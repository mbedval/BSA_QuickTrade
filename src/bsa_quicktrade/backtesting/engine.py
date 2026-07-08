"""Walk-forward backtesting engine.

Evaluates each analysis module (and the combined system) over historical
data using a rolling window approach.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bsa_quicktrade.core.config import AppConfig
from bsa_quicktrade.core.models import AnalysisResult, BacktestResult, Signal, StockData

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Walk-forward backtesting for scoring modules."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.bt_cfg = config.backtesting
        self.commission = config.backtesting.commission_pct / 100
        self.slippage = config.backtesting.slippage_pct / 100

    def backtest_module(
        self,
        analyzer: Any,  # BaseAnalyzer
        stock_data: StockData,
        window: int | None = None,
    ) -> BacktestResult:
        """Run a walk-forward backtest for a single analyzer module.

        The approach:
        1. For each date *t* from `window` to the end of the series:
           - Feed the analyzer data up to *t* (exclusive of future)
           - Record the signal
           - Measure the next-day return
        2. Compute aggregate metrics.
        """
        window = window or self.bt_cfg.walk_forward_window
        daily = stock_data.daily.copy()

        if len(daily) < window + 20:
            return self._empty_result(getattr(analyzer, "name", "unknown"))

        # Extract close prices
        close = _get_close(daily)
        if close is None or len(close) < window + 20:
            return self._empty_result(getattr(analyzer, "name", "unknown"))

        trades: list[dict[str, Any]] = []
        start_idx = max(window, 200)  # need enough history for long EMAs

        for i in range(start_idx, len(daily) - 1):
            # Create a truncated StockData up to index i
            truncated = StockData(
                ticker=stock_data.ticker,
                company_name=stock_data.company_name,
                daily=daily.iloc[: i + 1],
                weekly=stock_data.weekly,
                sector=stock_data.sector,
            )

            try:
                result: AnalysisResult = analyzer.analyze(truncated)
            except Exception:
                continue

            # Next-day return
            next_close = float(close.iloc[i + 1])
            curr_close = float(close.iloc[i])
            if curr_close <= 0:
                continue
            next_return = (next_close - curr_close) / curr_close

            # Adjust for costs
            net_return = next_return - (self.commission + self.slippage)

            # Record trade
            if result.signal.is_bullish:
                trades.append({
                    "date": daily.index[i],
                    "signal": "long",
                    "score": result.score,
                    "return": net_return,
                    "predicted_bullish": True,
                    "actual_bullish": next_return > 0,
                })
            elif result.signal.is_bearish:
                trades.append({
                    "date": daily.index[i],
                    "signal": "short",
                    "score": result.score,
                    "return": -net_return,  # short position profits on decline
                    "predicted_bullish": False,
                    "actual_bullish": next_return > 0,
                })

        if not trades:
            return self._empty_result(getattr(analyzer, "name", "unknown"))

        return self._compute_metrics(getattr(analyzer, "name", "unknown"), trades)

    def backtest_system(
        self,
        analyzers: list[Any],
        stock_data: StockData,
    ) -> BacktestResult:
        """Backtest the combined scoring system across all analyzers."""
        # Simplified: run each analyzer, combine signals via majority vote
        daily = stock_data.daily.copy()
        close = _get_close(daily)
        if close is None or len(daily) < 220:
            return self._empty_result("system")

        trades: list[dict[str, Any]] = []

        for i in range(200, len(daily) - 1):
            truncated = StockData(
                ticker=stock_data.ticker,
                company_name=stock_data.company_name,
                daily=daily.iloc[: i + 1],
                weekly=stock_data.weekly,
                sector=stock_data.sector,
            )

            bullish_votes = 0
            bearish_votes = 0
            total_score = 0.0
            n_modules = 0

            for analyzer in analyzers:
                try:
                    result = analyzer.analyze(truncated)
                    if result.signal.is_bullish:
                        bullish_votes += 1
                    elif result.signal.is_bearish:
                        bearish_votes += 1
                    total_score += result.score
                    n_modules += 1
                except Exception:
                    continue

            if n_modules == 0:
                continue

            avg_score = total_score / n_modules
            next_close = float(close.iloc[i + 1])
            curr_close = float(close.iloc[i])
            if curr_close <= 0:
                continue
            next_return = (next_close - curr_close) / curr_close
            net_return = next_return - (self.commission + self.slippage)

            if bullish_votes > bearish_votes and bullish_votes >= 3:
                trades.append({
                    "date": daily.index[i],
                    "signal": "long",
                    "score": avg_score,
                    "return": net_return,
                    "predicted_bullish": True,
                    "actual_bullish": next_return > 0,
                })
            elif bearish_votes > bullish_votes and bearish_votes >= 3:
                trades.append({
                    "date": daily.index[i],
                    "signal": "short",
                    "score": avg_score,
                    "return": -net_return,
                    "predicted_bullish": False,
                    "actual_bullish": next_return > 0,
                })

        if not trades:
            return self._empty_result("system")

        return self._compute_metrics("system", trades)

    # ── Metrics ─────────────────────────────────────────────────────────

    def _compute_metrics(
        self, module_name: str, trades: list[dict[str, Any]],
    ) -> BacktestResult:
        returns = [t["return"] for t in trades]
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r <= 0]

        total = len(trades)
        wins = len(winning)
        losses = len(losing)
        win_rate = wins / total if total > 0 else 0

        avg_return = float(np.mean(returns)) if returns else 0
        avg_win = float(np.mean(winning)) if winning else 0
        avg_loss = float(np.mean(losing)) if losing else 0

        # Sharpe ratio (annualised, assuming 252 trading days)
        if len(returns) > 1:
            ret_std = float(np.std(returns))
            sharpe = (avg_return / ret_std * np.sqrt(252)) if ret_std > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1e-10
        profit_factor = gross_profit / gross_loss

        # Max drawdown
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        # Accuracy / Precision / Recall (for bullish predictions)
        tp = sum(1 for t in trades if t["predicted_bullish"] and t["actual_bullish"])
        fp = sum(1 for t in trades if t["predicted_bullish"] and not t["actual_bullish"])
        fn = sum(1 for t in trades if not t["predicted_bullish"] and t["actual_bullish"])
        tn = sum(1 for t in trades if not t["predicted_bullish"] and not t["actual_bullish"])

        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Total return
        total_return = float(np.sum(returns)) * 100

        return BacktestResult(
            module_name=module_name,
            total_trades=total,
            winning_trades=wins,
            losing_trades=losses,
            win_rate=round(win_rate * 100, 1),
            avg_return_pct=round(avg_return * 100, 3),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            avg_holding_days=1.0,  # Intraday assumption
            accuracy=round(accuracy * 100, 1),
            precision=round(precision * 100, 1),
            recall=round(recall * 100, 1),
            total_return_pct=round(total_return, 2),
        )

    def _empty_result(self, name: str) -> BacktestResult:
        return BacktestResult(
            module_name=name,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_return_pct=0,
            sharpe_ratio=0,
            profit_factor=0,
            max_drawdown_pct=0,
            avg_holding_days=0,
            accuracy=0,
            precision=0,
            recall=0,
        )


def _get_close(df: pd.DataFrame) -> pd.Series | None:
    """Extract close series from DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            if "close" in str(col[-1]).lower():
                return df[col]
    else:
        for col in df.columns:
            if "close" in str(col).lower():
                return df[col]
    return None
