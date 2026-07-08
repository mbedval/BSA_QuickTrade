"""Module 12 — Statistical Analysis.

Searches historical data for periods similar to current conditions,
calculates forward returns, win rates, and expected moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsa_quicktrade.analyzers.base import BaseAnalyzer
from rsa_quicktrade.core.models import AnalysisResult, StockData


class StatisticalAnalyzer(BaseAnalyzer):
    name = "statistics"

    def analyze(self, data: StockData) -> AnalysisResult:
        _, _, _, close, _ = self.get_ohlcv(data.daily)
        if close is None or len(close) < 100:
            return self.make_result(50, 10, ["Insufficient data for statistical analysis"])

        cfg = self.config.statistics
        reasons: list[str] = []
        metadata: dict = {}
        c = close.values.astype(float)

        # ── 1. Extract & normalize current pattern ──────────────────────
        window = cfg.similarity_window
        if len(c) < window + max(cfg.forward_periods) + 10:
            return self.make_result(50, 15, ["Not enough history for similarity search"])

        current_pattern = c[-window:]
        current_norm = self._normalize(current_pattern)

        # ── 2. Sliding window similarity search ─────────────────────────
        matches: list[dict] = []
        max_fwd = max(cfg.forward_periods)

        for start in range(0, len(c) - window - max_fwd - 1):
            candidate = c[start: start + window]
            candidate_norm = self._normalize(candidate)

            if candidate_norm is None or current_norm is None:
                continue

            corr = float(np.corrcoef(current_norm, candidate_norm)[0, 1])
            if corr >= cfg.min_correlation:
                # Calculate forward returns
                fwd_returns = {}
                for period in cfg.forward_periods:
                    end_idx = start + window + period
                    if end_idx < len(c):
                        fwd_ret = (c[end_idx] - c[start + window - 1]) / c[start + window - 1] * 100
                        fwd_returns[period] = fwd_ret

                if fwd_returns:
                    matches.append({
                        "start": start,
                        "correlation": corr,
                        "forward_returns": fwd_returns,
                    })

        metadata["num_matches"] = len(matches)

        if len(matches) < cfg.min_similar_matches:
            reasons.append(f"Only {len(matches)} similar patterns found (need {cfg.min_similar_matches})")
            return self.make_result(50, 20, reasons, **metadata)

        # ── 3. Aggregate statistics ─────────────────────────────────────
        score = 50.0
        confidence = 50.0

        for period in cfg.forward_periods:
            returns = [m["forward_returns"][period]
                       for m in matches if period in m["forward_returns"]]
            if not returns:
                continue

            avg_ret = float(np.mean(returns))
            win_ratio = sum(1 for r in returns if r > 0) / len(returns) * 100
            max_up = float(np.max(returns))
            max_dd = float(np.min(returns))
            std_ret = float(np.std(returns))

            metadata[f"avg_{period}d_return"] = round(avg_ret, 2)
            metadata[f"win_ratio_{period}d"] = round(win_ratio, 1)
            metadata[f"max_upside_{period}d"] = round(max_up, 2)
            metadata[f"max_drawdown_{period}d"] = round(max_dd, 2)
            metadata[f"std_{period}d"] = round(std_ret, 2)

            if period == 1:
                # Primary scoring based on 1-day forward
                if win_ratio > 60 and avg_ret > 0:
                    score = 55 + min(win_ratio - 50, 40)
                    reasons.append(
                        f"Historical 1-day win rate {win_ratio:.0f}% "
                        f"(avg +{avg_ret:.2f}%) from {len(returns)} similar setups"
                    )
                elif win_ratio < 40 and avg_ret < 0:
                    score = 45 - min(50 - win_ratio, 40)
                    reasons.append(
                        f"Historical 1-day win rate only {win_ratio:.0f}% "
                        f"(avg {avg_ret:.2f}%)"
                    )
                else:
                    reasons.append(f"Historical 1-day: {win_ratio:.0f}% win, avg {avg_ret:+.2f}%")

                confidence = min(80, 30 + len(returns) * 2)

        # 5-day and 20-day context
        if f"avg_5d_return" in metadata:
            reasons.append(
                f"5-day outlook: {metadata.get('win_ratio_5d', 50):.0f}% win, "
                f"avg {metadata.get('avg_5d_return', 0):+.2f}%"
            )
        if f"avg_20d_return" in metadata:
            reasons.append(
                f"20-day outlook: {metadata.get('win_ratio_20d', 50):.0f}% win, "
                f"avg {metadata.get('avg_20d_return', 0):+.2f}%"
            )

        # ── 4. Expected move (ATR-based) ────────────────────────────────
        if len(c) > 20:
            daily_returns = np.diff(c) / c[:-1]
            daily_std = float(np.std(daily_returns[-20:]))
            expected_1d_move = daily_std * c[-1] * 100 / c[-1]  # as percentage
            metadata["expected_1d_move_pct"] = round(expected_1d_move, 2)
            reasons.append(f"Expected daily move: ±{expected_1d_move:.1f}% (1σ)")

        score = float(np.clip(score, 0, 100))
        return self.make_result(score, confidence, reasons, **metadata)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray | None:
        """Z-score normalize an array."""
        std = float(np.std(arr))
        if std < 1e-10:
            return None
        return (arr - np.mean(arr)) / std
