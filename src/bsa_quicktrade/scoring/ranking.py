"""Score aggregation and ranking engine.

Combines the outputs of all 12 analysis modules using configurable
weights, calculates aggregate bullish/bearish scores and confidence,
generates trade setups, and ranks stocks for the final top-N selection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bsa_quicktrade.core.config import AppConfig
from bsa_quicktrade.core.models import (
    AnalysisResult,
    ExpectedRange,
    PriceLevel,
    Signal,
    StockAnalysis,
    StockData,
    TradeSetup,
)

logger = logging.getLogger(__name__)


class RankingEngine:
    """Aggregate module scores, rank stocks, and generate trade setups."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.weights = config.weights.normalized()

    # ── Core Aggregation ────────────────────────────────────────────────

    def aggregate(
        self,
        ticker: str,
        data: StockData,
        results: dict[str, AnalysisResult],
    ) -> StockAnalysis:
        """Combine all module results into a single ``StockAnalysis``."""
        _, high, low, close, volume = _get_ohlcv(data.daily)
        current_price = float(close.iloc[-1]) if len(close) > 0 else 0.0

        # Weighted score
        overall_score = self._weighted_score(results)
        bullish_score = self._directional_score(results, bullish=True)
        bearish_score = self._directional_score(results, bullish=False)
        confidence = self._confidence(results)
        signal = self._aggregate_signal(overall_score, confidence)

        # Support / resistance from analyzers
        supports = self._extract_levels(results, "support", current_price)
        resistances = self._extract_levels(results, "resistance", current_price)

        # Trade setup
        trade_setup = self._build_trade_setup(
            current_price, signal, supports, resistances, data.daily,
        )

        # Expected ranges
        atr = self._calc_atr(high, low, close, period=14)
        expected_intraday = ExpectedRange(
            low=current_price - atr,
            high=current_price + atr,
            period="intraday",
        )
        expected_week = ExpectedRange(
            low=current_price - atr * 2.5,
            high=current_price + atr * 2.5,
            period="1_week",
        )
        expected_month = ExpectedRange(
            low=current_price - atr * 5,
            high=current_price + atr * 5,
            period="1_month",
        )

        # Reasons and risks
        reasons = self._compile_reasons(results, signal)
        risks = self._compile_risks(results, signal)
        patterns = self._compile_patterns(results)

        return StockAnalysis(
            ticker=ticker,
            company_name=data.company_name,
            current_price=current_price,
            sector=data.sector,
            overall_score=overall_score,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            confidence=confidence,
            signal=signal,
            module_results=results,
            trade_setup=trade_setup,
            support_levels=supports,
            resistance_levels=resistances,
            expected_intraday_range=expected_intraday,
            expected_week_range=expected_week,
            expected_month_range=expected_month,
            reasons_for_selection=reasons,
            risks=risks,
            patterns_detected=patterns,
        )

    def rank(self, analyses: list[StockAnalysis]) -> list[StockAnalysis]:
        """Sort analyses by composite ranking score, return top N."""
        # Composite = overall_score * (confidence / 100) — rewards both
        for a in analyses:
            a._rank_key = a.overall_score * (a.confidence / 100)  # type: ignore[attr-defined]

        ranked = sorted(analyses, key=lambda a: a._rank_key, reverse=True)  # type: ignore[attr-defined]

        # Filter: minimum score + minimum confirmations
        cfg = self.config.scoring
        filtered = [
            a for a in ranked
            if a.overall_score >= cfg.min_overall_score
            and self._count_confirmations(a) >= cfg.min_confirmations
        ]

        top_n = filtered[: cfg.top_n]
        logger.info(
            "Ranking: %d analysed → %d passed filters → top %d selected",
            len(analyses), len(filtered), len(top_n),
        )
        return top_n

    # ── Internals ───────────────────────────────────────────────────────

    def _weighted_score(self, results: dict[str, AnalysisResult]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for module, weight in self.weights.items():
            if module in results:
                weighted_sum += results[module].score * weight
                total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 50.0

    def _directional_score(
        self, results: dict[str, AnalysisResult], *, bullish: bool
    ) -> float:
        scores = []
        for r in results.values():
            if bullish and r.signal.is_bullish:
                scores.append(r.score)
            elif not bullish and r.signal.is_bearish:
                scores.append(100 - r.score)  # invert for bearish strength
        return float(np.mean(scores)) if scores else 0.0

    def _confidence(self, results: dict[str, AnalysisResult]) -> float:
        if not results:
            return 0.0
        # Agreement: what fraction of modules agree on direction?
        bullish_count = sum(1 for r in results.values() if r.signal.is_bullish)
        bearish_count = sum(1 for r in results.values() if r.signal.is_bearish)
        majority = max(bullish_count, bearish_count)
        agreement = majority / len(results) * 100

        # Also weight by individual module confidences
        avg_conf = float(np.mean([r.confidence for r in results.values()]))

        return (agreement * 0.6 + avg_conf * 0.4)

    def _aggregate_signal(self, score: float, confidence: float) -> Signal:
        if confidence < 30:
            return Signal.NEUTRAL
        if score >= 75:
            return Signal.STRONG_BULLISH
        if score >= 60:
            return Signal.BULLISH
        if score >= 40:
            return Signal.NEUTRAL
        if score >= 25:
            return Signal.BEARISH
        return Signal.STRONG_BEARISH

    def _count_confirmations(self, analysis: StockAnalysis) -> int:
        """Count how many modules agree with the overall signal direction."""
        if analysis.signal.is_bullish:
            return sum(1 for r in analysis.module_results.values() if r.signal.is_bullish)
        if analysis.signal.is_bearish:
            return sum(1 for r in analysis.module_results.values() if r.signal.is_bearish)
        return len(analysis.module_results)

    # ── Trade Setup ─────────────────────────────────────────────────────

    def _build_trade_setup(
        self,
        price: float,
        signal: Signal,
        supports: list[PriceLevel],
        resistances: list[PriceLevel],
        daily_df: pd.DataFrame,
    ) -> TradeSetup | None:
        if price <= 0:
            return None

        _, high, low, close, _ = _get_ohlcv(daily_df)
        atr = self._calc_atr(high, low, close)
        ts_cfg = self.config.trade_setup

        if signal.is_bullish:
            # Bullish setup
            nearest_support = min(
                (s.price for s in supports if s.price < price),
                default=price - atr * ts_cfg.atr_sl_multiplier,
            )
            stop_loss = max(nearest_support - atr * 0.3, price - atr * ts_cfg.atr_sl_multiplier)
            best_entry = price
            aggressive_entry = price + atr * ts_cfg.aggressive_entry_offset_atr
            conservative_entry = price - atr * ts_cfg.conservative_entry_offset_atr

            risk = abs(best_entry - stop_loss)
            target_1 = best_entry + risk * ts_cfg.target_1_rr
            target_2 = best_entry + risk * ts_cfg.target_2_rr
            target_3 = best_entry + risk * ts_cfg.target_3_rr

        elif signal.is_bearish:
            # Bearish setup (short)
            nearest_resistance = max(
                (r.price for r in resistances if r.price > price),
                default=price + atr * ts_cfg.atr_sl_multiplier,
            )
            stop_loss = min(nearest_resistance + atr * 0.3, price + atr * ts_cfg.atr_sl_multiplier)
            best_entry = price
            aggressive_entry = price - atr * ts_cfg.aggressive_entry_offset_atr
            conservative_entry = price + atr * ts_cfg.conservative_entry_offset_atr

            risk = abs(stop_loss - best_entry)
            target_1 = best_entry - risk * ts_cfg.target_1_rr
            target_2 = best_entry - risk * ts_cfg.target_2_rr
            target_3 = best_entry - risk * ts_cfg.target_3_rr
        else:
            # Neutral — provide wide range
            stop_loss = price - atr * ts_cfg.atr_sl_multiplier
            best_entry = price
            aggressive_entry = price
            conservative_entry = price
            risk = atr
            target_1 = price + atr
            target_2 = price + atr * 2
            target_3 = price + atr * 3

        risk = abs(best_entry - stop_loss)
        rr = abs(target_1 - best_entry) / risk if risk > 0 else 0

        return TradeSetup(
            best_entry=round(best_entry, 2),
            aggressive_entry=round(aggressive_entry, 2),
            conservative_entry=round(conservative_entry, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            target_3=round(target_3, 2),
            risk_reward=round(rr, 2),
            probability=0.0,  # filled by statistical module if available
        )

    # ── Level Extraction ────────────────────────────────────────────────

    def _extract_levels(
        self,
        results: dict[str, AnalysisResult],
        level_type: str,
        current_price: float,
    ) -> list[PriceLevel]:
        levels: list[PriceLevel] = []
        key = f"{level_type}_levels"

        for r in results.values():
            raw_levels = r.metadata.get(key, [])
            if isinstance(raw_levels, list):
                for item in raw_levels:
                    if isinstance(item, PriceLevel):
                        levels.append(item)
                    elif isinstance(item, (int, float)):
                        levels.append(PriceLevel(
                            price=float(item),
                            level_type=level_type,
                            strength=50,
                            source=r.module_name,
                        ))

        # Also check for single values
        for key_name in [f"oi_{level_type}", f"max_pain", f"nearest_{level_type}"]:
            for r in results.values():
                val = r.metadata.get(key_name)
                if isinstance(val, (int, float)) and val > 0:
                    levels.append(PriceLevel(
                        price=float(val),
                        level_type=level_type,
                        strength=60,
                        source=r.module_name,
                    ))

        # Sort by proximity to current price
        levels.sort(key=lambda l: abs(l.price - current_price))
        return levels[:10]

    # ── Reason Compilation ──────────────────────────────────────────────

    def _compile_reasons(
        self, results: dict[str, AnalysisResult], signal: Signal,
    ) -> list[str]:
        reasons: list[str] = []
        for r in sorted(results.values(), key=lambda x: x.score, reverse=True):
            if signal.is_bullish and r.signal.is_bullish:
                reasons.extend(r.reasons[:2])
            elif signal.is_bearish and r.signal.is_bearish:
                reasons.extend(r.reasons[:2])
            elif signal == Signal.NEUTRAL:
                reasons.extend(r.reasons[:1])
        return reasons[:15]

    def _compile_risks(
        self, results: dict[str, AnalysisResult], signal: Signal,
    ) -> list[str]:
        risks: list[str] = []
        for r in results.values():
            # Opposing signals are risks
            if signal.is_bullish and r.signal.is_bearish:
                risks.extend([f"[{r.module_name}] {reason}" for reason in r.reasons[:1]])
            elif signal.is_bearish and r.signal.is_bullish:
                risks.extend([f"[{r.module_name}] {reason}" for reason in r.reasons[:1]])
        if not risks:
            risks.append("No significant opposing signals detected")
        return risks[:10]

    def _compile_patterns(self, results: dict[str, AnalysisResult]) -> list[str]:
        patterns: list[str] = []
        for r in results.values():
            pats = r.metadata.get("patterns_detected", [])
            if isinstance(pats, list):
                patterns.extend(pats)
        return patterns

    # ── ATR Calculation ─────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float:
        if len(close) < period + 1:
            return float((high.iloc[-1] - low.iloc[-1])) if len(close) > 0 else 1.0
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else float(tr.iloc[-1])


def _get_ohlcv(df: pd.DataFrame):
    """Extract OHLCV — mirrors BaseAnalyzer.get_ohlcv."""
    cols = {}
    target_names = ["open", "high", "low", "close", "volume"]
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            name = str(col[-1]).lower() if isinstance(col, tuple) else str(col).lower()
            for tn in target_names:
                if tn in name and tn not in cols:
                    cols[tn] = df[col]
    else:
        for col in df.columns:
            cl = str(col).lower()
            for tn in target_names:
                if tn in cl and tn not in cols:
                    cols[tn] = df[col]
    return (
        cols.get("open", pd.Series(dtype=float)),
        cols.get("high", pd.Series(dtype=float)),
        cols.get("low", pd.Series(dtype=float)),
        cols.get("close", pd.Series(dtype=float)),
        cols.get("volume", pd.Series(dtype=float)),
    )
