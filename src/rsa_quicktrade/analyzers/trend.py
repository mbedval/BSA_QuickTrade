"""Module 1 — Trend Analysis.

Evaluates trend direction and strength using EMA alignment, ADX,
slope regression, swing-structure, and multi-timeframe confirmation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]
from scipy import stats
from scipy.signal import argrelextrema

from rsa_quicktrade.analyzers.base import BaseAnalyzer
from rsa_quicktrade.core.models import AnalysisResult, StockData


class TrendAnalyzer(BaseAnalyzer):
    name = "trend"

    def analyze(self, data: StockData) -> AnalysisResult:
        open_, high, low, close, volume = self.get_ohlcv(data.daily)
        if close is None or len(close) < 50:
            return self.make_result(50, 10, ["Insufficient data for trend analysis"])

        cfg = self.config.trend
        sub_scores: dict[str, float] = {}
        reasons: list[str] = []

        # ── 1. EMA Stack Alignment (25%) ────────────────────────────────
        emas: dict[int, pd.Series] = {}
        for p in cfg.ema_periods:
            e = ta.ema(close, length=p)
            if e is not None:
                emas[p] = e

        if len(emas) >= 2:
            last_vals = {p: float(e.iloc[-1]) for p, e in emas.items() if not np.isnan(e.iloc[-1])}
            if last_vals:
                sorted_periods = sorted(last_vals.keys())
                sorted_vals = [last_vals[p] for p in sorted_periods]
                # Perfect bullish: shortest EMA > longest, all descending by period
                bullish_pairs = sum(
                    1 for i in range(len(sorted_vals) - 1) if sorted_vals[i] > sorted_vals[i + 1]
                )
                total_pairs = len(sorted_vals) - 1
                ema_score = (bullish_pairs / total_pairs) * 100 if total_pairs > 0 else 50
                sub_scores["ema_stack"] = ema_score

                price = float(close.iloc[-1])
                above_count = sum(1 for v in last_vals.values() if price > v)
                if above_count == len(last_vals):
                    reasons.append(f"Price above all EMAs ({', '.join(str(p) for p in sorted_periods)})")
                elif above_count == 0:
                    reasons.append("Price below all EMAs — downtrend")
                else:
                    reasons.append(f"Price above {above_count}/{len(last_vals)} EMAs")
        else:
            sub_scores["ema_stack"] = 50

        # ── 2. Golden / Death Cross (15%) ───────────────────────────────
        if 50 in emas and 200 in emas:
            ema50 = emas[50].dropna()
            ema200 = emas[200].dropna()
            if len(ema50) > 5 and len(ema200) > 5:
                cross_score = 50.0
                # Check recent crossovers (last 10 bars)
                for i in range(-1, max(-11, -len(ema50)), -1):
                    try:
                        curr_diff = float(ema50.iloc[i]) - float(ema200.iloc[i])
                        prev_diff = float(ema50.iloc[i - 1]) - float(ema200.iloc[i - 1])
                        if curr_diff > 0 and prev_diff <= 0:
                            recency = 10 - abs(i + 1)
                            cross_score = 70 + recency * 3
                            reasons.append(f"Golden Cross (EMA 50/200) {abs(i+1)} bars ago")
                            break
                        elif curr_diff < 0 and prev_diff >= 0:
                            recency = 10 - abs(i + 1)
                            cross_score = 30 - recency * 3
                            reasons.append(f"Death Cross (EMA 50/200) {abs(i+1)} bars ago")
                            break
                    except (IndexError, ValueError):
                        break

                # Current state if no recent cross
                if cross_score == 50 and len(ema50) > 0 and len(ema200) > 0:
                    if float(ema50.iloc[-1]) > float(ema200.iloc[-1]):
                        cross_score = 65
                    else:
                        cross_score = 35
                sub_scores["cross"] = cross_score
        else:
            sub_scores["cross"] = 50

        # ── 3. ADX Trend Strength (20%) ─────────────────────────────────
        adx_data = ta.adx(high, low, close, length=cfg.adx_period)
        if adx_data is not None and len(adx_data) > 0:
            adx_cols = adx_data.columns.tolist()
            adx_val = float(adx_data[adx_cols[0]].iloc[-1]) if not np.isnan(adx_data[adx_cols[0]].iloc[-1]) else 0
            # +DI and -DI
            plus_di = float(adx_data[adx_cols[1]].iloc[-1]) if len(adx_cols) > 1 and not np.isnan(adx_data[adx_cols[1]].iloc[-1]) else 0
            minus_di = float(adx_data[adx_cols[2]].iloc[-1]) if len(adx_cols) > 2 and not np.isnan(adx_data[adx_cols[2]].iloc[-1]) else 0

            if adx_val < cfg.adx_strong_threshold:
                # Weak trend
                adx_score = 50  # neutral — no trend
                reasons.append(f"ADX {adx_val:.0f} — weak/no trend")
            elif plus_di > minus_di:
                strength = min((adx_val - cfg.adx_strong_threshold) / (cfg.adx_very_strong_threshold - cfg.adx_strong_threshold), 1.0)
                adx_score = 65 + strength * 35
                reasons.append(f"ADX {adx_val:.0f} with +DI > -DI — bullish trend")
            else:
                strength = min((adx_val - cfg.adx_strong_threshold) / (cfg.adx_very_strong_threshold - cfg.adx_strong_threshold), 1.0)
                adx_score = 35 - strength * 35
                reasons.append(f"ADX {adx_val:.0f} with -DI > +DI — bearish trend")

            sub_scores["adx"] = float(np.clip(adx_score, 0, 100))
        else:
            sub_scores["adx"] = 50

        # ── 4. Slope Analysis (15%) ─────────────────────────────────────
        period = min(cfg.slope_period, len(close) - 1)
        if period >= 10:
            y = close.iloc[-period:].values.astype(float)
            x = np.arange(len(y))
            slope, _, r_value, _, _ = stats.linregress(x, y)
            # Normalize slope relative to price level
            slope_pct = slope / float(np.mean(y)) * 100  # daily % change
            r_sq = r_value ** 2

            # Score: strong positive slope with high R² = bullish
            if slope_pct > 0:
                slope_score = 50 + min(slope_pct * 10, 50) * r_sq
            else:
                slope_score = 50 + max(slope_pct * 10, -50) * r_sq

            sub_scores["slope"] = float(np.clip(slope_score, 0, 100))
            if slope_pct > 0.1:
                reasons.append(f"Positive regression slope ({slope_pct:.2f}%/day, R²={r_sq:.2f})")
            elif slope_pct < -0.1:
                reasons.append(f"Negative regression slope ({slope_pct:.2f}%/day, R²={r_sq:.2f})")
        else:
            sub_scores["slope"] = 50

        # ── 5. Higher High / Higher Low (15%) ───────────────────────────
        order = cfg.swing_lookback
        if len(close) > order * 4:
            vals = close.values.astype(float)
            hi_idx = argrelextrema(vals, np.greater_equal, order=order)[0]
            lo_idx = argrelextrema(vals, np.less_equal, order=order)[0]

            swing_score = 50.0
            if len(hi_idx) >= 2 and len(lo_idx) >= 2:
                last_highs = vals[hi_idx[-2:]]
                last_lows = vals[lo_idx[-2:]]

                hh = last_highs[-1] > last_highs[-2]
                hl = last_lows[-1] > last_lows[-2]
                lh = last_highs[-1] < last_highs[-2]
                ll = last_lows[-1] < last_lows[-2]

                if hh and hl:
                    swing_score = 85
                    reasons.append("Higher High + Higher Low — bullish structure")
                elif hh or hl:
                    swing_score = 65
                elif lh and ll:
                    swing_score = 15
                    reasons.append("Lower High + Lower Low — bearish structure")
                elif lh or ll:
                    swing_score = 35

            sub_scores["swing"] = swing_score
        else:
            sub_scores["swing"] = 50

        # ── 6. Weekly Trend (10%) ───────────────────────────────────────
        weekly_score = 50.0
        if data.weekly is not None and len(data.weekly) > 20:
            _, _, _, w_close, _ = self.get_ohlcv(data.weekly)
            if w_close is not None and len(w_close) > 20:
                w_ema20 = ta.ema(w_close, length=20)
                w_ema50 = ta.ema(w_close, length=min(50, len(w_close) - 1))
                if w_ema20 is not None and w_ema50 is not None:
                    if float(w_ema20.iloc[-1]) > float(w_ema50.iloc[-1]):
                        weekly_score = 75
                        reasons.append("Weekly EMA 20 > 50 — bullish weekly trend")
                    else:
                        weekly_score = 25
                        reasons.append("Weekly EMA 20 < 50 — bearish weekly trend")
        sub_scores["weekly"] = weekly_score

        # ── Combine ─────────────────────────────────────────────────────
        weights = {
            "ema_stack": 0.25, "adx": 0.20, "cross": 0.15,
            "slope": 0.15, "swing": 0.15, "weekly": 0.10,
        }
        overall = sum(sub_scores.get(k, 50) * w for k, w in weights.items())
        confidence = min(90, len([s for s in sub_scores.values() if s != 50]) * 15)

        return self.make_result(overall, confidence, reasons)
