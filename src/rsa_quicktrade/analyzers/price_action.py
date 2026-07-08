"""Module 5 — Price Action Analysis.

Identifies swing structure, support/resistance, breakouts, pullbacks,
retests, market structure shifts, and liquidity sweeps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]
from scipy.signal import argrelextrema

from rsa_quicktrade.analyzers.base import BaseAnalyzer
from rsa_quicktrade.core.models import AnalysisResult, PriceLevel, StockData


class PriceActionAnalyzer(BaseAnalyzer):
    name = "price_action"

    def analyze(self, data: StockData) -> AnalysisResult:
        open_, high, low, close, volume = self.get_ohlcv(data.daily)
        if close is None or len(close) < 50:
            return self.make_result(50, 10, ["Insufficient data for price action"])

        cfg = self.config.price_action
        sub: dict[str, float] = {}
        reasons: list[str] = []
        metadata: dict = {}
        price = float(close.iloc[-1])

        # ── Detect swing points ─────────────────────────────────────────
        h_vals = high.values.astype(float)
        l_vals = low.values.astype(float)
        c_vals = close.values.astype(float)
        v_vals = volume.values.astype(float) if volume is not None else np.ones(len(close))

        hi_idx = argrelextrema(h_vals, np.greater_equal, order=cfg.swing_order)[0]
        lo_idx = argrelextrema(l_vals, np.less_equal, order=cfg.swing_order)[0]

        swing_highs = [(int(i), float(h_vals[i])) for i in hi_idx]
        swing_lows = [(int(i), float(l_vals[i])) for i in lo_idx]

        # ── 1. Support / Resistance (20%) ───────────────────────────────
        lookback = min(cfg.support_resistance_lookback, len(close))
        supports, resistances = self._cluster_levels(
            swing_highs, swing_lows, price, cfg.level_cluster_pct, lookback, len(close),
        )
        metadata["support_levels"] = [PriceLevel(p, "support", s, "pivot", t)
                                      for p, s, t in supports[:5]]
        metadata["resistance_levels"] = [PriceLevel(p, "resistance", s, "pivot", t)
                                         for p, s, t in resistances[:5]]

        nearest_sup = supports[0][0] if supports else price * 0.95
        nearest_res = resistances[0][0] if resistances else price * 1.05

        # Proximity scoring
        sup_dist = (price - nearest_sup) / price * 100 if nearest_sup < price else 99
        res_dist = (nearest_res - price) / price * 100 if nearest_res > price else 99

        if sup_dist < 1.5:
            sub["sr"] = 70  # Near support — potential bounce
            reasons.append(f"Price near support ₹{nearest_sup:,.0f} ({sup_dist:.1f}% away)")
        elif res_dist < 1.5:
            sub["sr"] = 65  # Near resistance — potential breakout
            reasons.append(f"Price approaching resistance ₹{nearest_res:,.0f} ({res_dist:.1f}% away)")
        else:
            sub["sr"] = 50

        # ── 2. Breakout Detection (30%) ─────────────────────────────────
        avg_vol = float(np.mean(v_vals[-20:])) if len(v_vals) > 20 else 0
        breakout_score = 50.0

        if len(swing_highs) > 0:
            recent_res = max((p for _, p in swing_highs[-5:]), default=0)
            if price > recent_res and recent_res > 0:
                breakout_pct = (price - recent_res) / recent_res * 100
                vol_ok = float(v_vals[-1]) > avg_vol * cfg.breakout_volume_mult if avg_vol > 0 else False

                if breakout_pct < 3 and vol_ok:
                    breakout_score = 90
                    reasons.append(f"Breakout above ₹{recent_res:,.0f} with volume confirmation")
                elif breakout_pct < 3:
                    breakout_score = 70
                    reasons.append(f"Breakout above ₹{recent_res:,.0f} — needs volume confirmation")

        # Check for false breakout
        if len(close) > 3 and len(swing_highs) > 0:
            prev_res = max((p for _, p in swing_highs[-5:]), default=0)
            if (float(high.iloc[-2]) > prev_res and
                    float(close.iloc[-1]) < prev_res and
                    prev_res > 0):
                breakout_score = 30
                reasons.append(f"False breakout above ₹{prev_res:,.0f} — bearish rejection")

        sub["breakout"] = breakout_score

        # ── 3. Market Structure (20%) ───────────────────────────────────
        struct_score = 50.0
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            last_sh = swing_highs[-1][1]
            prev_sh = swing_highs[-2][1]
            last_sl = swing_lows[-1][1]
            prev_sl = swing_lows[-2][1]

            hh = last_sh > prev_sh
            hl = last_sl > prev_sl
            lh = last_sh < prev_sh
            ll = last_sl < prev_sl

            if hh and hl:
                struct_score = 80
                reasons.append("Bullish market structure — HH + HL")
            elif lh and ll:
                struct_score = 20
                reasons.append("Bearish market structure — LH + LL")
            elif hh and ll:
                struct_score = 50
                reasons.append("Market structure in transition")
            elif hh:
                struct_score = 65
            elif ll:
                struct_score = 35

            # Break of structure
            if price > last_sh and last_sh > prev_sh:
                struct_score = min(struct_score + 10, 95)
                reasons.append("Break of structure — new high above recent swing")

        sub["structure"] = struct_score

        # ── 4. Pullback to EMA (15%) ────────────────────────────────────
        pullback_score = 50.0
        ema20 = ta.ema(close, length=20)
        ema50 = ta.ema(close, length=50)

        if ema20 is not None and struct_score > 60:  # Only in uptrends
            ema20_val = float(ema20.iloc[-1])
            dist_to_ema = abs(price - ema20_val) / price * 100
            if dist_to_ema < 1.0 and price > ema20_val * 0.99:
                pullback_score = 75
                reasons.append(f"Pullback to EMA 20 — potential entry in uptrend")
        if ema50 is not None and struct_score > 60:
            ema50_val = float(ema50.iloc[-1])
            dist_to_ema50 = abs(price - ema50_val) / price * 100
            if dist_to_ema50 < 1.5 and price > ema50_val * 0.99:
                pullback_score = max(pullback_score, 70)
                reasons.append(f"Pullback to EMA 50 — strong support zone")

        sub["pullback"] = pullback_score

        # ── 5. Liquidity Sweep (15%) ────────────────────────────────────
        sweep_score = 50.0
        if len(swing_lows) >= 1 and len(close) > 2:
            recent_swing_low = swing_lows[-1][1]
            # Wick below swing low but close above
            if float(low.iloc[-1]) < recent_swing_low and float(close.iloc[-1]) > recent_swing_low:
                sweep_score = 80
                reasons.append(f"Liquidity sweep below ₹{recent_swing_low:,.0f} — bullish reversal")
            # Wick above swing high but close below
            if len(swing_highs) >= 1:
                recent_swing_high = swing_highs[-1][1]
                if float(high.iloc[-1]) > recent_swing_high and float(close.iloc[-1]) < recent_swing_high:
                    sweep_score = 25
                    reasons.append(f"Liquidity sweep above ₹{recent_swing_high:,.0f} — bearish rejection")

        sub["sweep"] = sweep_score

        # ── Combine ─────────────────────────────────────────────────────
        weights = {
            "breakout": 0.30, "sr": 0.20, "structure": 0.20,
            "pullback": 0.15, "sweep": 0.15,
        }
        overall = sum(sub.get(k, 50) * w for k, w in weights.items())
        confidence = min(85, len([s for s in sub.values() if abs(s - 50) > 10]) * 15)

        return self.make_result(overall, confidence, reasons, **metadata)

    @staticmethod
    def _cluster_levels(
        swing_highs: list[tuple[int, float]],
        swing_lows: list[tuple[int, float]],
        current_price: float,
        cluster_pct: float,
        lookback: int,
        total_bars: int,
    ) -> tuple[list[tuple[float, float, int]], list[tuple[float, float, int]]]:
        """Cluster nearby swing points into support/resistance levels.

        Returns (supports, resistances) — each is a list of
        (price, strength, touches) sorted by proximity to current price.
        """
        all_levels: list[float] = []
        start_bar = max(0, total_bars - lookback)
        for idx, price in swing_highs:
            if idx >= start_bar:
                all_levels.append(price)
        for idx, price in swing_lows:
            if idx >= start_bar:
                all_levels.append(price)

        if not all_levels:
            return [], []

        # Cluster levels within cluster_pct %
        all_levels.sort()
        clusters: list[list[float]] = []
        current_cluster: list[float] = [all_levels[0]]

        for lvl in all_levels[1:]:
            if (lvl - current_cluster[0]) / current_cluster[0] * 100 <= cluster_pct:
                current_cluster.append(lvl)
            else:
                clusters.append(current_cluster)
                current_cluster = [lvl]
        clusters.append(current_cluster)

        # Build levels with strength = touches count
        supports = []
        resistances = []
        for cluster in clusters:
            avg_price = float(np.mean(cluster))
            touches = len(cluster)
            strength = min(touches * 25, 100)

            if avg_price < current_price:
                supports.append((avg_price, strength, touches))
            else:
                resistances.append((avg_price, strength, touches))

        # Sort by proximity
        supports.sort(key=lambda x: current_price - x[0])
        resistances.sort(key=lambda x: x[0] - current_price)

        return supports, resistances
