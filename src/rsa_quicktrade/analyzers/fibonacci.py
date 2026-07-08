"""Module 8 — Fibonacci Analysis.

Auto-detects significant swing points and calculates retracement
and extension levels, confluence zones, and proximity scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from rsa_quicktrade.analyzers.base import BaseAnalyzer
from rsa_quicktrade.core.models import AnalysisResult, PriceLevel, StockData


class FibonacciAnalyzer(BaseAnalyzer):
    name = "fibonacci"

    def analyze(self, data: StockData) -> AnalysisResult:
        _, high, low, close, _ = self.get_ohlcv(data.daily)
        if close is None or len(close) < 50:
            return self.make_result(50, 10, ["Insufficient data for Fibonacci"])

        cfg = self.config.fibonacci
        reasons: list[str] = []
        metadata: dict = {}
        price = float(close.iloc[-1])

        h_vals = high.values.astype(float)
        l_vals = low.values.astype(float)

        # ── Find significant swing points ───────────────────────────────
        order = cfg.swing_order
        hi_idx = argrelextrema(h_vals, np.greater_equal, order=order)[0]
        lo_idx = argrelextrema(l_vals, np.less_equal, order=order)[0]

        if len(hi_idx) == 0 or len(lo_idx) == 0:
            return self.make_result(50, 10, ["No significant swing points found"])

        # Most recent major swing high and low
        swing_high = float(h_vals[hi_idx[-1]])
        swing_low = float(l_vals[lo_idx[-1]])
        swing_high_idx = int(hi_idx[-1])
        swing_low_idx = int(lo_idx[-1])

        # Determine trend direction
        uptrend = swing_low_idx < swing_high_idx  # low before high = uptrend

        # ── Calculate Retracement Levels ────────────────────────────────
        if uptrend:
            diff = swing_high - swing_low
            ret_levels = {f"Fib {level*100:.1f}%": swing_high - diff * level
                          for level in cfg.retracement_levels}
            ext_levels = {f"Fib Ext {level*100:.1f}%": swing_low + diff * level
                          for level in cfg.extension_levels}
        else:
            diff = swing_high - swing_low
            ret_levels = {f"Fib {level*100:.1f}%": swing_low + diff * level
                          for level in cfg.retracement_levels}
            ext_levels = {f"Fib Ext {level*100:.1f}%": swing_high - diff * level
                          for level in cfg.extension_levels}

        metadata["retracement_levels"] = ret_levels
        metadata["extension_levels"] = ext_levels
        metadata["swing_high"] = swing_high
        metadata["swing_low"] = swing_low
        metadata["trend"] = "uptrend" if uptrend else "downtrend"

        # ── Price Proximity Scoring ─────────────────────────────────────
        all_levels = {**ret_levels, **ext_levels}
        closest_level = None
        closest_dist = float("inf")

        for name, level in all_levels.items():
            dist = abs(price - level) / price * 100
            if dist < closest_dist:
                closest_dist = dist
                closest_level = (name, level)

        score = 50.0
        confidence = 40.0

        if closest_level and closest_dist < cfg.confluence_threshold_pct:
            # Price is AT a Fib level
            level_name, level_price = closest_level

            if uptrend and "Ext" not in level_name:
                # Uptrend + at retracement = buy zone
                if "61.8" in level_name:
                    score = 82
                    reasons.append(f"Price at Golden Ratio (61.8%) retracement ₹{level_price:,.0f} — prime buy zone")
                    confidence = 75
                elif "38.2" in level_name:
                    score = 72
                    reasons.append(f"Price at 38.2% retracement ₹{level_price:,.0f} — shallow pullback buy")
                    confidence = 65
                elif "50.0" in level_name:
                    score = 70
                    reasons.append(f"Price at 50% retracement ₹{level_price:,.0f}")
                    confidence = 60
                else:
                    score = 60
                    reasons.append(f"Price near {level_name} level ₹{level_price:,.0f}")
            elif not uptrend and "Ext" not in level_name:
                # Downtrend + at retracement = sell zone
                if "61.8" in level_name:
                    score = 20
                    reasons.append(f"Price at 61.8% retracement in downtrend — potential resistance")
                    confidence = 70
                elif "38.2" in level_name:
                    score = 35
                    confidence = 55
            elif "Ext" in level_name:
                # At extension = potential target reached
                score = 45
                reasons.append(f"Price at extension level {level_name} ₹{level_price:,.0f} — potential target")
                confidence = 55
        elif closest_dist < 3.0 and closest_level:
            level_name, level_price = closest_level
            reasons.append(f"Price approaching {level_name} ₹{level_price:,.0f} ({closest_dist:.1f}% away)")
            if uptrend and "Ext" not in level_name:
                score = 60
            confidence = 45

        # ── Confluence Detection ────────────────────────────────────────
        # Check if multiple Fib levels from different swings cluster together
        if len(hi_idx) >= 2 and len(lo_idx) >= 2:
            alt_high = float(h_vals[hi_idx[-2]])
            alt_low = float(l_vals[lo_idx[-2]])
            alt_diff = alt_high - alt_low

            alt_ret = {level: alt_high - alt_diff * level if swing_low_idx < swing_high_idx
                       else alt_low + alt_diff * level
                       for level in cfg.retracement_levels}

            # Check confluence
            for name1, lvl1 in ret_levels.items():
                for lvl2 in alt_ret.values():
                    if abs(lvl1 - lvl2) / lvl1 * 100 < cfg.confluence_threshold_pct:
                        dist_to_conf = abs(price - lvl1) / price * 100
                        if dist_to_conf < 2.0:
                            score = max(score, 78)
                            confidence = max(confidence, 70)
                            reasons.append(f"Fibonacci confluence zone near ₹{lvl1:,.0f}")
                            metadata["confluence_zone"] = round(lvl1, 2)
                            break

        # Store as support/resistance
        support_fib = [PriceLevel(v, "support", 60, "fibonacci") for k, v in ret_levels.items() if v < price]
        resistance_fib = [PriceLevel(v, "resistance", 60, "fibonacci") for k, v in ret_levels.items() if v > price]
        metadata["support_levels"] = support_fib[:3]
        metadata["resistance_levels"] = resistance_fib[:3]

        if not reasons:
            reasons.append(f"Fibonacci levels calculated (swing {swing_low:,.0f}-{swing_high:,.0f})")

        return self.make_result(score, confidence, reasons, **metadata)
