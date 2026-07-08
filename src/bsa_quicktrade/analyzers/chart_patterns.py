"""Module 7 — Chart Pattern Analysis.

Detects classical chart patterns using scipy peak detection
and rule-based structural matching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from bsa_quicktrade.analyzers.base import BaseAnalyzer
from bsa_quicktrade.core.models import AnalysisResult, StockData


class ChartPatternAnalyzer(BaseAnalyzer):
    name = "chart_patterns"

    def analyze(self, data: StockData) -> AnalysisResult:
        _, high, low, close, volume = self.get_ohlcv(data.daily)
        if close is None or len(close) < 50:
            return self.make_result(50, 10, ["Insufficient data for chart patterns"])

        cfg = self.config.chart_patterns
        reasons: list[str] = []
        detected: list[str] = []
        net_signal = 0.0

        # Tolerance adjustment by strictness
        tol = {"conservative": cfg.tolerance_pct * 0.5,
               "moderate": cfg.tolerance_pct,
               "liberal": cfg.tolerance_pct * 1.5}.get(cfg.strictness, cfg.tolerance_pct)

        c = close.values.astype(float)
        h = high.values.astype(float)
        lo = low.values.astype(float)
        price = c[-1]

        for lookback in cfg.lookback_periods:
            if len(c) < lookback:
                continue
            seg_c = c[-lookback:]
            seg_h = h[-lookback:]
            seg_l = lo[-lookback:]

            order = max(cfg.pivot_order, lookback // 15)
            hi_idx = argrelextrema(seg_h, np.greater_equal, order=order)[0]
            lo_idx = argrelextrema(seg_l, np.less_equal, order=order)[0]

            if len(hi_idx) < 2 or len(lo_idx) < 2:
                continue

            peaks = [(int(i), float(seg_h[i])) for i in hi_idx]
            troughs = [(int(i), float(seg_l[i])) for i in lo_idx]

            # ── Double Top ──────────────────────────────────────────
            if len(peaks) >= 2:
                p1, p2 = peaks[-2], peaks[-1]
                if (abs(p1[1] - p2[1]) / p1[1] * 100 < tol and
                        abs(p1[0] - p2[0]) >= cfg.min_pattern_bars and
                        "Double Top" not in detected):
                    valley = min(seg_l[p1[0]:p2[0] + 1])
                    if price < valley:
                        net_signal -= 3.0
                        detected.append("Double Top")
                        reasons.append(f"Double Top at ₹{p1[1]:,.0f} (neckline broken)")
                    elif price < p1[1]:
                        net_signal -= 1.5
                        detected.append("Double Top (forming)")
                        reasons.append(f"Double Top forming at ₹{p1[1]:,.0f}")

            # ── Double Bottom ───────────────────────────────────────
            if len(troughs) >= 2:
                t1, t2 = troughs[-2], troughs[-1]
                if (abs(t1[1] - t2[1]) / t1[1] * 100 < tol and
                        abs(t1[0] - t2[0]) >= cfg.min_pattern_bars and
                        "Double Bottom" not in detected):
                    peak_between = max(seg_h[t1[0]:t2[0] + 1])
                    if price > peak_between:
                        net_signal += 3.0
                        detected.append("Double Bottom")
                        reasons.append(f"Double Bottom at ₹{t1[1]:,.0f} (neckline broken)")
                    elif price > t1[1]:
                        net_signal += 1.5
                        detected.append("Double Bottom (forming)")

            # ── Head & Shoulders ────────────────────────────────────
            if len(peaks) >= 3:
                for i in range(len(peaks) - 2):
                    left, head, right = peaks[i], peaks[i + 1], peaks[i + 2]
                    if (head[1] > left[1] and head[1] > right[1] and
                            abs(left[1] - right[1]) / left[1] * 100 < tol * 2 and
                            "Head & Shoulders" not in detected):
                        # Find neckline troughs
                        t_between = [t for t in troughs if left[0] < t[0] < right[0]]
                        if t_between:
                            neckline = min(t[1] for t in t_between)
                            if price < neckline:
                                net_signal -= 3.5
                                detected.append("Head & Shoulders")
                                reasons.append("Head & Shoulders complete — bearish breakdown")
                            else:
                                detected.append("Head & Shoulders (forming)")
                                net_signal -= 1.0

            # ── Inverse H&S ────────────────────────────────────────
            if len(troughs) >= 3:
                for i in range(len(troughs) - 2):
                    left, head, right = troughs[i], troughs[i + 1], troughs[i + 2]
                    if (head[1] < left[1] and head[1] < right[1] and
                            abs(left[1] - right[1]) / left[1] * 100 < tol * 2 and
                            "Inverse H&S" not in detected):
                        p_between = [p for p in peaks if left[0] < p[0] < right[0]]
                        if p_between:
                            neckline = max(p[1] for p in p_between)
                            if price > neckline:
                                net_signal += 3.5
                                detected.append("Inverse H&S")
                                reasons.append("Inverse Head & Shoulders — bullish breakout")

            # ── Triangles ───────────────────────────────────────────
            if len(peaks) >= 3 and len(troughs) >= 3:
                hi_vals = [p[1] for p in peaks[-3:]]
                lo_vals = [t[1] for t in troughs[-3:]]

                hi_slope = (hi_vals[-1] - hi_vals[0]) / max(1, peaks[-1][0] - peaks[-3][0])
                lo_slope = (lo_vals[-1] - lo_vals[0]) / max(1, troughs[-1][0] - troughs[-3][0])

                flat_hi = abs(hi_vals[-1] - hi_vals[0]) / hi_vals[0] * 100 < tol
                flat_lo = abs(lo_vals[-1] - lo_vals[0]) / lo_vals[0] * 100 < tol

                if flat_hi and lo_slope > 0 and "Ascending Triangle" not in detected:
                    net_signal += 2.0
                    detected.append("Ascending Triangle")
                    reasons.append(f"Ascending Triangle — bullish bias (flat top ~₹{hi_vals[-1]:,.0f})")
                elif flat_lo and hi_slope < 0 and "Descending Triangle" not in detected:
                    net_signal -= 2.0
                    detected.append("Descending Triangle")
                    reasons.append(f"Descending Triangle — bearish bias (flat bottom ~₹{lo_vals[-1]:,.0f})")
                elif hi_slope < 0 and lo_slope > 0 and "Symmetrical Triangle" not in detected:
                    detected.append("Symmetrical Triangle")
                    reasons.append("Symmetrical Triangle — awaiting breakout direction")

            # ── Wedge ───────────────────────────────────────────────
            if len(peaks) >= 3 and len(troughs) >= 3:
                hi_vals = [p[1] for p in peaks[-3:]]
                lo_vals = [t[1] for t in troughs[-3:]]
                hi_slope_w = hi_vals[-1] - hi_vals[0]
                lo_slope_w = lo_vals[-1] - lo_vals[0]

                if hi_slope_w > 0 and lo_slope_w > 0 and hi_slope_w < lo_slope_w and "Rising Wedge" not in detected:
                    net_signal -= 1.5
                    detected.append("Rising Wedge")
                    reasons.append("Rising Wedge — bearish reversal pattern")
                elif hi_slope_w < 0 and lo_slope_w < 0 and abs(hi_slope_w) < abs(lo_slope_w) and "Falling Wedge" not in detected:
                    net_signal += 1.5
                    detected.append("Falling Wedge")
                    reasons.append("Falling Wedge — bullish reversal pattern")

            # ── Flag / Pennant ──────────────────────────────────────
            if lookback <= 50 and len(seg_c) > 20:
                # Check for flagpole (sharp move in first half)
                mid = len(seg_c) // 2
                pole_move = (seg_c[mid] - seg_c[0]) / seg_c[0] * 100
                consolidation_range = (max(seg_h[mid:]) - min(seg_l[mid:])) / seg_c[mid] * 100

                if abs(pole_move) > 5 and consolidation_range < abs(pole_move) * 0.5:
                    if pole_move > 0 and "Bull Flag" not in detected:
                        net_signal += 2.0
                        detected.append("Bull Flag")
                        reasons.append(f"Bull Flag — {pole_move:.0f}% pole with tight consolidation")
                    elif pole_move < 0 and "Bear Flag" not in detected:
                        net_signal -= 2.0
                        detected.append("Bear Flag")

        # ── Score ───────────────────────────────────────────────────────
        score = 50 + net_signal * 4
        score = float(np.clip(score, 0, 100))
        confidence = min(80, len(detected) * 18)
        if not detected:
            confidence = 10
            reasons.append("No classical chart patterns detected in recent data")

        return self.make_result(score, confidence, reasons, patterns_detected=detected)
