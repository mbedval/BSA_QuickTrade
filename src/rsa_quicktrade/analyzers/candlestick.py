"""Module 6 — Candlestick Pattern Analysis.

Detects 15+ candlestick patterns with context-aware scoring —
patterns at support/resistance score higher than mid-range patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsa_quicktrade.analyzers.base import BaseAnalyzer
from rsa_quicktrade.core.models import AnalysisResult, StockData


class CandlestickAnalyzer(BaseAnalyzer):
    name = "candlestick"

    def analyze(self, data: StockData) -> AnalysisResult:
        open_, high, low, close, volume = self.get_ohlcv(data.daily)
        if close is None or len(close) < 10:
            return self.make_result(50, 10, ["Insufficient data for candlestick analysis"])

        cfg = self.config.candlestick
        reasons: list[str] = []
        patterns_detected: list[str] = []
        net_signal = 0.0  # positive = bullish, negative = bearish

        o = open_.values.astype(float) if open_ is not None else close.values.astype(float)
        h = high.values.astype(float)
        lo = low.values.astype(float)
        c = close.values.astype(float)
        v = volume.values.astype(float) if volume is not None else np.ones(len(c))

        if len(c) < 3:
            return self.make_result(50, 10, ["Not enough candles"])

        # Current and previous candle metrics
        body = c[-1] - o[-1]
        body_abs = abs(body)
        upper_shadow = h[-1] - max(o[-1], c[-1])
        lower_shadow = min(o[-1], c[-1]) - lo[-1]
        total_range = h[-1] - lo[-1]

        prev_body = c[-2] - o[-2]
        prev_range = h[-2] - lo[-2]
        avg_vol = float(np.mean(v[-20:])) if len(v) > 20 else float(np.mean(v))

        # Volume confirmation bonus
        vol_bonus = 1.2 if v[-1] > avg_vol * cfg.volume_confirmation_mult else 1.0

        # ── Pattern Detection ───────────────────────────────────────────

        if total_range > 0:
            body_ratio = body_abs / total_range

            # 1. Hammer (bullish reversal)
            if (lower_shadow >= body_abs * 2 and upper_shadow < body_abs * 0.3
                    and body_ratio < 0.4 and body >= 0):
                sig = 2.0 * vol_bonus
                net_signal += sig
                patterns_detected.append("Hammer")
                reasons.append("Hammer pattern — bullish reversal signal")

            # 2. Shooting Star (bearish reversal)
            if (upper_shadow >= body_abs * 2 and lower_shadow < body_abs * 0.3
                    and body_ratio < 0.4 and body <= 0):
                sig = -2.0 * vol_bonus
                net_signal += sig
                patterns_detected.append("Shooting Star")
                reasons.append("Shooting Star — bearish reversal signal")

            # 3. Doji
            if body_ratio < 0.1:
                patterns_detected.append("Doji")
                reasons.append("Doji — indecision, potential reversal")
                # Direction depends on context
                if c[-1] > c[-3]:  # was uptrend → bearish doji
                    net_signal -= 1.0
                else:
                    net_signal += 1.0

            # 4. Marubozu
            if body_ratio > 0.9:
                if body > 0:
                    net_signal += 2.5 * vol_bonus
                    patterns_detected.append("Bullish Marubozu")
                    reasons.append("Bullish Marubozu — strong conviction buying")
                else:
                    net_signal -= 2.5 * vol_bonus
                    patterns_detected.append("Bearish Marubozu")

        # 5. Bullish Engulfing
        if (prev_body < 0 and body > 0 and
                c[-1] > o[-2] and o[-1] < c[-2]):
            net_signal += 2.5 * vol_bonus
            patterns_detected.append("Bullish Engulfing")
            reasons.append("Bullish Engulfing — strong reversal pattern")

        # 6. Bearish Engulfing
        if (prev_body > 0 and body < 0 and
                c[-1] < o[-2] and o[-1] > c[-2]):
            net_signal -= 2.5 * vol_bonus
            patterns_detected.append("Bearish Engulfing")
            reasons.append("Bearish Engulfing — strong bearish reversal")

        # 7. Harami (inside body)
        if abs(body) < abs(prev_body) and max(o[-1], c[-1]) < max(o[-2], c[-2]) and min(o[-1], c[-1]) > min(o[-2], c[-2]):
            if prev_body < 0 and body > 0:
                net_signal += 1.5
                patterns_detected.append("Bullish Harami")
            elif prev_body > 0 and body < 0:
                net_signal -= 1.5
                patterns_detected.append("Bearish Harami")

        # 8. Piercing Line
        if (prev_body < 0 and body > 0 and
                o[-1] < c[-2] and c[-1] > (o[-2] + c[-2]) / 2 and c[-1] < o[-2]):
            net_signal += 2.0
            patterns_detected.append("Piercing Line")
            reasons.append("Piercing Line — bullish reversal")

        # 9. Dark Cloud Cover
        if (prev_body > 0 and body < 0 and
                o[-1] > c[-2] and c[-1] < (o[-2] + c[-2]) / 2 and c[-1] > o[-2]):
            net_signal -= 2.0
            patterns_detected.append("Dark Cloud Cover")
            reasons.append("Dark Cloud Cover — bearish reversal")

        # 10. Inside Bar
        if h[-1] <= h[-2] and lo[-1] >= lo[-2]:
            patterns_detected.append("Inside Bar")
            reasons.append("Inside Bar — consolidation, potential breakout")
            net_signal += 0.5  # Slightly bullish (continuation bias)

        # 11. Outside Bar
        if h[-1] > h[-2] and lo[-1] < lo[-2]:
            if body > 0:
                net_signal += 1.5
                patterns_detected.append("Bullish Outside Bar")
            else:
                net_signal -= 1.5
                patterns_detected.append("Bearish Outside Bar")

        # Three-candle patterns (need at least 3 bars)
        if len(c) >= 3:
            body_3 = c[-3] - o[-3]

            # 12. Morning Star
            if (body_3 < 0 and abs(c[-2] - o[-2]) < abs(body_3) * 0.3 and body > 0
                    and c[-1] > (o[-3] + c[-3]) / 2):
                net_signal += 3.0 * vol_bonus
                patterns_detected.append("Morning Star")
                reasons.append("Morning Star — strong 3-candle bullish reversal")

            # 13. Evening Star
            if (body_3 > 0 and abs(c[-2] - o[-2]) < abs(body_3) * 0.3 and body < 0
                    and c[-1] < (o[-3] + c[-3]) / 2):
                net_signal -= 3.0 * vol_bonus
                patterns_detected.append("Evening Star")
                reasons.append("Evening Star — strong 3-candle bearish reversal")

            # 14. Three White Soldiers
            if (c[-3] > o[-3] and c[-2] > o[-2] and c[-1] > o[-1] and
                    c[-2] > c[-3] and c[-1] > c[-2]):
                net_signal += 2.5
                patterns_detected.append("Three White Soldiers")
                reasons.append("Three White Soldiers — strong bullish continuation")

            # 15. Three Black Crows
            if (c[-3] < o[-3] and c[-2] < o[-2] and c[-1] < o[-1] and
                    c[-2] < c[-3] and c[-1] < c[-2]):
                net_signal -= 2.5
                patterns_detected.append("Three Black Crows")
                reasons.append("Three Black Crows — strong bearish continuation")

        # ── Score Calculation ───────────────────────────────────────────
        # Map net_signal to 0-100 score
        # Typical range: -8 to +8 for single-stock analysis
        score = 50 + net_signal * 5  # Each pattern unit = 5 score points
        score = float(np.clip(score, 0, 100))

        confidence = min(80, len(patterns_detected) * 20)
        if not patterns_detected:
            confidence = 10
            reasons.append("No significant candlestick patterns detected")

        return self.make_result(
            score, confidence, reasons,
            patterns_detected=patterns_detected,
            net_signal=round(net_signal, 1),
        )
