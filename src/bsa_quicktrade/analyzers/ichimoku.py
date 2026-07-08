"""Module 9 — Ichimoku Cloud Analysis.

Evaluates the complete Ichimoku Kinko Hyo system: cloud position,
TK cross, future cloud direction, Chikou span, and cloud twist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]

from bsa_quicktrade.analyzers.base import BaseAnalyzer
from bsa_quicktrade.core.models import AnalysisResult, StockData


class IchimokuAnalyzer(BaseAnalyzer):
    name = "ichimoku"

    def analyze(self, data: StockData) -> AnalysisResult:
        _, high, low, close, _ = self.get_ohlcv(data.daily)
        if close is None or len(close) < 80:
            return self.make_result(50, 10, ["Insufficient data for Ichimoku (need 80+ bars)"])

        cfg = self.config.ichimoku
        sub: dict[str, float] = {}
        reasons: list[str] = []
        price = float(close.iloc[-1])

        # Calculate Ichimoku components
        ich = ta.ichimoku(high, low, close,
                          tenkan=cfg.tenkan_period,
                          kijun=cfg.kijun_period,
                          senkou=cfg.senkou_b_period)

        if ich is None or len(ich) < 2:
            return self.make_result(50, 15, ["Ichimoku calculation failed"])

        # ich returns tuple: (ichimoku_df, span_df)
        ich_df = ich[0] if isinstance(ich, tuple) else ich

        # Find column names dynamically
        cols = ich_df.columns.tolist()
        tenkan_col = next((c for c in cols if "ITS" in c or "tenkan" in c.lower()), None)
        kijun_col = next((c for c in cols if "IKS" in c or "kijun" in c.lower()), None)
        span_a_col = next((c for c in cols if "ISA" in c or "senkou" in c.lower() and "a" in c.lower()), None)
        span_b_col = next((c for c in cols if "ISB" in c or "senkou" in c.lower() and "b" in c.lower()), None)
        chikou_col = next((c for c in cols if "ICS" in c or "chikou" in c.lower()), None)

        # Fallback to positional if named detection fails
        if tenkan_col is None and len(cols) >= 5:
            tenkan_col, kijun_col = cols[0], cols[1]
            span_a_col, span_b_col = cols[2], cols[3]
            chikou_col = cols[4] if len(cols) > 4 else None

        # ── 1. Cloud Position (30%) ─────────────────────────────────────
        if span_a_col and span_b_col:
            span_a = float(ich_df[span_a_col].iloc[-1])
            span_b = float(ich_df[span_b_col].iloc[-1])
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)

            if price > cloud_top:
                cloud_dist = (price - cloud_top) / price * 100
                sub["cloud"] = 75 + min(cloud_dist * 3, 20)
                reasons.append(f"Price above Ichimoku cloud — bullish ({cloud_dist:.1f}% above)")
            elif price < cloud_bottom:
                cloud_dist = (cloud_bottom - price) / price * 100
                sub["cloud"] = 25 - min(cloud_dist * 3, 20)
                reasons.append(f"Price below Ichimoku cloud — bearish")
            else:
                sub["cloud"] = 50
                reasons.append("Price inside Ichimoku cloud — neutral zone")
        else:
            sub["cloud"] = 50

        # ── 2. TK Cross (20%) ──────────────────────────────────────────
        if tenkan_col and kijun_col:
            tenkan = ich_df[tenkan_col].dropna()
            kijun = ich_df[kijun_col].dropna()

            if len(tenkan) > 2 and len(kijun) > 2:
                tk_diff = float(tenkan.iloc[-1]) - float(kijun.iloc[-1])
                tk_prev = float(tenkan.iloc[-2]) - float(kijun.iloc[-2])

                if tk_diff > 0 and tk_prev <= 0:
                    sub["tk_cross"] = 85
                    reasons.append("Tenkan-Kijun bullish cross — buy signal")
                elif tk_diff < 0 and tk_prev >= 0:
                    sub["tk_cross"] = 15
                    reasons.append("Tenkan-Kijun bearish cross — sell signal")
                elif tk_diff > 0:
                    sub["tk_cross"] = 65
                else:
                    sub["tk_cross"] = 35
            else:
                sub["tk_cross"] = 50
        else:
            sub["tk_cross"] = 50

        # ── 3. Future Cloud Direction (20%) ─────────────────────────────
        if span_a_col and span_b_col and len(ich_df) > 5:
            # Future cloud: compare recent span A vs span B values
            future_a = float(ich_df[span_a_col].iloc[-1])
            future_b = float(ich_df[span_b_col].iloc[-1])
            prev_a = float(ich_df[span_a_col].iloc[-5])
            prev_b = float(ich_df[span_b_col].iloc[-5])

            if future_a > future_b:
                sub["future_cloud"] = 70
                if prev_a <= prev_b:
                    reasons.append("Future cloud turning bullish — cloud twist")
            elif future_a < future_b:
                sub["future_cloud"] = 30
                if prev_a >= prev_b:
                    reasons.append("Future cloud turning bearish — cloud twist")
            else:
                sub["future_cloud"] = 50
        else:
            sub["future_cloud"] = 50

        # ── 4. Chikou Span (15%) ───────────────────────────────────────
        if chikou_col:
            chikou_vals = ich_df[chikou_col].dropna()
            if len(chikou_vals) > cfg.displacement and len(close) > cfg.displacement:
                chikou_val = float(chikou_vals.iloc[-1])
                price_26_ago = float(close.iloc[-cfg.displacement]) if len(close) > cfg.displacement else price

                if chikou_val > price_26_ago:
                    sub["chikou"] = 70
                    reasons.append("Chikou span above price — bullish confirmation")
                else:
                    sub["chikou"] = 30
            else:
                sub["chikou"] = 50
        else:
            sub["chikou"] = 50

        # ── 5. Cloud Twist Detection (15%) ─────────────────────────────
        if span_a_col and span_b_col and len(ich_df) > 10:
            twist_score = 50.0
            for i in range(-1, max(-11, -len(ich_df)), -1):
                try:
                    a_curr = float(ich_df[span_a_col].iloc[i])
                    b_curr = float(ich_df[span_b_col].iloc[i])
                    a_prev = float(ich_df[span_a_col].iloc[i - 1])
                    b_prev = float(ich_df[span_b_col].iloc[i - 1])

                    if a_curr > b_curr and a_prev <= b_prev:
                        twist_score = 75
                        reasons.append("Recent bullish cloud twist — trend change signal")
                        break
                    elif a_curr < b_curr and a_prev >= b_prev:
                        twist_score = 25
                        reasons.append("Recent bearish cloud twist")
                        break
                except (IndexError, ValueError):
                    break
            sub["twist"] = twist_score
        else:
            sub["twist"] = 50

        # ── Combine ─────────────────────────────────────────────────────
        weights = {
            "cloud": 0.30, "tk_cross": 0.20, "future_cloud": 0.20,
            "chikou": 0.15, "twist": 0.15,
        }
        overall = sum(sub.get(k, 50) * w for k, w in weights.items())
        confidence = min(80, len([s for s in sub.values() if abs(s - 50) > 10]) * 16)

        return self.make_result(overall, confidence, reasons)
