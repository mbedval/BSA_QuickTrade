"""Module 2 — Momentum Analysis.

Evaluates momentum via RSI, Stochastic RSI, MACD, CCI, ROC,
and divergence detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]
from scipy.signal import argrelextrema

from rsa_quicktrade.analyzers.base import BaseAnalyzer
from rsa_quicktrade.core.models import AnalysisResult, StockData


class MomentumAnalyzer(BaseAnalyzer):
    name = "momentum"

    def analyze(self, data: StockData) -> AnalysisResult:
        open_, high, low, close, volume = self.get_ohlcv(data.daily)
        if close is None or len(close) < 30:
            return self.make_result(50, 10, ["Insufficient data for momentum analysis"])

        cfg = self.config.momentum
        sub: dict[str, float] = {}
        reasons: list[str] = []

        # ── 1. RSI (25%) ────────────────────────────────────────────────
        rsi = ta.rsi(close, length=cfg.rsi_period)
        if rsi is not None and len(rsi.dropna()) > 0:
            rsi_val = float(rsi.iloc[-1])
            rsi_prev = float(rsi.iloc[-2]) if len(rsi) > 1 else rsi_val

            if rsi_val <= cfg.rsi_oversold:
                rsi_score = 80  # oversold — potential reversal bullish
                reasons.append(f"RSI {rsi_val:.0f} — oversold, potential bounce")
            elif rsi_val >= cfg.rsi_overbought:
                rsi_score = 20  # overbought
                reasons.append(f"RSI {rsi_val:.0f} — overbought, potential pullback")
            elif 40 <= rsi_val <= 60:
                rsi_score = 50
            elif rsi_val > 50:
                rsi_score = 50 + (rsi_val - 50) * 1.0  # 50-70 maps to 50-70
                if rsi_val > rsi_prev:
                    reasons.append(f"RSI {rsi_val:.0f} rising — bullish momentum")
            else:
                rsi_score = 50 - (50 - rsi_val) * 1.0
                if rsi_val < rsi_prev:
                    reasons.append(f"RSI {rsi_val:.0f} falling — bearish momentum")

            sub["rsi"] = float(np.clip(rsi_score, 0, 100))
        else:
            sub["rsi"] = 50

        # ── 2. Stochastic RSI (15%) ─────────────────────────────────────
        stoch = ta.stochrsi(close, length=cfg.stoch_rsi_period, rsi_length=cfg.stoch_rsi_period,
                            k=cfg.stoch_rsi_k, d=cfg.stoch_rsi_d)
        if stoch is not None and len(stoch.dropna()) > 0:
            cols = stoch.columns.tolist()
            k_val = float(stoch[cols[0]].iloc[-1])
            d_val = float(stoch[cols[1]].iloc[-1]) if len(cols) > 1 else k_val

            if k_val < 20 and k_val > d_val:
                stoch_score = 80
                reasons.append("StochRSI bullish crossover from oversold")
            elif k_val > 80 and k_val < d_val:
                stoch_score = 20
                reasons.append("StochRSI bearish crossover from overbought")
            elif k_val > d_val:
                stoch_score = 60
            elif k_val < d_val:
                stoch_score = 40
            else:
                stoch_score = 50
            sub["stoch_rsi"] = float(np.clip(stoch_score, 0, 100))
        else:
            sub["stoch_rsi"] = 50

        # ── 3. MACD (25%) ───────────────────────────────────────────────
        macd_data = ta.macd(close, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal)
        if macd_data is not None and len(macd_data.dropna()) > 0:
            cols = macd_data.columns.tolist()
            macd_line = float(macd_data[cols[0]].iloc[-1])
            signal_line = float(macd_data[cols[1]].iloc[-1])
            histogram = float(macd_data[cols[2]].iloc[-1])
            hist_prev = float(macd_data[cols[2]].iloc[-2]) if len(macd_data) > 1 else 0

            macd_score = 50.0
            # Histogram direction
            if histogram > 0 and histogram > hist_prev:
                macd_score += 15
            elif histogram < 0 and histogram < hist_prev:
                macd_score -= 15

            # Signal crossover
            if macd_line > signal_line:
                macd_score += 15
                reasons.append("MACD above signal line — bullish")
            else:
                macd_score -= 15

            # Zero line
            if macd_line > 0:
                macd_score += 10
            else:
                macd_score -= 10

            # Histogram momentum
            if histogram > 0 and histogram > hist_prev:
                reasons.append("MACD histogram expanding — strengthening momentum")
            elif histogram < 0 and histogram < hist_prev:
                reasons.append("MACD histogram expanding bearish")

            sub["macd"] = float(np.clip(macd_score, 0, 100))
        else:
            sub["macd"] = 50

        # ── 4. CCI (10%) ────────────────────────────────────────────────
        cci = ta.cci(high, low, close, length=cfg.cci_period)
        if cci is not None and len(cci.dropna()) > 0:
            cci_val = float(cci.iloc[-1])
            if cci_val > cfg.cci_overbought:
                cci_score = 70  # strong bullish momentum
                reasons.append(f"CCI {cci_val:.0f} — strong bullish momentum")
            elif cci_val < cfg.cci_oversold:
                cci_score = 30  # strong bearish
            elif cci_val > 0:
                cci_score = 50 + (cci_val / cfg.cci_overbought) * 20
            else:
                cci_score = 50 + (cci_val / abs(cfg.cci_oversold)) * 20
            sub["cci"] = float(np.clip(cci_score, 0, 100))
        else:
            sub["cci"] = 50

        # ── 5. ROC (10%) ────────────────────────────────────────────────
        roc = ta.roc(close, length=cfg.roc_period)
        if roc is not None and len(roc.dropna()) > 0:
            roc_val = float(roc.iloc[-1])
            roc_prev = float(roc.iloc[-2]) if len(roc) > 1 else 0
            if roc_val > 0 and roc_val > roc_prev:
                roc_score = 65 + min(roc_val * 2, 30)
            elif roc_val > 0:
                roc_score = 55 + min(roc_val, 15)
            elif roc_val < 0 and roc_val < roc_prev:
                roc_score = 35 + max(roc_val * 2, -30)
            else:
                roc_score = 45 + max(roc_val, -15)
            sub["roc"] = float(np.clip(roc_score, 0, 100))
        else:
            sub["roc"] = 50

        # ── 6. Divergence Detection (15%) ───────────────────────────────
        div_score = 50.0
        if rsi is not None and len(rsi.dropna()) > cfg.divergence_lookback:
            lookback = cfg.divergence_lookback
            price_seg = close.iloc[-lookback:].values.astype(float)
            rsi_seg = rsi.iloc[-lookback:].values.astype(float)

            # Remove NaNs
            mask = ~np.isnan(rsi_seg)
            if mask.sum() > 20:
                price_seg = price_seg[mask]
                rsi_seg = rsi_seg[mask]

                order = max(3, len(price_seg) // 10)
                try:
                    p_highs = argrelextrema(price_seg, np.greater_equal, order=order)[0]
                    p_lows = argrelextrema(price_seg, np.less_equal, order=order)[0]
                    r_highs = argrelextrema(rsi_seg, np.greater_equal, order=order)[0]
                    r_lows = argrelextrema(rsi_seg, np.less_equal, order=order)[0]

                    # Regular bearish: Price HH but RSI LH
                    if len(p_highs) >= 2 and len(r_highs) >= 2:
                        if (price_seg[p_highs[-1]] > price_seg[p_highs[-2]] and
                                rsi_seg[r_highs[-1]] < rsi_seg[r_highs[-2]]):
                            div_score = 30
                            reasons.append("Bearish RSI divergence — price HH but RSI LH")

                    # Regular bullish: Price LL but RSI HL
                    if len(p_lows) >= 2 and len(r_lows) >= 2:
                        if (price_seg[p_lows[-1]] < price_seg[p_lows[-2]] and
                                rsi_seg[r_lows[-1]] > rsi_seg[r_lows[-2]]):
                            div_score = 75
                            reasons.append("Bullish RSI divergence — price LL but RSI HL")

                    # Hidden bullish: Price HL but RSI LL
                    if len(p_lows) >= 2 and len(r_lows) >= 2:
                        if (price_seg[p_lows[-1]] > price_seg[p_lows[-2]] and
                                rsi_seg[r_lows[-1]] < rsi_seg[r_lows[-2]]):
                            div_score = max(div_score, 70)
                            reasons.append("Hidden bullish divergence detected")
                except (ValueError, IndexError):
                    pass

        sub["divergence"] = div_score

        # ── Combine ─────────────────────────────────────────────────────
        weights = {
            "rsi": 0.25, "macd": 0.25, "stoch_rsi": 0.15,
            "cci": 0.10, "roc": 0.10, "divergence": 0.15,
        }
        overall = sum(sub.get(k, 50) * w for k, w in weights.items())
        confidence = min(85, len([s for s in sub.values() if abs(s - 50) > 10]) * 14)

        return self.make_result(overall, confidence, reasons)
