"""Module 4 — Volume Analysis.

Evaluates buying/selling pressure through volume indicators,
smart-money accumulation, VWAP, and delivery percentage trends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]

from bsa_quicktrade.analyzers.base import BaseAnalyzer
from bsa_quicktrade.core.models import AnalysisResult, StockData


class VolumeAnalyzer(BaseAnalyzer):
    name = "volume"

    def analyze(self, data: StockData) -> AnalysisResult:
        open_, high, low, close, volume = self.get_ohlcv(data.daily)
        if close is None or volume is None or len(close) < 30:
            return self.make_result(50, 10, ["Insufficient data for volume analysis"])

        cfg = self.config.volume
        sub: dict[str, float] = {}
        reasons: list[str] = []
        metadata: dict = {}

        avg_vol = float(volume.tail(cfg.avg_period).mean())

        # ── 1. RVOL & Volume Spike (15%) ────────────────────────────────
        if avg_vol > 0:
            rvol = float(volume.iloc[-1]) / avg_vol
            metadata["rvol"] = round(rvol, 2)

            if rvol >= cfg.spike_multiplier:
                sub["rvol"] = 85
                reasons.append(f"Volume spike — RVOL {rvol:.1f}x average")
            elif rvol >= 1.5:
                sub["rvol"] = 70
                reasons.append(f"Above-average volume — RVOL {rvol:.1f}x")
            elif rvol <= cfg.dryup_threshold:
                sub["rvol"] = 65  # dry-up can be bullish (pre-breakout)
                reasons.append(f"Volume dry-up — RVOL {rvol:.1f}x (potential accumulation)")
            else:
                sub["rvol"] = 50
        else:
            sub["rvol"] = 50

        # ── 2. VWAP (15%) ──────────────────────────────────────────────
        # Calculate session VWAP (approximation using daily typical price)
        tp = (high + low + close) / 3
        cum_tpv = (tp * volume).cumsum()
        cum_vol = volume.cumsum()
        vwap = cum_tpv / cum_vol
        if len(vwap.dropna()) > 0:
            vwap_val = float(vwap.iloc[-1])
            price = float(close.iloc[-1])
            metadata["vwap"] = round(vwap_val, 2)

            if price > vwap_val * 1.01:
                sub["vwap"] = 70
                reasons.append(f"Price above VWAP (₹{vwap_val:,.0f}) — bullish control")
            elif price < vwap_val * 0.99:
                sub["vwap"] = 30
                reasons.append(f"Price below VWAP — bearish control")
            else:
                sub["vwap"] = 50
        else:
            sub["vwap"] = 50

        # ── 3. OBV Trend (20%) ──────────────────────────────────────────
        obv = ta.obv(close, volume)
        if obv is not None and len(obv.dropna()) > 20:
            obv_sma = obv.rolling(20).mean()
            obv_val = float(obv.iloc[-1])
            obv_sma_val = float(obv_sma.iloc[-1])

            # OBV trend
            obv_slope = float(obv.iloc[-1] - obv.iloc[-10]) if len(obv) > 10 else 0
            price_slope = float(close.iloc[-1] - close.iloc[-10]) if len(close) > 10 else 0

            if obv_val > obv_sma_val and obv_slope > 0:
                sub["obv"] = 70
                reasons.append("OBV rising above SMA — accumulation")
            elif obv_val < obv_sma_val and obv_slope < 0:
                sub["obv"] = 30
                reasons.append("OBV falling below SMA — distribution")
            else:
                sub["obv"] = 50

            # Smart money: OBV rising while price flat/declining
            if obv_slope > 0 and price_slope <= 0:
                sub["obv"] = max(sub["obv"], 80)
                reasons.append("Smart money accumulation — OBV rising despite flat/falling price")
                metadata["smart_money"] = True
        else:
            sub["obv"] = 50

        # ── 4. Chaikin Money Flow (15%) ──────────────────────────────────
        cmf = ta.cmf(high, low, close, volume, length=cfg.cmf_period)
        if cmf is not None and len(cmf.dropna()) > 0:
            cmf_val = float(cmf.iloc[-1])
            metadata["cmf"] = round(cmf_val, 3)

            if cmf_val > 0.1:
                sub["cmf"] = 75
                reasons.append(f"CMF {cmf_val:.2f} — strong buying pressure")
            elif cmf_val > 0:
                sub["cmf"] = 60
            elif cmf_val < -0.1:
                sub["cmf"] = 25
                reasons.append(f"CMF {cmf_val:.2f} — selling pressure")
            elif cmf_val < 0:
                sub["cmf"] = 40
            else:
                sub["cmf"] = 50
        else:
            sub["cmf"] = 50

        # ── 5. MFI (10%) ───────────────────────────────────────────────
        mfi = ta.mfi(high, low, close, volume, length=cfg.mfi_period)
        if mfi is not None and len(mfi.dropna()) > 0:
            mfi_val = float(mfi.iloc[-1])
            metadata["mfi"] = round(mfi_val, 1)

            if mfi_val > 80:
                sub["mfi"] = 30  # overbought on volume
            elif mfi_val < 20:
                sub["mfi"] = 75  # oversold on volume — potential bounce
                reasons.append(f"MFI {mfi_val:.0f} — volume-weighted oversold")
            elif mfi_val > 50:
                sub["mfi"] = 60
            else:
                sub["mfi"] = 40
        else:
            sub["mfi"] = 50

        # ── 6. Accumulation/Distribution (10%) ──────────────────────────
        ad = ta.ad(high, low, close, volume)
        if ad is not None and len(ad.dropna()) > 20:
            ad_slope = float(ad.iloc[-1] - ad.iloc[-10]) if len(ad) > 10 else 0
            if ad_slope > 0:
                sub["ad"] = 65
            elif ad_slope < 0:
                sub["ad"] = 35
            else:
                sub["ad"] = 50
        else:
            sub["ad"] = 50

        # ── 7. Volume Climax (5%) ──────────────────────────────────────
        if avg_vol > 0:
            last_vol = float(volume.iloc[-1])
            if last_vol > avg_vol * cfg.climax_volume_mult:
                # Climax + reversal candle?
                body = float(close.iloc[-1]) - float(open_.iloc[-1]) if open_ is not None else 0
                prev_body = float(close.iloc[-2]) - float(open_.iloc[-2]) if open_ is not None and len(close) > 1 else 0

                if body * prev_body < 0:  # reversal
                    sub["climax"] = 80
                    reasons.append(f"Volume climax ({last_vol/avg_vol:.0f}x avg) with reversal candle")
                else:
                    sub["climax"] = 65
            else:
                sub["climax"] = 50
        else:
            sub["climax"] = 50

        # ── 8. Delivery % Analysis (10%) ────────────────────────────────
        if data.delivery_data is not None and len(data.delivery_data) > 3:
            try:
                del_df = data.delivery_data
                # Try to find delivery % column
                del_col = None
                for c in del_df.columns:
                    if "dly" in str(c).lower() or "delivery" in str(c).lower() and "%" in str(c):
                        del_col = c
                        break

                if del_col is not None:
                    del_pcts = pd.to_numeric(del_df[del_col], errors="coerce").dropna()
                    if len(del_pcts) >= 3:
                        recent_avg = float(del_pcts.tail(3).mean())
                        older_avg = float(del_pcts.tail(cfg.delivery_lookback).mean())
                        metadata["delivery_pct_recent"] = round(recent_avg, 1)
                        metadata["delivery_pct_avg"] = round(older_avg, 1)

                        if recent_avg > older_avg + cfg.delivery_increase_threshold:
                            sub["delivery"] = 80
                            reasons.append(f"Delivery % rising ({recent_avg:.0f}% vs avg {older_avg:.0f}%) — institutional interest")
                        elif recent_avg > 50:
                            sub["delivery"] = 65
                            reasons.append(f"High delivery % ({recent_avg:.0f}%) — conviction buying")
                        else:
                            sub["delivery"] = 50
                    else:
                        sub["delivery"] = 50
                else:
                    sub["delivery"] = 50
            except Exception:
                sub["delivery"] = 50
        else:
            sub["delivery"] = 50

        # ── Combine ─────────────────────────────────────────────────────
        weights = {
            "obv": 0.20, "vwap": 0.15, "rvol": 0.15, "cmf": 0.15,
            "mfi": 0.10, "ad": 0.05, "delivery": 0.10, "climax": 0.05,
        }
        # Give extra weight to smart money if detected
        if metadata.get("smart_money"):
            weights["obv"] = 0.30
            weights["cmf"] = 0.10

        overall = sum(sub.get(k, 50) * w for k, w in weights.items())
        confidence = min(85, len([s for s in sub.values() if abs(s - 50) > 10]) * 12)

        return self.make_result(overall, confidence, reasons, **metadata)
