"""Module 3 — Volatility Analysis.

Measures volatility state (squeeze / expansion) to identify
high-probability breakout opportunities via ATR, Bollinger Bands,
Keltner Channels, and narrow-range patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # type: ignore[import-untyped]

from bsa_quicktrade.analyzers.base import BaseAnalyzer
from bsa_quicktrade.core.models import AnalysisResult, StockData


class VolatilityAnalyzer(BaseAnalyzer):
    name = "volatility"

    def analyze(self, data: StockData) -> AnalysisResult:
        open_, high, low, close, volume = self.get_ohlcv(data.daily)
        if close is None or len(close) < 30:
            return self.make_result(50, 10, ["Insufficient data for volatility analysis"])

        cfg = self.config.volatility
        sub: dict[str, float] = {}
        reasons: list[str] = []
        metadata: dict = {}

        # ── 1. ATR Expansion / Contraction (20%) ────────────────────────
        atr = ta.atr(high, low, close, length=cfg.atr_period)
        if atr is not None and len(atr.dropna()) > 10:
            atr_val = float(atr.iloc[-1])
            atr_avg = float(atr.tail(50).mean())
            atr_ratio = atr_val / atr_avg if atr_avg > 0 else 1.0
            metadata["atr"] = round(atr_val, 2)
            metadata["atr_pct"] = round(atr_val / float(close.iloc[-1]) * 100, 2)

            if atr_ratio > 1.3:
                sub["atr"] = 75  # expanding — active move
                reasons.append(f"ATR expanding ({atr_ratio:.1f}x avg) — active volatility")
            elif atr_ratio < 0.7:
                sub["atr"] = 80  # contracting — breakout imminent
                reasons.append(f"ATR contracting ({atr_ratio:.1f}x avg) — breakout setup")
            else:
                sub["atr"] = 50
        else:
            sub["atr"] = 50

        # ── 2. Historical Volatility (10%) ──────────────────────────────
        if len(close) > cfg.historical_vol_period:
            log_returns = np.log(close / close.shift(1)).dropna()
            if len(log_returns) > cfg.historical_vol_period:
                hv = float(log_returns.tail(cfg.historical_vol_period).std() * np.sqrt(252) * 100)
                hv_avg = float(log_returns.tail(50).std() * np.sqrt(252) * 100)
                metadata["historical_volatility"] = round(hv, 1)

                if hv > hv_avg * 1.3:
                    sub["hv"] = 70
                elif hv < hv_avg * 0.7:
                    sub["hv"] = 80  # low vol = compression
                    reasons.append(f"Historical vol {hv:.0f}% compressed — expansion likely")
                else:
                    sub["hv"] = 50
            else:
                sub["hv"] = 50
        else:
            sub["hv"] = 50

        # ── 3. Bollinger Band Width & Squeeze (15%) ─────────────────────
        bb = ta.bbands(close, length=cfg.bb_period, std=cfg.bb_std)
        if bb is not None and len(bb.dropna()) > 10:
            cols = bb.columns.tolist()
            # Find upper, middle, lower bands
            bbu = bb[[c for c in cols if "BBU" in c.upper()]]
            bbm = bb[[c for c in cols if "BBM" in c.upper()]]
            bbl = bb[[c for c in cols if "BBL" in c.upper()]]

            if len(bbu.columns) > 0 and len(bbl.columns) > 0 and len(bbm.columns) > 0:
                upper = bbu.iloc[:, 0]
                middle = bbm.iloc[:, 0]
                lower = bbl.iloc[:, 0]

                bb_width = (upper - lower) / middle
                bb_width_val = float(bb_width.iloc[-1])
                bb_width_min = float(bb_width.tail(20).min())

                metadata["bb_width"] = round(bb_width_val, 4)

                if bb_width_val <= bb_width_min * 1.05:
                    sub["bb"] = 85  # Squeeze!
                    reasons.append(f"Bollinger Band squeeze — width at 20-day low")
                elif bb_width_val > float(bb_width.tail(20).mean()) * 1.5:
                    sub["bb"] = 65  # Expansion
                    reasons.append("BB width expanding — active move in progress")
                else:
                    sub["bb"] = 50
            else:
                sub["bb"] = 50
        else:
            sub["bb"] = 50

        # ── 4. Keltner Squeeze (TTM Squeeze) (30%) ─────────────────────
        kc = ta.kc(high, low, close, length=cfg.keltner_period,
                   scalar=cfg.keltner_atr_mult)
        squeeze_on = False
        if kc is not None and bb is not None and len(kc.dropna()) > 0:
            kc_cols = kc.columns.tolist()
            kcu = kc[[c for c in kc_cols if "KCU" in c.upper()]]
            kcl = kc[[c for c in kc_cols if "KCL" in c.upper()]]

            bb_cols = bb.columns.tolist()
            bbu_s = bb[[c for c in bb_cols if "BBU" in c.upper()]]
            bbl_s = bb[[c for c in bb_cols if "BBL" in c.upper()]]

            if len(kcu.columns) > 0 and len(bbu_s.columns) > 0:
                # Squeeze = BB inside KC
                bb_upper = float(bbu_s.iloc[-1, 0])
                bb_lower = float(bbl_s.iloc[-1, 0])
                kc_upper = float(kcu.iloc[-1, 0])
                kc_lower = float(kcl.iloc[-1, 0])

                squeeze_on = bb_upper < kc_upper and bb_lower > kc_lower

                # Check for squeeze release (was on, now off)
                if len(bbu_s) > 5 and len(kcu) > 5:
                    prev_squeeze = (float(bbu_s.iloc[-3, 0]) < float(kcu.iloc[-3, 0]) and
                                    float(bbl_s.iloc[-3, 0]) > float(kcl.iloc[-3, 0]))
                    if prev_squeeze and not squeeze_on:
                        sub["squeeze"] = 90  # Squeeze just fired!
                        reasons.append("TTM Squeeze FIRED — high probability breakout")
                    elif squeeze_on:
                        sub["squeeze"] = 80
                        reasons.append("TTM Squeeze ON — volatility compressed, breakout building")
                    else:
                        sub["squeeze"] = 50
                else:
                    sub["squeeze"] = 70 if squeeze_on else 50
            else:
                sub["squeeze"] = 50
        else:
            sub["squeeze"] = 50

        # ── 5. NR4 / NR7 Patterns (15%) ────────────────────────────────
        ranges = (high - low).tail(max(cfg.nr_lookback_7 + 1, 10))
        if len(ranges) >= cfg.nr_lookback_7 + 1:
            today_range = float(ranges.iloc[-1])
            nr4 = today_range < float(ranges.iloc[-cfg.nr_lookback_4 - 1:-1].min())
            nr7 = today_range < float(ranges.iloc[-cfg.nr_lookback_7 - 1:-1].min())

            if nr7:
                sub["nr"] = 90
                reasons.append("NR7 pattern — narrowest range in 7 days, breakout imminent")
            elif nr4:
                sub["nr"] = 80
                reasons.append("NR4 pattern — narrowest range in 4 days")
            else:
                sub["nr"] = 50
        else:
            sub["nr"] = 50

        # ── 6. Gap Analysis (10%) ──────────────────────────────────────
        if len(close) > 2:
            prev_high = float(high.iloc[-2])
            prev_low = float(low.iloc[-2])
            curr_open = float(open_.iloc[-1]) if open_ is not None and len(open_) > 0 else 0
            curr_low = float(low.iloc[-1])
            curr_high = float(high.iloc[-1])

            gap_pct = 0.0
            if curr_low > prev_high:
                gap_pct = (curr_low - prev_high) / prev_high * 100
                if gap_pct > cfg.gap_threshold_pct:
                    sub["gap"] = 75
                    reasons.append(f"Gap up {gap_pct:.1f}% — momentum")
                else:
                    sub["gap"] = 55
            elif curr_high < prev_low:
                gap_pct = (prev_low - curr_high) / prev_low * 100
                if gap_pct > cfg.gap_threshold_pct:
                    sub["gap"] = 75  # gap = opportunity regardless of direction
                    reasons.append(f"Gap down {gap_pct:.1f}% — potential reversal play")
                else:
                    sub["gap"] = 55
            else:
                sub["gap"] = 50

            metadata["gap_pct"] = round(gap_pct, 2)
        else:
            sub["gap"] = 50

        # ── Combine (scores OPPORTUNITY, not direction) ─────────────────
        weights = {
            "squeeze": 0.30, "atr": 0.20, "bb": 0.15,
            "nr": 0.15, "gap": 0.10, "hv": 0.10,
        }
        overall = sum(sub.get(k, 50) * w for k, w in weights.items())
        confidence = min(85, len([s for s in sub.values() if s > 60]) * 15)

        return self.make_result(overall, confidence, reasons, **metadata)
