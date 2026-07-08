"""Module 11 — Options Analysis.

Analyzes NSE option chains: PCR, max pain, OI-based S/R,
IV rank, OI buildup classification, and unusual activity.
"""

from __future__ import annotations

import numpy as np

from rsa_quicktrade.analyzers.base import BaseAnalyzer
from rsa_quicktrade.core.models import AnalysisResult, PriceLevel, StockData


class OptionsAnalyzer(BaseAnalyzer):
    name = "options"

    def analyze(self, data: StockData) -> AnalysisResult:
        if not self.config.options.enabled:
            return self.make_result(50, 5, ["Options analysis disabled"])

        oc = data.option_chain
        if oc is None or not oc.get("records"):
            return self.make_result(50, 10, ["No option chain data available"])

        cfg = self.config.options
        sub: dict[str, float] = {}
        reasons: list[str] = []
        metadata: dict = {}

        records = oc["records"]
        underlying = float(oc.get("underlying_value", 0))
        total_ce = int(oc.get("total_ce_oi", 0))
        total_pe = int(oc.get("total_pe_oi", 0))

        if underlying <= 0:
            return self.make_result(50, 10, ["Invalid underlying price in option chain"])

        # ── 1. PCR (20%) ───────────────────────────────────────────────
        pcr = total_pe / total_ce if total_ce > 0 else 1.0
        metadata["pcr"] = round(pcr, 2)

        if pcr > 1.3:
            sub["pcr"] = 80
            reasons.append(f"PCR {pcr:.2f} — high put writing, strong support")
        elif pcr > 1.0:
            sub["pcr"] = 65
            reasons.append(f"PCR {pcr:.2f} — moderately bullish")
        elif pcr > 0.7:
            sub["pcr"] = 50
        elif pcr > 0.5:
            sub["pcr"] = 35
            reasons.append(f"PCR {pcr:.2f} — bearish bias")
        else:
            sub["pcr"] = 20
            reasons.append(f"PCR {pcr:.2f} — very low, bearish")

        # ── 2. Max Pain (15%) ──────────────────────────────────────────
        max_pain = self._calculate_max_pain(records, underlying)
        if max_pain > 0:
            metadata["max_pain"] = round(max_pain, 2)
            mp_dist = (max_pain - underlying) / underlying * 100

            if abs(mp_dist) < 1.0:
                sub["max_pain"] = 50
                reasons.append(f"Max Pain ₹{max_pain:,.0f} — price near max pain (neutral)")
            elif mp_dist > 0:
                sub["max_pain"] = 60  # Price below max pain — may drift up
                reasons.append(f"Max Pain ₹{max_pain:,.0f} — price may drift up toward max pain")
            else:
                sub["max_pain"] = 40
                reasons.append(f"Max Pain ₹{max_pain:,.0f} — price above, may pull back")
        else:
            sub["max_pain"] = 50

        # ── 3. OI-based Support / Resistance (20%) ─────────────────────
        max_pe_oi = 0
        max_pe_strike = 0
        max_ce_oi = 0
        max_ce_strike = 0

        for r in records:
            pe_oi = r.get("pe_oi", 0)
            ce_oi = r.get("ce_oi", 0)
            strike = r.get("strike", 0)

            if pe_oi > max_pe_oi and strike <= underlying:
                max_pe_oi = pe_oi
                max_pe_strike = strike
            if ce_oi > max_ce_oi and strike >= underlying:
                max_ce_oi = ce_oi
                max_ce_strike = strike

        metadata["oi_support"] = max_pe_strike
        metadata["oi_resistance"] = max_ce_strike
        metadata["support_levels"] = [PriceLevel(max_pe_strike, "support", 75, "option_oi")] if max_pe_strike else []
        metadata["resistance_levels"] = [PriceLevel(max_ce_strike, "resistance", 75, "option_oi")] if max_ce_strike else []

        if max_pe_strike and max_ce_strike:
            oi_range = (underlying - max_pe_strike) / underlying * 100
            reasons.append(f"OI Support ₹{max_pe_strike:,.0f} | OI Resistance ₹{max_ce_strike:,.0f}")

            # Score: price closer to support = bullish
            if oi_range < 2:
                sub["oi_sr"] = 70
            elif oi_range > 3:
                sub["oi_sr"] = 45
            else:
                sub["oi_sr"] = 55
        else:
            sub["oi_sr"] = 50

        # ── 4. ATM Analysis + OI Buildup (25%) ─────────────────────────
        atm_records = self._get_atm_strikes(records, underlying, cfg.strikes_range)
        if atm_records:
            total_ce_change = sum(r.get("ce_oi_change", 0) for r in atm_records)
            total_pe_change = sum(r.get("pe_oi_change", 0) for r in atm_records)

            # Get price change direction from daily data
            _, _, _, close, _ = self.get_ohlcv(data.daily)
            price_up = True
            if close is not None and len(close) > 1:
                price_up = float(close.iloc[-1]) >= float(close.iloc[-2])

            oi_increasing = (total_ce_change + total_pe_change) > 0

            if price_up and oi_increasing:
                sub["buildup"] = 80
                buildup_type = "Long Buildup"
                reasons.append("Long Buildup — price ↑ + OI ↑ (bullish)")
            elif not price_up and oi_increasing:
                sub["buildup"] = 20
                buildup_type = "Short Buildup"
                reasons.append("Short Buildup — price ↓ + OI ↑ (bearish)")
            elif not price_up and not oi_increasing:
                sub["buildup"] = 40
                buildup_type = "Long Unwinding"
                reasons.append("Long Unwinding — price ↓ + OI ↓")
            else:
                sub["buildup"] = 65
                buildup_type = "Short Covering"
                reasons.append("Short Covering — price ↑ + OI ↓")

            metadata["buildup_type"] = buildup_type
        else:
            sub["buildup"] = 50

        # ── 5. IV Analysis (10%) ───────────────────────────────────────
        ivs = [r.get("ce_iv", 0) for r in atm_records if r.get("ce_iv", 0) > 0]
        ivs += [r.get("pe_iv", 0) for r in atm_records if r.get("pe_iv", 0) > 0]

        if ivs:
            avg_iv = float(np.mean(ivs))
            metadata["atm_iv"] = round(avg_iv, 1)

            # High IV = expensive options
            if avg_iv > 40:
                sub["iv"] = 40
                reasons.append(f"ATM IV {avg_iv:.0f}% — high, potential mean reversion")
            elif avg_iv < 15:
                sub["iv"] = 70
                reasons.append(f"ATM IV {avg_iv:.0f}% — low, options cheap, breakout potential")
            else:
                sub["iv"] = 55
        else:
            sub["iv"] = 50

        # ── 6. Unusual OI (10%) ────────────────────────────────────────
        all_oi_changes = [abs(r.get("ce_oi_change", 0)) for r in records]
        all_oi_changes += [abs(r.get("pe_oi_change", 0)) for r in records]

        if all_oi_changes:
            mean_change = float(np.mean(all_oi_changes))
            std_change = float(np.std(all_oi_changes))
            threshold = mean_change + std_change * cfg.unusual_oi_std

            unusual = []
            for r in records:
                if abs(r.get("ce_oi_change", 0)) > threshold:
                    unusual.append(f"CE {r['strike']} (+{r['ce_oi_change']:,})")
                if abs(r.get("pe_oi_change", 0)) > threshold:
                    unusual.append(f"PE {r['strike']} (+{r['pe_oi_change']:,})")

            if unusual:
                sub["unusual"] = 70
                reasons.append(f"Unusual OI activity: {', '.join(unusual[:3])}")
                metadata["unusual_oi"] = unusual[:5]
            else:
                sub["unusual"] = 50
        else:
            sub["unusual"] = 50

        # ── Combine ─────────────────────────────────────────────────────
        weights = {
            "pcr": 0.20, "max_pain": 0.15, "oi_sr": 0.20,
            "buildup": 0.25, "iv": 0.10, "unusual": 0.10,
        }
        overall = sum(sub.get(k, 50) * w for k, w in weights.items())
        confidence = min(80, len([s for s in sub.values() if abs(s - 50) > 10]) * 14)

        return self.make_result(overall, confidence, reasons, **metadata)

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_max_pain(records: list[dict], underlying: float) -> float:
        """Calculate max pain — strike where total option buyer loss is minimized."""
        strikes = sorted(set(r["strike"] for r in records if r.get("strike")))
        if not strikes:
            return 0

        min_pain = float("inf")
        max_pain_strike = 0

        for test_strike in strikes:
            total_pain = 0
            for r in records:
                s = r.get("strike", 0)
                ce_oi = r.get("ce_oi", 0)
                pe_oi = r.get("pe_oi", 0)

                # Call buyer loss if expiry at test_strike
                if test_strike > s:
                    total_pain += (test_strike - s) * ce_oi
                # Put buyer loss
                if test_strike < s:
                    total_pain += (s - test_strike) * pe_oi

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return float(max_pain_strike)

    @staticmethod
    def _get_atm_strikes(records: list[dict], underlying: float, n: int) -> list[dict]:
        """Get ATM ± n strikes."""
        sorted_records = sorted(records, key=lambda r: abs(r.get("strike", 0) - underlying))
        return sorted_records[: 2 * n + 1]
