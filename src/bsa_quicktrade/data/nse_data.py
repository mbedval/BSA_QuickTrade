"""NSE-specific data — delivery percentage and option chains.

Uses ``nselib`` for delivery data and ``nsepython`` for option chains.
Both libraries hit the NSE website directly and may require proper
headers / session cookies.  Graceful fallbacks are provided.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from bsa_quicktrade.core.cache import DataCache

logger = logging.getLogger(__name__)


class NSEDataFetcher:
    """Fetch delivery percentage and option chain data from NSE."""

    def __init__(self, cache: DataCache) -> None:
        self.cache = cache

    # ── Delivery Data ───────────────────────────────────────────────────

    def get_delivery_data(self, symbol: str, num_days: int = 10) -> pd.DataFrame | None:
        """Fetch recent delivery percentage data for *symbol*.

        Returns a DataFrame with columns including delivery quantity
        and delivery percentage, or ``None`` on failure.
        """
        clean_symbol = symbol.replace(".NS", "")
        cached = self.cache.get("delivery", clean_symbol)
        if cached is not None:
            return cached

        try:
            from nselib import capital_market  # type: ignore[import-untyped]
            from datetime import datetime, timedelta

            end = datetime.now()
            start = end - timedelta(days=num_days * 2)  # extra buffer for non-trading days
            start_str = start.strftime("%d-%m-%Y")
            end_str = end.strftime("%d-%m-%Y")

            df = capital_market.price_volume_and_deliverable_position_data(
                symbol=clean_symbol,
                from_date=start_str,
                to_date=end_str,
            )

            if df is not None and not df.empty:
                self.cache.set("delivery", clean_symbol, df, ttl="delivery")
                logger.debug("Delivery data fetched for %s (%d rows)", clean_symbol, len(df))
                return df

        except ImportError:
            logger.warning("nselib not installed — delivery data unavailable")
        except Exception as exc:
            logger.debug("Delivery data fetch failed for %s: %s", clean_symbol, exc)

        return None

    # ── Option Chain ────────────────────────────────────────────────────

    def get_option_chain(self, symbol: str) -> dict[str, Any] | None:
        """Fetch the full option chain for *symbol* via nsepython.

        Returns the raw JSON dict from NSE's option-chain API, or
        ``None`` on failure.
        """
        clean_symbol = symbol.replace(".NS", "")
        cached = self.cache.get("option_chain", clean_symbol)
        if cached is not None:
            return cached

        try:
            from nsepython import nse_optionchain_scrapper  # type: ignore[import-untyped]

            data = nse_optionchain_scrapper(clean_symbol)

            if data:
                self.cache.set("option_chain", clean_symbol, data, ttl="option")
                logger.debug("Option chain fetched for %s", clean_symbol)
                return data

        except ImportError:
            logger.warning("nsepython not installed — option chain unavailable")
        except Exception as exc:
            logger.debug("Option chain fetch failed for %s: %s", clean_symbol, exc)

        return None

    def parse_option_chain(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse the raw NSE option chain JSON into a structured dict.

        Returns
        -------
        dict with keys:
            records:  list of per-strike dicts
            underlying_value: float
            expiry_dates: list[str]
            total_ce_oi: int
            total_pe_oi: int
            pcr: float
        """
        result: dict[str, Any] = {
            "records": [],
            "underlying_value": 0.0,
            "expiry_dates": [],
            "total_ce_oi": 0,
            "total_pe_oi": 0,
            "pcr": 0.0,
        }

        try:
            records = raw.get("records", raw.get("filtered", {}))
            data_rows = records.get("data", [])
            result["expiry_dates"] = records.get("expiryDates", [])
            result["underlying_value"] = records.get("underlyingValue", 0.0)

            total_ce_oi = 0
            total_pe_oi = 0

            for row in data_rows:
                strike = row.get("strikePrice", 0)
                entry: dict[str, Any] = {"strike": strike}

                ce = row.get("CE", {})
                pe = row.get("PE", {})

                if ce:
                    entry["ce_oi"] = ce.get("openInterest", 0)
                    entry["ce_oi_change"] = ce.get("changeinOpenInterest", 0)
                    entry["ce_volume"] = ce.get("totalTradedVolume", 0)
                    entry["ce_iv"] = ce.get("impliedVolatility", 0)
                    entry["ce_ltp"] = ce.get("lastPrice", 0)
                    total_ce_oi += entry["ce_oi"]

                if pe:
                    entry["pe_oi"] = pe.get("openInterest", 0)
                    entry["pe_oi_change"] = pe.get("changeinOpenInterest", 0)
                    entry["pe_volume"] = pe.get("totalTradedVolume", 0)
                    entry["pe_iv"] = pe.get("impliedVolatility", 0)
                    entry["pe_ltp"] = pe.get("lastPrice", 0)
                    total_pe_oi += entry["pe_oi"]

                result["records"].append(entry)

            result["total_ce_oi"] = total_ce_oi
            result["total_pe_oi"] = total_pe_oi
            result["pcr"] = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0

        except Exception as exc:
            logger.error("Option chain parse error: %s", exc)

        return result
