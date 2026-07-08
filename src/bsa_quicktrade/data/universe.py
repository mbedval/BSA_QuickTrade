"""Stock universe manager — builds and filters the tradeable universe."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bsa_quicktrade.core.config import AppConfig
from bsa_quicktrade.core.constants import (
    get_fno_symbols,
    get_sector,
    to_yfinance_ticker,
)

logger = logging.getLogger(__name__)


class UniverseManager:
    """Builds the tradeable stock universe with liquidity and price filters."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.universe

    def build_universe(self) -> list[str]:
        """Return yfinance-format tickers for all F&O stocks."""
        symbols = get_fno_symbols()
        tickers = [to_yfinance_ticker(s) for s in symbols]
        logger.info("Universe: %d F&O tickers", len(tickers))
        return tickers[: self.cfg.max_stocks]

    def filter_by_liquidity(
        self,
        tickers: list[str],
        daily_data: dict[str, pd.DataFrame],
    ) -> list[str]:
        """Remove tickers that don't meet minimum volume / price filters.

        Parameters
        ----------
        tickers:
            Full list of candidate tickers.
        daily_data:
            Mapping of ticker → daily OHLCV DataFrame (must have a
            ``Volume`` column and a ``Close`` column).
        """
        filtered: list[str] = []

        for t in tickers:
            df = daily_data.get(t)
            if df is None or df.empty:
                logger.debug("Excluded %s — no data", t)
                continue

            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            vol_col = None
            close_col = None
            for c in df.columns:
                cl = str(c).lower()
                if "volume" in cl:
                    vol_col = c
                if "close" in cl:
                    close_col = c

            if vol_col is None or close_col is None:
                logger.debug("Excluded %s — missing columns", t)
                continue

            avg_vol = float(np.nanmean(df[vol_col].tail(20)))
            last_price = float(df[close_col].iloc[-1])

            if avg_vol < self.cfg.min_avg_daily_volume:
                logger.debug(
                    "Excluded %s — avg vol %.0f < %.0f",
                    t, avg_vol, self.cfg.min_avg_daily_volume,
                )
                continue
            if last_price < self.cfg.min_price:
                logger.debug("Excluded %s — price ₹%.1f < ₹%.1f", t, last_price, self.cfg.min_price)
                continue

            filtered.append(t)

        logger.info(
            "Liquidity filter: %d → %d tickers (min vol=%d, min price=₹%.0f)",
            len(tickers), len(filtered),
            self.cfg.min_avg_daily_volume, self.cfg.min_price,
        )
        return filtered

    @staticmethod
    def get_sector(ticker: str) -> str:
        return get_sector(ticker)
