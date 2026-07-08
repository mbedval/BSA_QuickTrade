"""Abstract base class for all analysis modules.

Every analyzer must inherit from ``BaseAnalyzer`` and implement the
``analyze`` method, which receives a ``StockData`` bundle and returns
an ``AnalysisResult``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from rsa_quicktrade.core.config import AppConfig
from rsa_quicktrade.core.models import AnalysisResult, Signal, StockData


class BaseAnalyzer(ABC):
    """Base class that every analysis module inherits from."""

    name: str = "base"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"analyzer.{self.name}")

    @abstractmethod
    def analyze(self, data: StockData) -> AnalysisResult:
        """Run the analysis and return a scored result."""
        ...

    # ── Scoring Helpers ─────────────────────────────────────────────────

    @staticmethod
    def normalize_score(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """Clamp *value* to [0, 100]."""
        if max_val == min_val:
            return 50.0
        scaled = (value - min_val) / (max_val - min_val) * 100
        return float(np.clip(scaled, 0, 100))

    @staticmethod
    def score_to_signal(score: float) -> Signal:
        """Map a 0–100 score to a directional signal.

        0–20   → STRONG_BEARISH
        20–40  → BEARISH
        40–60  → NEUTRAL
        60–80  → BULLISH
        80–100 → STRONG_BULLISH
        """
        if score >= 80:
            return Signal.STRONG_BULLISH
        if score >= 60:
            return Signal.BULLISH
        if score >= 40:
            return Signal.NEUTRAL
        if score >= 20:
            return Signal.BEARISH
        return Signal.STRONG_BEARISH

    # ── Data Helpers ────────────────────────────────────────────────────

    @staticmethod
    def safe_col(df: pd.DataFrame, name: str) -> pd.Series | None:
        """Return a column by case-insensitive match, or None."""
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance grouped download may have (ticker, col) multi-index
            for col in df.columns:
                if isinstance(col, tuple):
                    if name.lower() in str(col[-1]).lower():
                        return df[col]
                elif name.lower() in str(col).lower():
                    return df[col]
        else:
            for col in df.columns:
                if name.lower() in str(col).lower():
                    return df[col]
        return None

    @staticmethod
    def get_ohlcv(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Extract Open, High, Low, Close, Volume series from a DataFrame.

        Handles both flat and MultiIndex column layouts from yfinance.
        """
        cols = {}
        target_names = ["open", "high", "low", "close", "volume"]

        if isinstance(df.columns, pd.MultiIndex):
            level = -1  # last level typically has the OHLCV names
            for col in df.columns:
                col_name = str(col[level]).lower() if isinstance(col, tuple) else str(col).lower()
                for tn in target_names:
                    if tn in col_name and tn not in cols:
                        cols[tn] = df[col]
        else:
            for col in df.columns:
                col_lower = str(col).lower()
                for tn in target_names:
                    if tn in col_lower and tn not in cols:
                        cols[tn] = df[col]

        o = cols.get("open", pd.Series(dtype=float))
        h = cols.get("high", pd.Series(dtype=float))
        lo = cols.get("low", pd.Series(dtype=float))
        c = cols.get("close", pd.Series(dtype=float))
        v = cols.get("volume", pd.Series(dtype=float))
        return o, h, lo, c, v

    def make_result(
        self,
        score: float,
        confidence: float,
        reasons: list[str],
        **metadata,
    ) -> AnalysisResult:
        """Convenience builder for ``AnalysisResult``."""
        return AnalysisResult(
            module_name=self.name,
            score=score,
            confidence=confidence,
            signal=self.score_to_signal(score),
            reasons=reasons,
            metadata=metadata,
        )
