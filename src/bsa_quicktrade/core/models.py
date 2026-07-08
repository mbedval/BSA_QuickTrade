"""Core data models used across the entire framework.

All inter-module communication flows through these dataclasses, ensuring
a consistent contract between analyzers, scoring, and output layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


# ── Signal Enum ─────────────────────────────────────────────────────────────

class Signal(Enum):
    """Directional signal produced by an analysis module."""

    STRONG_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    STRONG_BEARISH = -2

    @property
    def is_bullish(self) -> bool:
        return self.value > 0

    @property
    def is_bearish(self) -> bool:
        return self.value < 0

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


# ── Analysis Result ─────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Output of a single analysis module.

    Every analyzer must return exactly this structure so the scoring
    engine can combine results from independent modules uniformly.
    """

    module_name: str
    score: float  # 0–100  (0 = maximally bearish, 100 = maximally bullish)
    confidence: float  # 0–100
    signal: Signal
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(100.0, float(self.score)))
        self.confidence = max(0.0, min(100.0, float(self.confidence)))


# ── Support / Resistance Level ──────────────────────────────────────────────

@dataclass
class PriceLevel:
    """A support or resistance level with metadata."""

    price: float
    level_type: str  # "support" | "resistance"
    strength: float  # 0–100
    source: str  # e.g. "pivot", "option_oi", "fibonacci"
    touches: int = 0


# ── Trade Setup ─────────────────────────────────────────────────────────────

@dataclass
class TradeSetup:
    """Complete trade recommendation with entries, stops, and targets."""

    best_entry: float
    aggressive_entry: float
    conservative_entry: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward: float
    probability: float  # 0–100

    @property
    def risk_amount(self) -> float:
        return abs(self.best_entry - self.stop_loss)

    @property
    def reward_amount(self) -> float:
        return abs(self.target_1 - self.best_entry)


# ── Expected Range ──────────────────────────────────────────────────────────

@dataclass
class ExpectedRange:
    """Projected price range for a given time horizon."""

    low: float
    high: float
    period: str  # "intraday" | "1_week" | "1_month"

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def width_pct(self) -> float:
        mid = (self.high + self.low) / 2
        return (self.width / mid * 100) if mid else 0.0


# ── Stock Data Container ───────────────────────────────────────────────────

@dataclass
class StockData:
    """All downloaded data for a single stock, passed to analyzers."""

    ticker: str
    company_name: str
    daily: pd.DataFrame  # OHLCV, 2 years
    weekly: pd.DataFrame  # OHLCV, 2 years
    hourly: pd.DataFrame | None = None  # OHLCV, 60 days (optional)
    option_chain: dict[str, Any] | None = None
    delivery_data: pd.DataFrame | None = None
    index_daily: pd.DataFrame | None = None  # Nifty 50 daily
    sector: str = "Unknown"


# ── Complete Stock Analysis ─────────────────────────────────────────────────

@dataclass
class StockAnalysis:
    """Aggregated analysis for one stock — the final output structure.

    Contains results from all 12 modules plus derived trade setup.
    """

    ticker: str
    company_name: str
    current_price: float
    sector: str

    # Aggregate scores
    overall_score: float  # 0–100
    bullish_score: float  # 0–100
    bearish_score: float  # 0–100
    confidence: float  # 0–100
    signal: Signal

    # Per-module breakdown
    module_results: dict[str, AnalysisResult] = field(default_factory=dict)

    # Trade setup
    trade_setup: TradeSetup | None = None

    # Levels
    support_levels: list[PriceLevel] = field(default_factory=list)
    resistance_levels: list[PriceLevel] = field(default_factory=list)

    # Expected ranges
    expected_intraday_range: ExpectedRange | None = None
    expected_week_range: ExpectedRange | None = None
    expected_month_range: ExpectedRange | None = None

    # Reasoning
    reasons_for_selection: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    patterns_detected: list[str] = field(default_factory=list)

    @property
    def top_reasons(self) -> list[str]:
        """Return the top 5 reasons sorted by relevance."""
        return self.reasons_for_selection[:5]

    @property
    def module_summary(self) -> dict[str, float]:
        """Module name → score mapping for quick display."""
        return {name: r.score for name, r in self.module_results.items()}


# ── Backtest Result ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Metrics from backtesting a module or the full system."""

    module_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return_pct: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown_pct: float
    avg_holding_days: float
    accuracy: float
    precision: float
    recall: float
    total_return_pct: float = 0.0
