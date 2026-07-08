"""Configuration loader — reads YAML and exposes typed dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Section dataclasses ─────────────────────────────────────────────────────

@dataclass
class UniverseConfig:
    min_avg_daily_volume: int = 500_000
    min_price: float = 50.0
    prefer_fno: bool = True
    max_stocks: int = 250


@dataclass
class DataConfig:
    daily_period: str = "2y"
    weekly_period: str = "2y"
    hourly_period: str = "60d"
    download_chunk_size: int = 50
    download_delay_seconds: float = 1.0
    max_retries: int = 3
    retry_backoff_factor: float = 2.0


@dataclass
class CacheConfig:
    enabled: bool = True
    directory: str = ".cache/rsa_quicktrade"
    daily_ttl_hours: int = 4
    option_ttl_hours: int = 1
    delivery_ttl_hours: int = 12


@dataclass
class LoggingConfig:
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_file: str = "output/rsa_quicktrade.log"


@dataclass
class WeightsConfig:
    trend: float = 15.0
    momentum: float = 12.0
    volatility: float = 8.0
    volume: float = 15.0
    price_action: float = 10.0
    candlestick: float = 5.0
    chart_patterns: float = 5.0
    fibonacci: float = 3.0
    ichimoku: float = 5.0
    market_breadth: float = 5.0
    options: float = 12.0
    statistics: float = 5.0

    def as_dict(self) -> dict[str, float]:
        return {
            "trend": self.trend,
            "momentum": self.momentum,
            "volatility": self.volatility,
            "volume": self.volume,
            "price_action": self.price_action,
            "candlestick": self.candlestick,
            "chart_patterns": self.chart_patterns,
            "fibonacci": self.fibonacci,
            "ichimoku": self.ichimoku,
            "market_breadth": self.market_breadth,
            "options": self.options,
            "statistics": self.statistics,
        }

    @property
    def total(self) -> float:
        return sum(self.as_dict().values())

    def normalized(self) -> dict[str, float]:
        """Return weights normalized to sum to 1.0."""
        t = self.total
        return {k: v / t for k, v in self.as_dict().items()} if t else self.as_dict()


@dataclass
class TrendConfig:
    ema_periods: list[int] = field(default_factory=lambda: [20, 50, 100, 200])
    adx_period: int = 14
    adx_strong_threshold: int = 25
    adx_very_strong_threshold: int = 40
    slope_period: int = 20
    swing_lookback: int = 5


@dataclass
class MomentumConfig:
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    stoch_rsi_period: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    cci_period: int = 20
    cci_overbought: int = 100
    cci_oversold: int = -100
    roc_period: int = 12
    divergence_lookback: int = 50


@dataclass
class VolatilityConfig:
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    keltner_period: int = 20
    keltner_atr_mult: float = 1.5
    historical_vol_period: int = 20
    nr_lookback_4: int = 4
    nr_lookback_7: int = 7
    wide_range_atr_mult: float = 1.5
    gap_threshold_pct: float = 0.5


@dataclass
class VolumeConfig:
    avg_period: int = 20
    spike_multiplier: float = 2.0
    dryup_threshold: float = 0.5
    climax_volume_mult: float = 3.0
    cmf_period: int = 20
    mfi_period: int = 14
    delivery_lookback: int = 10
    delivery_increase_threshold: float = 5.0


@dataclass
class PriceActionConfig:
    swing_order: int = 5
    support_resistance_lookback: int = 100
    breakout_volume_mult: float = 1.5
    false_breakout_pct: float = 0.5
    consolidation_atr_mult: float = 0.5
    level_cluster_pct: float = 1.0


@dataclass
class CandlestickConfig:
    context_lookback: int = 20
    confirmation_candles: int = 1
    volume_confirmation_mult: float = 1.2


@dataclass
class ChartPatternsConfig:
    lookback_periods: list[int] = field(default_factory=lambda: [20, 50, 100, 250])
    pivot_order: int = 10
    tolerance_pct: float = 2.0
    min_pattern_bars: int = 10
    strictness: str = "moderate"


@dataclass
class FibonacciConfig:
    retracement_levels: list[float] = field(
        default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786]
    )
    extension_levels: list[float] = field(
        default_factory=lambda: [1.272, 1.618, 2.0, 2.618]
    )
    swing_order: int = 20
    confluence_threshold_pct: float = 1.0


@dataclass
class IchimokuConfig:
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    displacement: int = 26


@dataclass
class MarketBreadthConfig:
    rs_period: int = 50
    rs_momentum_period: int = 10
    benchmark_index: str = "^NSEI"


@dataclass
class OptionsConfig:
    strikes_range: int = 2
    unusual_oi_std: float = 2.0
    iv_rank_period: int = 252
    enabled: bool = True


@dataclass
class StatisticsConfig:
    similarity_window: int = 20
    min_correlation: float = 0.7
    forward_periods: list[int] = field(default_factory=lambda: [1, 5, 20])
    min_similar_matches: int = 5


@dataclass
class ScoringConfig:
    min_confirmations: int = 3
    top_n: int = 10
    min_overall_score: float = 40.0


@dataclass
class TradeSetupConfig:
    atr_sl_multiplier: float = 1.5
    target_1_rr: float = 1.5
    target_2_rr: float = 2.5
    target_3_rr: float = 4.0
    aggressive_entry_offset_atr: float = 0.3
    conservative_entry_offset_atr: float = 0.5


@dataclass
class VisualizationConfig:
    style: str = "yahoo"
    dpi: int = 300
    width: int = 20
    height: int = 14
    output_dir: str = "output/charts"


@dataclass
class BacktestingConfig:
    lookback_period: str = "2y"
    walk_forward_window: int = 20
    commission_pct: float = 0.03
    slippage_pct: float = 0.05
    output_dir: str = "output/backtests"


@dataclass
class OutputConfig:
    report_dir: str = "output/reports"
    format: str = "console"


# ── Root Config ─────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    """Root configuration — aggregates all sections."""

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    price_action: PriceActionConfig = field(default_factory=PriceActionConfig)
    candlestick: CandlestickConfig = field(default_factory=CandlestickConfig)
    chart_patterns: ChartPatternsConfig = field(default_factory=ChartPatternsConfig)
    fibonacci: FibonacciConfig = field(default_factory=FibonacciConfig)
    ichimoku: IchimokuConfig = field(default_factory=IchimokuConfig)
    market_breadth: MarketBreadthConfig = field(default_factory=MarketBreadthConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    trade_setup: TradeSetupConfig = field(default_factory=TradeSetupConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ── Loader ──────────────────────────────────────────────────────────────────

def _merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base*."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _build_section(cls: type, raw: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a raw dict, ignoring unknown keys."""
    import dataclasses

    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in valid})


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file with env-var overrides.

    Parameters
    ----------
    path:
        Path to YAML config file.  Falls back to ``config/default.yaml``
        relative to the project root, then to built-in defaults.
    """
    raw: dict[str, Any] = {}

    # 1. Load YAML
    if path is None:
        candidates = [
            Path("config/default.yaml"),
            Path(__file__).resolve().parents[3] / "config" / "default.yaml",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    if path and Path(path).exists():
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

    # 2. Env-var overrides (RSA_ prefix)
    env_overrides: dict[str, Any] = {}
    for key, val in os.environ.items():
        if key.startswith("RSA_"):
            parts = key[4:].lower().split("__")
            d = env_overrides
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            # Try to cast to number
            try:
                d[parts[-1]] = int(val)
            except ValueError:
                try:
                    d[parts[-1]] = float(val)
                except ValueError:
                    d[parts[-1]] = val
    if env_overrides:
        _merge(raw, env_overrides)

    # 3. Build typed config
    cfg = AppConfig()
    section_map: dict[str, tuple[str, type]] = {
        "universe": ("universe", UniverseConfig),
        "data": ("data", DataConfig),
        "cache": ("cache", CacheConfig),
        "logging": ("logging", LoggingConfig),
        "weights": ("weights", WeightsConfig),
        "trend": ("trend", TrendConfig),
        "momentum": ("momentum", MomentumConfig),
        "volatility": ("volatility", VolatilityConfig),
        "volume": ("volume", VolumeConfig),
        "price_action": ("price_action", PriceActionConfig),
        "candlestick": ("candlestick", CandlestickConfig),
        "chart_patterns": ("chart_patterns", ChartPatternsConfig),
        "fibonacci": ("fibonacci", FibonacciConfig),
        "ichimoku": ("ichimoku", IchimokuConfig),
        "market_breadth": ("market_breadth", MarketBreadthConfig),
        "options": ("options", OptionsConfig),
        "statistics": ("statistics", StatisticsConfig),
        "scoring": ("scoring", ScoringConfig),
        "trade_setup": ("trade_setup", TradeSetupConfig),
        "visualization": ("visualization", VisualizationConfig),
        "backtesting": ("backtesting", BacktestingConfig),
        "output": ("output", OutputConfig),
    }
    for yaml_key, (attr, cls) in section_map.items():
        if yaml_key in raw and isinstance(raw[yaml_key], dict):
            setattr(cfg, attr, _build_section(cls, raw[yaml_key]))

    return cfg
