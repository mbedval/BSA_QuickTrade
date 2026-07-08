# RSA QuickTrade

**Institutional-Quality NSE Stock Screening & Analysis Framework**

Identifies the Top 10 NSE stocks most likely to produce profitable intraday opportunities for the next trading session by combining 12 independent analysis modules with weighted scoring.

## Features

### 12 Analysis Modules
| Module | Weight | Key Indicators |
|--------|--------|----------------|
| **Trend** | 15% | EMA stack (20/50/100/200), ADX, Golden/Death cross, regression slope, HH/HL structure, weekly trend |
| **Momentum** | 12% | RSI (14), Stochastic RSI, MACD (12/26/9), CCI, ROC, divergence detection |
| **Volatility** | 8% | ATR, Bollinger Band squeeze, TTM Squeeze (Keltner/BB), NR4/NR7, gap analysis |
| **Volume** | 15% | RVOL, OBV, VWAP, CMF, MFI, A/D, delivery %, smart money detection |
| **Price Action** | 10% | Swing S/R clustering, breakout + volume, market structure shifts, liquidity sweeps |
| **Candlestick** | 5% | 15 patterns (hammer, engulfing, morning/evening star, doji, marubozu, etc.) |
| **Chart Patterns** | 5% | Double top/bottom, H&S, triangles, wedges, flags, cup & handle |
| **Fibonacci** | 3% | Auto swing detection, retracement/extension, confluence zones |
| **Ichimoku** | 5% | Cloud position, TK cross, future cloud, Chikou span, cloud twist |
| **Market Breadth** | 5% | Relative strength vs NIFTY, RS momentum, RRG quadrant |
| **Options** | 12% | PCR, max pain, OI-based S/R, OI buildup classification, IV rank |
| **Statistics** | 5% | Historical pattern similarity search, forward return analysis |

### Output
- Rich terminal reports with colored tables and stock cards
- Module-level score breakdown with reasons
- Trade setups: entry, stop-loss, 3 targets, risk/reward
- Support/resistance levels with sources
- Expected price ranges (intraday, 1-week, 1-month)
- Professional charts (candlestick + EMA/BB/RSI/MACD panels)
- JSON and CSV export
- Walk-forward backtesting

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with all dependencies
pip install -e ".[dev]"
```

## Usage

### Full Market Scan (Primary Use Case)
```bash
python -m rsa_quicktrade scan
```
Scans ~200 NSE stocks and outputs the Top 10 opportunities.

### Single Stock Deep-Dive
```bash
python -m rsa_quicktrade analyze RELIANCE
```

### Generate Charts Only
```bash
python -m rsa_quicktrade chart TATAMOTORS
```

### Backtest Modules
```bash
python -m rsa_quicktrade backtest RELIANCE
```

### CLI Options
```bash
python -m rsa_quicktrade scan --top 20          # Top 20 instead of 10
python -m rsa_quicktrade scan --min-volume 1000000  # Higher volume filter
python -m rsa_quicktrade scan -c config/custom.yaml # Custom config
```

## Configuration

All parameters are configurable via `config/default.yaml`. Key sections:

```yaml
weights:
  trend: 15
  momentum: 12
  volume: 15
  options: 12
  # ... adjust module weights

scoring:
  min_confirmations: 3   # Minimum modules agreeing
  top_n: 10              # Number of stocks to select
  min_overall_score: 40  # Minimum score threshold

universe:
  min_avg_daily_volume: 500000
  min_price: 50
  prefer_fno: true       # Prioritize F&O stocks
```

### Environment Variable Overrides
```bash
RSA_SCORING__TOP_N=20 python -m rsa_quicktrade scan
RSA_UNIVERSE__MIN_PRICE=100 python -m rsa_quicktrade scan
```

## Architecture

```
src/rsa_quicktrade/
├── core/            # Config, models, constants, cache, logging
├── data/            # yfinance downloader, universe manager, NSE data
├── analyzers/       # 12 independent analysis modules
│   ├── base.py      # Abstract base (BaseAnalyzer)
│   ├── trend.py     # EMA, ADX, slope, swing structure
│   ├── momentum.py  # RSI, MACD, StochRSI, divergence
│   ├── volatility.py # ATR, BB squeeze, TTM, NR patterns
│   ├── volume.py    # OBV, VWAP, CMF, delivery %
│   ├── price_action.py # S/R, breakouts, market structure
│   ├── candlestick.py  # 15 candlestick patterns
│   ├── chart_patterns.py # Classical chart patterns
│   ├── fibonacci.py # Retracement, extensions, confluence
│   ├── ichimoku.py  # Full Ichimoku cloud system
│   ├── market_breadth.py # RS vs NIFTY, RRG quadrant
│   ├── options.py   # Option chain analysis
│   └── statistics.py # Historical similarity search
├── scoring/         # Weighted aggregation + ranking
├── output/          # Rich reports + mplfinance charts
├── backtesting/     # Walk-forward backtesting engine
└── main.py          # CLI entry point
```

### Data Flow
```
main.py → UniverseManager (build stock list)
        → DataDownloader (yfinance OHLCV + NSE data)
        → 12 Analyzers (each returns AnalysisResult)
        → RankingEngine (weighted aggregation → StockAnalysis)
        → ReportGenerator (Rich console + JSON/CSV)
        → ChartGenerator (mplfinance charts)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | OHLCV data from Yahoo Finance |
| `pandas-ta-classic` | 130+ technical indicators |
| `scipy` | Peak detection, regression, statistics |
| `mplfinance` | Professional candlestick charts |
| `rich` | Beautiful terminal output |
| `diskcache` | Intelligent data caching |
| `nsepython` | NSE F&O list, option chains |
| `nselib` | Delivery data |

## Disclaimer

This is an analytical tool for educational purposes. It does not constitute financial advice. Always conduct your own research and consult a qualified financial advisor before making trading decisions.

## License

MIT
