"""Module 10 — Market Breadth Analysis.

Measures relative strength vs NIFTY, sector performance,
and classifies stocks into RRG quadrants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsa_quicktrade.analyzers.base import BaseAnalyzer
from rsa_quicktrade.core.models import AnalysisResult, StockData


class MarketBreadthAnalyzer(BaseAnalyzer):
    name = "market_breadth"

    def analyze(self, data: StockData) -> AnalysisResult:
        _, _, _, close, _ = self.get_ohlcv(data.daily)
        if close is None or len(close) < 50:
            return self.make_result(50, 10, ["Insufficient data for market breadth"])

        cfg = self.config.market_breadth
        reasons: list[str] = []
        metadata: dict = {}

        # ── 1. Relative Strength vs NIFTY ───────────────────────────────
        if data.index_daily is not None and len(data.index_daily) > cfg.rs_period:
            idx_close = self.safe_col(data.index_daily, "close")
            if idx_close is not None and len(idx_close) > cfg.rs_period:
                # Align lengths
                min_len = min(len(close), len(idx_close))
                stock_ret = float((close.iloc[-1] / close.iloc[-min(cfg.rs_period, min_len)]) - 1) * 100
                idx_ret = float((idx_close.iloc[-1] / idx_close.iloc[-min(cfg.rs_period, min_len)]) - 1) * 100

                rs_ratio = stock_ret / idx_ret if idx_ret != 0 else 1.0
                metadata["rs_ratio"] = round(rs_ratio, 2)
                metadata["stock_return"] = round(stock_ret, 1)
                metadata["index_return"] = round(idx_ret, 1)

                # RS Momentum (rate of change of RS ratio)
                if min_len > cfg.rs_period + cfg.rs_momentum_period:
                    rs_prev_stock = float((close.iloc[-cfg.rs_momentum_period] /
                                           close.iloc[-cfg.rs_period]) - 1) * 100
                    rs_prev_idx = float((idx_close.iloc[-cfg.rs_momentum_period] /
                                         idx_close.iloc[-cfg.rs_period]) - 1) * 100
                    rs_prev = rs_prev_stock / rs_prev_idx if rs_prev_idx != 0 else 1.0
                    rs_momentum = rs_ratio - rs_prev
                else:
                    rs_momentum = 0.0

                metadata["rs_momentum"] = round(rs_momentum, 3)

                # RRG Quadrant Classification
                if rs_ratio > 1.0 and rs_momentum > 0:
                    quadrant = "Leading"
                    score = 80 + min(rs_ratio * 5, 15)
                    reasons.append(f"RRG Leading quadrant — RS {rs_ratio:.2f}x, outperforming NIFTY")
                elif rs_ratio > 1.0 and rs_momentum <= 0:
                    quadrant = "Weakening"
                    score = 55
                    reasons.append(f"RRG Weakening — still outperforming but momentum fading")
                elif rs_ratio <= 1.0 and rs_momentum > 0:
                    quadrant = "Improving"
                    score = 62
                    reasons.append(f"RRG Improving — underperforming but momentum turning positive")
                else:
                    quadrant = "Lagging"
                    score = 25
                    reasons.append(f"RRG Lagging — underperforming NIFTY with negative momentum")

                metadata["rrg_quadrant"] = quadrant
                score = float(np.clip(score, 0, 100))

                # Add return comparison
                if stock_ret > idx_ret + 5:
                    reasons.append(f"Stock +{stock_ret:.0f}% vs NIFTY +{idx_ret:.0f}% ({cfg.rs_period}-day)")
                elif stock_ret < idx_ret - 5:
                    reasons.append(f"Stock +{stock_ret:.0f}% vs NIFTY +{idx_ret:.0f}% — underperforming")

                confidence = 65
                return self.make_result(score, confidence, reasons, **metadata)

        # Fallback: no index data
        reasons.append("No NIFTY benchmark data available for relative strength")
        return self.make_result(50, 15, reasons, **metadata)
