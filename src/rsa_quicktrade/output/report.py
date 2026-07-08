"""Rich console report generator.

Produces beautifully formatted terminal output with colored tables,
detailed stock analysis cards, and trade setup details.  Also exports
to JSON and CSV.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rsa_quicktrade.core.config import AppConfig
from rsa_quicktrade.core.models import Signal, StockAnalysis

logger = logging.getLogger(__name__)
console = Console()


class ReportGenerator:
    """Generate and display analysis reports."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output.report_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Main Entry ──────────────────────────────────────────────────────

    def generate(self, ranked: list[StockAnalysis]) -> None:
        """Generate the full report — console + file exports."""
        if not ranked:
            console.print("[bold red]No stocks met the selection criteria.[/]")
            return

        self._print_header()
        self._print_summary_table(ranked)

        for i, analysis in enumerate(ranked, 1):
            self._print_stock_card(analysis, rank=i)

        self._print_footer(len(ranked))

        # File exports
        fmt = self.config.output.format
        if fmt in ("json", "all"):
            self._export_json(ranked)
        if fmt in ("csv", "all"):
            self._export_csv(ranked)

    # ── Header / Footer ────────────────────────────────────────────────

    def _print_header(self) -> None:
        header = Text()
        header.append("╔══════════════════════════════════════════════════════════╗\n", style="bold cyan")
        header.append("║       RSA QuickTrade — Stock Analysis Report            ║\n", style="bold cyan")
        header.append(f"║       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'):>38s}  ║\n", style="bold cyan")
        header.append("╚══════════════════════════════════════════════════════════╝", style="bold cyan")
        console.print(header)
        console.print()

    def _print_footer(self, count: int) -> None:
        console.print()
        console.print(
            Panel(
                f"[bold]Total stocks analysed and ranked: {count}[/]\n"
                "[dim]Disclaimer: This is an analytical tool, not financial advice. "
                "Always do your own research before trading.[/]",
                title="[bold cyan]Report Complete[/]",
                border_style="cyan",
            )
        )

    # ── Summary Table ───────────────────────────────────────────────────

    def _print_summary_table(self, ranked: list[StockAnalysis]) -> None:
        table = Table(
            title="[bold white]Top Stocks — Summary Dashboard[/]",
            show_lines=True,
            border_style="bright_blue",
            header_style="bold white on dark_blue",
        )
        table.add_column("#", justify="center", width=3)
        table.add_column("Ticker", justify="left", width=14)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Score", justify="center", width=7)
        table.add_column("Conf%", justify="center", width=7)
        table.add_column("Signal", justify="center", width=16)
        table.add_column("R:R", justify="center", width=6)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("SL", justify="right", width=10)
        table.add_column("Target 1", justify="right", width=10)

        for i, a in enumerate(ranked, 1):
            sig_style = _signal_style(a.signal)
            ts = a.trade_setup

            table.add_row(
                str(i),
                f"[bold]{a.ticker.replace('.NS', '')}[/]",
                f"₹{a.current_price:,.2f}",
                f"[bold {sig_style}]{a.overall_score:.0f}[/]",
                f"{a.confidence:.0f}",
                f"[{sig_style}]{a.signal.label}[/]",
                f"{ts.risk_reward:.1f}" if ts else "—",
                f"₹{ts.best_entry:,.2f}" if ts else "—",
                f"₹{ts.stop_loss:,.2f}" if ts else "—",
                f"₹{ts.target_1:,.2f}" if ts else "—",
            )

        console.print(table)
        console.print()

    # ── Detailed Stock Card ─────────────────────────────────────────────

    def _print_stock_card(self, a: StockAnalysis, rank: int) -> None:
        sig_style = _signal_style(a.signal)

        # Title
        title = (
            f"[bold white]#{rank} {a.ticker.replace('.NS', '')} — "
            f"{a.company_name}[/]  "
            f"[{sig_style}]({a.signal.label})[/]"
        )

        # Module breakdown table
        mod_table = Table(show_header=True, border_style="dim", box=None, pad_edge=False)
        mod_table.add_column("Module", width=18)
        mod_table.add_column("Score", justify="center", width=7)
        mod_table.add_column("Signal", justify="center", width=16)
        mod_table.add_column("Conf%", justify="center", width=7)
        mod_table.add_column("Key Reason", width=50)

        for name, r in sorted(a.module_results.items(), key=lambda x: x[1].score, reverse=True):
            ms = _signal_style(r.signal)
            reason = r.reasons[0] if r.reasons else "—"
            mod_table.add_row(
                name.replace("_", " ").title(),
                f"[{ms}]{r.score:.0f}[/]",
                f"[{ms}]{r.signal.label}[/]",
                f"{r.confidence:.0f}",
                reason[:50],
            )

        # Trade setup
        ts = a.trade_setup
        trade_info = ""
        if ts:
            trade_info = (
                f"\n[bold]Trade Setup:[/]\n"
                f"  Best Entry:         ₹{ts.best_entry:,.2f}\n"
                f"  Aggressive Entry:   ₹{ts.aggressive_entry:,.2f}\n"
                f"  Conservative Entry: ₹{ts.conservative_entry:,.2f}\n"
                f"  Stop Loss:          ₹{ts.stop_loss:,.2f}\n"
                f"  Target 1:           ₹{ts.target_1:,.2f}  (R:R {ts.risk_reward:.1f})\n"
                f"  Target 2:           ₹{ts.target_2:,.2f}\n"
                f"  Target 3:           ₹{ts.target_3:,.2f}\n"
            )

        # Support / Resistance
        levels_info = ""
        if a.support_levels:
            sup_str = ", ".join(f"₹{s.price:,.2f}" for s in a.support_levels[:3])
            levels_info += f"\n[bold]Support:[/]    {sup_str}"
        if a.resistance_levels:
            res_str = ", ".join(f"₹{r.price:,.2f}" for r in a.resistance_levels[:3])
            levels_info += f"\n[bold]Resistance:[/] {res_str}"

        # Expected ranges
        range_info = ""
        if a.expected_intraday_range:
            r = a.expected_intraday_range
            range_info += f"\n[bold]Expected Intraday:[/] ₹{r.low:,.2f} — ₹{r.high:,.2f} ({r.width_pct:.1f}%)"
        if a.expected_week_range:
            r = a.expected_week_range
            range_info += f"\n[bold]Expected 1-Week:[/]  ₹{r.low:,.2f} — ₹{r.high:,.2f} ({r.width_pct:.1f}%)"
        if a.expected_month_range:
            r = a.expected_month_range
            range_info += f"\n[bold]Expected 1-Month:[/] ₹{r.low:,.2f} — ₹{r.high:,.2f} ({r.width_pct:.1f}%)"

        # Patterns
        pattern_info = ""
        if a.patterns_detected:
            pattern_info = f"\n[bold]Patterns:[/] {', '.join(a.patterns_detected[:5])}"

        # Reasons
        reasons_info = ""
        if a.reasons_for_selection:
            reasons_info = "\n[bold green]Why Selected:[/]\n"
            for reason in a.reasons_for_selection[:5]:
                reasons_info += f"  ✓ {reason}\n"

        # Risks
        risks_info = ""
        if a.risks:
            risks_info = "\n[bold red]Risks:[/]\n"
            for risk in a.risks[:3]:
                risks_info += f"  ⚠ {risk}\n"

        # Assemble card content
        card_content = (
            f"[bold]Price:[/] ₹{a.current_price:,.2f}  │  "
            f"[bold]Score:[/] [{sig_style}]{a.overall_score:.1f}[/]  │  "
            f"[bold]Confidence:[/] {a.confidence:.0f}%  │  "
            f"[bold]Sector:[/] {a.sector}\n"
            f"[bold]Bullish Score:[/] {a.bullish_score:.0f}  │  "
            f"[bold]Bearish Score:[/] {a.bearish_score:.0f}\n"
        )

        console.print(Panel(
            card_content,
            title=title,
            border_style=sig_style,
            width=90,
        ))
        console.print(mod_table)
        if trade_info:
            console.print(trade_info)
        if levels_info:
            console.print(levels_info)
        if range_info:
            console.print(range_info)
        if pattern_info:
            console.print(pattern_info)
        if reasons_info:
            console.print(reasons_info)
        if risks_info:
            console.print(risks_info)
        console.print("─" * 90)
        console.print()

    # ── File Exports ────────────────────────────────────────────────────

    def _export_json(self, ranked: list[StockAnalysis]) -> None:
        path = self.output_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.json"
        data = [self._analysis_to_dict(a) for a in ranked]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("JSON report saved to %s", path)

    def _export_csv(self, ranked: list[StockAnalysis]) -> None:
        import csv

        path = self.output_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.csv"
        fieldnames = [
            "rank", "ticker", "company", "price", "score", "confidence",
            "signal", "bullish_score", "bearish_score", "sector",
            "entry", "stop_loss", "target_1", "target_2", "target_3",
            "risk_reward", "reasons", "risks", "patterns",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, a in enumerate(ranked, 1):
                ts = a.trade_setup
                writer.writerow({
                    "rank": i,
                    "ticker": a.ticker.replace(".NS", ""),
                    "company": a.company_name,
                    "price": f"{a.current_price:.2f}",
                    "score": f"{a.overall_score:.1f}",
                    "confidence": f"{a.confidence:.0f}",
                    "signal": a.signal.label,
                    "bullish_score": f"{a.bullish_score:.0f}",
                    "bearish_score": f"{a.bearish_score:.0f}",
                    "sector": a.sector,
                    "entry": f"{ts.best_entry:.2f}" if ts else "",
                    "stop_loss": f"{ts.stop_loss:.2f}" if ts else "",
                    "target_1": f"{ts.target_1:.2f}" if ts else "",
                    "target_2": f"{ts.target_2:.2f}" if ts else "",
                    "target_3": f"{ts.target_3:.2f}" if ts else "",
                    "risk_reward": f"{ts.risk_reward:.2f}" if ts else "",
                    "reasons": " | ".join(a.reasons_for_selection[:3]),
                    "risks": " | ".join(a.risks[:3]),
                    "patterns": ", ".join(a.patterns_detected[:5]),
                })
        logger.info("CSV report saved to %s", path)

    @staticmethod
    def _analysis_to_dict(a: StockAnalysis) -> dict[str, Any]:
        ts = a.trade_setup
        return {
            "ticker": a.ticker,
            "company_name": a.company_name,
            "current_price": a.current_price,
            "sector": a.sector,
            "overall_score": round(a.overall_score, 1),
            "bullish_score": round(a.bullish_score, 1),
            "bearish_score": round(a.bearish_score, 1),
            "confidence": round(a.confidence, 1),
            "signal": a.signal.label,
            "module_scores": {
                name: {
                    "score": round(r.score, 1),
                    "confidence": round(r.confidence, 1),
                    "signal": r.signal.label,
                    "reasons": r.reasons,
                }
                for name, r in a.module_results.items()
            },
            "trade_setup": {
                "best_entry": ts.best_entry,
                "aggressive_entry": ts.aggressive_entry,
                "conservative_entry": ts.conservative_entry,
                "stop_loss": ts.stop_loss,
                "target_1": ts.target_1,
                "target_2": ts.target_2,
                "target_3": ts.target_3,
                "risk_reward": ts.risk_reward,
            } if ts else None,
            "support_levels": [{"price": s.price, "strength": s.strength, "source": s.source}
                               for s in a.support_levels[:5]],
            "resistance_levels": [{"price": r.price, "strength": r.strength, "source": r.source}
                                  for r in a.resistance_levels[:5]],
            "expected_intraday_range": {
                "low": a.expected_intraday_range.low,
                "high": a.expected_intraday_range.high,
            } if a.expected_intraday_range else None,
            "expected_week_range": {
                "low": a.expected_week_range.low,
                "high": a.expected_week_range.high,
            } if a.expected_week_range else None,
            "expected_month_range": {
                "low": a.expected_month_range.low,
                "high": a.expected_month_range.high,
            } if a.expected_month_range else None,
            "reasons_for_selection": a.reasons_for_selection,
            "risks": a.risks,
            "patterns_detected": a.patterns_detected,
        }


# ── Helpers ─────────────────────────────────────────────────────────────────

def _signal_style(signal: Signal) -> str:
    return {
        Signal.STRONG_BULLISH: "bold green",
        Signal.BULLISH: "green",
        Signal.NEUTRAL: "yellow",
        Signal.BEARISH: "red",
        Signal.STRONG_BEARISH: "bold red",
    }.get(signal, "white")
