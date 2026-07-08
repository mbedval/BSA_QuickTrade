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
        
        # HTML export (always generated for visual dashboards)
        self._export_html(ranked)

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

    def _export_html(self, ranked: list[StockAnalysis]) -> None:
        """Export the scan results to a beautiful, modern HTML dashboard."""
        import html

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')
        
        # Start constructing HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RSA QuickTrade — Analysis Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        html {{
            scroll-behavior: smooth;
        }}

        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #06b6d4;
            --primary-glow: rgba(6, 182, 212, 0.15);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.15);
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.15);
            --warning: #f59e0b;
            --border-color: #24324f;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 2.5rem 1.5rem;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, #131b2e 0%, #0d1527 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}

        header h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        header .timestamp {{
            color: var(--text-muted);
            font-size: 0.95rem;
            letter-spacing: 0.05em;
        }}

        .dashboard-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 3rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}

        .section-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #00f2fe;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Table styles */
        .table-responsive {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.95rem;
        }}

        th {{
            background-color: rgba(36, 50, 79, 0.5);
            color: #00f2fe;
            font-weight: 600;
            padding: 1rem;
            border-bottom: 2px solid var(--border-color);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }}

        #ticker-header {{
            cursor: pointer;
            user-select: none;
            transition: background-color 0.2s ease, color 0.2s ease;
        }}

        #ticker-header:hover {{
            color: #fff;
            background-color: rgba(6, 182, 212, 0.15);
        }}

        .sort-indicator {{
            font-size: 0.85rem;
            opacity: 0.7;
            margin-left: 4px;
        }}

        td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: rgba(6, 182, 212, 0.04);
        }}

        .ticker-link {{
            text-decoration: none;
            display: inline-block;
            transition: transform 0.2s ease;
        }}

        .ticker-link:hover .ticker-badge {{
            background: rgba(6, 182, 212, 0.2);
            border-color: var(--primary);
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
            transform: translateY(-1px);
        }}

        .ticker-badge {{
            font-weight: 700;
            font-size: 1rem;
            color: #fff;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        /* Signal badges */
        .signal-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .signal-strong-bullish {{
            color: #10b981;
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .signal-bullish {{
            color: #34d399;
            background-color: rgba(52, 211, 153, 0.12);
            border: 1px solid rgba(52, 211, 153, 0.25);
        }}
        .signal-neutral {{
            color: #fbbf24;
            background-color: rgba(251, 191, 36, 0.12);
            border: 1px solid rgba(251, 191, 36, 0.25);
        }}
        .signal-bearish {{
            color: #f87171;
            background-color: rgba(248, 113, 113, 0.12);
            border: 1px solid rgba(248, 113, 113, 0.25);
        }}
        .signal-strong-bearish {{
            color: #ef4444;
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        /* Cards for individual stocks */
        .stock-detail-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.5s ease;
        }}

        .stock-detail-card:target {{
            border-color: var(--primary);
            box-shadow: 0 0 25px rgba(6, 182, 212, 0.4);
            background-color: #1a253d;
        }}

        .stock-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.25rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .stock-title-container {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .stock-rank {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
            background: var(--primary-glow);
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            border: 1px solid rgba(6, 182, 212, 0.3);
        }}

        .stock-name-sector {{
            display: flex;
            flex-direction: column;
        }}

        .stock-name {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
        }}

        .stock-sector {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.1rem;
        }}

        .stock-price-score {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }}

        .stock-price {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
        }}

        .stock-score-block {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }}

        .stock-score-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        .stock-score-value {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
        }}

        .stock-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 1.5rem;
        }}

        @media (max-width: 900px) {{
            .stock-grid {{
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }}
        }}

        .block-title {{
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-left: 3px solid var(--primary);
            padding-left: 0.5rem;
        }}

        .setup-details {{
            background-color: rgba(36, 50, 79, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            height: 100%;
        }}

        .setup-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.45rem 0;
            font-size: 0.9rem;
            border-bottom: 1px dashed rgba(36, 50, 79, 0.4);
        }}

        .setup-row:last-child {{
            border-bottom: none;
            padding-bottom: 0;
        }}

        .setup-row:first-child {{
            padding-top: 0;
        }}

        .setup-val {{
            font-weight: 600;
        }}

        .setup-val.highlight {{
            color: #fff;
        }}

        .bullet-list {{
            list-style: none;
        }}

        .bullet-list li {{
            position: relative;
            padding-left: 1.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }}

        .reasons-list li::before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: var(--success);
            font-weight: bold;
        }}

        .risks-list li::before {{
            content: "⚠";
            position: absolute;
            left: 0;
            color: var(--danger);
            font-weight: bold;
        }}

        .levels-container {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }}

        .level-chip {{
            background: rgba(36, 50, 79, 0.3);
            border: 1px solid var(--border-color);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.85rem;
        }}

        .level-chip .label {{
            color: var(--text-muted);
            margin-right: 0.25rem;
        }}

        .level-chip .value {{
            font-weight: 600;
        }}

        .level-chip.support {{
            border-color: rgba(16, 185, 129, 0.2);
            background: rgba(16, 185, 129, 0.05);
        }}
        .level-chip.support .value {{
            color: var(--success);
        }}

        .level-chip.resistance {{
            border-color: rgba(239, 68, 68, 0.2);
            background: rgba(239, 68, 68, 0.05);
        }}
        .level-chip.resistance .value {{
            color: var(--danger);
        }}

        /* Modules sub-table */
        .modules-card {{
            grid-column: span 2;
            margin-top: 0.5rem;
        }}

        @media (max-width: 900px) {{
            .modules-card {{
                grid-column: span 1;
            }}
        }}

        .modules-table th {{
            padding: 0.6rem 1rem;
            font-size: 0.75rem;
        }}

        .modules-table td {{
            padding: 0.6rem 1rem;
            font-size: 0.85rem;
        }}

        footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 5rem;
            border-top: 1px solid var(--border-color);
            padding-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RSA QuickTrade — Stock Analysis Report</h1>
            <div class="timestamp">Generated: {timestamp_str}</div>
        </header>

        <!-- Summary Dashboard -->
        <div class="dashboard-card">
            <h2 class="section-title">Top Stocks — Summary Dashboard</h2>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 5%; text-align: center;">#</th>
                            <th id="ticker-header" style="width: 15%;" onclick="sortTableByTicker()">Ticker <span class="sort-indicator">↕</span></th>
                            <th style="width: 15%; text-align: right;">Price</th>
                            <th style="width: 10%; text-align: center;">Score</th>
                            <th style="width: 10%; text-align: center;">Conf%</th>
                            <th style="width: 15%; text-align: center;">Signal</th>
                            <th style="width: 10%; text-align: center;">R:R</th>
                            <th style="width: 10%; text-align: right;">Entry</th>
                            <th style="width: 10%; text-align: right;">SL</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        # Generate summary table rows
        for i, a in enumerate(ranked, 1):
            ts = a.trade_setup
            sig_class = _get_signal_class(a.signal)
            ticker_clean = a.ticker.replace(".NS", "")
            
            html_content += f"""
                        <tr>
                            <td style="text-align: center; font-weight: 600;">{i}</td>
                            <td><a href="#detail-{ticker_clean}" class="ticker-link"><span class="ticker-badge">{ticker_clean}</span></a></td>
                            <td style="text-align: right; font-weight: 600;">₹{a.current_price:,.2f}</td>
                            <td style="text-align: center; font-weight: 700;"><span class="signal-badge {sig_class}">{a.overall_score:.0f}</span></td>
                            <td style="text-align: center;">{a.confidence:.0f}%</td>
                            <td style="text-align: center;"><span class="signal-badge {sig_class}">{a.signal.label}</span></td>
                            <td style="text-align: center;">{f"{ts.risk_reward:.1f}" if ts else "—"}</td>
                            <td style="text-align: right; color: var(--success); font-weight: 500;">{f"₹{ts.best_entry:,.2f}" if ts else "—"}</td>
                            <td style="text-align: right; color: var(--danger); font-weight: 500;">{f"₹{ts.stop_loss:,.2f}" if ts else "—"}</td>
                        </tr>"""

        html_content += """
                    </tbody>
                </table>
            </div>
        </div>

        <h2 class="section-title" style="margin-bottom: 2rem;">Detailed Stock Analysis</h2>
"""

        # Generate individual cards
        for i, a in enumerate(ranked, 1):
            sig_class = _get_signal_class(a.signal)
            ticker_clean = a.ticker.replace(".NS", "")
            ts = a.trade_setup
            
            # Setup details column HTML
            setup_html = ""
            if ts:
                setup_html = f"""
                <div>
                    <div class="block-title">Trade Setup</div>
                    <div class="setup-details">
                        <div class="setup-row">
                            <span>Best Entry</span>
                            <span class="setup-val highlight" style="color: var(--success);">₹{ts.best_entry:,.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Aggressive Entry</span>
                            <span class="setup-val">₹{ts.aggressive_entry:,.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Conservative Entry</span>
                            <span class="setup-val">₹{ts.conservative_entry:,.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Stop Loss</span>
                            <span class="setup-val highlight" style="color: var(--danger);">₹{ts.stop_loss:,.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Target 1</span>
                            <span class="setup-val highlight">₹{ts.target_1:,.2f} (R:R {ts.risk_reward:.1f})</span>
                        </div>
                        <div class="setup-row">
                            <span>Target 2</span>
                            <span class="setup-val">₹{ts.target_2:,.2f}</span>
                        </div>
                        <div class="setup-row">
                            <span>Target 3</span>
                            <span class="setup-val">₹{ts.target_3:,.2f}</span>
                        </div>
                    </div>
                </div>"""
            else:
                setup_html = """
                <div>
                    <div class="block-title">Trade Setup</div>
                    <div class="setup-details" style="display: flex; align-items: center; justify-content: center; color: var(--text-muted);">
                        No trade setup available
                    </div>
                </div>"""

            # Reasons & Risks column HTML
            reasons_html = ""
            if a.reasons_for_selection:
                reasons_html += """<div class="block-title">Why Selected</div><ul class="bullet-list reasons-list" style="margin-bottom: 1.5rem;">"""
                for r in a.reasons_for_selection[:4]:
                    reasons_html += f"<li>{html.escape(r)}</li>"
                reasons_html += "</ul>"
            
            risks_html = ""
            if a.risks:
                risks_html += """<div class="block-title">Risks / Watchout</div><ul class="bullet-list risks-list" style="margin-bottom: 1.5rem;">"""
                for r in a.risks[:3]:
                    risks_html += f"<li>{html.escape(r)}</li>"
                risks_html += "</ul>"

            patterns_html = ""
            if a.patterns_detected:
                patterns_html += f"""
                <div style="margin-top: 1rem;">
                    <span class="block-title" style="display: inline-block; border: none; padding: 0; margin-right: 0.5rem; font-size: 0.85rem;">Detected Patterns:</span>
                    <span style="font-size: 0.9rem; color: var(--warning); font-weight: 500;">{", ".join(a.patterns_detected[:5])}</span>
                </div>"""

            # Support & Resistance levels
            levels_html = ""
            if a.support_levels or a.resistance_levels:
                levels_html += """<div class="block-title" style="margin-top: 1.25rem;">Key Levels</div><div class="levels-container">"""
                for s in a.support_levels[:3]:
                    levels_html += f"""<div class="level-chip support"><span class="label">S:</span><span class="value">₹{s.price:,.2f}</span></div>"""
                for r in a.resistance_levels[:3]:
                    levels_html += f"""<div class="level-chip resistance"><span class="label">R:</span><span class="value">₹{r.price:,.2f}</span></div>"""
                levels_html += "</div>"

            # Ranges
            ranges_html = ""
            if a.expected_intraday_range or a.expected_week_range or a.expected_month_range:
                ranges_html += """<div class="block-title" style="margin-top: 1.25rem;">Expected Ranges</div><div class="levels-container">"""
                if a.expected_intraday_range:
                    r = a.expected_intraday_range
                    ranges_html += f"""<div class="level-chip"><span class="label">Intraday:</span><span class="value">₹{r.low:,.2f} - ₹{r.high:,.2f}</span></div>"""
                if a.expected_week_range:
                    r = a.expected_week_range
                    ranges_html += f"""<div class="level-chip"><span class="label">1-Week:</span><span class="value">₹{r.low:,.2f} - ₹{r.high:,.2f}</span></div>"""
                if a.expected_month_range:
                    r = a.expected_month_range
                    ranges_html += f"""<div class="level-chip"><span class="label">1-Month:</span><span class="value">₹{r.low:,.2f} - ₹{r.high:,.2f}</span></div>"""
                ranges_html += "</div>"

            # Modules Table
            modules_rows = ""
            for name, r in sorted(a.module_results.items(), key=lambda x: x[1].score, reverse=True):
                m_sig_class = _get_signal_class(r.signal)
                reason = r.reasons[0] if r.reasons else "—"
                modules_rows += f"""
                            <tr>
                                <td style="font-weight: 500;">{name.replace('_', ' ').title()}</td>
                                <td style="text-align: center; font-weight: 600;"><span class="signal-badge {m_sig_class}">{r.score:.0f}</span></td>
                                <td style="text-align: center;"><span class="signal-badge {m_sig_class}">{r.signal.label}</span></td>
                                <td style="text-align: center;">{r.confidence:.0f}%</td>
                                <td style="color: var(--text-muted);">{html.escape(reason)}</td>
                            </tr>"""

            html_content += f"""
        <div class="stock-detail-card" id="detail-{ticker_clean}">
            <div class="stock-header">
                <div class="stock-title-container">
                    <div class="stock-rank">{i}</div>
                    <div class="stock-name-sector">
                        <span class="stock-name">{ticker_clean} <span style="font-size: 0.95rem; font-weight: 400; color: var(--text-muted);">| {a.company_name}</span></span>
                        <span class="stock-sector">{a.sector}</span>
                    </div>
                </div>
                <div class="stock-price-score">
                    <span class="stock-price">₹{a.current_price:,.2f}</span>
                    <div class="stock-score-block">
                        <span class="stock-score-label">Score / Confidence</span>
                        <span class="stock-score-value"><span class="signal-badge {sig_class}" style="font-size: 1.2rem; padding: 0.3rem 1rem;">{a.overall_score:.0f} ({a.confidence:.0f}%)</span></span>
                    </div>
                </div>
            </div>

            <div class="stock-grid">
                <!-- Trade Setup -->
                {setup_html}

                <!-- Context, Reasons, Risks -->
                <div>
                    {reasons_html}
                    {risks_html}
                    {patterns_html}
                    {levels_html}
                    {ranges_html}
                </div>

                <!-- Modules Table -->
                <div class="modules-card">
                    <div class="block-title">Module Score Breakdown</div>
                    <div class="table-responsive">
                        <table class="modules-table">
                            <thead>
                                <tr>
                                    <th style="width: 20%;">Module</th>
                                    <th style="width: 10%; text-align: center;">Score</th>
                                    <th style="width: 15%; text-align: center;">Signal</th>
                                    <th style="width: 10%; text-align: center;">Conf%</th>
                                    <th style="width: 45%;">Reason</th>
                                </tr>
                            </thead>
                            <tbody>
                                {modules_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>"""

        html_content += f"""
        <footer>
            <p><strong>Total stocks analysed and ranked: {len(ranked)}</strong></p>
            <p style="margin-top: 0.5rem; opacity: 0.6; font-size: 0.75rem;">Disclaimer: This is an analytical tool, not financial advice. Always do your own research before trading.</p>
        </footer>
    </div>
    <script>
        let currentSortDir = 'asc';
        function sortTableByTicker() {{
            const table = document.querySelector('.dashboard-card table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            rows.sort((a, b) => {{
                const tickerA = a.querySelector('.ticker-badge').textContent.trim();
                const tickerB = b.querySelector('.ticker-badge').textContent.trim();
                return currentSortDir === 'asc' 
                    ? tickerA.localeCompare(tickerB) 
                    : tickerB.localeCompare(tickerA);
            }});
            
            currentSortDir = currentSortDir === 'asc' ? 'desc' : 'asc';
            
            const indicator = document.querySelector('#ticker-header .sort-indicator');
            indicator.textContent = currentSortDir === 'asc' ? ' ▲' : ' ▼';
            indicator.style.opacity = '1';

            rows.forEach(row => tbody.appendChild(row));
        }}
    </script>
</body>
</html>"""

        # Write to outputs
        # 1. Output directory
        path_dir = self.output_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.html"
        with open(path_dir, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info("HTML report saved to %s", path_dir)

        # 2. Workspace root directory (as quicktrade_scan.html)
        path_root = Path("quicktrade_scan.html")
        with open(path_root, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info("HTML report saved to workspace root as %s", path_root)
        console.print(f"[green]✓ HTML report generated at [bold]quicktrade_scan.html[/][/]")

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


def _get_signal_class(signal: Signal) -> str:
    return {
        Signal.STRONG_BULLISH: "signal-strong-bullish",
        Signal.BULLISH: "signal-bullish",
        Signal.NEUTRAL: "signal-neutral",
        Signal.BEARISH: "signal-bearish",
        Signal.STRONG_BEARISH: "signal-strong-bearish",
    }.get(signal, "")
