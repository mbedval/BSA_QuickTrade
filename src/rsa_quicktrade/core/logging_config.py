"""Structured logging setup — console + rotating file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_file: str = "output/rsa_quicktrade.log",
) -> None:
    """Configure the root logger with console and file handlers."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        root.warning("Could not create log file at %s", log_file)

    # Suppress noisy third-party loggers
    for name in ("urllib3", "yfinance", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
