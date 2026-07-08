"""Disk-based caching layer with configurable TTL.

Uses ``diskcache`` for persistent storage so repeated runs within
the TTL window skip expensive network downloads.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import diskcache  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class DataCache:
    """Thin wrapper around diskcache.Cache with named TTL helpers."""

    def __init__(
        self,
        directory: str = ".cache/rsa_quicktrade",
        enabled: bool = True,
        daily_ttl_hours: int = 4,
        option_ttl_hours: int = 1,
        delivery_ttl_hours: int = 12,
    ) -> None:
        self._enabled = enabled
        self._daily_ttl = daily_ttl_hours * 3600
        self._option_ttl = option_ttl_hours * 3600
        self._delivery_ttl = delivery_ttl_hours * 3600

        if enabled:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(directory, size_limit=2**30)  # 1 GB
            logger.debug("Cache initialised at %s", directory)
        else:
            self._cache = None

    # ── Key builders ────────────────────────────────────────────────────

    @staticmethod
    def _key(namespace: str, identifier: str) -> str:
        return f"{namespace}:{identifier}"

    @staticmethod
    def _hash_key(namespace: str, identifier: str) -> str:
        raw = f"{namespace}:{identifier}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ── Public API ──────────────────────────────────────────────────────

    def get(self, namespace: str, identifier: str) -> Any | None:
        """Return cached value or ``None`` on miss / disabled."""
        if not self._enabled or self._cache is None:
            return None
        key = self._key(namespace, identifier)
        val = self._cache.get(key)
        if val is not None:
            logger.debug("Cache HIT  %s", key)
        return val

    def set(self, namespace: str, identifier: str, value: Any, ttl: str = "daily") -> None:
        """Store *value* under *namespace:identifier* with the named TTL."""
        if not self._enabled or self._cache is None:
            return
        expire = {
            "daily": self._daily_ttl,
            "option": self._option_ttl,
            "delivery": self._delivery_ttl,
        }.get(ttl, self._daily_ttl)
        key = self._key(namespace, identifier)
        self._cache.set(key, value, expire=expire)
        logger.debug("Cache SET  %s (ttl=%s)", key, ttl)

    def clear(self) -> None:
        """Evict all cached data."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Cache cleared")

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
