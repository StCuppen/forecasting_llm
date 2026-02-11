from __future__ import annotations

from src.connectors.base import BaseConnector
from src.connectors.http_client import SimpleHttpClient
from src.connectors.kalshi import KalshiConnector
from src.connectors.manifold import ManifoldConnector
from src.connectors.metaculus import MetaculusConnector
from src.connectors.polymarket import PolymarketConnector
from src.core.config import LeagueConfig


def build_connectors(config: LeagueConfig) -> dict[str, BaseConnector]:
    connectors: dict[str, BaseConnector] = {}
    for source_name, source_cfg in config.sources.items():
        if not source_cfg.enabled:
            continue
        client = SimpleHttpClient(
            timeout_seconds=source_cfg.timeout_seconds,
            max_retries=source_cfg.max_retries,
            cache_ttl_seconds=source_cfg.cache_ttl_seconds,
        )
        if source_name == "metaculus":
            connectors[source_name] = MetaculusConnector(base_url=source_cfg.base_url, http=client)
        elif source_name == "kalshi":
            connectors[source_name] = KalshiConnector(base_url=source_cfg.base_url, http=client)
        elif source_name == "polymarket":
            connectors[source_name] = PolymarketConnector(base_url=source_cfg.base_url, http=client)
        elif source_name == "manifold":
            connectors[source_name] = ManifoldConnector(base_url=source_cfg.base_url, http=client)
    return connectors
