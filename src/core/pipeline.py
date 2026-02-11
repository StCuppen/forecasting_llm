from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from src.core.config import LeagueConfig
from src.core.updater import detect_domain


def infer_domain_tag(tags: Iterable[str], config: LeagueConfig) -> str:
    return detect_domain(list(tags), config.domain_keywords)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)

