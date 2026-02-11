from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.schemas import QuestionCandidate, ResolutionCandidate


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Heuristic: milliseconds if very large.
        if value > 10_000_000_000:
            value = value / 1000.0
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    return None


def within_window(dt: datetime | None, window_days: int) -> bool:
    if dt is None:
        return False
    horizon = utc_now() + timedelta(days=window_days)
    return dt <= horizon


class BaseConnector(ABC):
    source: str

    @abstractmethod
    def list_candidates(self, window_days: int = 7) -> list[QuestionCandidate]:
        raise NotImplementedError

    @abstractmethod
    def fetch_details(self, source_id: str) -> QuestionCandidate | None:
        raise NotImplementedError

    @abstractmethod
    def get_resolution(self, source_id: str) -> ResolutionCandidate:
        raise NotImplementedError

