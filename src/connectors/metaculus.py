from __future__ import annotations

from typing import Any

from src.connectors.base import BaseConnector, parse_datetime, within_window
from src.connectors.http_client import SimpleHttpClient
from src.core.schemas import QuestionCandidate, ResolutionCandidate


class MetaculusConnector(BaseConnector):
    source = "metaculus"

    def __init__(self, base_url: str | None = None, http: SimpleHttpClient | None = None) -> None:
        self.base_url = base_url or "https://www.metaculus.com/api2"
        self.http = http or SimpleHttpClient()

    def list_candidates(self, window_days: int = 7) -> list[QuestionCandidate]:
        try:
            payload = self.http.get_json(
                f"{self.base_url}/questions/",
                params={"limit": 200, "offset": 0},
            )
        except Exception:
            return []

        results = payload.get("results", []) if isinstance(payload, dict) else []
        candidates: list[QuestionCandidate] = []
        for row in results:
            status = str(row.get("status", "")).lower()
            if status not in {"open", "upcoming"}:
                continue
            close_time = parse_datetime(row.get("scheduled_close_time") or row.get("close_time"))
            resolve_time = parse_datetime(row.get("scheduled_resolve_time") or row.get("resolve_time"))
            if not (within_window(close_time, window_days) or within_window(resolve_time, window_days)):
                continue

            question_id = row.get("id")
            if question_id is None:
                continue
            question_type = ((row.get("question") or {}).get("type") if isinstance(row.get("question"), dict) else row.get("type")) or "binary"
            candidates.append(
                QuestionCandidate(
                    source=self.source,
                    source_id=str(question_id),
                    title=str(row.get("title", "")),
                    description=str((row.get("question") or {}).get("description", row.get("description", ""))),
                    close_time=close_time,
                    resolve_time_expected=resolve_time,
                    tags=[str(x) for x in row.get("projects", []) if isinstance(x, (str, int))],
                    resolver_type="metaculus_api",
                    resolver_config={"question_type": question_type, "endpoint": f"{self.base_url}/questions/{question_id}/"},
                    status="open",
                    raw_payload=row,
                )
            )
        return candidates

    def fetch_details(self, source_id: str) -> QuestionCandidate | None:
        try:
            row = self.http.get_json(f"{self.base_url}/questions/{source_id}/")
        except Exception:
            return None
        if not isinstance(row, dict):
            return None
        close_time = parse_datetime(row.get("scheduled_close_time") or row.get("close_time"))
        resolve_time = parse_datetime(row.get("scheduled_resolve_time") or row.get("resolve_time"))
        question_type = ((row.get("question") or {}).get("type") if isinstance(row.get("question"), dict) else row.get("type")) or "binary"
        return QuestionCandidate(
            source=self.source,
            source_id=str(source_id),
            title=str(row.get("title", "")),
            description=str((row.get("question") or {}).get("description", row.get("description", ""))),
            close_time=close_time,
            resolve_time_expected=resolve_time,
            tags=[str(x) for x in row.get("projects", []) if isinstance(x, (str, int))],
            resolver_type="metaculus_api",
            resolver_config={"question_type": question_type, "endpoint": f"{self.base_url}/questions/{source_id}/"},
            status="open",
            raw_payload=row,
        )

    def get_resolution(self, source_id: str) -> ResolutionCandidate:
        try:
            row = self.http.get_json(f"{self.base_url}/questions/{source_id}/")
        except Exception:
            return ResolutionCandidate(
                source=self.source,
                source_id=source_id,
                status="unresolved",
            )
        if not isinstance(row, dict):
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved")

        status = str(row.get("status", "")).lower()
        resolved_at = parse_datetime(row.get("actual_resolve_time") or row.get("resolve_time"))
        raw_resolution = row.get("resolution")
        y: float | None = None
        confidence = 0.0

        if isinstance(raw_resolution, bool):
            y = 1.0 if raw_resolution else 0.0
            confidence = 0.95
        elif isinstance(raw_resolution, (int, float)):
            if raw_resolution in (0, 1):
                y = float(raw_resolution)
                confidence = 0.9
        elif isinstance(raw_resolution, str):
            normalized = raw_resolution.strip().upper()
            if normalized in {"YES", "TRUE", "1"}:
                y = 1.0
                confidence = 0.9
            elif normalized in {"NO", "FALSE", "0"}:
                y = 0.0
                confidence = 0.9

        if status in {"resolved", "closed"} and y is not None:
            return ResolutionCandidate(
                source=self.source,
                source_id=source_id,
                status="resolved",
                resolved_at=resolved_at,
                y=y,
                resolution_confidence=confidence,
                raw_payload=row,
            )
        if status in {"resolved", "closed"} and y is None:
            return ResolutionCandidate(
                source=self.source,
                source_id=source_id,
                status="manual_review",
                resolved_at=resolved_at,
                y=None,
                resolution_confidence=0.2,
                raw_payload=row,
            )
        return ResolutionCandidate(
            source=self.source,
            source_id=source_id,
            status="unresolved",
            raw_payload=row,
        )

