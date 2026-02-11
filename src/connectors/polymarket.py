from __future__ import annotations

from src.connectors.base import BaseConnector, parse_datetime, within_window
from src.connectors.http_client import SimpleHttpClient
from src.core.schemas import QuestionCandidate, ResolutionCandidate


class PolymarketConnector(BaseConnector):
    source = "polymarket"

    def __init__(self, base_url: str | None = None, http: SimpleHttpClient | None = None) -> None:
        self.base_url = base_url or "https://gamma-api.polymarket.com"
        self.http = http or SimpleHttpClient()

    def list_candidates(self, window_days: int = 7) -> list[QuestionCandidate]:
        try:
            payload = self.http.get_json(
                f"{self.base_url}/markets",
                params={"active": "true", "closed": "false", "limit": 500},
            )
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        output: list[QuestionCandidate] = []
        for row in payload:
            close_time = parse_datetime(
                row.get("endDate") or row.get("end_date") or row.get("closeTime")
            )
            resolve_time = parse_datetime(
                row.get("resolveDate") or row.get("resolve_date") or row.get("endDate")
            )
            if not (within_window(close_time, window_days) or within_window(resolve_time, window_days)):
                continue
            source_id = row.get("id") or row.get("conditionId")
            if source_id is None:
                continue
            output.append(
                QuestionCandidate(
                    source=self.source,
                    source_id=str(source_id),
                    title=str(row.get("question", row.get("title", ""))),
                    description=str(row.get("description", "")),
                    close_time=close_time,
                    resolve_time_expected=resolve_time,
                    tags=[str(t) for t in row.get("tags", []) if isinstance(t, str)],
                    resolver_type="polymarket_api",
                    resolver_config={"endpoint": f"{self.base_url}/markets/{source_id}"},
                    status="open",
                    raw_payload=row,
                )
            )
        return output

    def fetch_details(self, source_id: str) -> QuestionCandidate | None:
        try:
            row = self.http.get_json(f"{self.base_url}/markets/{source_id}")
        except Exception:
            return None
        if not isinstance(row, dict):
            return None
        return QuestionCandidate(
            source=self.source,
            source_id=source_id,
            title=str(row.get("question", row.get("title", ""))),
            description=str(row.get("description", "")),
            close_time=parse_datetime(row.get("endDate") or row.get("end_date") or row.get("closeTime")),
            resolve_time_expected=parse_datetime(row.get("resolveDate") or row.get("resolve_date") or row.get("endDate")),
            tags=[str(t) for t in row.get("tags", []) if isinstance(t, str)],
            resolver_type="polymarket_api",
            resolver_config={"endpoint": f"{self.base_url}/markets/{source_id}"},
            status="open",
            raw_payload=row,
        )

    def get_resolution(self, source_id: str) -> ResolutionCandidate:
        try:
            row = self.http.get_json(f"{self.base_url}/markets/{source_id}")
        except Exception:
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved")
        if not isinstance(row, dict):
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved")

        resolved_flag = bool(row.get("resolved") or row.get("closed"))
        winning = row.get("winningOutcome") or row.get("outcome")
        resolved_at = parse_datetime(
            row.get("resolvedTime") or row.get("resolveDate") or row.get("endDate")
        )
        if not resolved_flag:
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved", raw_payload=row)
        if isinstance(winning, str):
            w = winning.strip().upper()
            if w in {"YES", "TRUE", "1"}:
                return ResolutionCandidate(
                    source=self.source,
                    source_id=source_id,
                    status="resolved",
                    resolved_at=resolved_at,
                    y=1.0,
                    resolution_confidence=0.9,
                    raw_payload=row,
                )
            if w in {"NO", "FALSE", "0"}:
                return ResolutionCandidate(
                    source=self.source,
                    source_id=source_id,
                    status="resolved",
                    resolved_at=resolved_at,
                    y=0.0,
                    resolution_confidence=0.9,
                    raw_payload=row,
                )
        return ResolutionCandidate(
            source=self.source,
            source_id=source_id,
            status="manual_review",
            resolved_at=resolved_at,
            y=None,
            resolution_confidence=0.2,
            raw_payload=row,
        )

