from __future__ import annotations

from src.connectors.base import BaseConnector, parse_datetime, within_window
from src.connectors.http_client import SimpleHttpClient
from src.core.schemas import QuestionCandidate, ResolutionCandidate


class ManifoldConnector(BaseConnector):
    source = "manifold"

    def __init__(self, base_url: str | None = None, http: SimpleHttpClient | None = None) -> None:
        self.base_url = base_url or "https://api.manifold.markets/v0"
        self.http = http or SimpleHttpClient()

    def list_candidates(self, window_days: int = 7) -> list[QuestionCandidate]:
        try:
            payload = self.http.get_json(f"{self.base_url}/markets", params={"limit": 1000})
        except Exception:
            return []
        if not isinstance(payload, list):
            return []

        output: list[QuestionCandidate] = []
        for row in payload:
            if str(row.get("outcomeType", "")).upper() != "BINARY":
                continue
            is_resolved = bool(row.get("isResolved", False))
            if is_resolved:
                continue
            close_time = parse_datetime(row.get("closeTime"))
            resolve_time = parse_datetime(row.get("resolutionTime") or row.get("closeTime"))
            if not (within_window(close_time, window_days) or within_window(resolve_time, window_days)):
                continue
            market_id = row.get("id")
            if market_id is None:
                continue
            output.append(
                QuestionCandidate(
                    source=self.source,
                    source_id=str(market_id),
                    title=str(row.get("question", "")),
                    description=str(row.get("textDescription", "")),
                    close_time=close_time,
                    resolve_time_expected=resolve_time,
                    tags=[str(x) for x in row.get("groupSlugs", []) if isinstance(x, str)],
                    resolver_type="manifold_api",
                    resolver_config={"endpoint": f"{self.base_url}/market/{market_id}"},
                    status="open",
                    raw_payload=row,
                )
            )
        return output

    def fetch_details(self, source_id: str) -> QuestionCandidate | None:
        try:
            row = self.http.get_json(f"{self.base_url}/market/{source_id}")
        except Exception:
            return None
        if not isinstance(row, dict):
            return None
        return QuestionCandidate(
            source=self.source,
            source_id=source_id,
            title=str(row.get("question", "")),
            description=str(row.get("textDescription", "")),
            close_time=parse_datetime(row.get("closeTime")),
            resolve_time_expected=parse_datetime(row.get("resolutionTime") or row.get("closeTime")),
            tags=[str(x) for x in row.get("groupSlugs", []) if isinstance(x, str)],
            resolver_type="manifold_api",
            resolver_config={"endpoint": f"{self.base_url}/market/{source_id}"},
            status="open",
            raw_payload=row,
        )

    def get_resolution(self, source_id: str) -> ResolutionCandidate:
        try:
            row = self.http.get_json(f"{self.base_url}/market/{source_id}")
        except Exception:
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved")
        if not isinstance(row, dict):
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved")
        is_resolved = bool(row.get("isResolved", False))
        resolution = str(row.get("resolution", "")).upper()
        resolved_at = parse_datetime(row.get("resolutionTime"))
        if not is_resolved:
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved", raw_payload=row)
        if resolution == "YES":
            return ResolutionCandidate(
                source=self.source,
                source_id=source_id,
                status="resolved",
                resolved_at=resolved_at,
                y=1.0,
                resolution_confidence=0.95,
                raw_payload=row,
            )
        if resolution == "NO":
            return ResolutionCandidate(
                source=self.source,
                source_id=source_id,
                status="resolved",
                resolved_at=resolved_at,
                y=0.0,
                resolution_confidence=0.95,
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

