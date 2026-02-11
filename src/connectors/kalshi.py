from __future__ import annotations

from src.connectors.base import BaseConnector, parse_datetime, within_window
from src.connectors.http_client import SimpleHttpClient
from src.core.schemas import QuestionCandidate, ResolutionCandidate


class KalshiConnector(BaseConnector):
    source = "kalshi"

    def __init__(self, base_url: str | None = None, http: SimpleHttpClient | None = None) -> None:
        self.base_url = base_url or "https://api.elections.kalshi.com/trade-api/v2"
        self.http = http or SimpleHttpClient()

    def list_candidates(self, window_days: int = 7) -> list[QuestionCandidate]:
        try:
            payload = self.http.get_json(
                f"{self.base_url}/markets",
                params={"status": "open", "limit": 1000},
            )
        except Exception:
            return []
        markets = payload.get("markets", []) if isinstance(payload, dict) else []
        candidates: list[QuestionCandidate] = []
        for row in markets:
            close_time = parse_datetime(row.get("close_time") or row.get("expiration_time"))
            resolve_time = parse_datetime(row.get("settlement_time") or row.get("close_time"))
            if not (within_window(close_time, window_days) or within_window(resolve_time, window_days)):
                continue
            source_id = row.get("ticker")
            if source_id is None:
                continue
            candidates.append(
                QuestionCandidate(
                    source=self.source,
                    source_id=str(source_id),
                    title=str(row.get("title", row.get("subtitle", ""))),
                    description=str(row.get("rules_primary", row.get("subtitle", ""))),
                    close_time=close_time,
                    resolve_time_expected=resolve_time,
                    tags=[str(row.get("category", "kalshi"))],
                    resolver_type="kalshi_api",
                    resolver_config={"endpoint": f"{self.base_url}/markets/{source_id}"},
                    status="open",
                    raw_payload=row,
                )
            )
        return candidates

    def fetch_details(self, source_id: str) -> QuestionCandidate | None:
        try:
            row = self.http.get_json(f"{self.base_url}/markets/{source_id}")
        except Exception:
            return None
        if not isinstance(row, dict):
            return None
        market = row.get("market", row)
        if not isinstance(market, dict):
            market = row
        return QuestionCandidate(
            source=self.source,
            source_id=source_id,
            title=str(market.get("title", market.get("subtitle", ""))),
            description=str(market.get("rules_primary", market.get("subtitle", ""))),
            close_time=parse_datetime(market.get("close_time") or market.get("expiration_time")),
            resolve_time_expected=parse_datetime(market.get("settlement_time") or market.get("close_time")),
            tags=[str(market.get("category", "kalshi"))],
            resolver_type="kalshi_api",
            resolver_config={"endpoint": f"{self.base_url}/markets/{source_id}"},
            status="open",
            raw_payload=market,
        )

    def get_resolution(self, source_id: str) -> ResolutionCandidate:
        try:
            row = self.http.get_json(f"{self.base_url}/markets/{source_id}")
        except Exception:
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved")
        if not isinstance(row, dict):
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved")
        market = row.get("market", row)
        if not isinstance(market, dict):
            market = row
        status = str(market.get("status", "")).lower()
        result = str(market.get("result", "")).upper()
        resolved_at = parse_datetime(market.get("settlement_time") or market.get("close_time"))
        if status not in {"settled", "finalized", "closed", "resolved"}:
            return ResolutionCandidate(source=self.source, source_id=source_id, status="unresolved", raw_payload=market)
        if result in {"YES", "TRUE", "1"}:
            return ResolutionCandidate(
                source=self.source,
                source_id=source_id,
                status="resolved",
                resolved_at=resolved_at,
                y=1.0,
                resolution_confidence=0.95,
                raw_payload=market,
            )
        if result in {"NO", "FALSE", "0"}:
            return ResolutionCandidate(
                source=self.source,
                source_id=source_id,
                status="resolved",
                resolved_at=resolved_at,
                y=0.0,
                resolution_confidence=0.95,
                raw_payload=market,
            )
        return ResolutionCandidate(
            source=self.source,
            source_id=source_id,
            status="manual_review",
            resolved_at=resolved_at,
            y=None,
            resolution_confidence=0.2,
            raw_payload=market,
        )

