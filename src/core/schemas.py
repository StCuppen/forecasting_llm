from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from src.core.utils import from_iso, to_iso, utc_now


JsonDict = dict[str, Any]


@dataclass
class Question:
    id: str
    source: str
    source_id: str
    title: str
    description: str
    close_time: datetime | None
    resolve_time_expected: datetime | None
    tags: list[str]
    resolver_type: str
    resolver_config: JsonDict
    status: str
    duplicate_of: str | None = None
    raw_payload: JsonDict | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def to_record(self) -> JsonDict:
        record = asdict(self)
        record["close_time"] = to_iso(self.close_time)
        record["resolve_time_expected"] = to_iso(self.resolve_time_expected)
        record["created_at"] = to_iso(self.created_at)
        record["updated_at"] = to_iso(self.updated_at)
        return record

    @staticmethod
    def from_record(record: JsonDict) -> "Question":
        return Question(
            id=record["id"],
            source=record["source"],
            source_id=record["source_id"],
            title=record["title"],
            description=record.get("description", ""),
            close_time=from_iso(record.get("close_time")),
            resolve_time_expected=from_iso(record.get("resolve_time_expected")),
            tags=list(record.get("tags", [])),
            resolver_type=record.get("resolver_type", "unknown"),
            resolver_config=dict(record.get("resolver_config", {})),
            status=record.get("status", "open"),
            duplicate_of=record.get("duplicate_of"),
            raw_payload=record.get("raw_payload"),
            created_at=from_iso(record.get("created_at")) or utc_now(),
            updated_at=from_iso(record.get("updated_at")) or utc_now(),
        )


@dataclass
class Prediction:
    question_id: str
    run_id: str
    made_at: datetime
    p_ens: float
    p_agents: dict[str, float]
    model_versions: dict[str, str]
    evidence_bundle_id: str
    cost_estimate: float | None
    latency: float | None
    forecast_context: JsonDict = field(default_factory=dict)
    calibrator_version: str | None = None
    id: int | None = None

    def to_record(self) -> JsonDict:
        record = asdict(self)
        record["made_at"] = to_iso(self.made_at)
        return record

    @staticmethod
    def from_record(record: JsonDict) -> "Prediction":
        return Prediction(
            id=record.get("id"),
            question_id=record["question_id"],
            run_id=record["run_id"],
            made_at=from_iso(record["made_at"]) or utc_now(),
            p_ens=float(record["p_ens"]),
            p_agents=dict(record.get("p_agents", {})),
            model_versions=dict(record.get("model_versions", {})),
            evidence_bundle_id=record["evidence_bundle_id"],
            cost_estimate=record.get("cost_estimate"),
            latency=record.get("latency"),
            forecast_context=dict(record.get("forecast_context", {})),
            calibrator_version=record.get("calibrator_version"),
        )


@dataclass
class EvidenceItem:
    url: str
    retrieved_at: datetime
    snippet_hash: str
    trust_score: float
    rank: int
    snippet: str | None = None

    def to_record(self) -> JsonDict:
        return {
            "url": self.url,
            "retrieved_at": to_iso(self.retrieved_at),
            "snippet_hash": self.snippet_hash,
            "trust_score": self.trust_score,
            "rank": self.rank,
            "snippet": self.snippet,
        }

    @staticmethod
    def from_record(record: JsonDict) -> "EvidenceItem":
        return EvidenceItem(
            url=record["url"],
            retrieved_at=from_iso(record["retrieved_at"]) or utc_now(),
            snippet_hash=record["snippet_hash"],
            trust_score=float(record.get("trust_score", 0.0)),
            rank=int(record.get("rank", 0)),
            snippet=record.get("snippet"),
        )


@dataclass
class EvidenceBundle:
    bundle_id: str
    items: list[EvidenceItem]
    archived_text_hashes: list[str] | None = None
    created_at: datetime = field(default_factory=utc_now)

    def to_record(self) -> JsonDict:
        return {
            "bundle_id": self.bundle_id,
            "items": [item.to_record() for item in self.items],
            "archived_text_hashes": list(self.archived_text_hashes or []),
            "created_at": to_iso(self.created_at),
        }

    @staticmethod
    def from_record(record: JsonDict) -> "EvidenceBundle":
        return EvidenceBundle(
            bundle_id=record["bundle_id"],
            items=[EvidenceItem.from_record(item) for item in record.get("items", [])],
            archived_text_hashes=list(record.get("archived_text_hashes", [])),
            created_at=from_iso(record.get("created_at")) or utc_now(),
        )


@dataclass
class Resolution:
    question_id: str
    resolved_at: datetime
    y: float
    resolver_payload_raw: JsonDict
    resolution_confidence: float
    status: str = "resolved"
    id: int | None = None

    def to_record(self) -> JsonDict:
        record = asdict(self)
        record["resolved_at"] = to_iso(self.resolved_at)
        return record

    @staticmethod
    def from_record(record: JsonDict) -> "Resolution":
        return Resolution(
            id=record.get("id"),
            question_id=record["question_id"],
            resolved_at=from_iso(record["resolved_at"]) or utc_now(),
            y=float(record["y"]),
            resolver_payload_raw=dict(record.get("resolver_payload_raw", {})),
            resolution_confidence=float(record.get("resolution_confidence", 0.0)),
            status=record.get("status", "resolved"),
        )


@dataclass
class Score:
    question_id: str
    brier_ens: float
    logloss_ens: float
    brier_agents: dict[str, float]
    logloss_agents: dict[str, float]
    aggregates: JsonDict
    scored_at: datetime = field(default_factory=utc_now)
    id: int | None = None

    def to_record(self) -> JsonDict:
        record = asdict(self)
        record["scored_at"] = to_iso(self.scored_at)
        return record

    @staticmethod
    def from_record(record: JsonDict) -> "Score":
        return Score(
            id=record.get("id"),
            question_id=record["question_id"],
            brier_ens=float(record["brier_ens"]),
            logloss_ens=float(record["logloss_ens"]),
            brier_agents=dict(record.get("brier_agents", {})),
            logloss_agents=dict(record.get("logloss_agents", {})),
            aggregates=dict(record.get("aggregates", {})),
            scored_at=from_iso(record["scored_at"]) or utc_now(),
        )


@dataclass
class Diagnostic:
    question_id: str
    error_type: str
    structured_notes: list[str]
    recommended_patch: str
    created_at: datetime = field(default_factory=utc_now)
    id: int | None = None

    def to_record(self) -> JsonDict:
        record = asdict(self)
        record["created_at"] = to_iso(self.created_at)
        return record

    @staticmethod
    def from_record(record: JsonDict) -> "Diagnostic":
        return Diagnostic(
            id=record.get("id"),
            question_id=record["question_id"],
            error_type=record["error_type"],
            structured_notes=list(record.get("structured_notes", [])),
            recommended_patch=record.get("recommended_patch", ""),
            created_at=from_iso(record["created_at"]) or utc_now(),
        )


@dataclass
class QuestionCandidate:
    source: str
    source_id: str
    title: str
    description: str
    close_time: datetime | None
    resolve_time_expected: datetime | None
    tags: list[str]
    resolver_type: str
    resolver_config: JsonDict
    status: str = "open"
    raw_payload: JsonDict | None = None


@dataclass
class ResolutionCandidate:
    source: str
    source_id: str
    status: str
    resolved_at: datetime | None = None
    y: float | None = None
    resolution_confidence: float = 0.0
    raw_payload: JsonDict | None = None

