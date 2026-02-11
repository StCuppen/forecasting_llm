from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.core.schemas import (
    Diagnostic,
    EvidenceBundle,
    Prediction,
    Question,
    Resolution,
    Score,
)


class Storage(ABC):
    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_question(self, question: Question) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_questions(self, statuses: list[str] | None = None) -> list[Question]:
        raise NotImplementedError

    @abstractmethod
    def list_open_unforecasted_questions(self, limit: int | None = None) -> list[Question]:
        raise NotImplementedError

    @abstractmethod
    def list_due_unresolved_questions(self, now: datetime) -> list[Question]:
        raise NotImplementedError

    @abstractmethod
    def list_resolved_unscored_questions(self) -> list[Question]:
        raise NotImplementedError

    @abstractmethod
    def mark_question_status(self, question_id: str, status: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_question_duplicate(self, question_id: str, duplicate_of: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_question(self, question_id: str) -> Question | None:
        raise NotImplementedError

    @abstractmethod
    def upsert_evidence_bundle(self, bundle: EvidenceBundle) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_evidence_bundle(self, bundle_id: str) -> EvidenceBundle | None:
        raise NotImplementedError

    @abstractmethod
    def insert_prediction(self, prediction: Prediction) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_latest_prediction(self, question_id: str) -> Prediction | None:
        raise NotImplementedError

    @abstractmethod
    def count_predictions_since(self, since: datetime) -> int:
        raise NotImplementedError

    @abstractmethod
    def insert_resolution(self, resolution: Resolution) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_resolution(self, question_id: str) -> Resolution | None:
        raise NotImplementedError

    @abstractmethod
    def insert_score(self, score: Score) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert_diagnostic(self, diagnostic: Diagnostic) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_recent_resolved_with_scores(self, limit: int = 50) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_scores_after(self, last_score_id: int) -> list[Score]:
        raise NotImplementedError

    @abstractmethod
    def get_scores_for_calibration(
        self, domain_tag: str, limit: int
    ) -> list[tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_state_value(self, key: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def set_state_value(self, key: str, value: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_domain_weights(self, domain_tag: str) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def save_domain_weights(self, domain_tag: str, weights: dict[str, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_calibrator(self, domain_tag: str, version: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_calibrator(self, domain_tag: str) -> tuple[str, dict[str, Any]] | None:
        raise NotImplementedError
