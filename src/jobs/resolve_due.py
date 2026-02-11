from __future__ import annotations

import argparse

from src.connectors import build_connectors
from src.core.schemas import Resolution
from src.core.utils import utc_now
from src.jobs.common import bootstrap


def run_resolve_due(config_path: str = "league.toml") -> dict[str, int]:
    config, storage = bootstrap(config_path)
    try:
        connectors = build_connectors(config)
        due = storage.list_due_unresolved_questions(now=utc_now())
        resolved = 0
        manual = 0
        unresolved = 0
        for question in due:
            connector = connectors.get(question.source)
            if connector is None:
                unresolved += 1
                continue
            result = connector.get_resolution(question.source_id)
            if result.status == "resolved" and result.y is not None and result.resolved_at is not None:
                storage.insert_resolution(
                    Resolution(
                        question_id=question.id,
                        resolved_at=result.resolved_at,
                        y=float(result.y),
                        resolver_payload_raw=result.raw_payload or {},
                        resolution_confidence=result.resolution_confidence,
                        status="resolved",
                    )
                )
                storage.mark_question_status(question.id, "resolved")
                resolved += 1
            elif result.status == "manual_review":
                storage.mark_question_status(question.id, "manual_review")
                manual += 1
            else:
                unresolved += 1
        return {"resolved": resolved, "manual_review": manual, "unresolved": unresolved}
    finally:
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve due questions.")
    parser.add_argument("--config", default="league.toml")
    args = parser.parse_args()
    print(run_resolve_due(config_path=args.config))


if __name__ == "__main__":
    main()

