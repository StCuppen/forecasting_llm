from __future__ import annotations

import argparse
from dataclasses import asdict

from src.connectors import build_connectors
from src.core.dedupe import find_duplicate
from src.core.schemas import Question, QuestionCandidate
from src.core.utils import utc_now
from src.jobs.common import bootstrap


def candidate_to_question(candidate: QuestionCandidate) -> Question:
    now = utc_now()
    qid = f"{candidate.source}:{candidate.source_id}"
    return Question(
        id=qid,
        source=candidate.source,
        source_id=candidate.source_id,
        title=candidate.title,
        description=candidate.description,
        close_time=candidate.close_time,
        resolve_time_expected=candidate.resolve_time_expected,
        tags=list(candidate.tags),
        resolver_type=candidate.resolver_type,
        resolver_config=dict(candidate.resolver_config),
        status=candidate.status,
        raw_payload=candidate.raw_payload,
        created_at=now,
        updated_at=now,
    )


def run_ingest(config_path: str = "league.toml", window_days: int | None = None) -> dict[str, int]:
    config, storage = bootstrap(config_path)
    try:
        connectors = build_connectors(config)
        if window_days is None:
            window_days = config.window_days

        existing = storage.list_questions()
        all_seen = {q.id: q for q in existing}

        ingested = 0
        duplicates = 0
        errors = 0

        for source, connector in connectors.items():
            try:
                candidates = connector.list_candidates(window_days=window_days)
            except Exception:
                errors += 1
                continue
            for candidate in candidates:
                question = candidate_to_question(candidate)
                duplicate = find_duplicate(candidate, list(all_seen.values()))
                if duplicate.duplicate_of and duplicate.duplicate_of != question.id:
                    question.duplicate_of = duplicate.duplicate_of
                    duplicates += 1
                storage.upsert_question(question)
                all_seen[question.id] = question
                ingested += 1

        return {"ingested": ingested, "duplicates": duplicates, "errors": errors}
    finally:
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest 7-day candidates from enabled sources.")
    parser.add_argument("--config", default="league.toml", help="Path to league.toml")
    parser.add_argument("--window-days", type=int, default=None, help="Override window days")
    args = parser.parse_args()
    result = run_ingest(config_path=args.config, window_days=args.window_days)
    print(result)


if __name__ == "__main__":
    main()

