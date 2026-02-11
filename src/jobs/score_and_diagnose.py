from __future__ import annotations

import argparse

from src.core.diagnostics import diagnose_prediction
from src.core.pipeline import infer_domain_tag
from src.core.schemas import Score
from src.core.scoring import brier_score, log_loss
from src.core.utils import utc_now
from src.jobs.common import bootstrap


def run_score_and_diagnose(config_path: str = "league.toml") -> dict[str, int]:
    config, storage = bootstrap(config_path)
    try:
        questions = storage.list_resolved_unscored_questions()
        scored = 0
        skipped = 0
        for question in questions:
            prediction = storage.get_latest_prediction(question.id)
            resolution = storage.get_resolution(question.id)
            if prediction is None or resolution is None:
                skipped += 1
                continue
            brier_ens = brier_score(prediction.p_ens, resolution.y)
            logloss_ens = log_loss(prediction.p_ens, resolution.y, epsilon=config.scoring.logloss_epsilon)
            brier_agents = {
                name: brier_score(prob, resolution.y) for name, prob in prediction.p_agents.items()
            }
            logloss_agents = {
                name: log_loss(prob, resolution.y, epsilon=config.scoring.logloss_epsilon)
                for name, prob in prediction.p_agents.items()
            }
            evidence = storage.get_evidence_bundle(prediction.evidence_bundle_id)
            domain_tag = infer_domain_tag(question.tags, config)
            score = Score(
                question_id=question.id,
                brier_ens=brier_ens,
                logloss_ens=logloss_ens,
                brier_agents=brier_agents,
                logloss_agents=logloss_agents,
                aggregates={
                    "abs_error_ens": abs(prediction.p_ens - resolution.y),
                    "domain_tag": domain_tag,
                    "evidence_bundle_id": prediction.evidence_bundle_id,
                    "evidence_count": len(evidence.items) if evidence else 0,
                },
                scored_at=utc_now(),
            )
            storage.insert_score(score)
            diagnostic = diagnose_prediction(
                question=question,
                prediction=prediction,
                resolution=resolution,
                score=score,
                evidence=evidence,
                llm_assisted=config.diagnostics.llm_assisted and not config.diagnostics.deterministic_only,
            )
            storage.insert_diagnostic(diagnostic)
            scored += 1
        return {"scored": scored, "skipped": skipped}
    finally:
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score newly resolved questions and create diagnostics.")
    parser.add_argument("--config", default="league.toml")
    args = parser.parse_args()
    print(run_score_and_diagnose(config_path=args.config))


if __name__ == "__main__":
    main()

