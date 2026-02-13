from __future__ import annotations

import argparse
import uuid

from src.core.updater import check_signposts, update_online_weights
from src.core.forecast_runner import run_forecast_for_question
from src.core.pipeline import infer_domain_tag
from src.core.schemas import Prediction
from src.core.utils import stable_hash, utc_now
from src.jobs.common import bootstrap


def run_update_online(config_path: str = "league.toml") -> dict[str, object]:
    config, storage = bootstrap(config_path)
    try:
        updated = update_online_weights(
            storage=storage,
            eta=config.updater.eta,
            default_weight=config.updater.default_weight,
        )
        signpost_checks = check_signposts(storage=storage)
        reforecasted = 0
        reforecast_ids: list[str] = []
        for check in signpost_checks:
            if not bool(check.get("trigger_reforecast", False)):
                continue
            question_id = str(check.get("question_id", ""))
            if not question_id:
                continue
            question = storage.get_question(question_id)
            if question is None:
                continue
            try:
                domain_tag = infer_domain_tag(question.tags, config)
                calibrator = storage.get_calibrator(domain_tag) if config.forecast.apply_calibration else None
                result = run_forecast_for_question(
                    question,
                    dry_run=config.forecast.dry_run_default,
                    apply_calibration=config.forecast.apply_calibration,
                    calibrator=calibrator,
                )
                storage.upsert_evidence_bundle(result.evidence_bundle)
                made_at = utc_now()
                run_id = str(uuid.uuid4())
                forecast_context = dict(result.forecast_context)
                forecast_context.update(
                    {
                        "question_snapshot_hash": stable_hash(
                            f"{question.title}|{question.description}|{question.close_time}|{question.resolve_time_expected}|{question.tags}"
                        ),
                        "source": question.source,
                        "domain_tag": domain_tag,
                        "triggered_by_signpost": True,
                        "signpost_check": check,
                    }
                )
                prediction = Prediction(
                    question_id=question.id,
                    run_id=run_id,
                    made_at=made_at,
                    p_ens=result.p_ens,
                    p_agents=result.p_agents,
                    model_versions=result.model_versions,
                    evidence_bundle_id=result.evidence_bundle.bundle_id,
                    cost_estimate=result.cost_estimate,
                    latency=result.latency,
                    forecast_context=forecast_context,
                    calibrator_version=(calibrator[0] if calibrator else None),
                )
                storage.insert_prediction(prediction)
                storage.mark_question_status(question.id, "forecasted")
                reforecasted += 1
                reforecast_ids.append(question.id)
            except Exception:
                continue
        return {
            "weights": updated,
            "signpost_checks": signpost_checks,
            "reforecasted": reforecasted,
            "reforecast_question_ids": reforecast_ids,
        }
    finally:
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Update online ensemble weights.")
    parser.add_argument("--config", default="league.toml")
    args = parser.parse_args()
    print(run_update_online(config_path=args.config))


if __name__ == "__main__":
    main()
