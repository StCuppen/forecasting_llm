from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import uuid

from src.core.forecast_runner import run_forecast_for_question
from src.core.pipeline import infer_domain_tag
from src.core.schemas import Prediction, Question
from src.core.utils import stable_hash, to_iso, utc_now
from src.jobs.common import bootstrap


def _safe_name(value: str, max_len: int = 80) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    clean = clean.strip("_")
    if not clean:
        clean = "question"
    return clean[:max_len]


def write_prediction_markdown(
    log_dir: str,
    run_id: str,
    made_at_iso: str,
    question: Question,
    domain_tag: str,
    p_ens: float,
    p_agents: dict[str, float],
    model_versions: dict[str, str],
    rationale_markdown: str,
    evidence_urls: list[str],
) -> str:
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{made_at_iso.replace(':', '').replace('-', '')}_{_safe_name(question.id)}_{run_id[:8]}.md"
    path = out_dir / filename
    lines = [
        f"# Feedback Loop Prediction",
        "",
        f"- Run ID: `{run_id}`",
        f"- Made At: `{made_at_iso}`",
        f"- Question ID: `{question.id}`",
        f"- Source: `{question.source}` / `{question.source_id}`",
        f"- Domain Tag: `{domain_tag}`",
        f"- Ensemble Probability: `{p_ens:.6f}`",
        "",
        "## Agent Probabilities",
    ]
    if p_agents:
        for agent, prob in sorted(p_agents.items()):
            lines.append(f"- `{agent}`: `{prob:.6f}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Model Versions"])
    if model_versions:
        for agent, version in sorted(model_versions.items()):
            lines.append(f"- `{agent}`: `{version}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Evidence URLs"])
    if evidence_urls:
        for url in evidence_urls:
            lines.append(f"- {url}")
    else:
        lines.append("- none")
    lines.extend(["", rationale_markdown.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.as_posix())


def run_forecast_open(config_path: str = "league.toml", dry_run: bool | None = None) -> dict[str, int]:
    config, storage = bootstrap(config_path)
    try:
        if dry_run is None:
            dry_run = config.forecast.dry_run_default
        now = utc_now()
        since = now - timedelta(days=7)
        recent_prediction_count = storage.count_predictions_since(since)
        total_open_unforecasted = len(storage.list_open_unforecasted_questions(limit=None))
        remaining_budget = max(0, int(config.forecast.weekly_prediction_limit) - recent_prediction_count)
        if remaining_budget <= 0:
            return {
                "forecasted": 0,
                "errors": 0,
                "skipped_due_weekly_limit": total_open_unforecasted,
            }

        per_run_limit = min(int(config.forecast.max_questions_per_tick), remaining_budget)
        questions = storage.list_open_unforecasted_questions(limit=per_run_limit)
        forecasted = 0
        errors = 0
        skipped_due_weekly_limit = max(0, total_open_unforecasted - len(questions))
        for question in questions:
            try:
                domain_tag = infer_domain_tag(question.tags, config)
                calibrator = storage.get_calibrator(domain_tag) if config.forecast.apply_calibration else None
                result = run_forecast_for_question(
                    question,
                    dry_run=dry_run,
                    apply_calibration=config.forecast.apply_calibration,
                    calibrator=calibrator,
                )
                storage.upsert_evidence_bundle(result.evidence_bundle)
                run_id = str(uuid.uuid4())
                made_at = utc_now()
                made_at_iso = to_iso(made_at) or ""
                evidence_urls = [item.url for item in result.evidence_bundle.items]
                prediction_md_path = None
                if config.forecast.write_prediction_markdown:
                    prediction_md_path = write_prediction_markdown(
                        log_dir=config.forecast.prediction_log_dir,
                        run_id=run_id,
                        made_at_iso=made_at_iso,
                        question=question,
                        domain_tag=domain_tag,
                        p_ens=result.p_ens,
                        p_agents=result.p_agents,
                        model_versions=result.model_versions,
                        rationale_markdown=result.rationale_markdown,
                        evidence_urls=evidence_urls,
                    )
                forecast_context = dict(result.forecast_context)
                forecast_context.update(
                    {
                        "question_snapshot_hash": stable_hash(
                            f"{question.title}|{question.description}|{question.close_time}|{question.resolve_time_expected}|{question.tags}"
                        ),
                        "source": question.source,
                        "domain_tag": domain_tag,
                        "prediction_markdown_path": prediction_md_path,
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
                forecasted += 1
            except Exception:
                errors += 1
        return {
            "forecasted": forecasted,
            "errors": errors,
            "skipped_due_weekly_limit": skipped_due_weekly_limit,
        }
    finally:
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecasts for open questions.")
    parser.add_argument("--config", default="league.toml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true", help="Force non-dry-run")
    args = parser.parse_args()
    dry_run = True if args.dry_run else False if args.live else None
    result = run_forecast_open(config_path=args.config, dry_run=dry_run)
    print(result)


if __name__ == "__main__":
    main()
