from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.core.dashboard import render_dashboard
from src.jobs.common import bootstrap
from src.jobs.forecast_open import run_forecast_open
from src.jobs.ingest_7day import run_ingest
from src.jobs.resolve_due import run_resolve_due
from src.jobs.score_and_diagnose import run_score_and_diagnose
from src.jobs.update_online import run_update_online
from src.jobs.weekly_calibrate import run_weekly_calibrate


def run_tick(
    config_path: str = "league.toml",
    dry_run: bool = False,
    run_weekly: bool = False,
) -> dict:
    result: dict = {}
    result["ingest_7day"] = run_ingest(config_path=config_path)
    result["forecast_open"] = run_forecast_open(config_path=config_path, dry_run=dry_run)
    result["resolve_due"] = run_resolve_due(config_path=config_path)
    result["score_and_diagnose"] = run_score_and_diagnose(config_path=config_path)
    result["update_online"] = run_update_online(config_path=config_path)
    if run_weekly:
        result["weekly_calibrate"] = run_weekly_calibrate(config_path=config_path)

    _, storage = bootstrap(config_path)
    try:
        result["dashboard"] = render_dashboard(storage, limit=50)
    finally:
        storage.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full league tick.")
    parser.add_argument("--config", default="league.toml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-weekly", action="store_true")
    args = parser.parse_args()
    result = run_tick(config_path=args.config, dry_run=args.dry_run, run_weekly=args.run_weekly)
    print(result["ingest_7day"])
    print(result["forecast_open"])
    print(result["resolve_due"])
    print(result["score_and_diagnose"])
    print(result["update_online"])
    if "weekly_calibrate" in result:
        print(result["weekly_calibrate"])
    print(result["dashboard"])


if __name__ == "__main__":
    main()

