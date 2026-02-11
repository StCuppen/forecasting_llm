from __future__ import annotations

import argparse

from src.jobs.forecast_open import run_forecast_open
from src.jobs.ingest_7day import run_ingest
from src.jobs.resolve_due import run_resolve_due
from src.jobs.score_and_diagnose import run_score_and_diagnose
from src.jobs.tick import run_tick
from src.jobs.update_online import run_update_online
from src.jobs.weekly_calibrate import run_weekly_calibrate


def main() -> None:
    parser = argparse.ArgumentParser(description="7-day forecasting league jobs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest_7day")
    p_ingest.add_argument("--config", default="league.toml")
    p_ingest.add_argument("--window-days", type=int, default=None)

    p_forecast = sub.add_parser("forecast_open")
    p_forecast.add_argument("--config", default="league.toml")
    p_forecast.add_argument("--dry-run", action="store_true")
    p_forecast.add_argument("--live", action="store_true")

    p_resolve = sub.add_parser("resolve_due")
    p_resolve.add_argument("--config", default="league.toml")
    p_score = sub.add_parser("score_and_diagnose")
    p_score.add_argument("--config", default="league.toml")
    p_update = sub.add_parser("update_online")
    p_update.add_argument("--config", default="league.toml")
    p_weekly = sub.add_parser("weekly_calibrate")
    p_weekly.add_argument("--config", default="league.toml")

    p_tick = sub.add_parser("tick")
    p_tick.add_argument("--config", default="league.toml")
    p_tick.add_argument("--dry-run", action="store_true")
    p_tick.add_argument("--run-weekly", action="store_true")

    args = parser.parse_args()
    cfg = args.config
    if args.cmd == "ingest_7day":
        print(run_ingest(config_path=cfg, window_days=args.window_days))
    elif args.cmd == "forecast_open":
        dry = True if args.dry_run else False if args.live else None
        print(run_forecast_open(config_path=cfg, dry_run=dry))
    elif args.cmd == "resolve_due":
        print(run_resolve_due(config_path=cfg))
    elif args.cmd == "score_and_diagnose":
        print(run_score_and_diagnose(config_path=cfg))
    elif args.cmd == "update_online":
        print(run_update_online(config_path=cfg))
    elif args.cmd == "weekly_calibrate":
        print(run_weekly_calibrate(config_path=cfg))
    elif args.cmd == "tick":
        result = run_tick(config_path=cfg, dry_run=args.dry_run, run_weekly=args.run_weekly)
        print(result)


if __name__ == "__main__":
    main()
