from __future__ import annotations

import argparse

from src.core.updater import run_weekly_calibration
from src.jobs.common import bootstrap


def run_weekly_calibrate(config_path: str = "league.toml") -> dict[str, str]:
    config, storage = bootstrap(config_path)
    try:
        if not config.calibration.enabled:
            return {}
        domains = sorted(set(["general"] + list(config.domain_keywords.keys())))
        versions = run_weekly_calibration(
            storage=storage,
            domains=domains,
            window_size=config.calibration.window_size,
            min_points=config.calibration.min_points,
            bins=config.calibration.bins,
        )
        return versions
    finally:
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit weekly calibration models.")
    parser.add_argument("--config", default="league.toml")
    args = parser.parse_args()
    print(run_weekly_calibrate(config_path=args.config))


if __name__ == "__main__":
    main()

