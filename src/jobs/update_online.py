from __future__ import annotations

import argparse

from src.core.updater import update_online_weights
from src.jobs.common import bootstrap


def run_update_online(config_path: str = "league.toml") -> dict[str, dict[str, float]]:
    config, storage = bootstrap(config_path)
    try:
        updated = update_online_weights(
            storage=storage,
            eta=config.updater.eta,
            default_weight=config.updater.default_weight,
        )
        return updated
    finally:
        storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Update online ensemble weights.")
    parser.add_argument("--config", default="league.toml")
    args = parser.parse_args()
    print(run_update_online(config_path=args.config))


if __name__ == "__main__":
    main()

