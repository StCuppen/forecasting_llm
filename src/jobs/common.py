from __future__ import annotations

from src.core.config import LeagueConfig, load_config
from src.core.sqlite_storage import SQLiteStorage


def bootstrap(config_path: str) -> tuple[LeagueConfig, SQLiteStorage]:
    config = load_config(config_path)
    storage = SQLiteStorage(config.db_path)
    storage.init()
    return config, storage

