from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


@dataclass
class SourceConfig:
    enabled: bool = True
    base_url: str | None = None
    timeout_seconds: int = 20
    max_retries: int = 3
    cache_ttl_seconds: int = 300


@dataclass
class ForecastConfig:
    dry_run_default: bool = False
    apply_calibration: bool = False
    max_questions_per_tick: int = 25
    weekly_prediction_limit: int = 25
    prediction_log_dir: str = "predictions/feedback_loop"
    write_prediction_markdown: bool = True


@dataclass
class ScoringConfig:
    logloss_epsilon: float = 1e-9


@dataclass
class UpdaterConfig:
    eta: float = 0.4
    default_weight: float = 1.0


@dataclass
class CalibrationConfig:
    enabled: bool = True
    min_points: int = 50
    window_size: int = 500
    bins: int = 10


@dataclass
class DiagnosticsConfig:
    llm_assisted: bool = False
    deterministic_only: bool = True


@dataclass
class LeagueConfig:
    db_path: str = "league.sqlite3"
    window_days: int = 7
    sources: dict[str, SourceConfig] = field(default_factory=dict)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    updater: UpdaterConfig = field(default_factory=UpdaterConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    domain_keywords: dict[str, list[str]] = field(default_factory=dict)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        return {}
    return value


def load_config(path: str = "league.toml") -> LeagueConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return LeagueConfig()

    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    sources_raw = _section(raw, "sources")
    sources = {
        name: SourceConfig(**values)
        for name, values in sources_raw.items()
        if isinstance(values, dict)
    }

    league = LeagueConfig(
        db_path=str(raw.get("db_path", "league.sqlite3")),
        window_days=int(raw.get("window_days", 7)),
        sources=sources,
        forecast=ForecastConfig(**_section(raw, "forecast")),
        scoring=ScoringConfig(**_section(raw, "scoring")),
        updater=UpdaterConfig(**_section(raw, "updater")),
        calibration=CalibrationConfig(**_section(raw, "calibration")),
        diagnostics=DiagnosticsConfig(**_section(raw, "diagnostics")),
        domain_keywords={
            str(k): list(v) for k, v in _section(raw, "domain_keywords").items()
        },
    )
    return league
