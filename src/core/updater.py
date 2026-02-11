from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from src.core.schemas import Score
from src.core.storage import Storage


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def multiplicative_weights_update(
    weights: dict[str, float],
    losses: dict[str, float],
    eta: float,
    default_weight: float = 1.0,
) -> dict[str, float]:
    updated = dict(weights)
    for agent, loss in losses.items():
        current = updated.get(agent, default_weight)
        updated[agent] = current * math.exp(-eta * float(loss))
    return normalize_weights(updated)


def detect_domain(tags: list[str], domain_keywords: dict[str, list[str]]) -> str:
    tags_lower = {t.lower() for t in tags}
    for domain, keywords in domain_keywords.items():
        for kw in keywords:
            if kw.lower() in tags_lower:
                return domain
    return "general"


def update_online_weights(
    storage: Storage,
    eta: float,
    default_weight: float,
) -> dict[str, dict[str, float]]:
    last_seen_raw = storage.get_state_value("updater_last_score_id")
    last_seen = int(last_seen_raw) if last_seen_raw else 0
    new_scores = storage.list_scores_after(last_seen)
    latest_by_domain: dict[str, dict[str, float]] = {}
    max_id = last_seen
    for score in new_scores:
        domain = str(score.aggregates.get("domain_tag", "general"))
        current = storage.get_domain_weights(domain)
        updated = multiplicative_weights_update(
            current, score.brier_agents, eta=eta, default_weight=default_weight
        )
        storage.save_domain_weights(domain, updated)
        latest_by_domain[domain] = updated
        if score.id is not None:
            max_id = max(max_id, score.id)
    storage.set_state_value("updater_last_score_id", str(max_id))
    return latest_by_domain


def fit_histogram_calibrator(points: list[tuple[float, float]], bins: int) -> dict[str, Any]:
    bins = max(2, int(bins))
    bucket_counts = [0 for _ in range(bins)]
    bucket_sum = [0.0 for _ in range(bins)]
    for p, y in points:
        idx = min(bins - 1, max(0, int(p * bins)))
        bucket_counts[idx] += 1
        bucket_sum[idx] += y

    bucket_rate: list[float] = []
    for i in range(bins):
        if bucket_counts[i] == 0:
            bucket_rate.append((i + 0.5) / bins)
        else:
            bucket_rate.append(bucket_sum[i] / bucket_counts[i])

    return {
        "method": "histogram",
        "bins": bins,
        "bucket_counts": bucket_counts,
        "bucket_rate": bucket_rate,
    }


def apply_calibrator(payload: dict[str, Any], p: float) -> float:
    if payload.get("method") != "histogram":
        return p
    bins = int(payload.get("bins", 10))
    rates = payload.get("bucket_rate", [])
    if not rates:
        return p
    idx = min(bins - 1, max(0, int(float(p) * bins)))
    return float(rates[idx])


def run_weekly_calibration(
    storage: Storage,
    domains: list[str],
    window_size: int,
    min_points: int,
    bins: int,
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for domain in domains:
        points = storage.get_scores_for_calibration(domain, window_size)
        if len(points) < min_points:
            continue
        payload = fit_histogram_calibrator(points, bins=bins)
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        storage.save_calibrator(domain, version, payload)
        versions[domain] = version
    return versions

