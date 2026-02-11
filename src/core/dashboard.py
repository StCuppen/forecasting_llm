from __future__ import annotations

from collections import defaultdict

from src.core.storage import Storage


def render_dashboard(storage: Storage, limit: int = 50) -> str:
    rows = storage.list_recent_resolved_with_scores(limit=limit)
    if not rows:
        return "No resolved/scored questions yet."

    misses = sorted(rows, key=lambda r: abs(r["p_ens"] - r["y"]), reverse=True)[:10]
    domain_perf: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        tags = row.get("tags") or []
        domain = tags[0] if tags else "general"
        domain_perf[domain].append(row["brier_ens"])

    lines = ["# 7-Day League Dashboard", "", f"Last {len(rows)} resolved questions"]
    lines.append("")
    lines.append("## Biggest Misses")
    for row in misses:
        lines.append(
            f"- {row['question_id']}: p={row['p_ens']:.3f}, y={row['y']:.1f}, brier={row['brier_ens']:.3f} | {row['title'][:90]}"
        )
    lines.append("")
    lines.append("## Per-Domain Mean Brier")
    for domain, briers in sorted(domain_perf.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        lines.append(f"- {domain}: {sum(briers)/len(briers):.3f} (n={len(briers)})")
    return "\n".join(lines)

