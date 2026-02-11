import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

from bot.agent.agent_experiment import run_ensemble_forecast


def _load_dataset(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if "question" not in obj or "y" not in obj:
                continue
            y = float(obj["y"])
            if y < 0 or y > 1:
                continue
            items.append({"question": str(obj["question"]), "y": y})
    return items


def _stable_holdout_split(items: list[dict], holdout_ratio: float = 0.35) -> tuple[list[dict], list[dict]]:
    train, holdout = [], []
    for item in items:
        bucket = abs(hash(item["question"])) % 1000
        if bucket < int(holdout_ratio * 1000):
            holdout.append(item)
        else:
            train.append(item)
    return train, holdout


def _logloss(p: float, y: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _flag_combos() -> list[tuple[str, dict]]:
    return [
        (
            "baseline_all_off",
            {
                "spec_lock": False,
                "evidence_ledger": False,
                "numeric_provenance": False,
                "market_snapshot": False,
                "outlier_xexam": False,
            },
        ),
        (
            "all_on",
            {
                "spec_lock": True,
                "evidence_ledger": True,
                "numeric_provenance": True,
                "market_snapshot": True,
                "outlier_xexam": True,
            },
        ),
        (
            "all_on_no_outlier_xexam",
            {
                "spec_lock": True,
                "evidence_ledger": True,
                "numeric_provenance": True,
                "market_snapshot": True,
                "outlier_xexam": False,
            },
        ),
    ]


async def _run_one_combo(name: str, flags: dict, holdout: list[dict]) -> dict:
    rows = []
    spec_mismatch_count = 0
    market_hall_count = 0
    orphan_total = 0
    numeric_claim_total = 0
    token_total = 0
    t0 = time.time()

    for item in holdout:
        started = time.time()
        result = await run_ensemble_forecast(
            question=item["question"],
            publish_to_metaculus=False,
            use_react=True,
            feature_flags=flags,
        )
        elapsed = time.time() - started
        p = float(result["final_probability"])
        y = float(item["y"])
        brier = (p - y) ** 2
        ll = _logloss(p, y)
        rows.append({"question": item["question"], "p": p, "y": y, "brier": brier, "logloss": ll, "runtime_s": elapsed})

        for model_row in result.get("individual_results", []):
            model_res = model_row.get("result")
            if not model_res:
                continue
            token_total += int((model_res.token_usage or {}).get("total", 0))
            risks = set(model_res.risk_flags or [])
            if "SPEC_MAJORDRIFT" in risks:
                spec_mismatch_count += 1
            if "MARKET_HALLUCINATION" in risks:
                market_hall_count += 1
            diag = model_res.diagnostics or {}
            num = diag.get("numeric_provenance") or {}
            orphan_total += int(num.get("orphan", 0))
            numeric_claim_total += int(num.get("total_claims", 0))

    runtime_total = time.time() - t0
    n = len(rows) if rows else 1
    model_eval_count = max(1, len(rows) * 3)
    avg_brier = statistics.mean([r["brier"] for r in rows]) if rows else None
    avg_logloss = statistics.mean([r["logloss"] for r in rows]) if rows else None
    return {
        "name": name,
        "flags": flags,
        "n_holdout": len(rows),
        "avg_brier": avg_brier,
        "avg_logloss": avg_logloss,
        "spec_mismatch_rate": spec_mismatch_count / model_eval_count,
        "orphan_numeric_rate": (orphan_total / numeric_claim_total) if numeric_claim_total else 0.0,
        "market_hallucination_rate": market_hall_count / model_eval_count,
        "token_total": token_total,
        "runtime_s_total": runtime_total,
        "rows": rows,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run feature-flag ablations on holdout forecast questions.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL with fields: question, y")
    parser.add_argument("--max-holdout", type=int, default=6, help="Max holdout questions to run")
    parser.add_argument("--out", default="forecasts/ablation_results.json", help="Output JSON path")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    items = _load_dataset(ds_path)
    if not items:
        raise SystemExit("No valid dataset rows found.")

    _, holdout = _stable_holdout_split(items, holdout_ratio=0.35)
    if not holdout:
        holdout = items[:]
    holdout = holdout[: args.max_holdout]

    print(f"Loaded {len(items)} rows, using holdout size {len(holdout)}")
    results = []
    for name, flags in _flag_combos():
        print(f"\n=== Running combo: {name} ===")
        res = await _run_one_combo(name, flags, holdout)
        results.append(res)
        print(
            f"{name}: brier={res['avg_brier']:.4f} logloss={res['avg_logloss']:.4f} "
            f"spec_mismatch_rate={res['spec_mismatch_rate']:.3f} "
            f"orphan_numeric_rate={res['orphan_numeric_rate']:.3f} "
            f"market_hallucination_rate={res['market_hallucination_rate']:.3f} "
            f"tokens={res['token_total']} runtime_s={res['runtime_s_total']:.1f}"
        )

    base = next((r for r in results if r["name"] == "baseline_all_off"), None)
    if base:
        for r in results:
            if r is base:
                continue
            if base["avg_brier"] is not None and r["avg_brier"] is not None:
                r["delta_brier_vs_baseline"] = r["avg_brier"] - base["avg_brier"]
            if base["avg_logloss"] is not None and r["avg_logloss"] is not None:
                r["delta_logloss_vs_baseline"] = r["avg_logloss"] - base["avg_logloss"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved ablation results to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
