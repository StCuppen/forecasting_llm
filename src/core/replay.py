from __future__ import annotations

from typing import Any

from src.core.schemas import EvidenceBundle, Prediction


def replay_output_shape(
    prediction: Prediction,
    evidence_bundle: EvidenceBundle | None,
    config_version: str = "league-v1",
) -> dict[str, Any]:
    evidence_count = len(evidence_bundle.items) if evidence_bundle else 0
    agent_names = sorted(prediction.p_agents.keys())
    return {
        "run_id": prediction.run_id,
        "question_id": prediction.question_id,
        "p_ens_type": type(prediction.p_ens).__name__,
        "agent_count": len(agent_names),
        "agent_names": agent_names,
        "evidence_count": evidence_count,
        "has_model_versions": bool(prediction.model_versions),
        "config_version": config_version,
    }

