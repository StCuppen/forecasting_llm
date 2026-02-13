from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import os
import re
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

from forecasting_tools import MetaculusApi

from bot.aggregation import AggregatedForecast, ForecastRun, aggregate_forecasts
from bot.publish_gate import (
    EvidenceItem as GateEvidenceItem,
    SpecLockResult,
    evaluate_publish_gate,
    shrink_probability,
)
from bot.coherence import enforce_sum_to_one
from .prompts import (
    LEAN_BINARY_FORECAST_PROMPT,
    LEAN_MC_APPEND_PROMPT,
    LEAN_NUMERIC_APPEND_PROMPT,
)
from .retrieval import multi_provider_search
from .utils import ExaClient, call_openrouter_llm, clean_indents, get_token_usage, reset_token_usage

logger = logging.getLogger(__name__)


@dataclass
class LeanRunOutput:
    probability: float
    explanation: str
    search_count: int = 0
    sources: list | None = None
    token_usage: dict | None = None
    warnings: list | None = None
    risk_flags: list | None = None
    diagnostics: dict | None = None

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = []
        if self.token_usage is None:
            self.token_usage = {"prompt": 0, "completion": 0, "total": 0}
        if self.warnings is None:
            self.warnings = []
        if self.risk_flags is None:
            self.risk_flags = []
        if self.diagnostics is None:
            self.diagnostics = {}


def _default_forecast_model_families() -> list[dict[str, Any]]:
    return [
        {
            "model": "moonshotai/kimi-k2.5",
            "runs": 2,
            "label": "Kimi K2.5",
            "reasoning_effort": None,
            "max_tokens": 24000,
            "temperature": 0.35,
        },
        {
            "model": "openai/gpt-5-mini",
            "runs": 2,
            "label": "GPT-5 Mini",
            "reasoning_effort": "medium",
            "max_tokens": 20000,
            "temperature": 0.35,
        },
        {
            "model": "google/gemini-2.5-flash",
            "runs": 1,
            "label": "Gemini 2.5 Flash",
            "reasoning_effort": "medium",
            "max_tokens": 20000,
            "temperature": 0.35,
        },
    ]


def _expand_model_roster(raw_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for family in raw_models:
        model_name = str(family.get("model") or family.get("name") or "").strip()
        if not model_name:
            continue
        runs = max(1, int(family.get("runs", 1) or 1))
        label_base = str(family.get("label") or model_name)
        for idx in range(1, runs + 1):
            label = f"{label_base} [{idx}/{runs}]" if runs > 1 else label_base
            expanded.append(
                {
                    "name": model_name,
                    "label": label,
                    "family_label": label_base,
                    "run_index": idx,
                    "max_tokens": int(family.get("max_tokens", 12000)),
                    "temperature": float(family.get("temperature", 0.35)),
                    "reasoning_effort": family.get("reasoning_effort"),
                }
            )
    return expanded


def _parse_deadline(*texts: str) -> Optional[datetime]:
    blob = " ".join([t for t in texts if t]).strip()
    if not blob:
        return None
    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", blob)
    if iso_match:
        try:
            return datetime(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
                tzinfo=timezone.utc,
            )
        except Exception:
            pass
    month_match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(20\d{2})\b",
        blob,
        re.IGNORECASE,
    )
    if month_match:
        try:
            return datetime.strptime(
                f"{month_match.group(1)} {month_match.group(2)} {month_match.group(3)}",
                "%B %d %Y",
            ).replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _source_name_from_url(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    return host if host else "unknown"


_PRIMARY_SOURCE_TEMPLATES: list[tuple[list[str], list[str]]] = [
    (["openai", "gpt", "chatgpt", "api pricing"], ["openai.com"]),
    (["google", "gemini", "deepmind"], ["blog.google", "deepmind.google"]),
    (["anthropic", "claude"], ["anthropic.com"]),
    (["meta", "llama"], ["ai.meta.com"]),
    (["cpi", "inflation", "consumer price"], ["bls.gov"]),
    (["gdp", "gross domestic product"], ["bea.gov"]),
    (["unemployment", "jobs report", "nonfarm payroll"], ["bls.gov"]),
    (["fed", "federal reserve", "interest rate", "fomc"], ["federalreserve.gov"]),
    (["sec ", "securities"], ["sec.gov"]),
    (["election", "vote", "ballot"], ["fec.gov"]),
    (["fda ", "drug approval"], ["fda.gov"]),
    (["nasa", "spacex", "launch"], ["nasa.gov", "spacex.com"]),
    (["tesla"], ["tesla.com", "ir.tesla.com"]),
    (["apple"], ["apple.com"]),
    (["microsoft", "copilot"], ["microsoft.com", "blogs.microsoft.com"]),
    (["amazon", "aws"], ["aboutamazon.com", "aws.amazon.com"]),
    (["un ", "united nations"], ["un.org"]),
    (["who ", "world health"], ["who.int"]),
    (["imf "], ["imf.org"]),
    (["world bank"], ["worldbank.org"]),
    (["polymarket"], ["polymarket.com"]),
    (["metaculus"], ["metaculus.com"]),
    (["manifold"], ["manifold.markets"]),
]


def _identify_primary_source_urls(
    question_title: str,
    resolution_criteria: str,
) -> list[str]:
    """Heuristic: identify 0-3 likely canonical source domains for this question."""
    blob = f"{question_title} {resolution_criteria}".lower()
    matched_domains: list[str] = []
    for keywords, domains in _PRIMARY_SOURCE_TEMPLATES:
        if any(kw in blob for kw in keywords):
            for d in domains:
                if d not in matched_domains:
                    matched_domains.append(d)
    return matched_domains[:3]


def _is_primary_source_url(url: str, question_domains: list[str] | None = None) -> bool:
    host = (urlparse(url).netloc or "").lower()
    if any(marker in host for marker in [".gov", ".edu", ".int", "official", "sec.gov", "europa.eu"]):
        return True
    if question_domains:
        for domain in question_domains:
            if domain.lower() in host:
                return True
    return False


@dataclass
class EnrichedEvidence:
    """Evidence item with LLM-extracted content and scored relevance."""
    url: str
    source_name: str
    title: str
    extracted_claims: str
    relevance_score: float
    contains_concrete_datapoint: bool
    is_primary_source: bool
    raw_text_length: int


_EVIDENCE_EXTRACTION_PROMPT = """Extract the specific claims, datapoints, and facts from this document that are relevant to answering the forecasting question below. Ignore navigation elements, cookie banners, advertisements, and boilerplate.

QUESTION: {question}
RESOLUTION CRITERIA: {resolution_criteria}

DOCUMENT URL: {url}
DOCUMENT TEXT:
{text}

Output format (be concise):
CLAIMS:
- [claim 1]
- [claim 2]
...

RELEVANCE: [0.0-1.0 score: 0.0=completely irrelevant, 0.3=tangentially related, 0.6=somewhat relevant general context, 0.8=directly relevant with concrete data, 1.0=resolution-critical primary data]
HAS_DATAPOINT: [yes/no — does this contain a specific number, date, measurement, or verifiable fact bearing on the question?]
"""


async def _extract_and_score_single(
    doc: Any,
    question_title: str,
    resolution_criteria: str,
) -> EnrichedEvidence | None:
    """Extract relevant content from a single doc via cheap LLM call."""
    url = str(getattr(doc, "url", "") or "").strip()
    if not url.startswith("http"):
        return None
    raw_text = str(getattr(doc, "text", "") or getattr(doc, "snippet", "") or "")
    title = str(getattr(doc, "title", "") or "Untitled")
    text_for_extraction = re.sub(r"\s+", " ", raw_text).strip()[:3000]
    if len(text_for_extraction) < 50:
        return None

    prompt = clean_indents(_EVIDENCE_EXTRACTION_PROMPT.format(
        question=question_title,
        resolution_criteria=resolution_criteria or question_title,
        url=url,
        text=text_for_extraction,
    ))
    try:
        response = await call_openrouter_llm(
            prompt=prompt,
            model=os.getenv("EVIDENCE_EXTRACTION_MODEL", "google/gemini-2.5-flash"),
            temperature=0.1,
            max_tokens=800,
            usage_label="evidence_extraction",
            reasoning_effort="low",
        )
        rel_match = re.search(r"RELEVANCE:\s*([\d.]+)", response, re.IGNORECASE)
        relevance = float(rel_match.group(1)) if rel_match else 0.5
        relevance = max(0.0, min(1.0, relevance))

        dp_match = re.search(r"HAS_DATAPOINT:\s*(yes|no)", response, re.IGNORECASE)
        has_datapoint = dp_match.group(1).lower() == "yes" if dp_match else False

        claims_match = re.search(
            r"CLAIMS:\s*\n(.*?)(?:\n\s*RELEVANCE:|\Z)", response, re.DOTALL | re.IGNORECASE
        )
        claims_text = claims_match.group(1).strip() if claims_match else response[:500]

        return EnrichedEvidence(
            url=url,
            source_name=_source_name_from_url(url),
            title=title,
            extracted_claims=claims_text[:600],
            relevance_score=relevance,
            contains_concrete_datapoint=has_datapoint,
            is_primary_source=False,
            raw_text_length=len(raw_text),
        )
    except Exception as exc:
        logger.warning(f"Evidence extraction failed for {url}: {exc}")
        return None


async def _extract_and_score_evidence(
    docs: list[Any],
    question_title: str,
    resolution_criteria: str,
    max_docs: int = 12,
) -> list[EnrichedEvidence]:
    """Run parallel LLM extraction+scoring on docs."""
    tasks = [
        _extract_and_score_single(doc, question_title, resolution_criteria)
        for doc in docs[:max_docs]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    enriched: list[EnrichedEvidence] = []
    for r in results:
        if isinstance(r, EnrichedEvidence):
            enriched.append(r)
        elif isinstance(r, Exception):
            logger.warning(f"Evidence extraction task failed: {r}")
    enriched.sort(key=lambda e: e.relevance_score, reverse=True)
    return enriched


def _build_evidence_bundle_from_enriched(
    enriched: list[EnrichedEvidence],
    max_docs: int = 12,
    max_chars: int = 18000,
) -> tuple[str, list[GateEvidenceItem], list[dict[str, Any]]]:
    """Build evidence bundle from LLM-enriched evidence items."""
    now = datetime.now(timezone.utc)
    evidence_items: list[GateEvidenceItem] = []
    source_rows: list[dict[str, Any]] = []
    chunks: list[str] = []
    for ev in enriched[:max_docs]:
        evidence_items.append(
            GateEvidenceItem(
                content=ev.extracted_claims or ev.title,
                source_url=ev.url,
                source_name=ev.source_name,
                retrieved_at=now,
                published_at=None,
                relevance_score=ev.relevance_score,
                is_primary_source=ev.is_primary_source,
            )
        )
        source_rows.append(
            {
                "url": ev.url,
                "title": ev.title[:120],
                "date": "",
                "quality": "High" if ev.is_primary_source else ("Medium" if ev.relevance_score >= 0.7 else "Low"),
                "snippet": (ev.extracted_claims or ev.title)[:260],
            }
        )
        chunks.append(
            f"[{len(evidence_items)}] {ev.source_name} | {ev.url} | "
            f"retrieved={now.date().isoformat()} | primary={ev.is_primary_source} | "
            f"relevance={ev.relevance_score:.2f} | datapoint={ev.contains_concrete_datapoint}\n"
            f"{ev.extracted_claims or ev.title}"
        )
    bundle = "\n\n".join(chunks)
    if not bundle:
        bundle = "No evidence retrieved from search."
    return bundle[:max_chars], evidence_items, source_rows


def _build_evidence_bundle_from_docs(
    docs: list[Any], max_docs: int = 12, max_chars: int = 18000,
    question_domains: list[str] | None = None,
) -> tuple[str, list[GateEvidenceItem], list[dict[str, Any]]]:
    """Fallback: build evidence from raw docs when LLM extraction is skipped."""
    now = datetime.now(timezone.utc)
    evidence_items: list[GateEvidenceItem] = []
    source_rows: list[dict[str, Any]] = []
    chunks: list[str] = []
    for doc in docs[:max_docs]:
        url = str(getattr(doc, "url", "") or "").strip()
        if not url.startswith("http"):
            continue
        title = str(getattr(doc, "title", "") or "Untitled")
        raw_text = str(getattr(doc, "text", "") or getattr(doc, "snippet", "") or "")
        content = re.sub(r"\s+", " ", raw_text).strip()[:900]
        basket = str(getattr(doc, "basket_type", "") or "")
        relevance = 0.8 if basket in {"direct_status", "markets_news"} else 0.6
        primary = _is_primary_source_url(url, question_domains)
        source_name = _source_name_from_url(url)
        evidence_items.append(
            GateEvidenceItem(
                content=content or title,
                source_url=url,
                source_name=source_name,
                retrieved_at=now,
                published_at=None,
                relevance_score=relevance,
                is_primary_source=primary,
            )
        )
        source_rows.append(
            {
                "url": url,
                "title": title[:120],
                "date": "",
                "quality": "High" if primary else "Medium",
                "snippet": (content or title)[:260],
            }
        )
        chunks.append(
            f"[{len(evidence_items)}] {source_name} | {url} | retrieved={now.date().isoformat()} | primary={primary}\n"
            f"{content or title}"
        )
    bundle = "\n\n".join(chunks)
    if not bundle:
        bundle = "No evidence retrieved from search."
    return bundle[:max_chars], evidence_items, source_rows


def _parse_search_calls(text: str, max_queries: int) -> list[str]:
    queries = re.findall(r'SEARCH\s*\(\s*["\']([^"\']+)["\']\s*\)', text, flags=re.IGNORECASE)
    if not queries:
        for line in text.splitlines():
            line = line.strip().lstrip("-* ").strip()
            if not line:
                continue
            if len(line) < 12 or len(line) > 140:
                continue
            queries.append(line)
    dedup: list[str] = []
    seen: set[str] = set()
    for q in queries:
        q_clean = re.sub(r"\s+", " ", q).strip()
        key = q_clean.lower()
        if not q_clean or key in seen:
            continue
        seen.add(key)
        dedup.append(q_clean)
        if len(dedup) >= max_queries:
            break
    return dedup


async def _adaptive_search_queries(
    question_text: str,
    resolution_criteria: str,
    today_str: str,
    model: str,
    max_queries: int,
) -> tuple[list[str], str]:
    prompt = clean_indents(
        f"""
        You are planning a single adaptive search pass for a forecasting question.

        QUESTION:
        {question_text}

        RESOLUTION CRITERIA:
        {resolution_criteria}

        TODAY: {today_str}

        Produce 6-12 search calls that maximize decision-relevant signal.
        Prioritize:
        1) Latest status and trend
        2) Exact resolution mechanics/data source
        3) Strongest YES and NO evidence
        4) Relevant base-rate/history
        5) Market/expert aggregate odds if available

        Output only lines in this format:
        SEARCH("query")
        """
    )
    try:
        raw = await call_openrouter_llm(
            prompt=prompt,
            model=model,
            temperature=0.2,
            max_tokens=1200,
            usage_label="search_strategy",
        )
        parsed = _parse_search_calls(raw, max_queries=max_queries)
        if parsed:
            return parsed, raw
        return [question_text[:120]], raw
    except Exception as exc:
        return [question_text[:120]], f"Adaptive search planning failed: {exc}"


async def _red_team_adjust_probability(
    question_title: str,
    resolution_criteria: str,
    deadline_text: str,
    p_calibrated: float,
    best_reasoning: str,
) -> tuple[float, str]:
    critique_model = os.getenv("RED_TEAM_MODEL", "google/gemini-2.5-flash")
    prompt = clean_indents(
        f"""
        You are reviewing a forecast for errors. Be adversarial.

        QUESTION: {question_title}
        RESOLUTION CRITERIA: {resolution_criteria}
        DEADLINE: {deadline_text}
        DRAFT PROBABILITY: {p_calibrated * 100:.1f}%
        REASONING SUMMARY: {best_reasoning[-6000:]}

        Check for these specific failure modes:
        1. TIMELINE CONFUSION
        2. BASE RATE NEGLECT
        3. SINGLE-SOURCE FRAGILITY
        4. RESOLUTION MISREAD
        5. MISSING PATHWAYS

        Then state your verdict:
        ADJUSTMENT: [none / increase by Xpp / decrease by Xpp]
        CONFIDENCE IN ADJUSTMENT: [low / medium / high]
        """
    )
    try:
        critique = await call_openrouter_llm(
            prompt=prompt,
            model=critique_model,
            temperature=0.1,
            max_tokens=1400,
            usage_label="red_team_critique",
            reasoning_effort="low",
        )
        adj_pp = 0.0
        adj_match = re.search(r"ADJUSTMENT\s*:\s*([^\n]+)", critique, re.IGNORECASE)
        conf_match = re.search(
            r"CONFIDENCE IN ADJUSTMENT\s*:\s*(low|medium|high)",
            critique,
            re.IGNORECASE,
        )
        confidence = conf_match.group(1).lower() if conf_match else "low"
        if adj_match:
            adj_text = adj_match.group(1).strip().lower()
            if "increase" in adj_text or "decrease" in adj_text:
                mag = re.search(r"(\d+(?:\.\d+)?)\s*pp", adj_text)
                if mag:
                    amount = float(mag.group(1))
                    adj_pp = amount if "increase" in adj_text else -amount
        updated = p_calibrated
        if abs(adj_pp) > 5 and confidence in {"medium", "high"}:
            updated = max(0.01, min(0.99, p_calibrated + 0.5 * (adj_pp / 100.0)))
        return updated, critique
    except Exception as exc:
        return p_calibrated, f"Red-team critique failed: {exc}"


def _extract_numeric_claim_sentences(reasoning: str, max_items: int = 20) -> list[str]:
    claims: list[str] = []
    month_names = (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
    sentences = re.split(r"(?<=[.!?])\s+", reasoning)
    for sentence in sentences:
        s = sentence.strip()
        if len(s) < 18:
            continue
        if not re.search(r"\b\d[\d,]*(?:\.\d+)?\s*%?\b", s):
            continue
        s_lower = s.lower()
        # Filter obvious date-only claims (they pollute hinge analysis).
        if any(m in s_lower for m in month_names) and re.search(r"\b20\d{2}\b", s_lower):
            if len(re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", s_lower)) <= 2:
                continue
        claims.append(re.sub(r"\s+", " ", s)[:260])
        if len(claims) >= max_items:
            break
    return claims


def _claim_is_evidence_backed(claim: str, evidence_bundle_text: str) -> bool:
    evidence_lower = evidence_bundle_text.lower()
    claim_lower = claim.lower()
    number_tokens = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", claim_lower)
    stop = {
        "will",
        "with",
        "that",
        "this",
        "from",
        "into",
        "than",
        "have",
        "has",
        "been",
        "were",
        "their",
        "there",
        "about",
        "after",
        "before",
        "between",
    }
    keywords = [w for w in re.findall(r"[a-z]{4,}", claim_lower) if w not in stop]
    overlap = sum(1 for w in keywords if w in evidence_lower)
    number_hit = any(tok in evidence_lower for tok in number_tokens) if number_tokens else False
    if number_tokens:
        return number_hit and overlap >= 1
    return overlap >= 2


def _parse_claim_lines(raw: str) -> list[str]:
    claims: list[str] = []
    for line in raw.splitlines():
        t = line.strip().lstrip("-* ").strip()
        if not t:
            continue
        if t.lower().startswith("claim:"):
            t = t.split(":", 1)[1].strip()
        if len(t) < 12:
            continue
        claims.append(re.sub(r"\s+", " ", t)[:260])
    # preserve order, dedup
    dedup: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        k = claim.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(claim)
    return dedup[:8]


async def _extract_hinge_claims(
    question_title: str,
    resolution_criteria: str,
    reasoning: str,
) -> list[str]:
    prompt = clean_indents(
        f"""
        Extract the most important factual hinge claims from this forecast reasoning.
        A hinge claim is a factual assertion that, if false, would materially change the forecast.

        QUESTION: {question_title}
        RESOLUTION CRITERIA: {resolution_criteria}
        REASONING:
        {reasoning[-6000:]}

        Output 5-8 lines, one claim per line:
        CLAIM: <claim>
        """
    )
    try:
        raw = await call_openrouter_llm(
            prompt=prompt,
            model=os.getenv("CLAIM_AUDIT_MODEL", "google/gemini-2.5-flash"),
            temperature=0.1,
            max_tokens=1000,
            usage_label="claim_audit_hinges",
            reasoning_effort="low",
        )
        parsed = _parse_claim_lines(raw)
        if parsed:
            return parsed
    except Exception:
        pass
    return []


async def _claim_audit(
    question_title: str,
    resolution_criteria: str,
    reasoning: str,
    evidence_bundle_text: str,
) -> dict[str, Any]:
    numeric_claims = _extract_numeric_claim_sentences(reasoning, max_items=24)
    hinge_claim_text = await _extract_hinge_claims(
        question_title=question_title,
        resolution_criteria=resolution_criteria,
        reasoning=reasoning,
    )
    if not hinge_claim_text:
        hinge_claim_text = numeric_claims[:8]

    items: list[dict[str, str]] = []
    for claim in numeric_claims:
        status = "evidence_backed" if _claim_is_evidence_backed(claim, evidence_bundle_text) else "assumption"
        items.append({"claim": claim, "status": status, "type": "numeric"})

    hinge_claims: list[dict[str, str]] = []
    for claim in hinge_claim_text[:8]:
        status = "evidence_backed" if _claim_is_evidence_backed(claim, evidence_bundle_text) else "assumption"
        hinge_claims.append({"claim": claim, "status": status, "type": "hinge"})
        if not any(claim.lower() == it["claim"].lower() for it in items):
            items.append({"claim": claim, "status": status, "type": "hinge"})

    evidence_backed = sum(1 for it in items if it["status"] == "evidence_backed")
    hinge_assumptions = sum(1 for it in hinge_claims if it["status"] == "assumption")
    assumptions_note = (
        "Assumptions-heavy rationale: >50% hinge claims are assumptions."
        if hinge_claims and (hinge_assumptions / len(hinge_claims) > 0.5)
        else ""
    )
    return {
        "total_claims": len(items),
        "evidence_backed": evidence_backed,
        "assumption": len(items) - evidence_backed,
        "hinge_claims": hinge_claims,
        "assumptions_note": assumptions_note,
        "items": items,
    }


def _parse_signposts(text: str) -> list[dict[str, str]]:
    blocks = re.split(r"\n(?=SIGNPOST:)", text.strip(), flags=re.IGNORECASE)
    out: list[dict[str, str]] = []
    for block in blocks:
        event = ""
        direction = ""
        magnitude = ""
        for line in block.splitlines():
            line = line.strip()
            if line.lower().startswith("signpost:"):
                event = line.split(":", 1)[1].strip()
            elif line.lower().startswith("direction:"):
                direction = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("magnitude:"):
                magnitude = line.split(":", 1)[1].strip()
        if event:
            out.append(
                {
                    "event": event,
                    "direction": direction or "unknown",
                    "magnitude": magnitude or "unknown",
                }
            )
    return out[:5]


def _fallback_signposts(
    question_title: str,
    resolution_criteria: str,
    count: int,
) -> list[dict[str, str]]:
    templates = [
        {
            "event": (
                f"Primary resolution source for '{question_title}' publishes a materially stronger value "
                "than current baseline."
            ),
            "direction": "up",
            "magnitude": "+15pp",
        },
        {
            "event": (
                f"Primary resolution source for '{question_title}' publishes a materially weaker value "
                "than current baseline."
            ),
            "direction": "down",
            "magnitude": "-15pp",
        },
        {
            "event": (
                "Official publication is delayed, revised unexpectedly, or contradicts proxy indicators "
                "currently used in reasoning."
            ),
            "direction": "down",
            "magnitude": "-10pp",
        },
        {
            "event": (
                f"A high-confidence official preview directly aligned with the resolution criteria appears: "
                f"{resolution_criteria[:120]}"
            ),
            "direction": "up",
            "magnitude": "+12pp",
        },
    ]
    return templates[: max(0, count)]


async def _extract_signposts(
    question_title: str,
    resolution_criteria: str,
    final_probability: float,
    reasoning: str,
) -> dict[str, Any]:
    prompt = clean_indents(
        f"""
        Given this forecast and its reasoning, list 3-5 specific, observable events
        that would cause a probability update greater than 10 percentage points.

        QUESTION: {question_title}
        RESOLUTION CRITERIA: {resolution_criteria}
        CURRENT FORECAST: {final_probability * 100:.1f}%
        REASONING SUMMARY:
        {reasoning[-5000:]}

        Format exactly:
        SIGNPOST: [event description]
        DIRECTION: [up/down]
        MAGNITUDE: [+15pp / -20pp]
        """
    )
    try:
        raw = await call_openrouter_llm(
            prompt=prompt,
            model=os.getenv("SIGNPOST_MODEL", "google/gemini-2.5-flash"),
            temperature=0.2,
            max_tokens=1000,
            usage_label="signpost_extract",
            reasoning_effort="low",
        )
        parsed = _parse_signposts(raw)
        if len(parsed) < 3:
            repair_prompt = clean_indents(
                f"""
                Rewrite these signposts into 3-5 concrete, observable events.
                Keep format exactly as:
                SIGNPOST: ...
                DIRECTION: up/down
                MAGNITUDE: +15pp or -20pp

                QUESTION: {question_title}
                RESOLUTION CRITERIA: {resolution_criteria}
                EXISTING DRAFT:
                {raw}
                """
            )
            try:
                repaired = await call_openrouter_llm(
                    prompt=repair_prompt,
                    model=os.getenv("SIGNPOST_MODEL", "google/gemini-2.5-flash"),
                    temperature=0.1,
                    max_tokens=900,
                    usage_label="signpost_extract_repair",
                    reasoning_effort="low",
                )
                repaired_parsed = _parse_signposts(repaired)
                if len(repaired_parsed) > len(parsed):
                    raw = repaired
                    parsed = repaired_parsed
            except Exception:
                pass
        if len(parsed) < 3:
            parsed.extend(
                _fallback_signposts(
                    question_title=question_title,
                    resolution_criteria=resolution_criteria,
                    count=3 - len(parsed),
                )
            )
        return {"raw": raw, "signposts": parsed}
    except Exception as exc:
        return {"raw": f"signpost extraction failed: {exc}", "signposts": []}


def _anchor_signposts(
    signposts: list[dict[str, str]],
    final_probability: float,
) -> list[dict[str, str]]:
    """Re-anchor signpost magnitudes to final_probability, clipping to [1%, 99%]."""
    anchored: list[dict[str, str]] = []
    for s in signposts:
        event = s.get("event", "")
        direction = s.get("direction", "unknown")
        magnitude_str = s.get("magnitude", "unknown")

        mag_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*pp", magnitude_str, re.IGNORECASE)
        if mag_match:
            mag_pp = float(mag_match.group(1))
            adjusted = final_probability + (mag_pp / 100.0)
            adjusted_clipped = max(0.01, min(0.99, adjusted))
            effective_mag_pp = (adjusted_clipped - final_probability) * 100.0
            # Truncate toward zero so rounding doesn't overshoot the clip
            import math
            if effective_mag_pp >= 0:
                truncated = math.floor(effective_mag_pp)
                new_magnitude = f"+{truncated}pp"
            else:
                truncated = math.ceil(effective_mag_pp)
                new_magnitude = f"{truncated}pp"
            anchored.append({
                "event": event,
                "direction": direction,
                "magnitude": new_magnitude,
            })
        else:
            anchored.append(s)
    return anchored


def _build_structured_report(
    *,
    timestamp: str,
    question_title: str,
    q_type: str,
    canonical_spec_text: str,
    today_str: str,
    planner_text: str,
    planned_queries: list[str],
    search_queries: list[str],
    gate_evidence: list[GateEvidenceItem],
    top_evidence_rows: list[dict[str, Any]],
    evidence_bundle_text: str,
    settled: list[dict[str, Any]],
    aggregated: Any,
    final_probability: float,
    parsing_failure_rate: float,
    gate_report: Any,
    red_team_changed: bool,
    critique_artifact: str,
    claim_audit: dict[str, Any],
    signposts: list[dict[str, str]],
    signal_strength: float,
    informativeness: str,
    final_mc_probabilities: dict[str, float] | None = None,
    final_numeric_percentiles: dict[int, float] | None = None,
    numeric_unit: str = "",
) -> str:
    """Build a structured markdown forecast report with clear sections."""
    sections: list[str] = []

    # === 1. HEADER ===
    sections.append(
        f"# Forecast Report -- {timestamp}\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| Question | {question_title} |\n"
        f"| Type | {q_type} |\n"
        f"| Date | {today_str} |\n"
        f"| Final Prediction | {final_probability:.1%} |\n"
        f"| Informativeness | {informativeness} (signal={signal_strength:.3f}) |\n"
        f"| Confidence | {aggregated.confidence_class} |\n"
        f"| Gate Action | {gate_report.action} (score={gate_report.gate_score:.3f}) |\n\n"
        f"**Canonical Spec:**\n{canonical_spec_text}"
    )

    # === 2. SEARCH PROCESS ===
    sections.append(
        f"## Search Process\n\n"
        f"**Strategy Output:**\n```\n{planner_text}\n```\n\n"
        f"**Planned Queries ({len(planned_queries)}):**\n"
        + "\n".join(f"- {q}" for q in planned_queries) + "\n\n"
        f"**Executed Queries ({len(search_queries)}):**\n"
        + "\n".join(f"- {q}" for q in search_queries)
    )

    # === 3. EVIDENCE SUMMARY ===
    ev_header = "| # | Source | Primary | Relevance | Bearing | Snippet |\n|---|--------|---------|-----------|---------|---------|"
    ev_rows = []
    for row in top_evidence_rows:
        snippet_short = row.get("snippet", "")[:80].replace("|", "/")
        ev_rows.append(
            f"| {row['evidence_id']} | {row['source']} | "
            f"{'Yes' if row['primary'] else 'No'} | {row['relevance']:.2f} | "
            f"{row['bearing']} | {snippet_short}... |"
        )
    sections.append(
        f"## Evidence Summary\n\n"
        f"Items: {gate_report.evidence_count} | "
        f"Distinct sources: {gate_report.distinct_sources} | "
        f"Primary: {gate_report.primary_sources} | "
        f"Mean relevance: {gate_report.mean_relevance:.3f} | "
        f"Freshness: {gate_report.freshness_days:.1f} days\n\n"
        + ev_header + "\n" + "\n".join(ev_rows)
    )

    # === 4. MODEL RUNS ===
    run_header = "| Model | Probability | Tokens | Warnings |\n|-------|-------------|--------|----------|"
    run_rows = []
    for row in settled:
        result = row["result"]
        run_rows.append(
            f"| {row['config']['label']} | {result.probability:.1%} | "
            f"{result.token_usage.get('total', 0):,} | {len(result.warnings)} |"
        )
    sections.append(
        f"## Model Runs\n\n"
        f"Parsing failure rate: {parsing_failure_rate:.1%}\n\n"
        + run_header + "\n" + "\n".join(run_rows)
    )
    for row in settled:
        label = row["config"]["label"]
        result = row["result"]
        reasoning_preview = result.explanation[:3000]
        if len(result.explanation) > 3000:
            reasoning_preview += f"\n\n... [{len(result.explanation) - 3000} chars truncated]"
        sections.append(
            f"### {label} ({result.probability:.1%})\n\n"
            f"<details><summary>Full reasoning ({len(result.explanation):,} chars)</summary>\n\n"
            f"{reasoning_preview}\n\n</details>"
        )

    # === 5. AGGREGATION ===
    agg_text = (
        f"## Aggregation\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Raw trimmed mean | {aggregated.p_raw:.4f} ({aggregated.p_raw:.1%}) |\n"
        f"| Calibrated (extremized) | {aggregated.p_calibrated:.4f} ({aggregated.p_calibrated:.1%}) |\n"
        f"| Final probability | {final_probability:.4f} ({final_probability:.1%}) |\n"
        f"| Dispersion | {aggregated.dispersion:.4f} |\n"
        f"| N runs | {aggregated.n_runs} (trimmed: {aggregated.n_trimmed}) |"
    )
    if final_mc_probabilities:
        agg_text += "\n\n**MC Probabilities:**\n"
        for opt, p in sorted(final_mc_probabilities.items()):
            agg_text += f"- {opt}: {p:.1%}\n"
    if final_numeric_percentiles:
        unit_text = f" {numeric_unit}" if numeric_unit else ""
        agg_text += "\n\n**Numeric Percentiles:**\n"
        for pctl in [10, 25, 50, 75, 90]:
            agg_text += f"- P{pctl}: {final_numeric_percentiles[pctl]:.4f}{unit_text}\n"
    sections.append(agg_text)

    # === 6. GATE DECISION ===
    gate_text = (
        f"## Gate Decision\n\n"
        f"| Component | Score |\n"
        f"|-----------|-------|\n"
        f"| Spec confidence | {gate_report.spec_confidence:.3f} |\n"
        f"| Temporal relevance | {gate_report.temporal_relevance:.3f} |\n"
        f"| Evidence sufficiency | {gate_report.evidence_sufficiency:.3f} |\n"
        f"| Model agreement | {gate_report.model_agreement:.3f} |\n"
        f"| **Weighted gate score** | **{gate_report.gate_score:.3f}** |\n\n"
        f"**Action:** {gate_report.action}\n\n"
        f"**Reasons:**\n"
    )
    for reason in gate_report.reasons:
        gate_text += f"- {reason}\n"
    sections.append(gate_text)

    # === 7. RED TEAM ===
    sections.append(
        f"## Red Team\n\n"
        f"Status: {'adjusted' if red_team_changed else 'no change'}\n\n"
        f"<details><summary>Critique</summary>\n\n{critique_artifact}\n\n</details>"
    )

    # === 8. CLAIM AUDIT ===
    ca = claim_audit
    hinge_claims = ca.get("hinge_claims", [])
    backed = ca.get("evidence_backed", 0)
    assumed = ca.get("assumption", 0)
    note = ca.get("assumptions_note", "")
    audit_text = (
        f"## Claim Audit\n\n"
        f"Total claims: {ca.get('total_claims', 0)} | "
        f"Evidence-backed: {backed} | Assumptions: {assumed}"
    )
    if note:
        audit_text += f"\n\n**{note}**"
    if hinge_claims:
        audit_text += "\n\n**Hinge Claims:**\n"
        for hc in hinge_claims:
            tag = "evidence" if hc["status"] == "evidence_backed" else "ASSUMPTION"
            audit_text += f"- [{tag}] {hc['claim']}\n"
    sections.append(audit_text)

    # === 9. SIGNPOSTS ===
    if signposts:
        sp_text = (
            "## Signposts\n\n"
            "| Event | Direction | Magnitude |\n"
            "|-------|-----------|----------|"
        )
        for s in signposts[:5]:
            sp_text += f"\n| {s.get('event', '')} | {s.get('direction', '')} | {s.get('magnitude', '')} |"
        sections.append(sp_text)

    # === 10. FINAL SUMMARY ===
    sections.append(
        f"## Final Summary\n\n"
        f"**Prediction:** {final_probability:.1%}\n"
        f"**Confidence class:** {aggregated.confidence_class}\n"
        f"**Informativeness:** {informativeness}\n"
        f"**Gate action:** {gate_report.action}"
    )

    return "\n\n---\n\n".join(sections) + "\n"


def _trimmed_mean(values: list[float], trim_fraction: float = 0.2) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    trim_count = max(1, int(n * trim_fraction))
    if n < 4:
        trim_count = 0
    kept = sorted_vals[trim_count : n - trim_count] if trim_count > 0 else sorted_vals
    return sum(kept) / len(kept)


def _extremize(p: float, k: float = 1.73) -> float:
    if p <= 0.001 or p >= 0.999:
        return max(0.001, min(0.999, p))
    import math

    log_odds = math.log(p / (1.0 - p))
    out = 1.0 / (1.0 + math.exp(-k * log_odds))
    return max(0.001, min(0.999, out))


def _norm_option_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _infer_numeric_unit(
    question_title: str,
    resolution_criteria: str,
    explicit_unit: str,
) -> str:
    if explicit_unit.strip():
        return explicit_unit.strip()
    blob = f"{question_title}\n{resolution_criteria}".lower()
    if "%" in blob or " percent" in blob or "percentage" in blob:
        return "%"
    if "basis point" in blob or "bps" in blob:
        return "bps"
    if "$" in blob or " usd" in blob or "dollar" in blob:
        return "USD"
    if "celsius" in blob or "°c" in blob:
        return "C"
    if "fahrenheit" in blob or "°f" in blob:
        return "F"
    if "million" in blob:
        return "million"
    if "billion" in blob:
        return "billion"
    return ""


def _validate_numeric_percentiles(
    parsed: dict[int, float],
    response_text: str,
    expected_unit: str,
    lower_bound: float | None,
    upper_bound: float | None,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if not parsed:
        return False, ["Numeric percentiles missing."]

    values = [float(parsed[p]) for p in [10, 25, 50, 75, 90] if p in parsed]
    if len(values) < 3:
        return False, ["Too few numeric percentiles parsed."]
    for idx in range(1, len(values)):
        if values[idx] < values[idx - 1]:
            warnings.append("Parsed numeric percentiles were non-monotonic before projection.")
            break

    if lower_bound is not None and upper_bound is not None and upper_bound > lower_bound:
        span = max(1e-9, upper_bound - lower_bound)
        severe_outlier = any(v < lower_bound - 10 * span or v > upper_bound + 10 * span for v in values)
        if severe_outlier:
            return False, ["Numeric values appear outside expected order-of-magnitude (possible unit mismatch)."]
        if upper_bound <= 1.5 and any(v > 2.0 for v in values):
            return False, ["Numeric values look like percent units while bounds imply decimals."]

    unit = expected_unit.strip().lower()
    text = response_text.lower()
    if unit == "%" and ("%" not in response_text and "percent" not in text):
        warnings.append("Expected percentage units but model did not explicitly label output as percent.")
    elif unit == "usd" and ("$" not in response_text and "usd" not in text and "dollar" not in text):
        warnings.append("Expected USD units but output did not include an explicit currency marker.")
    elif unit == "bps" and ("bps" not in text and "basis point" not in text):
        warnings.append("Expected basis points but output did not explicitly mention bps.")

    return True, warnings


def _extract_mc_probabilities(text: str, options: list[str]) -> dict[str, float] | None:
    if not options:
        return None
    option_keys = {_norm_option_key(opt): opt for opt in options}
    out: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("*").strip()
        if not line or ":" not in line:
            continue
        left, right = line.split(":", 1)
        pct_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", right)
        if not pct_match:
            continue
        try:
            p = float(pct_match.group(1)) / 100.0
        except Exception:
            continue
        lk = _norm_option_key(left)
        # direct
        if lk in option_keys:
            out[option_keys[lk]] = p
            continue
        # fuzzy contains
        matched_opt = None
        for opt in options:
            ok = _norm_option_key(opt)
            if ok and (ok in lk or lk in ok):
                matched_opt = opt
                break
        if matched_opt:
            out[matched_opt] = p
    if not out:
        return None
    filled = dict(out)
    missing = [opt for opt in options if opt not in filled]
    if missing:
        remaining = max(0.0, 1.0 - sum(max(0.0, v) for v in filled.values()))
        share = remaining / len(missing) if missing else 0.0
        for opt in missing:
            filled[opt] = share
    return enforce_sum_to_one(filled)


def _extract_numeric_percentiles(text: str) -> dict[int, float] | None:
    target = [10, 25, 50, 75, 90]
    out: dict[int, float] = {}
    for p in target:
        pat = rf"\b{p}(?:th)?\s*percentile[^0-9\-]*(-?\d[\d,]*(?:\.\d+)?)"
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        try:
            out[p] = float(m.group(1).replace(",", ""))
        except Exception:
            continue
    if len(out) < 3:
        return None
    # fill missing with nearest known percentile
    for p in target:
        if p in out:
            continue
        lower = [x for x in target if x < p and x in out]
        upper = [x for x in target if x > p and x in out]
        if lower and upper:
            lo = max(lower)
            hi = min(upper)
            frac = (p - lo) / (hi - lo)
            out[p] = out[lo] + frac * (out[hi] - out[lo])
        elif lower:
            out[p] = out[max(lower)]
        elif upper:
            out[p] = out[min(upper)]
    return out


def _aggregate_mc_runs(
    per_run: list[dict[str, float]],
    options: list[str],
    trim_fraction: float,
    extremize_k: float,
) -> dict[str, float]:
    agg: dict[str, float] = {}
    for opt in options:
        vals = [max(0.0, min(1.0, run.get(opt, 0.0))) for run in per_run]
        mean_p = _trimmed_mean(vals, trim_fraction=trim_fraction)
        agg[opt] = _extremize(mean_p, k=extremize_k)
    return enforce_sum_to_one(agg)


def _aggregate_numeric_runs(
    per_run: list[dict[int, float]],
    trim_fraction: float,
    lower_bound: float | None,
    upper_bound: float | None,
) -> dict[int, float]:
    target = [10, 25, 50, 75, 90]
    out: dict[int, float] = {}
    for p in target:
        vals = [run[p] for run in per_run if p in run]
        out[p] = _trimmed_mean(vals, trim_fraction=trim_fraction) if vals else 0.0
    # enforce monotonic non-decreasing percentiles
    prev = None
    for p in target:
        v = out[p]
        if prev is not None and v < prev:
            v = prev
        out[p] = v
        prev = v
    if lower_bound is not None:
        for p in target:
            out[p] = max(lower_bound, out[p])
    if upper_bound is not None:
        for p in target:
            out[p] = min(upper_bound, out[p])
    # monotonic again after clamping
    prev = None
    for p in target:
        v = out[p]
        if prev is not None and v < prev:
            v = prev
        out[p] = v
        prev = v
    return out


async def run_lean_ensemble_forecast(
    question: str,
    models: list[dict] | None = None,
    publish_to_metaculus: bool = False,
    community_prior: float | None = None,
    use_react: bool = True,
    feature_flags: Optional[dict[str, Any]] = None,
    outlier_threshold_pp: float = 15.0,
    question_type: str = "binary",
    options: list[str] | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    unit: str | None = None,
    extract_probability_fn: Callable[[str], float] | None = None,
    canonical_spec_extractor: Callable[[str], Awaitable[Any]] | None = None,
    canonical_spec_formatter: Callable[[Any, str], str] | None = None,
    spec_consistency_checker: Callable[[str, str], Awaitable[tuple[str, str]]] | None = None,
) -> dict:
    del outlier_threshold_pp
    del use_react

    if extract_probability_fn is None:
        raise ValueError("extract_probability_fn is required")

    def _flag(name: str, default: bool = True) -> bool:
        if not feature_flags:
            return default
        return bool(feature_flags.get(name, default))

    reset_token_usage()
    original_question_input = question.strip()
    q_type = (question_type or "binary").strip().lower()
    if q_type not in {"binary", "multiple_choice", "mc", "numeric"}:
        q_type = "binary"
    if q_type == "mc":
        q_type = "multiple_choice"
    mc_options = [str(o).strip() for o in (options or []) if str(o).strip()]
    numeric_unit = (unit or "").strip()
    question_url_for_post: Optional[str] = None
    question_title = original_question_input
    resolution_criteria = ""
    fine_print = ""
    background_info = ""

    if original_question_input.lower().startswith("http") and "metaculus.com/questions/" in original_question_input:
        question_url_for_post = original_question_input
        try:
            meta_q = MetaculusApi.get_question_by_url(question_url_for_post)
            if meta_q is not None:
                question_title = str(getattr(meta_q, "question_text", "") or question_title)
                resolution_criteria = str(getattr(meta_q, "resolution_criteria", "") or "")
                fine_print = str(getattr(meta_q, "fine_print", "") or "")
                background_info = str(getattr(meta_q, "background_info", "") or "")
                cp = getattr(meta_q, "community_prediction_at_access_time", None)
                if community_prior is None and isinstance(cp, (float, int)) and 0 <= cp <= 1:
                    community_prior = float(cp)
                if q_type == "multiple_choice":
                    meta_opts = getattr(meta_q, "options", None)
                    if isinstance(meta_opts, (list, tuple)):
                        extracted = [str(o).strip() for o in meta_opts if str(o).strip()]
                        if extracted:
                            mc_options = extracted
                if q_type == "numeric":
                    lb = getattr(meta_q, "lower_bound", None)
                    ub = getattr(meta_q, "upper_bound", None)
                    if isinstance(lb, (float, int)):
                        lower_bound = float(lb)
                    if isinstance(ub, (float, int)):
                        upper_bound = float(ub)
        except Exception as exc:
            logger.warning(f"Could not enrich question from Metaculus URL: {exc}")

    if q_type == "numeric":
        numeric_unit = _infer_numeric_unit(
            question_title=question_title,
            resolution_criteria=(resolution_criteria or fine_print or background_info),
            explicit_unit=numeric_unit,
        )

    if models is None:
        models = _default_forecast_model_families()
    expanded_models = _expand_model_roster(models)
    if not expanded_models:
        raise ValueError("No forecast models configured.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    logs_dir = "forecasts"
    os.makedirs(logs_dir, exist_ok=True)

    spec_input = "\n\n".join([question_title, resolution_criteria, fine_print]).strip()
    canonical_spec = None
    canonical_spec_text = f"- One-line: {question_title}"
    if _flag("spec_lock", True) and canonical_spec_extractor is not None and canonical_spec_formatter is not None:
        try:
            canonical_spec = await canonical_spec_extractor(spec_input)
            canonical_spec_text = canonical_spec_formatter(canonical_spec, question_title)
        except Exception as exc:
            logger.warning(f"Canonical spec extraction failed: {exc}")
    spec_confidence = 0.9 if canonical_spec is not None else 0.35

    search_model = os.getenv("SEARCH_PLANNER_MODEL", "google/gemini-2.5-flash")
    planner_queries, planner_text = await _adaptive_search_queries(
        question_text=question_title,
        resolution_criteria=(resolution_criteria or canonical_spec_text),
        today_str=today_str,
        model=search_model,
        max_queries=14,
    )
    forced_resolution_query = (
        f"{question_title[:120]} how this question resolves official source announcement "
        "resolution criteria data source"
    )
    planned_queries: list[str] = list(planner_queries)
    seen: set[str] = set()
    search_queries: list[str] = []
    for q in planner_queries + [forced_resolution_query]:
        q_clean = q.strip()
        q_key = q_clean.lower()
        if not q_clean or q_key in seen:
            continue
        seen.add(q_key)
        search_queries.append(q_clean)
    max_queries = int(os.getenv("FORECAST_MAX_SEARCH_QUERIES", "10"))
    search_queries = search_queries[:max_queries]

    # Primary source targeting: add site-scoped queries for canonical domains
    primary_domains = _identify_primary_source_urls(question_title, resolution_criteria)
    for domain in primary_domains[:2]:
        targeted_q = f"site:{domain} {question_title[:80]}"
        if targeted_q.lower() not in seen:
            search_queries.append(targeted_q)
            seen.add(targeted_q.lower())

    docs: list[Any] = []
    exa_key = os.getenv("EXA_API_KEY")
    if exa_key:
        exa_client = ExaClient(api_key=exa_key)
        docs, _ = await multi_provider_search(
            queries=search_queries,
            exa_client=exa_client,
            use_sonar=False,
        )

    # LLM evidence extraction and scoring
    enriched_evidence = await _extract_and_score_evidence(
        docs=docs,
        question_title=question_title,
        resolution_criteria=resolution_criteria,
        max_docs=12,
    )
    for ev in enriched_evidence:
        ev.is_primary_source = _is_primary_source_url(ev.url, primary_domains)

    if enriched_evidence:
        evidence_bundle_text, gate_evidence, shared_sources = _build_evidence_bundle_from_enriched(
            enriched=enriched_evidence,
        )
    else:
        # Fallback to raw docs if extraction produced nothing
        evidence_bundle_text, gate_evidence, shared_sources = _build_evidence_bundle_from_docs(
            docs, question_domains=primary_domains,
        )

    if not gate_evidence:
        gate_evidence = [
            GateEvidenceItem(
                content=(resolution_criteria or question_title)[:600],
                source_url="local://question",
                source_name="question_text",
                retrieved_at=datetime.now(timezone.utc),
                published_at=None,
                relevance_score=0.4,
                is_primary_source=True,
            )
        ]
        shared_sources = [
            {
                "url": "local://question",
                "title": "Question text",
                "date": today_str,
                "quality": "Medium",
                "snippet": (resolution_criteria or question_title)[:260],
            }
        ]
    # Evidence score: mean relevance (not count-based)
    evidence_score = (
        sum(ev.relevance_score for ev in gate_evidence) / len(gate_evidence)
        if gate_evidence else 0.0
    )

    prompt = LEAN_BINARY_FORECAST_PROMPT.format(
        question_title=question_title,
        background_info=(background_info or "No additional background provided."),
        resolution_criteria=(resolution_criteria or canonical_spec_text),
        fine_print=fine_print or "",
        evidence_bundle=evidence_bundle_text,
        today=today_str,
    )
    if q_type == "multiple_choice":
        if not mc_options:
            mc_options = ["Option A", "Option B"]
        options_list = "\n".join([f"- {opt}" for opt in mc_options])
        prompt = (
            prompt
            + "\n\n"
            + LEAN_MC_APPEND_PROMPT.format(options_list=options_list)
        )
    elif q_type == "numeric":
        prompt = (
            prompt
            + "\n\n"
            + LEAN_NUMERIC_APPEND_PROMPT.format(
                lower_bound=lower_bound if lower_bound is not None else "unspecified",
                upper_bound=upper_bound if upper_bound is not None else "unspecified",
            )
        )
        if numeric_unit:
            prompt += f"\n\nRequired unit: {numeric_unit}\n"

    async def _run_one(config: dict[str, Any]) -> dict[str, Any]:
        usage_label = (
            f"forecast_run:{timestamp}:{config['family_label'].replace(' ', '_').lower()}:"
            f"{config['run_index']}"
        )
        try:
            response = await call_openrouter_llm(
                prompt=prompt,
                model=config["name"],
                temperature=float(config.get("temperature", 0.35)),
                max_tokens=int(config.get("max_tokens", 12000)),
                usage_label=usage_label,
                reasoning_effort=config.get("reasoning_effort"),
            )
            warnings: list[str] = []
            risk_flags: list[str] = []
            probability = 0.5
            mc_probs: dict[str, float] | None = None
            numeric_pct: dict[int, float] | None = None
            parse_success = True
            if q_type == "binary":
                probability = max(0.01, min(0.99, extract_probability_fn(response)))
            elif q_type == "multiple_choice":
                mc_probs = _extract_mc_probabilities(response, mc_options)
                if mc_probs is None:
                    parse_success = False
                    probability = 0.5
                else:
                    probability = max(mc_probs.values()) if mc_probs else 0.5
            else:
                numeric_pct = _extract_numeric_percentiles(response)
                if numeric_pct is None:
                    parse_success = False
                    probability = 0.5
                else:
                    valid_units, unit_warnings = _validate_numeric_percentiles(
                        parsed=numeric_pct,
                        response_text=response,
                        expected_unit=numeric_unit,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )
                    warnings.extend(unit_warnings)
                    if not valid_units:
                        parse_success = False
                        risk_flags.append("NUMERIC_UNIT_MISMATCH")
                        probability = 0.5
                    else:
                        mid = numeric_pct.get(50, 0.0)
                        if lower_bound is not None and upper_bound is not None and upper_bound > lower_bound:
                            probability = max(0.01, min(0.99, (mid - lower_bound) / (upper_bound - lower_bound)))
                        else:
                            probability = 0.5
            usage = get_token_usage().get(usage_label, {"prompt": 0, "completion": 0, "total": 0})
            spec_status = "NOT_CHECKED"
            spec_reason = ""
            if (
                _flag("spec_lock", True)
                and spec_consistency_checker is not None
                and canonical_spec_text
            ):
                spec_status, spec_reason = await spec_consistency_checker(
                    canonical_spec_text=canonical_spec_text,
                    model_answer=response,
                )
                if spec_status != "OK":
                    warnings.append(f"Spec consistency {spec_status}: {spec_reason}")
                if spec_status == "MAJOR_DRIFT":
                    risk_flags.append("SPEC_MAJORDRIFT")
            result = LeanRunOutput(
                probability=probability,
                explanation=response,
                search_count=len(search_queries),
                sources=[dict(s) for s in shared_sources],
                token_usage={
                    "prompt": int(usage.get("prompt", 0)),
                    "completion": int(usage.get("completion", 0)),
                    "total": int(usage.get("total", 0)),
                },
                warnings=warnings,
                risk_flags=risk_flags,
                diagnostics={
                    "mode": "lean_forecast_prompt",
                    "usage_label": usage_label,
                    "spec_status": spec_status,
                    "spec_reason": spec_reason,
                    "question_type": q_type,
                    "parse_success": parse_success,
                    "parsed_mc_probabilities": mc_probs,
                    "parsed_numeric_percentiles": numeric_pct,
                    "numeric_unit_expected": numeric_unit,
                },
            )
            run = ForecastRun(
                model=config["name"],
                probability=probability,
                reasoning=response,
                token_usage=int(usage.get("total", 0)),
            )
            return {
                "config": {
                    "name": config["name"],
                    "label": config["label"],
                    "max_tokens": config["max_tokens"],
                },
                "result": result,
                "forecast_run": run,
            }
        except Exception as exc:
            failed = LeanRunOutput(
                probability=0.5,
                explanation=f"Run failed: {exc}",
                search_count=len(search_queries),
                sources=[dict(s) for s in shared_sources],
                token_usage={"prompt": 0, "completion": 0, "total": 0},
                warnings=[f"Run failed: {exc}"],
                risk_flags=["RUN_ERROR"],
                diagnostics={"mode": "lean_forecast_prompt", "error": str(exc), "question_type": q_type, "parse_success": False},
            )
            return {
                "config": {
                    "name": config["name"],
                    "label": config["label"],
                    "max_tokens": config["max_tokens"],
                },
                "result": failed,
                "forecast_run": None,
                "error": str(exc),
            }

    settled = await asyncio.gather(*[_run_one(cfg) for cfg in expanded_models])
    successful_runs = [row["forecast_run"] for row in settled if row.get("forecast_run") is not None]

    if not successful_runs:
        fallback_prob = float(community_prior) if community_prior is not None else 0.5
        summary_text = (
            f"All forecast runs failed. Fallback probability: {fallback_prob:.1%} "
            f"(community_prior={community_prior if community_prior is not None else 'none'})."
        )
        full_log = (
            f"# FORECAST LOG - {timestamp}\n\n"
            f"Question: {question_title}\n\n"
            f"Search queries ({len(search_queries)}):\n- " + "\n- ".join(search_queries) + "\n\n"
            f"{summary_text}\n"
        )
        safe_q = "".join(c for c in question_title[:50] if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        with open(f"{logs_dir}/forecast_{timestamp}_{safe_q}.txt", "w", encoding="utf-8") as f:
            f.write(full_log)
        return {
            "final_probability": fallback_prob,
            "summary_text": summary_text,
            "full_reasoning": summary_text,
            "full_log": full_log,
            "individual_results": settled,
            "publish_action": "abstain",
            "feature_flags": {
                "spec_lock": _flag("spec_lock", True),
                "evidence_ledger": _flag("evidence_ledger", True),
                "numeric_provenance": _flag("numeric_provenance", True),
                "market_snapshot": _flag("market_snapshot", True),
                "outlier_xexam": _flag("outlier_xexam", True),
            },
        }

    parse_failures = sum(
        1 for row in settled if not bool((row.get("result").diagnostics or {}).get("parse_success", False))
    )
    parsing_failure_rate = parse_failures / max(len(settled), 1)
    trim_fraction = float(os.getenv("FORECAST_TRIM_FRACTION", "0.2"))
    extremize_k = float(os.getenv("FORECAST_EXTREMIZE_K", "1.73"))

    aggregated = aggregate_forecasts(
        runs=successful_runs,
        extremize_k=extremize_k,
        trim_fraction=trim_fraction,
        evidence_score=evidence_score,
    )

    best_reasoning = sorted(successful_runs, key=lambda r: abs(r.probability - aggregated.p_raw))[0].reasoning
    question_deadline = _parse_deadline(
        getattr(canonical_spec, "time_window", "") if canonical_spec is not None else "",
        resolution_criteria,
        fine_print,
        question_title,
    )
    deadline_text = question_deadline.date().isoformat() if question_deadline else "Unknown"
    critique_artifact = "Red-team critique skipped for non-binary question type."
    p_after_critique = aggregated.p_calibrated
    if q_type == "binary":
        p_after_critique, critique_artifact = await _red_team_adjust_probability(
            question_title=question_title,
            resolution_criteria=(resolution_criteria or canonical_spec_text),
            deadline_text=deadline_text,
            p_calibrated=aggregated.p_calibrated,
            best_reasoning=best_reasoning,
        )
    aggregated_for_gate = AggregatedForecast(
        p_raw=aggregated.p_raw,
        p_calibrated=p_after_critique,
        dispersion=aggregated.dispersion,
        confidence_class=aggregated.confidence_class,
        individual_runs=aggregated.individual_runs,
        n_runs=aggregated.n_runs,
        n_trimmed=aggregated.n_trimmed,
    )
    gate_report = evaluate_publish_gate(
        forecast=aggregated_for_gate,
        evidence=gate_evidence,
        spec_lock=SpecLockResult(confidence=spec_confidence, canonical_text=canonical_spec_text),
        question_deadline=question_deadline,
    )
    final_probability = aggregated_for_gate.p_calibrated
    if gate_report.action in {"publish_low_confidence", "abstain"}:
        final_probability = shrink_probability(
            p=final_probability,
            gate_score=gate_report.gate_score,
            base_rate=community_prior,
        )

    final_mc_probabilities: dict[str, float] | None = None
    final_numeric_percentiles: dict[int, float] | None = None
    if q_type == "multiple_choice":
        mc_runs = []
        for row in settled:
            diag = (row.get("result").diagnostics or {})
            parsed = diag.get("parsed_mc_probabilities")
            if diag.get("parse_success") and isinstance(parsed, dict):
                mc_runs.append({str(k): float(v) for k, v in parsed.items()})
        if mc_runs:
            final_mc_probabilities = _aggregate_mc_runs(
                per_run=mc_runs,
                options=mc_options,
                trim_fraction=trim_fraction,
                extremize_k=extremize_k,
            )
        else:
            base = 1.0 / max(len(mc_options), 1)
            final_mc_probabilities = {opt: base for opt in mc_options}
        final_probability = max(final_mc_probabilities.values()) if final_mc_probabilities else final_probability
    elif q_type == "numeric":
        numeric_runs = []
        for row in settled:
            diag = (row.get("result").diagnostics or {})
            parsed = diag.get("parsed_numeric_percentiles")
            if diag.get("parse_success") and isinstance(parsed, dict):
                try:
                    numeric_runs.append({int(k): float(v) for k, v in parsed.items()})
                except Exception:
                    continue
        if numeric_runs:
            final_numeric_percentiles = _aggregate_numeric_runs(
                per_run=numeric_runs,
                trim_fraction=trim_fraction,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        else:
            lb = 0.0 if lower_bound is None else lower_bound
            ub = 100.0 if upper_bound is None else upper_bound
            span = max(1e-9, ub - lb)
            final_numeric_percentiles = {
                10: lb + 0.15 * span,
                25: lb + 0.35 * span,
                50: lb + 0.50 * span,
                75: lb + 0.65 * span,
                90: lb + 0.85 * span,
            }
        if lower_bound is not None and upper_bound is not None and upper_bound > lower_bound:
            final_probability = max(
                0.01,
                min(0.99, (final_numeric_percentiles[50] - lower_bound) / (upper_bound - lower_bound)),
            )

    # Ticket 6A/6B: async post-hoc claim audit + signpost extraction (non-blocking to publish decision).
    claim_audit_task = _claim_audit(
        question_title=question_title,
        resolution_criteria=(resolution_criteria or canonical_spec_text),
        reasoning=best_reasoning,
        evidence_bundle_text=evidence_bundle_text,
    )
    signpost_task = _extract_signposts(
        question_title=question_title,
        resolution_criteria=(resolution_criteria or canonical_spec_text),
        final_probability=final_probability,
        reasoning=best_reasoning,
    )
    claim_audit, signpost_report = await asyncio.gather(claim_audit_task, signpost_task)
    signposts = signpost_report.get("signposts", [])
    signposts = _anchor_signposts(signposts, final_probability)
    red_team_changed = abs(p_after_critique - aggregated.p_calibrated) > 1e-9

    top_evidence_rows: list[dict[str, Any]] = []
    for idx, ev in enumerate(gate_evidence[:8], start=1):
        snippet = (ev.content or "")[:140]
        if ev.is_primary_source:
            bearing = "Direct resolution source/mechanics"
        elif ev.relevance_score >= 0.75:
            bearing = "High-directness status signal"
        else:
            bearing = "Proxy/context evidence"
        top_evidence_rows.append(
            {
                "evidence_id": f"E{idx}",
                "source": ev.source_name,
                "url": ev.source_url,
                "retrieved_at": ev.retrieved_at.date().isoformat(),
                "primary": ev.is_primary_source,
                "relevance": round(ev.relevance_score, 3),
                "bearing": bearing,
                "snippet_hash": hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16],
                "snippet": snippet,
            }
        )
    signal_strength = abs(final_probability - 0.5)
    informativeness = "strong_view" if signal_strength > 0.15 else "weak_view"

    summary_lines = [
        f"Question Type: {q_type}",
        f"Final Prediction: {final_probability:.1%}",
        f"Informativeness: {informativeness} (|p-0.5|={signal_strength:.3f})",
        f"Raw Trimmed Mean: {aggregated_for_gate.p_raw:.1%}",
        f"Calibrated Probability: {aggregated_for_gate.p_calibrated:.1%}",
        f"Dispersion: {aggregated_for_gate.dispersion:.3f}",
        f"Confidence Class: {aggregated_for_gate.confidence_class}",
        f"Publish Gate: {gate_report.action} (score={gate_report.gate_score:.2f})",
        f"Parsing Failure Rate: {parsing_failure_rate:.1%}",
        f"Search Queries: planned={len(planned_queries)} executed={len(search_queries)}",
        (
            "Gate Components: "
            f"spec={gate_report.spec_confidence:.2f}, "
            f"temporal={gate_report.temporal_relevance:.2f}, "
            f"evidence={gate_report.evidence_sufficiency:.2f}, "
            f"agreement={gate_report.model_agreement:.2f}"
        ),
        (
            "Gate Metrics: "
            f"evidence_count={gate_report.evidence_count}, "
            f"distinct_sources={gate_report.distinct_sources}, "
            f"primary_sources={gate_report.primary_sources}, "
            f"mean_relevance={gate_report.mean_relevance:.2f}, "
            f"freshness_days={gate_report.freshness_days:.1f}"
        ),
        f"Red Team: ran | {'adjusted' if red_team_changed else 'no_change'}",
        "Individual Runs:",
    ]
    assumptions_note = str(claim_audit.get("assumptions_note", "")).strip()
    if assumptions_note:
        summary_lines.append(f"Assumptions Note: {assumptions_note}")
    for row in settled:
        result = row["result"]
        summary_lines.append(
            f"- {row['config']['label']}: {result.probability:.1%} "
            f"(tokens={result.token_usage.get('total', 0):,}, warnings={len(result.warnings)})"
        )
    if final_mc_probabilities:
        summary_lines.append("MC Probabilities:")
        for opt, p in sorted(final_mc_probabilities.items(), key=lambda x: x[0]):
            summary_lines.append(f"- {opt}: {p:.1%}")
    if final_numeric_percentiles:
        summary_lines.append("Numeric Percentiles:")
        for pctl in [10, 25, 50, 75, 90]:
            unit_text = f" {numeric_unit}" if numeric_unit else ""
            summary_lines.append(f"- P{pctl}: {final_numeric_percentiles[pctl]:.4f}{unit_text}")
    if signposts:
        summary_lines.append("Signposts:")
        for s in signposts[:5]:
            summary_lines.append(
                f"- {s.get('event')} | direction={s.get('direction')} | magnitude={s.get('magnitude')}"
            )
    summary_text = "\n".join(summary_lines)

    full_reasoning_parts = [
        f"# Ensemble Forecast ({final_probability:.1%})",
        "## Search Strategy Output",
        planner_text,
        "## Planned Queries",
        "\n".join([f"- {q}" for q in planned_queries]),
        "## Executed Queries",
        "\n".join([f"- {q}" for q in search_queries]),
        "## Evidence Bundle",
        evidence_bundle_text,
        "## Top Evidence",
        "\n".join(
            [
                (
                    f"- {row['evidence_id']} | {row['source']} | {row['retrieved_at']} | "
                    f"primary={row['primary']} | relevance={row['relevance']} | "
                    f"bearing={row['bearing']} | snippet_hash={row['snippet_hash']} | {row['snippet']}"
                )
                for row in top_evidence_rows
            ]
        ),
        "## Aggregate",
        (
            f"- p_raw={aggregated_for_gate.p_raw:.4f}\n"
            f"- p_calibrated={aggregated_for_gate.p_calibrated:.4f}\n"
            f"- dispersion={aggregated_for_gate.dispersion:.4f}\n"
            f"- confidence={aggregated_for_gate.confidence_class}\n"
            f"- gate_action={gate_report.action}\n"
            f"- gate_components: spec={gate_report.spec_confidence:.3f}, temporal={gate_report.temporal_relevance:.3f}, "
            f"evidence={gate_report.evidence_sufficiency:.3f}, agreement={gate_report.model_agreement:.3f}\n"
            f"- parsing_failure_rate={parsing_failure_rate:.3f}\n"
        ),
        "## Red-Team Critique",
        critique_artifact,
        "## Claim Audit",
        str(claim_audit),
        "## Signposts",
        str(signpost_report),
    ]
    for row in settled:
        full_reasoning_parts.append(
            f"## {row['config']['label']} ({row['result'].probability:.1%})\n\n{row['result'].explanation}"
        )
    full_reasoning = "\n\n---\n\n".join(full_reasoning_parts)

    full_log = _build_structured_report(
        timestamp=timestamp,
        question_title=question_title,
        q_type=q_type,
        canonical_spec_text=canonical_spec_text,
        today_str=today_str,
        planner_text=planner_text,
        planned_queries=planned_queries,
        search_queries=search_queries,
        gate_evidence=gate_evidence,
        top_evidence_rows=top_evidence_rows,
        evidence_bundle_text=evidence_bundle_text,
        settled=settled,
        aggregated=aggregated_for_gate,
        final_probability=final_probability,
        parsing_failure_rate=parsing_failure_rate,
        gate_report=gate_report,
        red_team_changed=red_team_changed,
        critique_artifact=critique_artifact,
        claim_audit=claim_audit,
        signposts=signposts,
        signal_strength=signal_strength,
        informativeness=informativeness,
        final_mc_probabilities=final_mc_probabilities,
        final_numeric_percentiles=final_numeric_percentiles,
        numeric_unit=numeric_unit,
    )
    safe_q = "".join(c for c in question_title[:50] if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
    with open(f"{logs_dir}/forecast_{timestamp}_{safe_q}.txt", "w", encoding="utf-8") as f:
        f.write(full_log)

    if publish_to_metaculus and gate_report.action != "abstain" and q_type == "binary":
        metaculus_token = os.getenv("METACULUS_TOKEN")
        if metaculus_token:
            try:
                if question_url_for_post:
                    found_question = MetaculusApi.get_question_by_url(question_url_for_post)
                else:
                    found_question = MetaculusApi.get_question_by_url(question_title)
                if found_question:
                    q_id = (
                        getattr(found_question, "id", None)
                        or getattr(found_question, "question_id", None)
                        or getattr(found_question, "id_of_question", None)
                        or getattr(found_question, "id_of_post", None)
                        or getattr(found_question, "post_id", None)
                    )
                    if q_id:
                        MetaculusApi.post_binary_question_prediction(
                            question_id=q_id,
                            prediction_in_decimal=final_probability,
                        )
                        comment_text = (
                            "## Automated Ensemble Forecast\n\n"
                            f"{summary_text}\n\n---\n*Generated by forecasting bot*"
                        )
                        try:
                            post_id = (
                                getattr(found_question, "id_of_post", None)
                                or getattr(found_question, "post_id", None)
                                or (getattr(found_question, "api_json", {}) or {}).get("id")
                            )
                            MetaculusApi.post_question_comment(
                                post_id=int(post_id),
                                comment_text=comment_text,
                            )
                        except Exception as comment_err:
                            logger.warning(f"Prediction posted but comment failed: {comment_err}")
            except Exception as post_exc:
                logger.error(f"Failed to post to Metaculus: {post_exc}")

    return {
        "final_probability": final_probability,
        "summary_text": summary_text,
        "full_reasoning": full_reasoning,
        "full_log": full_log,
        "individual_results": settled,
        "aggregated_forecast": {
            "p_raw": aggregated_for_gate.p_raw,
            "p_calibrated": aggregated_for_gate.p_calibrated,
            "dispersion": aggregated_for_gate.dispersion,
            "confidence_class": aggregated_for_gate.confidence_class,
            "n_runs": aggregated_for_gate.n_runs,
            "n_trimmed": aggregated_for_gate.n_trimmed,
        },
        "gate_report": {
            "publishable": gate_report.publishable,
            "confidence_class": gate_report.confidence_class,
            "gate_score": gate_report.gate_score,
            "spec_confidence": gate_report.spec_confidence,
            "temporal_relevance": gate_report.temporal_relevance,
            "evidence_sufficiency": gate_report.evidence_sufficiency,
            "model_agreement": gate_report.model_agreement,
            "evidence_count": gate_report.evidence_count,
            "distinct_sources": gate_report.distinct_sources,
            "primary_sources": gate_report.primary_sources,
            "mean_relevance": gate_report.mean_relevance,
            "freshness_days": gate_report.freshness_days,
            "action": gate_report.action,
            "reasons": gate_report.reasons,
        },
        "publish_action": gate_report.action,
        "planned_queries": planned_queries,
        "executed_queries": search_queries,
        "top_evidence": top_evidence_rows,
        "parsing_failure_rate": parsing_failure_rate,
        "question_type": q_type,
        "signal_strength": signal_strength,
        "informativeness": informativeness,
        "mc_probabilities": final_mc_probabilities,
        "numeric_percentiles": final_numeric_percentiles,
        "red_team_artifact": critique_artifact,
        "claim_audit": claim_audit,
        "signpost_report": signpost_report,
        "signposts": signposts,
        "feature_flags": {
            "spec_lock": _flag("spec_lock", True),
            "evidence_ledger": _flag("evidence_ledger", True),
            "numeric_provenance": _flag("numeric_provenance", True),
            "market_snapshot": _flag("market_snapshot", True),
            "outlier_xexam": _flag("outlier_xexam", True),
        },
    }
