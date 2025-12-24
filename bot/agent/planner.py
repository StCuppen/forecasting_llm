import logging
from typing import List, Tuple

from .utils import call_openrouter_llm, clean_indents

logger = logging.getLogger(__name__)

async def make_search_plan(
    question_text: str,
    today_str: str,
    llm_model: str = "openai/gpt-5"
) -> Tuple[List[str], str]:
    """
    Generate a minimal search plan.

    Returns:
        queries: list of search query strings
        planner_text: markdown text with rationale, info baskets, and queries
    """
    prompt = clean_indents(f"""
        You are a research planner assisting a forecasting bot.

        Given a forecasting question and today's date, your job is to:
        1. Clarify the research objective.
        2. identify key sub questions / uncertainties
        3. Decompose the information needs into a small set of broad "information baskets".
        4. Propose search queries for each basket.

        for the following baskets:

BASE RATES & REFERENCE CLASSES (Outside View)
Historical analogues, frequencies, and patterns of change for similar situations. Use structural similarities (market type, competition structure, institutional rules, etc.) to pick the right reference classes.

STRUCTURAL DYNAMICS & CONSTRAINTS (Outside View)
How systems of this type usually evolve: typical swing sizes, durability of leaders, usual timescales, hard constraints (laws, physics, institutional rules).

CURRENT STATE & TRAJECTORY (Inside View)
Direct measurement of where things stand today and how they’re moving: polls, leaderboards, prices, adoption levels, recent shocks and trends.

ACTORS, INCENTIVES & CAPABILITIES (Inside View)
Who can act, who benefits or loses, who has resources or veto power, and what their past behavior suggests.

MECHANISMS, PATHS & SIGNPOSTS (Inside–Bridge)
Concrete paths to different outcomes (YES/NO or high/low), key gates and bottlenecks, and leading indicators that would significantly update the forecast.

MARKETS & AGGREGATED EXPERT SENTIMENT (Outside View)
Prediction markets, forecasting platforms (Metaculus, Manifold, Polymarket), expert consensus. 
CRITICAL: For any question that might have existing forecasts, you MUST include queries like:
- "[question topic] Metaculus probability forecast"
- "[question topic] Manifold market odds percentage"  
- "[question topic] prediction market forecast"
These community probabilities serve as strong Bayesian priors for calibration.
        Add or drop baskets as appropriate, but cover these dimensions when they are relevant.

        Output MUST follow this exact format in Markdown:

        # PLANNER

        ## A. Research Objective
        - Today's date: {today_str}
        - One-sentence objective: ...
        - Question type: <binary / numeric / multiple-choice / other>
        - identify key sub questions / uncertainties

        ## B. Information Baskets
        - Basket 1: <name> — <2-3 sentence description>
        - Basket 2: <name> — <2-3 sentence description>
        - Basket 3: <name> — <2-3 sentence description>
        (use all 6 baskets.)

        ## C. Queries
    generate max 15 queries in total, corresponding to the information baskets. you may weigh some baskets more heavily than others if you see fit given the specifics of a question

        **CRITICAL query guidelines:**
        1. **Named entities first**: Every query MUST include the key named entities from the question (company names, platform names, specific metrics, person names, etc.). Do NOT use generic synonyms.
        2. **Direct state lookup**: At least 2-3 queries in CURRENT STATE must be DIRECT lookups of the actual metric/platform/leaderboard named in the question. Example: if the question mentions "Chatbot Arena", query "Chatbot Arena leaderboard current leader December 2025", NOT "chatbot market leadership trends".
        3. **Resolution criteria**: Include queries that directly target HOW the question resolves (e.g., the specific ranking, the specific threshold, the specific date).
        4. **Recency markers**: Include date/year markers in queries to get fresh results (e.g., "2025", "December 2025", "latest").
        5. **Avoid vague abstractions**: Do NOT use generic queries like "market trends" or "industry dynamics" when the question names a specific platform or metric.

        Format:
        - Queries must be concise (<= 120 characters).
        - No URLs, no Boolean operators; use natural-language search strings.
        - Do NOT omit section C.
        - Do NOT change section labels or heading levels.
        - Do not output anything outside this template.

        Question:
        {question_text}

        Today's date: {today_str}
    """)
    
    try:
        response_text = None
        # Up to 3 attempts to get usable queries
        for attempt in range(3):
            response_text = await call_openrouter_llm(
                prompt=prompt,
                model=llm_model,
                temperature=0.3,
                max_tokens=8000,  # generous, but not absurd
                usage_label="planner",
            )
            text = response_text.strip()
            queries: List[str] = []
            in_queries = False
            for line in text.splitlines():
                stripped = line.strip()
                lower = stripped.lower()
                # Enter queries section on a reasonably broad match so we
                # don't get stuck if the model omits heading hashes.
                if (
                    not in_queries
                    and (
                        lower.startswith("## c. queries")
                        or lower.startswith("c. queries")
                        or lower.startswith("## c ") and "queries" in lower
                        or lower == "## c. queries"
                        or lower == "c. queries"
                    )
                ):
                    in_queries = True
                    continue
                if in_queries:
                    if stripped.startswith("## "):
                        # next section, stop reading queries
                        break
                    if not stripped:
                        continue

                    # Accept both bullet and bare lines
                    if stripped.startswith("- "):
                        q = stripped[2:].strip()
                    else:
                        q = stripped

                    # Strip leading "Q1:", "1.", etc. if present
                    if ":" in q:
                        head, tail = q.split(":", 1)
                        # short heads like "Q1", "1", "Q10 (Basket ...)" etc.
                        if len(head) <= 8:
                            q = tail.strip()

                    # Heuristic filters: drop obvious meta / fallback notes
                    lower_q = q.lower()
                    if (
                        q
                        and len(q) >= 8
                        and not lower_q.startswith("(planner fallback")
                    ):
                        queries.append(q)

            if queries:
                planner_text = text
                return queries, planner_text

            logger.warning(
                f"Planner attempt {attempt + 1} produced no queries; retrying."
            )
            logger.debug(f"Planner output (first 1000 chars): {text[:1000]}")

        # Fallback if all attempts produced no queries
        logger.warning("Planner returned no queries after retries; falling back to question text.")
        planner_text = (response_text.strip() if response_text else "") + "\n\n(Planner fallback: used question text as generic query.)"
        return [question_text], planner_text

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        fallback_text = (
            "# PLANNER\n\n## A. Research Objective\n- Today's date: {today_str}\n- One-sentence objective: planner failed\n"
            f"- Question type: binary\n\n## B. Information Baskets\n- Basket: General — planner failed\n\n"
            f"## C. Queries\n- Q1 (Fallback): {question_text}"
        )
        return [question_text], fallback_text
