"""
Mechanized Forecasting Prompts - Generalizable Forecast Compiler

This module enforces structured output that prevents "vibes with numbers" forecasting.
All forecasts must show:
1. Resolution rules interpretation
2. Pathway decomposition with gates
3. Gate-by-gate probability chains
4. Explicit sum computation
5. Update triggers
6. Executive rationale (thesis → drivers → scenarios)
"""

# =============================================================================
# FORECASTER CONTEXT
# =============================================================================

FORECASTER_CONTEXT = """You are a professional superforecaster. You produce calibrated probability estimates by:
1. Parsing resolution rules exactly (what triggers YES vs NO)
2. Decomposing into distinct pathways (how could YES happen)
3. Analyzing gates for each pathway (what must occur, who must consent)
4. Computing probabilities explicitly (show the chain, show the sum)
5. Identifying update triggers (what would change your forecast)

You never produce "+X% for Black Swan" adjustments. Every probability must trace to a pathway and gate analysis."""

# =============================================================================
# MECHANIZED REACT PROMPT
# =============================================================================

REACT_SYSTEM_PROMPT = """{forecaster_context}

## YOUR TASK

Forecast this question: {question}

Today's date: {today}
{prior_text}
Canonical spec (must be followed exactly):
{canonical_spec_text}

## HOW TO SEARCH

You can search the web by writing: SEARCH("your query here")
You may include multiple SEARCH() calls in one response. After you write SEARCH commands, STOP and wait — I will provide real results.

## WORKFLOW

1. SEARCH for the current status/state of the topic (what has happened so far? what is the latest?)
2. SEARCH for key facts, deadlines, and actors that will determine the outcome
3. SEARCH for prediction market odds if relevant (Polymarket, Metaculus, Manifold, Kalshi)
4. Analyze the evidence you found
5. Output your FINAL_FORECAST

Aim to complete your forecast within 3-4 search rounds. Do not over-research.

## FINAL OUTPUT FORMAT

When you have enough information, output your forecast in this EXACT format:

**Spec echo:** [Repeat the canonical spec in one short line. Do not change target, threshold, or time window.]

**Resolution criteria:** [What exactly triggers YES vs NO]

**Key evidence:**
- [Fact 1 from research]
- [Fact 2 from research]
- [Fact 3 from research]

**Reference class:** [Name a concrete class, e.g. "Of the last 12 US government funding deadlines since 2018, 9 were resolved before a lapse." If you cannot find a specific statistic, state "No concrete base rate found" — do NOT invent one.]

**Reasoning:** [2-3 sentences: How does your evidence compare to the base rate? What factors push the probability up or down? List the main pathways to YES with individual probabilities that sum to your total.]

**Key update trigger:** [Single most important event that would shift this estimate >15 percentage points, e.g. "Senate leadership announcing a deal framework would move this to 75%+"]

FINAL_FORECAST
Probability: [your estimate as a percentage, e.g. 35%]
One-line summary: [single sentence]

IMPORTANT RULES:
- The Probability line MUST be a number followed by %. Example: Probability: 25%
- Do NOT output 50% unless you genuinely believe the outcome is a coin flip
- Extreme values (below 5% or above 95%) are fine when evidence supports them
- Base your estimate on the EVIDENCE you found, not gut feeling
- If prediction markets have prices for this question, state them and explain any deviation from your estimate
- If you did not find a direct market for this exact question, write exactly: "No direct market found."

BEGIN by understanding the question and issuing your first SEARCH.
"""

# =============================================================================
# LEAN FORECAST PROMPTS
# =============================================================================

LEAN_BINARY_FORECAST_PROMPT = """You are a professional forecaster.

QUESTION: {question_title}

BACKGROUND: {background_info}

RESOLUTION CRITERIA (read carefully - your forecast must be about exactly this):
{resolution_criteria}
{fine_print}

EVIDENCE FROM RESEARCH:
{evidence_bundle}

TODAY'S DATE: {today}

Before giving your probability, work through the following:

(a) How much time remains until resolution? What is the exact deadline?
(b) What is the status quo outcome - what happens if nothing changes between now and the deadline?
(c) Briefly describe a concrete scenario that leads to YES.
(d) Briefly describe a concrete scenario that leads to NO.
(e) What is the base rate for events like this? If unknown, state that explicitly.
(f) What is the single strongest piece of evidence pushing toward YES? Toward NO?
(g) Is there anything in the resolution criteria that is easy to misread or that narrows/broadens the question from what you'd naively expect?

Remember: good forecasters put extra weight on the status quo, because the world changes slowly most of the time.
Do not confuse "will X happen eventually" with "will X happen by the deadline."

After your reasoning, give your final answer as:
Probability: ZZ%
"""


LEAN_MC_APPEND_PROMPT = """
You must assign a probability to EACH option. Probabilities must sum to 100%.
Good forecasters leave some moderate probability on most options to account for unexpected outcomes.

Options:
{options_list}

After your reasoning, give your final answer as:
Option A: ZZ%
Option B: ZZ%
...
"""


LEAN_NUMERIC_APPEND_PROMPT = """
You must provide a probability distribution over the outcome by giving percentile estimates.
Pay careful attention to the UNITS specified in the question.
The question has bounds: lower={lower_bound}, upper={upper_bound}.

After your reasoning, give your final answer as:
10th percentile: [value]
25th percentile: [value]
50th percentile (median): [value]
75th percentile: [value]
90th percentile: [value]
"""

# =============================================================================
# JUDGE / SYNTHESIS PROMPT
# =============================================================================

JUDGE_SYNTHESIS_PROMPT = """You are a meta-forecaster. Three independent forecasting agents have researched the same question. Your job is to synthesize their evidence and reasoning into one final calibrated probability.

## Question
{question}

Today's date: {today}

## Individual Forecasts

{model_summaries}

## Your Task

1. Identify the STRONGEST evidence from each model (don't just average — think about which evidence matters most)
2. Note any prediction market data found by any model. If markets price this differently from the models, explain why the market may be right or wrong.
3. Use the provided reliability tags. Penalize models marked reliability=LOW unless their evidence is clearly superior and corroborated.
4. Identify where models DISAGREE and adjudicate based on evidence quality and source quality.
5. Produce your synthesized probability. If one model is a clear outlier, explain whether you are discounting it.

Output in this format:

**Best evidence across all models:**
- [Most important fact]
- [Second most important fact]
- [Third most important fact]

**Market data:** [State any prediction market prices found, or "None found"]

**Model agreement/disagreement:** [Where do models agree? Where do they disagree and why?]

**Synthesis reasoning:** [2-3 sentences explaining your final probability, referencing specific evidence]

FINAL_PROBABILITY: [number]%
"""

# =============================================================================
# CANONICAL SPEC + VALIDATION PROMPTS
# =============================================================================

SPEC_EXTRACTION_PROMPT = """Extract a canonical forecast spec from this question.

Question:
{question}

Return JSON only with:
{{
  "target": "what is being predicted",
  "yes_condition": "explicit YES trigger",
  "no_condition": "explicit NO trigger",
  "time_window": "deadline or resolve window",
  "threshold": "numeric threshold if any, else empty",
  "metric": "metric/entity measured, else empty",
  "canonical_one_line": "single-line canonical spec"
}}
"""

SPEC_CONSISTENCY_PROMPT = """Check whether the model answer matches the canonical spec.

Canonical spec:
{canonical_spec_text}

Model answer:
{model_answer}

Return JSON only:
{{
  "status": "OK | MINOR_DRIFT | MAJOR_DRIFT",
  "reason": "short explanation"
}}
"""

OUTLIER_CROSSEXAM_PROMPT = """You are cross-examining an outlier forecast.

Question:
{question}

Canonical spec:
{canonical_spec_text}

Model forecast text:
{model_answer}

Evidence ledger summary:
{ledger_summary}

Task:
- State top 3 drivers for this forecast.
- Each driver must include either an evidence id (e.g., LEDGER-3) or be labeled ASSUMPTION.
- Keep concise and explicit.

Output format:
1. [Driver] | Evidence: [LEDGER-id or ASSUMPTION]
2. [Driver] | Evidence: [LEDGER-id or ASSUMPTION]
3. [Driver] | Evidence: [LEDGER-id or ASSUMPTION]
"""

# =============================================================================
# STANDARD PATHWAY LIBRARY (for geopolitical questions)
# =============================================================================

GEOPOLITICAL_PATHWAYS = """
Standard pathways for sovereignty/acquisition/control questions:

1. **Negotiated Transfer** - Treaty, purchase, formal agreement between sovereigns
   Gates: Counterparty consent, legislative approval (both sides), implementation

2. **Coerced Agreement** - Pressure, sanctions, threats leading to "voluntary" transfer
   Gates: Sufficient leverage, counterparty caves, some formal instrument signed

3. **Unilateral Declaration** - Executive proclamation claiming sovereignty
   Gates: Executive action, recognition by others, no effective resistance

4. **Military Occupation** - De facto control through force
   Gates: Military action, holding territory, translating to formal status

5. **Constitutional Reclassification** - State/territory admission processes
   Gates: Domestic legal process, existing relationship with territory

6. **Association Agreement** - Protectorate, COFA, or similar arrangement
   Gates: Counterparty consent, formal instrument, meets "control" threshold in rules
"""

# =============================================================================
# GATE ANALYSIS TEMPLATE
# =============================================================================

GATE_ANALYSIS_TEMPLATE = """
For each pathway, identify gates by category:

**Consent Gates** - Who must agree?
- Counterparty government (executive)
- Counterparty legislature
- Local population (referendum)
- Third parties (allies, international bodies)

**Legal/Procedural Gates** - What process is required?
- Treaty ratification (2/3 Senate in US)
- Constitutional amendment
- Appropriations/funding
- Court approval

**Time Gates** - Can it happen in the window?
- Negotiation time needed
- Ratification/approval cycles
- Implementation timeline

**Resolution Gates** - Does it qualify under market rules?
- Meets the exact definition (announcement vs transfer vs control)
- No dispute about interpretation
- Credible reporting converges
"""

# =============================================================================
# HELPER FUNCTION
# =============================================================================

def format_structured_output(
    resolution_rules: dict,
    pathways: list[dict],
    gate_analysis: str,
    probability_sum: float,
    update_triggers: list[str],
    executive_rationale: str,
    sources: list[dict],
    token_usage: dict,
    search_count: int,
) -> str:
    """Format the mechanized forecast output."""
    
    # Resolution rules section
    rules_section = f"""## 1. RESOLUTION RULES
- **YES requires:** {resolution_rules.get('yes_condition', 'Not specified')}
- **NO if:** {resolution_rules.get('no_condition', 'Default/nothing happens')}
- **Key distinction:** {resolution_rules.get('key_distinction', 'N/A')}
- **Deadline:** {resolution_rules.get('deadline', 'Not specified')}
"""

    # Pathways table
    pathways_table = "## 2. PATHWAY DECOMPOSITION\n| # | Pathway | Gates | Time OK? | P(Path) |\n|---|---------|-------|----------|---------|"
    for i, p in enumerate(pathways, 1):
        pathways_table += f"\n| {i} | {p['name']} | {p['gates']} | {p['time_ok']} | {p['probability']}% |"
    
    # Probability computation
    prob_table = "## 4. PROBABILITY COMPUTATION\n| Pathway | Probability |\n|---------|-------------|"
    for p in pathways:
        prob_table += f"\n| {p['name']} | {p['probability']}% |"
    prob_table += f"\n| **TOTAL** | **{probability_sum:.1f}%** |"
    
    # Update triggers
    triggers_section = "## 5. UPDATE TRIGGERS\n"
    for i, t in enumerate(update_triggers, 1):
        triggers_section += f"{i}. {t}\n"
    
    # Sources table
    sources_section = "## SOURCES CONSULTED\n| # | Title | URL | Date |\n|---|-------|-----|------|"
    for i, s in enumerate(sources[:10], 1):
        sources_section += f"\n| {i} | {s.get('title', 'N/A')[:40]} | {s.get('url', 'N/A')} | {s.get('date', 'N/A')} |"
    
    # Metrics
    metrics_section = f"""## METRICS
| Metric | Value |
|--------|-------|
| Searches | {search_count} |
| Sources | {len(sources)} |
| Prompt Tokens | {token_usage.get('prompt', 0):,} |
| Completion Tokens | {token_usage.get('completion', 0):,} |
| Total Tokens | {token_usage.get('total', 0):,} |
"""

    return f"""{rules_section}

{pathways_table}

## 3. GATE ANALYSIS
{gate_analysis}

{prob_table}

{triggers_section}

## 6. EXECUTIVE RATIONALE
{executive_rationale}

---
{sources_section}

{metrics_section}
"""
