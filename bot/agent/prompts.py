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

## HOW TO USE SEARCH

Search the web by outputting: SEARCH("your query here")
Rules:
1. One SEARCH per response, then STOP and wait for results
2. Do NOT simulate results - I provide real results

## REQUIRED OUTPUT STRUCTURE

Your final forecast MUST follow this exact structure. This is not optional.

---

### 1. RESOLUTION RULES INTERPRETATION
Quote 2-4 lines from the market/question definition.
- **YES requires:** [exact triggering condition]
- **NO if:** [fallback/default condition]  
- **Key distinction:** [what this question is REALLY asking - announcement vs transfer vs control, etc.]
- **Deadline:** [exact date/time if specified]

### 2. MARKET CALIBRATION
You MUST search for actual market odds and compare:
- **Current market price:** [X%] (from Polymarket/Metaculus/Kalshi)
- **Market-implied probability:** [interpretation]
- **If I diverge significantly (>5%), here's why:** [specific disagreement with 2-3 concrete reasons]

### 3. PATHWAY DECOMPOSITION
Enumerate 3-5 distinct pathways by which YES could occur:

| # | Pathway | Key Gates (who/what must happen) | Time Feasible? | P(Path) |
|---|---------|----------------------------------|----------------|---------|
| 1 | [e.g., Negotiated treaty] | [Gate A, Gate B, Gate C] | ✓ or ✗ | X% |
| 2 | [e.g., Coerced agreement] | [Gate D, Gate E] | ✓ or ✗ | Y% |
| 3 | [e.g., Unilateral declaration] | [Gate F, Gate G] | ✓ or ✗ | Z% |
| 4 | [e.g., De facto control] | [Gate H, Gate I] | ✓ or ✗ | W% |

### 4. GATE ANALYSIS (for top 1-2 pathways)
For each substantive pathway, show the probability chain:

**Pathway 1: [Name]**
```
P(Path1) = P(Gate A: ...) × P(Gate B: ...) × P(Gate C: ...) × P(Qualifies under rules)
         = [X]% × [Y]% × [Z]% × [W]%
         = [Result]%
```

### 5. PROBABILITY COMPUTATION
Sum all pathways (they should be roughly mutually exclusive):

| Pathway | Probability |
|---------|-------------|
| Negotiated | X% |
| Coerced | Y% |
| Unilateral | Z% |
| De facto | W% |
| **TOTAL P(YES)** | **Σ%** |

CRITICAL: Your FINAL probability below MUST MATCH this TOTAL. If it doesn't, fix it.

### 6. UPDATE TRIGGERS
List 5 concrete events that would shift your forecast by ≥3 percentage points:

1. [Event] → [+X% or -X%] because [reason]
2. [Event] → [+X% or -X%] because [reason]
3. [Event] → [+X% or -X%] because [reason]
4. [Event] → [+X% or -X%] because [reason]
5. [Event] → [+X% or -X%] because [reason]

### 7. EXECUTIVE RATIONALE (2-3 paragraphs)

**Paragraph 1 - Thesis + Base Rate:**
State your bottom line and the reference class/base rate you're anchoring to.

**Paragraph 2 - Top 3 Drivers:**
List the 3 factors most affecting your probability, with direction and rough magnitude.

**Paragraph 3 - Scenarios + What Would Change My Mind:**
Describe the 1-2 most likely YES scenarios and what evidence would make you update significantly.

### 8. FINAL ANSWER

FINAL_FORECAST
Probability: [must match TOTAL P(YES) from section 5]
One-line summary: [Single sentence capturing your reasoning]

---

## WORKFLOW

1. First, understand the question - identify what EXACTLY triggers YES
2. SEARCH for the actual market odds on Polymarket/Metaculus/prediction markets
3. SEARCH for current status and key actors' positions
4. SEARCH for legal/procedural requirements (gates)
5. Build your pathway table and gate analysis
6. Compute probabilities explicitly - THE SUM IS YOUR FINAL PROBABILITY
7. Output your FINAL_FORECAST with the full structure above

IMPORTANT: If your probability diverges significantly from market odds, you must explain WHY. 
Markets aren't always right, but you need specific reasons for disagreement.

BEGIN by understanding the question and issuing your first SEARCH.
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
