import logging
import asyncio
import httpx
import trafilatura
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Tuple

from .utils import (
    SerperClient,
    TavilyClient,
    SonarClient,
    BraveClient,
    ScrapingDogClient,
    LangSearchClient,
    ExaClient,
    call_openrouter_llm,
    clean_indents,
)

logger = logging.getLogger(__name__)


class BasketType:
    DIRECT_STATUS = "direct_status"
    BASE_RATES = "base_rates"
    MECHANISMS = "mechanisms"
    INCENTIVES = "incentives"
    MARKETS_NEWS = "markets_news"
    GENERAL = "general"


def classify_query(query: str) -> str:
    """
    Heuristic classification of a query into a basket type.

    Intentionally simple to avoid over-engineering; uses keyword matching.
    """
    q = query.lower()
    if any(k in q for k in ["current", "today", "now", "latest", "leaderboard", "standing", "rank"]):
        return BasketType.DIRECT_STATUS
    if any(k in q for k in ["history", "historical", "past", "trend", "over time", "since"]):
        return BasketType.BASE_RATES
    if any(k in q for k in ["mechanism", "how does", "process", "pathway", "gate", "bottleneck"]):
        return BasketType.MECHANISMS
    if any(k in q for k in ["incentive", "motivation", "actor", "stakeholder", "strategy"]):
        return BasketType.INCENTIVES
    if any(k in q for k in ["poll", "election", "market", "betting", "odds", "price"]):
        return BasketType.MARKETS_NEWS
    return BasketType.GENERAL


@dataclass
class RawHit:
    url: str
    title: Optional[str]
    snippet: Optional[str]
    provider: str
    score: float
    query: str
    basket_type: str
    full_content: Optional[str] = None


@dataclass
class Doc:
    url: str
    title: Optional[str]
    basket_type: str
    providers: Set[str]
    snippet: Optional[str]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

async def fetch_clean_text(url: str, timeout: int = 15) -> str:
    """
    Fetch and return reasonably clean text for a URL.
    Prefer trafilatura; fall back to Jina Reader, ScrapingDog (if key set), or httpx.
    """
    # 1. Try trafilatura first (fastest, cheapest)
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
        if text and len(text) > 500:
            return text

    # 2. Try Jina Reader (free, handles some dynamic content)
    # Use a separate timeout for Jina as it can be slow
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(jina_url)
            if resp.status_code == 200 and len(resp.text) > 500:
                return resp.text
    except Exception as e:
        logger.warning(f"Jina Reader failed for {url}: {e}")

    # 3. Try ScrapingDog for dynamic sites if trafilatura/Jina failed
    scraping_dog = ScrapingDogClient()
    if scraping_dog.api_key:
        text = await scraping_dog.scrape(url, dynamic=True)
        # Clean it up a bit with trafilatura if possible, or just return raw text
        if text:
            cleaned = trafilatura.extract(text)
            if cleaned and len(cleaned) > 300:
                return cleaned
            # If trafilatura can't clean the dynamic HTML, fallback to simple text extraction
            # or just return the raw HTML if needed, but let's try to be clean
            return text[:50000] # Limit size

    # 4. Fallback to httpx raw fetch
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "ForecastBot/0.1"},
            )
            resp.raise_for_status()
            return resp.text[:50000]
    except Exception:
        return ""


async def multi_provider_search(
    queries: List[str],
    serper_client: Optional[SerperClient] = None,
    tavily_client: Optional[TavilyClient] = None,
    brave_client: Optional[BraveClient] = None,
    sonar_client: Optional[SonarClient] = None,
    langsearch_client: Optional[LangSearchClient] = None,
    exa_client: Optional["ExaClient"] = None,
    use_sonar: bool = True,
) -> Tuple[List[Doc], Dict[str, Dict[str, Any]]]:
    """
    Run primary search (Exa) + secondary search (Serper) on the given queries.
    
    Search priority: Exa (primary, returns full content) > Serper (fallback).
    Exa provides AI-native search with full text content extraction.
    
    Note: Exa has aggressive rate limits, so we run searches in small batches
    with delays to avoid 429 errors.
    """
    # 1. Classify queries
    # Strip quotes to improve search recall
    cleaned_queries = [q.strip('"\'') for q in queries]
    classified = [(q, classify_query(q)) for q in cleaned_queries]

    # 2. Run Exa searches with rate limiting (batches of 3 with 0.5s delay)
    primary_results_lists: List[Any] = []
    primary_provider = "exa"
    
    if exa_client is not None:
        EXA_BATCH_SIZE = 3
        EXA_DELAY = 0.5  # seconds between batches
        
        for i in range(0, len(classified), EXA_BATCH_SIZE):
            batch = classified[i:i + EXA_BATCH_SIZE]
            batch_tasks = [exa_client.search(q) for q, _ in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            primary_results_lists.extend(batch_results)
            
            # Delay before next batch (skip delay after last batch)
            if i + EXA_BATCH_SIZE < len(classified):
                await asyncio.sleep(EXA_DELAY)
    else:
        primary_results_lists = [None] * len(classified)
        primary_provider = None

    # 3. Run Serper searches concurrently (no rate limit issues)
    secondary_results_lists: List[Any]
    secondary_provider = None
    
    if serper_client:
        secondary_provider = "serper"
        secondary_tasks = [serper_client.search(q) for q, _ in classified]
        secondary_results_lists = await asyncio.gather(*secondary_tasks, return_exceptions=True)
    else:
        secondary_results_lists = [None] * len(classified)

        primary_results_lists = [None] * len(classified)
        secondary_results_lists = await run_secondary()

    raw_hits: List[RawHit] = []

    # 3. Normalize Primary (Exa) results (top-5 per query, includes full content)
    for (q, basket), hits in zip(classified, primary_results_lists):
        if hits is None:
            continue
        if isinstance(hits, Exception):
            logger.error(f"{primary_provider} search failed for query '{q}': {hits}")
            continue
        hits = (hits or [])[:5]
        for r in hits:
            raw_hits.append(
                RawHit(
                    url=r.get("url", ""),
                    title=r.get("title"),
                    snippet=r.get("content"),
                    provider=primary_provider,
                    score=float(r.get("score", 0.0)),
                    query=q,
                    basket_type=basket,
                    # Exa provides text/raw_content directly
                    full_content=r.get("raw_content") or r.get("content"),
                )
            )

    # 4. Normalize Secondary (Serper) results (top-3 per query)
    for (q, basket), hits in zip(classified, secondary_results_lists):
        if hits is None or secondary_provider is None:
            continue
        if isinstance(hits, Exception):
            logger.error(f"{secondary_provider} search failed for query '{q}': {hits}")
            continue
        hits = (hits or [])[:3]
        for r in hits:
            raw_hits.append(
                RawHit(
                    url=r.get("url", ""),
                    title=r.get("title"),
                    snippet=r.get("content"),
                    provider=secondary_provider,
                    score=float(r.get("score", 0.0)),
                    query=q,
                    basket_type=basket,
                    full_content=None,  # Serper only provides snippets
                )
            )

    # 5. Deduplicate by URL, merging providers
    by_url: Dict[str, Doc] = {}
    for h in raw_hits:
        if not h.url:
            continue
        existing = by_url.get(h.url)
        if existing:
            existing.providers.add(h.provider)
            # If we have full content now and didn't before, update it
            if h.full_content and not existing.text:
                existing.text = h.full_content
            continue
        by_url[h.url] = Doc(
            url=h.url,
            title=h.title,
            basket_type=h.basket_type,
            providers={h.provider},
            snippet=h.snippet,
            text=h.full_content or "",
        )

    # 5. Scrape full content for all docs (skip synthetic Sonar URLs)
    docs: List[Doc] = list(by_url.values())

    # Filter for scrapeable HTTP URLs that don't have text yet
    to_scrape_docs = [d for d in docs if d.url.startswith("http") and not d.text]
    
    # If we have Tavily, use its Extract API which handles JS/dynamic sites better
    if tavily_client and to_scrape_docs:
        urls_to_extract = [d.url for d in to_scrape_docs[:5]]  # Limit to top 5 to save credits/time
        remaining_docs = to_scrape_docs[5:]
        
        if urls_to_extract:
            try:
                logger.info(f"Using Tavily Extract for {len(urls_to_extract)} URLs...")
                extract_resp = await tavily_client.extract(urls=urls_to_extract)
                results = extract_resp.get("results", [])
                
                # Map results back to docs
                results_map = {r["url"]: r.get("raw_content", "") for r in results}
                
                for d in to_scrape_docs:
                    if d.url in results_map:
                        d.text = results_map[d.url]
                        
            except Exception as e:
                logger.error(f"Tavily Extract failed: {e}")
                # Fallback to standard scraping for these if Tavily fails
                remaining_docs = to_scrape_docs

        # Standard scraping for remaining docs (or all if Tavily failed/missing)
        if remaining_docs:
            scrape_tasks = [fetch_clean_text(d.url) for d in remaining_docs]
            scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)
            for d, content in zip(remaining_docs, scraped_contents):
                if isinstance(content, Exception):
                    logger.error(f"Scrape failed for '{d.url}': {content}")
                    d.text = d.snippet or ""
                else:
                    d.text = content or d.snippet or ""
    else:
        # Fallback: Standard scraping for all if no Tavily client
        scrape_tasks = [fetch_clean_text(d.url) for d in to_scrape_docs]
        scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        for d, content in zip(to_scrape_docs, scraped_contents):
            if isinstance(content, Exception):
                logger.error(f"Scrape failed for '{d.url}': {content}")
                d.text = d.snippet or ""
            else:
                d.text = content or d.snippet or ""

    # 6. Optional: Sonar for key baskets (status + markets/news)
    sonar_notes: Dict[str, Dict[str, Any]] = {}
    if use_sonar and sonar_client is not None:
        seen_baskets: set[str] = set()
        for q, basket in classified:
            if basket not in {BasketType.DIRECT_STATUS, BasketType.MARKETS_NEWS}:
                continue
            if basket in seen_baskets:
                continue
            seen_baskets.add(basket)
            try:
                ans = await sonar_client.answer_with_citations(q, max_tokens=600)
                sonar_notes[basket] = {"query": q, "answer": ans.get("answer", "")}
                answer_text = ans.get("answer", "") or ""
                citations = ans.get("citations") or []

                # Add Sonar synthesis as a pseudo-doc
                sonar_doc = Doc(
                    url=f"sonar://{basket}",
                    title=f"Sonar synthesis for {basket}",
                    basket_type=basket,
                    providers={"sonar"},
                    snippet=None,
                    text=answer_text,
                )
                docs.append(sonar_doc)

                # Turn each citation URL into a real document
                for cit_url in citations:
                    if not isinstance(cit_url, str) or not cit_url.startswith("http"):
                        continue
                    existing_doc = by_url.get(cit_url)
                    if existing_doc:
                        existing_doc.providers.add("sonar_citation")
                        continue
                    new_doc = Doc(
                        url=cit_url,
                        title=None,
                        basket_type=basket,
                        providers={"sonar_citation"},
                        snippet=None,
                        text="",
                    )
                    by_url[cit_url] = new_doc
                    docs.append(new_doc)
            except Exception as e:
                logger.error(f"Sonar call failed for query '{q}': {e}")

    return docs, sonar_notes


async def perform_research(
    question_text: str,
    queries: List[str],
    today_str: str,
    serper_client: Optional[SerperClient] = None,
    tavily_client: Optional[TavilyClient] = None,
    brave_client: Optional[BraveClient] = None,
    langsearch_client: Optional[LangSearchClient] = None,
    exa_client: Optional["ExaClient"] = None,
    llm_model: str = "openai/gpt-5-mini",
    sonar_client: Optional[SonarClient] = None,
    use_sonar: bool = True,
) -> str:
    """
    Execute the search queries and synthesize a research memo (markdown).

    Returns:
        research_memo: structured markdown memo with fixed sections.
    """
    logger.info(f"Executing {len(queries)} queries via providers...")
    docs, sonar_notes = await multi_provider_search(
        queries=queries,
        serper_client=serper_client,
        tavily_client=tavily_client,
        brave_client=brave_client,
        langsearch_client=langsearch_client,
        exa_client=exa_client,
        sonar_client=sonar_client,
        use_sonar=use_sonar,
    )

    if docs:
        logger.info("Research documents used (top URLs):")
        for d in docs[:10]:
            logger.info(f"  - {d.title or 'Untitled'} ({d.url})")

    # Build context text with doc text
    context_lines = []
    for i, d in enumerate(docs, 1):
        title = d.title or "Untitled"
        url = d.url or ""
        content = d.text or d.snippet or ""
        context_lines.append(f"[Doc {i}] {title} ({url})\n{content}")
    context_text = "\n\n".join(context_lines)

    # 3. Synthesize Research Memo
    prompt = clean_indents(f"""
        You are a research assistant for a forecasting bot.

        You are given:
        - A forecasting question.
        - Today's date.
        - A set of web documents (title, url, content, and sometimes raw_content).

        Your job is to produce a structured research memo with NO probabilities and NO final forecast.

        Output MUST follow exactly this format:

        # RESEARCH MEMO

        ## 1. Triage
        - Question (in my own words): ...
        - Resolution criteria: ...
        - Time horizon: ...
        - Status quo / current known state (if any): ...

        ## 2. Canonical Data
        ### 2.1 Core Facts
        - Fact 1: ...
        - Fact 2: ...
        - Fact 3: ...
        (Include 5–10 of the most important, non-overlapping facts.)

        ### 2.2 Current Competition State (if applicable)
        Only include this subsection if the question is about a leaderboard, competition, poll, market ranking, or "who will be #1".

        - Current leader: ...
        - Entity of interest current rank: ...
        - Score/metric: ...
        - Gap to leader (if known): ...
        - Last update timestamp (if known): ...
        - Source(s): ...
        (If a field cannot be found, write 'Unknown' and explain briefly.)

        ## 3. Outside View (Descriptive)
        - Relevant reference classes:
          - RC1: ...
          - RC2: ...
        - Historical patterns:
          - Pattern 1: ...
          - Pattern 2: ...
        - Qualitative notes on base rates (no numeric probabilities): ...

        ## 4. Inside View (Descriptive)
        - Current context and major recent developments:
          - ...
        - Mechanisms / gates that drive the outcome:
          - Gate 1: ...
          - Gate 2: ...
          - Gate 3: ...
        - Structural constraints (time left, resource limits, regulatory constraints, etc.): ...

        ## 5. Key Uncertainties
        - U1: ...
        - U2: ...
        - U3: ...

        Rules:
        - Use only information supported by the documents.
        - If a value is not clearly given, do NOT invent a number; mark it as 'Unknown' and describe qualitatively.
        - You MUST include all sections 1 through 5, even if some are short.
        - Keep Canonical Data concise so you have room for sections 3–5.
        - Do not output anything outside this template.

        Question: {question_text}
        Today: {today_str}

        Documents:
        {context_text[:50000]}  # Truncate if too long
    """)
    
    try:
        response_text = await call_openrouter_llm(
            prompt=prompt,
            model=llm_model,
            temperature=0.3,  # Low temperature for factual extraction
            max_tokens=50000,  # High limit to accommodate reasoning models and long memos
            usage_label="research",
        )
        memo = response_text.strip()

        # Ensure required sections exist; append placeholders if missing
        required_sections = {
            "## 1. Triage": (
                "## 1. Triage\n"
                "- Question (in my own words): Unknown\n"
                "- Resolution criteria: Unknown\n"
                "- Time horizon: Unknown\n"
                "- Status quo / current known state (if any): Unknown\n"
            ),
            "## 2. Canonical Data": (
                "## 2. Canonical Data\n"
                "### 2.1 Core Facts\n"
                "- Fact: Unknown\n"
            ),
            "## 3. Outside View (Descriptive)": (
                "## 3. Outside View (Descriptive)\n"
                "- Relevant reference classes:\n"
                "  - RC: Unknown\n"
                "- Historical patterns:\n"
                "  - Pattern: Unknown\n"
                "- Qualitative notes on base rates (no numeric probabilities): Unknown\n"
            ),
            "## 4. Inside View (Descriptive)": (
                "## 4. Inside View (Descriptive)\n"
                "- Current context and major recent developments:\n"
                "  - Unknown\n"
                "- Mechanisms / gates that drive the outcome:\n"
                "  - Gate: Unknown\n"
                "- Structural constraints (time left, resource limits, regulatory constraints, etc.): Unknown\n"
            ),
            "## 5. Key Uncertainties": (
                "## 5. Key Uncertainties\n"
                "- U1: Unknown\n"
            ),
        }

        for marker, placeholder in required_sections.items():
            if marker not in memo:
                memo = f"{memo.rstrip()}\n\n{placeholder}"

        # Generic check for missing direct measurement on status-like questions.
        # If the memo explicitly reports the current leader as Unknown, mark this
        # so forecasters can treat the situation as low-information.
        direct_status_fail = False
        for line in memo.splitlines():
            if line.strip().startswith("Current leader:"):
                if "Unknown" in line:
                    direct_status_fail = True
                    break
        if direct_status_fail:
            memo = (
                memo.rstrip()
                + "\n\nRESEARCH FAILURE: direct measurement of current status missing.\n"
            )

        return memo

    except Exception as e:
        logger.error(f"Research synthesis failed: {e}")
        return (
            "# RESEARCH MEMO\n"
            "## 1. Triage\n"
            f"- Question (in my own words): {question_text}\n"
            f"- Resolution criteria: Research failed: {e}\n"
            "- Time horizon: Unknown\n"
            "- Status quo / current known state (if any): Unknown\n"
        )
