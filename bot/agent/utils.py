import os
import logging
import asyncio
import httpx
import json
from typing import Optional, Dict

# Set up logging
logger = logging.getLogger(__name__)

# Global token usage tracker (per run / process)
TOKEN_USAGE: Dict[str, Dict[str, int]] = {}

def reset_token_usage() -> None:
    """Reset aggregated token usage (e.g. at start of a question pipeline)."""
    global TOKEN_USAGE
    TOKEN_USAGE = {}

def record_token_usage(label: str, prompt: int, completion: int, total: int) -> None:
    """Aggregate token usage under a human-readable label."""
    global TOKEN_USAGE
    agg = TOKEN_USAGE.setdefault(label, {"prompt": 0, "completion": 0, "total": 0})
    agg["prompt"] += prompt
    agg["completion"] += completion
    agg["total"] += total

def get_token_usage() -> Dict[str, Dict[str, int]]:
    """Return aggregated token usage snapshot."""
    return TOKEN_USAGE.copy()

def clean_indents(text: str) -> str:
    """Remove common indentation from a multi-line string."""
    lines = text.split('\n')
    if not lines:
        return text
    
    # Find minimum indentation of non-empty lines
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
            
    if min_indent == float('inf'):
        return text
        
    # Remove indentation
    cleaned_lines = []
    for line in lines:
        if len(line) >= min_indent:
            cleaned_lines.append(line[min_indent:])
        else:
            cleaned_lines.append(line)
            
    return '\n'.join(cleaned_lines).strip()


def extract_search_queries(response_text: str) -> list[str]:
    """Extract clean search queries from LLM response text.
    
    Filters out:
    - Reasoning text with markdown bold formatting (**...**)
    - Meta-commentary about needing information
    - Lines that are too long to be queries (>150 chars)
    - Lines starting with common reasoning prefixes
    
    Args:
        response_text: Raw LLM response asking for more research
        
    Returns:
        List of clean, valid search queries
    """
    queries = []
    
    # Reasoning prefixes to skip
    skip_prefixes = [
        "i need", "i'm ", "i am ", "i think", "i'll ", "i will",
        "we need", "we should", "let me", "deciding", "evaluating",
        "considering", "determining", "gathering", "identifying",
        "the ", "this ", "since ", "because ", "so,", "yes,", "no,",
        "**", "##", "#",
    ]
    
    # Skip patterns (regex-like checks)
    skip_contains = [
        "**",  # Bold markdown = reasoning, not a query
        "need to", "should ", "would ", "could ",
        "information is", "more info", "enough info",
        "suggest", "provide", "specific web search",
    ]
    
    for line in response_text.split('\n'):
        line = line.strip()
        
        # Skip empty or too short
        if not line or len(line) < 15:
            continue
            
        # Skip if too long (likely reasoning, not a query)
        if len(line) > 150:
            continue
        
        # Strip leading bullet/dash/number
        if line.startswith('- '):
            line = line[2:].strip()
        elif line.startswith('* '):
            line = line[2:].strip()
        elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
            line = line[2:].strip()
        elif len(line) > 3 and line[0].isdigit() and line[1].isdigit() and line[2] in '.):':
            line = line[3:].strip()
            
        # Strip quotes
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        if line.startswith("'") and line.endswith("'"):
            line = line[1:-1].strip()
            
        # Skip if starts with reasoning prefix
        line_lower = line.lower()
        if any(line_lower.startswith(prefix) for prefix in skip_prefixes):
            continue
            
        # Skip if contains reasoning patterns
        if any(pattern in line_lower for pattern in skip_contains):
            continue
            
        # Skip if it's mostly punctuation or formatting
        alpha_ratio = sum(1 for c in line if c.isalpha()) / max(len(line), 1)
        if alpha_ratio < 0.5:
            continue
            
        # Valid query candidate
        if len(line) >= 15:
            queries.append(line)
    
    # Return at most 3 queries
    return queries[:3]


def extract_market_probabilities(text: str) -> list[dict]:
    """Extract prediction market/forecast probabilities from research text.
    
    Looks for patterns like:
    - "Metaculus community: 7%"
    - "Manifold: 15%"  
    - "Polymarket odds: 0.12"
    - "community prediction is 8%"
    
    Returns:
        List of dicts with 'source' and 'probability' keys
    """
    import re
    
    priors = []
    
    # Common patterns for probabilities
    patterns = [
        # "Metaculus: 7%" or "community: 7%"
        r'(?:metaculus|manifold|polymarket|predictit|community|crowd|forecast)\s*(?:prediction|odds|probability|forecast)?[:\s]+(\d+(?:\.\d+)?)\s*%',
        # "7% on Metaculus"
        r'(\d+(?:\.\d+)?)\s*%\s+(?:on\s+)?(?:metaculus|manifold|polymarket)',
        # "probability of 0.07" near market names
        r'(?:metaculus|manifold|polymarket).*?(?:probability|odds)[:\s]+(?:of\s+)?(\d*\.\d+)',
        # "0.07 probability" after market name
        r'(?:metaculus|manifold|polymarket).*?(\d*\.\d+)\s*(?:probability|odds)',
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        for match in re.finditer(pattern, text_lower):
            try:
                prob_str = match.group(1)
                prob = float(prob_str)
                
                # Convert percentage to decimal if needed
                if prob > 1:
                    prob = prob / 100
                    
                # Sanity check
                if 0 < prob < 1:
                    # Try to identify source
                    source = "unknown"
                    context = text_lower[max(0, match.start()-50):match.end()+50]
                    if "metaculus" in context:
                        source = "metaculus"
                    elif "manifold" in context:
                        source = "manifold"
                    elif "polymarket" in context:
                        source = "polymarket"
                    elif "community" in context:
                        source = "community"
                        
                    priors.append({"source": source, "probability": prob})
            except (ValueError, IndexError):
                continue
                
    # Deduplicate by source, keeping first found
    seen_sources = set()
    unique_priors = []
    for p in priors:
        if p["source"] not in seen_sources:
            seen_sources.add(p["source"])
            unique_priors.append(p)
            
    return unique_priors


def extract_json_from_response(response_text: str) -> dict:
    """Robust JSON extraction from LLM responses.
    
    Handles:
    - Markdown code fences (```json, ```)
    - Extra commentary before/after JSON
    - Malformed responses
    
    Args:
        response_text: Raw LLM response
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If JSON cannot be extracted
    """
    # Clean up markdown code blocks
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        logger.warning(f"Direct JSON parse failed, attempting substring extraction")
        logger.debug(f"Raw response (first 500 chars): {response_text[:500]}")
        
        # Try to find JSON object boundaries
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_substring = text[start:end]
            return json.loads(json_substring)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"JSON extraction failed completely")
            logger.error(f"Full response: {response_text[:1000]}")
            raise first_error  # Raise the original error with full context

async def call_openrouter_llm(
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    usage_label: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Call OpenRouter API for LLM inference.

    Args:
        prompt: The prompt to send
        model: OpenRouter model name (e.g., "openai/gpt-5-mini")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        reasoning_effort: "low", "medium", or "high" (for reasoning models)

    Returns:
        Response text from the model

    Raises:
        Exception: If API call fails or OPENROUTER_API_KEY not set
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Optional: Add HTTP-Referer and X-Title for better rate limits
    # Note: HTTP headers must be latin-1 encodable, so sanitize unicode characters
    site_url = os.getenv("OPENROUTER_SITE_URL", "https://github.com/forecasting-bot")
    site_name = os.getenv("OPENROUTER_SITE_NAME", "Forecasting Bot")

    # Sanitize headers: replace common unicode chars with ASCII equivalents
    def sanitize_header(value: str) -> str:
        """Replace unicode characters with ASCII equivalents for HTTP headers."""
        replacements = {
            '\u2013': '-',  # en dash
            '\u2014': '-',  # em dash
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u201C': '"',  # left double quote
            '\u201D': '"',  # right double quote
        }
        for unicode_char, ascii_char in replacements.items():
            value = value.replace(unicode_char, ascii_char)
        # Remove any remaining non-ASCII characters
        return value.encode('ascii', 'ignore').decode('ascii')

    headers["HTTP-Referer"] = sanitize_header(site_url)
    headers["X-Title"] = sanitize_header(site_name)


    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Add reasoning parameters if provided
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    # Legacy check for o1/o3 models if reasoning_effort not explicitly passed
    elif "o1" in model or "o3" in model:
        # Default to medium for o-series if not specified
        # payload["reasoning_effort"] = "medium" 
        pass

    logger.info(f"Calling OpenRouter with model: {model}")

    # Use httpx for proper async support (avoids memory corruption from concurrent requests)
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(url, headers=headers, json=payload)

    # Better error handling with response body
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        error_body = ""
        try:
            error_body = response.json()
            logger.error(f"OpenRouter API error: {error_body}")
        except Exception:
            error_body = response.text
            logger.error(f"OpenRouter API error (raw): {error_body}")
        raise

    result = response.json()

    # Handle OpenRouter response format
    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0].get("message", {}) or {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning") or "").strip()

        usage = result.get("usage", {})
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens))
            logger.info(
                f"OpenRouter usage for {model}: prompt={prompt_tokens}, "
                f"completion={completion_tokens}, total={total_tokens}"
            )
            if usage_label:
                record_token_usage(usage_label, prompt_tokens, completion_tokens, total_tokens)

        # Fallback: if content is empty but the model returned non-empty
        # hidden reasoning, use the reasoning as the visible content.
        if not content:
            if reasoning:
                logger.warning(
                    f"Model {model} returned empty content but non-empty reasoning; "
                    "using reasoning as fallback content."
                )
                content = reasoning
            else:
                logger.error(f"Empty response from OpenRouter model {model}")
                logger.error(f"Full response: {result}")
                raise Exception(
                    f"Empty response from model {model}. Check if model supports the requested parameters."
                )
        logger.info(f"OpenRouter response received ({len(content)} chars)")
        return content
    else:
        raise Exception(f"Unexpected OpenRouter response format: {result}")


class SerperClient:
    """Client for Serper.dev search API."""

    def __init__(self, api_key: str | None = None, max_results: int = 8):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not set")
        self.max_results = max_results

    async def search(self, query: str, num_results: int | None = None) -> list[dict]:
        """Execute a Serper search and return normalized results."""
        n = num_results or self.max_results
        payload = {"q": query, "num": n, "gl": "us", "hl": "en"}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for rank, item in enumerate(data.get("organic", [])[:n]):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "content": item.get("snippet", ""),
                "score": 1.0 / (rank + 1),
            })
        return results


class SonarClient:
    """
    Client for Perplexity Sonar via OpenRouter. Used as a research
    assistant, not as the main forecaster.

    Usage pattern: given a query, return a concise answer + list of cited URLs.
    """

    def __init__(self, model: str = "perplexity/sonar"):
        """
        Initialize SonarClient using OpenRouter.

        Args:
            model: OpenRouter model name (default: "perplexity/sonar")
                  Options: "perplexity/sonar", "perplexity/sonar-pro", etc.
        """
        # Check that OPENROUTER_API_KEY is set
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set for SonarClient")
        self.model = model

    async def answer_with_citations(self, query: str, max_tokens: int = 600) -> dict:
        """
        Call Perplexity Sonar via OpenRouter with `query` and return a dict:
            {
                "answer": str,         # short synthesis
                "citations": [urls...] # urls mentioned / cited
            }

        Citations are extracted from the response text using regex.
        """
        try:
            answer = await call_openrouter_llm(
                prompt=query,
                model=self.model,
                temperature=0.1,
                max_tokens=max_tokens,
                usage_label="sonar",
            )
        except Exception as e:
            logger.error(f"SonarClient call failed: {e}")
            answer = ""

        # Extract URLs from the answer text
        citations: list[str] = []
        if answer:
            try:
                import re as _re
                # Match http/https URLs
                citations = _re.findall(r"https?://[^\s\)]+", answer)
            except Exception:
                citations = []

        return {"answer": answer, "citations": citations}


class TavilyClient:
    """Client for Tavily AI-optimized search API."""

    def __init__(self, api_key: str, max_results: int = 5, search_depth: str = "advanced"):
        """Initialize Tavily client.

        Args:
            api_key: Tavily API key
            max_results: Maximum results per query (default 5)
            search_depth: "basic" or "advanced" (default "advanced")
        """
        self.api_key = api_key
        self.max_results = max_results
        self.search_depth = search_depth
        self.search_url = "https://api.tavily.com/search"
        self.extract_url = "https://api.tavily.com/extract"
        self.crawl_url = "https://api.tavily.com/crawl"

    async def search(self, query: str, **kwargs) -> list[dict]:
        """Execute search query via Tavily API.

        Args:
            query: Search query string
            **kwargs: Additional parameters for Tavily API (e.g., topic, days, include_domains)

        Returns:
            List of search results with title, url, content, score
        """
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": kwargs.get("max_results", self.max_results),
            "search_depth": kwargs.get("search_depth", self.search_depth),
            "include_answer": False,
            "include_raw_content": False,
        }
        # Update payload with any additional arguments
        payload.update(kwargs)

        logger.info(f"Tavily search: {query} (params: {kwargs})")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.search_url, json=payload)
        response.raise_for_status()

        result = response.json()
        results = result.get("results", [])
        logger.info(f"Tavily retrieved {len(results)} results")
        return results

    async def search_multiple(self, queries: list[str]) -> list[dict]:
        """Execute multiple search queries and aggregate results.

        Args:
            queries: List of search query strings

        Returns:
            Aggregated list of all search results
        """
        all_results = []
        for query in queries:
            try:
                results = await self.search(query)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Tavily search failed for '{query}': {e}")

        logger.info(f"Total Tavily results from {len(queries)} queries: {len(all_results)}")
        return all_results

    async def extract(
        self,
        urls: list[str],
        extract_depth: str = "basic",
        include_images: bool = False,
        include_favicon: bool = False,
        fmt: str = "markdown",
    ) -> dict:
        """
        Use Tavily Extract to fetch full content for given URLs.

        Returns:
            dict with keys:
                - results: list[{url, raw_content, images?}]
                - failed_results: list[{url, error}]
        """
        if not urls:
            return {"results": [], "failed_results": []}

        payload = {
            "api_key": self.api_key,
            "urls": urls,
            "extract_depth": extract_depth,
            "include_images": include_images,
            "include_favicon": include_favicon,
            "format": fmt,
        }

        logger.info(f"Tavily extract: {len(urls)} urls")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.extract_url, json=payload)
        response.raise_for_status()
        return response.json()

    async def crawl(
        self,
        url: str,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
        instructions: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Use Tavily Crawl to traverse a site from a base URL.

        Returns:
            dict with keys:
                - base_url
                - results: list[{url, raw_content}]
        """
        payload = {
            "api_key": self.api_key,
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
        }
        if instructions:
            payload["instructions"] = instructions
        payload.update(kwargs)

        logger.info(f"Tavily crawl: {url} (max_depth={max_depth}, limit={limit})")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self.crawl_url, json=payload)
        response.raise_for_status()
        return response.json()


class ExaClient:
    """Client for Exa.ai search API - AI-native search with full content retrieval."""

    def __init__(self, api_key: str | None = None, max_results: int = 10):
        """Initialize Exa client.

        Args:
            api_key: Exa API key (or set EXA_API_KEY env var)
            max_results: Maximum results per query (default 10)
        """
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY not set")
        self.max_results = max_results
        self.base_url = "https://api.exa.ai"

    async def search(
        self,
        query: str,
        num_results: int | None = None,
        use_autoprompt: bool = True,
        include_text: bool = True,
        start_published_date: str | None = None,
    ) -> list[dict]:
        """Execute search query via Exa API with content retrieval.

        Args:
            query: Search query string
            num_results: Number of results (default uses max_results)
            use_autoprompt: Let Exa optimize the query (default True)
            include_text: Include full text content (default True)
            start_published_date: Filter for recent content (ISO format, e.g. "2024-01-01")

        Returns:
            List of search results with title, url, text (full content), score
        """
        n = num_results or self.max_results
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Use search_and_contents endpoint to get full text
        endpoint = f"{self.base_url}/search"
        
        payload = {
            "query": query,
            "numResults": n,
            "useAutoprompt": use_autoprompt,
            "contents": {
                "text": True,  # Get full text content
            }
        }
        
        if start_published_date:
            payload["startPublishedDate"] = start_published_date

        logger.info(f"Exa search: '{query}' (num_results={n})")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Exa search failed for '{query}': {e}")
            raise

        results = []
        for item in data.get("results", []):
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("text", "") or item.get("snippet", ""),  # Full text or snippet
                "score": item.get("score", 0.0),
                "published_date": item.get("publishedDate", ""),
            }
            # Also include raw_content for compatibility with existing code
            if item.get("text"):
                result["raw_content"] = item.get("text")
            results.append(result)

        logger.info(f"Exa retrieved {len(results)} results")
        return results

    async def find_similar(self, url: str, num_results: int | None = None) -> list[dict]:
        """Find pages similar to a given URL.

        Args:
            url: URL to find similar pages for
            num_results: Number of results

        Returns:
            List of similar pages with title, url, text, score
        """
        n = num_results or self.max_results

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "url": url,
            "numResults": n,
            "contents": {
                "text": True,
            }
        }

        logger.info(f"Exa find_similar: '{url}'")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/findSimilar", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("text", ""),
                "score": item.get("score", 0.0),
            })

        return results

class BraveClient:
    """Client for Brave Search API."""

    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY not set")
        self.max_results = max_results

    async def search(self, query: str, num_results: int | None = None) -> list[dict]:
        """Execute a Brave search and return normalized results."""
        n = num_results or self.max_results
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        params = {"q": query, "count": n}
        
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        # Brave structure: data['web']['results'] -> list of items
        web_results = data.get("web", {}).get("results", [])
        
        for rank, item in enumerate(web_results[:n]):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("description", "") or item.get("snippet", ""),
                "score": 1.0 / (rank + 1),
            })
        return results


class ScrapingDogClient:
    """Client for ScrapingDog API (for dynamic/JS scraping)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("SCRAPINGDOG_API_KEY")
        if not self.api_key:
            # Don't raise error here, just log warning if used
            logger.warning("SCRAPINGDOG_API_KEY not set")

    async def scrape(self, url: str, dynamic: bool = True) -> str:
        """Scrape a URL using ScrapingDog."""
        if not self.api_key:
            return ""

        params = {
            "api_key": self.api_key,
            "url": url,
            "dynamic": str(dynamic).lower(),
        }
        
        logger.info(f"ScrapingDog: fetching {url} (dynamic={dynamic})")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get("https://api.scrapingdog.com/scrape", params=params)
            if resp.status_code == 200:
                return resp.text
            else:
                logger.error(f"ScrapingDog failed for {url}: {resp.status_code} {resp.text}")
                return ""


class LangSearchClient:
    """Client for LangSearch Web Search API.
    
    LangSearch provides AI-optimized web search with long text summaries,
    hybrid search (keyword + vector), and Bing-compatible response format.
    
    Docs: https://langsearch.com
    """

    def __init__(self, api_key: str | None = None, max_results: int = 5):
        """Initialize LangSearch client.
        
        Args:
            api_key: LangSearch API key (or set LANGSEARCH_API_KEY env var)
            max_results: Maximum results per query (default 5, max 10)
        """
        self.api_key = api_key or os.getenv("LANGSEARCH_API_KEY")
        if not self.api_key:
            raise ValueError("LANGSEARCH_API_KEY not set")
        self.max_results = min(max_results, 10)  # API max is 10
        self.search_url = "https://api.langsearch.com/v1/web-search"

    async def search(
        self,
        query: str,
        num_results: int | None = None,
        freshness: str | None = None,
        include_summary: bool = True,
    ) -> list[dict]:
        """Execute a LangSearch web search and return normalized results.
        
        Args:
            query: Search query string
            num_results: Number of results (1-10, default uses max_results)
            freshness: Time range filter - "noLimit", "day", "week", "month", "year"
            include_summary: Include long text summaries from source pages
            
        Returns:
            List of search results with title, url, content, score, and optional summary
        """
        n = min(num_results or self.max_results, 10)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "query": query,
            "count": n,
            "summary": include_summary,
        }
        
        if freshness:
            payload["freshness"] = freshness
        
        logger.info(f"LangSearch: searching '{query}' (count={n}, summary={include_summary})")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.search_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        
        results = []
        # LangSearch returns Bing-compatible format with webPages.value
        web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
        
        for rank, item in enumerate(web_pages[:n]):
            result = {
                "title": item.get("name", "") or item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("snippet", ""),
                "score": 1.0 / (rank + 1),
            }
            # Include summary if available (LangSearch's long text summary feature)
            if item.get("summary"):
                result["summary"] = item.get("summary")
            results.append(result)
        
        logger.info(f"LangSearch retrieved {len(results)} results")
        return results

    async def search_with_summaries(self, query: str, num_results: int | None = None) -> list[dict]:
        """Search with summaries enabled - convenience method.
        
        Returns results with 'summary' field containing extracted page content.
        """
        return await self.search(query, num_results=num_results, include_summary=True)
