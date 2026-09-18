import time
from dataclasses import dataclass

from ddgs import DDGS

from financial_rag_agent.core.config import get_settings

_PRIMARY_BACKEND = "duckduckgo"
_FALLBACK_BACKEND = "bing"
_RETRY_DELAY_SECONDS = 1.0


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


def search_web(query: str, max_results: int | None = None) -> list[WebSearchResult]:
    """Real web search for current news/context outside the filings. Free
    search backends are genuinely flaky (confirmed live: duckduckgo and its
    bing fallback both failed transiently in the same request) — retries
    the primary backend once after a short delay before falling back, and
    lets a final failure propagate as ddgs.exceptions.DDGSException, which
    main.py maps to a 502 rather than a raw 500."""
    limit = max_results if max_results is not None else get_settings().web_search_max_results

    try:
        raw_results = DDGS().text(query, max_results=limit, backend=_PRIMARY_BACKEND)
    except Exception:
        time.sleep(_RETRY_DELAY_SECONDS)
        try:
            raw_results = DDGS().text(query, max_results=limit, backend=_PRIMARY_BACKEND)
        except Exception:
            raw_results = DDGS().text(query, max_results=limit, backend=_FALLBACK_BACKEND)

    return [
        WebSearchResult(
            title=r.get("title", ""),
            url=r.get("href", ""),
            snippet=r.get("body", ""),
        )
        for r in raw_results
    ]
