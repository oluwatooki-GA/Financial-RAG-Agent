import requests

from financial_rag_agent.discovery.ngx_source import NGXSource
from financial_rag_agent.discovery.schemas import DiscoveredDocument
from financial_rag_agent.discovery.sec_source import SECSource
from financial_rag_agent.discovery.web_source import WebSource

# Priority order per PROJECT_BUILD_PROMPT.md: official regulatory filing ->
# official exchange disclosure -> company IR -> other reliable source.
# There's no separate CompanyIRSource yet (no generic way to find "the"
# IR page for an arbitrary company without a search step, which WebSource
# already provides) — see discovery/web_source.py.
_SOURCES = [SECSource(), NGXSource(), WebSource()]


def discover_documents(company_name: str, cik: str | None = None) -> list[DiscoveredDocument]:
    """Tries each registered source in priority order and returns the
    first non-empty result. A source that isn't implemented yet (NGX) or
    that genuinely finds nothing for this company doesn't abort the whole
    search — it just falls through to the next source, so an unverified
    source degrades gracefully instead of blocking discovery."""
    for source in _SOURCES:
        try:
            results = source.search_documents(company_name, cik=cik)
        except (NotImplementedError, ValueError, requests.RequestException):
            continue
        if results:
            return results
    return []
