import requests

from financial_rag_agent.discovery.base import DocumentSource
from financial_rag_agent.discovery.ngx_source import NGXSource
from financial_rag_agent.discovery.schemas import DiscoveredDocument
from financial_rag_agent.discovery.sec_source import SECSource
from financial_rag_agent.discovery.web_source import WebSource

# Priority order per PROJECT_BUILD_PROMPT.md: official regulatory filing ->
# official exchange disclosure -> company IR -> other reliable source.
# There's no separate CompanyIRSource yet (no generic way to find "the"
# IR page for an arbitrary company without a search step, which WebSource
# already provides) — see discovery/web_source.py.
#
# To add a new source (a real NGX document scraper later, a different
# country's exchange, a dedicated company-IR crawler, ...):
#   1. Write a class implementing discovery/base.py's DocumentSource
#      (a `source_type` string and a `search_documents(...)` method) —
#      NGXSource is a template for a source that's honestly not ready yet.
#   2. Add one line here, in priority order.
# Nothing else in this file, or in discover_documents() below, changes —
# that's the whole point of depending on the DocumentSource shape instead
# of on any specific source's internals.
_SOURCES: list[DocumentSource] = [SECSource(), NGXSource(), WebSource()]


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
