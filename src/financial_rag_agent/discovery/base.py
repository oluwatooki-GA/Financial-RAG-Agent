from typing import Protocol

from financial_rag_agent.discovery.schemas import DiscoveredDocument


class DocumentSource(Protocol):
    """The Strategy interface every discovery source implements (SEC, NGX,
    web search, and any future source — company IR, another country's
    exchange, ...). discovery/registry.py depends only on this shape, never
    on a specific source's internals, so adding a new source never means
    touching SECSource/NGXSource/WebSource or the registry's dispatch
    logic — see registry.py's module docstring for the two-step recipe."""

    source_type: str

    def search_documents(self, company_name: str, cik: str | None = None) -> list[DiscoveredDocument]: ...
