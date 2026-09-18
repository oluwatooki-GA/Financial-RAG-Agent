from financial_rag_agent.discovery.schemas import DiscoveredDocument
from financial_rag_agent.tools.web_search import search_web


class WebSource:
    """General web search as the last-resort discovery mechanism — lowest
    priority in the registry. A hit here is a candidate URL to validate
    and download, never evidence by itself: per PROJECT_BUILD_PROMPT.md
    design constraint #9, search-result text is data, not a verified
    fact, until the underlying document is actually fetched."""

    source_type = "web"

    def search_documents(self, company_name: str, cik: str | None = None) -> list[DiscoveredDocument]:
        results = search_web(f"{company_name} annual report financial statements filetype:pdf")
        return [
            DiscoveredDocument(
                title=r.title,
                source_url=r.url,
                source_type=self.source_type,
                company_name=company_name,
            )
            for r in results
        ]
