from financial_rag_agent.discovery.schemas import DiscoveredDocument


class NGXSource:
    """Not implemented — verified honestly, not assumed. Live-checked this
    session: ngxgroup.com resolves, but no documented, stable public API
    or reliable listing endpoint for company disclosures was found (unlike
    SEC EDGAR's documented JSON endpoints). Building a scraper against an
    unverified page structure would be fragile and could silently return
    wrong or stale results, which is worse than admitting the gap
    (PROJECT_BUILD_PROMPT.md design constraint #11).

    Nigerian-company discovery currently falls through to WebSource in
    the registry's priority order until this is built and verified
    against real NGX pages."""

    source_type = "ngx"

    def search_documents(self, company_name: str, cik: str | None = None) -> list[DiscoveredDocument]:
        raise NotImplementedError(
            "NGX discovery is not implemented: no verified, reliable public "
            "NGX disclosures endpoint has been confirmed."
        )
