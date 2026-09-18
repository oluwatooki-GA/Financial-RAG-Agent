from dataclasses import dataclass


@dataclass
class DiscoveredDocument:
    """One candidate document returned by a source, before it has been
    downloaded/hashed/validated. Not yet evidence — per
    PROJECT_BUILD_PROMPT.md's design constraint #9, a discovered document
    (especially from WebSource) is data to fetch and check, not something
    to treat as a verified fact."""

    title: str
    source_url: str
    source_type: str  # "sec" | "ngx" | "company_ir" | "web"
    document_type: str | None = None
    company_name: str | None = None
    fiscal_year: int | None = None
