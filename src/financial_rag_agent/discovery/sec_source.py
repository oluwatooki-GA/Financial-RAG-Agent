from financial_rag_agent.discovery.schemas import DiscoveredDocument
from financial_rag_agent.ingestion.edgar_client import get_latest_10k


class SECSource:
    """Real and working today: wraps the existing EDGAR client (already
    proven live for NVIDIA and Apple). Needs a CIK — EDGAR's free-text
    company-name search is a separate API not yet integrated, so a
    company with no known CIK simply isn't discoverable through this
    source (search falls through to the next one in the registry)."""

    source_type = "sec"

    def search_documents(self, company_name: str, cik: str | None = None) -> list[DiscoveredDocument]:
        if not cik:
            return []

        filing_ref = get_latest_10k(cik)
        return [
            DiscoveredDocument(
                title=f"{filing_ref.company_name} {filing_ref.form_type} ({filing_ref.filing_date})",
                source_url=filing_ref.source_url,
                source_type=self.source_type,
                document_type=filing_ref.form_type,
                company_name=filing_ref.company_name,
            )
        ]
