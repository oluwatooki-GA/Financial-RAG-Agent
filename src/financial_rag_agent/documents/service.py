from sqlmodel import select

from financial_rag_agent.core import Company, Filing, get_session
from financial_rag_agent.discovery.registry import discover_documents
from financial_rag_agent.documents.downloader import DownloadError, download_pdf, save_downloaded_pdf
from financial_rag_agent.documents.registry import get_or_register_document


class DiscoveryError(Exception):
    """No usable document could be found and downloaded for this company."""


def _get_or_create_company(session, company_name: str, cik: str | None) -> Company:
    if cik:
        existing = session.exec(select(Company).where(Company.cik == cik)).first()
    else:
        existing = session.exec(select(Company).where(Company.name == company_name)).first()
    if existing:
        return existing

    company = Company(name=company_name, cik=cik)
    session.add(company)
    session.commit()
    session.refresh(company)
    return company


def discover_and_register_document(company_name: str, cik: str | None = None) -> tuple[Filing, bool]:
    """The Phase 4 runtime-discovery entry point: search → download →
    validate → hash → dedup-or-register, wiring discovery/registry.py,
    documents/downloader.py, and documents/registry.py into one flow.
    Returns (document, created) — created=False means this exact content
    was already known (by hash) and nothing new was downloaded to disk.

    Tries each discovered candidate in turn so one dead link doesn't fail
    the whole search (per PROJECT_BUILD_PROMPT.md's "handle failures
    gracefully" principle).

    Deliberately stops at "downloaded and registered", not "chunked and
    embedded": the existing parser/chunker are SEC-HTML-only, and there is
    no PDF-aware equivalent yet. A registered document's ingestion_status
    stays "downloaded" rather than "complete" — an honest signal that
    it isn't retrievable yet, not a silent claim that it is. Building the
    PDF parsing/chunking path is future work, not something to fake here.
    """
    candidates = discover_documents(company_name, cik=cik)
    if not candidates:
        raise DiscoveryError(f"No document found for {company_name!r}")

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            content, file_hash = download_pdf(candidate.source_url)
        except DownloadError as exc:
            last_error = exc
            continue

        with get_session() as session:
            company = _get_or_create_company(session, company_name, cik)
            document, created = get_or_register_document(
                session,
                company_id=company.id,
                file_hash=file_hash,
                source_url=candidate.source_url,
                source_type=candidate.source_type,
                document_type=candidate.document_type,
                title=candidate.title,
                fiscal_year=candidate.fiscal_year,
                primary_document_filename=candidate.source_url.rsplit("/", 1)[-1] or "document.pdf",
            )
            if created:
                local_path = save_downloaded_pdf(content, file_hash)
                document.local_raw_path = str(local_path)
                document.ingestion_status = "downloaded"
                session.add(document)
                session.commit()
                session.refresh(document)

            return document, created

    raise DiscoveryError(
        f"Found candidates for {company_name!r} but none downloaded as a valid PDF: {last_error}"
    )
