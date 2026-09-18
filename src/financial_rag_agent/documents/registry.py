from datetime import date, datetime
from uuid import UUID

from sqlmodel import Session, select

from financial_rag_agent.core import Filing


def find_document_by_hash(session: Session, file_hash: str) -> Filing | None:
    """Real lookup against the relational table — the dedup check every
    discovery/download path must run before creating a new row."""
    return session.exec(select(Filing).where(Filing.file_hash == file_hash)).first()


def get_or_register_document(
    session: Session,
    *,
    company_id: UUID,
    file_hash: str,
    source_url: str,
    source_type: str,
    primary_document_filename: str,
    document_type: str | None = None,
    title: str | None = None,
    fiscal_year: int | None = None,
    period: str | None = None,
    filing_date: date | None = None,
    period_of_report: date | None = None,
    local_raw_path: str | None = None,
) -> tuple[Filing, bool]:
    """Registers a discovered document, or returns the existing one if this
    exact content (by file_hash) is already known. Returns (document,
    created) so callers can tell "found it, no download needed" apart from
    "just persisted a new one" without a second query.

    This is the Phase 4 document-registry entry point: it makes "the same
    PDF found through two different URLs becomes one row" (and "same
    document queried again later" -> no re-download) an enforced property
    of the data model, not a convention callers have to remember.
    """
    existing = find_document_by_hash(session, file_hash)
    if existing is not None:
        return existing, False

    document = Filing(
        company_id=company_id,
        file_hash=file_hash,
        source_url=source_url,
        source_type=source_type,
        document_type=document_type,
        title=title,
        fiscal_year=fiscal_year,
        period=period,
        filing_date=filing_date,
        period_of_report=period_of_report,
        primary_document_filename=primary_document_filename,
        local_raw_path=local_raw_path,
        downloaded_at=datetime.utcnow(),
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    return document, True
