from datetime import date, datetime

from sqlmodel import select

from financial_rag_agent.core import Chunk, Company, Filing, get_session
from financial_rag_agent.ingestion.chunker import SECFilingChunker
from financial_rag_agent.ingestion.edgar_client import FilingRef, fetch_filing_html, get_latest_10k
from financial_rag_agent.ingestion.indexing import persist_and_embed
from financial_rag_agent.ingestion.parser import parse_filing_html


def _get_or_create_company(session, filing_ref: FilingRef) -> Company:
    company = session.exec(select(Company).where(Company.cik == filing_ref.cik)).first()
    if company:
        return company

    company = Company(
        cik=filing_ref.cik,
        ticker=filing_ref.ticker,
        name=filing_ref.company_name,
        sic=filing_ref.sic,
    )
    session.add(company)
    session.commit()
    session.refresh(company)
    return company


def _get_or_create_filing(session, company: Company, filing_ref: FilingRef, local_path: str) -> Filing:
    existing = session.exec(
        select(Filing).where(Filing.accession_number == filing_ref.accession_number)
    ).first()
    if existing:
        return existing

    filing = Filing(
        company_id=company.id,
        accession_number=filing_ref.accession_number,
        form_type=filing_ref.form_type,
        document_type=filing_ref.form_type,
        source_type="sec",
        filing_date=date.fromisoformat(filing_ref.filing_date),
        period_of_report=date.fromisoformat(filing_ref.period_of_report)
        if filing_ref.period_of_report
        else None,
        primary_document_filename=filing_ref.primary_document,
        source_url=filing_ref.source_url,
        local_raw_path=local_path,
        downloaded_at=datetime.utcnow(),
    )
    session.add(filing)
    session.commit()
    session.refresh(filing)
    return filing


def ingest_filing(cik: str) -> Filing:
    filing_ref = get_latest_10k(cik)
    raw_path = fetch_filing_html(filing_ref)

    with get_session() as session:
        company = _get_or_create_company(session, filing_ref)
        filing = _get_or_create_filing(session, company, filing_ref, str(raw_path))

        drafts = []
        already_chunked = session.exec(select(Chunk.id).where(Chunk.filing_id == filing.id).limit(1)).first()
        if not already_chunked:
            blocks = parse_filing_html(raw_path)
            drafts = SECFilingChunker().chunk(blocks)

        persist_and_embed(session, filing, company.id, drafts)

        return filing
