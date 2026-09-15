from datetime import date, datetime

from sqlmodel import select

from financial_rag_agent.db import Chunk, Company, Filing, get_session
from financial_rag_agent.ingestion.chunker import SECFilingChunker
from financial_rag_agent.ingestion.edgar_client import FilingRef, fetch_filing_html, get_latest_10k
from financial_rag_agent.ingestion.parser import parse_filing_html
from financial_rag_agent.retrieval.vector_store import get_vector_store


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
        filing_date=date.fromisoformat(filing_ref.filing_date),
        period_of_report=date.fromisoformat(filing_ref.period_of_report)
        if filing_ref.period_of_report
        else None,
        primary_document_filename=filing_ref.primary_document,
        source_url=filing_ref.source_url,
        local_raw_path=local_path,
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

        if filing.ingestion_status == "complete":
            return filing

        filing.ingestion_status = "parsing"
        session.add(filing)
        session.commit()

        blocks = parse_filing_html(raw_path)
        drafts = SECFilingChunker().chunk(blocks)

        chunks: list[Chunk] = []
        for draft in drafts:
            chunk = Chunk(
                filing_id=filing.id,
                chunk_index=draft.chunk_index,
                part_label=draft.part_label,
                item_label=draft.item_label,
                item_heading=draft.item_heading,
                section_path=draft.section_path,
                text=draft.text,
                token_count=len(draft.text) // 4,
            )
            chunk.embedding_id = str(chunk.id)
            chunks.append(chunk)

        filing.ingestion_status = "embedding"
        session.add(filing)
        session.add_all(chunks)
        session.commit()
        for c in chunks:
            session.refresh(c)

        vector_store = get_vector_store()
        vector_store.add_texts(
            texts=[c.text for c in chunks],
            metadatas=[
                {
                    "chunk_id": str(c.id),
                    "filing_id": str(filing.id),
                    "company_id": str(company.id),
                    "item_label": c.item_label,
                    "item_heading": c.item_heading,
                }
                for c in chunks
            ],
            ids=[str(c.id) for c in chunks],
        )

        filing.ingestion_status = "complete"
        filing.chunk_count = len(chunks)
        filing.ingested_at = datetime.utcnow()
        session.add(filing)
        session.commit()
        session.refresh(filing)

        return filing
