from pathlib import Path

from sqlmodel import select

from financial_rag_agent.core import Chunk, Company, Filing, get_session
from financial_rag_agent.discovery.registry import discover_documents
from financial_rag_agent.documents.downloader import DownloadError, download_pdf, save_downloaded_pdf
from financial_rag_agent.documents.pdf_chunker import PDFChunker
from financial_rag_agent.documents.pdf_parser import parse_pdf
from financial_rag_agent.documents.registry import get_or_register_document
from financial_rag_agent.ingestion.indexing import persist_and_embed


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
    embedded" — download and processing are kept as two separate steps
    (a discovered document can sit "known but unprocessed" for a while,
    matching the persistent-vs-temporary distinction in
    PROJECT_BUILD_PROMPT.md). Call process_downloaded_document() on the
    returned document to actually make it retrievable.
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


def process_downloaded_document(document_id) -> Filing:
    """The "Step B" discover_and_register_document() deliberately stops
    short of: parses the downloaded PDF (page-based text + real tables,
    see documents/pdf_parser.py), chunks it (documents/pdf_chunker.py),
    and persists/embeds those chunks through the exact same shared path
    SEC ingestion uses (ingestion/indexing.py's persist_and_embed) — so a
    Phase-4-discovered document becomes retrievable through the identical
    query path as a SEC filing, not a separate one.

    Skips re-parsing/re-chunking if Chunk rows already exist for this
    document (e.g. a prior run got through parsing but failed during
    embedding) — mirrors ingest_filing()'s same cheap-existence-check
    optimization, not persist_and_embed's job since it only decides
    whether to reuse *drafts already computed this call*.
    """
    with get_session() as session:
        document = session.get(Filing, document_id)
        if document is None:
            raise ValueError(f"No document with id {document_id}")
        if document.local_raw_path is None:
            raise ValueError(f"Document {document_id} has no local file to process")

        drafts = []
        already_chunked = session.exec(select(Chunk.id).where(Chunk.filing_id == document.id).limit(1)).first()
        if not already_chunked:
            blocks = parse_pdf(Path(document.local_raw_path))
            drafts = PDFChunker().chunk(blocks)

        persist_and_embed(session, document, document.company_id, drafts)
        return document
