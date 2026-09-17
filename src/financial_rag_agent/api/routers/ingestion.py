import re
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException

from financial_rag_agent.core import Filing
from financial_rag_agent.ingestion.schemas import IngestRequest, IngestResponse
from financial_rag_agent.ingestion.service import ingest_filing

router = APIRouter(prefix="/filings", tags=["ingestion"])

_CIK_RE = re.compile(r"^\d{1,10}$")


def get_ingest_service() -> Callable[[str], Filing]:
    """Dependency provider for the ingestion service — routes depend on this
    injectable callable rather than importing ingest_filing directly, so
    tests can override it via app.dependency_overrides instead of
    monkeypatching the ingestion module."""
    return ingest_filing


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, ingest_service=Depends(get_ingest_service)) -> IngestResponse:
    if not _CIK_RE.match(request.cik):
        raise HTTPException(status_code=400, detail=f"Invalid CIK {request.cik!r}: expected 1-10 digits")

    filing = ingest_service(request.cik)
    return IngestResponse(
        accession_number=filing.accession_number,
        chunk_count=filing.chunk_count,
        ingestion_status=filing.ingestion_status,
    )
