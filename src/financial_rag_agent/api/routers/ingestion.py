import re

from fastapi import APIRouter, Depends, HTTPException

from financial_rag_agent.api.dependencies import get_ingest_service
from financial_rag_agent.api.schemas import IngestRequest, IngestResponse

router = APIRouter(prefix="/filings", tags=["ingestion"])

_CIK_RE = re.compile(r"^\d{1,10}$")


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
