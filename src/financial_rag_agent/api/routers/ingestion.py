from fastapi import APIRouter

from financial_rag_agent.api.schemas import IngestRequest, IngestResponse
from financial_rag_agent.ingestion.pipeline import ingest_filing

router = APIRouter(prefix="/filings", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    filing = ingest_filing(request.cik)
    return IngestResponse(
        accession_number=filing.accession_number,
        chunk_count=filing.chunk_count,
        ingestion_status=filing.ingestion_status,
    )
