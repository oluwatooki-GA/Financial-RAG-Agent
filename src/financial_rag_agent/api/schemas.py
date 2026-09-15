from uuid import UUID

from pydantic import BaseModel


class IngestRequest(BaseModel):
    cik: str


class IngestResponse(BaseModel):
    accession_number: str
    chunk_count: int
    ingestion_status: str


class RetrievedChunkResponse(BaseModel):
    chunk_id: UUID
    score: float
    text: str
    item_label: str | None
    item_heading: str | None
    filing_accession_number: str


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunkResponse]
