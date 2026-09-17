from uuid import UUID

from pydantic import BaseModel


class IngestRequest(BaseModel):
    cik: str


class IngestResponse(BaseModel):
    accession_number: str
    chunk_count: int
    ingestion_status: str


class CitationSentenceResponse(BaseModel):
    text: str
    char_start: int
    char_end: int
    score: float


class RetrievedChunkResponse(BaseModel):
    chunk_id: UUID
    score: float
    text: str
    item_label: str | None
    item_heading: str | None
    filing_accession_number: str
    modality: str
    table_data: list[list[str]] | None
    citation_sentences: list[CitationSentenceResponse]


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunkResponse]
