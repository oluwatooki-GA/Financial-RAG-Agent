from pydantic import BaseModel


class IngestRequest(BaseModel):
    cik: str


class IngestResponse(BaseModel):
    accession_number: str
    chunk_count: int
    ingestion_status: str
