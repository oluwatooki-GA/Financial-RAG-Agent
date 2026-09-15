import uuid
from datetime import date, datetime
from typing import Optional

from sqlmodel import Field, SQLModel, UniqueConstraint


class Company(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    cik: str = Field(index=True, unique=True)
    ticker: Optional[str] = None
    name: str
    sic: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Filing(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    company_id: uuid.UUID = Field(foreign_key="company.id", index=True)
    accession_number: str = Field(unique=True)
    form_type: str = Field(index=True)
    filing_date: date
    period_of_report: Optional[date] = None
    primary_document_filename: str
    source_url: str
    local_raw_path: Optional[str] = None
    ingestion_status: str = Field(default="pending")
    chunk_count: int = Field(default=0)
    ingested_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("filing_id", "chunk_index"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    filing_id: uuid.UUID = Field(foreign_key="filing.id", index=True)
    chunk_index: int
    part_label: Optional[str] = None
    item_label: Optional[str] = Field(default=None, index=True)
    item_heading: Optional[str] = None
    section_path: Optional[str] = None
    text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    token_count: Optional[int] = None
    embedding_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
