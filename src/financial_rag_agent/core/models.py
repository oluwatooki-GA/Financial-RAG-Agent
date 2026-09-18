import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel, UniqueConstraint


class Company(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # cik is nullable+non-unique-by-null for non-SEC (e.g. NGX) companies that
    # have no CIK; uniqueness only matters among rows that actually have one,
    # which Postgres already gives us since it treats NULLs as distinct.
    cik: Optional[str] = Field(default=None, index=True, unique=True)
    ticker: Optional[str] = None
    name: str
    sic: Optional[str] = None
    country: Optional[str] = Field(default=None, index=True)
    exchange: Optional[str] = Field(default=None, index=True)
    sector: Optional[str] = None
    # ISO 4217 code (USD, NGN, ...). Figures are never converted across
    # currencies (see PROJECT_BUILD_PROMPT.md's design constraint 12) — this
    # is what lets a future analysis step refuse to silently compare them.
    currency: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Filing(SQLModel, table=True):
    """A regulatory/company disclosure document. Originally SEC-only
    (accession_number/form_type); generalized in Phase 4 so the same table
    holds NGX/company-IR/web-discovered documents too, identified by
    file_hash (SHA-256 of the raw bytes) instead of an accession number."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    company_id: uuid.UUID = Field(foreign_key="company.id", index=True)

    # SEC-specific; None for documents discovered from other sources.
    accession_number: Optional[str] = Field(default=None, unique=True)
    form_type: Optional[str] = Field(default=None, index=True)

    # Generalized metadata, populated for every source.
    document_type: Optional[str] = Field(default=None, index=True)  # "10-K", "annual_report", ...
    source_type: str = Field(default="sec", index=True)  # sec | ngx | company_ir | regulatory | web
    title: Optional[str] = None
    fiscal_year: Optional[int] = Field(default=None, index=True)
    period: Optional[str] = None  # e.g. "FY2025", "Q3 2025"

    filing_date: Optional[date] = None
    period_of_report: Optional[date] = None
    primary_document_filename: str
    source_url: str
    local_raw_path: Optional[str] = None
    # SHA-256 hex digest of the downloaded bytes. The dedup key for anything
    # discovered at runtime — the same PDF found via two different URLs must
    # resolve to one row (see documents/registry.py).
    file_hash: Optional[str] = Field(default=None, unique=True, index=True)

    ingestion_status: str = Field(default="pending")
    chunk_count: int = Field(default=0)
    downloaded_at: Optional[datetime] = None
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
    modality: str = Field(default="text", index=True)
    table_data: Optional[list] = Field(default=None, sa_column=Column(JSON))
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    token_count: Optional[int] = None
    embedding_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
