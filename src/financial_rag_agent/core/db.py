from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

import financial_rag_agent.core.models  # noqa: F401  (registers tables with SQLModel.metadata)
from financial_rag_agent.core.config import get_settings

_settings = get_settings()
engine = create_engine(_settings.database_url)


_MIGRATIONS = (
    # Phase 4: generalize `filing` beyond SEC-only documents. create_all()
    # only creates missing tables, never alters existing ones, so a table
    # created under the old schema needs these run explicitly. Every
    # statement here is safe to re-run.
    "ALTER TABLE company ADD COLUMN IF NOT EXISTS country VARCHAR",
    "ALTER TABLE company ADD COLUMN IF NOT EXISTS exchange VARCHAR",
    "ALTER TABLE company ADD COLUMN IF NOT EXISTS sector VARCHAR",
    "ALTER TABLE company ADD COLUMN IF NOT EXISTS currency VARCHAR",
    "ALTER TABLE company ALTER COLUMN cik DROP NOT NULL",
    "ALTER TABLE filing ADD COLUMN IF NOT EXISTS document_type VARCHAR",
    "ALTER TABLE filing ADD COLUMN IF NOT EXISTS source_type VARCHAR NOT NULL DEFAULT 'sec'",
    "ALTER TABLE filing ADD COLUMN IF NOT EXISTS title VARCHAR",
    "ALTER TABLE filing ADD COLUMN IF NOT EXISTS fiscal_year INTEGER",
    "ALTER TABLE filing ADD COLUMN IF NOT EXISTS period VARCHAR",
    "ALTER TABLE filing ADD COLUMN IF NOT EXISTS file_hash VARCHAR",
    "ALTER TABLE filing ADD COLUMN IF NOT EXISTS downloaded_at TIMESTAMP",
    "ALTER TABLE filing ALTER COLUMN accession_number DROP NOT NULL",
    "ALTER TABLE filing ALTER COLUMN form_type DROP NOT NULL",
    "ALTER TABLE filing ALTER COLUMN filing_date DROP NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_filing_file_hash ON filing (file_hash)",
    "CREATE INDEX IF NOT EXISTS ix_filing_document_type ON filing (document_type)",
    "CREATE INDEX IF NOT EXISTS ix_filing_source_type ON filing (source_type)",
    "CREATE INDEX IF NOT EXISTS ix_filing_fiscal_year ON filing (fiscal_year)",
    "CREATE INDEX IF NOT EXISTS ix_company_country ON company (country)",
    "CREATE INDEX IF NOT EXISTS ix_company_exchange ON company (exchange)",
)


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        for statement in _MIGRATIONS:
            conn.execute(text(statement))


def get_session() -> Session:
    return Session(engine)
