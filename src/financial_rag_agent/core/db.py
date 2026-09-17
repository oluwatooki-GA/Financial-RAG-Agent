from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

import financial_rag_agent.core.models  # noqa: F401  (registers tables with SQLModel.metadata)
from financial_rag_agent.core.config import get_settings

_settings = get_settings()
engine = create_engine(_settings.database_url)


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
