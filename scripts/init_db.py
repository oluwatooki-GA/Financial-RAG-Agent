import _bootstrap  # noqa: F401
from financial_rag_agent.db import init_db

if __name__ == "__main__":
    init_db()
    print("Database initialized: pgvector extension enabled, tables created.")
