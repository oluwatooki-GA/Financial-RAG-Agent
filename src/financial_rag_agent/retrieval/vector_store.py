from functools import lru_cache

from langchain_postgres import PGVector

from financial_rag_agent.config import get_settings
from financial_rag_agent.embeddings.factory import get_embeddings_client

COLLECTION_NAME = "sec_filing_chunks"


@lru_cache
def get_vector_store() -> PGVector:
    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings_client(),
        collection_name=COLLECTION_NAME,
        connection=settings.database_url,
        use_jsonb=True,
    )
