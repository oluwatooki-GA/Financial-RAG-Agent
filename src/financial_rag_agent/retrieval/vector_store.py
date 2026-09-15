import re
from functools import lru_cache

from langchain_postgres import PGVector

from financial_rag_agent.config import get_settings
from financial_rag_agent.embeddings.factory import get_embeddings_client

BASE_COLLECTION_NAME = "sec_filing_chunks"


def _collection_name(settings) -> str:
    # Different embedding providers/models produce incompatible vector spaces.
    # Keying the collection name on both means switching EMBEDDING_PROVIDER in
    # .env can never silently mix vectors from two different models.
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", settings.embedding_model).strip("_")
    return f"{BASE_COLLECTION_NAME}__{settings.embedding_provider}__{model_slug}"


@lru_cache
def get_vector_store() -> PGVector:
    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings_client(),
        collection_name=_collection_name(settings),
        connection=settings.database_url,
        use_jsonb=True,
    )
