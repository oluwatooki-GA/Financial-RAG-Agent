import re
import uuid
from functools import lru_cache

from langchain_postgres import PGVector

from financial_rag_agent.core.config import get_settings
from financial_rag_agent.services.embeddings.factory import get_embeddings_client

BASE_COLLECTION_NAME = "sec_filing_chunks"


def _collection_name(settings) -> str:
    # Different embedding providers/models produce incompatible vector spaces.
    # Keying the collection name on both means switching EMBEDDING_PROVIDER in
    # .env can never silently mix vectors from two different models.
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", settings.embedding_model).strip("_")
    return f"{BASE_COLLECTION_NAME}__{settings.embedding_provider}__{model_slug}"


def vector_row_id(collection_name: str, chunk_id: uuid.UUID | str) -> str:
    """Derives the pgvector row id for a chunk within a given collection.

    langchain-postgres's langchain_pg_embedding table has a single global
    primary key on `id` (not scoped per collection), so the same chunk_id
    cannot be reused as the row id across two different collections without
    one collection's write silently clobbering the other's row. Deriving a
    per-collection id keeps chunk_id (the stable identity used for the
    relational round-trip) out of the primary key while still being
    deterministic, so re-ingesting the same chunk into the same collection
    upserts instead of duplicating.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{chunk_id}"))


@lru_cache
def get_vector_store() -> PGVector:
    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings_client(),
        collection_name=_collection_name(settings),
        connection=settings.database_url,
        use_jsonb=True,
    )
