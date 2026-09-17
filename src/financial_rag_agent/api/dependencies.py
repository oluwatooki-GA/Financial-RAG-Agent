from typing import Callable

from financial_rag_agent.db import Filing
from financial_rag_agent.ingestion.pipeline import ingest_filing
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk, baseline_vector_search


def get_ingest_service() -> Callable[[str], Filing]:
    """Dependency provider for the ingestion service. Routes depend on this
    injectable callable rather than importing ingest_filing directly, so
    tests can override it via app.dependency_overrides instead of
    monkeypatching the ingestion module."""
    return ingest_filing


def get_query_service() -> Callable[..., list[RetrievedChunk]]:
    return baseline_vector_search
