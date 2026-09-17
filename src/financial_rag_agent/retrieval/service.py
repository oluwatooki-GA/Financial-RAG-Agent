from financial_rag_agent.core.config import get_settings
from financial_rag_agent.retrieval.hybrid_retriever import hybrid_search, hybrid_search_reranked
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk, baseline_vector_search

_STRATEGIES = {
    "baseline": lambda query, k, modality: baseline_vector_search(query, k=k, modality=modality),
    "hybrid": lambda query, k, modality: hybrid_search(query, k=k, modality=modality),
    "hybrid_reranked": lambda query, k, modality: hybrid_search_reranked(query, k=k, modality=modality),
}


def search(query: str, k: int = 5, modality: str | None = None) -> list[RetrievedChunk]:
    """Single retrieval entrypoint the router calls. Strategy is chosen via
    the RETRIEVAL_STRATEGY setting (default: "hybrid") rather than hardcoded,
    so it can change without a code change — and so baseline/hybrid/
    hybrid_reranked stay genuinely distinct, swappable configurations
    rather than one path that silently no-ops the others."""
    settings = get_settings()
    strategy = _STRATEGIES.get(settings.retrieval_strategy)
    if strategy is None:
        raise ValueError(f"Unsupported retrieval_strategy: {settings.retrieval_strategy!r}")
    return strategy(query, k, modality)
