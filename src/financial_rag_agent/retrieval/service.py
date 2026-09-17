from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk, baseline_vector_search


def search(query: str, k: int = 5, modality: str | None = None) -> list[RetrievedChunk]:
    """Single retrieval entrypoint the router calls. Currently always uses
    the baseline vector retriever; hybrid_search/hybrid_search_reranked
    exist in hybrid_retriever.py and can be routed to from here later
    (e.g. by a query-shape heuristic or a request parameter) without the
    router needing to change."""
    return baseline_vector_search(query, k=k, modality=modality)
