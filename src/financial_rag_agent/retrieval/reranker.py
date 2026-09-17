from dataclasses import replace
from functools import lru_cache

from sentence_transformers import CrossEncoder

from financial_rag_agent.core.config import get_settings
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk


@lru_cache
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Actually calls the cross-encoder reranker on the candidate pool and
    returns the top_k by reranker score. This function must be called from
    the retrieval path that claims to rerank — never left plumbed through
    and unused."""
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)

    reordered = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    # dataclasses.replace copies every field from the original candidate and
    # only overrides score, so this can never silently drop a field that
    # gets added to RetrievedChunk later (as manually listing each field
    # here once did — it caused the multimodal fields to go missing).
    return [replace(c, score=float(score)) for c, score in reordered[:top_k]]
