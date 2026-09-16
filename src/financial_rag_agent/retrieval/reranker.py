from functools import lru_cache

from sentence_transformers import CrossEncoder

from financial_rag_agent.config import get_settings
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
    return [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            score=float(score),
            text=c.text,
            item_label=c.item_label,
            item_heading=c.item_heading,
            filing_accession_number=c.filing_accession_number,
            citation_sentences=c.citation_sentences,
        )
        for c, score in reordered[:top_k]
    ]
