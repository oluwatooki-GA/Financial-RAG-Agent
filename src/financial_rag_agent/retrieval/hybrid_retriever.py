from uuid import UUID

from financial_rag_agent.retrieval.bm25_index import bm25_search
from financial_rag_agent.retrieval.citations import extract_citation_sentences
from financial_rag_agent.retrieval.reranker import rerank
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk, attach_details, raw_vector_search

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[UUID]], rrf_k: int = DEFAULT_RRF_K) -> list[tuple[UUID, float]]:
    """Standard RRF: score(d) = sum over retrievers r that returned d of
    1 / (rrf_k + rank_r(d)), rank_r being the 1-indexed rank in that
    retriever's list. A doc missing from a retriever's list contributes 0
    for that retriever."""
    scores: dict[UUID, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _fused_candidate_ids(
    query: str, filing_id: UUID | None, candidate_pool_size: int, modality: str | None = None
) -> list[tuple[UUID, float]]:
    vector_results = raw_vector_search(query, k=candidate_pool_size, filing_id=filing_id, modality=modality)
    bm25_results = bm25_search(query, k=candidate_pool_size, filing_id=filing_id, modality=modality)
    return reciprocal_rank_fusion(
        [
            [chunk_id for chunk_id, _score in vector_results],
            [chunk_id for chunk_id, _score in bm25_results],
        ]
    )


def hybrid_search(
    query: str,
    k: int = 5,
    filing_id: UUID | None = None,
    candidate_pool_size: int = 20,
    modality: str | None = None,
    with_citations: bool = True,
) -> list[RetrievedChunk]:
    """Vector search + BM25, fused via reciprocal rank fusion. No reranking —
    a distinct configuration from baseline_vector_search and
    hybrid_search_reranked, not a flag that silently no-ops either."""
    fused = _fused_candidate_ids(query, filing_id, candidate_pool_size, modality=modality)
    return attach_details(query, fused[:k], with_citations=with_citations)


def hybrid_search_reranked(
    query: str,
    k: int = 5,
    filing_id: UUID | None = None,
    candidate_pool_size: int = 20,
    modality: str | None = None,
) -> list[RetrievedChunk]:
    """Hybrid (vector + BM25 + RRF) candidate pool, then the cross-encoder
    reranker is actually called on that pool to produce the final top_k."""
    fused = _fused_candidate_ids(query, filing_id, candidate_pool_size, modality=modality)
    candidates = attach_details(query, fused, with_citations=False)
    reranked = rerank(query, candidates, top_k=k)

    for r in reranked:
        r.citation_sentences = extract_citation_sentences(query, r.text)

    return reranked
