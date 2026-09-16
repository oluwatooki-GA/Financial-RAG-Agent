import uuid

from financial_rag_agent.retrieval.hybrid_retriever import reciprocal_rank_fusion

A, B, C, D = (uuid.uuid4() for _ in range(4))


def test_doc_ranked_first_by_both_retrievers_wins():
    fused = reciprocal_rank_fusion([[A, B, C], [A, C, B]])
    assert fused[0][0] == A


def test_doc_missing_from_one_retriever_still_scored():
    # B is only in the second list; should still appear, just lower than
    # docs found by both retrievers.
    fused = reciprocal_rank_fusion([[A, C], [D, B]])
    fused_ids = [chunk_id for chunk_id, _score in fused]
    assert B in fused_ids
    assert D in fused_ids


def test_higher_rank_in_a_single_retriever_scores_higher():
    fused = reciprocal_rank_fusion([[A, B, C]])
    scores = dict(fused)
    assert scores[A] > scores[B] > scores[C]


def test_appearing_in_both_lists_beats_appearing_in_only_one():
    # A appears in both lists (rank 2 in each); B only appears once, at rank 1.
    fused = reciprocal_rank_fusion([[C, A], [D, A]])
    scores = dict(fused)
    assert scores[A] > scores[C]
    assert scores[A] > scores[D]
