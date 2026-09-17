import uuid

from financial_rag_agent.retrieval.bm25_index import _rank_and_filter, _tokenize
from rank_bm25 import BM25Okapi


def test_exact_keyword_match_ranks_first():
    docs = [
        "NVIDIA Data Center revenue grew significantly in fiscal year 2026.",
        "Our Gaming segment includes GeForce GPUs for consumers.",
        "Risk factors include supply chain constraints and competition.",
    ]
    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(_tokenize("Data Center revenue"))
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)

    assert ranked[0] == 0
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_tokenize_lowercases_and_strips_punctuation():
    assert _tokenize("NVIDIA's Data-Center, Revenue!") == ["nvidia", "s", "data", "center", "revenue"]


def test_modality_filter_restricts_results():
    a, b, c = (uuid.uuid4() for _ in range(3))
    chunk_ids = [a, b, c]
    scores = [3.0, 2.0, 1.0]
    modalities = ["text", "table", "table"]

    unfiltered = _rank_and_filter(chunk_ids, scores, modalities, k=10, modality=None)
    assert [cid for cid, _ in unfiltered] == [a, b, c]

    table_only = _rank_and_filter(chunk_ids, scores, modalities, k=10, modality="table")
    assert [cid for cid, _ in table_only] == [b, c]

    text_only = _rank_and_filter(chunk_ids, scores, modalities, k=10, modality="text")
    assert [cid for cid, _ in text_only] == [a]


def test_zero_score_results_are_dropped_even_within_modality():
    a, b = (uuid.uuid4() for _ in range(2))
    results = _rank_and_filter([a, b], [0.0, 5.0], ["table", "table"], k=10, modality="table")
    assert [cid for cid, _ in results] == [b]
