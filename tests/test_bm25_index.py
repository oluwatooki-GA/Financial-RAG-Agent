from financial_rag_agent.retrieval.bm25_index import _tokenize
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
