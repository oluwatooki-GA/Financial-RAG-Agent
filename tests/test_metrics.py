import math

import pytest

from financial_rag_agent.eval.metrics import evaluate_retrieval

# One query, 3 docs. d1 and d2 are relevant (1), d3 is not (0). The run
# ranks them d1 (highest score) > d3 > d2 — same fixture used to
# cross-validate this project's original hand-rolled formulas against ranx.
QRELS = {"q1": {"d1": 1, "d2": 1, "d3": 0}}
RUN = {"q1": {"d1": 0.9, "d3": 0.8, "d2": 0.5}}


def test_precision_and_recall_at_2():
    m = evaluate_retrieval(QRELS, RUN, k=2)
    assert m["precision_at_k"] == pytest.approx(0.5)
    assert m["recall_at_k"] == pytest.approx(0.5)


def test_mrr_first_relevant_at_rank_1():
    m = evaluate_retrieval(QRELS, RUN, k=2)
    assert m["mrr"] == pytest.approx(1.0)


def test_ndcg_at_2_matches_hand_computed_value():
    dcg = 1 / math.log2(2) + 0 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg
    m = evaluate_retrieval(QRELS, RUN, k=2)
    assert m["ndcg_at_k"] == pytest.approx(expected)


def test_perfect_ranking_gives_precision_one():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    run = {"q1": {"d1": 0.9, "d2": 0.8}}
    m = evaluate_retrieval(qrels, run, k=2)
    assert m["precision_at_k"] == pytest.approx(1.0)
    assert m["ndcg_at_k"] == pytest.approx(1.0)
