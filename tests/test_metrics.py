import math

import pytest

from financial_rag_agent.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

RELEVANCES = [True, False, True, False, True]


def test_precision_at_k():
    assert precision_at_k(RELEVANCES, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RELEVANCES, 5) == pytest.approx(3 / 5)


def test_recall_at_k():
    assert recall_at_k(RELEVANCES, 3, total_relevant=3) == pytest.approx(2 / 3)
    assert recall_at_k(RELEVANCES, 5, total_relevant=3) == pytest.approx(1.0)
    assert recall_at_k(RELEVANCES, 5, total_relevant=0) == 0.0


def test_mrr_first_relevant_rank():
    assert mrr(RELEVANCES) == pytest.approx(1.0)
    assert mrr([False, False, True]) == pytest.approx(1 / 3)
    assert mrr([False, False, False]) == 0.0


def test_ndcg_matches_hand_computed_value():
    dcg = 1 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    expected = dcg / idcg
    assert ndcg_at_k(RELEVANCES, 3) == pytest.approx(expected)


def test_ndcg_perfect_ordering_is_one():
    assert ndcg_at_k([True, True, False], 3) == pytest.approx(1.0)


def test_ndcg_no_relevant_docs_is_zero():
    assert ndcg_at_k([False, False, False], 3) == 0.0
