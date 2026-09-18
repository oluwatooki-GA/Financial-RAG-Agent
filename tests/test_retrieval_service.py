from types import SimpleNamespace

import pytest

from financial_rag_agent.retrieval import service


def _fake_settings(strategy: str):
    return SimpleNamespace(retrieval_strategy=strategy)


def test_hybrid_is_the_default_strategy_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "get_settings", lambda: _fake_settings("hybrid"))
    monkeypatch.setattr(
        service, "hybrid_search", lambda query, k, filing_id, modality: calls.append("hybrid")
    )

    service.search("some query", k=3)

    assert calls == ["hybrid"]


def test_baseline_strategy_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "get_settings", lambda: _fake_settings("baseline"))
    monkeypatch.setattr(
        service,
        "baseline_vector_search",
        lambda query, k, filing_id, modality: calls.append("baseline"),
    )

    service.search("some query", k=3)

    assert calls == ["baseline"]


def test_hybrid_reranked_strategy_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "get_settings", lambda: _fake_settings("hybrid_reranked"))
    monkeypatch.setattr(
        service,
        "hybrid_search_reranked",
        lambda query, k, filing_id, modality: calls.append("hybrid_reranked"),
    )

    service.search("some query", k=3)

    assert calls == ["hybrid_reranked"]


def test_unsupported_strategy_raises_value_error(monkeypatch):
    monkeypatch.setattr(service, "get_settings", lambda: _fake_settings("not-a-real-strategy"))

    with pytest.raises(ValueError, match="not-a-real-strategy"):
        service.search("some query", k=3)
