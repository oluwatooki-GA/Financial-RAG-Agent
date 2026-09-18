from types import SimpleNamespace

import pytest

from financial_rag_agent.services.llm import factory as factory_module
from financial_rag_agent.services.llm.factory import get_llm_client


@pytest.fixture(autouse=True)
def _clear_cache():
    get_llm_client.cache_clear()
    yield
    get_llm_client.cache_clear()


def test_ollama_provider_dispatch(monkeypatch):
    monkeypatch.setattr(
        factory_module, "get_settings", lambda: SimpleNamespace(llm_provider="ollama", llm_model="llama3.2:3b", ollama_base_url="http://localhost:11434")
    )
    calls = []
    monkeypatch.setattr(factory_module, "ChatOllama", lambda **kwargs: calls.append(kwargs) or "ollama-client")

    client = get_llm_client()

    assert client == "ollama-client"
    assert calls[0]["model"] == "llama3.2:3b"


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        factory_module,
        "get_settings",
        lambda: SimpleNamespace(llm_provider="anthropic", llm_model="claude-haiku-4-5-20251001", anthropic_api_key=None),
    )

    with pytest.raises(RuntimeError, match="anthropic_api_key"):
        get_llm_client()


def test_anthropic_provider_dispatch(monkeypatch):
    import langchain_anthropic

    monkeypatch.setattr(
        factory_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="anthropic", llm_model="claude-haiku-4-5-20251001", anthropic_api_key="sk-ant-fake"
        ),
    )
    calls = []
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **kwargs: calls.append(kwargs) or "anthropic-client")

    client = get_llm_client()

    assert client == "anthropic-client"
    assert calls[0]["model"] == "claude-haiku-4-5-20251001"
    assert calls[0]["api_key"] == "sk-ant-fake"


def test_unsupported_provider_raises_value_error(monkeypatch):
    monkeypatch.setattr(factory_module, "get_settings", lambda: SimpleNamespace(llm_provider="not-a-real-provider"))

    with pytest.raises(ValueError, match="not-a-real-provider"):
        get_llm_client()
