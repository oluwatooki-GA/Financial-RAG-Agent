import ddgs.exceptions
import pytest

from financial_rag_agent.tools import web_search as web_search_module
from financial_rag_agent.tools.web_search import search_web


class _FakeDDGS:
    def __init__(self, fail_backends: dict[str, int]):
        # fail_backends: backend -> number of times it should fail before succeeding
        self._fail_backends = dict(fail_backends)
        self.calls: list[str] = []

    def text(self, query, max_results, backend):
        self.calls.append(backend)
        remaining = self._fail_backends.get(backend, 0)
        if remaining > 0:
            self._fail_backends[backend] = remaining - 1
            raise ddgs.exceptions.TimeoutException(f"{backend} backend failed")
        return [{"title": f"result from {backend}", "href": "https://example.com", "body": "snippet"}]


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(web_search_module.time, "sleep", lambda _seconds: None)


def test_primary_backend_used_when_it_works(monkeypatch):
    fake = _FakeDDGS(fail_backends={})
    monkeypatch.setattr(web_search_module, "DDGS", lambda: fake)

    results = search_web("test query", max_results=3)

    assert len(results) == 1
    assert results[0].title == "result from duckduckgo"
    assert fake.calls == ["duckduckgo"]


def test_retries_primary_once_before_falling_back(monkeypatch):
    fake = _FakeDDGS(fail_backends={"duckduckgo": 1})
    monkeypatch.setattr(web_search_module, "DDGS", lambda: fake)

    results = search_web("test query", max_results=3)

    assert results[0].title == "result from duckduckgo"
    assert fake.calls == ["duckduckgo", "duckduckgo"]


def test_falls_back_to_secondary_backend_when_primary_fails_twice(monkeypatch):
    fake = _FakeDDGS(fail_backends={"duckduckgo": 2})
    monkeypatch.setattr(web_search_module, "DDGS", lambda: fake)

    results = search_web("test query", max_results=3)

    assert results[0].title == "result from bing"
    assert fake.calls == ["duckduckgo", "duckduckgo", "bing"]


def test_raises_ddgs_exception_when_every_backend_fails(monkeypatch):
    fake = _FakeDDGS(fail_backends={"duckduckgo": 2, "bing": 1})
    monkeypatch.setattr(web_search_module, "DDGS", lambda: fake)

    with pytest.raises(ddgs.exceptions.DDGSException):
        search_web("test query", max_results=3)
