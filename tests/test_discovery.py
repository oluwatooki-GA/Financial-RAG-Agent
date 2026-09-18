import pytest

from financial_rag_agent.discovery import registry as registry_module
from financial_rag_agent.discovery.ngx_source import NGXSource
from financial_rag_agent.discovery.registry import discover_documents
from financial_rag_agent.discovery.schemas import DiscoveredDocument
from financial_rag_agent.discovery.sec_source import SECSource
from financial_rag_agent.discovery.web_source import WebSource
from financial_rag_agent.ingestion.edgar_client import FilingRef


def _fake_filing_ref(cik: str) -> FilingRef:
    return FilingRef(
        cik=cik,
        company_name="NVIDIA CORP",
        ticker="NVDA",
        sic="3674",
        accession_number="0001045810-26-000021",
        form_type="10-K",
        filing_date="2026-02-20",
        period_of_report="2026-01-25",
        primary_document="nvda-20260125.htm",
        source_url="https://www.sec.gov/fake/nvda.htm",
    )


def test_sec_source_returns_empty_without_a_cik():
    assert SECSource().search_documents("Some Company", cik=None) == []


def test_sec_source_wraps_get_latest_10k(monkeypatch):
    import financial_rag_agent.discovery.sec_source as sec_source_module

    monkeypatch.setattr(sec_source_module, "get_latest_10k", _fake_filing_ref)

    results = SECSource().search_documents("NVIDIA", cik="0001045810")

    assert len(results) == 1
    assert results[0].source_type == "sec"
    assert results[0].source_url == "https://www.sec.gov/fake/nvda.htm"
    assert results[0].document_type == "10-K"


def test_ngx_source_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        NGXSource().search_documents("GTCO")


def test_web_source_wraps_search_web(monkeypatch):
    import financial_rag_agent.discovery.web_source as web_source_module
    from financial_rag_agent.tools.web_search import WebSearchResult

    monkeypatch.setattr(
        web_source_module,
        "search_web",
        lambda query, max_results=None: [
            WebSearchResult(title="GTCO 2025 Annual Report", url="https://gtco.com/ar2025.pdf", snippet="...")
        ],
    )

    results = WebSource().search_documents("GTCO")

    assert len(results) == 1
    assert results[0].source_type == "web"
    assert results[0].source_url == "https://gtco.com/ar2025.pdf"


def test_discover_documents_prefers_sec_when_available(monkeypatch):
    calls = []

    class _StubSEC:
        source_type = "sec"

        def search_documents(self, company_name, cik=None):
            calls.append("sec")
            return [DiscoveredDocument(title="NVIDIA 10-K", source_url="https://sec.gov/x", source_type="sec")]

    class _StubWeb:
        source_type = "web"

        def search_documents(self, company_name, cik=None):
            calls.append("web")
            return [DiscoveredDocument(title="should not be reached", source_url="https://example.com", source_type="web")]

    monkeypatch.setattr(registry_module, "_SOURCES", [_StubSEC(), _StubWeb()])

    results = discover_documents("NVIDIA", cik="0001045810")

    assert calls == ["sec"]
    assert results[0].source_url == "https://sec.gov/x"


def test_discover_documents_falls_through_unimplemented_and_empty_sources(monkeypatch):
    calls = []

    class _StubSEC:
        source_type = "sec"

        def search_documents(self, company_name, cik=None):
            calls.append("sec")
            return []  # e.g. no cik known for this company

    class _StubNGX:
        source_type = "ngx"

        def search_documents(self, company_name, cik=None):
            calls.append("ngx")
            raise NotImplementedError

    class _StubWeb:
        source_type = "web"

        def search_documents(self, company_name, cik=None):
            calls.append("web")
            return [DiscoveredDocument(title="GTCO report", source_url="https://gtco.com/ar.pdf", source_type="web")]

    monkeypatch.setattr(registry_module, "_SOURCES", [_StubSEC(), _StubNGX(), _StubWeb()])

    results = discover_documents("GTCO")

    assert calls == ["sec", "ngx", "web"]
    assert results[0].source_type == "web"


def test_discover_documents_returns_empty_when_every_source_finds_nothing(monkeypatch):
    class _StubEmpty:
        source_type = "web"

        def search_documents(self, company_name, cik=None):
            return []

    monkeypatch.setattr(registry_module, "_SOURCES", [_StubEmpty()])

    assert discover_documents("Unknown Company") == []
