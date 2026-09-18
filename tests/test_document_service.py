import pytest
from sqlmodel import Session, SQLModel, create_engine

from financial_rag_agent.discovery.schemas import DiscoveredDocument
from financial_rag_agent.documents import service as service_module
from financial_rag_agent.documents.downloader import DownloadError
from financial_rag_agent.documents.service import DiscoveryError, discover_and_register_document


@pytest.fixture
def sqlite_session(monkeypatch):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(service_module, "get_session", lambda: Session(engine))


def test_raises_discovery_error_when_no_candidates_found(monkeypatch, sqlite_session):
    monkeypatch.setattr(service_module, "discover_documents", lambda company_name, cik=None: [])

    with pytest.raises(DiscoveryError, match="No document found"):
        discover_and_register_document("Nonexistent Co")


def test_tries_next_candidate_when_first_download_fails(monkeypatch, sqlite_session):
    candidates = [
        DiscoveredDocument(title="dead link", source_url="https://example.com/dead.pdf", source_type="web"),
        DiscoveredDocument(title="real one", source_url="https://example.com/real.pdf", source_type="web"),
    ]
    monkeypatch.setattr(service_module, "discover_documents", lambda company_name, cik=None: candidates)

    def _fake_download(url, **kwargs):
        if "dead" in url:
            raise DownloadError("404")
        return b"%PDF-1.4 real content", "real-hash"

    monkeypatch.setattr(service_module, "download_pdf", _fake_download)
    monkeypatch.setattr(service_module, "save_downloaded_pdf", lambda content, file_hash: f"/tmp/{file_hash}.pdf")

    document, created = discover_and_register_document("Some Co")

    assert created is True
    assert document.source_url == "https://example.com/real.pdf"
    assert document.file_hash == "real-hash"
    assert document.ingestion_status == "downloaded"


def test_raises_discovery_error_when_every_candidate_fails_to_download(monkeypatch, sqlite_session):
    candidates = [
        DiscoveredDocument(title="dead 1", source_url="https://example.com/a.pdf", source_type="web"),
        DiscoveredDocument(title="dead 2", source_url="https://example.com/b.pdf", source_type="web"),
    ]
    monkeypatch.setattr(service_module, "discover_documents", lambda company_name, cik=None: candidates)
    monkeypatch.setattr(
        service_module, "download_pdf", lambda url, **kwargs: (_ for _ in ()).throw(DownloadError("dead"))
    )

    with pytest.raises(DiscoveryError, match="none downloaded"):
        discover_and_register_document("Some Co")


def test_second_discovery_of_the_same_content_does_not_create_a_new_row(monkeypatch, sqlite_session):
    candidate = DiscoveredDocument(title="report", source_url="https://example.com/report.pdf", source_type="web")
    monkeypatch.setattr(service_module, "discover_documents", lambda company_name, cik=None: [candidate])
    monkeypatch.setattr(service_module, "download_pdf", lambda url, **kwargs: (b"%PDF-1.4 x", "same-hash"))

    save_calls = []
    monkeypatch.setattr(
        service_module,
        "save_downloaded_pdf",
        lambda content, file_hash: save_calls.append(file_hash) or f"/tmp/{file_hash}.pdf",
    )

    first, first_created = discover_and_register_document("Some Co")
    second, second_created = discover_and_register_document("Some Co")

    assert first_created is True
    assert second_created is False
    assert second.id == first.id
    assert save_calls == ["same-hash"]  # only saved to disk once, not on the dedup hit
