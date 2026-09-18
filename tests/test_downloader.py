import requests

from financial_rag_agent.documents import downloader as downloader_module
from financial_rag_agent.documents.downloader import DownloadError, download_pdf, save_downloaded_pdf


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200, content_type: str = "application/pdf"):
        self._content = content
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_get(response: _FakeResponse):
    def _get(url, stream=True, timeout=None):
        return response

    return _get


def test_valid_pdf_downloads_and_hashes_correctly(monkeypatch):
    pdf_bytes = b"%PDF-1.4 fake but real-looking pdf content"
    monkeypatch.setattr(downloader_module.requests, "get", _fake_get(_FakeResponse(pdf_bytes)))

    content, file_hash = download_pdf("https://example.com/report.pdf")

    assert content == pdf_bytes
    from financial_rag_agent.documents.hashing import compute_file_hash

    assert file_hash == compute_file_hash(pdf_bytes)


def test_html_error_page_is_rejected_by_content_type(monkeypatch):
    html_bytes = b"<html><body>404 not found</body></html>"
    monkeypatch.setattr(
        downloader_module.requests, "get", _fake_get(_FakeResponse(html_bytes, content_type="text/html"))
    )

    try:
        download_pdf("https://example.com/dead-link.pdf")
        assert False, "expected DownloadError"
    except DownloadError as exc:
        assert "not a document" in str(exc)


def test_non_pdf_content_is_rejected_by_magic_bytes_even_with_correct_content_type(monkeypatch):
    # Content-Type says application/pdf, but the bytes aren't a real PDF —
    # the header alone must never be trusted.
    fake_bytes = b"this is not actually a pdf despite the header"
    monkeypatch.setattr(downloader_module.requests, "get", _fake_get(_FakeResponse(fake_bytes)))

    try:
        download_pdf("https://example.com/lying.pdf")
        assert False, "expected DownloadError"
    except DownloadError as exc:
        assert "magic bytes" in str(exc)


def test_oversized_response_is_aborted_before_reading_everything(monkeypatch):
    big_pdf = b"%PDF-1.4 " + (b"x" * 1000)
    monkeypatch.setattr(downloader_module.requests, "get", _fake_get(_FakeResponse(big_pdf)))

    try:
        download_pdf("https://example.com/huge.pdf", max_bytes=100)
        assert False, "expected DownloadError"
    except DownloadError as exc:
        assert "size limit" in str(exc)


def test_transient_network_failure_is_retried_once_then_succeeds(monkeypatch):
    pdf_bytes = b"%PDF-1.4 real content after a retry"
    calls = {"count": 0}

    def _flaky_get(url, stream=True, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ConnectionError("transient network blip")
        return _FakeResponse(pdf_bytes)

    monkeypatch.setattr(downloader_module.requests, "get", _flaky_get)
    monkeypatch.setattr(downloader_module.time, "sleep", lambda _seconds: None)

    content, _file_hash = download_pdf("https://example.com/flaky.pdf")

    assert content == pdf_bytes
    assert calls["count"] == 2


def test_persistent_network_failure_raises_download_error(monkeypatch):
    def _always_fails(url, stream=True, timeout=None):
        raise requests.Timeout("server never responded")

    monkeypatch.setattr(downloader_module.requests, "get", _always_fails)
    monkeypatch.setattr(downloader_module.time, "sleep", lambda _seconds: None)

    try:
        download_pdf("https://example.com/timeout.pdf")
        assert False, "expected DownloadError"
    except DownloadError as exc:
        assert "after retrying" in str(exc)


def test_save_downloaded_pdf_is_content_addressed_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "RAW_DISCOVERED_DIR", tmp_path)
    content = b"%PDF-1.4 some real bytes"
    file_hash = "deadbeef"

    first_path = save_downloaded_pdf(content, file_hash)
    assert first_path.exists()
    assert first_path.read_bytes() == content

    # calling again with the same hash must not error or duplicate anything
    second_path = save_downloaded_pdf(content, file_hash)
    assert second_path == first_path
    assert list(tmp_path.iterdir()) == [first_path]
