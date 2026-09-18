import time
from pathlib import Path

import requests

from financial_rag_agent.core.config import get_settings
from financial_rag_agent.documents.hashing import compute_file_hash

RAW_DISCOVERED_DIR = Path("data/raw_discovered")

_PDF_MAGIC = b"%PDF-"
_CHUNK_SIZE = 65536
_RETRY_DELAY_SECONDS = 1.0


class DownloadError(Exception):
    """A candidate URL could not be downloaded and validated as a real PDF."""


def _looks_like_pdf(content: bytes) -> bool:
    # The only real proof a response is a PDF — never trust the URL ending
    # in .pdf or the server's declared Content-Type, either of which can
    # be wrong (a dead link commonly redirects to an HTML error page that
    # still has a ".pdf"-looking URL).
    return content[:5] == _PDF_MAGIC


def _stream_download(url: str, timeout: int, max_bytes: int) -> bytes:
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise DownloadError(f"{url!r} returned Content-Type {content_type!r}, not a document")

        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
            total += len(chunk)
            if total > max_bytes:
                # Abort as soon as the limit is crossed, mid-stream — never
                # buffer the whole oversized body just to reject it after.
                raise DownloadError(f"{url!r} exceeded the {max_bytes}-byte size limit")
            chunks.append(chunk)

        return b"".join(chunks)


def download_pdf(url: str, *, max_bytes: int | None = None, timeout: int | None = None) -> tuple[bytes, str]:
    """Downloads a candidate document and validates it is genuinely a PDF
    before returning (content_bytes, sha256_hash). Retries once after a
    transient network failure; a validation failure (wrong content-type,
    not really a PDF, too large) is not retried since it won't change.

    Never downloads a file just because its URL ends in .pdf — see
    PROJECT_BUILD_PROMPT.md's design principle for runtime discovery.
    """
    settings = get_settings()
    max_bytes = max_bytes if max_bytes is not None else settings.max_document_size_bytes
    timeout = timeout if timeout is not None else settings.download_timeout_seconds

    try:
        content = _stream_download(url, timeout, max_bytes)
    except requests.RequestException:
        time.sleep(_RETRY_DELAY_SECONDS)
        try:
            content = _stream_download(url, timeout, max_bytes)
        except requests.RequestException as exc:
            raise DownloadError(f"Failed to download {url!r} after retrying: {exc}") from exc

    if not _looks_like_pdf(content):
        raise DownloadError(f"{url!r} did not return a real PDF (checked magic bytes, not the URL or Content-Type)")

    return content, compute_file_hash(content)


def save_downloaded_pdf(content: bytes, file_hash: str) -> Path:
    """Persists validated PDF bytes to local storage, named by content
    hash rather than any filename derived from the URL — sidesteps unsafe-
    filename/path-traversal concerns entirely rather than sanitizing one,
    and doubles as a disk-level dedup check."""
    RAW_DISCOVERED_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = RAW_DISCOVERED_DIR / f"{file_hash}.pdf"
    if not dest_path.exists():
        dest_path.write_bytes(content)
    return dest_path
