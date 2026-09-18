import hashlib


def compute_file_hash(content: bytes) -> str:
    """SHA-256 hex digest of raw document bytes — the dedup key for any
    runtime-discovered document. Content-addressed rather than
    URL-addressed, so the same PDF found via two different URLs (or
    re-discovered after the KB already has it) resolves to one row."""
    return hashlib.sha256(content).hexdigest()
