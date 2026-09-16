import re
from dataclasses import dataclass
from uuid import UUID

from rank_bm25 import BM25Okapi
from sqlmodel import select

from financial_rag_agent.db import Chunk, get_session

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _CachedIndex:
    fingerprint: tuple
    bm25: BM25Okapi
    chunk_ids: list[UUID]


_cache: dict[UUID | None, _CachedIndex] = {}


def _fingerprint(filing_id: UUID | None) -> tuple:
    # Cheap enough to run on every call; real BM25 (re)tokenization only
    # happens when this actually changes, i.e. a re-ingest happened.
    with get_session() as session:
        stmt = select(Chunk.chunk_index)
        if filing_id is not None:
            stmt = stmt.where(Chunk.filing_id == filing_id)
        indices = session.exec(stmt).all()
    return (len(indices), max(indices, default=-1))


def _build_index(filing_id: UUID | None, fingerprint: tuple) -> _CachedIndex:
    with get_session() as session:
        stmt = select(Chunk).order_by(Chunk.chunk_index)
        if filing_id is not None:
            stmt = stmt.where(Chunk.filing_id == filing_id)
        chunks = session.exec(stmt).all()

    chunk_ids = [c.id for c in chunks]
    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return _CachedIndex(fingerprint=fingerprint, bm25=bm25, chunk_ids=chunk_ids)


def get_bm25_index(filing_id: UUID | None = None) -> _CachedIndex:
    fingerprint = _fingerprint(filing_id)
    cached = _cache.get(filing_id)
    if cached is None or cached.fingerprint != fingerprint:
        cached = _build_index(filing_id, fingerprint)
        _cache[filing_id] = cached
    return cached


def bm25_search(query: str, k: int = 20, filing_id: UUID | None = None) -> list[tuple[UUID, float]]:
    """Real BM25 lexical search over the relational chunk table (not a
    vector-store convenience method). Returns (chunk_id, score) pairs,
    highest score first, dropping zero-score (no lexical overlap) results."""
    index = get_bm25_index(filing_id)
    if index.bm25 is None:
        return []

    scores = index.bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(index.chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [(chunk_id, float(score)) for chunk_id, score in ranked[:k] if score > 0]
