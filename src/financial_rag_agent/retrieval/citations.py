from dataclasses import dataclass

import numpy as np
from syntok.segmenter import process as syntok_process

from financial_rag_agent.embeddings.factory import get_embeddings_client


@dataclass
class Sentence:
    text: str
    char_start: int
    char_end: int


@dataclass
class CitationSentence:
    text: str
    char_start: int
    char_end: int
    score: float


def split_sentences(text: str) -> list[Sentence]:
    """Splits text into sentences with char offsets into the original text.

    Uses syntok's segmenter, which handles financial/legal text (decimals,
    abbreviations like "U.S." or "Jan.", "Item 7.") far more reliably than a
    naive regex split on periods.
    """
    sentences: list[Sentence] = []

    for paragraph in syntok_process(text):
        for sentence_tokens in paragraph:
            if not sentence_tokens:
                continue
            start = sentence_tokens[0].offset
            last_tok = sentence_tokens[-1]
            end = last_tok.offset + len(last_tok.value)
            sentences.append(Sentence(text=text[start:end], char_start=start, char_end=end))

    return sentences


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def extract_citation_sentences(query: str, chunk_text: str, top_n: int = 2) -> list[CitationSentence]:
    """Finds the exact sentence(s) within a retrieved chunk that best support
    the query, so an answer can cite a precise sentence instead of a whole
    chunk. Reuses whichever embedding provider EMBEDDING_PROVIDER points at."""
    sentences = split_sentences(chunk_text)
    if not sentences:
        return []

    client = get_embeddings_client()
    query_vec = client.embed_query(query)
    sentence_vecs = client.embed_documents([s.text for s in sentences])

    scored = [
        CitationSentence(
            text=s.text,
            char_start=s.char_start,
            char_end=s.char_end,
            score=_cosine_similarity(query_vec, vec),
        )
        for s, vec in zip(sentences, sentence_vecs)
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_n]
