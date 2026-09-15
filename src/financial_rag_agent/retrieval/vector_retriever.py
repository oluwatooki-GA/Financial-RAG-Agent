from dataclasses import dataclass, field
from uuid import UUID

from sqlmodel import select

from financial_rag_agent.db import Chunk, Filing, get_session
from financial_rag_agent.retrieval.citations import CitationSentence, extract_citation_sentences
from financial_rag_agent.retrieval.vector_store import get_vector_store


@dataclass
class RetrievedChunk:
    chunk_id: UUID
    score: float
    text: str
    item_label: str | None
    item_heading: str | None
    filing_accession_number: str
    citation_sentences: list[CitationSentence] = field(default_factory=list)


def baseline_vector_search(
    query: str, k: int = 5, filing_id: UUID | None = None, with_citations: bool = True
) -> list[RetrievedChunk]:
    vector_store = get_vector_store()

    search_filter = {"filing_id": str(filing_id)} if filing_id else None
    results = vector_store.similarity_search_with_score(query, k=k, filter=search_filter)

    chunk_ids = [UUID(doc.metadata["chunk_id"]) for doc, _score in results]

    with get_session() as session:
        rows = session.exec(
            select(Chunk, Filing)
            .join(Filing, Filing.id == Chunk.filing_id)
            .where(Chunk.id.in_(chunk_ids))
        ).all()
        by_id = {chunk.id: (chunk, filing) for chunk, filing in rows}

    retrieved: list[RetrievedChunk] = []
    for doc, score in results:
        chunk_id = UUID(doc.metadata["chunk_id"])
        chunk, filing = by_id[chunk_id]
        citation_sentences = extract_citation_sentences(query, chunk.text) if with_citations else []
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                score=score,
                text=chunk.text,
                item_label=chunk.item_label,
                item_heading=chunk.item_heading,
                filing_accession_number=filing.accession_number,
                citation_sentences=citation_sentences,
            )
        )

    return retrieved
