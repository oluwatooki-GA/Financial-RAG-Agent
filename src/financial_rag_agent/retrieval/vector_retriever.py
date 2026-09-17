from dataclasses import dataclass, field
from uuid import UUID

from sqlmodel import select

from financial_rag_agent.core import Chunk, Filing, get_session
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
    modality: str = "text"
    table_data: list[list[str]] | None = None
    citation_sentences: list[CitationSentence] = field(default_factory=list)


def raw_vector_search(
    query: str, k: int = 20, filing_id: UUID | None = None, modality: str | None = None
) -> list[tuple[UUID, float]]:
    """Real vector similarity search, returning (chunk_id, score) pairs with
    the vector store's actual similarity_search_with_score (never a
    hardcoded score). Used both by the baseline retriever directly and by
    hybrid retrieval as one of the two ranked lists fed into RRF."""
    vector_store = get_vector_store()
    filter_clauses = {}
    if filing_id:
        filter_clauses["filing_id"] = str(filing_id)
    if modality:
        filter_clauses["modality"] = modality
    results = vector_store.similarity_search_with_score(query, k=k, filter=filter_clauses or None)
    return [(UUID(doc.metadata["chunk_id"]), score) for doc, score in results]


def attach_details(
    query: str, scored_chunk_ids: list[tuple[UUID, float]], with_citations: bool = True
) -> list[RetrievedChunk]:
    """Joins (chunk_id, score) pairs back to the relational Chunk/Filing rows
    and optionally computes citation sentences. Shared by every retrieval
    configuration (baseline, hybrid, hybrid+reranked) so there is one path
    from a ranked chunk_id list to a RetrievedChunk, not several."""
    if not scored_chunk_ids:
        return []

    chunk_ids = [chunk_id for chunk_id, _score in scored_chunk_ids]
    with get_session() as session:
        rows = session.exec(
            select(Chunk, Filing)
            .join(Filing, Filing.id == Chunk.filing_id)
            .where(Chunk.id.in_(chunk_ids))
        ).all()
        by_id = {chunk.id: (chunk, filing) for chunk, filing in rows}

    retrieved: list[RetrievedChunk] = []
    for chunk_id, score in scored_chunk_ids:
        if chunk_id not in by_id:
            continue
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
                modality=chunk.modality,
                table_data=chunk.table_data,
                citation_sentences=citation_sentences,
            )
        )

    return retrieved


def baseline_vector_search(
    query: str,
    k: int = 5,
    filing_id: UUID | None = None,
    modality: str | None = None,
    with_citations: bool = True,
) -> list[RetrievedChunk]:
    scored_chunk_ids = raw_vector_search(query, k=k, filing_id=filing_id, modality=modality)
    return attach_details(query, scored_chunk_ids, with_citations=with_citations)
