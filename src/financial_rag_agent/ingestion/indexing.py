from datetime import datetime
from uuid import UUID

from sqlmodel import Session, select

from financial_rag_agent.core import Chunk, Filing
from financial_rag_agent.core.config import get_settings
from financial_rag_agent.ingestion.chunker import ChunkDraft
from financial_rag_agent.retrieval.vector_store import get_vector_store, vector_row_id


def persist_and_embed(session: Session, filing: Filing, company_id: UUID, drafts: list[ChunkDraft]) -> list[Chunk]:
    """The one real chunk-persistence + embedding path, shared by every
    ingestion source (SEC HTML today, PDF as of Phase 4) — written once
    here instead of copied into documents/service.py, per the "one
    document ingestion path" design constraint: two copies of this logic
    could quietly drift apart (a fix or a new embedding field added to
    one path and forgotten in the other).

    Idempotent: if Chunks already exist for this filing, reuses them
    instead of re-parsing drafts (a re-run after a partial failure won't
    duplicate rows), and always re-embeds into whichever vector-store
    collection EMBEDDING_PROVIDER currently points at.
    """
    existing = session.exec(select(Chunk).where(Chunk.filing_id == filing.id).order_by(Chunk.chunk_index)).all()

    if not existing:
        filing.ingestion_status = "parsing"
        session.add(filing)
        session.commit()

        chunks: list[Chunk] = []
        for draft in drafts:
            chunk = Chunk(
                filing_id=filing.id,
                chunk_index=draft.chunk_index,
                part_label=draft.part_label,
                item_label=draft.item_label,
                item_heading=draft.item_heading,
                section_path=draft.section_path,
                text=draft.text,
                modality=draft.modality,
                table_data=draft.table_data,
                token_count=len(draft.text) // 4,
            )
            chunk.embedding_id = str(chunk.id)
            chunks.append(chunk)

        session.add_all(chunks)
        session.commit()
        for c in chunks:
            session.refresh(c)
    else:
        chunks = existing

    filing.ingestion_status = "embedding"
    session.add(filing)
    session.commit()

    vector_store = get_vector_store()
    batch_size = get_settings().embedding_batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vector_store.add_texts(
            texts=[c.text for c in batch],
            metadatas=[
                {
                    "chunk_id": str(c.id),
                    "filing_id": str(filing.id),
                    "company_id": str(company_id),
                    "item_label": c.item_label,
                    "item_heading": c.item_heading,
                    "modality": c.modality,
                }
                for c in batch
            ],
            ids=[vector_row_id(vector_store.collection_name, c.id) for c in batch],
        )

    filing.ingestion_status = "complete"
    filing.chunk_count = len(chunks)
    filing.ingested_at = datetime.utcnow()
    session.add(filing)
    session.commit()
    session.refresh(filing)

    return chunks
