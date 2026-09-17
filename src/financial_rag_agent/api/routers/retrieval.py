from typing import Callable

from fastapi import APIRouter, Depends, Query

from financial_rag_agent.retrieval import service
from financial_rag_agent.retrieval.schemas import (
    CitationSentenceResponse,
    QueryResponse,
    RetrievedChunkResponse,
)
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk

router = APIRouter(tags=["retrieval"])


def get_query_service() -> Callable[..., list[RetrievedChunk]]:
    return service.search


@router.get("/query", response_model=QueryResponse)
def query(
    q: str = Query(..., min_length=1),
    k: int = Query(default=5, ge=1, le=20),
    modality: str | None = Query(default=None, pattern="^(text|table)$"),
    query_service=Depends(get_query_service),
) -> QueryResponse:
    results = query_service(q, k=k, modality=modality)
    return QueryResponse(
        query=q,
        results=[
            RetrievedChunkResponse(
                chunk_id=r.chunk_id,
                score=r.score,
                text=r.text,
                item_label=r.item_label,
                item_heading=r.item_heading,
                filing_accession_number=r.filing_accession_number,
                modality=r.modality,
                table_data=r.table_data,
                citation_sentences=[
                    CitationSentenceResponse(
                        text=c.text, char_start=c.char_start, char_end=c.char_end, score=c.score
                    )
                    for c in r.citation_sentences
                ],
            )
            for r in results
        ],
    )
