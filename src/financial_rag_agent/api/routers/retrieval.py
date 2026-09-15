from fastapi import APIRouter, Query

from financial_rag_agent.api.schemas import QueryResponse, RetrievedChunkResponse
from financial_rag_agent.retrieval.vector_retriever import baseline_vector_search

router = APIRouter(tags=["retrieval"])


@router.get("/query", response_model=QueryResponse)
def query(q: str = Query(..., min_length=1), k: int = Query(default=5, ge=1, le=20)) -> QueryResponse:
    results = baseline_vector_search(q, k=k)
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
            )
            for r in results
        ],
    )
