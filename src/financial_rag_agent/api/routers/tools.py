from fastapi import APIRouter, HTTPException, Query

from financial_rag_agent.tools import calculator
from financial_rag_agent.tools.schemas import (
    CalculateRequest,
    CalculateResponse,
    WebSearchResponse,
    WebSearchResultResponse,
    XBRLFactResponse,
    XBRLResponse,
)
from financial_rag_agent.tools.web_search import search_web
from financial_rag_agent.tools.xbrl import get_company_concept

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post("/calculate", response_model=CalculateResponse)
def calculate(request: CalculateRequest) -> CalculateResponse:
    func, required_fields = calculator.OPERATIONS[request.operation]

    missing = [f for f in required_fields if getattr(request, f) is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Operation {request.operation!r} requires: {', '.join(missing)}",
        )

    values = [getattr(request, f) for f in required_fields]
    result = func(*values)

    return CalculateResponse(operation=request.operation, result=result)


@router.get("/xbrl", response_model=XBRLResponse)
def xbrl_lookup(
    cik: str = Query(..., min_length=1, max_length=10),
    concept: str = Query(...),
    taxonomy: str = Query(default="us-gaap"),
) -> XBRLResponse:
    facts = get_company_concept(cik, concept, taxonomy=taxonomy)
    return XBRLResponse(
        cik=cik,
        concept=concept,
        facts=[
            XBRLFactResponse(
                concept=f.concept,
                end_date=f.end_date,
                value=f.value,
                unit=f.unit,
                fiscal_year=f.fiscal_year,
                fiscal_period=f.fiscal_period,
                form=f.form,
                filed_date=f.filed_date,
            )
            for f in facts
        ],
    )


@router.get("/web-search", response_model=WebSearchResponse)
def web_search(
    q: str = Query(..., min_length=1),
    max_results: int = Query(default=5, ge=1, le=20),
) -> WebSearchResponse:
    results = search_web(q, max_results=max_results)
    return WebSearchResponse(
        query=q,
        results=[WebSearchResultResponse(title=r.title, url=r.url, snippet=r.snippet) for r in results],
    )
