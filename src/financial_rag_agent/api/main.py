import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from financial_rag_agent.api.routers import ingestion, retrieval

app = FastAPI(title="Financial RAG Agent")
app.include_router(ingestion.router)
app.include_router(retrieval.router)


@app.exception_handler(ValueError)
def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    # e.g. edgar_client.get_latest_10k raises ValueError when a CIK has no
    # 10-K on record — that's a client-correctable request, not a server bug.
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(requests.RequestException)
def handle_upstream_error(request: Request, exc: requests.RequestException) -> JSONResponse:
    # EDGAR (or any future upstream HTTP dependency) being unreachable or
    # erroring is not this service's fault — 502, not a raw 500 traceback.
    return JSONResponse(status_code=502, content={"detail": f"Upstream request failed: {exc}"})


@app.exception_handler(RuntimeError)
def handle_runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
    # e.g. embeddings/factory.py's dimension-mismatch fail-fast check — a
    # real configuration bug, not something the caller can fix.
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
