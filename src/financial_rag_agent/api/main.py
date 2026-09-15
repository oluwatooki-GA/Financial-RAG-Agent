from fastapi import FastAPI

from financial_rag_agent.api.routers import ingestion, retrieval

app = FastAPI(title="Financial RAG Agent")
app.include_router(ingestion.router)
app.include_router(retrieval.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
