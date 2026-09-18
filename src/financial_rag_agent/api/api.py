from fastapi import APIRouter

from financial_rag_agent.api.routers.ingestion import router as ingestion_router
from financial_rag_agent.api.routers.retrieval import router as retrieval_router
from financial_rag_agent.api.routers.tools import router as tools_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(ingestion_router)
api_router.include_router(retrieval_router)
api_router.include_router(tools_router)
