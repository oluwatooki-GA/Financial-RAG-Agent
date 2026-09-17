from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from financial_rag_agent.config import get_settings


@lru_cache
def get_llm_client() -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "ollama":
        return ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider!r}")
