from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from financial_rag_agent.core.config import get_settings


@lru_cache
def get_llm_client() -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "ollama":
        return ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "llm_provider='anthropic' requires anthropic_api_key "
                "(ANTHROPIC_API_KEY in .env) — a Claude.ai/Claude Code "
                "subscription does not include this."
            )
        return ChatAnthropic(model=settings.llm_model, api_key=settings.anthropic_api_key, temperature=0)

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider!r}")
