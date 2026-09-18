from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str

    sec_user_agent: str

    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    ollama_base_url: str = "http://localhost:11434"

    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # "baseline" | "hybrid" | "hybrid_reranked" — default is "hybrid" per real
    # eval results (eval_report.md): hybrid beat both baseline and
    # hybrid+reranked on every metric, so reranking is opt-in, not default.
    retrieval_strategy: str = "hybrid"

    chunk_target_tokens: int = 900
    chunk_overlap_tokens: int = 150

    web_search_max_results: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
