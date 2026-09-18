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

    # Runtime document discovery/download (Phase 4). Explicit and
    # configurable per PROJECT_BUILD_PROMPT.md's engineering principles —
    # not magic numbers buried in downloader.py.
    download_timeout_seconds: int = 30
    max_document_size_bytes: int = 25 * 1024 * 1024  # 25 MB

    # Retrieval-sufficiency gate (Phase 4): minimum best citation-sentence
    # cosine similarity (retrieval/citations.py) for KB evidence to count
    # as "good enough" before falling back to runtime discovery. Measured
    # live, not guessed: an on-topic query against the right company's
    # filing scored 0.834; an off-topic query against that same filing
    # scored 0.455. 0.6 sits between them. This is a first-pass default,
    # not a tuned/validated one — see retrieval/sufficiency.py for why
    # this signal alone can't catch a wrong-company match (that's the
    # company/filing metadata check, not this threshold).
    retrieval_sufficiency_min_score: float = 0.6


@lru_cache
def get_settings() -> Settings:
    return Settings()
