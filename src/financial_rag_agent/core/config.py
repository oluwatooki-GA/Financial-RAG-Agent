from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str

    sec_user_agent: str

    # sentence-transformers runs in-process (no separate server to crash
    # under load — verified live: embedding 719 chunks in one Ollama batch
    # crashed its server process mid-request), so it's the default here.
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    ollama_base_url: str = "http://localhost:11434"

    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    # Only required when llm_provider="anthropic". Billed separately via
    # console.anthropic.com — not covered by a Claude.ai/Claude Code
    # subscription. Never logged or committed; lives in .env only.
    anthropic_api_key: str | None = None

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # "baseline" | "hybrid" | "hybrid_reranked" — default is "hybrid" per real
    # eval results (eval_report.md): hybrid beat both baseline and
    # hybrid+reranked on every metric, so reranking is opt-in, not default.
    retrieval_strategy: str = "hybrid"

    chunk_target_tokens: int = 900
    chunk_overlap_tokens: int = 150

    # Embedding a large document's chunks in one single add_texts() call
    # can overwhelm a local Ollama server - verified live: a real 719-chunk
    # PDF crashed Ollama's internal tokenizer subprocess mid-request ("dial
    # tcp ...: connectex: actively refused") when embedded as one batch,
    # while NVIDIA's 256-chunk and Apple's 187-chunk filings never hit this.
    # Batching keeps each request small enough to be reliable.
    embedding_batch_size: int = 64

    web_search_max_results: int = 5

    # Runtime document discovery/download (Phase 4). Explicit and
    # configurable per PROJECT_BUILD_PROMPT.md's engineering principles —
    # not magic numbers buried in downloader.py.
    download_timeout_seconds: int = 30
    max_document_size_bytes: int = 25 * 1024 * 1024  # 25 MB

    # Retrieval-sufficiency gate (Phase 4): minimum best citation-sentence
    # cosine similarity (retrieval/citations.py) for KB evidence to count
    # as "good enough" before falling back to runtime discovery.
    #
    # This threshold is TIED TO THE EMBEDDING MODEL — the score scale
    # shifts between models, so it was re-measured when the default
    # embedding provider changed. Measured live, not guessed:
    #   all-MiniLM-L6-v2 (current default): on-topic 0.841 (NVIDIA) and
    #     0.531 (GTCO, noisier PDF-extracted text); off-topic 0.235/0.228.
    #   Ollama nomic-embed-text (previous):  on-topic 0.834/0.731;
    #     off-topic 0.455.
    # 0.4 sits between the current model's max off-topic (0.235) and min
    # on-topic (0.531). Keeping the old 0.6 threshold under the current
    # model would wrongly reject GTCO's on-topic query at 0.531 —
    # re-measure this if the embedding model changes again. See retrieval/
    # sufficiency.py for why this signal alone can't catch a wrong-company
    # match (that's the company/filing metadata check, not this number).
    retrieval_sufficiency_min_score: float = 0.4


@lru_cache
def get_settings() -> Settings:
    return Settings()
