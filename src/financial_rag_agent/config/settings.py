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

    chunk_target_tokens: int = 900
    chunk_overlap_tokens: int = 150


@lru_cache
def get_settings() -> Settings:
    return Settings()
