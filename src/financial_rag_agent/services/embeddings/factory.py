from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from financial_rag_agent.core.config import get_settings


@lru_cache
def get_embeddings_client() -> Embeddings:
    settings = get_settings()

    if settings.embedding_provider == "ollama":
        client: Embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    elif settings.embedding_provider == "sentence-transformers":
        from langchain_huggingface import HuggingFaceEmbeddings

        client = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider!r}")

    probe = client.embed_query("dimension probe")
    if len(probe) != settings.embedding_dimension:
        raise RuntimeError(
            f"configured embedding_dimension={settings.embedding_dimension} but "
            f"model {settings.embedding_model!r} returned a {len(probe)}-dim vector"
        )

    return client
