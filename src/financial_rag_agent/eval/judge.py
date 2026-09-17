from functools import lru_cache

from financial_rag_agent.services.llm.factory import get_llm_client

_JUDGE_PROMPT = """You are judging retrieval relevance for a financial research system.

Question: {query}

Passage from an SEC filing:
\"\"\"
{passage}
\"\"\"

Does this passage contain information that would help answer the question? Reply with exactly one word: YES or NO."""


@lru_cache(maxsize=4096)
def judge_relevance(query: str, chunk_text: str) -> bool:
    """LLM-as-judge binary relevance label for one (query, chunk) pair, via
    the small local Ollama model. Cached per (query, chunk_text) so the same
    chunk surfacing across multiple retrieval configurations in one eval run
    is only judged once."""
    llm = get_llm_client()
    prompt = _JUDGE_PROMPT.format(query=query, passage=chunk_text[:2000])
    response = llm.invoke(prompt)
    answer = str(response.content).strip().upper()
    return answer.startswith("YES")
