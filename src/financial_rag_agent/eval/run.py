from financial_rag_agent.eval.dataset import EVAL_QUERIES
from financial_rag_agent.eval.judge import judge_relevance
from financial_rag_agent.eval.metrics import evaluate_retrieval
from financial_rag_agent.retrieval.hybrid_retriever import hybrid_search, hybrid_search_reranked
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk, baseline_vector_search

POOL_SIZE = 8
FINAL_K = 6

CONFIGS = {
    "baseline_vector": lambda query: baseline_vector_search(query, k=POOL_SIZE, with_citations=False),
    "hybrid": lambda query: hybrid_search(query, k=POOL_SIZE, with_citations=False),
    "hybrid_reranked": lambda query: hybrid_search_reranked(query, k=POOL_SIZE),
}


def run_eval() -> dict[str, dict[str, float]]:
    """Runs every config against every eval query, judges relevance with the
    small local LLM, and computes real Precision@K/Recall@K/MRR/NDCG via
    ranx. Every config is scored against the same qrels per query (the union
    of judged-relevant chunks across all three configs' pools) — an honest
    proxy for recall, since there is no exhaustive corpus-wide relevance
    labeling to compute it against."""
    qrels: dict[str, dict[str, dict[str, int]]] = {name: {} for name in CONFIGS}
    runs: dict[str, dict[str, dict[str, float]]] = {name: {} for name in CONFIGS}

    for i, eval_query in enumerate(EVAL_QUERIES):
        query = eval_query.query
        query_id = f"q{i}"

        results_by_config: dict[str, list[RetrievedChunk]] = {
            name: fn(query) for name, fn in CONFIGS.items()
        }

        judged: dict = {}
        for results in results_by_config.values():
            for r in results:
                if r.chunk_id not in judged:
                    judged[r.chunk_id] = judge_relevance(query, r.text)

        shared_qrels = {str(chunk_id): int(is_relevant) for chunk_id, is_relevant in judged.items()}

        for name, results in results_by_config.items():
            qrels[name][query_id] = shared_qrels
            runs[name][query_id] = {str(r.chunk_id): r.score for r in results}

    return {name: evaluate_retrieval(qrels[name], runs[name], FINAL_K) for name in CONFIGS}


def summarize(results: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Retrieval evaluation report",
        "",
        f"Queries: {len(EVAL_QUERIES)} | K={FINAL_K} | candidate pool={POOL_SIZE} | "
        "judge=Ollama local LLM (see llm/factory.py) | metrics computed via ranx",
        "",
        "Recall is computed against the union of judged-relevant chunks across all "
        "three configs' pools for each query (no exhaustive corpus-wide ground truth exists).",
        "",
        "| Config | Precision@K | Recall@K | MRR | NDCG@K |",
        "|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['precision_at_k']:.3f} | {m['recall_at_k']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg_at_k']:.3f} |"
        )

    return "\n".join(lines)
