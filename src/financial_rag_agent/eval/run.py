from dataclasses import dataclass

from financial_rag_agent.eval.dataset import EVAL_QUERIES
from financial_rag_agent.eval.judge import judge_relevance
from financial_rag_agent.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from financial_rag_agent.retrieval.hybrid_retriever import hybrid_search, hybrid_search_reranked
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk, baseline_vector_search

POOL_SIZE = 8
FINAL_K = 6

CONFIGS = {
    "baseline_vector": lambda query: baseline_vector_search(query, k=POOL_SIZE, with_citations=False),
    "hybrid": lambda query: hybrid_search(query, k=POOL_SIZE, with_citations=False),
    "hybrid_reranked": lambda query: hybrid_search_reranked(query, k=POOL_SIZE),
}


@dataclass
class ConfigResult:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def _judge_all(query: str, results: list[RetrievedChunk]) -> dict:
    return {r.chunk_id: judge_relevance(query, r.text) for r in results}


def run_eval() -> dict[str, list[ConfigResult]]:
    """Runs every config against every eval query, judging relevance with the
    small local LLM and computing real P@K/R@K/MRR/NDCG per query. Recall is
    computed against the union of judged-relevant chunks across all three
    configs' candidate pools for that query (an honest proxy — there is no
    exhaustive, corpus-wide relevance labeling to compute recall against)."""
    per_config_results: dict[str, list[ConfigResult]] = {name: [] for name in CONFIGS}

    for eval_query in EVAL_QUERIES:
        query = eval_query.query
        results_by_config: dict[str, list[RetrievedChunk]] = {
            name: fn(query) for name, fn in CONFIGS.items()
        }

        judged: dict = {}
        for results in results_by_config.values():
            judged.update(_judge_all(query, results))

        total_relevant = sum(1 for is_relevant in judged.values() if is_relevant)

        for name, results in results_by_config.items():
            relevances = [judged[r.chunk_id] for r in results]
            per_config_results[name].append(
                ConfigResult(
                    precision_at_k=precision_at_k(relevances, FINAL_K),
                    recall_at_k=recall_at_k(relevances, FINAL_K, total_relevant=total_relevant),
                    mrr=mrr(relevances),
                    ndcg_at_k=ndcg_at_k(relevances, FINAL_K),
                )
            )

    return per_config_results


def summarize(per_config_results: dict[str, list[ConfigResult]]) -> str:
    lines = [
        f"# Retrieval evaluation report",
        "",
        f"Queries: {len(EVAL_QUERIES)} | K={FINAL_K} | candidate pool={POOL_SIZE} | "
        f"judge=Ollama local LLM (see llm/factory.py)",
        "",
        "Recall is computed against the union of judged-relevant chunks across all "
        "three configs' pools for each query (no exhaustive corpus-wide ground truth exists).",
        "",
        "| Config | Precision@K | Recall@K | MRR | NDCG@K |",
        "|---|---|---|---|---|",
    ]
    for name, results in per_config_results.items():
        n = len(results)
        avg_p = sum(r.precision_at_k for r in results) / n
        avg_r = sum(r.recall_at_k for r in results) / n
        avg_mrr = sum(r.mrr for r in results) / n
        avg_ndcg = sum(r.ndcg_at_k for r in results) / n
        lines.append(f"| {name} | {avg_p:.3f} | {avg_r:.3f} | {avg_mrr:.3f} | {avg_ndcg:.3f} |")

    return "\n".join(lines)
