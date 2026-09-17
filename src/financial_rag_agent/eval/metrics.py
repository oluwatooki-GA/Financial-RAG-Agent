from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate


def evaluate_retrieval(
    qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]], k: int
) -> dict[str, float]:
    """Real Precision@K, Recall@K, MRR, and NDCG@K, computed via ranx (a
    validated, widely used IR evaluation library) instead of hand-rolled
    formulas.

    qrels: {query_id: {chunk_id: relevance}} (0/1 for binary relevance)
    run:   {query_id: {chunk_id: score}} — score determines rank order
    """
    results = ranx_evaluate(
        Qrels(qrels),
        Run(run),
        [f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}"],
    )
    return {
        "precision_at_k": float(results[f"precision@{k}"]),
        "recall_at_k": float(results[f"recall@{k}"]),
        "mrr": float(results["mrr"]),
        "ndcg_at_k": float(results[f"ndcg@{k}"]),
    }
