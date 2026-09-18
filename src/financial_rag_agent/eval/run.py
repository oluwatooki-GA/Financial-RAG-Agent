import json
import logging
import time
from pathlib import Path

from sqlmodel import func, select

from financial_rag_agent.core import Filing, get_session
from financial_rag_agent.core.config import get_settings
from financial_rag_agent.eval.dataset import EVAL_QUERIES
from financial_rag_agent.eval.judge import judge_relevance
from financial_rag_agent.eval.metrics import evaluate_retrieval
from financial_rag_agent.retrieval.hybrid_retriever import hybrid_search, hybrid_search_reranked
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk, baseline_vector_search

logger = logging.getLogger(__name__)

POOL_SIZE = 5
FINAL_K = 4
RERANK_CANDIDATE_POOL_SIZE = 10  # was the hybrid_search_reranked default of 20
PROGRESS_FILE = Path("eval_progress.json")

CONFIGS = {
    "baseline_vector": lambda query: baseline_vector_search(query, k=POOL_SIZE, with_citations=False),
    "hybrid": lambda query: hybrid_search(query, k=POOL_SIZE, with_citations=False),
    "hybrid_reranked": lambda query: hybrid_search_reranked(
        query, k=POOL_SIZE, candidate_pool_size=RERANK_CANDIDATE_POOL_SIZE
    ),
}


def _save_progress(qrels: dict, runs: dict, queries_done: int) -> None:
    # Written after every query so a crash mid-run doesn't lose everything —
    # this file is scratch/resilience only, not a deliverable (gitignored).
    PROGRESS_FILE.write_text(
        json.dumps({"queries_done": queries_done, "qrels": qrels, "runs": runs}, default=str, indent=2),
        encoding="utf-8",
    )


def run_eval() -> dict[str, dict[str, float]]:
    """Runs every config against every eval query, judges relevance with
    whichever LLM provider is configured, and computes real Precision@K/
    Recall@K/MRR/NDCG via ranx. Every config is scored against the same
    qrels per query (the union of judged-relevant chunks across all three
    configs' pools) — an honest proxy for recall, since there is no
    exhaustive corpus-wide relevance labeling to compute it against.

    Searches are unscoped (no filing_id), so with several companies in the
    KB the eval also exercises cross-company discrimination — the NVIDIA-
    worded queries have to find NVIDIA's chunks among everyone else's."""
    qrels: dict[str, dict[str, dict[str, int]]] = {name: {} for name in CONFIGS}
    runs: dict[str, dict[str, dict[str, float]]] = {name: {} for name in CONFIGS}

    overall_start = time.perf_counter()

    for i, eval_query in enumerate(EVAL_QUERIES):
        query = eval_query.query
        query_id = f"q{i}"
        query_start = time.perf_counter()
        logger.info("[%d/%d] query: %r", i + 1, len(EVAL_QUERIES), query)

        results_by_config: dict[str, list[RetrievedChunk]] = {}
        for name, fn in CONFIGS.items():
            t0 = time.perf_counter()
            results = fn(query)
            logger.info(
                "  [%s] retrieved %d results in %.2fs", name, len(results), time.perf_counter() - t0
            )
            results_by_config[name] = results

        judged: dict = {}
        judge_start = time.perf_counter()
        judge_calls = 0
        for results in results_by_config.values():
            for r in results:
                if r.chunk_id not in judged:
                    t0 = time.perf_counter()
                    judged[r.chunk_id] = judge_relevance(query, r.text)
                    judge_calls += 1
                    logger.info(
                        "    [judge] chunk %s -> %s in %.2fs",
                        r.chunk_id,
                        judged[r.chunk_id],
                        time.perf_counter() - t0,
                    )
        logger.info(
            "  judged %d unique chunks in %.2fs", judge_calls, time.perf_counter() - judge_start
        )

        shared_qrels = {str(chunk_id): int(is_relevant) for chunk_id, is_relevant in judged.items()}

        for name, results in results_by_config.items():
            qrels[name][query_id] = shared_qrels
            runs[name][query_id] = {str(r.chunk_id): r.score for r in results}

        logger.info(
            "[%d/%d] done in %.2fs (elapsed total: %.1fs)",
            i + 1,
            len(EVAL_QUERIES),
            time.perf_counter() - query_start,
            time.perf_counter() - overall_start,
        )
        _save_progress(qrels, runs, i + 1)

    return {name: evaluate_retrieval(qrels[name], runs[name], FINAL_K) for name in CONFIGS}


def _corpus_description() -> str:
    with get_session() as session:
        complete = session.exec(
            select(func.count()).select_from(Filing).where(Filing.ingestion_status == "complete")
        ).one()
    return f"{complete} fully-ingested filing(s) in the KB, searched unscoped"


def summarize(results: dict[str, dict[str, float]]) -> str:
    # The header records the real conditions the numbers were measured
    # under — embedding model, judge, corpus size — so a later run under
    # different conditions can't be mistaken for a like-for-like comparison.
    settings = get_settings()
    lines = [
        "# Retrieval evaluation report",
        "",
        f"Queries: {len(EVAL_QUERIES)} | K={FINAL_K} | candidate pool={POOL_SIZE} | metrics via ranx",
        "",
        f"Embeddings: {settings.embedding_provider} / {settings.embedding_model} "
        f"({settings.embedding_dimension}-dim)",
        "",
        f"Judge: {settings.llm_provider} / {settings.llm_model}",
        "",
        f"Corpus: {_corpus_description()}",
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
