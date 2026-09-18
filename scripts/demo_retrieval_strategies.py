import _bootstrap  # noqa: F401

from sqlmodel import select

from financial_rag_agent.core import Company, Filing, get_session
from financial_rag_agent.retrieval.hybrid_retriever import hybrid_search, hybrid_search_reranked
from financial_rag_agent.retrieval.vector_retriever import baseline_vector_search

QUERY = "What was NVIDIA's total revenue and what drove the increase?"


def show(label, results):
    print(f"\n=== {label} ===")
    for r in results:
        heading = r.item_heading or "(no heading)"
        print(f"  score={r.score:.4f}  {heading}")
        print(f"    {r.text[:120].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    with get_session() as session:
        nvda_filing = session.exec(select(Filing).join(Company).where(Company.ticker == "NVDA")).first()

    if nvda_filing is None:
        raise SystemExit("NVIDIA isn't ingested in this DB yet -- run scripts/ingest_nvidia_10k.py first.")

    print(f"Query: {QUERY!r}")
    print("Comparing the three retrieval strategies against the same real NVIDIA 10-K.")
    print("hybrid is the production default because it measurably won on every metric")
    print("in eval_report.md -- this is that same comparison, live, on one query.")

    show("baseline (pure vector similarity)", baseline_vector_search(QUERY, k=3, filing_id=nvda_filing.id))
    show("hybrid (vector + BM25 + RRF)", hybrid_search(QUERY, k=3, filing_id=nvda_filing.id))
    show("hybrid + reranked (+ cross-encoder)", hybrid_search_reranked(QUERY, k=3, filing_id=nvda_filing.id))
