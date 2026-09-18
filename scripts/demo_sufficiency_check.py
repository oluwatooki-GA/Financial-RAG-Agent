import _bootstrap  # noqa: F401

from financial_rag_agent.retrieval.sufficiency import check_retrieval_sufficiency

CASES = [
    (
        "NVIDIA's own real revenue question, scoped to NVIDIA's real 10-K",
        "What was NVIDIA revenue in fiscal year 2026?",
        "NVIDIA CORP",
    ),
    (
        "A company with zero data in the knowledge base",
        "What was the profit after tax?",
        "Totally Fake Company Inc",
    ),
    (
        "A company that's known but hasn't been chunked/embedded yet",
        "What was the profit after tax in 2025?",
        "GTCO",
    ),
]

if __name__ == "__main__":
    print("The Phase 4 sufficiency gate: decides whether the KB has good enough")
    print("evidence to answer a question, or whether runtime document discovery")
    print("should run instead. Three real scenarios against the actual dev DB:\n")

    for label, query, company in CASES:
        result = check_retrieval_sufficiency(query, company_name=company)
        print(f"[{label}]")
        print(f"  query: {query!r}")
        print(f"  sufficient = {result.sufficient}")
        print(f"  reason: {result.reason}\n")
