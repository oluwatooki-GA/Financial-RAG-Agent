from financial_rag_agent.retrieval.vector_retriever import baseline_vector_search

QUESTIONS = [
    "What was NVIDIA's total revenue for fiscal year 2026?",
    "What are the key risk factors related to supply chain and manufacturing?",
    "Describe NVIDIA's Data Center segment performance.",
]

if __name__ == "__main__":
    for question in QUESTIONS:
        print(f"\n=== {question}")
        results = baseline_vector_search(question, k=5)
        scores = [r.score for r in results]
        assert len(set(scores)) == len(scores), "similarity scores must be distinct, not a constant"

        for r in results:
            snippet = r.text[:120].replace("\n", " ")
            print(f"  {r.score:.4f}  {r.item_label:<10} chunk={r.chunk_id}  {snippet}")

    print("\nAll queries returned distinct, real similarity scores.")
