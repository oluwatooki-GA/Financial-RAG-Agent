# Retrieval evaluation report

Queries: 10 | K=4 | candidate pool=5 | judge=Ollama local LLM (see llm/factory.py) | metrics computed via ranx

Recall is computed against the union of judged-relevant chunks across all three configs' pools for each query (no exhaustive corpus-wide ground truth exists).

| Config | Precision@K | Recall@K | MRR | NDCG@K |
|---|---|---|---|---|
| baseline_vector | 0.200 | 0.500 | 0.432 | 0.357 |
| hybrid | 0.250 | 0.700 | 0.578 | 0.559 |
| hybrid_reranked | 0.225 | 0.675 | 0.467 | 0.482 |
