# Retrieval evaluation report

Queries: 10 | K=4 | candidate pool=5 | metrics via ranx

Embeddings: sentence-transformers / sentence-transformers/all-MiniLM-L6-v2 (384-dim)

Judge: anthropic / claude-haiku-4-5-20251001

Corpus: 3 fully-ingested filing(s) in the KB, searched unscoped

Recall is computed against the union of judged-relevant chunks across all three configs' pools for each query (no exhaustive corpus-wide ground truth exists).

| Config | Precision@K | Recall@K | MRR | NDCG@K |
|---|---|---|---|---|
| baseline_vector | 0.250 | 0.227 | 0.342 | 0.264 |
| hybrid | 0.425 | 0.484 | 0.583 | 0.526 |
| hybrid_reranked | 0.475 | 0.664 | 0.620 | 0.597 |
