import math


def precision_at_k(relevances: list[bool], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = relevances[:k]
    return sum(top_k) / k


def recall_at_k(relevances: list[bool], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    top_k = relevances[:k]
    return sum(top_k) / total_relevant


def mrr(relevances: list[bool]) -> float:
    for rank, is_relevant in enumerate(relevances, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(relevances: list[bool], k: int) -> float:
    return sum(
        (1.0 if rel else 0.0) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(relevances: list[bool], k: int) -> float:
    dcg = _dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0
