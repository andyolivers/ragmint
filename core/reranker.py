from typing import List, Dict, Any
import numpy as np


class Reranker:
    """
    Supports:
      - MMR (Maximal Marginal Relevance)
      - Dummy CrossEncoder (for demonstration)
    """

    def __init__(self, mode: str = "mmr", lambda_param: float = 0.5):
        self.mode = mode
        self.lambda_param = lambda_param

    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.mode == "crossencoder":
            return self._crossencoder_rerank(query, docs)
        return self._mmr_rerank(query, docs)

    def _mmr_rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected = []
        remaining = docs.copy()
        while remaining and len(selected) < len(docs):
            if not selected:
                best = max(remaining, key=lambda d: d["score"])
            else:
                diversity_scores = [self._similarity(d["text"], s["text"]) for d in selected for d in remaining]
                best = max(
                    remaining,
                    key=lambda d: self.lambda_param * d["score"]
                    - (1 - self.lambda_param) * np.mean([self._similarity(d["text"], s["text"]) for s in selected]),
                )
            selected.append(best)
            remaining.remove(best)
        return selected

    def _crossencoder_rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for d in docs:
            d["score"] += np.random.uniform(0, 0.1)  # placeholder for model score
        return sorted(docs, key=lambda d: d["score"], reverse=True)

    def _similarity(self, a: str, b: str) -> float:
        return np.random.rand()  # placeholder similarity
