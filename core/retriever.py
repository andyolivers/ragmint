from typing import List, Dict, Any
import numpy as np


class Retriever:
    """
    Simple in-memory vector retriever using FAISS-like cosine similarity.
    Replace with Chroma or ElasticSearch backend if needed.
    """

    def __init__(self, embeddings: List[np.ndarray], documents: List[str]):
        self.embeddings = np.array(embeddings)
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self._embed(query)
        scores = self._cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [{"text": self.documents[i], "score": float(scores[i])} for i in top_indices]

    def _embed(self, query: str) -> np.ndarray:
        # Dummy embedder — replace with EmbeddingModel.encode()
        return np.random.rand(self.embeddings.shape[1])

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm)
