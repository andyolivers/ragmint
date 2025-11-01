import numpy as np
from ragmint.core.retriever import Retriever


def test_retriever_basic():
    docs = ["Doc one", "Doc two", "Doc three"]
    embeddings = [np.random.rand(5) for _ in docs]
    retriever = Retriever(embeddings, docs)

    results = retriever.retrieve("test query", top_k=2)
    assert len(results) == 2
    assert "text" in results[0]
