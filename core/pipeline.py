from typing import Any, Dict, List
from .retriever import Retriever
from .reranker import Reranker
from .evaluation import Evaluator
from .embeddings import EmbeddingModel
from .chunking import Chunker


class RAGPipeline:
    """
    Core Retrieval-Augmented Generation pipeline.
    Connects retriever → reranker → generator → evaluator.
    """

    def __init__(self, retriever: Retriever, reranker: Reranker, generator: Any, evaluator: Evaluator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.evaluator = evaluator

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # Step 1: Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        # Step 2: Rerank documents
        reranked_docs = self.reranker.rerank(query, retrieved_docs)

        # Step 3: Generate answer
        context = "\n".join([d["text"] for d in reranked_docs])
        answer = self.generator.generate(query, context)

        # Step 4: Evaluate
        eval_metrics = self.evaluator.evaluate(query, answer, context)

        return {
            "query": query,
            "answer": answer,
            "docs": reranked_docs,
            "metrics": eval_metrics,
        }
