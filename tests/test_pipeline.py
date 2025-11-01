from ragmint.core.pipeline import RAGPipeline
from ragmint.core.retriever import Retriever
from ragmint.core.reranker import Reranker
from ragmint.core.evaluation import Evaluator


class DummyGenerator:
    def generate(self, query, context):
        return f"Answer for: {query}"


def test_pipeline_run():
    retriever = Retriever([[0.1, 0.2], [0.3, 0.4]], ["Doc 1", "Doc 2"])
    reranker = Reranker()
    generator = DummyGenerator()
    evaluator = Evaluator()

    pipeline = RAGPipeline(retriever, reranker, generator, evaluator)
    result = pipeline.run("What is AI?", top_k=1)

    assert "answer" in result
    assert "metrics" in result
