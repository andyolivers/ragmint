from ragmint.core.evaluation import Evaluator


def test_evaluator_similarity():
    evaluator = Evaluator()
    result = evaluator.evaluate(
        query="What is RAG?",
        answer="Retrieval-Augmented Generation method",
        context="Retrieval-Augmented Generation improves factual accuracy."
    )
    assert "faithfulness" in result
    assert result["faithfulness"] >= 0.0
