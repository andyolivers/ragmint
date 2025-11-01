import json
import random
from typing import List, Tuple
from pathlib import Path

from .core.pipeline import RAGPipeline
from .core.evaluation import Evaluator
from .utils.data_loader import load_validation_set
from .utils.logger import get_logger

logger = get_logger("RAGMint")


class RAGMint:
    """
    Main class for optimizing RAG pipeline parameters.
    Supports: grid search, random search, Bayesian optimization (via Optuna).
    """

    def __init__(
        self,
        docs_path: str,
        retrievers: List[str] = None,
        embeddings: List[str] = None,
        rerankers: List[str] = None,
        default_params: dict = None,
    ):
        self.docs_path = docs_path
        self.retrievers = retrievers or ["faiss"]
        self.embeddings = embeddings or [
            "openai/text-embedding-3-small",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
        self.rerankers = rerankers or ["mmr", "cross-encoder/ms-marco-MiniLM-L-6-v2"]
        self.default_params = default_params or {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "top_k": 5,
        }

    def optimize(
        self,
        validation_set: str,
        metric: str = "faithfulness",
        search_type: str = "bayesian",
        trials: int = 20,
        param_space: dict = None,
        random_seed: int = 42,
    ) -> Tuple[dict, list]:
        """
        Optimizes RAG parameters via Grid, Random, or Bayesian search.
        Returns (best_params, all_results).
        """
        random.seed(random_seed)

        if param_space is None:
            param_space = {
                "chunk_size": [256, 512, 1024],
                "chunk_overlap": [0, 32, 64],
                "retriever": self.retrievers,
                "embedding_model": self.embeddings,
                "reranker": self.rerankers,
                "top_k": [3, 5, 8],
            }

        validation = load_validation_set(validation_set)
        search_type = search_type.lower()

        if search_type == "grid":
            from .optimization.search import GridSearch
            searcher = GridSearch(param_space)
        elif search_type == "random":
            from .optimization.search import RandomSearch
            searcher = RandomSearch(param_space, n_trials=trials)
        elif search_type == "bayesian":
            from .optimization.search import BayesianSearch
            searcher = BayesianSearch(param_space, n_trials=trials)
        else:
            raise ValueError("search_type must be 'grid', 'random', or 'bayesian'")

        best_config, results = searcher.run(self._evaluate_config, metric, validation)

        best_config = {**self.default_params, **best_config}
        logger.info(f"✅ Best configuration found: {best_config}")

        # Save results
        results_path = Path("experiments/results")
        results_path.mkdir(parents=True, exist_ok=True)
        with open(results_path / "best_config.json", "w", encoding="utf-8") as f:
            json.dump({"best": best_config, "results": results}, f, indent=2)

        return best_config, results

    def _evaluate_config(self, config: dict, validation_set: list) -> dict:
        """
        Builds and evaluates a single pipeline configuration.
        Returns a dict of metric_name -> score.
        """
        pipeline = RAGPipeline(
            docs_path=self.docs_path,
            retriever_name=config.get("retriever"),
            embedding_model=config.get("embedding_model"),
            reranker_name=config.get("reranker"),
            chunk_size=int(config.get("chunk_size")),
            chunk_overlap=int(config.get("chunk_overlap")),
            top_k=int(config.get("top_k")),
        )
        evaluator = Evaluator(metric_names=["faithfulness", "recall"])
        return evaluator.evaluate(pipeline, validation_set)
