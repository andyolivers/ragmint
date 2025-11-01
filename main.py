from ragmint.tuner import RAGMint

def main():
    rag = RAGMint(
        docs_path="experiments/corpus",
        retrievers=["faiss"],
        embeddings=["openai/text-embedding-3-small"],
        rerankers=["mmr"],
    )

    best, results = rag.optimize(
        validation_set="experiments/validation_qa.json",
        metric="faithfulness",
        search_type="bayesian",  # 'grid' | 'random' | 'bayesian'
        trials=10,
    )

    print("Best config found:\n", best)

if __name__ == "__main__":
    main()
