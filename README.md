# Ragmint

![](/images/ragmint.png)

**Ragmint** (Retrieval-Augmented Generation Model Inspection & Tuning) is a modular Python library for optimizing, evaluating, and tuning RAG (Retrieval-Augmented Generation) pipelines.

It provides:
- ✅ Automated hyperparameter optimization (Grid, Random, Bayesian)
- 🔍 RAG evaluation metrics (faithfulness, recall, latency, BLEU, ROUGE)
- ⚙️ Flexible retrievers (FAISS, Chroma, ElasticSearch)
- 🧩 Embedding wrappers (OpenAI, HuggingFace)
- 🧠 Rerankers (MMR, CrossEncoder)
- 💾 Caching, logging, and reproducible experiments

---

## 🚀 Quick Start

### 1️⃣ Install
```bash
git clone https://github.com/yourusername/ragmint.git
cd ragmint
pip install -e .
```

> The `-e` flag installs in editable mode for local development.

---

### 2️⃣ Run an experiment
```bash
python ragmint/main.py --config configs/default.yaml --search bayesian
```

---

### 3️⃣ Evaluate your RAG pipeline manually
```python
from ragmint.core.pipeline import RAGPipeline

pipeline = RAGPipeline({
    "embedding_model": "text-embedding-3-small",
    "retriever": "faiss",
})
result = pipeline.run("What is retrieval-augmented generation?")
print(result)
```

---

### 🧩 Folder Structure
```
ragmint/
├── tuner.py
├── core/
├── utils/
├── configs/
├── experiments/
├── tests/
└── main.py
```

---

## 🧪 Run Tests
```bash
pytest -v
```

---

## ⚙️ License
Licensed under the **Apache License 2.0** — free for personal, research, and commercial use.

---

## 👤 Author
**André Oliveira**  
[andyolivers.com](https://andyolivers.com)  
Data Scientist | AI Engineer
