# 🧠 RAG Quality Checker

A **multi-pipeline Retrieval-Augmented Generation (RAG) evaluation system** that compares different RAG configurations and automatically determines the best-performing pipeline using an LLM-based evaluator.

This project focuses on **evaluating RAG quality**, not just generating answers.

---

## 🚀 What This Project Does

- Builds **multiple RAG pipelines** with different:
  - chunk sizes
  - chunk overlaps
  - embedding models
- Runs the **same question** through all pipelines
- Uses an **LLM-as-a-Judge** to:
  - score each pipeline’s answer
  - detect hallucinations
  - select the best pipeline
- Provides:
  - a FastAPI backend
  - a Streamlit dashboard for visualization


## 🔑 Key Features

- ✅ Multi-pipeline RAG comparison
- ✅ Configurable chunking & embeddings
- ✅ Context-grounded answer generation
- ✅ LLM-based evaluation & scoring
- ✅ Pipeline winner selection

## 🧪 RAG Pipelines Used

| Pipeline | Chunk Size | Overlap | Embedding Model |
|--------|-----------|---------|----------------|
| A | 300 | 50 | all-MiniLM-L6-v2 |
| B | 500 | 100 | all-mpnet-base-v2 |
| C | 800 | 150 | paraphrase-MiniLM-L6-v2 |
| D | 1200 | 200 | paraphrase-mpnet-base-v2 |

## ⚙️ Tech Stack

### Backend
- FastAPI
- LangChain
- FAISS
- HuggingFace sentence-transformers
- OpenAI / Ollama (pluggable LLM)
  
### Frontend
- Streamlit
- Requests



