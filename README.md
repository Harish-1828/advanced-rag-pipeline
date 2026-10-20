# Comprehensive RAG (Retrieval-Augmented Generation) Pipeline

An end-to-end implementation of advanced Retrieval-Augmented Generation (RAG) techniques using Python, LangChain, Sentence-Transformers, ChromaDB, and local LLMs (via Ollama).

---

## 📑 Overview

This project explores the progressive implementation of RAG systems, starting from fundamental manual document ingestion and cosine similarity to advanced query transformation and fusion retrieval techniques.

### Key Capabilities Demonstrated:
1. **Document Parsing**: Extracting raw text from PDF documents using `pypdf`.
2. **Chunking Strategy**: Fixed-size windowing (`500` characters) with configurable overlap (`100` characters) to preserve boundary context.
3. **Dense Vector Embeddings**: Generating embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
4. **Vector Store Integration**: Storing and indexing document embeddings using **ChromaDB**.
5. **Local LLM Inference**: Offline generation powered by **Ollama** (`llama2`).
6. **Advanced Retrieval Techniques**:
   - **Multi-Query Retrieval**: Generating query variations using prompt engineering to capture alternative phrasings and improve recall.
   - **RAG-Fusion**: Executing multiple queries against the vector store and re-ranking documents using **Reciprocal Rank Fusion (RRF)**.
   - **Query Decomposition**: Deconstructing complex composite queries into distinct sub-questions for granular information retrieval.

---

## 🏗️ Architecture Flow

```
[ PDF Document ]
       │
       ▼
[ Text Extraction (pypdf) ]
       │
       ▼
[ Sliding Window Chunking (500 chars / 100 overlap) ]
       │
       ▼
[ Embeddings: all-MiniLM-L6-v2 ] ──► [ Vector Store: ChromaDB ]
                                               ▲
                                               │ (Similarity Search / RRF)
                                               │
[ User Query ] ──► [ Query Transformation ] ───┘
                         ├── Multi-Query Expansion
                         ├── RAG-Fusion (RRF Ranking)
                         └── Query Decomposition
                                   │
                                   ▼
                       [ Retrieved Context Chunks ]
                                   │
                                   ▼
                    [ LLM Prompt (Ollama / LLaMA 2) ]
                                   │
                                   ▼
                            [ Final Answer ]
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally with the target model:
  ```bash
  ollama pull llama2
  ```

### 2. Installation
Install the necessary Python dependencies:

```bash
pip install pypdf sentence-transformers scikit-learn numpy chromadb
pip install langchain-core langchain-classic langchain-community langchain-chroma langchain-huggingface langchain-ollama
```

---

## 🔬 Deep-Dive: Advanced Retrieval Techniques

### 1. Multi-Query Retrieval
Standard similarity searches often miss relevant chunks if the user's vocabulary differs from the source document. Multi-Query retrieval prompts the LLM to generate alternative formulations of the user's question, querying the vector database across each phrasing to maximize recall.

### 2. RAG-Fusion & Reciprocal Rank Fusion (RRF)
RAG-Fusion extends Multi-Query by applying a statistical re-ranking algorithm across all retrieved result lists:
$$\text{RRF Score}(d) = \sum_{q \in Q} \frac{1}{k + \text{rank}(q, d)}$$
*(where $k = 60$ is a constant smoothing factor).*
Documents that rank consistently high across multiple query variations are prioritized, providing a more robust context to the generator.

### 3. Query Decomposition
Breaks multi-faceted inquiries into independent sub-questions, executes parallel searches, and synthesizes a structured, comprehensive answer.

---

## 🔒 Security Notice
- Ensure that sensitive keys or environment files (`.env`, secret tokens) are never pushed to public repositories. This repository includes a preconfigured `.gitignore` to prevent secret leaks.
