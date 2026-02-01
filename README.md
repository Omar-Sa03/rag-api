# RAG API

A comprehensive Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, and Ollama. This service enables semantic search and question-answering over a knowledge base by combining vector similarity search, BM25 keyword search, and reciprocal rank fusion with a re-ranking layer.

## Problem Statement and Motivation

Traditional keyword-based search systems often fail to capture semantic meaning and context, leading to suboptimal results. This RAG API addresses this limitation by:

- Enabling semantic search over unstructured text documents
- Combining the precision of keyword search (BM25) with the recall of vector search
- Providing context-aware question-answering capabilities
- Allowing dynamic knowledge base updates without full system restarts
- Offering a robust, versioned REST API with comprehensive observability

## Key Features

- **Advanced Hybrid Search**: Combines Dense Vector Search with BM25 Keyword Search
- **Reciprocal Rank Fusion (RRF)**: Merges results from multiple search strategies
- **Cross-Encoder Re-ranking**: High-precision re-ranking layer for superior relevance
- **Semantic Query Processing**: Query the knowledge base using natural language
- **Document Processing**: Support for PDF, DOCX, and Markdown formats
- **Smart Chunking**: Multiple chunking strategies (recursive, semantic, PDF page-aware)
- **API Versioning**: versioned endpoints at `/v1` and `/v2`
- **Structured Logging**: JSON-formatted logs for better observability
- **Observability**: Prometheus metrics (`/metrics`) and health checks (`/health`)
- **Rate Limiting**: Protection against abuse on critical endpoints
- **Validation**: Strict Request/Response validation using Pydantic models
- **OpenAPI Documentation**: Fully documented API with examples

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **ChromaDB**: Vector database for embeddings storage and similarity search
- **Ollama**: Local LLM inference engine (using tinyllama model)
- **Rank-BM25**: Keyword search library
- **Sentence-Transformers**: For cross-encoder re-ranking
- **Structlog**: Structured logging
- **Slowapi**: Rate limiting
- **Prometheus-Fastapi-Instrumentator**: Metrics exposure

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ HTTP POST (Rate Limited)
       ▼
┌─────────────────────────────────┐
│           FastAPI App           │
│  (Middleware: Logs, CORS, Auth) │
└──────┬────────────────┬─────────┘
       │                │
       ▼                ▼
┌──────────────┐   ┌──────────────┐
│ Hybrid Search│   │  Re-Ranker   │
│ (Vector+BM25)│   │(CrossEncoder)│
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
┌─────────────┐    ┌──────────┐
│  ChromaDB   │    │  Ollama  │
│ (Vector DB) │    │   (LLM)  │
└─────────────┘    └──────────┘
```

## How to Run

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- tinyllama model downloaded in Ollama (run `ollama pull tinyllama`)

### Setup

1. **Clone the repository**:
   ```bash
   cd rag-api
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running**:
   ```bash
   ollama pull tinyllama
   ```

### Run the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

### API Endpoints

The API is versioned. The current stable version is `v1`.

#### Core Endpoints

- `GET /v1/`: API Information
- `POST /v1/query`: Hybrid search with LLM generation
- `POST /v1/add`: Add raw text content
- `POST /v1/upload`: Upload and process files
- `POST /v1/rebuild-index`: Manually rebuild BM25 index

#### Observability & Management

- `GET /health`: Health check (Status & Version)
- `GET /metrics`: Prometheus metrics
- `GET /docs`: OpenAPI/Swagger Documentation
- `GET /redoc`: ReDoc Documentation

### Example Usage

#### Querying (Hybrid Search)

```json
POST /v1/query
{
  "q": "What is Kubernetes?",
  "mode": "hybrid",
  "n_results": 5,
  "rerank": true
}
```

**Response**:
```json
{
  "answer": "Kubernetes is a container orchestration platform...",
  "sources": [...],
  "search_mode": "hybrid",
  "reranked": true,
  "total_results": 5
}
```

#### Uploading a Document

```bash
curl -X POST "http://localhost:8000/v1/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@manual.pdf" \
  -F "strategy=recursive"
```

## Assumptions and Limitations

- **Local Deployment**: System is configured for local execution with ChromaDB in persistent mode.
- **Model Dependencies**: Requires Ollama running locally.
- **Rate Limits**:
  - `/query`: 10 requests/minute
  - `/add` & `/upload`: 5 requests/minute
- **Auth**: No authentication enabled by default (add JWT/OAuth for production).
