# RAG API

A lightweight Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, and Ollama. This service enables semantic search and question-answering over a knowledge base by combining vector similarity search with large language model generation.

## Problem Statement and Motivation

Traditional keyword-based search systems often fail to capture semantic meaning and context, leading to suboptimal results. This RAG API addresses this limitation by:

- Enabling semantic search over unstructured text documents
- Providing context-aware question-answering capabilities
- Allowing dynamic knowledge base updates without full system restarts
- Offering a simple REST API interface for integration with other applications

## Key Features

- **Semantic Query Processing**: Query the knowledge base using natural language questions
- **Dynamic Knowledge Base**: Add new documents to the knowledge base via API endpoints
- **Vector-Based Retrieval**: Uses ChromaDB for efficient similarity search
- **LLM-Powered Answers**: Generates contextual answers using Ollama's tinyllama model
- **Persistent Storage**: ChromaDB persists embeddings locally for data durability

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **ChromaDB**: Open-source vector database for embeddings storage and similarity search
- **Ollama**: Local LLM inference engine (using tinyllama model)
- **Python**: Core programming language

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ HTTP POST
       ▼
┌─────────────────┐
│   FastAPI App   │
│  (app.py)       │
└──────┬──────────┘
       │
       ├──────────────┐
       ▼              ▼
┌─────────────┐  ┌──────────┐
│  ChromaDB    │  │  Ollama  │
│  (Vector DB) │  │   (LLM)  │
└─────────────┘  └──────────┘
```

### Workflow

1. **Initial Setup**: Run `embed.py` to populate the knowledge base with initial documents
2. **Query Flow**:
   - Client sends a question to `/query` endpoint
   - FastAPI queries ChromaDB for semantically similar documents
   - Retrieved context is passed to Ollama along with the question
   - Ollama generates an answer based on the context
   - Response is returned to the client
3. **Knowledge Addition**: Use `/add` endpoint to dynamically add new content to the knowledge base

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
   pip install fastapi uvicorn chromadb ollama
   ```

4. **Ensure Ollama is running and tinyllama model is available**:
   ```bash
   ollama pull tinyllama
   ```

### Build/Initialize Knowledge Base

Run the embedding script to populate the knowledge base with initial documents:

```bash
python embed.py
```

This will read `k8s.txt` and store it in ChromaDB. You can modify `embed.py` to add your own documents.

### Run the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### POST `/query`
Query the knowledge base with a question.

**Request**:
```json
{
  "q": "What is Kubernetes?"
}
```

**Response**:
```json
{
  "answer": "Kubernetes is a container orchestration platform..."
}
```

#### POST `/add`
Add new content to the knowledge base.

**Request**:
```json
{
  "text": "Your document content here..."
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Content added to knowledge base",
  "id": "uuid-here"
}
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Assumptions and Limitations

### Assumptions

- Ollama service is running locally and accessible
- tinyllama model is pre-downloaded and available
- Documents are provided as plain text (no preprocessing for PDFs, markdown, etc.)
- Single ChromaDB collection is sufficient for the use case
- Local file system storage is acceptable (ChromaDB persistent client)

### Limitations

- **No authentication/authorization**: API endpoints are publicly accessible
- **Single result retrieval**: Currently returns only the top 1 most similar document
- **No document chunking**: Large documents are stored as single entries, which may impact retrieval quality
- **No metadata filtering**: Cannot filter queries by document metadata or categories
- **Local-only deployment**: Designed for local development; not configured for production deployment
- **Error handling**: Basic error handling; may not cover all edge cases
- **No rate limiting**: API endpoints have no request throttling
- **Model dependency**: Tied to Ollama's tinyllama model; changing models requires code modification

## License

[Specify your license here]
