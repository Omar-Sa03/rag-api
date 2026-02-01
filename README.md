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
- **Document Processing**: Support for PDF, DOCX, and Markdown file formats
- **Smart Chunking**: Multiple chunking strategies (recursive, semantic, PDF page-aware)
- **Metadata Preservation**: Track source, page numbers, and sections for each chunk
- **Hybrid Search**: Combines Dense Vector Search with BM25 Keyword Search
- **RRF Fusion**: Combines search results using Reciprocal Rank Fusion
- **Cross-Encoder Re-ranking**: High-precision re-ranking layer for better accuracy
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
   pip install -r requirements.txt
   ```
   
   This will install:
   - fastapi, uvicorn (API framework)
   - chromadb (vector database)
   - ollama (LLM integration)
   - PyPDF2, pdfplumber (PDF processing)
   - python-docx (DOCX processing)
   - langchain-text-splitters (chunking)
   - python-multipart (file uploads)
   - rank-bm25 (keyword search)
   - sentence-transformers (re-ranking)

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

#### GET `/`
Get API information and available endpoints.

**Response**:
```json
{
  "name": "RAG API",
  "version": "2.0",
  "description": "Retrieval-Augmented Generation API with document processing",
  "endpoints": {...},
  "supported_formats": [".pdf", ".docx", ".md", ".txt"]
}
```

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
  "answer": "Kubernetes is a container orchestration platform...",
  "sources": [
    {
      "text_preview": "First 100 chars of source...",
      "metadata": {
        "source": "k8s.pdf",
        "page_number": 1,
        "chunk_index": 0
      }
    }
  ]
}
```

#### POST `/add`
Add new text content to the knowledge base.

**Request**:
```json
{
  "text": "Your document content here...",
  "chunk": true,
  "strategy": "recursive"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Content chunked and added to knowledge base (5 chunks)",
  "chunks": 5,
  "ids": ["uuid1", "uuid2", ...]
}
```

#### POST `/upload`
Upload and process a document (PDF, DOCX, Markdown).

**Request** (multipart/form-data):
- `file`: Document file to upload
- `strategy`: Chunking strategy ("recursive", "semantic", "pdf_page_aware")
- `chunk_size`: Maximum chunk size in characters (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "strategy=recursive" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

**Response**:
```json
{
  "status": "success",
  "message": "Document processed and added to knowledge base",
  "filename": "document.pdf",
  "file_type": "pdf",
  "chunks": 12,
  "metadata": {
    "source": "document.pdf",
    "pages": 5,
    "file_type": "pdf"
  },
  "chunk_ids": ["uuid1", "uuid2", ...]
}
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Chunking Strategies

The API supports three chunking strategies:

### 1. Recursive Character Splitting (Default)
- Splits text by paragraphs, then sentences, then characters
- Best for general-purpose text processing
- Preserves natural text boundaries

### 2. Semantic Chunking
- Splits text by sentences and groups them semantically
- Better for question-answering tasks
- Maintains sentence coherence

### 3. PDF Page-Aware Chunking
- Preserves PDF page boundaries
- Chunks within each page using recursive splitting
- Maintains page number metadata for citations

All strategies support configurable chunk size and overlap for context preservation.

## Assumptions and Limitations

### Assumptions

- Ollama service is running locally and accessible
- tinyllama model is pre-downloaded and available
- Single ChromaDB collection is sufficient for the use case
- Local file system storage is acceptable (ChromaDB persistent client)
- Uploaded files are temporarily stored during processing

### Limitations

- **No authentication/authorization**: API endpoints are publicly accessible
- **Multiple result retrieval**: Returns top 3 most similar chunks (configurable)
- **No metadata filtering**: Cannot filter queries by document metadata or categories
- **Local-only deployment**: Designed for local development; not configured for production deployment
- **Error handling**: Basic error handling; may not cover all edge cases
- **No rate limiting**: API endpoints have no request throttling
- **Model dependency**: Tied to Ollama's tinyllama model; changing models requires code modification
- **File size limits**: No explicit file size limits; large files may cause memory issues

