# Company Wiki & HR Assistant API

An internal **Retrieval-Augmented Generation (RAG)** API built with FastAPI, ChromaDB, and Ollama — specialized for company knowledge management. This service enables employees to search and ask questions about **company policies, benefits, onboarding guides, procedures, and organizational information** using hybrid search with re-ranking.

## Problem Statement and Motivation

Employees often struggle to find the right policy document, benefit details, or procedural guide buried across multiple internal systems. This HR Assistant API solves this by:

- Providing a single, searchable knowledge base for all company documents
- **Department-scoped search** — filter results by HR, Engineering, Finance, Legal, etc.
- **Category-aware retrieval** — search by policy, onboarding, benefits, FAQ, and more
- Context-aware question-answering with HR-tuned prompts and source citations
- Dynamic knowledge base updates without system restarts
- Professional, disclaimered responses that reference specific policy documents

## Key Features

- **Department & Category Filtering**: Scope queries to specific departments and document types
- **HR-Tuned LLM Prompts**: Professional tone with policy citations and disclaimers
- **Advanced Hybrid Search**: Combines Dense Vector Search with BM25 Keyword Search
- **Reciprocal Rank Fusion (RRF)**: Merges results from multiple search strategies
- **Cross-Encoder Re-ranking**: High-precision re-ranking layer for superior relevance
- **Document Processing**: Support for PDF, DOCX, and Markdown formats
- **Smart Chunking**: Multiple chunking strategies (recursive, semantic, PDF page-aware)
- **Policy Metadata**: Track effective dates, authors, departments, and categories
- **Web UI**: Built-in interactive assistant interface
- **API Versioning**: Versioned endpoints at `/v1`
- **Observability**: Prometheus metrics (`/metrics`) and health checks (`/health`)
- **Rate Limiting**: Protection against abuse on critical endpoints

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
┌─────────────────┐
│    Employee      │
│  (Web UI / API)  │
└────────┬─────────┘
         │
         │ HTTP POST (Rate Limited)
         ▼
┌──────────────────────────────────────┐
│         FastAPI App                  │
│  (Middleware: Logs, CORS, Auth)      │
│  Title: Company Wiki & HR Assistant  │
└────────┬───────────────────┬─────────┘
         │                   │
         ▼                   ▼
┌────────────────┐   ┌──────────────┐
│ Hybrid Search  │   │  HR-Tuned    │
│ (Vector + BM25)│   │  LLM Prompt  │
│ + Dept/Cat     │   │  + Disclaimer│
│   Filters      │   │(CrossEncoder)│
└────────┬───────┘   └──────┬───────┘
         │                  │
         ▼                  ▼
┌──────────────┐    ┌──────────┐
│  ChromaDB    │    │  Ollama  │
│ (company_wiki│    │   (LLM)  │
│  collection) │    │          │
└──────────────┘    └──────────┘
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
The Web UI (HR Assistant) is served at the root `/`.

### API Endpoints

The API is versioned. The current stable version is `v1`.

#### Core Endpoints

- `GET /v1/`: API Information
- `POST /v1/query`: Search company knowledge base with department/category filters
- `POST /v1/add`: Add text content with department, category, and metadata
- `POST /v1/upload`: Upload and process documents (PDF/DOCX/MD/TXT)
- `POST /v1/rebuild-index`: Manually rebuild BM25 index

#### Observability & Management

- `GET /health`: Health check (Status & Version)
- `GET /metrics`: Prometheus metrics
- `GET /docs`: OpenAPI/Swagger Documentation
- `GET /redoc`: ReDoc Documentation

### Example Usage

#### Querying (with Department Filter)

```json
POST /v1/query
{
  "q": "What is our PTO policy?",
  "mode": "hybrid",
  "n_results": 5,
  "rerank": true,
  "department": "hr",
  "category": "policy"
}
```

**Response**:
```json
{
  "answer": "According to the PTO Policy, full-time employees accrue 15 days of paid time off per year...",
  "sources": [...],
  "search_mode": "hybrid",
  "reranked": true,
  "total_results": 5,
  "department_filter": "hr",
  "category_filter": "policy",
  "disclaimer": "This information is provided for reference only. For official decisions, please contact your HR representative or department lead."
}
```

#### Adding Knowledge

```json
POST /v1/add
{
  "text": "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval...",
  "department": "hr",
  "category": "policy",
  "author": "HR Team",
  "effective_date": "2026-01-01"
}
```

#### Uploading a Document

```bash
curl -X POST "http://localhost:8000/v1/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@employee_handbook.pdf" \
  -F "strategy=recursive" \
  -F "department=hr" \
  -F "category=handbook"
```

### Departments

| Value | Description |
|-------|-------------|
| `hr` | Human Resources |
| `engineering` | Engineering |
| `finance` | Finance |
| `legal` | Legal |
| `operations` | Operations |
| `marketing` | Marketing |
| `all` | Search across all departments (query only) |

### Document Categories

| Value | Description |
|-------|-------------|
| `policy` | Company policies (PTO, remote work, etc.) |
| `onboarding` | New hire onboarding guides |
| `benefits` | Benefits information (health, 401k, etc.) |
| `org-chart` | Organizational charts and team structure |
| `handbook` | Employee handbooks |
| `procedure` | Standard operating procedures |
| `faq` | Frequently asked questions |
| `announcement` | Company announcements |

## Assumptions and Limitations

- **Local Deployment**: System is configured for local execution with ChromaDB in persistent mode.
- **Model Dependencies**: Requires Ollama running locally.
- **Rate Limits**:
  - `/query`: 10 requests/minute
  - `/add` & `/upload`: 5 requests/minute
- **Auth**: No authentication enabled by default (add JWT/OAuth for production).
- **Disclaimer**: All responses include a disclaimer — this is an assistant, not a legal authority.
