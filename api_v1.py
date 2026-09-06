from fastapi import APIRouter, UploadFile, File, Form, Request, BackgroundTasks
from typing import Optional
import uuid
import time
import structlog
import ollama

from schemas import (
    QueryRequest, QueryResponse, AddKnowledgeRequest, AddKnowledgeResponse,
    RebuildIndexResponse, UploadResponse, DepartmentEnum, CategoryEnum
)
from services import hybrid_search, collection, doc_processor, chunking_manager
from hybrid_search import format_search_results
from exceptions import (
    DocumentProcessingError, VectorDBError, LLMGenerationError
)
from limiter import limiter
from rag_api.core.metrics import (
    RAG_QUERY_LATENCY,
    RAG_RERANK_SCORE,
    RAG_DOCUMENTS_INDEXED,
    RAG_SEARCH_MODE,
    RAG_LLM_ERRORS,
    RAG_SOURCES_RETURNED,
)

logger = structlog.get_logger()
router = APIRouter()

# --- HR Assistant System Prompt ---

HR_SYSTEM_PROMPT = """You are the Company Wiki & HR Assistant, an internal knowledge assistant for employees.

Your role:
- Answer employee questions about company policies, benefits, onboarding, procedures, and organizational information.
- Always base your answers on the provided context documents. Do NOT invent information.
- Reference specific policy or document names when available (e.g. "According to the Remote Work Policy...").
- Use a professional, friendly, and helpful tone.
- If the context does not contain enough information to fully answer, say so clearly.
- Always end with the disclaimer: "For official decisions, please contact your HR representative or department lead."

Context from company documents:
{context}

Employee Question: {question}

Answer clearly and helpfully:"""

DISCLAIMER = "This information is provided for reference only. For official decisions, please contact your HR representative or department lead."


def _build_search_filters(
    department: Optional[DepartmentEnum],
    category: Optional[CategoryEnum],
) -> Optional[dict]:
    """Build a metadata filter dict from department/category enums."""
    filters = {}
    if department and department != DepartmentEnum.ALL:
        filters["department"] = department.value
    if category:
        filters["category"] = category.value
    return filters if filters else None


@router.get("/")
async def root():
    """Root endpoint for V1 API information."""
    return {
        "version": "1.0",
        "description": "Company Wiki & HR Assistant API",
        "domain": "internal-hr",
        "features": [
            "Department-scoped search",
            "Policy & benefits retrieval",
            "Hybrid search with re-ranking",
            "Document categorization",
        ],
    }

@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    """
    Query the company knowledge base with hybrid search.
    Optionally filter by department and/or document category.
    """
    logger.info(
        "query_received",
        query=body.q,
        mode=body.mode,
        department=body.department.value if body.department else None,
        category=body.category.value if body.category else None,
    )

    # Track search mode usage
    RAG_SEARCH_MODE.labels(mode=body.mode).inc()

    reranked_label = "true" if body.rerank else "false"
    query_start = time.perf_counter()

    # Build metadata filters
    filters = _build_search_filters(body.department, body.category)

    try:
        # Perform hybrid search with optional filters
        search_results = hybrid_search.search(
            query=body.q,
            mode=body.mode,
            n_results=body.n_results,
            rerank=body.rerank,
            filters=filters,
        )

        if not search_results:
            logger.info("no_results_found", query=body.q)
            RAG_SOURCES_RETURNED.observe(0)
            RAG_QUERY_LATENCY.labels(mode=body.mode, reranked=reranked_label).observe(
                time.perf_counter() - query_start
            )
            return QueryResponse(
                answer="No relevant documents found in the company knowledge base for your query.",
                sources=[],
                search_mode=body.mode,
                reranked=body.rerank,
                total_results=0,
                department_filter=body.department.value if body.department else None,
                category_filter=body.category.value if body.category else None,
                disclaimer=DISCLAIMER,
            )

        # Track number of sources returned
        RAG_SOURCES_RETURNED.observe(len(search_results))

        # Observe top reranker score if available
        if body.rerank and search_results:
            top_score = search_results[0].get("score", search_results[0].get("rerank_score", None))
            if top_score is not None:
                RAG_RERANK_SCORE.observe(float(top_score))

        # Format results
        formatted_sources = format_search_results(search_results, include_scores=body.include_scores)

        # Extract contexts for LLM
        contexts = [result['document'] for result in search_results]
        combined_context = "\n\n---\n\n".join(contexts)

        # Generate answer with HR-tuned prompt
        answer = None
        try:
            response = ollama.generate(
                model="tinyllama",
                prompt=HR_SYSTEM_PROMPT.format(
                    context=combined_context,
                    question=body.q,
                ),
            )
            answer = response.get("response")
        except Exception as e:
            logger.warning("ollama_unavailable_fallback_to_extractive", error=str(e))
            RAG_LLM_ERRORS.labels(error_type=type(e).__name__).inc()
            
            # Graceful extractive fallback: present top retrieved policy passages directly
            top_passages = [c.strip() for c in contexts[:3] if c.strip()]
            if top_passages:
                bullet_points = "\n\n".join([f"• {p}" for p in top_passages])
                answer = (
                    f"Based on our company documents, here is the relevant information for your question:\n\n"
                    f"{bullet_points}\n\n"
                    f"*(Note: Local generative model (Ollama) is currently offline, so the verified policy excerpts above were retrieved directly. To enable generative answers, start Ollama with `ollama run tinyllama`.)*"
                )
            else:
                answer = "Relevant document references were found, but no text excerpts could be displayed."

        # Record end-to-end latency
        RAG_QUERY_LATENCY.labels(mode=body.mode, reranked=reranked_label).observe(
            time.perf_counter() - query_start
        )

        logger.info("query_processed", results_count=len(search_results))
        return QueryResponse(
            answer=answer,
            sources=formatted_sources,
            search_mode=body.mode,
            reranked=body.rerank,
            total_results=len(search_results),
            department_filter=body.department.value if body.department else None,
            category_filter=body.category.value if body.category else None,
            disclaimer=DISCLAIMER,
        )

    except LLMGenerationError:
        raise
    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise VectorDBError(f"Search failed: {str(e)}")

@router.post("/add", response_model=AddKnowledgeResponse)
@limiter.limit("5/minute")
async def add_knowledge(request: Request, body: AddKnowledgeRequest, background_tasks: BackgroundTasks):
    """Add new text content to the company knowledge base with department/category metadata."""
    logger.info(
        "add_knowledge_request",
        chunk_strategy=body.strategy if body.chunk else "none",
        department=body.department.value,
        category=body.category.value if body.category else None,
    )
    
    # Build HR-specific metadata
    hr_metadata = {
        "source": "direct_text",
        "department": body.department.value,
    }
    if body.category:
        hr_metadata["category"] = body.category.value
    if body.effective_date:
        hr_metadata["effective_date"] = body.effective_date
    if body.author:
        hr_metadata["author"] = body.author

    try:
        if body.chunk:
            chunks = chunking_manager.chunk_document(body.text, metadata=hr_metadata)
            doc_ids = []
            
            for chunk_data in chunks:
                doc_id = str(uuid.uuid4())
                clean_metadata = doc_processor.clean_metadata_for_chromadb(chunk_data['metadata'])
                
                collection.add(
                    documents=[chunk_data['text']],
                    ids=[doc_id],
                    metadatas=[clean_metadata]
                )
                doc_ids.append(doc_id)
            
            background_tasks.add_task(hybrid_search.rebuild_index)

            # Track indexed chunks
            RAG_DOCUMENTS_INDEXED.labels(source="add_text").inc(len(chunks))

            logger.info("knowledge_added", chunks=len(chunks), department=body.department.value)
            return AddKnowledgeResponse(
                status="success",
                message=f"Content chunked and added ({len(chunks)} chunks) to {body.department.value} department. Index rebuild scheduled.",
                chunks=len(chunks),
                ids=doc_ids,
                department=body.department.value,
                category=body.category.value if body.category else None,
            )
        else:
            doc_id = str(uuid.uuid4())
            clean_metadata = doc_processor.clean_metadata_for_chromadb(hr_metadata)
            collection.add(
                documents=[body.text],
                ids=[doc_id],
                metadatas=[clean_metadata],
            )
            background_tasks.add_task(hybrid_search.rebuild_index)

            # Track single document
            RAG_DOCUMENTS_INDEXED.labels(source="add_text").inc(1)

            logger.info("knowledge_added_single", department=body.department.value)
            return AddKnowledgeResponse(
                status="success",
                message=f"Content added to {body.department.value} department. Index rebuild scheduled.",
                id=doc_id,
                department=body.department.value,
                category=body.category.value if body.category else None,
            )
            
    except Exception as e:
        logger.error("add_knowledge_failed", error=str(e))
        raise VectorDBError(f"Failed to add content: {str(e)}")

@router.post("/rebuild-index", response_model=RebuildIndexResponse)
async def rebuild_index():
    """Manually rebuild the BM25 index."""
    try:
        hybrid_search.rebuild_index()
        logger.info("index_rebuilt")
        return RebuildIndexResponse(status="success", message="BM25 index rebuilt successfully")
    except Exception as e:
        logger.error("rebuild_index_failed", error=str(e))
        raise VectorDBError(f"Index rebuild failed: {str(e)}")

@router.post("/upload", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    department: str = Form(..., description="Department this document belongs to"),
    category: Optional[str] = Form(None, description="Document category (e.g. policy, onboarding)"),
):
    """Upload and process a document into the company knowledge base."""
    logger.info("upload_request", filename=file.filename, department=department)
    
    try:
        file_ext = file.filename.split('.')[-1].lower()
        if f'.{file_ext}' not in doc_processor.SUPPORTED_FORMATS:
            raise DocumentProcessingError(
                f"Unsupported file format: {file_ext}. Supported: {doc_processor.SUPPORTED_FORMATS}"
            )
        
        content = await file.read()
        processed = doc_processor.process_uploaded_file(content, file.filename)
        
        # Inject HR-specific metadata
        processed['metadata']['department'] = department
        if category:
            processed['metadata']['category'] = category

        # We need a custom chunker here as in original code
        custom_chunker = chunking_manager
        if strategy != "recursive" or chunk_size != 1000 or chunk_overlap != 200:
             from chunking import ChunkingManager
             custom_chunker = ChunkingManager(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks = custom_chunker.chunk_document(processed['text'], metadata=processed['metadata'])
        
        doc_ids = []
        for chunk_data in chunks:
            doc_id = str(uuid.uuid4())
            clean_metadata = doc_processor.clean_metadata_for_chromadb(chunk_data['metadata'])
            collection.add(
                documents=[chunk_data['text']],
                ids=[doc_id],
                metadatas=[clean_metadata]
            )
            doc_ids.append(doc_id)
            
        hybrid_search.rebuild_index()

        # Track uploaded chunks
        RAG_DOCUMENTS_INDEXED.labels(source="upload").inc(len(chunks))

        logger.info("upload_processed", filename=file.filename, chunks=len(chunks), department=department)
        return UploadResponse(
            status="success",
            message=f"Document processed and added to {department} department.",
            filename=file.filename,
            file_type=processed['metadata'].get('file_type', 'unknown'),
            chunks=len(chunks),
            metadata=doc_processor.clean_metadata_for_chromadb(processed['metadata']),
            chunk_ids=doc_ids,
            department=department,
            category=category,
        )
        
    except DocumentProcessingError:
        raise
    except Exception as e:
        logger.error("upload_failed", error=str(e))
        raise DocumentProcessingError(f"Error processing document: {str(e)}")
