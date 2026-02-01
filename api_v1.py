from fastapi import APIRouter, UploadFile, File, Form, Request, BackgroundTasks
from typing import Optional
import uuid
import structlog
import ollama

from schemas import (
    QueryRequest, QueryResponse, AddKnowledgeRequest, AddKnowledgeResponse,
    RebuildIndexResponse, UploadResponse
)
from services import hybrid_search, collection, doc_processor, chunking_manager
from hybrid_search import format_search_results
from exceptions import (
    DocumentProcessingError, VectorDBError, LLMGenerationError
)
from limiter import limiter

logger = structlog.get_logger()
router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint for V1 API information."""
    return {
        "version": "1.0",
        "description": "RAG API V1"
    }

@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    """
    Query the knowledge base with hybrid search.
    """
    logger.info("query_received", query=body.q, mode=body.mode)
    
    try:
        # Perform hybrid search
        search_results = hybrid_search.search(
            query=body.q,
            mode=body.mode,
            n_results=body.n_results,
            rerank=body.rerank
        )
        
        if not search_results:
            logger.info("no_results_found", query=body.q)
            return QueryResponse(
                answer="No relevant context found in the knowledge base.",
                sources=[],
                search_mode=body.mode,
                reranked=body.rerank,
                total_results=0
            )
        
        # Format results
        formatted_sources = format_search_results(search_results, include_scores=body.include_scores)
        
        # Extract contexts for LLM
        contexts = [result['document'] for result in search_results]
        combined_context = "\n\n---\n\n".join(contexts)
        
        # Generate answer
        try:
            response = ollama.generate(
                model="tinyllama",
                prompt=f"Context: \n{combined_context}\n\nQuestion: {body.q}\nAnswer clearly and concisely:",
            )
            answer = response["response"]
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            raise LLMGenerationError(f"Failed to generate answer: {str(e)}")

        logger.info("query_processed", results_count=len(search_results))
        return QueryResponse(
            answer=answer,
            sources=formatted_sources,
            search_mode=body.mode,
            reranked=body.rerank,
            total_results=len(search_results)
        )
        
    except LLMGenerationError:
        raise
    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise VectorDBError(f"Search failed: {str(e)}")

@router.post("/add", response_model=AddKnowledgeResponse)
@limiter.limit("5/minute")
async def add_knowledge(request: Request, body: AddKnowledgeRequest, background_tasks: BackgroundTasks):
    """Add new text content to the knowledge base."""
    logger.info("add_knowledge_request", chunk_strategy=body.strategy if body.chunk else "none")
    
    try:
        if body.chunk:
            chunks = chunking_manager.chunk_document(body.text, metadata={"source": "direct_text"})
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
            
            logger.info("knowledge_added", chunks=len(chunks))
            return AddKnowledgeResponse(
                status="success",
                message=f"Content chunked and added ({len(chunks)} chunks). Index rebuild scheduled.",
                chunks=len(chunks),
                ids=doc_ids
            )
        else:
            doc_id = str(uuid.uuid4())
            collection.add(documents=[body.text], ids=[doc_id])
            background_tasks.add_task(hybrid_search.rebuild_index)
            
            logger.info("knowledge_added_single")
            return AddKnowledgeResponse(
                status="success",
                message="Content added. Index rebuild scheduled.",
                id=doc_id
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
    chunk_overlap: int = Form(200)
):
    """Upload and process a document."""
    logger.info("upload_request", filename=file.filename)
    
    try:
        file_ext = file.filename.split('.')[-1].lower()
        if f'.{file_ext}' not in doc_processor.SUPPORTED_FORMATS:
            raise DocumentProcessingError(
                f"Unsupported file format: {file_ext}. Supported: {doc_processor.SUPPORTED_FORMATS}"
            )
        
        content = await file.read()
        processed = doc_processor.process_uploaded_file(content, file.filename)
        
        # We need a custom chunker here as in original code
        # But instantiating ChunkingManager is cheap
        custom_chunker = chunking_manager # Use default if params match, otherwise new
        if strategy != "recursive" or chunk_size != 1000 or chunk_overlap != 200:
             # Actually we need to import ChunkingManager class to instantiate a new one
             # I'll rely on the default one for now to save imports or import it inside
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
        
        logger.info("upload_processed", filename=file.filename, chunks=len(chunks))
        return UploadResponse(
            status="success",
            message="Document processed and added.",
            filename=file.filename,
            file_type=processed['metadata'].get('file_type', 'unknown'),
            chunks=len(chunks),
            metadata=doc_processor.clean_metadata_for_chromadb(processed['metadata']),
            chunk_ids=doc_ids
        )
        
    except DocumentProcessingError:
        raise
    except Exception as e:
        logger.error("upload_failed", error=str(e))
        raise DocumentProcessingError(f"Error processing document: {str(e)}")
