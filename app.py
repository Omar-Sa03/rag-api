from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import chromadb
import ollama
import uuid
from document_processor import DocumentProcessor
from chunking import ChunkingManager
from hybrid_search import HybridSearchEngine, format_search_results

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection(name="docs")

# Initialize document processor and chunking manager
doc_processor = DocumentProcessor()
chunking_manager = ChunkingManager(strategy='recursive', chunk_size=1000, chunk_overlap=200)

# Initialize hybrid search engine
print("Initializing hybrid search engine...")
hybrid_search = HybridSearchEngine(collection, use_reranker=True)
print("Hybrid search engine ready!")

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG API",
        "version": "3.0",
        "description": "Retrieval-Augmented Generation API with hybrid search and re-ranking",
        "endpoints": {
            "GET /": "API information",
            "POST /query": "Query the knowledge base with hybrid search",
            "POST /add": "Add text to knowledge base",
            "POST /upload": "Upload and process documents (PDF, DOCX, Markdown)",
            "POST /rebuild-index": "Rebuild BM25 index"
        },
        "features": {
            "search_modes": ["vector", "bm25", "hybrid"],
            "reranking": "Cross-encoder re-ranking available",
            "fusion": "Reciprocal Rank Fusion (RRF)"
        },
        "supported_formats": list(doc_processor.SUPPORTED_FORMATS)
    }


@app.post("/query")
def query(
    q: str,
    mode: str = "hybrid",
    n_results: int = 5,
    rerank: bool = True,
    include_scores: bool = True
):
    """
    Query the knowledge base with hybrid search.
    
    Args:
        q: Query text
        mode: Search mode - "vector", "bm25", or "hybrid" (default: "hybrid")
        n_results: Number of results to retrieve (default: 5)
        rerank: Whether to apply re-ranking (default: True)
        include_scores: Whether to include scoring information (default: True)
    """
    try:
        # Perform hybrid search
        search_results = hybrid_search.search(
            query=q,
            mode=mode,
            n_results=n_results,
            rerank=rerank
        )
        
        if not search_results:
            return {
                "answer": "No relevant context found in the knowledge base.",
                "sources": [],
                "search_mode": mode,
                "reranked": rerank
            }
        
        # Format results for response
        formatted_sources = format_search_results(search_results, include_scores=include_scores)
        
        # Extract contexts for LLM
        contexts = [result['document'] for result in search_results]
        combined_context = "\n\n---\n\n".join(contexts)
        
        # Generate answer using LLM
        answer = ollama.generate(
            model="tinyllama",
            prompt=f"Context: \n{combined_context}\n\nQuestion: {q}\nAnswer clearly and concisely:",
        )
        
        return {
            "answer": answer["response"],
            "sources": formatted_sources,
            "search_mode": mode,
            "reranked": rerank,
            "total_results": len(search_results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/add")
def add_knowledge(text: str, chunk: bool = True, strategy: str = "recursive"):
    """Add new text content to the knowledge base dynamically."""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        if chunk:
            # Use chunking manager to split text
            chunks = chunking_manager.chunk_document(text, metadata={"source": "direct_text"})
            
            # Add all chunks to ChromaDB
            doc_ids = []
            for chunk_data in chunks:
                doc_id = str(uuid.uuid4())
                
                # Clean metadata for ChromaDB (only simple types allowed)
                clean_metadata = doc_processor.clean_metadata_for_chromadb(chunk_data['metadata'])
                
                collection.add(
                    documents=[chunk_data['text']],
                    ids=[doc_id],
                    metadatas=[clean_metadata]
                )
                doc_ids.append(doc_id)
            
            # Trigger BM25 index rebuild
            hybrid_search.rebuild_index()
            
            return {
                "status": "success",
                "message": f"Content chunked and added to knowledge base ({len(chunks)} chunks). BM25 index rebuilt.",
                "chunks": len(chunks),
                "ids": doc_ids
            }
        else:
            # Add without chunking (backward compatibility)
            doc_id = str(uuid.uuid4())
            collection.add(documents=[text], ids=[doc_id])
            
            # Trigger BM25 index rebuild
            hybrid_search.rebuild_index()
            
            return {
                "status": "success",
                "message": "Content added to knowledge base. BM25 index rebuilt.",
                "id": doc_id
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild-index")
def rebuild_index():
    """Manually rebuild the BM25 index from scratch."""
    try:
        hybrid_search.rebuild_index()
        return {"status": "success", "message": "BM25 index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """
    Upload and process a document (PDF, DOCX, Markdown).
    The document will be processed, chunked, and added to the knowledge base.
    """
    try:
        # Validate file type
        file_ext = file.filename.split('.')[-1].lower()
        if f'.{file_ext}' not in doc_processor.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported: {doc_processor.SUPPORTED_FORMATS}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process document
        processed = doc_processor.process_uploaded_file(file_content, file.filename)
        
        # Create chunking manager with specified parameters
        custom_chunker = ChunkingManager(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Chunk the document
        chunks = custom_chunker.chunk_document(
            processed['text'],
            metadata=processed['metadata']
        )
        
        # Add chunks to ChromaDB
        doc_ids = []
        for chunk_data in chunks:
            doc_id = str(uuid.uuid4())
            
            # Clean metadata for ChromaDB (only simple types allowed)
            clean_metadata = doc_processor.clean_metadata_for_chromadb(chunk_data['metadata'])
            
            collection.add(
                documents=[chunk_data['text']],
                ids=[doc_id],
                metadatas=[clean_metadata]
            )
            doc_ids.append(doc_id)
        
        # Trigger BM25 index rebuild
        hybrid_search.rebuild_index()
        
        return {
            "status": "success",
            "message": f"Document processed and added to knowledge base. BM25 index rebuilt.",
            "filename": file.filename,
            "file_type": processed['metadata']['file_type'],
            "chunks": len(chunks),
            "metadata": processed['metadata'],
            "chunk_ids": doc_ids
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

