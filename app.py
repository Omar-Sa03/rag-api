from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import chromadb
import ollama
import uuid
from document_processor import DocumentProcessor
from chunking import ChunkingManager

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection(name="docs")

# Initialize document processor and chunking manager
doc_processor = DocumentProcessor()
chunking_manager = ChunkingManager(strategy='recursive', chunk_size=1000, chunk_overlap=200)

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG API",
        "version": "2.0",
        "description": "Retrieval-Augmented Generation API with document processing",
        "endpoints": {
            "GET /": "API information",
            "POST /query": "Query the knowledge base",
            "POST /add": "Add text to knowledge base",
            "POST /upload": "Upload and process documents (PDF, DOCX, Markdown)"
        },
        "supported_formats": list(doc_processor.SUPPORTED_FORMATS)
    }


@app.post("/query")
def query(q: str):
    """Query the knowledge base with a question."""
    results = collection.query(query_texts=[q], n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        return {
            "answer": "No relevant context found in the knowledge base.",
            "sources": []
        }
    
    # Get top results with metadata
    contexts = []
    sources = []
    
    for i, doc in enumerate(results['documents'][0]):
        contexts.append(doc)
        
        # Extract metadata if available
        metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
        sources.append({
            "text_preview": doc[:100] + "..." if len(doc) > 100 else doc,
            "metadata": metadata
        })
    
    # Combine contexts for LLM
    combined_context = "\n\n---\n\n".join(contexts)
    
    answer = ollama.generate(
        model="tinyllama",
        prompt=f"Context: \n{combined_context}\n\nQuestion: {q}\nAnswer clearly and concisely:",
    )

    return {
        "answer": answer["response"],
        "sources": sources
    }

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
            
            return {
                "status": "success",
                "message": f"Content chunked and added to knowledge base ({len(chunks)} chunks)",
                "chunks": len(chunks),
                "ids": doc_ids
            }
        else:
            # Add without chunking (backward compatibility)
            doc_id = str(uuid.uuid4())
            collection.add(documents=[text], ids=[doc_id])
            
            return {
                "status": "success",
                "message": "Content added to knowledge base",
                "id": doc_id
            }
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
        
        return {
            "status": "success",
            "message": f"Document processed and added to knowledge base",
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

