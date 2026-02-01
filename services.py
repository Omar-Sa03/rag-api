import chromadb
from document_processor import DocumentProcessor
from chunking import ChunkingManager
from hybrid_search import HybridSearchEngine
import os

# Initialize components globally
print("Initializing services...")

# Ensure db directory exists
os.makedirs("./db", exist_ok=True)

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection(name="docs")

doc_processor = DocumentProcessor()
# Default chunking manager - endpoints might create their own if parameters differ
chunking_manager = ChunkingManager(strategy='recursive', chunk_size=1000, chunk_overlap=200)

hybrid_search = HybridSearchEngine(collection, use_reranker=True)
print("Services initialized.")
