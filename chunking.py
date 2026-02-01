"""
Chunking module for splitting documents into smaller pieces with various strategies.
Supports recursive character splitting, semantic chunking, and overlap preservation.
"""

from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
import re


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Original document metadata
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        raise NotImplementedError


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive character text splitter.
    Tries to split by paragraphs, then sentences, then characters.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text using recursive character splitting."""
        if not text or not text.strip():
            return []
        
        chunks = self.splitter.split_text(text)
        
        result = []
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = {
                'chunk_index': idx,
                'chunk_size': len(chunk_text),
                'total_chunks': len(chunks),
                'chunking_strategy': 'recursive'
            }
            
            # Merge with original metadata
            if metadata:
                chunk_metadata.update(metadata)
            
            result.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return result


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunker that splits by sentences.
    Groups sentences together until chunk size is reached.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text using sentence-based semantic chunking."""
        if not text or not text.strip():
            return []
        
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Calculate overlap: keep last few sentences
                overlap_text = ' '.join(current_chunk)
                overlap_sentences = []
                overlap_size = 0
                
                # Add sentences from the end until we reach overlap size
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        result = []
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = {
                'chunk_index': idx,
                'chunk_size': len(chunk_text),
                'total_chunks': len(chunks),
                'chunking_strategy': 'semantic'
            }
            
            # Merge with original metadata
            if metadata:
                chunk_metadata.update(metadata)
            
            result.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return result


class PDFPageChunker(ChunkingStrategy):
    """
    Special chunker for PDFs that preserves page boundaries.
    Chunks within each page using recursive splitting.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF page chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_chunker = RecursiveChunker(chunk_size, chunk_overlap)
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split PDF text while preserving page information."""
        if not metadata or 'page_texts' not in metadata:
            # Fall back to recursive chunking if no page info
            return self.recursive_chunker.chunk(text, metadata)
        
        all_chunks = []
        
        for page_info in metadata['page_texts']:
            page_num = page_info['page_number']
            page_text = page_info['text']
            
            if not page_text.strip():
                continue
            
            # Create metadata for this page
            page_metadata = metadata.copy()
            page_metadata['page_number'] = page_num
            
            # Chunk the page text
            page_chunks = self.recursive_chunker.chunk(page_text, page_metadata)
            
            # Update chunk metadata to include page-specific info
            for chunk in page_chunks:
                chunk['metadata']['page_number'] = page_num
            
            all_chunks.extend(page_chunks)
        
        # Re-index chunks globally
        for idx, chunk in enumerate(all_chunks):
            chunk['metadata']['chunk_index'] = idx
            chunk['metadata']['total_chunks'] = len(all_chunks)
            chunk['metadata']['chunking_strategy'] = 'pdf_page_aware'
        
        return all_chunks


class ChunkingManager:
    """Manager class to handle different chunking strategies."""
    
    STRATEGIES = {
        'recursive': RecursiveChunker,
        'semantic': SemanticChunker,
        'pdf_page_aware': PDFPageChunker
    }
    
    def __init__(self, strategy: str = 'recursive', chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunking manager.
        
        Args:
            strategy: Chunking strategy to use ('recursive', 'semantic', 'pdf_page_aware')
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.STRATEGIES.keys())}")
        
        self.strategy_name = strategy
        self.chunker = self.STRATEGIES[strategy](chunk_size, chunk_overlap)
    
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk a document using the configured strategy.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        return self.chunker.chunk(text, metadata)
