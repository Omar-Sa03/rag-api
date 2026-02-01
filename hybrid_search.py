"""
Hybrid search module combining vector search (ChromaDB) with BM25 keyword search.
Implements reciprocal rank fusion (RRF) and cross-encoder re-ranking.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridSearchEngine:
    """
    Hybrid search engine that combines vector search and BM25 keyword search.
    Supports multiple search modes and re-ranking.
    """
    
    def __init__(self, 
                 collection,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_reranker: bool = True):
        """
        Initialize hybrid search engine.
        
        Args:
            collection: ChromaDB collection for vector search
            reranker_model: Cross-encoder model for re-ranking
            use_reranker: Whether to use re-ranking
        """
        self.collection = collection
        self.use_reranker = use_reranker
        
        # Initialize cross-encoder for re-ranking
        if self.use_reranker:
            print(f"Loading re-ranker model: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
        else:
            self.reranker = None
        
        # BM25 index (will be built on demand)
        self.bm25_index = None
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
    
    def build_bm25_index(self):
        """Build BM25 index from all documents in ChromaDB collection."""
        print("Building BM25 index...")
        
        # Get all documents from ChromaDB
        results = self.collection.get()
        
        if not results['documents']:
            print("No documents found in collection")
            return
        
        self.documents = results['documents']
        self.doc_ids = results['ids']
        self.metadatas = results.get('metadatas', [{}] * len(self.documents))
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        print(f"BM25 index built with {len(self.documents)} documents")
    
    def vector_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Perform vector similarity search using ChromaDB.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of search results with scores
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        search_results = []
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'id': results['ids'][0][i],
                    'document': doc,
                    'score': 1.0 - results['distances'][0][i] if results.get('distances') else 1.0,
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'rank': i + 1
                }
                search_results.append(result)
        
        return search_results
    
    def bm25_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if self.bm25_index is None:
            self.build_bm25_index()
        
        if not self.documents:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top N results
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        search_results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:  # Only include results with positive scores
                result = {
                    'id': self.doc_ids[idx],
                    'document': self.documents[idx],
                    'score': float(scores[idx]),
                    'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {},
                    'rank': rank
                }
                search_results.append(result)
        
        return search_results
    
    def reciprocal_rank_fusion(self,
                               vector_results: List[Dict],
                               bm25_results: List[Dict],
                               k: int = 60) -> List[Dict]:
        """
        Combine results from vector and BM25 search using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = sum(1 / (k + rank(d)))
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Constant for RRF (default: 60)
            
        Returns:
            Fused and ranked results
        """
        # Create a dictionary to store RRF scores
        rrf_scores = {}
        doc_data = {}
        
        # Add vector search results
        for result in vector_results:
            doc_id = result['id']
            rank = result['rank']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (k + rank))
            doc_data[doc_id] = result
        
        # Add BM25 search results
        for result in bm25_results:
            doc_id = result['id']
            rank = result['rank']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (k + rank))
            
            # Store document data if not already present
            if doc_id not in doc_data:
                doc_data[doc_id] = result
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            result = doc_data[doc_id].copy()
            result['rrf_score'] = score
            result['rank'] = rank
            fused_results.append(result)
        
        return fused_results
    
    def rerank(self, query: str, results: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Re-rank results using a cross-encoder model.
        
        Args:
            query: Search query
            results: Search results to re-rank
            top_k: Number of top results to return (None = all)
            
        Returns:
            Re-ranked results
        """
        if not self.use_reranker or self.reranker is None or not results:
            return results
        
        # Prepare query-document pairs
        pairs = [[query, result['document']] for result in results]
        
        # Get cross-encoder scores
        scores = self.reranker.predict(pairs)
        
        # Add scores to results and sort
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        # Update ranks
        for rank, result in enumerate(reranked, 1):
            result['rank'] = rank
        
        # Return top K if specified
        if top_k:
            return reranked[:top_k]
        
        return reranked
    
    def search(self,
               query: str,
               mode: str = "hybrid",
               n_results: int = 10,
               rerank: bool = True) -> List[Dict]:
        """
        Perform search with specified mode.
        
        Args:
            query: Search query
            mode: Search mode - "vector", "bm25", or "hybrid"
            n_results: Number of results to return
            rerank: Whether to apply re-ranking
            
        Returns:
            Search results
        """
        if mode == "vector":
            # Vector-only search
            results = self.vector_search(query, n_results * 2)
            
        elif mode == "bm25":
            # BM25-only search
            results = self.bm25_search(query, n_results * 2)
            
        elif mode == "hybrid":
            # Hybrid search with RRF
            vector_results = self.vector_search(query, n_results * 2)
            bm25_results = self.bm25_search(query, n_results * 2)
            results = self.reciprocal_rank_fusion(vector_results, bm25_results)
            
        else:
            raise ValueError(f"Unknown search mode: {mode}. Use 'vector', 'bm25', or 'hybrid'")
        
        # Apply re-ranking if requested
        if rerank and self.use_reranker:
            results = self.rerank(query, results, top_k=n_results)
        else:
            results = results[:n_results]
        
        return results
    
    def rebuild_index(self):
        """Rebuild the BM25 index (call after adding new documents)."""
        self.build_bm25_index()


def format_search_results(results: List[Dict], include_scores: bool = True) -> List[Dict]:
    """
    Format search results for API response.
    
    Args:
        results: Raw search results
        include_scores: Whether to include score information
        
    Returns:
        Formatted results
    """
    formatted = []
    
    for result in results:
        formatted_result = {
            'text': result['document'],
            'text_preview': result['document'][:150] + "..." if len(result['document']) > 150 else result['document'],
            'metadata': result.get('metadata', {}),
            'rank': result.get('rank', 0)
        }
        
        if include_scores:
            scores = {}
            if 'score' in result:
                scores['similarity_score'] = result['score']
            if 'rrf_score' in result:
                scores['rrf_score'] = result['rrf_score']
            if 'rerank_score' in result:
                scores['rerank_score'] = result['rerank_score']
            
            if scores:
                formatted_result['scores'] = scores
        
        formatted.append(formatted_result)
    
    return formatted
