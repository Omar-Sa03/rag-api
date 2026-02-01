from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union

# --- Request Models ---

class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, description="The query text to search for")
    mode: str = Field("hybrid", description="Search mode: 'vector', 'bm25', or 'hybrid'")
    n_results: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    rerank: bool = Field(True, description="Whether to apply re-ranking")
    include_scores: bool = Field(True, description="Whether to include scoring information")

    @validator('mode')
    def validate_mode(cls, v):
        allowed = ['vector', 'bm25', 'hybrid']
        if v not in allowed:
            raise ValueError(f"Mode must be one of {allowed}")
        return v

class AddKnowledgeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content to add")
    chunk: bool = Field(True, description="Whether to chunk the text")
    strategy: str = Field("recursive", description="Chunking strategy")

# --- Response Models ---

class SearchResultSource(BaseModel):
    # Depending on format_search_results return structure, usually has content/metadata
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] # Using Dict for flexibility, or could use SearchResultSource
    search_mode: str
    reranked: bool
    total_results: int

class AddKnowledgeResponse(BaseModel):
    status: str
    message: str
    chunks: Optional[int] = None
    ids: Optional[List[str]] = None
    id: Optional[str] = None # For non-chunked

class RebuildIndexResponse(BaseModel):
    status: str
    message: str

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    file_type: str
    chunks: int
    metadata: Dict[str, Any]
    chunk_ids: List[str]

class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]
