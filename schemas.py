from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union


# --- Enums ---

class DepartmentEnum(str, Enum):
    HR = "hr"
    ENGINEERING = "engineering"
    FINANCE = "finance"
    LEGAL = "legal"
    OPERATIONS = "operations"
    MARKETING = "marketing"
    ALL = "all"


class CategoryEnum(str, Enum):
    POLICY = "policy"
    ONBOARDING = "onboarding"
    BENEFITS = "benefits"
    ORG_CHART = "org-chart"
    HANDBOOK = "handbook"
    PROCEDURE = "procedure"
    FAQ = "faq"
    ANNOUNCEMENT = "announcement"


# --- Request Models ---

class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, description="The query text to search for")
    mode: str = Field("hybrid", description="Search mode: 'vector', 'bm25', or 'hybrid'")
    n_results: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    rerank: bool = Field(True, description="Whether to apply re-ranking")
    include_scores: bool = Field(True, description="Whether to include scoring information")
    department: Optional[DepartmentEnum] = Field(
        None, description="Filter results by department (e.g. 'hr', 'engineering')"
    )
    category: Optional[CategoryEnum] = Field(
        None, description="Filter results by document category (e.g. 'policy', 'benefits')"
    )

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
    department: DepartmentEnum = Field(
        ..., description="Department this content belongs to (e.g. 'hr', 'engineering')"
    )
    category: Optional[CategoryEnum] = Field(
        None, description="Document category (e.g. 'policy', 'onboarding', 'benefits')"
    )
    effective_date: Optional[str] = Field(
        None, description="Effective date of the policy/document (ISO format, e.g. '2026-01-15')"
    )
    author: Optional[str] = Field(
        None, description="Author or owner of the document"
    )

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
    department_filter: Optional[str] = None
    category_filter: Optional[str] = None
    disclaimer: str = "This information is provided for reference only. For official decisions, please contact your HR representative or department lead."

class AddKnowledgeResponse(BaseModel):
    status: str
    message: str
    chunks: Optional[int] = None
    ids: Optional[List[str]] = None
    id: Optional[str] = None # For non-chunked
    department: Optional[str] = None
    category: Optional[str] = None

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
    department: Optional[str] = None
    category: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]
