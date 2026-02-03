from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum


class FileType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "md"
    TEXT = "txt"
    JSON = "json"


class ChunkMetadata(BaseModel):
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    domain: Optional[str] = None
    content_type: Optional[str] = None
    tech_specs: Optional[Dict[str, Any]] = None

    source_file: str
    file_type: str
    chunk_index: int
    total_chunks: int
    char_count: int
    page_number: Optional[int] = None


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    dense_embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None


class UploadResponse(BaseModel):
    file_name: str
    file_type: str
    total_chunks: int
    processing_time_seconds: float
    chunk_ids: List[str]


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    domain_filter: Optional[str] = None
    content_type_filter: Optional[str] = None
    entity_filter: Optional[Dict[str, List[str]]] = None
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata


class QueryResponse(BaseModel):
    query: str
    answer: str
    results: List[SearchResult]
    retrieval_time_seconds: float
    llm_time_seconds: float
