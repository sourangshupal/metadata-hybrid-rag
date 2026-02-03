# RAG System with Metadata Enrichment - Implementation Guide

**Version:** 1.0  
**Python Version:** 3.12+  
**Architecture:** FastAPI + Docling + GLiNER2 + Qdrant + OpenAI

---

## 🎯 Project Overview

A production-ready RAG system focused on **metadata enrichment** for advanced retrieval. This implementation uses:

- **Document Processing**: Docling's HybridChunker for intelligent parsing
- **Metadata Extraction**: GLiNER2 for zero-shot entity extraction, classification, and structured data
- **Vector Storage**: Qdrant with hybrid search (BM25 + dense embeddings)
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o-mini for responses
- **No LangChain**: Direct implementations for clarity and control

---

## 📁 Project Structure

```
rag-metadata-enrichment/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── models.py               # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py   # Docling integration
│   │   ├── metadata_extractor.py   # GLiNER2 integration
│   │   ├── vector_store.py         # Qdrant operations
│   │   └── retrieval.py            # Hybrid search
│   └── api/
│       ├── __init__.py
│       ├── upload.py           # Upload endpoints
│       └── query.py            # Query endpoints
├── requirements.txt
├── .env.example
├── docker-compose.yml          # Qdrant setup
└── README.md
```

---

## 🔧 Dependencies (requirements.txt)

```txt
# Core
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pydantic==2.9.2
pydantic-settings==2.6.0

# Document Processing
docling==2.7.0
docling-core==2.4.1

# Metadata Extraction
gliner2==1.1.2

# Vector Store
qdrant-client==1.12.1
fastembed==0.4.2

# OpenAI
openai==1.54.4

# Utilities
python-dotenv==1.0.1
loguru==0.7.2
httpx==0.27.2
```

---

## ⚙️ Configuration (.env.example)

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_DIMENSION=1536

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents_metadata_enriched
QDRANT_API_KEY=  # Optional for cloud

# GLiNER Configuration
GLINER_MODEL=fastino/gliner2-large-v1

# Application Configuration
MAX_FILE_SIZE_MB=10
CHUNK_SIZE=512
CHUNK_OVERLAP=50
LOG_LEVEL=INFO
```

---

## 🐳 Docker Compose (docker-compose.yml)

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.12.1
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

volumes:
  qdrant_storage:
```

**Start Qdrant:**
```bash
docker-compose up -d
```

---

## 📝 Implementation

### 1. Configuration (app/config.py)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_dimension: int = 1536
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents_metadata_enriched"
    qdrant_api_key: str | None = None
    
    # GLiNER
    gliner_model: str = "fastino/gliner2-large-v1"
    
    # Application
    max_file_size_mb: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 50
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


settings = Settings()
```

### 2. Pydantic Models (app/models.py)

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum


class FileType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "md"
    TEXT = "txt"
    JSON = "json"


class ChunkMetadata(BaseModel):
    """Metadata extracted for each chunk"""
    
    # GLiNER2 extracted metadata
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    domain: Optional[str] = None
    content_type: Optional[str] = None
    tech_specs: Optional[Dict[str, Any]] = None
    
    # Document metadata
    source_file: str
    file_type: str
    chunk_index: int
    total_chunks: int
    char_count: int
    
    # Optional: page number for PDFs
    page_number: Optional[int] = None


class DocumentChunk(BaseModel):
    """Processed document chunk ready for storage"""
    
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    dense_embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None


class UploadResponse(BaseModel):
    """Response after document upload"""
    
    file_name: str
    file_type: str
    total_chunks: int
    processing_time_seconds: float
    chunk_ids: List[str]


class QueryRequest(BaseModel):
    """Query request with optional filters"""
    
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    domain_filter: Optional[str] = None
    content_type_filter: Optional[str] = None
    entity_filter: Optional[Dict[str, List[str]]] = None
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # 0=sparse, 1=dense


class SearchResult(BaseModel):
    """Single search result"""
    
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata


class QueryResponse(BaseModel):
    """Response with retrieval results and LLM answer"""
    
    query: str
    answer: str
    results: List[SearchResult]
    retrieval_time_seconds: float
    llm_time_seconds: float
```

### 3. Document Processor (app/services/document_processor.py)

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import tempfile


class DocumentProcessor:
    """Process documents using Docling's HybridChunker"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.converter = DocumentConverter()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def process_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Process a file and return chunks with document structure
        
        Args:
            file_path: Path to the file
            file_type: Type of file (pdf, md, txt, json)
        
        Returns:
            Dictionary with chunks and metadata
        """
        logger.info(f"Processing file: {file_path} (type: {file_type})")
        
        # Convert document
        result = self.converter.convert(file_path)
        
        # Extract full text
        full_text = result.document.export_to_markdown()
        
        # Use HybridChunker for intelligent chunking
        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=self.chunk_size,
            merge_peers=True  # Merge semantically similar chunks
        )
        
        # Get chunks
        chunks_iter = chunker.chunk(result.document)
        chunks = list(chunks_iter)
        
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        
        # Extract page information if available (for PDFs)
        page_mapping = {}
        if hasattr(result.document, 'pages'):
            for page_num, page in enumerate(result.document.pages, start=1):
                page_text = page.export_to_markdown()
                page_mapping[page_num] = page_text
        
        # Convert chunks to our format
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.text
            
            # Try to determine page number (for PDFs)
            page_num = None
            if page_mapping:
                for page_no, page_text in page_mapping.items():
                    if chunk_text in page_text:
                        page_num = page_no
                        break
            
            processed_chunks.append({
                "chunk_index": idx,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "page_number": page_num,
                "total_chunks": len(chunks)
            })
        
        return {
            "file_name": Path(file_path).name,
            "file_type": file_type,
            "total_chunks": len(chunks),
            "full_text": full_text,
            "chunks": processed_chunks,
            "page_mapping": page_mapping
        }
    
    def process_uploaded_file(self, file_bytes: bytes, filename: str, file_type: str) -> Dict[str, Any]:
        """
        Process an uploaded file from bytes
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            file_type: Type of file
        
        Returns:
            Processed chunks and metadata
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        try:
            result = self.process_file(tmp_path, file_type)
            result["file_name"] = filename  # Override with original name
            return result
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
```

### 4. Metadata Extractor (app/services/metadata_extractor.py)

```python
from gliner2 import GLiNER2
from typing import Dict, Any, List
from loguru import logger


class MetadataExtractor:
    """Extract rich metadata using GLiNER2"""
    
    def __init__(self, model_name: str = "fastino/gliner2-large-v1"):
        logger.info(f"Loading GLiNER2 model: {model_name}")
        self.model = GLiNER2.from_pretrained(model_name)
        logger.info("GLiNER2 model loaded successfully")
        
        # Define metadata extraction schema
        self.schema = self._create_schema()
    
    def _create_schema(self):
        """Create comprehensive metadata extraction schema"""
        
        return (self.model.create_schema()
            # Extract key entities
            .entities({
                "technology": "Technologies, tools, frameworks, databases, programming languages, APIs",
                "company": "Companies, vendors, organizations, institutions",
                "product": "Products, services, platforms, software, applications",
                "concept": "Technical concepts, algorithms, methodologies, design patterns",
                "metric": "Performance metrics, benchmarks, KPIs, measurements, statistics",
                "person": "People, authors, developers, researchers, experts",
                "location": "Cities, countries, regions, addresses",
                "date": "Dates, time periods, versions, releases"
            })
            
            # Classify by domain
            .classification("domain", [
                "database", "cloud_computing", "machine_learning", "backend_development",
                "frontend_development", "devops", "security", "networking",
                "data_science", "mobile_development", "general"
            ])
            
            # Classify by content type
            .classification("content_type", [
                "tutorial", "api_documentation", "architecture_guide",
                "troubleshooting", "best_practices", "code_example",
                "research_paper", "blog_post", "technical_specification",
                "case_study", "general_information"
            ])
            
            # Extract structured technical specifications
            .structure("tech_specs")
                .field("mentioned_products", dtype="list", 
                       description="List of products or tools mentioned")
                .field("versions", dtype="list",
                       description="Version numbers or release information")
                .field("requirements", dtype="list",
                       description="System requirements, prerequisites, dependencies")
                .field("capabilities", dtype="list",
                       description="Features, capabilities, or functionalities described")
        )
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text chunk using GLiNER2
        
        Args:
            text: Text chunk to analyze
        
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            # Extract using the predefined schema
            result = self.model.extract(text, self.schema)
            
            # Clean up None values and empty lists
            cleaned_result = {}
            for key, value in result.items():
                if value is not None:
                    if isinstance(value, dict):
                        # Clean nested dicts (like entities)
                        cleaned_value = {k: v for k, v in value.items() if v}
                        if cleaned_value:
                            cleaned_result[key] = cleaned_value
                    elif isinstance(value, list):
                        # Keep non-empty lists
                        if value:
                            cleaned_result[key] = value
                    else:
                        cleaned_result[key] = value
            
            logger.debug(f"Extracted metadata: {cleaned_result}")
            return cleaned_result
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "entities": {},
                "domain": "general",
                "content_type": "general_information",
                "tech_specs": []
            }
    
    def batch_extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract metadata from multiple text chunks
        
        Args:
            texts: List of text chunks
        
        Returns:
            List of metadata dictionaries
        """
        logger.info(f"Batch extracting metadata for {len(texts)} chunks")
        results = []
        
        for text in texts:
            metadata = self.extract_metadata(text)
            results.append(metadata)
        
        return results
```

### 5. Vector Store (app/services/vector_store.py)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
    SparseVectorParams, SparseIndexParams
)
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI
from fastembed import SparseTextEmbedding
import uuid


class VectorStore:
    """Qdrant vector store with hybrid search support"""
    
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        openai_api_key: str,
        embedding_model: str,
        embedding_dimension: int,
        api_key: Optional[str] = None
    ):
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        if api_key:
            self.client = QdrantClient(url=f"https://{host}", api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port)
        
        # Initialize OpenAI for dense embeddings
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        
        # Initialize FastEmbed for sparse embeddings (BM25)
        logger.info("Loading FastEmbed sparse model for BM25...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        logger.info(f"VectorStore initialized for collection: {collection_name}")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection with hybrid search configuration"""
        
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if not collection_exists:
            logger.info(f"Creating new collection: {self.collection_name}")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )
            logger.info("Collection created successfully with hybrid search support")
        else:
            logger.info(f"Collection {self.collection_name} already exists")
    
    def _get_dense_embedding(self, text: str) -> List[float]:
        """Generate dense embedding using OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def _get_sparse_embedding(self, text: str) -> Dict[int, float]:
        """Generate sparse embedding using FastEmbed BM25"""
        sparse_vectors = list(self.sparse_model.embed([text]))
        
        if sparse_vectors and len(sparse_vectors) > 0:
            sparse_vector = sparse_vectors[0]
            # Convert to dict format expected by Qdrant
            return {int(idx): float(val) for idx, val in zip(sparse_vector.indices, sparse_vector.values)}
        
        return {}
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add document chunks to vector store with hybrid embeddings
        
        Args:
            chunks: List of processed chunks with metadata
        
        Returns:
            List of inserted chunk IDs
        """
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        points = []
        chunk_ids = []
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            text = chunk["text"]
            metadata = chunk["metadata"]
            
            # Generate embeddings
            dense_embedding = self._get_dense_embedding(text)
            sparse_embedding = self._get_sparse_embedding(text)
            
            # Prepare payload with metadata
            payload = {
                "text": text,
                "metadata": metadata.dict() if hasattr(metadata, 'dict') else metadata
            }
            
            # Create point with both dense and sparse vectors
            point = PointStruct(
                id=chunk_id,
                vector={
                    "dense": dense_embedding,
                    "sparse": sparse_embedding
                },
                payload=payload
            )
            points.append(point)
        
        # Batch upload
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Successfully added {len(points)} chunks")
        return chunk_ids
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        domain_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        entity_filter: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (BM25 + dense) with metadata filtering
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Balance between sparse (0.0) and dense (1.0). 0.5 = equal weight
            domain_filter: Filter by domain classification
            content_type_filter: Filter by content type
            entity_filter: Filter by extracted entities
        
        Returns:
            List of search results with scores and metadata
        """
        logger.info(f"Hybrid search: '{query}' (top_k={top_k}, alpha={alpha})")
        
        # Generate query embeddings
        dense_query = self._get_dense_embedding(query)
        sparse_query = self._get_sparse_embedding(query)
        
        # Build metadata filters
        must_conditions = []
        
        if domain_filter:
            must_conditions.append(
                FieldCondition(
                    key="metadata.domain",
                    match=MatchValue(value=domain_filter)
                )
            )
        
        if content_type_filter:
            must_conditions.append(
                FieldCondition(
                    key="metadata.content_type",
                    match=MatchValue(value=content_type_filter)
                )
            )
        
        if entity_filter:
            for entity_type, entity_values in entity_filter.items():
                must_conditions.append(
                    FieldCondition(
                        key=f"metadata.entities.{entity_type}",
                        match=MatchAny(any=entity_values)
                    )
                )
        
        query_filter = Filter(must=must_conditions) if must_conditions else None
        
        # Perform hybrid search
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Sparse search
                {
                    "query": sparse_query,
                    "using": "sparse",
                    "limit": top_k * 2,
                    "filter": query_filter
                },
                # Dense search
                {
                    "query": dense_query,
                    "using": "dense",
                    "limit": top_k * 2,
                    "filter": query_filter
                }
            ],
            query={
                # Fusion: RRF (Reciprocal Rank Fusion)
                "fusion": "rrf"
            },
            limit=top_k
        )
        
        results = []
        for point in search_results.points:
            results.append({
                "chunk_id": point.id,
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {})
            })
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
```

### 6. Retrieval Service (app/services/retrieval.py)

```python
from typing import List, Dict, Any, Optional
from openai import OpenAI
from loguru import logger


class RetrievalService:
    """Handle query processing and answer generation"""
    
    def __init__(self, vector_store, openai_api_key: str, llm_model: str):
        self.vector_store = vector_store
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        domain_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        entity_filter: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search
        """
        return self.vector_store.hybrid_search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            domain_filter=domain_filter,
            content_type_filter=content_type_filter,
            entity_filter=entity_filter
        )
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer using retrieved context and OpenAI LLM
        
        Args:
            query: User query
            context_chunks: Retrieved chunks with metadata
        
        Returns:
            Generated answer
        """
        # Build context from retrieved chunks
        context_parts = []
        for idx, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source_file", "Unknown")
            domain = metadata.get("domain", "")
            
            context_parts.append(
                f"[Source {idx}: {source} | Domain: {domain}]\n{chunk['text']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information, say so honestly.
Always cite which source(s) you used in your answer."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        logger.info("Generating answer with LLM")
        
        # Call OpenAI
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        logger.info("Answer generated successfully")
        
        return answer
```

### 7. FastAPI Application (app/main.py)

```python
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import sys
import time

from app.config import settings
from app.models import (
    UploadResponse, QueryRequest, QueryResponse, 
    SearchResult, FileType
)
from app.services.document_processor import DocumentProcessor
from app.services.metadata_extractor import MetadataExtractor
from app.services.vector_store import VectorStore
from app.services.retrieval import RetrievalService

# Configure logging
logger.remove()
logger.add(sys.stderr, level=settings.log_level)

# Global service instances
document_processor: DocumentProcessor = None
metadata_extractor: MetadataExtractor = None
vector_store: VectorStore = None
retrieval_service: RetrievalService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global document_processor, metadata_extractor, vector_store, retrieval_service
    
    logger.info("Initializing services...")
    
    # Initialize document processor
    document_processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # Initialize metadata extractor
    metadata_extractor = MetadataExtractor(
        model_name=settings.gliner_model
    )
    
    # Initialize vector store
    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        openai_api_key=settings.openai_api_key,
        embedding_model=settings.openai_embedding_model,
        embedding_dimension=settings.openai_embedding_dimension,
        api_key=settings.qdrant_api_key
    )
    
    # Initialize retrieval service
    retrieval_service = RetrievalService(
        vector_store=vector_store,
        openai_api_key=settings.openai_api_key,
        llm_model=settings.openai_llm_model
    )
    
    logger.info("All services initialized successfully")
    
    yield
    
    logger.info("Shutting down services...")


app = FastAPI(
    title="RAG with Metadata Enrichment",
    description="Document upload and retrieval with GLiNER2 metadata extraction",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Metadata Enrichment API",
        "version": "1.0.0"
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, MD, TXT, JSON)
    
    - Extracts text using Docling
    - Chunks using HybridChunker
    - Extracts metadata using GLiNER2
    - Stores in Qdrant with hybrid embeddings
    """
    start_time = time.time()
    
    # Validate file type
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in [ft.value for ft in FileType]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Supported: pdf, md, txt, json"
        )
    
    # Validate file size
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size_mb:.2f}MB. Max: {settings.max_file_size_mb}MB"
        )
    
    logger.info(f"Processing upload: {file.filename} ({file_size_mb:.2f}MB)")
    
    try:
        # Step 1: Process document and chunk
        processed = document_processor.process_uploaded_file(
            file_bytes=file_bytes,
            filename=file.filename,
            file_type=file_extension
        )
        
        # Step 2: Extract metadata for each chunk
        chunks_with_metadata = []
        for chunk_data in processed["chunks"]:
            # Extract metadata using GLiNER2
            metadata_dict = metadata_extractor.extract_metadata(chunk_data["text"])
            
            # Combine with chunk metadata
            from app.models import ChunkMetadata
            chunk_metadata = ChunkMetadata(
                entities=metadata_dict.get("entities", {}),
                domain=metadata_dict.get("domain"),
                content_type=metadata_dict.get("content_type"),
                tech_specs=metadata_dict.get("tech_specs"),
                source_file=processed["file_name"],
                file_type=file_extension,
                chunk_index=chunk_data["chunk_index"],
                total_chunks=chunk_data["total_chunks"],
                char_count=chunk_data["char_count"],
                page_number=chunk_data.get("page_number")
            )
            
            chunks_with_metadata.append({
                "text": chunk_data["text"],
                "metadata": chunk_metadata
            })
        
        # Step 3: Add to vector store
        chunk_ids = vector_store.add_chunks(chunks_with_metadata)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully processed {file.filename} in {processing_time:.2f}s")
        
        return UploadResponse(
            file_name=file.filename,
            file_type=file_extension,
            total_chunks=len(chunks_with_metadata),
            processing_time_seconds=round(processing_time, 2),
            chunk_ids=chunk_ids
        )
        
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents with hybrid search and metadata filtering
    
    - Performs hybrid search (BM25 + dense embeddings)
    - Applies metadata filters (domain, content_type, entities)
    - Generates answer using OpenAI LLM
    """
    logger.info(f"Query: '{request.query}' with filters: domain={request.domain_filter}, content_type={request.content_type_filter}")
    
    try:
        # Step 1: Retrieve relevant chunks
        retrieval_start = time.time()
        
        results = retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            alpha=request.hybrid_alpha,
            domain_filter=request.domain_filter,
            content_type_filter=request.content_type_filter,
            entity_filter=request.entity_filter
        )
        
        retrieval_time = time.time() - retrieval_start
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Try adjusting your filters."
            )
        
        # Step 2: Generate answer
        llm_start = time.time()
        
        answer = retrieval_service.generate_answer(
            query=request.query,
            context_chunks=results
        )
        
        llm_time = time.time() - llm_start
        
        # Format results
        from app.models import SearchResult, ChunkMetadata
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result["chunk_id"],
                text=result["text"],
                score=result["score"],
                metadata=ChunkMetadata(**result["metadata"])
            ))
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            results=search_results,
            retrieval_time_seconds=round(retrieval_time, 2),
            llm_time_seconds=round(llm_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "document_processor": document_processor is not None,
            "metadata_extractor": metadata_extractor is not None,
            "vector_store": vector_store is not None,
            "retrieval_service": retrieval_service is not None
        },
        "config": {
            "gliner_model": settings.gliner_model,
            "qdrant_collection": settings.qdrant_collection_name,
            "embedding_model": settings.openai_embedding_model,
            "llm_model": settings.openai_llm_model
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🚀 Setup and Running

### 1. Clone and Install

```bash
# Create project directory
mkdir rag-metadata-enrichment
cd rag-metadata-enrichment

# Create virtual environment (Python 3.12+)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy and edit .env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Start Qdrant

```bash
docker-compose up -d

# Verify Qdrant is running
curl http://localhost:6333/collections
```

### 4. Run Application

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Application will be available at: **http://localhost:8000**

API docs: **http://localhost:8000/docs**

---

## 📚 API Usage Examples

### 1. Upload Document

```bash
# Upload PDF
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@oracle_vector_search.pdf"

# Response:
{
  "file_name": "oracle_vector_search.pdf",
  "file_type": "pdf",
  "total_chunks": 42,
  "processing_time_seconds": 8.35,
  "chunk_ids": ["uuid-1", "uuid-2", ...]
}
```

### 2. Query with Basic Search

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to use vector search in Oracle?",
    "top_k": 5,
    "hybrid_alpha": 0.5
  }'

# Response:
{
  "query": "How to use vector search in Oracle?",
  "answer": "Oracle Database 26ai provides vector search through...",
  "results": [
    {
      "chunk_id": "uuid-1",
      "text": "Oracle vector search allows...",
      "score": 0.89,
      "metadata": {
        "entities": {
          "technology": ["Oracle Database 26ai", "vector search"],
          "product": ["Oracle Database"]
        },
        "domain": "database",
        "content_type": "tutorial"
      }
    }
  ],
  "retrieval_time_seconds": 0.45,
  "llm_time_seconds": 1.23
}
```

### 3. Query with Metadata Filters

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Best practices for RAG systems",
    "top_k": 5,
    "domain_filter": "machine_learning",
    "content_type_filter": "best_practices",
    "entity_filter": {
      "technology": ["RAG", "vector search"]
    },
    "hybrid_alpha": 0.7
  }'
```

### 4. Python Client Example

```python
import requests

# Upload document
with open("technical_doc.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )
    print(response.json())

# Query with filters
query_request = {
    "query": "How to implement hybrid search?",
    "top_k": 5,
    "domain_filter": "database",
    "content_type_filter": "tutorial",
    "hybrid_alpha": 0.6
}

response = requests.post(
    "http://localhost:8000/query",
    json=query_request
)
result = response.json()

print(f"Answer: {result['answer']}")
print(f"\nSources:")
for idx, res in enumerate(result['results'], 1):
    print(f"{idx}. {res['metadata']['source_file']} (score: {res['score']:.3f})")
```

---

## 🎯 Key Features Implemented

### ✅ Feature 1: Multi-Format Upload
- Supports PDF, Markdown, TXT, JSON
- File size validation
- Async processing

### ✅ Feature 2: Docling HybridChunker
- Intelligent semantic chunking
- Preserves document structure
- Page number tracking for PDFs

### ✅ Feature 3: GLiNER2 Metadata Extraction
- Entity extraction (8 types: technology, company, product, concept, metric, person, location, date)
- Domain classification (11 categories)
- Content type classification (11 types)
- Structured technical specifications
- Zero-shot, no training required

### ✅ Feature 4: Qdrant Vector Store
- Hybrid search support (dense + sparse)
- OpenAI embeddings for dense vectors
- FastEmbed BM25 for sparse vectors
- Metadata-based filtering

### ✅ Feature 5: Hybrid Search Retrieval
- Reciprocal Rank Fusion (RRF)
- Configurable alpha (sparse/dense balance)
- Multi-level filtering:
  - Domain filter
  - Content type filter
  - Entity filter (multiple entity types)
- LLM answer generation with citations

---

## 🔍 Advanced Usage

### Custom Metadata Schema

To customize metadata extraction, modify `app/services/metadata_extractor.py`:

```python
def _create_schema(self):
    return (self.model.create_schema()
        # Add your custom entities
        .entities({
            "custom_entity": "Description of what to extract",
            # ... more entities
        })
        
        # Add your custom classifications
        .classification("custom_category", [
            "option1", "option2", "option3"
        ])
        
        # Add custom structured fields
        .structure("custom_specs")
            .field("field1", dtype="str")
            .field("field2", dtype="list")
    )
```

### Adjusting Hybrid Search Balance

```python
# More emphasis on semantic (dense) search
query_request = {
    "query": "Your query",
    "hybrid_alpha": 0.8  # 80% dense, 20% sparse
}

# More emphasis on keyword (sparse/BM25) search
query_request = {
    "query": "Your query",
    "hybrid_alpha": 0.2  # 20% dense, 80% sparse
}

# Balanced hybrid search
query_request = {
    "query": "Your query",
    "hybrid_alpha": 0.5  # 50/50 balance
}
```

---

## 🧪 Testing

### Test Document Upload

```python
# test_upload.py
import requests

files = [
    "sample.pdf",
    "readme.md",
    "notes.txt",
    "config.json"
]

for file_path in files:
    with open(file_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/upload",
            files={"file": f}
        )
        print(f"{file_path}: {response.status_code}")
        print(response.json())
```

### Test Metadata Filtering

```python
# test_query.py
import requests

# Test different filter combinations
test_cases = [
    {
        "name": "No filters",
        "query": "vector search",
        "filters": {}
    },
    {
        "name": "Domain filter only",
        "query": "vector search",
        "filters": {"domain_filter": "database"}
    },
    {
        "name": "Multiple filters",
        "query": "best practices",
        "filters": {
            "domain_filter": "machine_learning",
            "content_type_filter": "best_practices"
        }
    },
    {
        "name": "Entity filter",
        "query": "Oracle features",
        "filters": {
            "entity_filter": {
                "technology": ["Oracle", "vector search"]
            }
        }
    }
]

for test in test_cases:
    print(f"\n=== {test['name']} ===")
    response = requests.post(
        "http://localhost:8000/query",
        json={
            "query": test["query"],
            "top_k": 3,
            **test["filters"]
        }
    )
    result = response.json()
    print(f"Found {len(result['results'])} results")
    print(f"Answer: {result['answer'][:100]}...")
```

---

## 📊 Performance Optimization

### 1. Batch Processing
For bulk uploads, process multiple files in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def upload_file(file_path):
    with open(file_path, "rb") as f:
        return requests.post(
            "http://localhost:8000/upload",
            files={"file": f}
        )

files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(upload_file, files)
```

### 2. Caching Embeddings
Consider caching frequently accessed embeddings:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    return self._get_dense_embedding(text)
```

### 3. Increase Qdrant Performance
```yaml
# docker-compose.yml
services:
  qdrant:
    environment:
      - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4
      - QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=2
```

---

## 🐛 Troubleshooting

### Issue: "Collection already exists" error
```bash
# Delete and recreate collection
curl -X DELETE "http://localhost:6333/collections/documents_metadata_enriched"
```

### Issue: Out of memory with large PDFs
Adjust chunk size in `.env`:
```env
CHUNK_SIZE=256  # Smaller chunks
MAX_FILE_SIZE_MB=5  # Limit file size
```

### Issue: Slow metadata extraction
Use base model instead of large:
```env
GLINER_MODEL=fastino/gliner2-base-v1
```

### Issue: OpenAI rate limits
Add retry logic or use smaller batches.

---

## 📈 Next Steps

### Enhancements to Consider

1. **Query Expansion**: Add query rewriting for better retrieval
2. **Reranking**: Add cross-encoder reranking after hybrid search
3. **Streaming**: Implement streaming responses for LLM
4. **Multi-tenancy**: Add user isolation and access control
5. **Analytics**: Track query performance and metadata distribution
6. **Background Processing**: Use Celery for async document processing
7. **Evaluation**: Integrate RAGAS/DeepEval for retrieval quality

---

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

## 🙏 Acknowledgments

- **GLiNER2**: [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2)
- **Docling**: [DS4SD/docling](https://github.com/DS4SD/docling)
- **Qdrant**: [qdrant/qdrant](https://github.com/qdrant/qdrant)
- **FastEmbed**: [qdrant/fastembed](https://github.com/qdrant/fastembed)

---

**Built with ❤️ for production RAG systems**
