from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import sys
import time

from app.config import settings
from app.models import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
    FileType,
)
from app.services.document_processor import DocumentProcessor
from app.services.metadata_extractor import MetadataExtractor
from app.services.vector_store import VectorStore
from app.services.retrieval import RetrievalService

logger.remove()
logger.add(sys.stderr, level=settings.log_level)

document_processor: DocumentProcessor = None
metadata_extractor: MetadataExtractor = None
vector_store: VectorStore = None
retrieval_service: RetrievalService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_processor, metadata_extractor, vector_store, retrieval_service

    logger.info("Initializing services...")

    document_processor = DocumentProcessor(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    metadata_extractor = MetadataExtractor(model_name=settings.gliner_model)

    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        openai_api_key=settings.openai_api_key,
        embedding_model=settings.openai_embedding_model,
        embedding_dimension=settings.openai_embedding_dimension,
        api_key=settings.qdrant_api_key,
    )

    retrieval_service = RetrievalService(
        vector_store=vector_store,
        openai_api_key=settings.openai_api_key,
        llm_model=settings.openai_llm_model,
    )

    logger.info("All services initialized successfully")

    yield

    logger.info("Shutting down services...")


app = FastAPI(
    title="RAG with Metadata Enrichment",
    description="Document upload and retrieval with GLiNER2 metadata extraction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "RAG Metadata Enrichment API",
        "version": "1.0.0",
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    start_time = time.time()

    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in [ft.value for ft in FileType]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Supported: pdf, md, txt, json",
        )

    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size_mb:.2f}MB. Max: {settings.max_file_size_mb}MB",
        )

    logger.info(f"Processing upload: {file.filename} ({file_size_mb:.2f}MB)")

    try:
        processed = document_processor.process_uploaded_file(
            file_bytes=file_bytes, filename=file.filename, file_type=file_extension
        )

        chunks_with_metadata = []
        for chunk_data in processed["chunks"]:
            metadata_dict = metadata_extractor.extract_metadata(chunk_data["text"])

            tech_specs = metadata_dict.get("tech_specs")
            if isinstance(tech_specs, list) and len(tech_specs) > 0:
                tech_specs = tech_specs[0]
            elif not isinstance(tech_specs, dict):
                tech_specs = {}

            from app.models import ChunkMetadata

            chunk_metadata = ChunkMetadata(
                entities=metadata_dict.get("entities", {}),
                domain=metadata_dict.get("domain"),
                content_type=metadata_dict.get("content_type"),
                tech_specs=tech_specs,
                source_file=processed["file_name"],
                file_type=file_extension,
                chunk_index=chunk_data["chunk_index"],
                total_chunks=chunk_data["total_chunks"],
                char_count=chunk_data["char_count"],
                page_number=chunk_data.get("page_number"),
            )

            chunks_with_metadata.append(
                {"text": chunk_data["text"], "metadata": chunk_metadata}
            )

        chunk_ids = vector_store.add_chunks(chunks_with_metadata)

        processing_time = time.time() - start_time

        logger.info(f"Successfully processed {file.filename} in {processing_time:.2f}s")

        return UploadResponse(
            file_name=file.filename,
            file_type=file_extension,
            total_chunks=len(chunks_with_metadata),
            processing_time_seconds=round(processing_time, 2),
            chunk_ids=chunk_ids,
        )

    except Exception as e:
        logger.exception(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    logger.info(
        f"Query: '{request.query}' with filters: domain={request.domain_filter}, content_type={request.content_type_filter}"
    )

    try:
        retrieval_start = time.time()

        results = retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            alpha=request.hybrid_alpha,
            domain_filter=request.domain_filter,
            content_type_filter=request.content_type_filter,
            entity_filter=request.entity_filter,
        )

        retrieval_time = time.time() - retrieval_start

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Try adjusting your filters.",
            )

        llm_start = time.time()

        answer = retrieval_service.generate_answer(
            query=request.query, context_chunks=results
        )

        llm_time = time.time() - llm_start

        from app.models import SearchResult, ChunkMetadata

        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    chunk_id=result["chunk_id"],
                    text=result["text"],
                    score=result["score"],
                    metadata=ChunkMetadata(**result["metadata"]),
                )
            )

        return QueryResponse(
            query=request.query,
            answer=answer,
            results=search_results,
            retrieval_time_seconds=round(retrieval_time, 2),
            llm_time_seconds=round(llm_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "document_processor": document_processor is not None,
            "metadata_extractor": metadata_extractor is not None,
            "vector_store": vector_store is not None,
            "retrieval_service": retrieval_service is not None,
        },
        "config": {
            "gliner_model": settings.gliner_model,
            "qdrant_collection": settings.qdrant_collection_name,
            "embedding_model": settings.openai_embedding_model,
            "llm_model": settings.openai_llm_model,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
