# HYBRID RAG with Metadata Enrichment

Production-ready RAG system with metadata enrichment using GLiNER2, Qdrant, and OpenAI.

## Features

- Multi-format document upload (PDF, Markdown, TXT, JSON)
- Docling HybridChunker for intelligent semantic chunking
- GLiNER2 for zero-shot metadata extraction (entities, domain, content type, tech specs)
- Qdrant vector store with hybrid search (BM25 + dense embeddings)
- OpenAI GPT-4o-mini for answer generation
- Metadata filtering by domain, content type, and entities

## Project Structure

```
metadata-hybrid-rag/
├── app/                          # FastAPI application
│   ├── main.py                   # API entrypoint
│   ├── models.py                 # Pydantic schemas (ChunkMetadata, QueryRequest)
│   ├── config.py                 # Settings via pydantic-settings
│   └── services/
│       ├── document_processor.py # Docling HybridChunker pipeline
│       ├── metadata_extractor.py # GLiNER2 extraction (entities, domain, content_type)
│       ├── vector_store.py       # Qdrant client wrapper with metadata filters
│       └── retrieval.py          # Retrieval + GPT-4o-mini generation
├── data/
│   └── raw/                      # Sample PDFs (RAG, Transformers, AWS, OWASP, K8s)
├── notebooks/
│   ├── metadata_enrichment_tutorial.ipynb  # Tutorial: baseline vs enriched RAG
│   └── gliner2_complete_features.ipynb     # GLiNER2 all features showcase
└── docker-compose.yml            # Qdrant + app
```

## Notebooks

### `notebooks/metadata_enrichment_tutorial.ipynb`
End-to-end tutorial comparing RAG without metadata vs with GLiNER2-enriched metadata.
- How to design metadata for any RAG system (4 categories, decision framework)
- Baseline ingestion (`baseline_rag` collection) — structural fields only
- Enriched ingestion (`enriched_rag` collection) — full GLiNER2 pipeline with timing
- Domain, content-type, entity, and combined filter demos
- Side-by-side retrieval quality comparison across 4 benchmark queries
- Ingestion and retrieval timing tables

**Requirements:** OpenAI API key, Qdrant running (`docker-compose up -d`)

### `notebooks/gliner2_complete_features.ipynb`
Comprehensive showcase of all GLiNER2 capabilities — no Qdrant or OpenAI key needed.
- Basic NER (list labels vs. descriptions)
- Single-label and multi-label text classification
- Confidence scores with filtering
- Structured extraction (basic fields, choices/enum, per-field thresholds)
- RegexValidator (full match, partial match, exclude modes)
- Multi-task combined extraction in one forward pass
- Batch processing (`batch_extract`, `batch_classify_text`) with timing
- Schema Builder API (full HR/CV pipeline)
- Schema caching best practices
- Base vs Large model comparison
- End-to-end news intelligence pipeline

**Requirements:** `pip install gliner` only (models download on first run ~500MB)

## Setup

1. Copy environment variables:
   ```bash
   cp .env.example .env
   ```

2. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. Start Qdrant:
   ```bash
   docker-compose up -d
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `POST /upload` - Upload and process documents
- `POST /query` - Query documents with metadata filters
- `GET /` - Health check
- `GET /health` - Detailed health and configuration info

## API Documentation

Interactive API docs available at: http://localhost:8000/docs
