# HYBRID RAG with Metadata Enrichment

Production-ready RAG system with metadata enrichment using GLiNER2, Qdrant, and OpenAI.

## Features

- Multi-format document upload (PDF, Markdown, TXT, JSON)
- Docling HybridChunker for intelligent semantic chunking
- GLiNER2 for zero-shot metadata extraction
- Qdrant vector store with hybrid search (BM25 + dense embeddings)
- OpenAI GPT-4o-mini for answer generation
- Metadata filtering by domain, content type, and entities

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
