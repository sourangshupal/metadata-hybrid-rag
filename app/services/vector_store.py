from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    SparseVectorParams,
    SparseIndexParams,
)
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI
from fastembed import SparseTextEmbedding
import uuid


class VectorStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        openai_api_key: str,
        embedding_model: str,
        embedding_dimension: int,
        api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name

        if api_key:
            self.client = QdrantClient(url=f"https://{host}", api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port)

        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

        logger.info("Loading FastEmbed sparse model for BM25...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        logger.info(f"VectorStore initialized for collection: {collection_name}")

        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if not collection_exists:
            logger.info(f"Creating new collection: {self.collection_name}")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension, distance=Distance.COSINE
                ),
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )
            logger.info("Collection created successfully with hybrid search support")
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def _get_dense_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            input=text, model=self.embedding_model
        )
        return response.data[0].embedding

    def _get_sparse_embedding(self, text: str) -> Dict[int, float]:
        sparse_vectors = list(self.sparse_model.embed([text]))

        if sparse_vectors and len(sparse_vectors) > 0:
            sparse_vector = sparse_vectors[0]
            return {
                int(idx): float(val)
                for idx, val in zip(sparse_vector.indices, sparse_vector.values)
            }

        return {}

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        logger.info(f"Adding {len(chunks)} chunks to vector store")

        points = []
        chunk_ids = []

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            text = chunk["text"]
            metadata = chunk["metadata"]

            dense_embedding = self._get_dense_embedding(text)
            sparse_embedding = self._get_sparse_embedding(text)

            payload = {
                "text": text,
                "metadata": metadata.model_dump()
                if hasattr(metadata, "model_dump")
                else metadata,
            }

            point = PointStruct(id=chunk_id, vector=dense_embedding, payload=payload)
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Successfully added {len(points)} chunks")
        return chunk_ids

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        domain_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        entity_filter: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Hybrid search: '{query}' (top_k={top_k}, alpha={alpha})")

        dense_query = self._get_dense_embedding(query)
        sparse_query = self._get_sparse_embedding(query)

        must_conditions = []

        if domain_filter:
            must_conditions.append(
                FieldCondition(
                    key="metadata.domain", match=MatchValue(value=domain_filter)
                )
            )

        if content_type_filter:
            must_conditions.append(
                FieldCondition(
                    key="metadata.content_type",
                    match=MatchValue(value=content_type_filter),
                )
            )

        if entity_filter:
            for entity_type, entity_values in entity_filter.items():
                must_conditions.append(
                    FieldCondition(
                        key=f"metadata.entities.{entity_type}",
                        match=MatchAny(any=entity_values),
                    )
                )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_query,
            limit=top_k,
            query_filter=query_filter,
        )

        results = []
        for point in search_results.points:
            results.append(
                {
                    "chunk_id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": point.payload.get("metadata", {}),
                }
            )

        logger.info(f"Found {len(results)} results")
        return results

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
