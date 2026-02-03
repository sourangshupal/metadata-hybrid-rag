from typing import List, Dict, Any, Optional
from openai import OpenAI
from loguru import logger


class RetrievalService:
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
        entity_filter: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        return self.vector_store.hybrid_search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            domain_filter=domain_filter,
            content_type_filter=content_type_filter,
            entity_filter=entity_filter,
        )

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for idx, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source_file", "Unknown")
            domain = metadata.get("domain", "")

            context_parts.append(
                f"[Source {idx}: {source} | Domain: {domain}]\n{chunk['text']}\n"
            )

        context = "\n---\n".join(context_parts)

        system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information, say so honestly.
Always cite which source(s) you used in your answer."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        logger.info("Generating answer with LLM")

        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        logger.info("Answer generated successfully")

        return answer
