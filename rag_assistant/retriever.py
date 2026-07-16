from typing import Optional

from langchain_core.documents import Document

from rag_assistant.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[Document]:
        if self.score_threshold is not None:
            results = self.vector_store.similarity_search_with_score(
                query, k=self.top_k
            )
            filtered = [
                doc for doc, score in results if score <= self.score_threshold
            ]
            return filtered
        return self.vector_store.similarity_search(query, k=self.top_k)

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(
            query, k=self.top_k
        )

    def format_context(self, documents: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "desconhecido")
            parts.append(f"[{i + 1}] (Fonte: {source})\n{doc.page_content}")
        return "\n\n".join(parts)
