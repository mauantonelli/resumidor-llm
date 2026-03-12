import os
from typing import Optional

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStore:
    def __init__(
        self,
        embedding_model: str = "neuralmind/bert-base-portuguese-cased",
        persist_directory: Optional[str] = None,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.persist_directory = persist_directory
        self.store: Optional[FAISS] = None

    def create_from_documents(self, documents: list[Document]) -> None:
        self.store = FAISS.from_documents(documents, self.embeddings)
        if self.persist_directory:
            self.save()

    def add_documents(self, documents: list[Document]) -> None:
        if self.store is None:
            self.create_from_documents(documents)
        else:
            self.store.add_documents(documents)
            if self.persist_directory:
                self.save()

    def save(self) -> None:
        if self.store is None:
            raise ValueError("Nenhum vector store para salvar.")
        if self.persist_directory is None:
            raise ValueError("Diretorio de persistencia nao definido.")
        os.makedirs(self.persist_directory, exist_ok=True)
        self.store.save_local(self.persist_directory)

    def load(self) -> None:
        if self.persist_directory is None:
            raise ValueError("Diretorio de persistencia nao definido.")
        self.store = FAISS.load_local(
            self.persist_directory,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if self.store is None:
            raise ValueError("Vector store nao inicializado. Crie ou carregue primeiro.")
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        if self.store is None:
            raise ValueError("Vector store nao inicializado.")
        return self.store.similarity_search_with_score(query, k=k)
