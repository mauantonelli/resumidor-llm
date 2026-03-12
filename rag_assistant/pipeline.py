from typing import Optional

from rag_assistant.document_loader import DocumentLoader
from rag_assistant.vector_store import VectorStore
from rag_assistant.retriever import Retriever
from rag_assistant.generator import RAGGenerator


class RAGPipeline:
    def __init__(
        self,
        generator_model: str = "gpt2",
        embedding_model: str = "neuralmind/bert-base-portuguese-cased",
        persist_directory: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 4,
        max_new_tokens: int = 256,
    ):
        self.doc_loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = VectorStore(
            embedding_model=embedding_model,
            persist_directory=persist_directory,
        )
        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=top_k,
        )
        self.generator = RAGGenerator(
            model_name=generator_model,
            max_new_tokens=max_new_tokens,
        )
        self._indexed = False

    def index_documents(self, path: str) -> int:
        documents = self.doc_loader.load_from_path(path)
        if not documents:
            raise ValueError(f"Nenhum documento encontrado em: {path}")
        self.vector_store.create_from_documents(documents)
        self._indexed = True
        return len(documents)

    def load_index(self) -> None:
        self.vector_store.load()
        self._indexed = True

    def query(
        self,
        question: str,
        temperature: float = 0.7,
    ) -> dict[str, str]:
        if not self._indexed:
            raise ValueError("Nenhum indice carregado. Use index_documents() ou load_index().")

        docs = self.retriever.retrieve(question)
        context = self.retriever.format_context(docs)

        answer = self.generator.generate(
            query=question,
            context=context,
            temperature=temperature,
        )

        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "desconhecido")
            if source not in sources:
                sources.append(source)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": sources,
        }

    def interactive(self) -> None:
        if not self._indexed:
            raise ValueError("Nenhum indice carregado.")

        print("RAG Assistant - IFSP")
        print("Digite 'sair' para encerrar.\n")

        while True:
            question = input("Pergunta: ").strip()
            if question.lower() in ("sair", "exit", "quit"):
                print("Encerrando.")
                break
            if not question:
                continue

            result = self.query(question)
            print(f"\nResposta: {result['answer']}")
            print(f"\nFontes: {', '.join(result['sources'])}")
            print("-" * 40 + "\n")
