import os
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.schema import Document


class DocumentLoader:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_text_file(self, filepath: str) -> list[Document]:
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        return self.splitter.split_documents(docs)

    def load_pdf_file(self, filepath: str) -> list[Document]:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        return self.splitter.split_documents(docs)

    def load_directory(
        self,
        directory: str,
        glob_pattern: str = "**/*.txt",
    ) -> list[Document]:
        loader = DirectoryLoader(
            directory,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        return self.splitter.split_documents(docs)

    def load_from_path(self, path: str) -> list[Document]:
        if os.path.isdir(path):
            txt_docs = self.load_directory(path, "**/*.txt")
            pdf_docs = self.load_directory_pdfs(path)
            return txt_docs + pdf_docs
        elif path.endswith(".pdf"):
            return self.load_pdf_file(path)
        else:
            return self.load_text_file(path)

    def load_directory_pdfs(self, directory: str) -> list[Document]:
        all_docs = []
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.lower().endswith(".pdf"):
                    fpath = os.path.join(root, fname)
                    all_docs.extend(self.load_pdf_file(fpath))
        return all_docs
