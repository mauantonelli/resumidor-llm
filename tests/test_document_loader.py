import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_assistant.document_loader import DocumentLoader


def test_load_text_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("Este e um texto de teste para o carregamento de documentos.")
        temp_path = f.name

    try:
        loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
        docs = loader.load_text_file(temp_path)
        assert len(docs) >= 1
        assert "texto de teste" in docs[0].page_content
    finally:
        os.unlink(temp_path)


def test_load_text_file_large():
    content = "Esta e uma sentenca de teste. " * 200
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
        docs = loader.load_text_file(temp_path)
        assert len(docs) > 1
        for doc in docs:
            assert len(doc.page_content) <= 150
    finally:
        os.unlink(temp_path)


def test_load_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            filepath = os.path.join(tmpdir, f"doc_{i}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Documento numero {i} com conteudo de teste.")

        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        docs = loader.load_directory(tmpdir)
        assert len(docs) >= 3


def test_load_from_path_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("Teste de carregamento via load_from_path.")
        temp_path = f.name

    try:
        loader = DocumentLoader()
        docs = loader.load_from_path(temp_path)
        assert len(docs) >= 1
    finally:
        os.unlink(temp_path)


def test_load_from_path_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "teste.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Conteudo do arquivo de teste.")

        loader = DocumentLoader()
        docs = loader.load_from_path(tmpdir)
        assert len(docs) >= 1


def test_chunk_overlap():
    content = "Palavra " * 500
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
        docs = loader.load_text_file(temp_path)
        assert len(docs) > 1
    finally:
        os.unlink(temp_path)
