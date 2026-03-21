import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from summarization.preprocessing import TextPreprocessor


def test_clean_text_whitespace():
    preprocessor = TextPreprocessor()
    result = preprocessor.clean_text("  texto   com   espacos   ")
    assert result == "texto com espacos"


def test_clean_text_unicode():
    preprocessor = TextPreprocessor()
    result = preprocessor.clean_text("caf\u00e9")
    assert "caf" in result


def test_remove_references_brackets():
    preprocessor = TextPreprocessor()
    text = "Estudos mostram [1,2] que a IA evolui [3]."
    result = preprocessor.remove_references(text)
    assert "[1,2]" not in result
    assert "[3]" not in result
    assert "Estudos mostram" in result


def test_remove_references_section():
    preprocessor = TextPreprocessor()
    text = "Texto principal.\n\nReferências Bibliográficas\nALVES, J. 2020."
    result = preprocessor.remove_references(text)
    assert "ALVES" not in result
    assert "Texto principal" in result


def test_truncate_text_within_limit():
    preprocessor = TextPreprocessor(max_length=10)
    text = "uma dois tres"
    result = preprocessor.truncate_text(text)
    assert result == "uma dois tres"


def test_truncate_text_exceeds_limit():
    preprocessor = TextPreprocessor(max_length=3)
    text = "uma dois tres quatro cinco"
    result = preprocessor.truncate_text(text)
    words = result.split()
    assert len(words) == 3


def test_truncate_text_custom_max():
    preprocessor = TextPreprocessor()
    text = "uma dois tres quatro cinco seis"
    result = preprocessor.truncate_text(text, max_words=2)
    assert result == "uma dois"


def test_prepare_for_summarization():
    preprocessor = TextPreprocessor(max_length=50)
    text = "  Texto  com espacos [1,2] e  referencias.  \n\nReferências Bibliográficas\nSILVA, 2020."
    result = preprocessor.prepare_for_summarization(text)
    assert "[1,2]" not in result
    assert "SILVA" not in result
    assert "Texto com espacos" in result
