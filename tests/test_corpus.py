import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.corpus import Corpus, BUILTIN_SAMPLES


def test_builtin_samples_count():
    assert len(BUILTIN_SAMPLES) == 10


def test_sample_structure():
    required_keys = {"id", "title", "text", "reference_summary"}
    for sample in BUILTIN_SAMPLES:
        assert required_keys.issubset(sample.keys())


def test_sample_content_not_empty():
    for sample in BUILTIN_SAMPLES:
        assert len(sample["text"]) > 0
        assert len(sample["reference_summary"]) > 0
        assert len(sample["title"]) > 0


def test_get_builtin_samples():
    corpus = Corpus()
    samples = corpus.get_builtin_samples()
    assert len(samples) == 10
    samples[0]["title"] = "MODIFICADO"
    original = corpus.get_builtin_samples()
    assert original[0]["title"] != "MODIFICADO"


def test_get_texts_and_references():
    corpus = Corpus()
    texts, references = corpus.get_texts_and_references()
    assert len(texts) == 10
    assert len(references) == 10
    assert isinstance(texts[0], str)
    assert isinstance(references[0], str)


def test_save_and_load_json():
    corpus = Corpus()
    samples = corpus.get_builtin_samples()[:2]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        corpus.save_to_json(samples, temp_path)
        loaded = corpus.load_from_json(temp_path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == samples[0]["id"]
        assert loaded[1]["title"] == samples[1]["title"]
    finally:
        os.unlink(temp_path)
