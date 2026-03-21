import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from summarization.model_loader import ModelLoader, SUPPORTED_MODELS


def test_supported_models_keys():
    expected = {"gpt2", "distilgpt2", "llama2", "bertimbau"}
    assert set(SUPPORTED_MODELS.keys()) == expected


def test_list_available_models():
    loader = ModelLoader(device="cpu")
    models = loader.list_available_models()
    assert "gpt2" in models
    assert "distilgpt2" in models
    assert "bertimbau" in models
    assert "llama2" in models


def test_get_model_info_gpt2():
    loader = ModelLoader(device="cpu")
    info = loader.get_model_info("gpt2")
    assert info["model_id"] == "gpt2"
    assert info["type"] == "causal"
    assert info["quantize"] is False


def test_get_model_info_bertimbau():
    loader = ModelLoader(device="cpu")
    info = loader.get_model_info("bertimbau")
    assert info["type"] == "extractive"


def test_get_model_info_llama2():
    loader = ModelLoader(device="cpu")
    info = loader.get_model_info("llama2")
    assert info["quantize"] is True


def test_invalid_model_info():
    loader = ModelLoader(device="cpu")
    with pytest.raises(ValueError, match="não suportado"):
        loader.get_model_info("modelo_invalido")


def test_invalid_model_load():
    loader = ModelLoader(device="cpu")
    with pytest.raises(ValueError, match="não suportado"):
        loader.load_model("modelo_invalido")


def test_extractive_model_raises():
    loader = ModelLoader(device="cpu")
    with pytest.raises(ValueError, match="extrativo"):
        loader.load_model("bertimbau")


def test_get_model_info_returns_copy():
    loader = ModelLoader(device="cpu")
    info1 = loader.get_model_info("gpt2")
    info1["model_id"] = "MODIFICADO"
    info2 = loader.get_model_info("gpt2")
    assert info2["model_id"] == "gpt2"
