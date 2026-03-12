from typing import Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SUPPORTED_MODELS = {
    "gpt2": {
        "model_id": "gpt2",
        "type": "causal",
        "quantize": False,
    },
    "distilgpt2": {
        "model_id": "distilgpt2",
        "type": "causal",
        "quantize": False,
    },
    "llama2": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "type": "causal",
        "quantize": True,
    },
    "bertimbau": {
        "model_id": "neuralmind/bert-base-portuguese-cased",
        "type": "extractive",
        "quantize": False,
    },
}


class ModelLoader:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if self.device == "cuda":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def load_model(self, model_name: str) -> Tuple[Any, AutoTokenizer]:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' não suportado. "
                f"Modelos disponíveis: {list(SUPPORTED_MODELS.keys())}"
            )

        config = SUPPORTED_MODELS[model_name]
        model_id = config["model_id"]

        if config["type"] == "extractive":
            raise ValueError(
                f"Modelo '{model_name}' é extrativo. "
                f"Use ExtractiveSummarizer para este modelo."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        load_kwargs = {}
        if config["quantize"]:
            quant_config = self._get_quantization_config()
            if quant_config:
                load_kwargs["quantization_config"] = quant_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if "device_map" not in load_kwargs:
            model = model.to(self.device)

        model.eval()
        return model, tokenizer

    def get_model_info(self, model_name: str) -> dict:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Modelo '{model_name}' não suportado.")
        return SUPPORTED_MODELS[model_name].copy()

    def list_available_models(self) -> list[str]:
        return list(SUPPORTED_MODELS.keys())
