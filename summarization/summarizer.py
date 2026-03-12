from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from summarization.model_loader import ModelLoader
from summarization.preprocessing import TextPreprocessor


class Summarizer:
    def __init__(
        self,
        model_name: str = "gpt2",
        max_input_length: int = 512,
        max_summary_length: int = 150,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_summary_length = max_summary_length
        self.preprocessor = TextPreprocessor(max_length=max_input_length)
        self.loader = ModelLoader(device=device)
        self.model, self.tokenizer = self.loader.load_model(model_name)
        self.device = self.loader.device

    def _build_prompt(self, text: str) -> str:
        return (
            f"Texto científico:\n{text}\n\n"
            f"Resumo do texto acima:\n"
        )

    def generate_summary(
        self,
        text: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
        repetition_penalty: float = 1.2,
    ) -> str:
        processed_text = self.preprocessor.prepare_for_summarization(text)
        prompt = self._build_prompt(processed_text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_summary_length,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][prompt_length:]
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        summary = summary.strip()

        return summary

    def batch_summarize(self, texts: list[str], **kwargs) -> list[str]:
        return [self.generate_summary(text, **kwargs) for text in texts]
