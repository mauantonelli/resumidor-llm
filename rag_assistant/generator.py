from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RAGGenerator:
    def __init__(
        self,
        model_name: str = "pierreguillou/gpt2-small-portuguese",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            f"Contexto:\n{context}\n\n"
            f"Pergunta: {query}\n\n"
            f"Resposta baseada no contexto acima:\n"
        )

    def generate(
        self,
        query: str,
        context: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> str:
        prompt = self._build_prompt(query, context)

        # GPT-2 (e afins) tem janela de posicoes limitada (n_positions).
        # O prompt precisa ser truncado deixando espaco para os tokens
        # gerados, senao prompt + max_new_tokens estoura o limite de
        # posicoes e o embedding posicional lanca IndexError.
        model_max = (
            getattr(self.model.config, "n_positions", None)
            or getattr(self.model.config, "max_position_embeddings", 1024)
        )
        max_input = max(1, model_max - self.max_new_tokens)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][prompt_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
