from typing import Optional

import torch

from summarization.model_loader import ModelLoader
from summarization.preprocessing import TextPreprocessor


class Seq2SeqSummarizer:
    """Sumarizador abstrativo baseado em modelo seq2seq (ex.: PTT5 ajustado
    para sumarizacao em portugues).

    Diferente do Summarizer causal (que recebe um prompt e gera a
    continuacao), aqui o texto pre-processado e a entrada do encoder e a
    saida do decoder e o resumo — nao ha prompt a remover.

    Decodificacao deterministica por beam search (do_sample=False), coerente
    com a politica de reprodutibilidade da avaliacao.
    """

    def __init__(
        self,
        model_name: str = "ptt5-summ",
        max_input_length: int = 512,
        max_summary_length: int = 150,
        num_beams: int = 4,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_summary_length = max_summary_length
        self.num_beams = num_beams
        self.preprocessor = TextPreprocessor(max_length=max_input_length)
        self.loader = ModelLoader(device=device)
        self.model, self.tokenizer = self.loader.load_model(model_name)
        self.device = self.loader.device

    def generate_summary(
        self,
        text: str,
        num_beams: Optional[int] = None,
    ) -> str:
        if num_beams is None:
            num_beams = self.num_beams

        processed_text = self.preprocessor.prepare_for_summarization(text)

        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_summary_length,
                num_beams=num_beams,
                do_sample=False,
                repetition_penalty=1.2,
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()

    def batch_summarize(self, texts: list[str], **kwargs) -> list[str]:
        return [self.generate_summary(text, **kwargs) for text in texts]
