import re
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from summarization.preprocessing import TextPreprocessor


class ExtractiveSummarizer:
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        num_sentences: int = 3,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_sentences = num_sentences

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.preprocessor = TextPreprocessor()

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return cls_embedding.cpu().numpy()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def generate_summary(self, text: str, num_sentences: Optional[int] = None) -> str:
        if num_sentences is None:
            num_sentences = self.num_sentences

        cleaned = self.preprocessor.clean_text(text)
        sentences = self._split_sentences(cleaned)

        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        doc_embedding = self._get_embedding(cleaned[:512])
        sentence_scores = []

        for i, sentence in enumerate(sentences):
            sent_embedding = self._get_embedding(sentence)
            similarity = self._cosine_similarity(doc_embedding, sent_embedding)
            sentence_scores.append((i, similarity))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, _ in sentence_scores[:num_sentences]])
        summary_sentences = [sentences[i] for i in top_indices]

        return " ".join(summary_sentences)

    def batch_summarize(self, texts: list[str], **kwargs) -> list[str]:
        return [self.generate_summary(text, **kwargs) for text in texts]
