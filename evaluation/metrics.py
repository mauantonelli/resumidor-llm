from typing import Optional

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer


class SummaryEvaluator:
    def __init__(self, metrics: Optional[list[str]] = None):
        if metrics is None:
            metrics = ["rouge1", "rouge2", "rougeL"]
        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=False)

    def evaluate_single(self, reference: str, hypothesis: str) -> dict[str, dict[str, float]]:
        scores = self.scorer.score(reference, hypothesis)
        result = {}
        for metric_name, score in scores.items():
            result[metric_name] = {
                "precision": round(score.precision, 4),
                "recall": round(score.recall, 4),
                "fmeasure": round(score.fmeasure, 4),
            }
        return result

    def evaluate_batch(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, dict[str, float]]:
        all_scores = {m: {"precision": [], "recall": [], "fmeasure": []} for m in self.metrics}

        for ref, hyp in zip(references, hypotheses):
            scores = self.evaluate_single(ref, hyp)
            for metric_name, values in scores.items():
                for key, value in values.items():
                    all_scores[metric_name][key].append(value)

        averaged = {}
        for metric_name, values in all_scores.items():
            averaged[metric_name] = {
                key: round(float(np.mean(vals)), 4)
                for key, vals in values.items()
            }
        return averaged

    def format_results(self, results: dict[str, dict[str, float]]) -> str:
        lines = []
        lines.append(f"{'Metrica':<15} {'Precision':>10} {'Recall':>10} {'F-measure':>10}")
        lines.append("-" * 48)
        for metric_name, values in results.items():
            lines.append(
                f"{metric_name:<15} "
                f"{values['precision']:>10.4f} "
                f"{values['recall']:>10.4f} "
                f"{values['fmeasure']:>10.4f}"
            )
        return "\n".join(lines)


class SemanticEvaluator:
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

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

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        token_embeddings = outputs.last_hidden_state
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        mean_pooled = summed / counts

        return mean_pooled.squeeze().cpu().numpy()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def evaluate_single(self, reference: str, hypothesis: str) -> dict[str, float]:
        ref_emb = self._get_embedding(reference)
        hyp_emb = self._get_embedding(hypothesis)
        similarity = self._cosine_similarity(ref_emb, hyp_emb)
        return {"semantic_similarity": round(similarity, 4)}

    def evaluate_batch(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, float]:
        similarities = []
        for ref, hyp in zip(references, hypotheses):
            result = self.evaluate_single(ref, hyp)
            similarities.append(result["semantic_similarity"])

        return {
            "semantic_similarity_mean": round(float(np.mean(similarities)), 4),
            "semantic_similarity_std": round(float(np.std(similarities)), 4),
            "semantic_similarity_min": round(float(np.min(similarities)), 4),
            "semantic_similarity_max": round(float(np.max(similarities)), 4),
        }

    def format_results(self, results: dict[str, float]) -> str:
        lines = []
        lines.append(f"{'Metrica':<30} {'Valor':>10}")
        lines.append("-" * 42)
        for metric_name, value in results.items():
            lines.append(f"{metric_name:<30} {value:>10.4f}")
        return "\n".join(lines)
