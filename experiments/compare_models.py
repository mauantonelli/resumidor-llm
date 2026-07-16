import argparse
import json
import time
import os
from typing import Optional

from data.corpus import Corpus
from evaluation.metrics import SummaryEvaluator, SemanticEvaluator
from summarization.model_loader import SUPPORTED_MODELS
from utils import set_seed, SEED_PADRAO


def run_generative_model(model_name: str, texts: list[str]) -> tuple[list[str], float]:
    from summarization.summarizer import Summarizer

    print(f"\n  Carregando {model_name}...")
    start = time.time()
    summarizer = Summarizer(model_name=model_name)
    load_time = time.time() - start
    print(f"  Modelo carregado em {load_time:.1f}s")

    summaries = []
    for i, text in enumerate(texts):
        print(f"  Gerando resumo {i + 1}/{len(texts)}...", end=" ", flush=True)
        t0 = time.time()
        # decodificacao deterministica (greedy) para reprodutibilidade
        summary = summarizer.generate_summary(text, do_sample=False)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")
        summaries.append(summary)

    total_time = time.time() - start
    return summaries, total_time


def run_extractive_model(texts: list[str]) -> tuple[list[str], float]:
    from summarization.extractive_summarizer import ExtractiveSummarizer

    print("\n  Carregando bertimbau...")
    start = time.time()
    summarizer = ExtractiveSummarizer()
    load_time = time.time() - start
    print(f"  Modelo carregado em {load_time:.1f}s")

    summaries = []
    for i, text in enumerate(texts):
        print(f"  Gerando resumo {i + 1}/{len(texts)}...", end=" ", flush=True)
        t0 = time.time()
        summary = summarizer.generate_summary(text)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")
        summaries.append(summary)

    total_time = time.time() - start
    return summaries, total_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparacao entre modelos de sumarizacao")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Modelos a comparar (ex: gpt2 distilgpt2 bertimbau)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Arquivo JSON para salvar resultados",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Incluir avaliacao de similaridade semantica",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_PADRAO,
        help="Seed para reprodutibilidade",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.models:
        models_to_test = args.models
    else:
        models_to_test = ["gpt2", "distilgpt2", "bertimbau"]

    corpus = Corpus()
    texts, references = corpus.get_texts_and_references()
    rouge_evaluator = SummaryEvaluator()
    semantic_evaluator = None

    if args.semantic:
        print("Carregando modelo de similaridade semantica...")
        semantic_evaluator = SemanticEvaluator()

    print("=" * 60)
    print("COMPARACAO DE MODELOS DE SUMARIZACAO")
    print("=" * 60)
    print(f"Textos no corpus: {len(texts)}")
    print(f"Modelos a testar: {models_to_test}")
    if semantic_evaluator:
        print("Avaliacao semantica: ATIVADA")

    all_results = {}

    for model_name in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"MODELO: {model_name}")
        print("=" * 60)

        model_config = SUPPORTED_MODELS.get(model_name, {})

        if model_config.get("type") == "extractive":
            summaries, total_time = run_extractive_model(texts)
        else:
            summaries, total_time = run_generative_model(model_name, texts)

        rouge_scores = rouge_evaluator.evaluate_batch(references, summaries)

        result_entry = {
            "rouge_scores": rouge_scores,
            "time": round(total_time, 2),
            "summaries": summaries,
        }

        print(f"\n  Resultados ROUGE:")
        print(f"  {rouge_evaluator.format_results(rouge_scores)}")

        if semantic_evaluator:
            semantic_scores = semantic_evaluator.evaluate_batch(references, summaries)
            result_entry["semantic_scores"] = semantic_scores
            print(f"\n  Similaridade Semantica:")
            print(f"  {semantic_evaluator.format_results(semantic_scores)}")

        print(f"  Tempo total: {total_time:.1f}s")
        all_results[model_name] = result_entry

    print(f"\n\n{'=' * 60}")
    print("RESUMO COMPARATIVO")
    print("=" * 60)

    header = f"{'Modelo':<15} {'ROUGE-1 F':>10} {'ROUGE-2 F':>10} {'ROUGE-L F':>10}"
    if semantic_evaluator:
        header += f" {'Sem. Sim.':>10}"
    header += f" {'Tempo':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for model_name, result in all_results.items():
        scores = result["rouge_scores"]
        line = (
            f"{model_name:<15} "
            f"{scores['rouge1']['fmeasure']:>10.4f} "
            f"{scores['rouge2']['fmeasure']:>10.4f} "
            f"{scores['rougeL']['fmeasure']:>10.4f}"
        )
        if "semantic_scores" in result:
            line += f" {result['semantic_scores']['semantic_similarity_mean']:>10.4f}"
        line += f" {result['time']:>7.1f}s"
        print(line)

    if args.output:
        output_data = {}
        for model_name, result in all_results.items():
            entry = {
                "rouge_scores": result["rouge_scores"],
                "time": result["time"],
                "summaries": result["summaries"],
            }
            if "semantic_scores" in result:
                entry["semantic_scores"] = result["semantic_scores"]
            output_data[model_name] = entry

        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
