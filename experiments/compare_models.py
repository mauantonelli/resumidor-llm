import argparse
import hashlib
import json
import time
import os
from typing import Optional

from data.corpus import Corpus
from evaluation.metrics import SummaryEvaluator, SemanticEvaluator
from summarization.model_loader import SUPPORTED_MODELS
from utils import set_seed, SEED_PADRAO


def _hash_corpus(texts: list[str]) -> str:
    """Identifica o corpus para invalidar checkpoints de outra execucao."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def _ckpt_path(checkpoint_dir: Optional[str], model_name: str) -> Optional[str]:
    if not checkpoint_dir:
        return None
    return os.path.join(checkpoint_dir, f"ckpt_{model_name.replace('/', '_')}.json")


def _carregar_ckpt(path, corpus_hash):
    if not path or not os.path.exists(path):
        return [], []
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return [], []
    if d.get("corpus_hash") != corpus_hash:
        return [], []  # checkpoint de outro corpus: ignora
    return d.get("summaries", []), d.get("tempos", [])


def _salvar_ckpt(path, corpus_hash, summaries, tempos):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"corpus_hash": corpus_hash, "summaries": summaries,
                   "tempos": tempos}, f, ensure_ascii=False)
    os.replace(tmp, path)  # escrita atomica: nao corrompe se morrer no meio


def _fabricar_resumidor(model_name: str, config: dict):
    """Devolve uma funcao resumir(texto) -> str para o tipo do modelo."""
    tipo = config.get("type")
    if tipo == "extractive":
        from summarization.extractive_summarizer import ExtractiveSummarizer
        s = ExtractiveSummarizer()
        return lambda t: s.generate_summary(t)
    if tipo == "seq2seq":
        from summarization.seq2seq_summarizer import Seq2SeqSummarizer
        s = Seq2SeqSummarizer(model_name=model_name)
        return lambda t: s.generate_summary(t)
    if tipo == "seq2seq_chunk":
        from summarization.chunked_summarizer import ChunkedSeq2SeqSummarizer
        s = ChunkedSeq2SeqSummarizer(model_name=model_name)
        return lambda t: s.generate_summary(t)
    from summarization.summarizer import Summarizer
    s = Summarizer(model_name=model_name)
    # decodificacao deterministica (greedy) para reprodutibilidade
    return lambda t: s.generate_summary(t, do_sample=False)


def gerar_resumos(model_name, config, texts, checkpoint_dir=None):
    """Gera os resumos de um modelo, com checkpoint por artigo.

    Rodadas longas (ex.: chunking) podem ser interrompidas; o checkpoint
    permite retomar de onde parou em vez de perder tudo. O tempo reportado
    e load + soma dos tempos de geracao (exclui pausas entre execucoes).
    """
    corpus_hash = _hash_corpus(texts)
    path = _ckpt_path(checkpoint_dir, model_name)
    summaries, tempos = _carregar_ckpt(path, corpus_hash)

    if len(summaries) >= len(texts):
        print(f"\n  [ckpt] {model_name}: {len(texts)}/{len(texts)} ja concluido — "
              f"reaproveitando")
        return summaries[:len(texts)], float(sum(tempos[:len(texts)]))

    if summaries:
        print(f"\n  [ckpt] {model_name}: retomando de {len(summaries)}/{len(texts)}")

    print(f"\n  Carregando {model_name}...")
    t0 = time.time()
    resumir = _fabricar_resumidor(model_name, config)
    load_time = time.time() - t0
    print(f"  Modelo carregado em {load_time:.1f}s")

    for i in range(len(summaries), len(texts)):
        print(f"  Gerando resumo {i + 1}/{len(texts)}...", end=" ", flush=True)
        t0 = time.time()
        resumo = resumir(texts[i])
        dt = time.time() - t0
        print(f"({dt:.1f}s)")
        summaries.append(resumo)
        tempos.append(dt)
        _salvar_ckpt(path, corpus_hash, summaries, tempos)

    return summaries, load_time + float(sum(tempos))


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
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Diretorio de checkpoints por artigo (permite retomar rodadas longas)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Caminho para um corpus JSON (padrao: corpus sintetico embutido)",
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
    if args.corpus:
        samples = corpus.load_from_json(args.corpus)
        texts, references = corpus.get_texts_and_references(samples)
        print(f"Corpus externo: {args.corpus} ({len(texts)} textos)")
    else:
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
        summaries, total_time = gerar_resumos(
            model_name, model_config, texts, checkpoint_dir=args.checkpoint_dir
        )

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
