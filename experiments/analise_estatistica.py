"""Analise de dispersao dos resultados da comparacao.

Recalcula o ROUGE **por amostra** a partir dos resumos ja salvos por
experiments/compare_models.py (nao roda modelo de novo) e reporta:

  - media, desvio-padrao e intervalo de confianca de 95% (bootstrap) por modelo
  - o IC 95% da **diferenca pareada** entre dois modelos

Objetivo: verificar se diferencas pequenas entre modelos se sustentam ou se
estao dentro do ruido amostral. E estatistica **descritiva** — reporta
incerteza, nao executa teste de hipotese nem declara significancia.

Uso:
    python -m experiments.analise_estatistica \
        --input experiments/results/comparacao_scielo.json \
        --corpus data/processed/corpus_scielo.json \
        --comparar bertimbau ptt5-summ

Reproduzivel: bootstrap com seed fixa.
"""

import argparse
import json

import numpy as np

from data.corpus import Corpus
from evaluation.metrics import SummaryEvaluator, SemanticEvaluator
from utils import set_seed, SEED_PADRAO


N_BOOTSTRAP = 10000
ROUGE_METRICAS = ["rouge1", "rouge2", "rougeL"]
SEMANTICA = "semantica"


def scores_por_amostra(referencias, resumos, semantic_ev=None):
    """Devolve {metrica: np.array de score por amostra}."""
    ev = SummaryEvaluator()
    out = {m: [] for m in ROUGE_METRICAS}
    for ref, hyp in zip(referencias, resumos):
        s = ev.evaluate_single(ref, hyp)
        for m in ROUGE_METRICAS:
            out[m].append(s[m]["fmeasure"])
    if semantic_ev is not None:
        out[SEMANTICA] = [
            semantic_ev.evaluate_single(ref, hyp)["semantic_similarity"]
            for ref, hyp in zip(referencias, resumos)
        ]
    return {m: np.array(v, dtype=float) for m, v in out.items()}


def ic_bootstrap(valores, n_boot=N_BOOTSTRAP, alpha=0.05):
    """IC percentil da media via bootstrap."""
    n = len(valores)
    idx = np.random.randint(0, n, size=(n_boot, n))
    medias = valores[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ic_bootstrap_diferenca(a, b, n_boot=N_BOOTSTRAP, alpha=0.05):
    """IC da media da diferenca pareada (a - b). Reamostra as amostras
    (mesmos indices para os dois modelos: pareado)."""
    d = a - b
    n = len(d)
    idx = np.random.randint(0, n, size=(n_boot, n))
    medias = d[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def parse_args():
    p = argparse.ArgumentParser(description="Dispersao/IC dos resultados")
    p.add_argument("--input", default="experiments/results/comparacao_scielo.json")
    p.add_argument("--corpus", default=None,
                   help="Corpus JSON usado na execucao (padrao: sintetico embutido)")
    p.add_argument("--comparar", nargs=2, default=None,
                   metavar=("MODELO_A", "MODELO_B"),
                   help="Par de modelos para o IC da diferenca pareada")
    p.add_argument("--semantic", action="store_true",
                   help="Incluir similaridade semantica (carrega o encoder BERTimbau)")
    p.add_argument("--seed", type=int, default=SEED_PADRAO)
    p.add_argument("--out", default=None, help="Salvar analise em JSON")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.input, encoding="utf-8") as f:
        resultados = json.load(f)

    corpus = Corpus()
    if args.corpus:
        amostras = corpus.load_from_json(args.corpus)
        _, referencias = corpus.get_texts_and_references(amostras)
    else:
        _, referencias = corpus.get_texts_and_references()

    print(f"Amostras no corpus: {len(referencias)}")
    print(f"Bootstrap: {N_BOOTSTRAP} reamostragens, seed {args.seed}\n")

    semantic_ev = None
    if args.semantic:
        print("Carregando encoder para similaridade semantica...")
        semantic_ev = SemanticEvaluator()

    metricas = list(ROUGE_METRICAS) + ([SEMANTICA] if args.semantic else [])

    por_modelo = {}
    for modelo, dados in resultados.items():
        resumos = dados["summaries"]
        if len(resumos) != len(referencias):
            print(f"[aviso] {modelo}: {len(resumos)} resumos x {len(referencias)} "
                  f"referencias — pulando")
            continue
        por_modelo[modelo] = scores_por_amostra(referencias, resumos, semantic_ev)

    print(f"{'Modelo':<14} {'Metrica':<9} {'Media':>7} {'DP':>7}   IC 95%")
    print("-" * 60)
    analise = {}
    for modelo, scores in por_modelo.items():
        analise[modelo] = {}
        for m in metricas:
            v = scores[m]
            lo, hi = ic_bootstrap(v)
            analise[modelo][m] = {
                "media": float(v.mean()), "desvio": float(v.std(ddof=1)),
                "ic95": [lo, hi],
            }
            print(f"{modelo:<14} {m:<9} {v.mean():>7.4f} {v.std(ddof=1):>7.4f}   "
                  f"[{lo:.4f}, {hi:.4f}]")
        print()

    if args.comparar:
        a, b = args.comparar
        if a in por_modelo and b in por_modelo:
            print(f"Diferenca pareada ({a} - {b}), IC 95% bootstrap:")
            print("-" * 60)
            analise["_diferenca"] = {"modelos": [a, b]}
            for m in metricas:
                d, lo, hi = ic_bootstrap_diferenca(por_modelo[a][m], por_modelo[b][m])
                cruza_zero = lo <= 0 <= hi
                marca = "inclui zero" if cruza_zero else "NAO inclui zero"
                analise["_diferenca"][m] = {
                    "diferenca_media": d, "ic95": [lo, hi], "inclui_zero": cruza_zero,
                }
                print(f"  {m:<9} {d:>+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {marca}")
            print("\nIC que inclui zero => a diferenca nao se distingue do ruido "
                  "amostral\n(estatistica descritiva; nao e teste de hipotese).")
        else:
            print(f"[aviso] modelos {a}/{b} nao encontrados nos resultados")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(analise, f, ensure_ascii=False, indent=2)
        print(f"\nAnalise salva em: {args.out}")


if __name__ == "__main__":
    main()
