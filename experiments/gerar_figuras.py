"""Gera a tabela comparativa e as figuras a partir do JSON produzido por
experiments/compare_models.py.

Uso:
    python -m experiments.gerar_figuras \
        --input experiments/results/comparacao.json \
        --outdir experiments/results

Nao inventa dados: le exclusivamente o JSON de uma execucao real.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta Okabe-Ito (segura para daltonismo), padrao em figuras cientificas.
COR_R1 = "#0072B2"  # azul
COR_R2 = "#E69F00"  # laranja
COR_RL = "#009E73"  # verde
COR_UNICA = "#0072B2"
COR_GRID = "#D9D9D9"

ROTULOS_MODELO = {
    "gpt2": "GPT-2",
    "distilgpt2": "DistilGPT-2",
    "bertimbau": "BERTimbau",
    "ptt5-summ": "PTT5-summ",
    "ptt5-summ-chunk": "PTT5-summ\n(chunking)",
}


def _estilo_eixo(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _rotular_barras(ax, barras, fmt="{:.3f}"):
    for b in barras:
        altura = b.get_height()
        ax.annotate(
            fmt.format(altura),
            xy=(b.get_x() + b.get_width() / 2, altura),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _yerr(analise, modelos, metrica, valores):
    """Constroi o yerr assimetrico a partir do IC 95% da analise estatistica."""
    if not analise:
        return None
    lo_err, hi_err = [], []
    for m, v in zip(modelos, valores):
        ic = (analise.get(m, {}).get(metrica, {}) or {}).get("ic95")
        if not ic:
            return None  # sem IC para algum modelo: nao desenha barra de erro
        lo_err.append(max(0.0, v - ic[0]))
        hi_err.append(max(0.0, ic[1] - v))
    return [lo_err, hi_err]


def figura_rouge(dados, modelos, outpath, analise=None):
    r1 = [dados[m]["rouge_scores"]["rouge1"]["fmeasure"] for m in modelos]
    r2 = [dados[m]["rouge_scores"]["rouge2"]["fmeasure"] for m in modelos]
    rl = [dados[m]["rouge_scores"]["rougeL"]["fmeasure"] for m in modelos]
    rotulos = [ROTULOS_MODELO.get(m, m) for m in modelos]

    x = range(len(modelos))
    largura = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ekw = dict(ecolor="#444444", capsize=3, error_kw={"linewidth": 1.0})
    b1 = ax.bar([i - largura for i in x], r1, largura, label="ROUGE-1", color=COR_R1,
                yerr=_yerr(analise, modelos, "rouge1", r1), **ekw)
    b2 = ax.bar(list(x), r2, largura, label="ROUGE-2", color=COR_R2,
                yerr=_yerr(analise, modelos, "rouge2", r2), **ekw)
    b3 = ax.bar([i + largura for i in x], rl, largura, label="ROUGE-L", color=COR_RL,
                yerr=_yerr(analise, modelos, "rougeL", rl), **ekw)

    if not analise:
        _rotular_barras(ax, b1)
        _rotular_barras(ax, b2)
        _rotular_barras(ax, b3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(rotulos)
    ax.set_ylabel("F-measure")
    titulo = "ROUGE por modelo (F-measure)"
    if analise:
        titulo += " — barras de erro: IC 95%"
    ax.set_title(titulo)
    ax.legend(frameon=False)
    _estilo_eixo(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def figura_barra_simples(dados, modelos, chave, titulo, ylabel, outpath, fmt="{:.3f}"):
    valores = [chave(dados[m]) for m in modelos]
    rotulos = [ROTULOS_MODELO.get(m, m) for m in modelos]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    barras = ax.bar(rotulos, valores, 0.5, color=COR_UNICA)
    _rotular_barras(ax, barras, fmt=fmt)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    _estilo_eixo(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def gerar_tabela_markdown(dados, modelos, tem_semantica):
    cab = "| Modelo | ROUGE-1 F | ROUGE-2 F | ROUGE-L F |"
    sep = "|---|---|---|---|"
    if tem_semantica:
        cab += " Sim. semântica |"
        sep += "---|"
    cab += " Tempo (s) |"
    sep += "---|"
    linhas = [cab, sep]
    for m in modelos:
        d = dados[m]
        rs = d["rouge_scores"]
        linha = (
            f"| {ROTULOS_MODELO.get(m, m)} "
            f"| {rs['rouge1']['fmeasure']:.4f} "
            f"| {rs['rouge2']['fmeasure']:.4f} "
            f"| {rs['rougeL']['fmeasure']:.4f} |"
        )
        if tem_semantica:
            linha += f" {d['semantic_scores']['semantic_similarity_mean']:.4f} |"
        linha += f" {d['time']:.1f} |"
        linhas.append(linha)
    return "\n".join(linhas)


def parse_args():
    p = argparse.ArgumentParser(description="Gera tabela e figuras da comparacao")
    p.add_argument("--input", default="experiments/results/comparacao.json")
    p.add_argument("--outdir", default="experiments/results")
    p.add_argument("--analise", default=None,
                   help="JSON de experiments.analise_estatistica; desenha IC 95%% "
                        "como barras de erro")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input, encoding="utf-8") as f:
        dados = json.load(f)

    modelos = list(dados.keys())
    tem_semantica = all("semantic_scores" in dados[m] for m in modelos)
    os.makedirs(args.outdir, exist_ok=True)

    analise = None
    if args.analise:
        with open(args.analise, encoding="utf-8") as f:
            analise = json.load(f)

    fig1 = figura_rouge(dados, modelos, os.path.join(args.outdir, "fig_rouge.png"),
                        analise=analise)
    fig3 = figura_barra_simples(
        dados, modelos, lambda d: d["time"],
        "Tempo total por modelo", "Tempo (s)",
        os.path.join(args.outdir, "fig_tempo.png"), fmt="{:.1f}",
    )
    gerados = [fig1, fig3]
    if tem_semantica:
        fig2 = figura_barra_simples(
            dados, modelos,
            lambda d: d["semantic_scores"]["semantic_similarity_mean"],
            "Similaridade semântica média por modelo", "Similaridade (cosseno)",
            os.path.join(args.outdir, "fig_semantica.png"),
        )
        gerados.append(fig2)

    tabela = gerar_tabela_markdown(dados, modelos, tem_semantica)
    tabela_path = os.path.join(args.outdir, "tabela_comparativa.md")
    with open(tabela_path, "w", encoding="utf-8") as f:
        f.write(tabela + "\n")

    print(tabela)
    print()
    print("Figuras geradas:")
    for g in gerados:
        print(f"  {g}")
    print(f"Tabela: {tabela_path}")


if __name__ == "__main__":
    main()
