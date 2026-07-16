# Metodologia (rascunho)

> Rascunho gerado a partir do que o código efetivamente executa. Os números
> vieram de **uma execução real** (seed 42, CPU, Python 3.14). Reproduza com os
> comandos da seção "Reprodução". Nenhum valor foi estimado.

## 1. Corpus de avaliação

A avaliação usa um corpus de **10 textos científicos em português** (`data/corpus.py`,
`BUILTIN_SAMPLES`), cada um acompanhado de um **resumo de referência**. Os textos
cobrem temas de computação/IA (PLN, sumarização, aprendizado de máquina, RAG,
ética em IA etc.), com ~150–200 palavras cada.

> **Limitação (registrada):** o corpus é sintético — textos e resumos de
> referência foram redigidos pelo próprio autor. Serve para validar o pipeline,
> mas não permite generalização; uma rodada final deve usar artigos reais.

## 2. Pré-processamento

`summarization/preprocessing.py` (`TextPreprocessor`) aplica, em ordem:

1. Normalização Unicode **NFKC** e colapso de espaços múltiplos.
2. Remoção de referências bibliográficas: citações no padrão `[n]`/`[n,m]` e
   seções iniciadas por "Referências Bibliográficas"/"References".
3. Truncamento por número de palavras (`max_length`).

Nos modelos gerativos, a entrada é truncada em `max_input_length = 512` tokens.

## 3. Modelos comparados

| Modelo | Identificador HF | Abordagem |
|---|---|---|
| GPT-2 | `gpt2` | Gerativa (abstrativa) |
| DistilGPT-2 | `distilgpt2` | Gerativa (abstrativa) |
| BERTimbau | `neuralmind/bert-base-portuguese-cased` | Extrativa |

**Gerativos (GPT-2, DistilGPT-2).** Prompt em português:

```
Texto científico:
{texto pré-processado}

Resumo do texto acima:
```

Geração **determinística (greedy)**: `do_sample=False`, `num_beams=1`,
`repetition_penalty=1.2`, `max_new_tokens=150`. Apenas os tokens gerados após o
prompt compõem o resumo.

**Extrativo (BERTimbau).** `summarization/extractive_summarizer.py`: o texto é
segmentado em sentenças; para cada sentença calcula-se o embedding do token `[CLS]`
e a similaridade de cosseno com o embedding do documento; selecionam-se as
**3 sentenças** de maior similaridade, preservando a ordem original.

> **LLaMA-2** está configurado em `summarization/model_loader.py`, mas ficou
> **fora desta rodada** (acesso gated no Hugging Face + quantização 4-bit que
> exige GPU/CUDA). Registrado como trabalho futuro.

## 4. Reprodutibilidade

`utils.set_seed(42)` fixa as seeds de `random`, `numpy` e `torch` no início da
comparação, e a decodificação gerativa é determinística (greedy). Com isso, cada
número da tabela é 100% reproduzível.

## 5. Métricas

- **ROUGE-1, ROUGE-2, ROUGE-L** (`rouge-score`, sem stemming), reportando
  F-measure médio sobre o corpus (`evaluation/metrics.py`, `SummaryEvaluator`).
- **Similaridade semântica**: embeddings *mean-pooled* do BERTimbau para
  referência e hipótese, com similaridade de cosseno; reporta-se a média
  (`SemanticEvaluator`).

## 6. Resultados (execução real — seed 42, CPU)

| Modelo | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | Sim. semântica | Tempo (s) |
|---|---|---|---|---|---|
| GPT-2 | 0.0212 | 0.0013 | 0.0212 | 0.4918 | 44.2 |
| DistilGPT-2 | 0.0209 | 0.0000 | 0.0209 | 0.4814 | 19.1 |
| BERTimbau | 0.3429 | 0.1318 | 0.2459 | 0.8988 | 2.8 |

Figuras correspondentes em `experiments/results/` (`fig_rouge.png`,
`fig_semantica.png`, `fig_tempo.png`), geradas por `experiments/gerar_figuras.py`.

## 7. Discussão e limitações

- **Assimetria extrativo × gerativo.** O BERTimbau, por ser extrativo, copia
  sentenças do texto-fonte; como os resumos de referência são próximos desse
  texto, o ROUGE (baseado em sobreposição de n-gramas) o favorece fortemente.
  GPT-2/DistilGPT-2 são pré-treinados majoritariamente em inglês e sem ajuste
  para a tarefa em português, o que explica o ROUGE próximo de zero.
- **ROUGE mede sobreposição léxica**, não qualidade semântica; por isso a métrica
  semântica complementar. Ainda assim, ROUGE aqui não usa stemming para PT.
- **Viés potencial na métrica semântica.** O `SemanticEvaluator` usa o mesmo
  encoder (BERTimbau) que gera os resumos extrativos, o que pode favorecer o
  BERTimbau nessa métrica específica.
- **Corpus pequeno e sintético** (ver §1).
- **Ambiente CPU**; LLaMA-2 fora desta rodada.

## Reprodução

```bash
python -m experiments.compare_models \
    --models gpt2 distilgpt2 bertimbau --semantic --seed 42 \
    --output experiments/results/comparacao.json

python -m experiments.gerar_figuras \
    --input experiments/results/comparacao.json \
    --outdir experiments/results
```
