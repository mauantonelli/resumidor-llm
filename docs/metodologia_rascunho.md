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
| PTT5-summ | `recogna-nlp/ptt5-base-summ` | Gerativa (abstrativa, PT) |

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

**Seq2seq abstrativo (PTT5-summ).** `summarization/seq2seq_summarizer.py`: modelo
encoder-decoder (PTT5) ajustado para sumarização em português. O texto
pré-processado é a entrada do encoder e a saída do decoder é o resumo (não há
prompt a remover). Decodificação determinística por **beam search**
(`num_beams=4`, `do_sample=False`, `repetition_penalty=1.2`).

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
| GPT-2 | 0.0212 | 0.0013 | 0.0212 | 0.4918 | 39.9 |
| DistilGPT-2 | 0.0209 | 0.0000 | 0.0209 | 0.4814 | 11.7 |
| BERTimbau | 0.3429 | 0.1318 | 0.2459 | **0.8988** | 3.3 |
| PTT5-summ | **0.3913** | **0.2250** | **0.3270** | 0.8943 | 40.2 |

(Melhor valor de cada coluna em negrito; tempo é a soma de carregamento + geração
sobre 10 amostras em CPU.)

Figuras correspondentes em `experiments/results/` (`fig_rouge.png`,
`fig_semantica.png`, `fig_tempo.png`), geradas por `experiments/gerar_figuras.py`.

## 7. Discussão e limitações

- **O modelo em português importa mais que o paradigma.** O PTT5-summ (abstrativo,
  ajustado para PT) obteve o melhor ROUGE em todas as variantes (R-1 0.391, R-2
  0.225, R-L 0.327), superando inclusive o BERTimbau extrativo. Isso mostra que o
  fraco desempenho de GPT-2/DistilGPT-2 não é da abordagem gerativa em si, mas de
  usar modelos pré-treinados em inglês e sem ajuste para a tarefa em português
  (ROUGE ≈ 0, geração incoerente).
- **Extrativo × abstrativo em similaridade semântica.** BERTimbau (0.899) e
  PTT5-summ (0.894) ficam praticamente empatados na métrica semântica, apesar da
  grande diferença em ROUGE — indício de que o resumo abstrativo preserva o
  sentido mesmo reformulando as palavras.
- **Viés potencial na métrica semântica.** O `SemanticEvaluator` usa o mesmo
  encoder (BERTimbau) que gera os resumos extrativos, o que pode favorecer o
  BERTimbau nessa métrica específica — a leitura do quase-empate deve considerar
  isso.
- **ROUGE mede sobreposição léxica**, não qualidade semântica, e aqui não usa
  stemming para PT; por isso a métrica semântica complementar.
- **Custo.** O melhor modelo (PTT5-summ) é também o mais lento em CPU (~40 s no
  total); BERTimbau é o mais rápido (~3 s). Há um trade-off qualidade × tempo.
- **Corpus pequeno e sintético** (ver §1) — principal limitação para
  generalização.
- **Ambiente CPU**; LLaMA-2 fora desta rodada.

## Reprodução

```bash
python -m experiments.compare_models \
    --models gpt2 distilgpt2 bertimbau ptt5-summ --semantic --seed 42 \
    --output experiments/results/comparacao.json

python -m experiments.gerar_figuras \
    --input experiments/results/comparacao.json \
    --outdir experiments/results
```
