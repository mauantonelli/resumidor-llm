# Metodologia (rascunho)

> Rascunho gerado a partir do que o código efetivamente executa. Os números
> vieram de **uma execução real** (seed 42, CPU, Python 3.14). Reproduza com os
> comandos da seção "Reprodução". Nenhum valor foi estimado.

## 1. Corpora de avaliação

Foram usados **dois** corpora, em ordem crescente de realismo.

### 1.1. Corpus sintético (piloto)

10 textos em português (`data/corpus.py`, `BUILTIN_SAMPLES`), cada um com um
resumo de referência, sobre temas de computação/IA, com ~150–200 palavras cada.

> **Limitação:** textos e referências foram redigidos pelo próprio autor. Serve
> para validar o pipeline, **não** para generalizar.

### 1.2. Corpus científico real — SciELO (principal)

30 artigos científicos **reais** em PT-BR, coletados do SciELO Brasil por
`data/coletar_scielo.py`:

- **Entrada** = corpo do artigo (`<body>` do JATS XML).
- **Referência** = resumo (abstract) escrito pelos próprios autores
  (`<abstract>` do JATS).
- **Diversidade**: 6 artigos de cada um de 5 periódicos — Revista de Saúde
  Pública, Cadernos de Saúde Pública, Educação & Sociedade, Interface
  (Comunicação, Saúde, Educação) e Estudos Avançados — cobrindo saúde, educação
  e temas interdisciplinares.
- **Dimensões**: corpo com mediana de ~4.815 palavras; abstract com mediana de
  ~144 palavras.
- **Controle de vazamento**: artigos retro-digitalizados replicam o resumo
  dentro do `<body>`. O coletor remove do corpo os parágrafos que replicam o
  abstract, o título e cabeçalhos/autores. Verificado: **0 vazamentos em 30**.

> **Ruído residual conhecido:** o início do corpo pode conter o cabeçalho
> bilíngue (título em inglês, autores, afiliação) de alguns artigos.

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

### 6.1. Corpus científico real — SciELO, 30 artigos (resultado principal)

| Modelo | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | Sim. semântica | Tempo (s) |
|---|---|---|---|---|---|
| GPT-2 | 0.0370 | 0.0000 | 0.0277 | 0.3896 | 104.7 |
| DistilGPT-2 | 0.0380 | 0.0002 | 0.0271 | 0.3936 | 58.0 |
| BERTimbau | 0.1978 | **0.0451** | 0.1173 | 0.7727 | 150.3 |
| PTT5-summ | **0.1981** | 0.0373 | **0.1304** | **0.8030** | 173.2 |

Figuras em `experiments/results/scielo/`.

### 6.2. Corpus sintético — 10 textos (piloto)

| Modelo | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | Sim. semântica | Tempo (s) |
|---|---|---|---|---|---|
| GPT-2 | 0.0212 | 0.0013 | 0.0212 | 0.4918 | 39.9 |
| DistilGPT-2 | 0.0209 | 0.0000 | 0.0209 | 0.4814 | 11.7 |
| BERTimbau | 0.3429 | 0.1318 | 0.2459 | **0.8988** | 3.3 |
| PTT5-summ | **0.3913** | **0.2250** | **0.3270** | 0.8943 | 40.2 |

Figuras em `experiments/results/`.

(Melhor valor de cada coluna em negrito; tempo = carregamento + geração sobre
todo o corpus, em CPU. Figuras geradas por `experiments/gerar_figuras.py`.)

## 7. Discussão e limitações

- **Modelos em inglês não servem para PT-BR.** GPT-2 e DistilGPT-2 ficam com
  ROUGE ≈ 0 e similaridade semântica ~0.39 nos dois corpora, gerando texto
  incoerente. O problema não é a abordagem gerativa em si, mas usar modelos
  pré-treinados em inglês, sem ajuste para a tarefa em português.

- **O corpus sintético inflou os resultados.** Comparando §6.1 e §6.2, todos os
  scores caem fortemente ao sair do corpus sintético para artigos reais — o
  PTT5-summ vai de ROUGE-1 0.391 para 0.198, e o BERTimbau de 0.343 para 0.198.
  Textos científicos reais são substancialmente mais difíceis, e um corpus
  redigido pelo próprio autor favorece artificialmente os modelos. **Este é o
  principal argumento a favor de avaliar em dados reais.**

- **No corpus real, extrativo e abstrativo empatam — a vantagem do PTT5-summ não
  se sustenta.** No sintético, o PTT5-summ dominava todas as métricas ROUGE. Nos
  artigos reais o quadro é misto e essencialmente um empate: ROUGE-1 praticamente
  idêntico (0.1981 vs 0.1978), o BERTimbau **vence** em ROUGE-2 (0.0451 vs
  0.0373), e o PTT5-summ vence em ROUGE-L (0.1304 vs 0.1173) e na similaridade
  semântica (0.8030 vs 0.7727). Com n = 30 e sem teste estatístico, **não há base
  para declarar um vencedor em ROUGE-1**.

- **Limitação estrutural: truncamento da entrada.** Os sumarizadores truncam a
  entrada em 512 tokens, mas os artigos reais têm mediana de ~4.815 palavras. Ou
  seja, os modelos leem apenas o começo do artigo (aproximadamente a introdução),
  enquanto o resumo de referência sintetiza o **artigo inteiro**. Isso limita
  estruturalmente o ROUGE alcançável e é a explicação mais provável para a queda
  em §6.1. Tratar textos longos (chunking, modelos de contexto longo) é o próximo
  passo natural.

- **Viés potencial na métrica semântica.** O `SemanticEvaluator` usa o mesmo
  encoder (BERTimbau) que gera os resumos extrativos, o que poderia favorecer o
  BERTimbau. Vale notar que, mesmo assim, o PTT5-summ **supera** o BERTimbau na
  métrica semântica no corpus real — o que reforça esse resultado específico.

- **ROUGE mede sobreposição léxica**, não qualidade semântica, e aqui não usa
  stemming para PT; por isso a métrica semântica complementar.

- **Custo.** No corpus real, BERTimbau (150 s) e PTT5-summ (173 s) têm custo
  parecido; os modelos em inglês são mais rápidos, mas inúteis para a tarefa.

- **Sem significância estatística.** n = 30, sem intervalo de confiança nem teste
  de hipótese — diferenças pequenas não devem ser interpretadas como superioridade.

- **Ambiente CPU**; LLaMA-2 fora desta rodada.

## Reprodução

**Corpus científico real (SciELO) — resultado principal (§6.1):**

```bash
# 1. coletar os 30 artigos reais
python -m data.coletar_scielo --n 30 --out data/processed/corpus_scielo.json

# 2. rodar a comparação
python -m experiments.compare_models \
    --models gpt2 distilgpt2 bertimbau ptt5-summ --semantic --seed 42 \
    --corpus data/processed/corpus_scielo.json \
    --output experiments/results/comparacao_scielo.json

# 3. tabela e figuras
python -m experiments.gerar_figuras \
    --input experiments/results/comparacao_scielo.json \
    --outdir experiments/results/scielo
```

**Corpus sintético — piloto (§6.2):**

```bash
python -m experiments.compare_models \
    --models gpt2 distilgpt2 bertimbau ptt5-summ --semantic --seed 42 \
    --output experiments/results/comparacao.json

python -m experiments.gerar_figuras \
    --input experiments/results/comparacao.json \
    --outdir experiments/results
```

> A coleta do SciELO depende de disponibilidade da API/site; o corpus coletado
> fica em `data/processed/` (fora do versionamento). Os números acima vieram da
> coleta de 30 artigos descrita em §1.2.
