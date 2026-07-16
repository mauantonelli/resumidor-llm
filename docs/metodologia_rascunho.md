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
| PTT5-summ | `recogna-nlp/ptt5-base-summ` | Gerativa (abstrativa, PT) — trunca a entrada |
| PTT5-summ-chunk | `recogna-nlp/ptt5-base-summ` | Igual ao anterior, mas com chunking (lê o artigo inteiro) |

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

**Chunking hierárquico (PTT5-summ-chunk).** `summarization/chunked_summarizer.py`.
Motivação: há uma **assimetria de cobertura** entre os modelos. O BERTimbau
extrativo percorre *todas* as sentenças do artigo, enquanto o PTT5-summ e os
modelos causais truncam a entrada em 512 tokens — ou seja, leem apenas o começo
de artigos com mediana de ~4.815 palavras. O chunking remove essa desvantagem:

1. *map*: divide o texto em janelas de 512 tokens (sobreposição de 50) e resume
   cada uma em ~100 tokens;
2. *reduce*: concatena os resumos parciais e resume de novo; se a concatenação
   ainda exceder a janela, repete (até 3 níveis).

Entra como **modelo separado**, ao lado do PTT5-summ, para que "truncar ×
chunking" seja uma hipótese testável isolando uma variável — o mesmo `model_id`,
a mesma referência, o mesmo corpus.

> **Diferença de decodificação (registrada):** o resumo **final** usa beam search
> (igual ao PTT5-summ), mas os resumos **intermediários** de cada pedaço usam
> greedy. Com beam em todos os pedaços, o custo medido foi de ~10 min por artigo
> (~5 h no corpus), inviável em CPU; com greedy no *map* caiu para ~24 s por
> artigo. Os resumos de pedaço são artefatos internos, não a saída avaliada.

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
| GPT-2 | 0.0370 | 0.0000 | 0.0277 | 0.3896 | 89.6 |
| DistilGPT-2 | 0.0380 | 0.0002 | 0.0271 | 0.3936 | 54.7 |
| BERTimbau | 0.1978 | 0.0451 | 0.1173 | 0.7727 | 156.1 |
| PTT5-summ | 0.1981 | 0.0373 | 0.1304 | 0.8030 | 184.1 |
| PTT5-summ-chunk | 0.2181 | 0.0396 | 0.1340 | 0.8305 | 1029.5 |

> **Não destacamos um "vencedor"**: as diferenças entre BERTimbau, PTT5-summ e
> PTT5-summ-chunk estão dentro do ruído amostral em todas as métricas — ver
> §6.1.1. Os tempos variam entre execuções (mesma máquina, CPU compartilhada);
> as métricas são determinísticas e reproduzem exatamente.

Figuras em `experiments/results/scielo5/`.

### 6.1.1. Dispersão e incerteza (corpus real, n = 30)

Média ± desvio-padrão e IC 95% (bootstrap, 10.000 reamostragens, seed 42),
recalculados por amostra a partir dos resumos salvos
(`experiments/analise_estatistica.py`):

| Modelo | ROUGE-1 (IC 95%) | ROUGE-L (IC 95%) | Semântica (IC 95%) |
|---|---|---|---|
| GPT-2 | 0.0370 ± 0.0232 [0.0292, 0.0452] | 0.0277 [0.0220, 0.0338] | 0.3896 [0.3548, 0.4192] |
| DistilGPT-2 | 0.0380 ± 0.0211 [0.0304, 0.0454] | 0.0271 [0.0220, 0.0322] | 0.3936 [0.3575, 0.4262] |
| BERTimbau | 0.1978 ± 0.1251 [0.1545, 0.2422] | 0.1173 [0.0971, 0.1385] | 0.7727 [0.7217, 0.8216] |
| PTT5-summ | 0.1981 ± 0.0895 [0.1675, 0.2310] | 0.1304 [0.1132, 0.1485] | 0.8030 [0.7713, 0.8336] |
| PTT5-summ-chunk | 0.2181 ± 0.0815 [0.1893, 0.2472] | 0.1340 [0.1183, 0.1500] | 0.8305 [0.8076, 0.8507] |

**Diferença pareada PTT5-summ − BERTimbau** (IC 95% bootstrap):

| Métrica | Diferença | IC 95% | Conclusão |
|---|---|---|---|
| ROUGE-1 | +0.0003 | [−0.0395, +0.0417] | inclui zero |
| ROUGE-2 | −0.0078 | [−0.0288, +0.0112] | inclui zero |
| ROUGE-L | +0.0131 | [−0.0066, +0.0333] | inclui zero |
| Semântica | +0.0303 | [−0.0181, +0.0820] | inclui zero |

**Diferença pareada PTT5-summ-chunk − PTT5-summ** (chunking × truncamento):

| Métrica | Diferença | IC 95% | Conclusão |
|---|---|---|---|
| ROUGE-1 | +0.0200 | [−0.0112, +0.0518] | inclui zero |
| ROUGE-2 | +0.0023 | [−0.0080, +0.0128] | inclui zero |
| ROUGE-L | +0.0036 | [−0.0134, +0.0216] | inclui zero |
| Semântica | +0.0275 | [−0.0019, +0.0580] | inclui zero |

**Todas as diferenças incluem zero**: com n = 30, BERTimbau, PTT5-summ e
PTT5-summ-chunk são **indistinguíveis** entre si em todas as métricas.
(Estatística descritiva — reporta incerteza; não é teste de hipótese.)

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

- **Modelos em inglês não servem para PT-BR** — o achado mais robusto. GPT-2 e
  DistilGPT-2 ficam com ROUGE ≈ 0 e similaridade semântica ~0.39 nos dois
  corpora, gerando texto incoerente. Aqui a diferença **não** é ruído: os IC 95%
  de ROUGE-1 (~[0.029, 0.045]) não chegam perto dos de BERTimbau/PTT5-summ
  (~[0.155, 0.242]). O problema não é a abordagem gerativa em si, mas usar
  modelos pré-treinados em inglês, sem ajuste para a tarefa em português.

- **O corpus sintético inflou os resultados.** Comparando §6.1 e §6.2, todos os
  scores caem fortemente ao sair do corpus sintético para artigos reais — o
  PTT5-summ vai de ROUGE-1 0.391 para 0.198, e o BERTimbau de 0.343 para 0.198.
  Textos científicos reais são substancialmente mais difíceis, e um corpus
  redigido pelo próprio autor favorece artificialmente os modelos. **Este é o
  principal argumento a favor de avaliar em dados reais.**

- **No corpus real, extrativo e abstrativo são indistinguíveis — a vantagem do
  PTT5-summ não se sustenta.** No sintético, o PTT5-summ dominava todas as
  métricas ROUGE. Nos artigos reais, a análise de incerteza (§6.1.1) mostra que
  o IC 95% da diferença pareada **inclui zero nas quatro métricas** (ROUGE-1/2/L
  e semântica). Nenhum dos dois pode ser declarado superior ao outro com n = 30 —
  nem mesmo nas métricas em que as médias diferem (ROUGE-2 e ROUGE-L), cujas
  diferenças estão dentro do ruído amostral.

- **O PTT5-summ é mais consistente.** Embora as médias empatem, o desvio-padrão
  do PTT5-summ é sensivelmente menor que o do BERTimbau (ROUGE-1: 0.0895 vs
  0.1251; semântica: 0.0879 vs 0.1420): o sumarizador abstrativo varia menos
  entre artigos. Observação descritiva — a diferença de dispersão em si não foi
  testada.

- **Truncamento da entrada e a assimetria de cobertura.** Os modelos causais e o
  PTT5-summ truncam a entrada em 512 tokens, mas os artigos reais têm mediana de
  ~4.815 palavras: eles leem apenas o começo do artigo (aproximadamente a
  introdução), enquanto o resumo de referência sintetiza o **artigo inteiro**. O
  BERTimbau extrativo, por percorrer todas as sentenças, **não** sofre essa
  restrição — o que torna a comparação assimétrica em cobertura de texto.

- **O chunking melhora de forma consistente, mas a melhora não é confirmável e
  custa caro.** O PTT5-summ-chunk, que lê o artigo inteiro, ficou acima do
  PTT5-summ (truncado) nas **quatro** métricas — ROUGE-1 +0.0200, ROUGE-2
  +0.0023, ROUGE-L +0.0036 e semântica +0.0275. A consistência da direção é
  sugestiva, e a semântica quase exclui zero ([−0.0019, +0.0580]). Ainda assim,
  **os IC 95% das quatro diferenças incluem zero**: com n = 30 não é possível
  afirmar que o chunking supera o truncamento. E o custo é alto: **1029.5 s
  contra 184.1 s (≈5,6×)**. Ou seja, remover a desvantagem de cobertura **não**
  produziu o salto que a hipótese previa — as explicações candidatas são (a) o
  efeito ser pequeno, (b) n = 30 não ter poder para detectá-lo, e (c) o *reduce*
  do map-reduce sintetizar mal (nas inspeções manuais, o resumo final tendia a
  refletir o começo do artigo em vez de integrar o todo). Distinguir entre essas
  hipóteses exige mais amostras.

- **Viés potencial na métrica semântica.** O `SemanticEvaluator` usa o mesmo
  encoder (BERTimbau) que gera os resumos extrativos, o que poderia favorecer o
  BERTimbau. No corpus real a média do PTT5-summ é maior (0.8030 vs 0.7727), mas
  a diferença inclui zero no IC 95% (§6.1.1), então **não** se pode concluir
  superioridade — nem medir o efeito desse viés com os dados atuais.

- **ROUGE mede sobreposição léxica**, não qualidade semântica, e aqui não usa
  stemming para PT; por isso a métrica semântica complementar.

- **Custo.** BERTimbau (156 s) e PTT5-summ (184 s) têm custo parecido; o
  PTT5-summ-chunk custa ≈5,6× o PTT5-summ. Os modelos em inglês são os mais
  rápidos, mas inúteis para a tarefa.

- **Incerteza reportada, mas sem teste de hipótese.** A §6.1.1 traz IC 95% por
  bootstrap (descritivo). Não foi executado teste de hipótese formal, e n = 30 é
  pequeno: o corpus atual **não tem poder** para separar BERTimbau, PTT5-summ e
  PTT5-summ-chunk. Aumentar n é o caminho para decidir esses empates — é a
  limitação que mais restringe as conclusões deste trabalho.

- **Ambiente CPU**; LLaMA-2 fora desta rodada.

## Reprodução

**Corpus científico real (SciELO) — resultado principal (§6.1):**

```bash
# 1. coletar os 30 artigos reais
python -m data.coletar_scielo --n 30 --out data/processed/corpus_scielo.json

# 2. rodar a comparação (--checkpoint-dir permite retomar se for interrompida)
python -m experiments.compare_models \
    --models gpt2 distilgpt2 bertimbau ptt5-summ ptt5-summ-chunk \
    --semantic --seed 42 \
    --corpus data/processed/corpus_scielo.json \
    --checkpoint-dir experiments/results/.ckpt \
    --output experiments/results/comparacao_scielo5.json

# 3. incerteza (IC 95%) e diferença pareada chunking × truncamento
python -m experiments.analise_estatistica \
    --input experiments/results/comparacao_scielo5.json \
    --corpus data/processed/corpus_scielo.json \
    --comparar ptt5-summ-chunk ptt5-summ --semantic \
    --out experiments/results/scielo5/analise_estatistica.json

# 4. tabela e figuras (com barras de erro do IC 95%)
python -m experiments.gerar_figuras \
    --input experiments/results/comparacao_scielo5.json \
    --analise experiments/results/scielo5/analise_estatistica.json \
    --outdir experiments/results/scielo5
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
