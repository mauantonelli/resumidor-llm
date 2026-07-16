# CLAUDE.md

Orientações para agentes (Claude Code) trabalhando neste repositório.

## O que é

Iniciação Científica do IFSP São Carlos: resumidor automático de textos
científicos em PT-BR com LLMs open-source. Três frentes — sumarização
(GPT-2/DistilGPT-2 gerativos, BERTimbau extrativo), avaliação comparativa
(ROUGE + similaridade semântica) e assistente institucional via RAG.
Ver `README.md` para visão geral e uso.

## Regras inegociáveis

- **Nunca invente, estime ou "exemplifique" resultado numérico.** Se um número
  (ROUGE, similaridade, tempo) não veio de execução real, escreva
  `NÃO EXECUTADO` e siga. Isto vale para README, notebook, relatórios e commits.
- **Toda decisão metodológica PARA e pergunta ao autor** — escolha de métrica,
  split, hiperparâmetro, baseline, dataset, política de decodificação. Não decida
  sozinho o que afeta a validade científica da IC.
- **Ao final de cada execução que gere números, forneça o comando exato** para o
  autor reproduzir cada número.
- **Commits** pequenos, mensagem descritiva, **em português** (formato
  `tipo(escopo): resumo`). Terminar mensagens de commit com a linha
  `Co-Authored-By`.
- **Não commitar**: datasets/PDFs, checkpoints (`*.pt/*.bin/*.safetensors`),
  índice FAISS, dados processados, nem saídas de notebook (já cobertos pelo
  `.gitignore`).
- Push só quando o autor pedir.

## Ambiente

- **Python 3.14.** Dependências travadas em `requirements.txt`; lock completo em
  `requirements-lock.txt`.
- As versões são as **major de 2026**: `transformers 5.x`, `langchain 1.x`,
  `langchain-community 0.4`, `torch 2.13`. O código foi adaptado a elas — não
  reverta para APIs antigas (langchain 0.1 provavelmente nem tem wheel para 3.14).
- Modelos são baixados do Hugging Face em runtime (precisa de internet).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/
```

## Arquitetura (rápido)

- `summarization/` — `TextPreprocessor` (limpeza/refs/truncamento) →
  `ModelLoader` (dict `SUPPORTED_MODELS`, tipos `causal`/`extractive`/`seq2seq`/
  `seq2seq_chunk`) → `Summarizer` (causal, GPT-2/DistilGPT-2),
  `ExtractiveSummarizer` (BERTimbau, similaridade de embeddings),
  `Seq2SeqSummarizer` (PTT5-summ, abstrativo PT) e `ChunkedSeq2SeqSummarizer`
  (PTT5-summ com chunking map-reduce).
- `data/coletar_scielo.py` — coletor do corpus científico real (SciELO):
  corpo do artigo → abstract do autor, com limpeza anti-vazamento.
- `experiments/analise_estatistica.py` — IC 95% (bootstrap) por amostra e da
  diferença pareada. **Use antes de afirmar que um modelo é melhor que outro.**

### Cuidados aprendidos (não repetir)

- **Cobertura assimétrica**: `ExtractiveSummarizer` lê o artigo inteiro; os
  demais truncam em 512 tokens. Ao comparar, considere esse confundidor.
- **Rodadas longas morrem**: use sempre `--checkpoint-dir` (checkpoint por
  artigo, retomada automática) e `nohup` para runs de horas. O chunking custa
  ~6 min/artigo em CPU (até 23 min nos artigos de ~10k palavras).
- **Nunca afirme "modelo X vence"** sem olhar o IC da diferença — três
  afirmações desse tipo já foram desmentidas pelo bootstrap.
- `argparse` no Python 3.14 rejeita `%` em `help=` (use `%%`).
- `evaluation/metrics.py` — `SummaryEvaluator` (ROUGE) e `SemanticEvaluator`
  (similaridade por embeddings BERTimbau).
- `experiments/compare_models.py` — harness CLI que roda os modelos sobre o
  corpus e monta a tabela comparativa.
- `data/corpus.py` — corpus de avaliação: 10 amostras `BUILTIN_SAMPLES`
  (texto + `reference_summary`), hoje **sintéticas** (escritas pelo autor).
- `rag_assistant/` — `DocumentLoader` → `VectorStore` (FAISS + embeddings
  BERTimbau) → `Retriever` → `RAGGenerator` (GPT-2), orquestrados por
  `RAGPipeline`; CLI em `rag_assistant/cli.py` (`index`/`query`/`chat`).

## Estado e pendências (jul/2026)

- Reparos concluídos: imports do RAG para LangChain 1.x, bug de cópia rasa no
  corpus, dependências travadas, matplotlib, overflow de posições do GPT-2 no RAG.
- Reprodutibilidade: `utils.set_seed(42)` + decodificação determinística.
- **Avaliação comparativa executada** (seed 42, CPU): GPT-2, DistilGPT-2,
  BERTimbau e PTT5-summ. PTT5-summ lidera em ROUGE; GPT-2/DistilGPT-2 (inglês)
  vão mal em PT. Números e figuras em `experiments/results/` (gitignored);
  metodologia em `docs/metodologia_rascunho.md`. Suíte: **39 passando**.

### Decisões metodológicas já tomadas pelo autor (07/2026)

1. **Corpus**: rodar com as 10 amostras sintéticas por ora (limitação registrada).
2. **LLaMA-2**: fora (gated + GPU); trabalho futuro.
3. **Determinismo**: seed 42 + greedy (causal) / beam (seq2seq).
4. **Extrativo × gerativo**: comparação mantida, assimetria registrada.
5. **Modelo PT-BR**: adicionado PTT5-summ (`recogna-nlp/ptt5-base-summ`).

### Ainda em aberto (perguntar antes)

- Trocar o corpus sintético por artigos reais PT-BR (rodada final).
- **RAG**: gerador trocado para GPorTuguese-2 (`pierreguillou/gpt2-small-portuguese`)
  → português coerente, mas é LM base (não instruction-tuned): **não fundamenta a
  resposta no contexto recuperado (alucina)**. A recuperação já retorna as fontes
  certas. Grounding real exige um modelo PT ajustado a instruções (decisão em
  aberto, provavelmente maior/gated).
- Rodar/salvar o notebook com as saídas reais.

## Convenções de código

- Português nos identificadores de domínio, docstrings e mensagens ao usuário.
- Sem dependências novas sem necessidade; se adicionar, travar versão em
  `requirements.txt` e atualizar o lock.
