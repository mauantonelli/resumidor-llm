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
  `ModelLoader` (dict `SUPPORTED_MODELS`) → `Summarizer` (gerativo) e
  `ExtractiveSummarizer` (BERTimbau, similaridade de embeddings).
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
  corpus, dependências travadas, matplotlib adicionado. Suíte: **38 passando**.
- **A avaliação comparativa nunca foi executada de ponta a ponta.** Não há
  nenhum número gerado no repositório. Executar isso é o próximo grande passo —
  e depende das decisões metodológicas abaixo.

### Decisões metodológicas em aberto (perguntar antes de executar o experimento)

1. **Corpus**: manter as 10 amostras sintéticas ou trocar por artigos
   científicos reais em PT-BR com resumos de referência?
2. **LLaMA-2**: fora por padrão (gated + exige GPU/CUDA). Incluir depende de
   acesso/hardware.
3. **Determinismo**: não há seeds; o gerativo usa `do_sample=True, temperature=0.7`
   → não reproduzível. Definir seed e/ou decodificação determinística
   (greedy/beam) para a avaliação.
4. **Comparação extrativo × gerativo**: BERTimbau (extrativo) vs GPT-2/DistilGPT-2
   (gerativo) medidos por ROUGE contra referências abstrativas — assimetria a
   registrar como limitação ou repensar o baseline.

## Convenções de código

- Português nos identificadores de domínio, docstrings e mensagens ao usuário.
- Sem dependências novas sem necessidade; se adicionar, travar versão em
  `requirements.txt` e atualizar o lock.
