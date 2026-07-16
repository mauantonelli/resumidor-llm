# Resumidor Automático de Textos Científicos em Português

Iniciação Científica (PIVICT 2025) — IFSP, Campus São Carlos.
**Aluno:** Maurício Antonelli de Oliveira · **Orientador:** Prof. Rodrigo Elias Bianchi.

Sistema de geração automática de resumos de textos científicos em PT-BR usando
modelos de linguagem open-source, com avaliação comparativa entre eles e um
assistente institucional baseado em RAG.

O projeto tem três frentes:

1. **Sumarização** — GPT-2 e DistilGPT-2 (gerativos), BERTimbau (extrativo, por
   similaridade de embeddings) e PTT5-summ (abstrativo ajustado para PT).
2. **Avaliação comparativa** — ROUGE (ROUGE-1/2/L) e similaridade semântica,
   comparando os modelos sobre um corpus com resumos de referência.
3. **Assistente RAG** — perguntas e respostas sobre documentos institucionais do
   IFSP (edital, guia do aluno, normas de TCC, política de laboratórios,
   regulamento de IC), com LangChain + FAISS.

---

## Estado atual (jul/2026)

- ✅ Ambiente reproduzível com dependências travadas (Python 3.14).
- ✅ Módulos de sumarização e avaliação importam e passam nos testes.
- ✅ Assistente RAG corrigido para a API 1.x do LangChain.
- ✅ Suíte de testes: **39 passando**.
- ✅ **Avaliação comparativa executada** (GPT-2, DistilGPT-2, BERTimbau, PTT5-summ;
  seed 42, CPU). Resultados, tabela e figuras em `experiments/results/` (gerados
  localmente, fora do versionamento). Metodologia e números em
  `docs/metodologia_rascunho.md`. Resumo: **PTT5-summ** obteve o melhor ROUGE;
  GPT-2/DistilGPT-2 (pré-treino em inglês) tiveram desempenho fraco em PT-BR.
- ℹ️ O notebook `notebooks/analise_comparativa.ipynb` contém o código de análise,
  mas ainda sem saídas salvas (o caminho reprodutível é via `experiments/`).

> **Reprodutibilidade:** a geração dos sumarizadores gerativos usa amostragem
> (`do_sample=True`), portanto **ainda não é determinística** — a política de
> seeds/decodificação para a avaliação está em definição.

---

## Requisitos

- **Python 3.14** (versão usada no desenvolvimento; ver `requirements-lock.txt`).
- Os modelos (GPT-2, DistilGPT-2, BERTimbau) são baixados do Hugging Face na
  primeira execução — é necessário acesso à internet.
- LLaMA-2 está configurado em `summarization/model_loader.py`, mas exige acesso
  gated no Hugging Face e GPU (quantização 4-bit só funciona em CUDA); não é
  utilizado por padrão.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# ambiente reproduzível (versões travadas)
pip install -r requirements.txt

# para rodar os testes
pip install -r requirements-dev.txt

# alternativa: reprodução bit-a-bit de toda a árvore de dependências
# pip install -r requirements-lock.txt
```

## Uso

### Sumarização (CLI)

```bash
python main.py                          # usa texto de exemplo embutido
python main.py --model distilgpt2       # escolhe o modelo
python main.py --file artigo.txt        # resume um arquivo
python main.py --model bertimbau        # sumarização extrativa
python main.py --list-models            # lista modelos disponíveis
```

### Comparação entre modelos

```bash
python -m experiments.compare_models \
    --models gpt2 distilgpt2 bertimbau ptt5-summ \
    --semantic --seed 42 \
    --output experiments/results/comparacao.json

# gerar tabela e figuras a partir do JSON
python -m experiments.gerar_figuras \
    --input experiments/results/comparacao.json --outdir experiments/results
```

Roda cada modelo sobre o corpus de avaliação, calcula ROUGE (e, com `--semantic`,
similaridade semântica) e imprime uma tabela comparativa. `resultados.json` fica
fora do versionamento (ver `.gitignore`).

### Assistente RAG (IFSP)

```bash
# 1. indexar os documentos institucionais (gera o índice FAISS)
python -m rag_assistant.cli index

# 2. perguntar
python -m rag_assistant.cli query "Qual o prazo para entrega do TCC?"

# 3. modo interativo
python -m rag_assistant.cli chat
```

O índice vetorial é gravado em `data/vector_store/` (fora do versionamento).

### Notebook de análise comparativa

```bash
jupyter notebook notebooks/analise_comparativa.ipynb
```

## Testes

```bash
pytest tests/
```

## Estrutura

```
resumidor-llm/
├── data/
│   ├── corpus.py              # corpus de avaliação (10 amostras + resumos de referência)
│   └── raw/                   # documentos institucionais do IFSP (.txt) para o RAG
├── summarization/            # pré-processamento, carregamento de modelos, sumarizadores
├── evaluation/               # métricas ROUGE e similaridade semântica
├── experiments/             # harness de comparação entre modelos
├── rag_assistant/           # pipeline RAG (loader, vector store, retriever, gerador, CLI)
├── utils/                    # utilitários (timer, logging)
├── notebooks/               # análise comparativa
├── tests/                    # testes unitários (pytest)
├── main.py                   # CLI de sumarização
├── requirements.txt          # deps de runtime (travadas)
├── requirements-dev.txt      # deps de teste
└── requirements-lock.txt     # freeze completo (reprodução bit-a-bit)
```

## Notas

- Não são versionados: datasets/PDFs (`data/raw/*.pdf`), dados processados
  (`data/processed/*.json`), índice FAISS (`data/vector_store/`), checkpoints de
  modelo (`*.pt/*.bin/*.safetensors`) e saídas de notebook.
