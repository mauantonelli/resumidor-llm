import argparse
import sys
import os

from rag_assistant.pipeline import RAGPipeline


DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "vector_store"
)
DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "raw"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Assistant - IFSP")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Indexar documentos")
    index_parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_DOCS_DIR,
        help="Caminho dos documentos",
    )
    index_parser.add_argument(
        "--persist-dir",
        type=str,
        default=DEFAULT_PERSIST_DIR,
        help="Diretorio para salvar o indice",
    )

    query_parser = subparsers.add_parser("query", help="Fazer uma pergunta")
    query_parser.add_argument("question", type=str, help="Pergunta")
    query_parser.add_argument(
        "--persist-dir",
        type=str,
        default=DEFAULT_PERSIST_DIR,
        help="Diretorio do indice",
    )
    query_parser.add_argument(
        "--model",
        type=str,
        default="pierreguillou/gpt2-small-portuguese",
        help="Modelo gerador",
    )

    chat_parser = subparsers.add_parser("chat", help="Modo interativo")
    chat_parser.add_argument(
        "--persist-dir",
        type=str,
        default=DEFAULT_PERSIST_DIR,
        help="Diretorio do indice",
    )
    chat_parser.add_argument(
        "--model",
        type=str,
        default="pierreguillou/gpt2-small-portuguese",
        help="Modelo gerador",
    )

    return parser.parse_args()


def cmd_index(args) -> None:
    pipeline = RAGPipeline(persist_directory=args.persist_dir)
    print(f"Indexando documentos de: {args.path}")
    num_chunks = pipeline.index_documents(args.path)
    print(f"Indexados {num_chunks} chunks.")
    print(f"Indice salvo em: {args.persist_dir}")


def cmd_query(args) -> None:
    pipeline = RAGPipeline(
        generator_model=args.model,
        persist_directory=args.persist_dir,
    )
    pipeline.load_index()
    result = pipeline.query(args.question)
    print(f"\nPergunta: {result['question']}")
    print(f"\nResposta: {result['answer']}")
    print(f"\nFontes: {', '.join(result['sources'])}")


def cmd_chat(args) -> None:
    pipeline = RAGPipeline(
        generator_model=args.model,
        persist_directory=args.persist_dir,
    )
    pipeline.load_index()
    pipeline.interactive()


def main():
    args = parse_args()

    if args.command is None:
        print("Use: python -m rag_assistant.cli [index|query|chat]")
        print("  index  - Indexar documentos")
        print("  query  - Fazer uma pergunta")
        print("  chat   - Modo interativo")
        sys.exit(1)

    commands = {
        "index": cmd_index,
        "query": cmd_query,
        "chat": cmd_chat,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
