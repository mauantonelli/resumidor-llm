import argparse
import sys

from summarization.model_loader import ModelLoader, SUPPORTED_MODELS


SAMPLE_TEXT = (
    "A inteligência artificial tem transformado diversas áreas do conhecimento, "
    "incluindo a medicina, a engenharia e as ciências sociais. Modelos de linguagem "
    "de grande escala, como GPT e BERT, demonstraram capacidade notável de compreender "
    "e gerar texto em linguagem natural. No contexto brasileiro, a aplicação dessas "
    "tecnologias enfrenta desafios específicos relacionados à língua portuguesa e à "
    "disponibilidade de dados de treinamento. Este trabalho investiga o uso de modelos "
    "de linguagem open-source para a tarefa de sumarização automática de textos "
    "científicos em português. A sumarização automática é uma tarefa fundamental em "
    "processamento de linguagem natural que visa produzir versões condensadas de textos "
    "longos, preservando as informações mais relevantes. Os resultados preliminares "
    "indicam que, apesar das limitações dos modelos menores, é possível obter resumos "
    "coerentes quando se utilizam técnicas adequadas de prompt engineering e "
    "pós-processamento dos textos gerados."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumidor Automático de Textos Científicos"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Nome do modelo a ser utilizado",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Texto a ser resumido",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Caminho para arquivo de texto a ser resumido",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=150,
        help="Número máximo de tokens no resumo",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura para geração",
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=3,
        help="Número de frases para sumarização extrativa",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Listar modelos disponíveis",
    )
    return parser.parse_args()


def load_text_from_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def create_summarizer(model_name: str, args):
    model_config = SUPPORTED_MODELS.get(model_name)
    if model_config is None:
        print(f"Modelo '{model_name}' não suportado.")
        print(f"Modelos disponíveis: {list(SUPPORTED_MODELS.keys())}")
        sys.exit(1)

    if model_config["type"] == "extractive":
        from summarization.extractive_summarizer import ExtractiveSummarizer
        return ExtractiveSummarizer(num_sentences=args.num_sentences)
    elif model_config["type"] == "seq2seq":
        from summarization.seq2seq_summarizer import Seq2SeqSummarizer
        return Seq2SeqSummarizer(
            model_name=model_name,
            max_summary_length=args.max_length,
        )
    elif model_config["type"] == "seq2seq_chunk":
        from summarization.chunked_summarizer import ChunkedSeq2SeqSummarizer
        return ChunkedSeq2SeqSummarizer(
            model_name=model_name,
            max_summary_length=args.max_length,
        )
    else:
        from summarization.summarizer import Summarizer
        return Summarizer(
            model_name=model_name,
            max_summary_length=args.max_length,
        )


def main():
    args = parse_args()

    if args.list_models:
        loader = ModelLoader()
        models = loader.list_available_models()
        print("Modelos disponíveis:")
        for m in models:
            info = SUPPORTED_MODELS[m]
            tipo = "extrativo" if info["type"] == "extractive" else "generativo"
            print(f"  - {m} ({tipo})")
        return

    if args.text:
        input_text = args.text
    elif args.file:
        input_text = load_text_from_file(args.file)
    else:
        print("Usando texto de exemplo...")
        input_text = SAMPLE_TEXT

    print(f"\nModelo: {args.model}")
    print(f"Carregando modelo...")

    summarizer = create_summarizer(args.model, args)

    print(f"Gerando resumo...\n")
    print("=" * 60)
    print("TEXTO ORIGINAL:")
    print("=" * 60)
    print(input_text[:500])
    if len(input_text) > 500:
        print(f"... ({len(input_text)} caracteres no total)")

    model_config = SUPPORTED_MODELS[args.model]
    if model_config["type"] == "causal":
        summary = summarizer.generate_summary(
            input_text,
            temperature=args.temperature,
        )
    else:
        # extrativo e seq2seq nao usam temperatura
        summary = summarizer.generate_summary(input_text)

    print("\n" + "=" * 60)
    print("RESUMO GERADO:")
    print("=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
