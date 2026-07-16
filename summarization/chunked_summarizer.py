from typing import Optional

import torch

from summarization.model_loader import ModelLoader
from summarization.preprocessing import TextPreprocessor


class ChunkedSeq2SeqSummarizer:
    """Sumarizador abstrativo com chunking hierarquico (map-reduce).

    Motivacao: o Seq2SeqSummarizer trunca a entrada em `max_input_length`
    tokens. Em artigos cientificos reais (mediana ~4.800 palavras) isso faz o
    modelo ler apenas o comeco do texto, enquanto o resumo de referencia
    (abstract) sintetiza o artigo inteiro. O BERTimbau extrativo, por outro
    lado, percorre todas as sentencas do artigo — ou seja, os dois nao veem a
    mesma quantidade de texto.

    Estrategia:
      1. map    — divide o texto em janelas de `max_input_length` tokens (com
                  sobreposicao) e resume cada janela em `map_tokens` tokens;
      2. reduce — concatena os resumos parciais e resume de novo; se a
                  concatenacao ainda exceder a janela do modelo, repete o
                  processo (hierarquico), ate `max_niveis`.

    Decodificacao deterministica por beam search, igual ao Seq2SeqSummarizer.
    """

    def __init__(
        self,
        model_name: str = "ptt5-summ-chunk",
        max_input_length: int = 512,
        max_summary_length: int = 150,
        map_tokens: int = 100,
        overlap_tokens: int = 50,
        num_beams: int = 4,
        max_niveis: int = 3,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_summary_length = max_summary_length
        self.map_tokens = map_tokens
        self.overlap_tokens = overlap_tokens
        self.num_beams = num_beams
        self.max_niveis = max_niveis
        self.preprocessor = TextPreprocessor()
        self.loader = ModelLoader(device=device)
        self.model, self.tokenizer = self.loader.load_model(model_name)
        self.device = self.loader.device

    def _preparar(self, text: str) -> str:
        # limpeza + remocao de referencias, mas SEM truncar: o chunking existe
        # justamente para nao descartar o resto do artigo.
        texto = self.preprocessor.clean_text(text)
        return self.preprocessor.remove_references(texto)

    def _n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _dividir(self, text: str) -> list[str]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        janela = max(16, self.max_input_length - 8)  # margem p/ tokens especiais
        passo = max(1, janela - self.overlap_tokens)
        pedacos = []
        for ini in range(0, len(ids), passo):
            bloco = ids[ini:ini + janela]
            if not bloco:
                break
            pedacos.append(self.tokenizer.decode(bloco, skip_special_tokens=True))
            if ini + janela >= len(ids):
                break
        return pedacos or [text]

    def _resumir_um(self, text: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                repetition_penalty=1.2,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def generate_summary(self, text: str) -> str:
        texto = self._preparar(text)
        pedacos = self._dividir(texto)

        # texto curto: identico ao sumarizador sem chunking
        if len(pedacos) == 1:
            return self._resumir_um(pedacos[0], self.max_summary_length)

        # map
        parciais = [self._resumir_um(p, self.map_tokens) for p in pedacos]
        combinado = " ".join(parciais)

        # reduce hierarquico enquanto nao couber na janela do modelo
        nivel = 0
        while self._n_tokens(combinado) > self.max_input_length and nivel < self.max_niveis:
            parciais = [self._resumir_um(p, self.map_tokens) for p in self._dividir(combinado)]
            combinado = " ".join(parciais)
            nivel += 1

        return self._resumir_um(combinado, self.max_summary_length)

    def batch_summarize(self, texts: list[str], **kwargs) -> list[str]:
        return [self.generate_summary(text, **kwargs) for text in texts]
