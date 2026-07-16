"""Coletor de corpus cientifico real em portugues a partir do SciELO.

Estrategia (par entrada -> referencia):
  - entrada  = corpo do artigo (secoes <body> do JATS XML)
  - referencia = resumo (abstract) do autor em portugues (<abstract> do JATS)

Fonte: artigos open-access em PT-BR do SciELO Brasil. O texto completo e
obtido em JATS XML limpo (o site novo do SciELO serve `?format=xml`), o que
separa <abstract> de <body> e evita mistura de idiomas / vazamento do resumo
para dentro do texto de entrada.

Uso:
    python -m data.coletar_scielo --n 30 --out data/processed/corpus_scielo.json

Saida: JSON no mesmo schema do corpus sintetico
    [{"id", "title", "text", "reference_summary", "source"}]

Nao versionar a saida (data/processed/ e gitignored).
"""

import argparse
import json
import os
import time
import urllib.request

from bs4 import BeautifulSoup


UA = {"User-Agent": "resumidor-ic/1.0 (mailto:mauantonelli2@gmail.com)"}
ARTICLEMETA = "http://articlemeta.scielo.org/api/v1"

# Periodicos SciELO Brasil majoritariamente em PT, de areas variadas.
JORNAIS_PT = {
    "0034-8910": "Revista de Saude Publica",
    "0102-311X": "Cadernos de Saude Publica",
    "0100-4042": "Quimica Nova",
    "0101-7330": "Educacao & Sociedade",
    "1414-3283": "Interface - Comunicacao, Saude, Educacao",
    "0103-4014": "Estudos Avancados",
}


def _get(url, timeout=45):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.geturl(), r.read()


def listar_pids(issn, limit=50, offset=0):
    url = f"{ARTICLEMETA}/article/identifiers/?collection=scl&issn={issn}&limit={limit}&offset={offset}"
    _, body = _get(url)
    return [o["code"] for o in json.loads(body).get("objects", [])]


def meta_artigo(pid):
    url = f"{ARTICLEMETA}/article/?collection=scl&code={pid}"
    _, body = _get(url)
    return json.loads(body).get("article", {})


def idioma_pt(meta):
    return meta.get("v40") == [{"_": "pt"}]


def resolver_xml(pid):
    """Segue o redirect da URL antiga para a nova e retorna a URL do JATS XML."""
    final_url, _ = _get(f"https://www.scielo.br/scielo.php?script=sci_arttext&pid={pid}")
    base = final_url.split("?")[0]
    return base + "?format=xml"


def parse_jats(xml_bytes):
    """Extrai (titulo, abstract_pt, corpo) do JATS XML. Usa o artigo principal
    (nao os <sub-article> de traducao)."""
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    art = soup.find("article")
    if art is None:
        return None
    if art.get("{http://www.w3.org/XML/1998/namespace}lang", art.get("xml:lang")) not in (None, "pt"):
        # alguns parsers expõem o atributo como 'xml:lang'
        pass

    front = art.find("front")
    body = art.find("body")
    if front is None or body is None:
        return None

    title_el = front.find("article-title")
    title = title_el.get_text(" ", strip=True) if title_el else ""

    # abstract do artigo principal (o <abstract>, nao <trans-abstract>)
    abs_el = front.find("abstract")
    if abs_el is None:
        return None
    abstract = abs_el.get_text(" ", strip=True)

    # corpo: paragrafos das secoes, LIMPO para evitar vazamento do abstract
    # (artigos retro-digitalizados as vezes replicam titulo/autores/abstract
    # dentro do <body>).
    paras = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    corpo = _montar_corpo(paras, abstract, title)

    return title, abstract, corpo


def _overlap_tokens(a: str, b_tokens: set) -> float:
    at = a.lower().split()
    if not at:
        return 1.0
    return sum(1 for w in at if w in b_tokens) / len(at)


def _montar_corpo(paras, abstract, title):
    """Remove paragrafos que replicam o abstract (vazamento da referencia),
    cabecalhos/autores curtos e o titulo, e devolve o corpo concatenado."""
    abs_tokens = set(abstract.lower().split())
    title_norm = title.lower().strip()
    limpos = []
    for p in paras:
        if len(p.split()) < 8:
            continue  # titulos de secao, autores, legendas
        if title_norm and title_norm in p.lower():
            continue  # linha de titulo replicada
        if _overlap_tokens(p, abs_tokens) >= 0.5:
            continue  # paragrafo que e (parte do) abstract
        limpos.append(p)
    return " ".join(limpos)


def coletar(n_alvo=30, min_palavras_corpo=200, min_palavras_abstract=30, pausa=0.7):
    coletados = []
    vistos = set()
    for issn, nome in JORNAIS_PT.items():
        if len(coletados) >= n_alvo:
            break
        try:
            pids = listar_pids(issn, limit=80)
        except Exception as e:
            print(f"  [skip issn {issn}] listar_pids falhou: {e}")
            continue
        for pid in pids:
            if len(coletados) >= n_alvo:
                break
            if pid in vistos:
                continue
            vistos.add(pid)
            try:
                meta = meta_artigo(pid)
                if not idioma_pt(meta):
                    continue
                xml_url = resolver_xml(pid)
                if "/j/" not in xml_url:
                    continue  # sem versao XML no site novo
                _, xml = _get(xml_url)
                parsed = parse_jats(xml)
                if parsed is None:
                    continue
                title, abstract, corpo = parsed
                if len(abstract.split()) < min_palavras_abstract:
                    continue
                if len(corpo.split()) < min_palavras_corpo:
                    continue
                coletados.append({
                    "id": pid,
                    "title": title,
                    "text": corpo,
                    "reference_summary": abstract,
                    "source": {"journal": nome, "issn": issn, "url": xml_url.split("?")[0]},
                })
                print(f"  [{len(coletados)}/{n_alvo}] {nome}: {title[:60]} "
                      f"(corpo {len(corpo.split())}p / abs {len(abstract.split())}p)")
            except Exception as e:
                print(f"  [erro pid {pid}] {type(e).__name__}: {e}")
            finally:
                time.sleep(pausa)
    return coletados


def parse_args():
    p = argparse.ArgumentParser(description="Coletor de corpus SciELO (PT-BR)")
    p.add_argument("--n", type=int, default=30, help="Numero de artigos a coletar")
    p.add_argument("--out", default="data/processed/corpus_scielo.json")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Coletando ~{args.n} artigos cientificos PT-BR do SciELO...")
    corpus = coletar(n_alvo=args.n)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"\nColetados {len(corpus)} artigos. Salvo em: {args.out}")


if __name__ == "__main__":
    main()
