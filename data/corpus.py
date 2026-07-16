import copy
import json
import os
from typing import Optional


BUILTIN_SAMPLES = [
    {
        "id": "sample_01",
        "title": "Inteligencia Artificial na Educacao",
        "text": (
            "A inteligencia artificial tem transformado diversas areas do conhecimento, "
            "incluindo a medicina, a engenharia e as ciencias sociais. Modelos de linguagem "
            "de grande escala, como GPT e BERT, demonstraram capacidade notavel de compreender "
            "e gerar texto em linguagem natural. No contexto brasileiro, a aplicacao dessas "
            "tecnologias enfrenta desafios especificos relacionados a lingua portuguesa e a "
            "disponibilidade de dados de treinamento. Este trabalho investiga o uso de modelos "
            "de linguagem open-source para a tarefa de sumarizacao automatica de textos "
            "cientificos em portugues. A sumarizacao automatica e uma tarefa fundamental em "
            "processamento de linguagem natural que visa produzir versoes condensadas de textos "
            "longos, preservando as informacoes mais relevantes. Os resultados preliminares "
            "indicam que, apesar das limitacoes dos modelos menores, e possivel obter resumos "
            "coerentes quando se utilizam tecnicas adequadas de prompt engineering e "
            "pos-processamento dos textos gerados."
        ),
        "reference_summary": (
            "Este trabalho investiga o uso de modelos de linguagem open-source para "
            "sumarizacao automatica de textos cientificos em portugues. Resultados "
            "preliminares mostram que modelos menores podem gerar resumos coerentes "
            "com tecnicas adequadas de prompt engineering."
        ),
    },
    {
        "id": "sample_02",
        "title": "Processamento de Linguagem Natural para o Portugues",
        "text": (
            "O processamento de linguagem natural e uma subarea da inteligencia artificial "
            "dedicada a interacao entre computadores e linguagem humana. Nos ultimos anos, "
            "avancos significativos foram alcancados com o desenvolvimento de modelos baseados "
            "em transformers, como BERT, GPT e T5. Esses modelos sao pre-treinados em grandes "
            "volumes de texto e podem ser adaptados para diversas tarefas, como classificacao "
            "de texto, traducao automatica, resposta a perguntas e sumarizacao. Para a lingua "
            "portuguesa, o BERTimbau representa um marco importante, sendo o primeiro modelo "
            "BERT treinado especificamente com textos em portugues brasileiro. O modelo foi "
            "treinado com dados do BrWaC, um corpus com 2,7 bilhoes de tokens. Experimentos "
            "mostraram que o BERTimbau supera o BERT multilingue em tarefas como analise de "
            "sentimentos e reconhecimento de entidades nomeadas em portugues. A disponibilidade "
            "de modelos especializados para o portugues e essencial para o avanco das pesquisas "
            "em PLN no Brasil."
        ),
        "reference_summary": (
            "O BERTimbau e o primeiro modelo BERT treinado especificamente para portugues "
            "brasileiro, utilizando o corpus BrWaC. Ele supera o BERT multilingue em tarefas "
            "como analise de sentimentos e reconhecimento de entidades em portugues."
        ),
    },
    {
        "id": "sample_03",
        "title": "Sumarizacao Automatica de Textos",
        "text": (
            "A sumarizacao automatica de textos e uma tarefa classica em processamento de "
            "linguagem natural que consiste em produzir uma versao reduzida de um documento, "
            "mantendo suas informacoes mais relevantes. Existem duas abordagens principais: "
            "a sumarizacao extrativa, que seleciona frases importantes do texto original, e "
            "a sumarizacao abstrativa, que gera novas frases para expressar o conteudo. "
            "Metodos extrativos sao tradicionalmente mais simples e confiaveis, pois nao "
            "introduzem informacoes novas. Ja os metodos abstrativos, potencializados por "
            "modelos de linguagem modernos, podem produzir resumos mais fluentes e concisos. "
            "A avaliacao de resumos automaticos e tipicamente realizada com metricas como "
            "ROUGE, que compara n-gramas entre o resumo gerado e um resumo de referencia. "
            "Apesar de amplamente utilizada, a metrica ROUGE possui limitacoes conhecidas, "
            "como a incapacidade de capturar adequadamente a qualidade semantica dos resumos. "
            "Pesquisas recentes propoem o uso de metricas baseadas em embeddings, como "
            "BERTScore, para uma avaliacao mais robusta."
        ),
        "reference_summary": (
            "A sumarizacao automatica pode ser extrativa ou abstrativa. Metodos extrativos "
            "selecionam frases do texto original, enquanto abstrativos geram novas frases. "
            "A avaliacao usa metricas como ROUGE, embora pesquisas recentes proponham "
            "alternativas baseadas em embeddings como BERTScore."
        ),
    },
    {
        "id": "sample_04",
        "title": "Aprendizado de Maquina na Saude",
        "text": (
            "O aprendizado de maquina tem se mostrado uma ferramenta promissora para diversas "
            "aplicacoes na area da saude. Algoritmos de classificacao e regressao sao utilizados "
            "para diagnostico auxiliado por computador, predicao de doencas e personalizacao de "
            "tratamentos. Redes neurais convolucionais revolucionaram a analise de imagens medicas, "
            "alcancando desempenho comparavel ao de especialistas em tarefas como deteccao de "
            "tumores em mamografias e classificacao de lesoes dermatologicas. Modelos de linguagem "
            "sao aplicados na extracao de informacoes de prontuarios eletronicos e na analise de "
            "literatura medica. No entanto, desafios importantes persistem, incluindo a "
            "necessidade de grandes volumes de dados anotados, questoes de privacidade e "
            "regulamentacao, e a dificuldade de interpretacao dos modelos. A explicabilidade "
            "das decisoes algoritmicas e fundamental para a aceitacao dessas tecnologias por "
            "profissionais de saude e pacientes."
        ),
        "reference_summary": (
            "O aprendizado de maquina e utilizado na saude para diagnostico, predicao de "
            "doencas e analise de imagens medicas com desempenho comparavel a especialistas. "
            "Desafios incluem necessidade de dados anotados, privacidade e explicabilidade "
            "dos modelos."
        ),
    },
    {
        "id": "sample_05",
        "title": "Modelos de Linguagem de Grande Escala",
        "text": (
            "Modelos de linguagem de grande escala representam uma das principais tendencias "
            "em inteligencia artificial nos ultimos anos. Esses modelos, treinados com bilhoes "
            "de parametros em vastos corpora de texto, demonstram capacidades emergentes como "
            "raciocinio logico, geracao de codigo e compreensao de contexto complexo. A familia "
            "GPT da OpenAI e os modelos LLaMA da Meta sao exemplos notaveis dessa categoria. "
            "O treinamento desses modelos exige recursos computacionais significativos, o que "
            "levanta preocupacoes sobre sustentabilidade ambiental e concentracao de poder em "
            "grandes corporacoes de tecnologia. Iniciativas de codigo aberto, como o LLaMA, "
            "buscam democratizar o acesso a essas tecnologias. Tecnicas de eficiencia como "
            "quantizacao e destilacao de conhecimento permitem executar versoes reduzidas "
            "desses modelos em hardware mais acessivel, ampliando as possibilidades de "
            "pesquisa e aplicacao em instituicoes com recursos limitados."
        ),
        "reference_summary": (
            "Modelos de linguagem de grande escala como GPT e LLaMA possuem capacidades "
            "emergentes mas exigem recursos computacionais significativos. Iniciativas "
            "open-source e tecnicas como quantizacao buscam democratizar o acesso a "
            "essas tecnologias."
        ),
    },
    {
        "id": "sample_06",
        "title": "Redes Neurais Convolucionais em Visao Computacional",
        "text": (
            "Redes neurais convolucionais transformaram o campo da visao computacional nas "
            "ultimas decadas. Arquiteturas como AlexNet, VGG, ResNet e EfficientNet estabeleceram "
            "marcos sucessivos em tarefas de classificacao de imagens. O mecanismo de convolucao "
            "permite que essas redes aprendam hierarquias de caracteristicas visuais, desde bordas "
            "simples nas camadas iniciais ate padroes complexos nas camadas profundas. Transfer "
            "learning tornou-se uma pratica padrao, permitindo que modelos pre-treinados no ImageNet "
            "sejam adaptados para dominios especificos com poucos dados. Aplicacoes incluem "
            "diagnostico medico por imagem, veiculos autonomos, reconhecimento facial e analise "
            "de imagens de satelite. Desafios atuais envolvem a necessidade de grandes conjuntos "
            "de dados anotados, o custo computacional do treinamento e questoes eticas relacionadas "
            "ao uso de reconhecimento facial em vigilancia."
        ),
        "reference_summary": (
            "Redes neurais convolucionais revolucionaram a visao computacional com arquiteturas "
            "progressivamente mais eficientes. Transfer learning permite adaptacao para dominios "
            "especificos. Desafios incluem custo computacional e questoes eticas do reconhecimento facial."
        ),
    },
    {
        "id": "sample_07",
        "title": "Aprendizado por Reforco e suas Aplicacoes",
        "text": (
            "O aprendizado por reforco e um paradigma de aprendizado de maquina onde um agente "
            "aprende a tomar decisoes sequenciais para maximizar uma recompensa acumulada. "
            "Algoritmos como Q-Learning, SARSA e Policy Gradient formam a base teorica da area. "
            "O marco mais notavel foi o AlphaGo, que derrotou o campeao mundial de Go em 2016, "
            "demonstrando que agentes de aprendizado por reforco podem superar humanos em tarefas "
            "de alta complexidade. Aplicacoes praticas incluem controle de robotica, otimizacao "
            "de sistemas de recomendacao, gerenciamento de trafego urbano e design de moleculas "
            "para farmacos. O aprendizado por reforco profundo combina redes neurais com algoritmos "
            "de reforco, permitindo lidar com espacos de estado de alta dimensionalidade. Porem, "
            "desafios como instabilidade do treinamento, necessidade de muitas interacoes com o "
            "ambiente e dificuldade de definir funcoes de recompensa adequadas limitam a adocao "
            "em cenarios do mundo real."
        ),
        "reference_summary": (
            "O aprendizado por reforco permite que agentes aprendam decisoes sequenciais para "
            "maximizar recompensas. O AlphaGo demonstrou seu potencial ao derrotar humanos em Go. "
            "Desafios incluem instabilidade do treinamento e definicao de funcoes de recompensa."
        ),
    },
    {
        "id": "sample_08",
        "title": "Etica e Vies em Inteligencia Artificial",
        "text": (
            "A etica em inteligencia artificial tornou-se um tema central nas discussoes sobre "
            "tecnologia e sociedade. Sistemas de IA podem perpetuar e amplificar vieses presentes "
            "nos dados de treinamento, levando a discriminacao em areas como contratacao, credito "
            "e justica criminal. Estudos demonstraram que modelos de linguagem treinados em dados "
            "da internet absorvem estereotipos de genero, raca e religiao. Frameworks de fairness "
            "em machine learning buscam quantificar e mitigar esses vieses, utilizando metricas "
            "como paridade demografica, igualdade de oportunidade e calibracao. A transparencia "
            "algoritmica e a explicabilidade dos modelos sao consideradas essenciais para a "
            "construcao de sistemas confiaveis. Regulamentacoes como o AI Act da Uniao Europeia "
            "e a LGPD no Brasil estabelecem diretrizes para o uso responsavel de IA. O debate "
            "atual envolve equilibrar inovacao tecnologica com protecao de direitos fundamentais "
            "e justica social."
        ),
        "reference_summary": (
            "Sistemas de IA podem perpetuar vieses dos dados de treinamento, causando discriminacao. "
            "Frameworks de fairness buscam mitigar esses problemas. Regulamentacoes como o AI Act "
            "e a LGPD estabelecem diretrizes para uso responsavel de IA."
        ),
    },
    {
        "id": "sample_09",
        "title": "Computacao em Nuvem e Infraestrutura para IA",
        "text": (
            "A computacao em nuvem revolucionou a forma como recursos computacionais sao "
            "provisionados e utilizados para treinamento de modelos de inteligencia artificial. "
            "Plataformas como AWS, Google Cloud e Azure oferecem GPUs e TPUs sob demanda, "
            "democratizando o acesso a hardware especializado. O paradigma de MLOps combina "
            "praticas de DevOps com machine learning, automatizando pipelines de treinamento, "
            "validacao e deploy de modelos. Ferramentas como Kubernetes, Docker e plataformas "
            "como MLflow e Kubeflow facilitam o gerenciamento do ciclo de vida de modelos. "
            "O treinamento distribuido permite escalar o processamento de grandes modelos "
            "utilizando multiplas GPUs em paralelo. No entanto, custos podem escalar rapidamente, "
            "e questoes de latencia, privacidade de dados e dependencia de fornecedores "
            "representam desafios para organizacoes que migram workloads de IA para a nuvem."
        ),
        "reference_summary": (
            "A computacao em nuvem democratizou o acesso a hardware para IA com GPUs e TPUs "
            "sob demanda. MLOps automatiza o ciclo de vida de modelos. Desafios incluem custos "
            "escalantes, privacidade e dependencia de fornecedores."
        ),
    },
    {
        "id": "sample_10",
        "title": "Geracao Aumentada por Recuperacao (RAG)",
        "text": (
            "A Geracao Aumentada por Recuperacao, conhecida como RAG, e uma tecnica que combina "
            "modelos de linguagem com sistemas de recuperacao de informacao para gerar respostas "
            "mais precisas e fundamentadas. O processo envolve tres etapas: indexacao de documentos "
            "em um vector store, recuperacao de trechos relevantes com base na consulta do usuario, "
            "e geracao de resposta utilizando o contexto recuperado. Essa abordagem mitiga problemas "
            "comuns de LLMs como alucinacoes e informacoes desatualizadas, pois ancora as respostas "
            "em documentos verificaveis. Embeddings semanticos sao utilizados para representar "
            "documentos e consultas em espacos vetoriais, permitindo busca por similaridade eficiente. "
            "Ferramentas como LangChain e LlamaIndex simplificam a implementacao de pipelines RAG. "
            "Aplicacoes incluem chatbots corporativos, assistentes de pesquisa e sistemas de "
            "perguntas e respostas sobre bases documentais especificas."
        ),
        "reference_summary": (
            "RAG combina modelos de linguagem com recuperacao de informacao para gerar respostas "
            "fundamentadas em documentos. A tecnica mitiga alucinacoes de LLMs ancorando respostas "
            "em dados verificaveis. Ferramentas como LangChain facilitam sua implementacao."
        ),
    },
]


class Corpus:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        else:
            self.data_dir = data_dir

    def get_builtin_samples(self) -> list[dict]:
        return copy.deepcopy(BUILTIN_SAMPLES)

    def load_from_json(self, filepath: str) -> list[dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_to_json(self, samples: list[dict], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    def get_texts_and_references(
        self, samples: Optional[list[dict]] = None
    ) -> tuple[list[str], list[str]]:
        if samples is None:
            samples = self.get_builtin_samples()
        texts = [s["text"] for s in samples]
        references = [s["reference_summary"] for s in samples]
        return texts, references
