# Análise de Sentimento e Modelagem de Tópicos para Hub de Restaurantes

## 🎯 Objetivo do Projeto

Este projeto visa desenvolver uma solução para analisar avaliações de clientes de restaurantes em uma plataforma hub. O objetivo principal é classificar o sentimento (positivo, negativo, neutro) dos comentários e identificar os tópicos de discussão, a fim de auxiliar os estabelecimentos parceiros a melhorarem seus serviços.

Um requisito chave é que o algoritmo de classificação de sentimento seja eficiente e possa ser alocado em infraestruturas de baixo custo ou potencialmente no "cliente" (frontend ou serviço leve) para reduzir os custos de infraestrutura centralizada.

## 💡 Solução Proposta

A solução combina o poder de um Large Language Model (LLM) para a **anotação inicial** da base de treinamento com a eficiência de um algoritmo de Machine Learning (ML) tradicional para a **classificação em produção**.

O pipeline geral inclui:

1.  **Limpeza e Pré-processamento de Dados:** Preparar os comentários textuais para análise.
2.  **Rotulação da Base de Treinamento:** Utilizar um LLM (via API Groq) para classificar o sentimento e extrair metadados (como aspectos e razões) dos comentários da base de treinamento. Este passo é feito **uma vez** para preparar os dados de treinamento.
3.  **Modelagem de Tópicos:** Analisar os tópicos de discussão dentro das diferentes categorias de sentimento e/ou metadados extraídos.
4.  **Treinamento do Modelo ML:** Treinar um algoritmo de Machine Learning - MLP por Tensorflow -  em cima dos comentários (representados numericamente via Embeddings) e dos rótulos de sentimento gerados pelo LLM. Este modelo será o responsável pela classificação rápida e eficiente em produção.
5.  **Rastreamento de Experimentos:** Utilizar MLflow para registrar parâmetros, métricas e artefatos (modelos, vetorizadores) durante a fase de treinamento e experimentação.
6.  **Predição:** Usar o modelo ML treinado para classificar novos comentários (base de validação e dados futuros) - em container Docker

## ✨ Funcionalidades Principais

* Processamento de dados textuais brutos (tratamento de formatos, remoção de ruído).
* Rotulação automática de sentimentos em escala usando APIs de LLMs (implementação assíncrona para eficiência).
* Extração de metadados baseada em LLM durante a rotulação (aspectos, responsáveis, razões).
* Implementação de limpeza de texto customizável (stopwords, contrações).
* Vetorização de texto usando um modelo de embedding pré-treinado (`minishlab/potion-base-32M`).
* Definição e treinamento de uma Rede Neural Feedforward (MLP) customizável.
* Foco na otimização e avaliação da métrica de Recall para a classe Negativa durante o tuning.
* Capacidade de prever sentimentos em novos comentários de forma eficiente.
* Análise exploratória via modelagem de tópicos.

## ⚙️ Configuração e Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [SEU REPOSITÓRIO]
    cd [NOME DO REPOSITÓRIO]
    ```
2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv .venv
    # No Windows: .venv\Scripts\activate
    # No macOS/Linux: source .venv/bin/activate
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Obtenha sua chave API de LLM:**
    * Para Groq (conforme código de rotulação assíncrona implementado): Crie uma conta e uma chave API em [https://console.groq.com/](https://console.groq.com/).
5.  **Configure a variável de ambiente da API Key:**
    * **Groq:** Defina a variável `groq_key`.
        * No Linux/macOS (para a sessão atual do terminal):
            ```bash
            export groq_key='SUA_CHAVE_GROQ_AQUI'
            ```
        * No PowerShell (Windows):
            ```powershell
            $env:groq_key='SUA_CHAVE_GROQ_AQUI'
            ```
        * No Command Prompt (Windows):
            ```cmd
            set groq_key='SUA_CHAVE_GROQ_AQUI'
            ```

## Pré-requisitos

* Python 3.7+
* Docker (recomendado para ambiente isolado, conforme discutido anteriormente)
* As seguintes bibliotecas Python:
    * `tensorflow`
    * `numpy`
    * `scikit-learn` (para `TfidfVectorizer`, `StratifiedKFold`, `accuracy_score`, `f1_score`, `classification_report`, `recall_score`)
    * `pandas` (para `y_train.iloc`)
    * `time` (built-in)
    * `model2vec` (para embedding)

## Modelos Utilizados

* **Embedding:** O script utiliza o modelo pré-treinado `minishlab/potion-base-32M` (através da classe `Model2VecEmbeddings`) para converter texto em vetores numéricos.
* **Classificação:** Uma Rede Neural Feedforward (MLP) com duas camadas densas ocultas (com ativação 'relu' e dropout) e uma camada de saída com 3 neurônios e ativação 'softmax' é treinada nos embeddings.

# Uso com Docker

# Exemplo para um criar a imagem
docker build -t tensorflow-nlp-train .

# Exemplo teórico para um serviço rodando na porta 8000 dentro do contêiner
docker run -p 8000:8000 tensorflow-nlp-train

## Rodando o Modelo pelo APP do Docker
curl http://localhost:8000/health

### Predição

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "I simply loved the food!"}' \
  http://localhost:8000/predict
    