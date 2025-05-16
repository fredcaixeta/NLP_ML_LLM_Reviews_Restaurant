from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
import numpy as np

import pickle
import uvicorn

import os
import warnings
import sys # Importe sys para adicionar path

# Importar as mesmas classes e modelos usados no treinamento
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering, KMeans

from model2vec import StaticModel
from typing import List, Tuple, Dict

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class Model2VecEmbeddings():
    """Wrapper para o Model2Vec como Embeddings do LangChain"""
    def __init__(self, model_name: str = "minishlab/potion-base-2M", similarity_threshold: float = 0.85):
        self.model = StaticModel.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]

# Adicione o diretório raiz do seu projeto ao sys.path para poder importar seus módulos customizados
# Assumindo que Model2VecEmbeddings está definido em algum lugar acessível a partir da raiz.
# Ajuste o caminho se necessário. Ex: se Model2VecEmbeddings está em seu_projeto/src/models.py
# sys.path.append("/app") # No container, /app é a raiz

# --- Configurações do Caminho dos Artefatos no CONTAINER ---
# Estes caminhos são relativos ao diretório de trabalho /app no container
ARTIFACTS_PATH = r"/app/artifacts"
CLASSIFIER_PKL_PATH = os.path.join(ARTIFACTS_PATH, "v1_final.pkl")
TRAIN_EMBEDDINGS_NPY_PATH = os.path.join(ARTIFACTS_PATH, "X_train_vec.npy")
TRAIN_CLUSTERS_NPY_PATH = os.path.join(ARTIFACTS_PATH, "train_clusters.npy")

LOGGED_MODEL_PARAMS = {
    'embedding_model': 'minishlab/potion-base-8M', # Nome do modelo de embedding
    'clustering_algorithm': 'AgglomerativeClustering', # Nome do algoritmo de clustering
    # Use os parâmetros EXATOS que foram logados para a CONFIGURAÇÃO ALVO
    'clustering_params': {'n_clusters': 3, 'linkage': 'ward'},
    'classification_model': 'LogisticRegression', # Nome do classificador
    # Use os parâmetros EXATOS que foram logados para a CONFIGURAÇÃO ALVO
    'classification_model_params': {'C': 0.01, 'max_iter': 500, 'penalty': 'l1', 'solver': 'liblinear'}
}

ALL_LABELS = ['Negative', 'Neutral', 'Positive']

# --- Variáveis Globais para os Componentes do Pipeline Carregados ---
classifier_model = None
embedding_model_instance = None # Use um nome diferente para evitar conflito com o dicionário
X_train_vec_loaded = None
train_clusters_loaded = None
nn_model_loaded = None

# --- Definição do Aplicativo FastAPI ---
app = FastAPI(title="Sentiment Analysis Model Inference")

# --- Definição do Modelo de Request ---
class TextInput(BaseModel):
    text: str

# --- Endpoint Health Check ---
@app.get("/health")
def health_check():
    """Health check endpoint to verify the API is running."""
    # Verifica se os componentes essenciais foram carregados
    status = "ok" if classifier_model and embedding_model_instance and nn_model_loaded is not None and X_train_vec_loaded is not None and train_clusters_loaded is not None else "loading_error"
    detail = "All components loaded" if status == "ok" else "One or more components failed to load during startup"
    return {"status": status, "detail": detail}

# --- Evento de Startup: Carregar Modelos e Dados ---
@app.on_event("startup")
async def load_models():
    global classifier_model, embedding_model_instance, X_train_vec_loaded, train_clusters_loaded, nn_model_loaded

    print("Iniciando carregamento de modelos e dados...")

    # 1. Carregar Classificador
    try:
        with open(CLASSIFIER_PKL_PATH, 'rb') as f:
            classifier_model = pickle.load(f)
        print(f"Classificador carregado com sucesso de: {CLASSIFIER_PKL_PATH}")
    except FileNotFoundError:
        print(f"ERRO FATAL: Arquivo do classificador não encontrado em {CLASSIFIER_PKL_PATH}")
        # Em um ambiente de produção, você pode querer logar isso em um sistema de monitoramento
        sys.exit(1) # Encerrar o container se o modelo não puder ser carregado
    except Exception as e:
        print(f"ERRO FATAL ao carregar o classificador: {e}")
        sys.exit(1)

    # 2. Instanciar Modelo de Embedding
    try:
        # A classe Model2VecEmbeddings deve estar disponível (definida ou importada)
        # O carregamento real do modelo de embedding acontece dentro do __init__
        embedding_model_instance = Model2VecEmbeddings(LOGGED_MODEL_PARAMS['embedding_model'])
        print(f"Modelo de Embedding '{LOGGED_MODEL_PARAMS['embedding_model']}' instanciado.")
    except Exception as e:
        print(f"ERRO FATAL ao instanciar o modelo de embedding: {e}")
        sys.exit(1)

    # 3. Carregar Embeddings e Clusters de Treino
    try:
        X_train_vec_loaded = np.load(TRAIN_EMBEDDINGS_NPY_PATH)
        train_clusters_loaded = np.load(TRAIN_CLUSTERS_NPY_PATH)
        print(f"Embeddings de treino carregados de: {TRAIN_EMBEDDINGS_NPY_PATH} (shape: {X_train_vec_loaded.shape})")
        print(f"Clusters de treino carregados de: {TRAIN_CLUSTERS_NPY_PATH} (shape: {train_clusters_loaded.shape})")
        # Verifica se as formas são consistentes (número de amostras deve ser igual)
        if X_train_vec_loaded.shape[0] != train_clusters_loaded.shape[0]:
             print(f"AVISO: Número de amostras nos embeddings de treino ({X_train_vec_loaded.shape[0]}) difere dos clusters de treino ({train_clusters_loaded.shape[0]}).")
             # Isso pode causar problemas. Considere sair aqui também.
             # sys.exit(1)
    except FileNotFoundError:
        print(f"ERRO FATAL: Arquivos .npy de treino não encontrados ({TRAIN_EMBEDDINGS_NPY_PATH}, {TRAIN_CLUSTERS_NPY_PATH})")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO FATAL ao carregar arquivos .npy de treino: {e}")
        sys.exit(1)

    # 4. Fitar NearestNeighbors nos Embeddings de Treino Carregados
    try:
        # O NearestNeighbors NÃO foi salvo no treino, precisa ser instanciado e fitado AQUI
        nn_model_loaded = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_model_loaded.fit(X_train_vec_loaded) # Fita NN nos embeddings de treino carregados
        print("NearestNeighbors fitado nos embeddings de treino.")
    except Exception as e:
        print(f"ERRO FATAL ao fitar NearestNeighbors: {e}")
        sys.exit(1)

    print("Todos os modelos e dados necessários carregados com sucesso.")


@app.post("/predict")
async def predict_sentiment(item: TextInput):
    """
    Recebe texto de entrada, aplica o pipeline de pré-processamento e retorna a predição de sentimento.
    """
    # Verifica se todos os componentes globais foram carregados corretamente no startup
    if classifier_model is None or embedding_model_instance is None or nn_model_loaded is None or X_train_vec_loaded is None or train_clusters_loaded is None:
        print("ERRO: Componentes do modelo não carregados. Predição impossível.")
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"message": "Model components not loaded. Server is not ready."})

    try:
        # 1. Embedding
        # O método embed_documents deve receber uma lista de strings
        new_text_list = [item.text]
        new_embedding = embedding_model_instance.embed_documents(new_text_list) # Retorna array numpy (1, embedding_dim)

        # 2. Atribuição de Cluster usando Nearest Neighbors
        # Encontra o vizinho mais próximo do NOVO embedding nos embeddings de TREINO
        distances, indices = nn_model_loaded.kneighbors(new_embedding) # indices retorna array aninhado como [[indice]]
        # Obtém o cluster ID do vizinho mais próximo. indices.flatten()[0] pega o índice numérico
        assigned_cluster_id = train_clusters_loaded[indices.flatten()][0] # Pega o ID do cluster de treino correspondente

        # 3. Aumentar Features
        # Concatena o novo embedding com o cluster ID atribuído
        # O cluster ID deve ser um array 2D [[ID]] para hstack funcionar corretamente com new_embedding (1, dim)
        new_augmented_features = np.hstack([new_embedding, np.array([[assigned_cluster_id]])]) # Resulta em shape (1, embedding_dim + 1)

        # 4. Classificação
        # predict retorna um numpy array, take the first element (since batch size is 1)
        raw_prediction_output = classifier_model.predict(new_augmented_features)[0]

        print(f"DEBUG: raw_prediction_output: {raw_prediction_output}")
        print(f"DEBUG: type(raw_prediction_output): {type(raw_prediction_output)}")
        print(f"DEBUG: ALL_LABELS: {ALL_LABELS}")

        predicted_label = None

        # --- Ajuste AQUI: Verifique o tipo da predição e use-a diretamente se for string ---
        if isinstance(raw_prediction_output, str):
            # Se a predição já é uma string, usamos ela diretamente como o label previsto.
            predicted_label = raw_prediction_output
            # Opcional: Verificar se o label previsto está na sua lista ALL_LABELS
            if predicted_label not in ALL_LABELS:
                 print(f"AVISO: Predição de label inesperada não encontrada em ALL_LABELS: {predicted_label}")
                 # Trate como um label desconhecido se necessário
                 # predicted_label = "unknown_label"

        elif isinstance(raw_prediction_output, (int, np.integer)):
             # Caso padrão (se o modelo algum dia voltar a prever IDs numéricos): Mapear ID para label
             predicted_numeric_id = int(raw_prediction_output) # Garante que é int padrão
             if 0 <= predicted_numeric_id < len(ALL_LABELS):
                  predicted_label = ALL_LABELS[predicted_numeric_id]
             else:
                  # Se o ID numérico estiver fora do range esperado
                  print(f"AVISO: Predição numérica fora do range esperado ({len(ALL_LABELS)} classes): {predicted_numeric_id}")
                  predicted_label = "unknown_id" # Indique que é um ID numérico inválido

        else:
            # Lidar com qualquer outro tipo inesperado retornado pelo modelo
            print(f"ERRO: Tipo de predição inesperado retornado pelo modelo: {type(raw_prediction_output)}")
            predicted_label = "prediction_type_error" # Indique um erro no tipo de saída


        # --- Retornar a predição ---
        # Retorna a predição SE for um dos labels esperados ou um dos placeholders de erro
        if predicted_label in (ALL_LABELS + ["unknown_label", "unknown_id", "prediction_type_error"]):
             # Você pode querer retornar um status code diferente para erros
             if predicted_label in ["unknown_label", "unknown_id", "prediction_type_error"]:
                  from fastapi.responses import JSONResponse
                  return JSONResponse(status_code=500, content={"text": item.text, "predicted_label": predicted_label, "message": "Model prediction resulted in an invalid or unexpected label."})
             else:
                  # Retorna o label previsto se for um dos labels esperados
                  return {"text": item.text, "predicted_label": predicted_label}
        else:
             # Segurança caso a lógica acima falhe ou retorne algo totalmente inesperado
             print(f"ERRO FATAL: Lógica de tratamento de predição falhou para output: {raw_prediction_output}")
             from fastapi.responses import JSONResponse
             return JSONResponse(status_code=500, content={"message": "Critical error processing model prediction."})


    except Exception as e:
        # Captura qualquer outra exceção não tratada acima (ex: erro de embedding, nn, hstack)
        print(f"Erro inesperado durante a inferência: {e}")
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"message": f"Internal server error during inference process: {e}"})



# O bloco if __name__ == "__main__": não é estritamente necessário para o Docker,
# pois o comando CMD no Dockerfile irá executá-lo diretamente.
# Mas é útil para testar a API localmente sem o Docker.
if __name__ == "__main__":
    print("Rodando API localmente (para desenvolvimento/teste)...")
    # Para teste local, certifique-se de que os arquivos model.pkl, X_train_vec.npy,
    # train_clusters.npy estejam na pasta ./artifacts ou ajuste ARTIFACTS_PATH.
    # Ex: Crie uma pasta 'artifacts' no mesmo nível do app.py e copie os arquivos para lá.
    # No Dockerfile, eles estarão em /app/artifacts.
    # Se você estiver rodando localmente, pode ajustar ARTIFACTS_PATH temporariamente.
    # ARTIFACTS_PATH = "./artifacts" # Descomente para teste local se os arquivos estiverem aqui
    # E rode o script diretamente: python app/app.py
    uvicorn.run("model_inference_app:app", host="0.0.0.0", port=8000, reload=True) # reload=True para desenvolvimento