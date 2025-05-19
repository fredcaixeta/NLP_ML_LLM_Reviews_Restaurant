from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
import os
import sys
from tensorflow.keras.models import load_model
from model2vec import StaticModel
from typing import List

class Model2VecEmbeddings:
    """Wrapper para o Model2Vec como Embeddings do LangChain"""
    def __init__(self, model_name: str = "minishlab/potion-base-32M"):
        self.model = StaticModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]

# --- Configurações ---
ARTIFACTS_PATH = r"/app/artifacts"
MODEL_H5_PATH = os.path.join(ARTIFACTS_PATH, "best_sentiment_model.h5")
EMBEDDING_MODEL_NAME = "minishlab/potion-base-32M"  # Mesmo usado no treino

ALL_LABELS = ['Negative', 'Neutral', 'Positive']

# --- Variáveis Globais ---
keras_model = None
embedding_model = None

# --- FastAPI ---
app = FastAPI(title="Sentiment Analysis API")

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health_check():
    status = "ok" if keras_model and embedding_model else "loading_error"
    return {"status": status}

@app.on_event("startup")
async def load_models():
    global keras_model, embedding_model
    
    print("Carregando modelos...")
    
    # 1. Carregar modelo Keras
    try:
        keras_model = load_model(MODEL_H5_PATH)
        print(f"Modelo Keras carregado: {MODEL_H5_PATH}")
    except Exception as e:
        print(f"ERRO ao carregar modelo Keras: {e}")
        sys.exit(1)
    
    # 2. Inicializar modelo de embedding
    try:
        embedding_model = Model2VecEmbeddings(EMBEDDING_MODEL_NAME)
        print(f"Modelo de embedding carregado: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"ERRO ao carregar modelo de embedding: {e}")
        sys.exit(1)

@app.post("/predict")
async def predict_sentiment(item: TextInput):
    if not keras_model or not embedding_model:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"message": "Modelo não carregado. Tente novamente mais tarde."}
        )
    
    try:
        # 1. Gerar embedding em tempo real
        embedding = embedding_model.embed_query(item.text)
        
        # 2. Preparar entrada para o modelo Keras
        input_array = np.array([embedding])  # Adiciona dimensão de batch
        
        # 3. Fazer predição
        predictions = keras_model.predict(input_array, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_label = ALL_LABELS[predicted_class_idx]
        
        # 4. Formatar resposta
        return {
            "text": item.text,
            "predicted_sentiment": predicted_label,
            "confidence": float(np.max(predictions)),
            "probabilities": {
                "Negative": float(predictions[0][0]),
                "Neutral": float(predictions[0][1]),
                "Positive": float(predictions[0][2])
            }
        }
    
    except Exception as e:
        print(f"Erro durante predição: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante processamento: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)