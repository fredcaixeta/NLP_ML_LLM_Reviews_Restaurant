from model2vec import StaticModel
from langchain.embeddings.base import Embeddings
from typing import List, Tuple, Dict

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import pandas as pd
import numpy as np

model = StaticModel.from_pretrained("minishlab/potion-base-2M")
model.encode()

class Model2VecEmbeddings(Embeddings):
    """Wrapper para o Model2Vec como Embeddings do LangChain"""
    def __init__(self, model_name: str = "minishlab/potion-base-2M", similarity_threshold: float = 0.85):
        self.model = StaticModel.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]
    
    def find_similar_pairs(self, embeddings: np.ndarray) -> List[Tuple[int, int]]:
        """Identifica pares similares usando similaridade de cosseno"""
        print("Calculando similaridades...")
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)  # Ignora auto-similaridade
        
        similar_pairs = []
        n = sim_matrix.shape[0]
        
        # Encontra pares acima do threshold
        for i in tqdm(range(n), desc="Processando similaridades"):
            for j in range(i+1, n):
                if sim_matrix[i, j] > self.similarity_threshold:
                    similar_pairs.append((i, j))
        
        return similar_pairs