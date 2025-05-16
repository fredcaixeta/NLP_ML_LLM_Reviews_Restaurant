# Usar uma imagem Python base leve
FROM python:3.9-slim

# Definir o diretório de trabalho no container
WORKDIR /app

# Copiar o arquivo de requisitos e instalar as dependências
COPY model_inference_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Criar o diretório para os artefatos no container
RUN mkdir /app/artifacts

# --- Copiar os artefatos do MLflow para o container ---
# Substitua 'path/to/your/mlruns/...' pelos caminhos REAIS
# dos arquivos .pkl, .npy na sua máquina local, dentro da sua run MLflow.
# Você pode encontrar esses caminhos olhando na UI do MLflow ou na estrutura de pastas mlruns/.
# Ex: mlruns/1/abcdef1234567890/artifacts/model/model.pkl
# Ex: mlruns/1/abcdef1234567890/artifacts/X_train_vec.npy
# Ex: mlruns/1/abcdef1234567890/artifacts/train_clusters.npy

# Copiar o classificador PKL
COPY model_inference_app/artifacts/v1_final.pkl /app/artifacts/v1_final.pkl

# Copiar os embeddings de treino NPY
COPY model_inference_app/artifacts/X_train_vec.npy /app/artifacts/X_train_vec.npy

# Copiar os cluster IDs de treino NPY
COPY model_inference_app/artifacts/train_clusters.npy /app/artifacts/train_clusters.npy

# --- Copiar o código da aplicação FastAPI e módulos customizados ---
COPY model_inference_app/app.py /app/app.py

# Expor a porta que o Uvicorn vai usar
EXPOSE 8000

# Comando para rodar a aplicação usando Uvicorn
# A flag --host 0.0.0.0 é necessária para o container ser acessível externamente
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]