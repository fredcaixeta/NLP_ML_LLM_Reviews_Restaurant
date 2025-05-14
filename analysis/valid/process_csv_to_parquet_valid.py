# %%
import sys
sys.path.append('../')
sys.path.append('./')
import pandas as pd
import requests
import os
import asyncio

import json
import time
try:
    from src.agent_llm import SmartAgentSystem
except:
    from .src.agent_llm import SmartAgentSystem
# %%
import asyncio
from typing import List, Dict, Any
import pandas as pd


async def process_batch(batch: List[str]) -> List[Dict[str, Any]]:
    """Processa um batch de comentários usando o SmartAgentSystem"""
    async with SmartAgentSystem() as agent:
        tasks = [agent.expert_agent(comment) for _, comment in batch]
        results = await asyncio.gather(*tasks)
        
        # Associa cada resultado ao texto original e índice
        return [
            {
                "original_index": idx,
                "original_text": text,
                "sentiment": result.get("sentiment") if result else None,
                "metadata": result.get("metadata") if result else None,
                "raw_response": result
            }
            for (idx, text), result in zip(batch, results)
        ]
        

async def process_comments_in_batches(comments: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
    """Processa todos os comentários em batches assíncronos mantendo a associação completa"""
    all_results = []
    
    # Adiciona índices para rastreamento
    indexed_comments = list(enumerate(comments))
    
    for i in range(0, len(indexed_comments), batch_size):
        batch = indexed_comments[i:i + batch_size]
        print(f"Processando batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, len(comments))} de {len(comments)})")
        
        try:
            batch_results = await process_batch(batch)
            all_results.extend(batch_results)
            
            # Pequena pausa entre batches
            await asyncio.sleep(3)
        except Exception as e:
            print(f"Erro no batch {i//batch_size + 1}: {str(e)}")
            # Adiciona registros vazios para os itens falhos mantendo a ordem
            all_results.extend([
                {
                    "original_index": idx,
                    "original_text": text,
                    "sentiment": None,
                    "metadata": None,
                    "raw_response": None
                }
                for idx, text in batch
            ])
    
    return sorted(all_results, key=lambda x: x["original_index"])  # Garante ordem original  

def expand_results_to_dataframe(df: pd.DataFrame, results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Expande os resultados completos para o DataFrame original"""
    # Cria DataFrame com os resultados
    results_df = pd.DataFrame(results)
    
    # Remove colunas temporárias que não queremos manter
    results_df.drop(columns=['original_index', 'original_text'], inplace=True, errors='ignore')
    
    # Junta com o DataFrame original
    expanded_df = df.copy()
    expanded_df = pd.concat([expanded_df, results_df], axis=1)
    
    # Expande os metadados em colunas separadas
    if 'metadata' in expanded_df.columns:
        metadata_df = expanded_df['metadata'].apply(
            lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series()
        )
        metadata_df = metadata_df.add_prefix('metadata_')
        expanded_df = pd.concat([expanded_df.drop(columns=['metadata']), metadata_df], axis=1)
    
    return expanded_df

def prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara o DataFrame para serialização em Parquet"""
    df = df.copy()
    
    # Converte a coluna raw_response para string JSON
    if 'raw_response' in df.columns:
        df['raw_response'] = df['raw_response'].apply(
            lambda x: json.dumps(x) if x is not None and not isinstance(x, str) else x
        )
    
    # Converte outras colunas de objetos complexos se necessário
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if col != 'raw_response':  # Já tratamos essa
            df[col] = df[col].astype(str)
    
    return df

if __name__ == "__main__":
    # Processa todos os comentários em batches de 10
    # Carrega os dados do arquivo Parquet
    
    try:
        #df = pd.read_parquet('../data/dataset_train_cleaned.parquet')
        df = pd.read_parquet('../data/dataset_valid_cleaned.parquet')
    except:
        #df = pd.read_parquet('./data/dataset_train_cleaned.parquet')
        df = pd.read_parquet('./data/dataset_valid_cleaned.parquet')

    # Verifica se a coluna existe
    if 'comment_cleaned' not in df.columns:
        raise ValueError("O DataFrame não contém a coluna 'comment_cleaned'")

    comments_to_process = df['comment_cleaned'].tolist()

    print(f"Iniciando processamento de {len(comments_to_process)} comentários...")
    results = asyncio.run(process_comments_in_batches(comments_to_process))

    # Expande os resultados para o DataFrame original
    final_df = expand_results_to_dataframe(df, results)
    
    # Prepara o DataFrame para serialização
    final_df = prepare_for_parquet(final_df)
    
    # Salva os resultados em um novo arquivo Parquet
    output_path = 'dataset_valid_with_sentiment.parquet'
    final_df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Processamento concluído! Resultados salvos em {output_path}")
    print("\nResumo dos resultados:")
    print(final_df[['sentiment', 'metadata_responsible', 'metadata_reason']].head())
    print(f"\nTotal de comentários processados: {len(results)}")


# %%
