import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.svm import SVC
import numpy as np

# Carregar dados
try:
    df_train = pd.read_parquet(r'../data/dataset_train_with_sentiment_fix_negative_trimmed_similarity.parquet')
except:
    df_train = pd.read_parquet(r'data/dataset_train_with_sentiment_fix_negative_trimmed_similarity.parquet')

if 'target' not in df_train.columns:
    df_train['target'] = df_train['sentiment']

# Separar features e target
X = df_train['comment_cleaned']
y = df_train['target']

# Split em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Configuração do MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Train_Trimmed_Fix-Negative_Sentiment_Analysis_Restaurant")

def run_grid_search():
    # Definir os modelos e parâmetros para grid search
    models_params = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'tfidf__max_features': [3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 5]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'tfidf__max_features': [3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'clf__C': [0.1, 1, 10],
                'clf__penalty': ['l2', 'none'],
                'clf__solver': ['lbfgs', 'saga']
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'tfidf__max_features': [5000, 7000],
                'tfidf__ngram_range': [(1, 2)],
                'clf__C': [0.1, 1, 10],
                'clf__kernel': ['linear', 'rbf'],
                'clf__gamma': ['scale', 'auto']
            }
        }
    }

    for model_name, mp in models_params.items():
        with mlflow.start_run(run_name=f"GS_{model_name}", nested=True):
            # Criar pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', mp['model'])
            ])
            
            # Configurar GridSearchCV
            gs = GridSearchCV(
                pipeline,
                mp['params'],
                cv=3,
                n_jobs=-1,
                scoring='f1_weighted',
                verbose=1
            )
            
            # Treinar com grid search
            mlflow.sklearn.autolog()
            gs.fit(X_train, y_train)
            
            # Log adicional manual
            mlflow.log_params(gs.best_params_)
            mlflow.log_metric("best_cv_score", gs.best_score_)
            
            # Avaliar no conjunto de teste
            y_pred = gs.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_metrics({
                "test_accuracy": test_accuracy,
                "test_f1_weighted": test_f1
            })
            
            # Log do relatório de classificação
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f"classification_report_{model_name}.csv")
            mlflow.log_artifact(f"classification_report_{model_name}.csv")
            
            print(f"\n{model_name} - Melhores parâmetros:")
            print(gs.best_params_)
            print(f"Acurácia no teste: {test_accuracy:.4f}")
            print(f"F1-Score no teste: {test_f1:.4f}")

if __name__ == "__main__":
    run_grid_search()