from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os

from app.utils.helpers import setup_logger
from app.services.explain_service import ExplainService
from app.config import MODELS_DIR, FEATURE_STORE_DATA_DIR

logger = setup_logger(__name__)

router = APIRouter(prefix="/explain", tags=["Explicabilidade e XAI"])


class ExplainResponse(BaseModel):
    type: str
    description: str
    image_base64: str


def load_model(version: str):
    """Carrega o modelo do joblib."""
    import joblib
    # Aqui precisamos usar MODELS_DIR
    model_path = os.path.join(MODELS_DIR, f"xgboost_defasagem_v{version}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Modelo v{version} não encontrado.")
    return joblib.load(model_path)


def load_training_data(version: str):
    """Carrega as features de treinamento usadas na versão para gerar SHAP e PCA."""
    # Para simplificar, estamos pegando dados de amostra. 
    # Em produção, ou você grava um slice X_test na hora do joblib, ou busca do feature store.
    # Vamos simular um X (sem a variavel target) da pasta do dataset original ou feature store
    try:
        data_path = os.path.join(FEATURE_STORE_DATA_DIR, "materialized", "features.parquet")
        if not os.path.exists(data_path):
             # Retorna None quando não tem base para não explodir
             return None, None
             
        df = pd.read_parquet(data_path)
        
        # O ALVO no projeto Datathon é "status_aluno"
        target_col = "status_aluno"
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = np.zeros(len(df)) # Dummy Y se não achar o target
            
        # Pega só as features usadas no treinamento original lendo os metadados do modelo
        model_path = os.path.join(MODELS_DIR, f"xgboost_defasagem_v{version}.joblib")
        import joblib
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict) and "features" in model_data:
            features = model_data["features"]
            X = X[features]
            
        return X, y
    except Exception as e:
        logger.error(f"Erro ao carregar dados p/ explicabilidade: {e}")
        return None, None


@router.post("/shap/{version}", response_model=ExplainResponse, summary="Gera gráfico Global SHAP")
async def get_shap_explanation(version: str, max_display: int = 15):
    """
    Retorna o base64 de um gráfico Summary Plot do SHAP.
    Atualmente suportado para XGBoost, CatBoost e LightGBM.
    """
    model_data = load_model(version)
    if not isinstance(model_data, dict) or "model" not in model_data:
        raise HTTPException(status_code=400, detail="Formato de modelo salvo inválido.")
        
    model = model_data["model"]
    model_type = model.__class__.__name__.lower()
    
    if "xgb" not in model_type and "catboost" not in model_type and "lgbm" not in model_type:
         raise HTTPException(
             status_code=400, 
             detail="O endpoint SHAP é específico para modelos de árvore (XGBoost, LightGBM, CatBoost)."
         )
         
    X, _ = load_training_data(version)
    if X is None:
        raise HTTPException(status_code=404, detail="Não foi possível carregar os dados de treino para gerar o SHAP.")
        
    try:
        # Pega uma amostra para o SHAP não demorar muito (máx 500 linhas)
        X_sample = X.sample(n=min(500, len(X)), random_state=42)
        response_data = ExplainService.explain_tree_model(model, X_sample, max_display=max_display)
        return ExplainResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/boundary/{version}", response_model=ExplainResponse, summary="Gera Fronteira de Decisão 2D")
async def get_decision_boundary(version: str):
    """
    Retorna o base64 de um plot da Fronteira de Decisão usando PCA.
    Fundamental para entender hiperplanos de SVMs.
    """
    model_data = load_model(version)
    if not isinstance(model_data, dict) or "model" not in model_data:
        raise HTTPException(status_code=400, detail="Formato de modelo salvo inválido.")
        
    model = model_data["model"]
    
    X, y = load_training_data(version)
    if X is None:
        raise HTTPException(status_code=404, detail="Não foi possível carregar os dados de treino para gerar a fronteira.")
        
    try:
        # Pega uma amostra de 1000 linhas p/ renderização rápida
        import pandas as pd
        
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        if isinstance(y, pd.Series):
             y_sample = y.loc[X_sample.index]
        else:
             y_sample = y[:sample_size]
             
        response_data = ExplainService.explain_svm_boundary(model, X_sample, y_sample)
        return ExplainResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

@router.get("/coefficients/{version}", response_model=ExplainResponse, summary="Gera o Peso dos Coeficientes")
async def get_linear_coefficients(version: str):
    """
    Retorna o gráfico de barras dos coeficientes de modelos Lineares Positivos/Negativos.
    """
    model_data = load_model(version)
    if not isinstance(model_data, dict) or "model" not in model_data:
        raise HTTPException(status_code=400, detail="Formato de modelo salvo inválido.")
        
    model = model_data["model"]
    
    model_type = model.__class__.__name__.lower()
    if "logisticregression" not in model_type:
         raise HTTPException(
             status_code=400, 
             detail="O endpoint de coeficientes é apenas para Regressão Logística e modelos Lineares."
         )
         
    features = model_data.get("features", [])
    if not features:
        raise HTTPException(status_code=400, detail="Não foi possível encontrar a lista de features no modelo salvo.")

    try:
        response_data = ExplainService.explain_linear_model(model, features)
        return ExplainResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
