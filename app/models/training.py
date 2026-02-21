"""
Schemas Pydantic para treinamento e informações de modelo.
"""
from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    """Resposta do health check."""
    status: str = "ok"
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Informações sobre o modelo em produção."""
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    model_version: Optional[str] = None
    metrics: dict = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    feature_importance: list[dict] = Field(default_factory=list)
    n_training_samples: int = 0
    confusion_matrix: dict = Field(default_factory=dict)
    cv_results: Optional[dict] = None
    trained_at: Optional[str] = None


class TrainRequest(BaseModel):
    """Request para treinar um novo modelo."""
    model_type: str = Field("xgboost", description="Tipo: 'xgboost' (Fase 2: lightgbm, logistic_regression, svm, stacking, tabnet)")
    features: Optional[list[str]] = Field(None, description="Features a usar (None = todas as padrão)")
    optimize: bool = Field(False, description="Busca de hiperparâmetros")
    n_iter: int = Field(50, ge=10, le=200, description="Iterações para RandomizedSearchCV")
    include_ian: bool = Field(False, description="Incluir IAN (⚠️ data leakage)")
    run_cv: bool = Field(True, description="K-Fold Cross-Validation")
    run_learning_curves: bool = Field(True, description="Gerar learning curves")


class TrainResponse(BaseModel):
    """Resposta do treinamento."""
    model_id: str
    model_type: str
    metrics: dict
    feature_names: list[str]
    n_train: int
    n_test: int
    cv_results: Optional[dict] = None
    message: str


class ModelListResponse(BaseModel):
    """Lista de modelos treinados."""
    models: list[dict]
    total: int


class AvailableFeaturesResponse(BaseModel):
    """Features disponíveis para seleção."""
    features: list[dict]


class AvailableModelsResponse(BaseModel):
    """Tipos de modelo disponíveis para treinamento."""
    models: list[dict]
