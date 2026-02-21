"""Shim de compatibilidade — os schemas estão divididos por domínio:

- app.models.prediction   → StudentInput, PredictionResponse, Batch*
- app.models.training     → TrainRequest, TrainResponse, ModelInfo*, Health*
- app.models.monitoring   → DriftResponse, FeedbackRequest, FeedbackResponse
- app.models.feature_store → FeatureStore*
"""
from app.models.prediction import (
    StudentInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from app.models.training import (
    HealthResponse,
    ModelInfoResponse,
    TrainRequest,
    TrainResponse,
    ModelListResponse,
    AvailableFeaturesResponse,
    AvailableModelsResponse,
)
from app.models.monitoring import (
    DriftResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from app.models.feature_store import (
    FeatureStoreStatusResponse,
    FeatureStoreFeaturesResponse,
    FeatureStoreMaterializeResponse,
)

__all__ = [
    "StudentInput", "PredictionResponse", "BatchPredictionRequest", "BatchPredictionResponse",
    "HealthResponse", "ModelInfoResponse", "TrainRequest", "TrainResponse",
    "ModelListResponse", "AvailableFeaturesResponse", "AvailableModelsResponse",
    "DriftResponse", "FeedbackRequest", "FeedbackResponse",
    "FeatureStoreStatusResponse", "FeatureStoreFeaturesResponse", "FeatureStoreMaterializeResponse",
]
