"""
Schemas Pydantic divididos por domínio:
- prediction   → StudentInput, PredictionResponse, Batch*
- training     → TrainRequest, TrainResponse, ModelInfo*, Health*
- monitoring   → DriftResponse, FeedbackRequest, FeedbackResponse
- feature_store → FeatureStore*
"""
from app.models.prediction import (
    StudentInput, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse,
)
from app.models.training import (
    HealthResponse, ModelInfoResponse, TrainRequest, TrainResponse,
    ModelListResponse, AvailableFeaturesResponse, AvailableModelsResponse,
)
from app.models.monitoring import DriftResponse, FeedbackRequest, FeedbackResponse
from app.models.feature_store import (
    FeatureStoreStatusResponse, FeatureStoreFeaturesResponse, FeatureStoreMaterializeResponse,
)
