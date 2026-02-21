"""Shim de compatibilidade — use app.services.prediction.predict_service."""
from app.services.prediction.predict_service import *  # noqa: F401, F403
from app.services.prediction.predict_service import (
    load_model, clear_model_cache, prepare_input_features,
    predict, predict_batch, predict_from_store, ensure_feature_order,
)
