"""Shim de compatibilidade — use app.services.training.model_registry."""
from app.services.training.model_registry import *  # noqa: F401, F403
from app.services.training.model_registry import (
    MODEL_REGISTRY, get_available_models, get_param_grid,
    supports_hyperparam_search, supports_scale_pos_weight, create_model,
)
