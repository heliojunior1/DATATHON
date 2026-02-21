"""Shim de compatibilidade — use app.services.training.train_service."""
from app.services.training.train_service import *  # noqa: F401, F403
from app.services.training.train_service import (
    get_xgb_param_grid, train_model, run_training_pipeline,
)
