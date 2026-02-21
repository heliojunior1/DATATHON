"""Shim de compatibilidade — use app.services.ml.feature_engineering."""
from app.services.ml.feature_engineering import *  # noqa: F401, F403
from app.services.ml.feature_engineering import (
    create_derived_features, select_features, run_feature_engineering,
)
