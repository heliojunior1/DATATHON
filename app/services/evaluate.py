"""Shim de compatibilidade — use app.services.training.evaluate."""
from app.services.training.evaluate import *  # noqa: F401, F403
from app.services.training.evaluate import (
    calculate_metrics, get_classification_report, get_confusion_matrix,
    get_feature_importance, log_evaluation_results,
    cross_validate_model, generate_learning_curves,
)
