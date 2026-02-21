"""Shim de compatibilidade — use app.services.storage.model_storage."""
from app.services.storage.model_storage import *  # noqa: F401, F403
from app.services.storage.model_storage import (
    save_trained_model, load_trained_model, list_trained_models,
    get_latest_model_id, delete_model, clear_cache, load_reference_data,
    _generate_model_id,
)
