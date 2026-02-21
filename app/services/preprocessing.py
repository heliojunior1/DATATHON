"""Shim de compatibilidade — use app.services.ml.preprocessing."""
from app.services.ml.preprocessing import *  # noqa: F401, F403
from app.services.ml.preprocessing import (
    load_dataset, create_target_variable, encode_categorical_columns,
    handle_missing_values, preprocess_dataset, _encode_rec_avaliador,
)
