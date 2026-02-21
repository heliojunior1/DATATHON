"""
Armazenamento de log de predições em memória.

Responsabilidade única: registrar, limitar e expor o histórico de predições
realizadas durante o ciclo de vida da aplicação, com thread safety.
"""
import uuid
import threading
import numpy as np
from datetime import datetime

from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# Log de predições para monitoramento
_prediction_log: list[dict] = []
MAX_LOG_SIZE = 10000

# Rastreamento de valores ausentes por feature
_missing_log: dict[str, dict] = {}

# Lock para thread safety
_lock = threading.Lock()


def log_prediction(input_features: dict, prediction: dict, latency_ms: float | None = None) -> str:
    """
    Registra uma predição para monitoramento posterior.

    Args:
        input_features: Features de entrada.
        prediction: Resultado da predição.
        latency_ms: Latência de inferência em milissegundos.

    Returns:
        prediction_id: UUID único gerado para esta predição.
    """
    global _prediction_log, _missing_log

    prediction_id = str(uuid.uuid4())

    entry = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now().isoformat(),
        "features": input_features,
        "prediction": prediction.get("prediction"),
        "probability": prediction.get("probability"),
        "latency_ms": latency_ms,
        "actual_outcome": None,
        "feedback_timestamp": None,
    }

    with _lock:
        _prediction_log.append(entry)

        for feat_name, value in input_features.items():
            if feat_name not in _missing_log:
                _missing_log[feat_name] = {"total": 0, "missing": 0}
            _missing_log[feat_name]["total"] += 1
            is_missing = value is None or (isinstance(value, float) and np.isnan(value))
            if is_missing:
                _missing_log[feat_name]["missing"] += 1

        if len(_prediction_log) > MAX_LOG_SIZE:
            _prediction_log = _prediction_log[-MAX_LOG_SIZE:]

    return prediction_id


def get_prediction_log() -> list[dict]:
    """Retorna uma cópia do log de predições."""
    with _lock:
        return _prediction_log.copy()


def get_missing_log_snapshot() -> dict[str, dict]:
    """Retorna uma cópia thread-safe do log de valores ausentes."""
    with _lock:
        return {k: dict(v) for k, v in _missing_log.items()}


def update_prediction_entry(prediction_id: str, **fields) -> bool:
    """
    Atualiza campos de uma entrada do log pelo prediction_id.

    Args:
        prediction_id: UUID da predição.
        **fields: Campos a atualizar (ex: actual_outcome=1, feedback_timestamp="...").

    Returns:
        True se encontrou e atualizou, False caso contrário.
    """
    with _lock:
        for entry in _prediction_log:
            if entry.get("prediction_id") == prediction_id:
                entry.update(fields)
                return True
    return False


def get_recent_entries(limit: int) -> list[dict]:
    """Retorna as `limit` entradas mais recentes (em ordem reversa)."""
    with _lock:
        return list(reversed(_prediction_log[-limit:]))


def get_confirmed_entries() -> list[dict]:
    """Retorna apenas as entradas com actual_outcome preenchido."""
    with _lock:
        return [e for e in _prediction_log if e.get("actual_outcome") is not None]
