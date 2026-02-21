"""
Serviço de detecção de concept drift via feedback loop.

Responsabilidade única: registrar outcomes reais, comparar com predições
e detectar degradação de performance ao longo do tempo.
"""
import numpy as np
from datetime import datetime

from app.domain.risk_level import classify_risk
from app.services.monitoring.prediction_log_store import (
    get_recent_entries,
    get_confirmed_entries,
    update_prediction_entry,
)
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


def submit_feedback(prediction_id: str, actual_outcome: int) -> bool:
    """
    Registra o outcome real de uma predição para cálculo de concept drift.

    Args:
        prediction_id: UUID da predição a atualizar.
        actual_outcome: 0 = sem risco real, 1 = ficou defasado.

    Returns:
        True se encontrou e atualizou, False se prediction_id não existe.
    """
    return update_prediction_entry(
        prediction_id,
        actual_outcome=actual_outcome,
        feedback_timestamp=datetime.now().isoformat(),
    )


def get_predictions_for_feedback(limit: int = 50) -> list[dict]:
    """
    Retorna as predições mais recentes para confirmação de outcome.

    Args:
        limit: Número máximo de predições a retornar.

    Returns:
        Lista de predições com campos básicos, ordenada da mais recente para a mais antiga.
    """
    recent = get_recent_entries(limit)
    return [
        {
            "prediction_id": e["prediction_id"],
            "timestamp": e["timestamp"],
            "prediction": e["prediction"],
            "probability": round(e["probability"], 4) if e["probability"] is not None else None,
            "risk_level": classify_risk(e["probability"]) if e["probability"] is not None else "—",
            "actual_outcome": e.get("actual_outcome"),
            "feedback_timestamp": e.get("feedback_timestamp"),
        }
        for e in recent
    ]


def get_concept_drift_stats(window_size: int = 20) -> dict:
    """
    Detecta concept drift comparando F1/Recall entre janelas de predições confirmadas.

    Args:
        window_size: Número de predições por janela.

    Returns:
        Dicionário com status (OK/WARNING/DRIFT_DETECTED), métricas por janela e delta de F1.
    """
    confirmed = get_confirmed_entries()

    if not confirmed:
        return {
            "status": "NO_DATA",
            "confirmed_count": 0,
            "windows": [],
            "latest_f1": None,
            "baseline_f1": None,
            "f1_delta": None,
            "alert_message": None,
            "message": "Nenhum feedback confirmado ainda. Confirme outcomes na aba Concept Drift.",
        }

    windows = []
    for i in range(0, len(confirmed), window_size):
        chunk = confirmed[i:i + window_size]
        if len(chunk) < 5:
            continue

        y_true = [e["actual_outcome"] for e in chunk]
        y_pred = [e["prediction"] for e in chunk]

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(chunk)

        windows.append({
            "label": f"W{len(windows) + 1}",
            "n": len(chunk),
            "f1": round(f1, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "accuracy": round(accuracy, 4),
            "start_time": chunk[0]["timestamp"],
            "end_time": chunk[-1]["timestamp"],
        })

    if not windows:
        return {
            "status": "INSUFFICIENT_DATA",
            "confirmed_count": len(confirmed),
            "windows": [],
            "latest_f1": None,
            "baseline_f1": None,
            "f1_delta": None,
            "alert_message": None,
            "message": f"Necessário pelo menos 5 feedbacks confirmados (atual: {len(confirmed)}).",
        }

    latest_f1 = windows[-1]["f1"]
    baseline_f1 = float(np.mean([w["f1"] for w in windows[:-1]])) if len(windows) >= 2 else latest_f1
    f1_delta = latest_f1 - baseline_f1

    if f1_delta > -0.05:
        status = "OK"
        alert_message = None
    elif f1_delta > -0.10:
        status = "WARNING"
        alert_message = f"F1 caiu {abs(f1_delta) * 100:.1f}% na última janela (atenção recomendada)"
    else:
        status = "DRIFT_DETECTED"
        alert_message = (
            f"Concept drift detectado! F1 caiu {abs(f1_delta) * 100:.1f}% "
            "na última janela (retreinamento recomendado)"
        )

    return {
        "status": status,
        "confirmed_count": len(confirmed),
        "windows": windows,
        "latest_f1": round(latest_f1, 4),
        "baseline_f1": round(baseline_f1, 4),
        "f1_delta": round(f1_delta, 4),
        "alert_message": alert_message,
        "message": None,
    }
