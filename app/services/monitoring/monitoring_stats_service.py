"""
Serviço de estatísticas operacionais de monitoramento.

Responsabilidade única: calcular e expor métricas de predições,
latência, throughput e valores ausentes a partir do log em memória.
"""
import numpy as np
from datetime import datetime, timedelta

from app.services.monitoring.prediction_log_store import (
    get_prediction_log,
    get_missing_log_snapshot,
)
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


def get_prediction_stats() -> dict:
    """
    Calcula estatísticas das predições realizadas.

    Returns:
        Dicionário com estatísticas (total, taxa de risco, etc.).
    """
    log = get_prediction_log()

    if not log:
        return {
            "total_predictions": 0,
            "risk_rate": 0.0,
            "avg_probability": 0.0,
        }

    predictions = [entry["prediction"] for entry in log]
    probabilities = [entry["probability"] for entry in log]

    return {
        "total_predictions": len(log),
        "risk_rate": float(np.mean(predictions)),
        "avg_probability": float(np.mean(probabilities)),
        "min_probability": float(np.min(probabilities)),
        "max_probability": float(np.max(probabilities)),
        "std_probability": float(np.std(probabilities)),
        "last_prediction_time": log[-1]["timestamp"],
    }


def get_latency_stats() -> dict:
    """
    Retorna estatísticas de latência de inferência.

    Returns:
        Dicionário com min, max, avg, p50, p95, p99 em milissegundos.
    """
    log = get_prediction_log()
    latencies = [e["latency_ms"] for e in log if e.get("latency_ms") is not None]

    if not latencies:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

    return {
        "count": len(latencies),
        "avg_ms": round(float(np.mean(latencies)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }


def get_throughput(window_minutes: int = 60) -> dict:
    """
    Retorna throughput de requisições na janela de tempo especificada.

    Args:
        window_minutes: Tamanho da janela em minutos (padrão: 60).

    Returns:
        Dicionário com total de requisições e média por minuto.
    """
    cutoff_str = (datetime.now() - timedelta(minutes=window_minutes)).isoformat()
    log = get_prediction_log()
    recent = [e for e in log if e["timestamp"] >= cutoff_str]

    return {
        "window_minutes": window_minutes,
        "requests_in_window": len(recent),
        "avg_per_minute": round(len(recent) / window_minutes, 2) if window_minutes > 0 else 0.0,
    }


def get_missing_values_stats() -> dict:
    """
    Retorna estatísticas de valores ausentes por feature.

    Returns:
        Dicionário {feature: {total, missing, missing_rate}}.
    """
    snapshot = get_missing_log_snapshot()

    return {
        feat: {
            "total": vals["total"],
            "missing": vals["missing"],
            "missing_rate": round(vals["missing"] / vals["total"], 4) if vals["total"] > 0 else 0.0,
        }
        for feat, vals in snapshot.items()
    }
