"""
Módulo de monitoramento de data drift.

Compara distribuições de features entre dados de treinamento (referência)
e dados de produção para detectar drift.
"""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from scipy import stats as scipy_stats
from pathlib import Path

from app.core.config import REFERENCE_DIST_PATH
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# Log de predições para monitoramento
_prediction_log: list[dict] = []
MAX_LOG_SIZE = 10000


def log_prediction(input_features: dict, prediction: dict) -> None:
    """
    Registra uma predição para monitoramento posterior.

    Args:
        input_features: Features de entrada.
        prediction: Resultado da predição.
    """
    global _prediction_log

    entry = {
        "timestamp": datetime.now().isoformat(),
        "features": input_features,
        "prediction": prediction.get("prediction"),
        "probability": prediction.get("probability"),
    }
    _prediction_log.append(entry)

    # Limitar tamanho do log
    if len(_prediction_log) > MAX_LOG_SIZE:
        _prediction_log = _prediction_log[-MAX_LOG_SIZE:]


def get_prediction_log() -> list[dict]:
    """Retorna o log de predições."""
    return _prediction_log.copy()


def get_prediction_stats() -> dict:
    """
    Calcula estatísticas das predições realizadas.

    Returns:
        Dicionário com estatísticas (total, taxa de risco, etc.).
    """
    if not _prediction_log:
        return {
            "total_predictions": 0,
            "risk_rate": 0.0,
            "avg_probability": 0.0,
        }

    predictions = [entry["prediction"] for entry in _prediction_log]
    probabilities = [entry["probability"] for entry in _prediction_log]

    return {
        "total_predictions": len(_prediction_log),
        "risk_rate": float(np.mean(predictions)),
        "avg_probability": float(np.mean(probabilities)),
        "min_probability": float(np.min(probabilities)),
        "max_probability": float(np.max(probabilities)),
        "std_probability": float(np.std(probabilities)),
        "last_prediction_time": _prediction_log[-1]["timestamp"],
    }


def calculate_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """
    Calcula o Population Stability Index (PSI).

    PSI < 0.1  → Sem drift significativo
    PSI 0.1-0.2 → Drift moderado
    PSI > 0.2  → Drift significativo

    Args:
        reference: Distribuição de referência (treinamento).
        production: Distribuição de produção.
        bins: Número de bins para discretização.

    Returns:
        Valor PSI.
    """
    eps = 1e-6

    # Criar bins baseados na distribuição de referência
    breakpoints = np.linspace(
        min(np.min(reference), np.min(production)),
        max(np.max(reference), np.max(production)),
        bins + 1,
    )

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
    prod_counts = np.histogram(production, bins=breakpoints)[0] / len(production) + eps

    psi = np.sum((prod_counts - ref_counts) * np.log(prod_counts / ref_counts))
    return float(psi)


def check_drift(feature_name: str, production_values: np.ndarray) -> dict:
    """
    Verifica drift para uma feature específica.

    Args:
        feature_name: Nome da feature.
        production_values: Valores de produção.

    Returns:
        Dicionário com resultados do teste de drift.
    """
    # Carregar distribuição de referência
    if not REFERENCE_DIST_PATH.exists():
        return {"error": "Distribuição de referência não encontrada. Treine o modelo primeiro."}

    reference_stats = joblib.load(REFERENCE_DIST_PATH)

    if feature_name not in reference_stats:
        return {"error": f"Feature '{feature_name}' não encontrada na referência."}

    ref = reference_stats[feature_name]
    prod_values = np.array(production_values, dtype=float)
    prod_values = prod_values[~np.isnan(prod_values)]

    if len(prod_values) < 5:
        return {"warning": "Poucos dados de produção para análise de drift."}

    # Statisticas de produção
    prod_stats = {
        "mean": float(np.mean(prod_values)),
        "std": float(np.std(prod_values)),
        "min": float(np.min(prod_values)),
        "max": float(np.max(prod_values)),
    }

    # Teste KS (Kolmogorov-Smirnov) simulado contra distribuição normal de referência
    ref_simulated = np.random.normal(ref["mean"], max(ref["std"], 1e-6), size=1000)
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(ref_simulated, prod_values)

    # Desvio da média
    mean_shift = abs(prod_stats["mean"] - ref["mean"]) / max(ref["std"], 1e-6)

    # Classificação do drift
    if ks_pvalue < 0.01 or mean_shift > 2.0:
        drift_status = "DRIFT_DETECTED"
    elif ks_pvalue < 0.05 or mean_shift > 1.0:
        drift_status = "WARNING"
    else:
        drift_status = "OK"

    return {
        "feature": feature_name,
        "drift_status": drift_status,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "mean_shift_sigmas": float(mean_shift),
        "reference_mean": ref["mean"],
        "production_mean": prod_stats["mean"],
        "reference_std": ref["std"],
        "production_std": prod_stats["std"],
    }


def check_all_drift() -> dict:
    """
    Verifica drift para todas as features logadas.

    Returns:
        Dicionário com status de drift por feature.
    """
    if not _prediction_log:
        return {"status": "NO_DATA", "message": "Nenhuma predição registrada ainda."}

    if not REFERENCE_DIST_PATH.exists():
        return {"status": "NO_REFERENCE", "message": "Distribuição de referência não encontrada."}

    reference_stats = joblib.load(REFERENCE_DIST_PATH)

    # Coletar valores por feature dos logs
    feature_values = {}
    for entry in _prediction_log:
        features = entry.get("features", {})
        for feat_name, value in features.items():
            if value is not None and feat_name in reference_stats:
                if feat_name not in feature_values:
                    feature_values[feat_name] = []
                try:
                    feature_values[feat_name].append(float(value))
                except (ValueError, TypeError):
                    pass

    results = {}
    drift_count = 0
    warning_count = 0

    for feat_name, values in feature_values.items():
        if len(values) >= 5:
            result = check_drift(feat_name, np.array(values))
            results[feat_name] = result
            if result.get("drift_status") == "DRIFT_DETECTED":
                drift_count += 1
            elif result.get("drift_status") == "WARNING":
                warning_count += 1

    overall_status = "OK"
    if drift_count > 0:
        overall_status = "DRIFT_DETECTED"
    elif warning_count > 0:
        overall_status = "WARNING"

    return {
        "status": overall_status,
        "total_features_checked": len(results),
        "drift_detected": drift_count,
        "warnings": warning_count,
        "details": results,
        "prediction_stats": get_prediction_stats(),
    }
