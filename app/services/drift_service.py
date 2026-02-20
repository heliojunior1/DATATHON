"""
Módulo de monitoramento de data drift.

Compara distribuições de features entre dados de treinamento (referência)
e dados de produção para detectar drift.
"""
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats as scipy_stats

from app.services.model_storage import load_reference_data
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# Log de predições para monitoramento
_prediction_log: list[dict] = []
MAX_LOG_SIZE = 10000

# Rastreamento de valores ausentes por feature
_missing_log: dict[str, dict] = {}

# Lock para thread safety
_lock = threading.Lock()


def log_prediction(input_features: dict, prediction: dict, latency_ms: float | None = None) -> None:
    """
    Registra uma predição para monitoramento posterior.

    Args:
        input_features: Features de entrada.
        prediction: Resultado da predição.
        latency_ms: Latência de inferência em milissegundos.
    """
    global _prediction_log, _missing_log

    entry = {
        "timestamp": datetime.now().isoformat(),
        "features": input_features,
        "prediction": prediction.get("prediction"),
        "probability": prediction.get("probability"),
        "latency_ms": latency_ms,
    }

    with _lock:
        _prediction_log.append(entry)

        # Rastrear valores ausentes por feature
        for feat_name, value in input_features.items():
            if feat_name not in _missing_log:
                _missing_log[feat_name] = {"total": 0, "missing": 0}
            _missing_log[feat_name]["total"] += 1
            is_missing = value is None or (isinstance(value, float) and np.isnan(value))
            if is_missing:
                _missing_log[feat_name]["missing"] += 1

        # Limitar tamanho do log
        if len(_prediction_log) > MAX_LOG_SIZE:
            _prediction_log = _prediction_log[-MAX_LOG_SIZE:]


def get_prediction_log() -> list[dict]:
    """Retorna o log de predições."""
    with _lock:
        return _prediction_log.copy()


def get_prediction_stats() -> dict:
    """
    Calcula estatísticas das predições realizadas.

    Returns:
        Dicionário com estatísticas (total, taxa de risco, etc.).
    """
    with _lock:
        log = _prediction_log.copy()

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


def get_missing_values_stats() -> dict:
    """
    Retorna estatísticas de valores ausentes por feature.

    Returns:
        Dicionário {feature: {total, missing, missing_rate}}.
    """
    with _lock:
        snapshot = {k: dict(v) for k, v in _missing_log.items()}

    return {
        feat: {
            "total": vals["total"],
            "missing": vals["missing"],
            "missing_rate": round(vals["missing"] / vals["total"], 4) if vals["total"] > 0 else 0.0,
        }
        for feat, vals in snapshot.items()
    }


def get_latency_stats() -> dict:
    """
    Retorna estatísticas de latência de inferência.

    Returns:
        Dicionário com min, max, avg, p50, p95, p99 em milissegundos.
    """
    with _lock:
        latencies = [e["latency_ms"] for e in _prediction_log if e.get("latency_ms") is not None]

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
    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    cutoff_str = cutoff.isoformat()

    with _lock:
        recent = [e for e in _prediction_log if e["timestamp"] >= cutoff_str]

    return {
        "window_minutes": window_minutes,
        "requests_in_window": len(recent),
        "avg_per_minute": round(len(recent) / window_minutes, 2) if window_minutes > 0 else 0.0,
    }


def calculate_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """
    Calcula o Population Stability Index (PSI).

    PSI < 0.1  → Sem drift significativo
    PSI 0.1-0.2 → Drift moderado
    PSI > 0.2  → Drift significativo
    """
    eps = 1e-6

    try:
        breakpoints = np.linspace(
            min(np.min(reference), np.min(production)),
            max(np.max(reference), np.max(production)),
            bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
        prod_counts = np.histogram(production, bins=breakpoints)[0] / len(production) + eps

        psi = np.sum((prod_counts - ref_counts) * np.log(prod_counts / ref_counts))
        return float(psi)
    except Exception:
        return 0.0


def calculate_kl_divergence(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """
    Calcula a Divergência de Kullback-Leibler KL(produção || referência).

    KL < 0.1  → Distribuições similares
    KL > 0.5  → Divergência significativa
    """
    eps = 1e-6

    try:
        breakpoints = np.linspace(
            min(np.min(reference), np.min(production)),
            max(np.max(reference), np.max(production)),
            bins + 1,
        )

        ref_dist = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
        prod_dist = np.histogram(production, bins=breakpoints)[0] / len(production) + eps
        ref_dist /= ref_dist.sum()
        prod_dist /= prod_dist.sum()

        kl = float(np.sum(prod_dist * np.log(prod_dist / ref_dist)))
        return round(kl, 6)
    except Exception:
        return 0.0


def check_drift(feature_name: str, production_values: np.ndarray, reference_sample: pd.DataFrame) -> dict:
    """
    Verifica drift para uma feature específica usando KS-test contra dados reais.

    Args:
        feature_name: Nome da feature.
        production_values: Valores de produção.
        reference_sample: DataFrame com amostra de treino.

    Returns:
        Dicionário com resultados do teste de drift (KS-test, PSI, KL Divergence).
    """
    if feature_name not in reference_sample.columns:
        return {"error": f"Feature '{feature_name}' não encontrada na referência."}

    ref_values = reference_sample[feature_name].dropna().values
    prod_values = np.array(production_values, dtype=float)
    prod_values = prod_values[~np.isnan(prod_values)]

    if len(prod_values) < 5:
        return {"warning": "Poucos dados de produção para análise de drift."}

    ref_mean = float(np.mean(ref_values))
    ref_std = float(np.std(ref_values))
    prod_mean = float(np.mean(prod_values))

    ks_stat, ks_pvalue = scipy_stats.ks_2samp(ref_values, prod_values)

    div = ref_std if ref_std > 1e-6 else 1.0
    mean_shift = abs(prod_mean - ref_mean) / div

    psi = calculate_psi(ref_values, prod_values)
    kl_divergence = calculate_kl_divergence(ref_values, prod_values)

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
        "mean_shift": float(mean_shift),
        "reference_mean": ref_mean,
        "production_mean": prod_mean,
        "psi": round(psi, 6),
        "psi_status": "OK" if psi < 0.1 else ("WARNING" if psi < 0.2 else "DRIFT_DETECTED"),
        "kl_divergence": kl_divergence,
    }


def check_all_drift() -> dict:
    """
    Verifica drift para todas as features logadas.
    Usa model_storage para carregar a referência do modelo mais recente.

    Returns:
        Dicionário com status de drift por feature.
    """
    with _lock:
        log = _prediction_log.copy()

    if not log:
        return {"status": "NO_DATA", "message": "Nenhuma predição registrada ainda."}

    # Carregar referência do modelo mais recente via model_storage
    try:
        reference_sample = load_reference_data(None)
    except Exception:
        reference_sample = None

    if reference_sample is None:
        return {"status": "NO_REFERENCE", "message": "Distribuição de referência não encontrada."}

    if not isinstance(reference_sample, pd.DataFrame):
        return {"status": "ERROR", "message": "Formato de referência incompatível (re-treine o modelo)."}

    # Coletar valores por feature dos logs
    feature_values = {}
    for entry in log:
        features = entry.get("features", {})
        for feat_name, value in features.items():
            if value is not None and feat_name in reference_sample.columns:
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
            result = check_drift(feat_name, np.array(values), reference_sample)
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
