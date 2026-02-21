"""
Serviço de detecção de data drift.

Responsabilidade única: comparar distribuições de features entre dados de
treinamento (referência) e dados de produção usando KS-test, PSI e KL-Divergência.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.config import DRIFT_THRESHOLDS
from app.repositories.model_repository import ModelRepository

_repo = ModelRepository()


def load_reference_data(model_id: str | None = None):
    """Carrega dados de referência via repository (sem acesso direto ao storage)."""
    return _repo.load_reference(model_id)
from app.services.monitoring.prediction_log_store import get_prediction_log
from app.services.monitoring.monitoring_stats_service import get_prediction_stats
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


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
        return float(np.sum((prod_counts - ref_counts) * np.log(prod_counts / ref_counts)))
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
        return round(float(np.sum(prod_dist * np.log(prod_dist / ref_dist))), 6)
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

    if ks_pvalue < DRIFT_THRESHOLDS["ks_pvalue_critical"] or mean_shift > DRIFT_THRESHOLDS["mean_shift_critical"]:
        drift_status = "DRIFT_DETECTED"
    elif ks_pvalue < DRIFT_THRESHOLDS["ks_pvalue_warning"] or mean_shift > DRIFT_THRESHOLDS["mean_shift_warning"]:
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
    log = get_prediction_log()

    if not log:
        return {"status": "NO_DATA", "message": "Nenhuma predição registrada ainda."}

    try:
        reference_sample = load_reference_data(None)
    except Exception:
        reference_sample = None

    if reference_sample is None:
        return {"status": "NO_REFERENCE", "message": "Distribuição de referência não encontrada."}

    if not isinstance(reference_sample, pd.DataFrame):
        return {"status": "ERROR", "message": "Formato de referência incompatível (re-treine o modelo)."}

    feature_values: dict[str, list] = {}
    for entry in log:
        for feat_name, value in entry.get("features", {}).items():
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
