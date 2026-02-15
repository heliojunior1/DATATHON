"""
Módulo de avaliação de modelos.

Calcula métricas de desempenho e gera relatórios de avaliação.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """
    Calcula todas as métricas de avaliação do modelo.

    Args:
        y_true: Valores reais (0 ou 1).
        y_pred: Valores preditos (0 ou 1).
        y_proba: Probabilidades preditas para a classe positiva (opcional).

    Returns:
        Dicionário com todas as métricas.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auc_roc"] = 0.0
            logger.warning("AUC-ROC não pôde ser calculado (apenas uma classe nos dados)")

    return metrics


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Gera relatório de classificação formatado.

    Args:
        y_true: Valores reais.
        y_pred: Valores preditos.

    Returns:
        String com o relatório de classificação.
    """
    target_names = ["Sem Risco (0)", "Em Risco (1)"]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula e retorna a confusion matrix.

    Returns:
        Dicionário com TN, FP, FN, TP.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def get_feature_importance(model, feature_names: list[str]) -> list[dict]:
    """
    Extrai e ordena a importância das features do modelo.

    Args:
        model: Modelo treinado com atributo feature_importances_.
        feature_names: Lista de nomes das features.

    Returns:
        Lista de dicts {feature, importance} ordenada por importância.
    """
    importances = model.feature_importances_
    importance_list = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(feature_names, importances)
    ]
    importance_list.sort(key=lambda x: x["importance"], reverse=True)
    return importance_list


def log_evaluation_results(metrics: dict, report: str, confusion: dict) -> None:
    """Loga todas as métricas de avaliação."""
    logger.info("=" * 60)
    logger.info("RESULTADOS DA AVALIAÇÃO")
    logger.info("=" * 60)

    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    logger.info("\nClassification Report:")
    logger.info(f"\n{report}")

    logger.info("Confusion Matrix:")
    logger.info(f"  TN={confusion['true_negatives']}, FP={confusion['false_positives']}")
    logger.info(f"  FN={confusion['false_negatives']}, TP={confusion['true_positives']}")
