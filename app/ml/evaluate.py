"""
Módulo de avaliação de modelos.

Calcula métricas de desempenho e gera relatórios de avaliação.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, learning_curve, cross_val_predict
from sklearn.base import clone

from app.core.config import RANDOM_STATE, CV_FOLDS, MODELS_DIR
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


def cross_validate_model(
    model, X: pd.DataFrame, y: pd.Series, cv_folds: int = CV_FOLDS
) -> dict:
    """
    Realiza K-Fold Cross-Validation independente para validar o modelo.

    Diferente do CV usado no RandomizedSearchCV (que otimiza hiperparâmetros),
    esta função avalia o modelo final treinado para garantir que não há overfitting.

    Args:
        model: Modelo treinado (será clonado para cada fold).
        X: Features completas.
        y: Target completo.
        cv_folds: Número de folds (padrão: config.CV_FOLDS).

    Returns:
        Dicionário com métricas {metric: {mean, std, folds: [v1, v2, ...]}}.
    """
    logger.info("=" * 60)
    logger.info(f"CROSS-VALIDATION INDEPENDENTE ({cv_folds}-Fold)")
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    metric_names = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    fold_results = {m: [] for m in metric_names}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Clonar e treinar modelo do zero em cada fold
        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)

        y_pred = fold_model.predict(X_val_fold)
        y_proba = fold_model.predict_proba(X_val_fold)[:, 1]

        fold_metrics = calculate_metrics(y_val_fold, y_pred, y_proba)
        for m in metric_names:
            fold_results[m].append(fold_metrics.get(m, 0.0))

        logger.info(
            f"  Fold {fold_idx}: "
            f"F1={fold_metrics['f1_score']:.4f}  "
            f"AUC={fold_metrics.get('auc_roc', 0):.4f}  "
            f"Acc={fold_metrics['accuracy']:.4f}"
        )

    # Calcular média e desvio padrão
    cv_results = {}
    for m in metric_names:
        values = fold_results[m]
        cv_results[m] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": [float(v) for v in values],
        }

    logger.info("-" * 60)
    for m in metric_names:
        r = cv_results[m]
        logger.info(f"  {m:15s}: {r['mean']:.4f} ± {r['std']:.4f}")
    logger.info("=" * 60)

    return cv_results


def generate_learning_curves(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
    n_points: int = 10,
    output_path: Path | None = None,
) -> str:
    """
    Gera gráfico de learning curves para diagnóstico de overfitting/underfitting.

    O gráfico mostra o score de treino e validação em função do tamanho do dataset.
    - Se ambas as curvas convergem alto → bom modelo.
    - Se treino alto e validação baixo → overfitting.
    - Se ambas baixas → underfitting.

    Args:
        model: Modelo treinado (será clonado internamente).
        X: Features completas.
        y: Target completo.
        cv_folds: Número de folds para CV.
        n_points: Número de pontos na curva.
        output_path: Caminho de saída do PNG (padrão: models/learning_curves.png).

    Returns:
        Caminho do arquivo PNG gerado.
    """
    import matplotlib
    matplotlib.use("Agg")  # Backend não-interativo
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = MODELS_DIR / "learning_curves.png"

    logger.info("Gerando learning curves...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    train_sizes = np.linspace(0.1, 1.0, n_points)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        clone(model),
        X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Criar gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    # Cores
    train_color = "#58a6ff"
    val_color = "#f78166"

    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color=train_color,
    )
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color=val_color,
    )
    ax.plot(
        train_sizes_abs, train_mean, "o-",
        color=train_color, linewidth=2, markersize=5,
        label=f"Treino (final: {train_mean[-1]:.3f})",
    )
    ax.plot(
        train_sizes_abs, val_mean, "o-",
        color=val_color, linewidth=2, markersize=5,
        label=f"Validação (final: {val_mean[-1]:.3f})",
    )

    # Gap de overfitting
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.05:
        diagnosis = f"⚠️ Gap treino-validação: {gap:.3f} (possível overfitting)"
    elif val_mean[-1] < 0.7:
        diagnosis = "⚠️ Score baixo: possível underfitting"
    else:
        diagnosis = f"✅ Modelo saudável (gap: {gap:.3f})"

    ax.set_title(
        f"Learning Curves — F1 Score\n{diagnosis}",
        fontsize=14, fontweight="bold", color="white", pad=15,
    )
    ax.set_xlabel("Amostras de Treinamento", fontsize=12, color="white")
    ax.set_ylabel("F1 Score", fontsize=12, color="white")
    ax.legend(loc="lower right", fontsize=11, facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info(f"Learning curves salvas em: {output_path}")
    logger.info(f"  Diagnóstico: {diagnosis}")

    return str(output_path)

