"""
Módulo de treinamento de modelos.

Pipeline completa: pré-processamento → feature engineering → treinamento → avaliação → serialização.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier

from app.core.config import (
    MODEL_PATH,
    TRAIN_METADATA_PATH,
    REFERENCE_DIST_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    MODEL_NAME,
    MODEL_VERSION,
    INCLUDE_IAN,
)
from app.ml.preprocessing import preprocess_dataset
from app.ml.feature_engineering import run_feature_engineering, select_features
from app.ml.evaluate import (
    calculate_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_feature_importance,
    log_evaluation_results,
    cross_validate_model,
    generate_learning_curves,
)
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


def get_xgb_param_grid() -> dict:
    """Retorna o grid de hiperparâmetros para RandomizedSearchCV."""
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [1, 1.5, 2],
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    optimize: bool = True,
    n_iter: int = 50,
) -> XGBClassifier:
    """
    Treina o modelo XGBoost com otimização de hiperparâmetros.

    Args:
        X_train: Features de treinamento.
        y_train: Target de treinamento.
        optimize: Se True, realiza RandomizedSearchCV.
        n_iter: Número de iterações para RandomizedSearchCV.

    Returns:
        Modelo XGBoost treinado.
    """
    # Calcular scale_pos_weight para balanceamento
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    logger.info(f"Scale pos weight: {scale_pos_weight:.2f} (neg={n_negative}, pos={n_positive})")

    base_model = XGBClassifier(
        # Regularização para evitar overfitting (train score = 1.0 sem regularização)
        max_depth=4,              # Limitar profundidade (padrão: 6)
        min_child_weight=5,       # Mínimo de amostras por folha
        subsample=0.8,            # Usar 80% das amostras por árvore
        colsample_bytree=0.8,     # Usar 80% das features por árvore
        reg_alpha=0.1,            # Regularização L1
        reg_lambda=1.0,           # Regularização L2
        learning_rate=0.1,        # Taxa de aprendizado
        n_estimators=200,         # Mais árvores com learning_rate menor
        # Balanceamento e geral
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )

    if optimize:
        logger.info(f"Iniciando RandomizedSearchCV com {n_iter} iterações e {CV_FOLDS} folds...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=get_xgb_param_grid(),
            n_iter=n_iter,
            cv=cv,
            scoring="f1",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        logger.info(f"Melhor F1 (CV): {search.best_score_:.4f}")
        logger.info(f"Melhores parâmetros: {search.best_params_}")

        model = search.best_estimator_
    else:
        logger.info("Treinando modelo com parâmetros padrão...")
        base_model.fit(X_train, y_train)
        model = base_model

    return model


def save_model_artifacts(
    model: XGBClassifier,
    feature_names: list[str],
    metrics: dict,
    confusion: dict,
    feature_importance: list[dict],
    X_train: pd.DataFrame,
    cv_results: dict | None = None,
    learning_curve_path: str | None = None,
) -> None:
    """
    Salva o modelo treinado e metadados.

    Args:
        model: Modelo treinado.
        feature_names: Nomes das features.
        metrics: Métricas de avaliação.
        confusion: Confusion matrix.
        feature_importance: Ranking de features.
        X_train: Dados de treinamento (para distribuição de referência).
        cv_results: Resultados do K-Fold CV (opcional).
        learning_curve_path: Caminho do gráfico de learning curves (opcional).
    """
    # Salvar modelo
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Modelo salvo em: {MODEL_PATH}")

    # Salvar metadados
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_names": feature_names,
        "metrics": metrics,
        "confusion_matrix": confusion,
        "feature_importance": feature_importance,
        "n_training_samples": len(X_train),
        "cv_results": cv_results,
        "learning_curve_path": learning_curve_path,
    }
    joblib.dump(metadata, TRAIN_METADATA_PATH)
    logger.info(f"Metadados salvos em: {TRAIN_METADATA_PATH}")

    # Salvar distribuição de referência (para monitoramento de drift)
    # Salvar uma amostra dos dados de treino para comparação de drift real (KS-test)
    sample_size = min(len(X_train), 1000)
    reference_sample = X_train.sample(n=sample_size, random_state=42)
    joblib.dump(reference_sample, REFERENCE_DIST_PATH)
    logger.info(f"Amostra de referência ({sample_size} registros) salva em: {REFERENCE_DIST_PATH}")


def run_training_pipeline(
    filepath: str | Path | None = None,
    include_ian: bool | None = None,
    optimize: bool = True,
    n_iter: int = 50,
    run_cv: bool = True,
    run_learning_curves: bool = True,
) -> dict:
    """
    Executa a pipeline completa de treinamento.

    Args:
        filepath: Caminho do dataset.
        include_ian: Se True, inclui a feature IAN. Se None, usa config.INCLUDE_IAN.
        optimize: Se True, faz busca de hiperparâmetros.
        n_iter: Número de iterações para busca.
        run_cv: Se True, executa K-Fold CV independente.
        run_learning_curves: Se True, gera gráfico de learning curves.

    Returns:
        Dicionário com métricas e resultados.
    """
    if include_ian is None:
        include_ian = INCLUDE_IAN
    logger.info("=" * 70)
    logger.info("  PIPELINE DE TREINAMENTO - Datathon Passos Mágicos  ")
    logger.info("=" * 70)

    # 1. Pré-processamento
    df = preprocess_dataset(filepath)

    # 2. Feature engineering
    df = run_feature_engineering(df)

    # 3. Seleção de features
    X, y = select_features(df, include_ian=include_ian)
    feature_names = list(X.columns)

    # 4. Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Split: treino={len(X_train)}, teste={len(X_test)}")
    logger.info(f"Proporção treino — risco: {y_train.mean():.2%}")
    logger.info(f"Proporção teste  — risco: {y_test.mean():.2%}")

    # 5. Treinamento
    model = train_model(X_train, y_train, optimize=optimize, n_iter=n_iter)

    # 6. Avaliação
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_proba)
    report = get_classification_report(y_test, y_pred)
    confusion = get_confusion_matrix(y_test, y_pred)
    importance = get_feature_importance(model, feature_names)

    log_evaluation_results(metrics, report, confusion)

    logger.info("\nTop 10 Features Mais Importantes:")
    for i, feat in enumerate(importance[:10], 1):
        logger.info(f"  {i:2d}. {feat['feature']:30s} — {feat['importance']:.4f}")

    # 6.1 Cross-Validation independente
    cv_results = None
    if run_cv:
        cv_results = cross_validate_model(model, X, y)

    # 6.2 Learning Curves
    learning_curve_path = None
    if run_learning_curves:
        learning_curve_path = generate_learning_curves(model, X, y)

    # 7. Salvar artefatos
    save_model_artifacts(
        model, feature_names, metrics, confusion, importance, X_train,
        cv_results=cv_results,
        learning_curve_path=learning_curve_path,
    )

    results = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "metrics": metrics,
        "confusion_matrix": confusion,
        "feature_importance": importance,
        "feature_names": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_results": cv_results,
        "learning_curve_path": learning_curve_path,
    }

    logger.info("=" * 70)
    logger.info("  TREINAMENTO CONCLUÍDO COM SUCESSO  ")
    logger.info("=" * 70)

    return results
