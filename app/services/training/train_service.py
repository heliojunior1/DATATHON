"""
Módulo de treinamento de modelos.

Pipeline completa: pré-processamento → feature engineering → treinamento → avaliação → serialização.
Suporta múltiplos tipos de modelo via Model Registry.
Usa ModelRepository para persistência (injetável para testes).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from app.config import (
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    INCLUDE_IAN,
    AVAILABLE_FEATURES,
    USE_FEATURE_STORE,
)
from app.services.training.model_registry import create_model, get_param_grid, supports_hyperparam_search, supports_scale_pos_weight
from app.repositories.model_repository import ModelRepository
from app.services.ml.preprocessing import preprocess_dataset
from app.services.ml.feature_engineering import run_feature_engineering, select_features
from app.services.training.evaluate import (
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
    """Wrapper de compatibilidade — delega para model_registry."""
    return get_param_grid("xgboost")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
    optimize: bool = True,
    n_iter: int = 50,
) -> object:
    """
    Treina um modelo com otimização opcional de hiperparâmetros.

    Args:
        X_train: Features de treinamento.
        y_train: Target de treinamento.
        model_type: Tipo do modelo (default: "xgboost").
        optimize: Se True, realiza RandomizedSearchCV.
        n_iter: Número de iterações para RandomizedSearchCV.

    Returns:
        Modelo treinado (sklearn-compatible).
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    logger.info(f"Scale pos weight: {scale_pos_weight:.2f} (neg={n_negative}, pos={n_positive})")

    model_kwargs = {}
    if supports_scale_pos_weight(model_type):
        model_kwargs["scale_pos_weight"] = scale_pos_weight

    base_model = create_model(model_type=model_type, **model_kwargs)

    if optimize and supports_hyperparam_search(model_type):
        logger.info(f"Iniciando RandomizedSearchCV com {n_iter} iterações e {CV_FOLDS} folds...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        param_grid = get_param_grid(model_type)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
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
        logger.info(f"Treinando modelo {model_type} com parâmetros padrão...")
        base_model.fit(X_train, y_train)
        model = base_model

    return model


def run_training_pipeline(
    filepath: str | Path | None = None,
    model_type: str = "xgboost",
    selected_features: list[str] | None = None,
    include_ian: bool | None = None,
    optimize: bool = True,
    n_iter: int = 50,
    run_cv: bool = True,
    run_learning_curves: bool = True,
    use_feature_store: bool | None = None,
    repo: ModelRepository | None = None,
) -> dict:
    """
    Executa a pipeline completa de treinamento.

    Args:
        filepath: Caminho do dataset.
        model_type: Tipo do modelo ("xgboost", etc.).
        selected_features: Lista de features a usar. Se None, usa todas.
        include_ian: Se True, inclui a feature IAN. Se None, usa config.INCLUDE_IAN.
        optimize: Se True, faz busca de hiperparâmetros.
        n_iter: Número de iterações para busca.
        run_cv: Se True, executa K-Fold CV independente.
        run_learning_curves: Se True, gera gráfico de learning curves.
        use_feature_store: Se True, ingere features no Feature Store.
                          Se None, usa config.USE_FEATURE_STORE.
        repo: Repository para persistir o modelo. Se None, usa uma instância padrão.
              Passe um mock em testes para evitar acesso a disco.

    Returns:
        Dicionário com métricas, model_id e resultados.
    """
    _repo = repo if repo is not None else ModelRepository()

    if use_feature_store is None:
        use_feature_store = USE_FEATURE_STORE
    if include_ian is None:
        include_ian = INCLUDE_IAN

    logger.info("=" * 70)
    logger.info("  PIPELINE DE TREINAMENTO - Datathon Passos Mágicos  ")
    logger.info(f"  Modelo: {model_type}")
    logger.info("=" * 70)

    # 1. Pré-processamento
    df = preprocess_dataset(filepath)

    # 2. Feature engineering
    df = run_feature_engineering(df)

    # 2.1 Feature Store — ingestão e materialização
    if use_feature_store:
        try:
            from feature_store.feature_store_manager import FeatureStoreManager
            fs_manager = FeatureStoreManager()
            fs_manager.ingest_features(df)
            fs_manager.apply()
            fs_manager.materialize()
            logger.info("Features ingeridas e materializadas no Feature Store")
        except Exception as e:
            logger.warning(f"Falha ao usar Feature Store (continuando sem): {e}")

    # 3. Seleção de features
    X, y = select_features(df, include_ian=include_ian)

    # 4. Filtrar features selecionadas pelo usuário (se especificadas)
    if selected_features:
        available = [f for f in selected_features if f in X.columns]
        missing = [f for f in selected_features if f not in X.columns]
        if missing:
            logger.warning(f"Features não encontradas (ignoradas): {missing}")
        if not available:
            raise ValueError("Nenhuma das features selecionadas está disponível no dataset.")
        X = X[available]
        logger.info(f"Features filtradas pelo usuário: {len(available)} de {len(selected_features)}")

    feature_names = list(X.columns)

    # 5. Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Split: treino={len(X_train)}, teste={len(X_test)}")
    logger.info(f"Proporção treino — risco: {y_train.mean():.2%}")
    logger.info(f"Proporção teste  — risco: {y_test.mean():.2%}")

    # 6. Treinamento
    model = train_model(X_train, y_train, model_type=model_type, optimize=optimize, n_iter=n_iter)

    # 7. Avaliação
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_proba)
    report = get_classification_report(y_test, y_pred)
    confusion = get_confusion_matrix(y_test, y_pred)
    importance = get_feature_importance(model, feature_names, X_test, y_test)

    log_evaluation_results(metrics, report, confusion)

    logger.info("\nTop 10 Features Mais Importantes:")
    for i, feat in enumerate(importance[:10], 1):
        logger.info(f"  {i:2d}. {feat['feature']:30s} — {feat['importance']:.4f}")

    # 7.1 Cross-Validation independente
    cv_results = None
    if run_cv:
        cv_results = cross_validate_model(model, X, y)

    # 7.2 Learning Curves
    learning_curve_path = None
    if run_learning_curves:
        learning_curve_path = generate_learning_curves(model, X, y)

    # 8. Persistir artefatos via repository
    metadata = {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "feature_importance": importance,
        "n_training_samples": len(X_train),
        "cv_results": cv_results,
        "learning_curve_path": learning_curve_path,
    }
    model_id = _repo.save(
        model=model,
        metadata=metadata,
        model_type=model_type,
        feature_names=feature_names,
        X_train=X_train,
    )

    results = {
        "model_id": model_id,
        "model_type": model_type,
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
    logger.info(f"  TREINAMENTO CONCLUÍDO — model_id: {model_id}")
    logger.info("=" * 70)

    return results
