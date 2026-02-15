"""
Model Registry — Factory pattern para criação de modelos.

Centraliza a configuração de todos os tipos de modelo suportados.
Fase 1: XGBoost apenas. Fase 2 adicionará LightGBM, LR, SVM, Stacking, TabNet.
"""
from xgboost import XGBClassifier

from app.core.config import RANDOM_STATE
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# ── Registro de Modelos ──────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient Boosting otimizado — robusto a NaN e rápido",
        "supports_nan": True,
        "default_params": {
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
        },
        "param_grid": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2, 0.3],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [1, 1.5, 2],
        },
    },
    # Fase 2 — modelos futuros (placeholder)
    # "lightgbm": { ... },
    # "logistic_regression": { ... },
    # "svm": { ... },
    # "stacking": { ... },
    # "tabnet": { ... },
}


def get_available_models() -> list[dict]:
    """
    Retorna lista de tipos de modelo disponíveis para treinamento.

    Returns:
        Lista de dicts com {type, name, description, supports_nan}.
    """
    return [
        {
            "type": key,
            "name": cfg["name"],
            "description": cfg["description"],
            "supports_nan": cfg["supports_nan"],
        }
        for key, cfg in MODEL_REGISTRY.items()
    ]


def get_param_grid(model_type: str) -> dict:
    """
    Retorna o grid de hiperparâmetros para busca.

    Args:
        model_type: Tipo do modelo registrado.

    Returns:
        Dicionário com distribuições de parâmetros.

    Raises:
        ValueError: Se model_type não está registrado.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_type}' não registrado. "
            f"Disponíveis: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]["param_grid"]


def create_model(
    model_type: str,
    scale_pos_weight: float = 1.0,
    params: dict | None = None,
) -> object:
    """
    Cria uma instância de modelo com parâmetros padrão + overrides.

    Args:
        model_type: Tipo do modelo ("xgboost", etc.).
        scale_pos_weight: Peso para balanceamento de classes.
        params: Overrides aos parâmetros padrão (opcional).

    Returns:
        Instância do modelo (sklearn-compatible).

    Raises:
        ValueError: Se model_type não está registrado.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_type}' não registrado. "
            f"Disponíveis: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[model_type]
    merged_params = {**cfg["default_params"]}
    if params:
        merged_params.update(params)

    if model_type == "xgboost":
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            **merged_params,
        )
    else:
        raise ValueError(f"Factory não implementada para '{model_type}'.")

    logger.info(f"Modelo criado: {cfg['name']} (type={model_type})")
    return model
