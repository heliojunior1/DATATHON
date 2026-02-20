"""
Model Registry — Factory pattern para criação de modelos.

Centraliza a configuração de todos os tipos de modelo suportados.
"""
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import RANDOM_STATE
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
    "catboost": {
        "name": "CatBoost",
        "description": "Gradient Boosting com codificação categórica nativa — robusto a NaN e overfitting",
        "supports_nan": True,
        "default_params": {
            "depth": 4,
            "iterations": 200,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "border_count": 128,
            "eval_metric": "Logloss",
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "allow_writing_files": False,
        },
        "param_grid": {
            "iterations": [100, 200, 300, 500],
            "depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 7, 10],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "random_strength": [0.5, 1.0, 2.0, 5.0],
            "bagging_temperature": [0, 0.5, 1.0, 2.0],
            "border_count": [32, 64, 128, 254],
        },
    },
    "lightgbm": {
        "name": "LightGBM",
        "description": "Gradient Boosting leve e eficiente — rápido para datasets grandes",
        "supports_nan": True,
        "default_params": {
            "max_depth": 4,
            "num_leaves": 31,
            "n_estimators": 200,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "verbose": -1,
        },
        "param_grid": {
            "n_estimators": [100, 200, 300, 500],
            "num_leaves": [15, 31, 50, 70],
            "max_depth": [3, 4, 5, 6, 7, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [5, 10, 20, 50],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 1, 2, 5],
        },
    },
    "tabpfn": {
        "name": "TabPFN",
        "description": "Transformer pré-treinado para dados tabulares — ideal para datasets pequenos (<1k amostras, <=100 features)",
        "supports_nan": False,
        "supports_scale_pos_weight": False,
        "supports_hyperparam_search": False,
        "default_params": {
            "device": "cpu",
            "N_ensemble_configurations": 8,
            "seed": 42,
        },
        "param_grid": {},  # TabPFN é pré-treinado, sem busca de hiperparâmetros
    },
    "logistic_regression": {
        "name": "Regressão Logística",
        "description": "Modelo linear clássico — rápido, interpretável e bom baseline",
        "supports_nan": False,
        "supports_scale_pos_weight": False,
        "default_params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "param_grid": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["lbfgs", "liblinear", "saga"],
            "max_iter": [500, 1000, 2000],
        },
    },
    "svm": {
        "name": "SVM",
        "description": "Support Vector Machine — eficaz em espaços de alta dimensionalidade",
        "supports_nan": False,
        "supports_scale_pos_weight": False,
        "default_params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "param_grid": {
            "svc__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "svc__kernel": ["rbf", "linear", "poly"],
            "svc__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        },
    },
    # Fase 2 — modelos futuros (placeholder)
    # "stacking": { ... },
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


def supports_hyperparam_search(model_type: str) -> bool:
    """Retorna se o modelo suporta busca de hiperparâmetros."""
    cfg = MODEL_REGISTRY.get(model_type, {})
    return cfg.get("supports_hyperparam_search", True)


def supports_scale_pos_weight(model_type: str) -> bool:
    """Retorna se o modelo suporta scale_pos_weight."""
    cfg = MODEL_REGISTRY.get(model_type, {})
    return cfg.get("supports_scale_pos_weight", True)


def create_model(
    model_type: str,
    scale_pos_weight: float = 1.0,
    params: dict | None = None,
) -> object:
    """
    Cria uma instância de modelo com parâmetros padrão + overrides.

    Args:
        model_type: Tipo do modelo ("xgboost", "catboost", etc.).
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
    elif model_type == "catboost":
        model = CatBoostClassifier(
            scale_pos_weight=scale_pos_weight,
            **merged_params,
        )
    elif model_type == "lightgbm":
        model = LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            **merged_params,
        )
    elif model_type == "tabpfn":
        # TabPFN não aceita scale_pos_weight
        model = TabPFNClassifier(**merged_params)
    elif model_type == "logistic_regression":
        # LogisticRegression usa class_weight='balanced' ao invés de scale_pos_weight
        model = LogisticRegression(**merged_params)
    elif model_type == "svm":
        # SVM em Pipeline com StandardScaler para normalização automática
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(**merged_params)),
        ])
    else:
        raise ValueError(f"Factory não implementada para '{model_type}'.")

    logger.info(f"Modelo criado: {cfg['name']} (type={model_type})")
    return model
