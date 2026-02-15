"""
Model Storage — Gerencia múltiplos modelos treinados no disco.

Cada modelo é salvo em seu próprio diretório com ID único.
Mantém um index.json com a lista de todos os modelos.
"""
import json
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

from app.core.config import MODELS_DIR
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# Cache em memória
_model_cache: dict[str, object] = {}
_metadata_cache: dict[str, dict] = {}

# Caminho do índice
INDEX_PATH = MODELS_DIR / "index.json"


def _generate_model_id(model_type: str) -> str:
    """Gera um ID único baseado no tipo e timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix_map = {
        "xgboost": "xgb",
        "lightgbm": "lgb",
        "logistic_regression": "lr",
        "svm": "svm",
        "stacking": "stack",
        "tabnet": "tab",
    }
    prefix = prefix_map.get(model_type, model_type[:3])
    return f"{prefix}_{ts}"


def _load_index() -> list[dict]:
    """Carrega o índice de modelos do disco."""
    if INDEX_PATH.exists():
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Índice corrompido, recriando...")
            return []
    return []


def _save_index(index: list[dict]) -> None:
    """Salva o índice de modelos no disco."""
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def save_trained_model(
    model: object,
    metadata: dict,
    model_type: str,
    feature_names: list[str],
    X_train: pd.DataFrame,
) -> str:
    """
    Salva um modelo treinado com seus artefatos.

    Args:
        model: Modelo treinado (sklearn-compatible).
        metadata: Metadados (métricas, confusion_matrix, etc.).
        model_type: Tipo do modelo ("xgboost", etc.).
        feature_names: Nomes das features usadas.
        X_train: Dados de treinamento (para referência de drift).

    Returns:
        model_id gerado.
    """
    model_id = _generate_model_id(model_type)
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Salvar modelo
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Enriquecer metadados
    full_metadata = {
        **metadata,
        "model_id": model_id,
        "model_type": model_type,
        "feature_names": feature_names,
        "trained_at": datetime.now().isoformat(),
    }
    meta_path = model_dir / "metadata.joblib"
    joblib.dump(full_metadata, meta_path)

    # Salvar amostra de referência para drift
    sample_size = min(len(X_train), 1000)
    reference_sample = X_train.sample(n=sample_size, random_state=42)
    ref_path = model_dir / "reference.joblib"
    joblib.dump(reference_sample, ref_path)

    # Atualizar índice
    index = _load_index()
    index_entry = {
        "model_id": model_id,
        "model_type": model_type,
        "trained_at": full_metadata["trained_at"],
        "metrics": metadata.get("metrics", {}),
        "feature_count": len(feature_names),
        "n_training_samples": metadata.get("n_training_samples", 0),
    }
    index.append(index_entry)
    _save_index(index)

    # Atualizar cache
    _model_cache[model_id] = model
    _metadata_cache[model_id] = full_metadata

    logger.info(f"Modelo salvo: {model_id} em {model_dir}")
    return model_id


def load_trained_model(model_id: str | None = None) -> tuple[object, dict]:
    """
    Carrega um modelo treinado do disco.

    Args:
        model_id: ID do modelo. Se None, carrega o mais recente.

    Returns:
        Tupla (model, metadata).

    Raises:
        FileNotFoundError: Se o modelo não existe.
    """
    if model_id is None:
        model_id = get_latest_model_id()
        if model_id is None:
            raise FileNotFoundError(
                "Nenhum modelo treinado encontrado. "
                "Execute o treinamento primeiro."
            )

    # Cache hit
    if model_id in _model_cache and model_id in _metadata_cache:
        return _model_cache[model_id], _metadata_cache[model_id]

    model_dir = MODELS_DIR / model_id
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo '{model_id}' não encontrado em {model_dir}")

    model = joblib.load(model_path)
    metadata = joblib.load(meta_path) if meta_path.exists() else {}

    # Atualizar cache
    _model_cache[model_id] = model
    _metadata_cache[model_id] = metadata

    logger.info(f"Modelo carregado: {model_id}")
    return model, metadata


def list_trained_models() -> list[dict]:
    """
    Lista todos os modelos treinados.

    Returns:
        Lista de dicts ordenada por data (mais recente primeiro).
    """
    index = _load_index()
    # Ordenar por data (mais recente primeiro)
    index.sort(key=lambda x: x.get("trained_at", ""), reverse=True)
    return index


def get_latest_model_id() -> str | None:
    """Retorna o model_id mais recente ou None."""
    models = list_trained_models()
    return models[0]["model_id"] if models else None


def delete_model(model_id: str) -> bool:
    """
    Deleta um modelo treinado.

    Args:
        model_id: ID do modelo a deletar.

    Returns:
        True se deletado com sucesso.
    """
    import shutil

    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        return False

    shutil.rmtree(model_dir)

    # Remover do índice
    index = _load_index()
    index = [m for m in index if m["model_id"] != model_id]
    _save_index(index)

    # Limpar cache
    _model_cache.pop(model_id, None)
    _metadata_cache.pop(model_id, None)

    logger.info(f"Modelo deletado: {model_id}")
    return True


def clear_cache() -> None:
    """Limpa o cache de modelos em memória."""
    _model_cache.clear()
    _metadata_cache.clear()
    logger.info("Cache de modelos limpo")


def load_reference_data(model_id: str | None = None) -> pd.DataFrame | None:
    """
    Carrega a distribuição de referência para drift detection.

    Args:
        model_id: ID do modelo. Se None, usa o mais recente.

    Returns:
        DataFrame com amostra de referência ou None.
    """
    if model_id is None:
        model_id = get_latest_model_id()
        if model_id is None:
            return None

    ref_path = MODELS_DIR / model_id / "reference.joblib"
    if ref_path.exists():
        return joblib.load(ref_path)
    return None
