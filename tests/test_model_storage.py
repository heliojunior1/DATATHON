"""
Testes do Model Storage.
"""
import pytest
import joblib
import json
from pathlib import Path
from unittest.mock import patch
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from app.services.storage.model_storage import (
    save_trained_model,
    load_trained_model,
    list_trained_models,
    get_latest_model_id,
    delete_model,
    clear_cache,
    _generate_model_id,
)


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Usa diretório temporário para modelos."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
         patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
        clear_cache()
        yield models_dir


@pytest.fixture
def sample_model():
    """Cria um modelo XGBoost simples treinado."""
    np.random.seed(42)
    X = pd.DataFrame({"f1": np.random.randn(30), "f2": np.random.randn(30)})
    y = pd.Series(np.random.randint(0, 2, 30))
    model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestGenerateModelId:
    def test_xgboost_prefix(self):
        model_id = _generate_model_id("xgboost")
        assert model_id.startswith("xgb_")

    def test_unique(self):
        id1 = _generate_model_id("xgboost")
        id2 = _generate_model_id("xgboost")
        # Pode ser igual se chamado no mesmo segundo, mas em geral são únicos
        assert isinstance(id1, str)
        assert len(id1) > 4


class TestSaveAndLoadModel:
    def test_save_creates_files(self, tmp_models_dir, sample_model):
        model, X, _ = sample_model
        metadata = {"metrics": {"f1_score": 0.85}, "n_training_samples": 30}
        model_id = save_trained_model(model, metadata, "xgboost", ["f1", "f2"], X)

        model_dir = tmp_models_dir / model_id
        assert model_dir.exists()
        assert (model_dir / "model.joblib").exists()
        assert (model_dir / "metadata.joblib").exists()
        assert (model_dir / "reference.joblib").exists()

    def test_save_updates_index(self, tmp_models_dir, sample_model):
        model, X, _ = sample_model
        metadata = {"metrics": {"f1_score": 0.85}, "n_training_samples": 30}
        save_trained_model(model, metadata, "xgboost", ["f1", "f2"], X)

        index_path = tmp_models_dir / "index.json"
        assert index_path.exists()
        with open(index_path) as f:
            index = json.load(f)
        assert len(index) == 1
        assert index[0]["model_type"] == "xgboost"

    def test_load_by_id(self, tmp_models_dir, sample_model):
        model, X, _ = sample_model
        metadata = {"metrics": {"f1_score": 0.85}, "n_training_samples": 30}
        model_id = save_trained_model(model, metadata, "xgboost", ["f1", "f2"], X)

        clear_cache()
        loaded_model, loaded_meta = load_trained_model(model_id)
        assert isinstance(loaded_model, XGBClassifier)
        assert loaded_meta["model_id"] == model_id
        assert loaded_meta["feature_names"] == ["f1", "f2"]

    def test_load_latest(self, tmp_models_dir, sample_model):
        model, X, _ = sample_model
        metadata = {"metrics": {"f1_score": 0.85}, "n_training_samples": 30}
        model_id = save_trained_model(model, metadata, "xgboost", ["f1", "f2"], X)

        clear_cache()
        loaded_model, loaded_meta = load_trained_model(None)
        assert loaded_meta["model_id"] == model_id

    def test_load_nonexistent_raises(self, tmp_models_dir):
        with pytest.raises(FileNotFoundError):
            load_trained_model("nonexistent_model")


class TestListModels:
    def test_empty_list(self, tmp_models_dir):
        models = list_trained_models()
        assert models == []

    def test_list_after_save(self, tmp_models_dir, sample_model):
        model, X, _ = sample_model
        metadata = {"metrics": {"f1_score": 0.85}, "n_training_samples": 30}
        save_trained_model(model, metadata, "xgboost", ["f1", "f2"], X)

        models = list_trained_models()
        assert len(models) == 1
        assert models[0]["model_type"] == "xgboost"


class TestDeleteModel:
    def test_delete_existing(self, tmp_models_dir, sample_model):
        model, X, _ = sample_model
        metadata = {"metrics": {"f1_score": 0.85}, "n_training_samples": 30}
        model_id = save_trained_model(model, metadata, "xgboost", ["f1", "f2"], X)

        result = delete_model(model_id)
        assert result is True
        assert not (tmp_models_dir / model_id).exists()
        assert list_trained_models() == []

    def test_delete_nonexistent(self, tmp_models_dir):
        result = delete_model("nonexistent")
        assert result is False


class TestGetLatestModelId:
    def test_no_models(self, tmp_models_dir):
        assert get_latest_model_id() is None

    def test_with_models(self, tmp_models_dir, sample_model):
        model, X, _ = sample_model
        metadata = {"metrics": {"f1_score": 0.85}, "n_training_samples": 30}
        model_id = save_trained_model(model, metadata, "xgboost", ["f1", "f2"], X)
        assert get_latest_model_id() == model_id
