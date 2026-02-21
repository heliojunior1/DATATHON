"""
Testes da API FastAPI.
"""
import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from app.main import app
from app.services.training.train_service import train_model
from app.services.ml.feature_engineering import select_features
from app.services.prediction.predict_service import clear_model_cache
from app.services.storage.model_storage import save_trained_model, clear_cache as clear_storage_cache
from app.config import RANDOM_STATE, TEST_SIZE


@pytest.fixture
def client():
    """Retorna um TestClient FastAPI."""
    return TestClient(app)


@pytest.fixture
def trained_model(engineered_data, tmp_path):
    """Treina e salva modelo para testes de API (usando model_storage)."""
    X, y = select_features(engineered_data)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = train_model(X_train, y_train, model_type="xgboost", optimize=False)
    feature_names = list(X_train.columns)

    # Usar model_storage com diretório temporário
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
         patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
        clear_storage_cache()

        metadata = {
            "metrics": {"f1_score": 0.8, "accuracy": 0.85, "precision": 0.75, "recall": 0.9, "auc_roc": 0.88},
            "feature_importance": [{"feature": f, "importance": 0.1} for f in feature_names],
            "n_training_samples": len(X_train),
            "confusion_matrix": {"true_negatives": 30, "false_positives": 3, "false_negatives": 2, "true_positives": 15},
        }
        model_id = save_trained_model(model, metadata, "xgboost", feature_names, X_train)
        yield models_dir, model_id


class TestHealthEndpoint:
    """Testes para GET /health."""

    def test_health_no_model(self, client):
        """Health check sem modelo carregado."""
        clear_model_cache()
        with patch("app.routers.prediction.load_model", side_effect=FileNotFoundError("No model")):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is False

    def test_health_with_model(self, client, trained_model):
        """Health check com modelo carregado."""
        models_dir, model_id = trained_model
        clear_model_cache()
        with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Testes para POST /predict."""

    def test_predict_success(self, client, trained_model, sample_student_input):
        """Predição com dados válidos."""
        models_dir, model_id = trained_model
        clear_model_cache()
        with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            response = client.post("/predict", json=sample_student_input)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert data["prediction"] in [0, 1]

    def test_predict_with_model_id(self, client, trained_model, sample_student_input):
        """Predição especificando model_id."""
        models_dir, model_id = trained_model
        clear_model_cache()
        with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            response = client.post(f"/predict?model_id={model_id}", json=sample_student_input)
            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == model_id

    def test_predict_no_model(self, client, sample_student_input):
        """Predição sem modelo carregado."""
        clear_model_cache()
        with patch("app.routers.prediction.predict", side_effect=FileNotFoundError("No model")):
            response = client.post("/predict", json=sample_student_input)
            assert response.status_code == 503

    def test_predict_invalid_input(self, client, trained_model):
        """Predição com dados inválidos."""
        models_dir, _ = trained_model
        clear_model_cache()
        with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            response = client.post("/predict", json={"invalid": "data"})
            assert response.status_code == 422  # Validation error


class TestBatchPredictEndpoint:
    """Testes para POST /predict/batch."""

    def test_batch_predict_success(self, client, trained_model, sample_student_input):
        """Predição em lote com dados válidos."""
        models_dir, _ = trained_model
        clear_model_cache()
        with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            response = client.post("/predict/batch", json={"students": [sample_student_input, sample_student_input]})
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert len(data["predictions"]) == 2


class TestModelInfoEndpoint:
    """Testes para GET /model-info."""

    def test_model_info_success(self, client, trained_model):
        """Informações do modelo."""
        models_dir, model_id = trained_model
        clear_model_cache()
        with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            response = client.get("/model-info")
            assert response.status_code == 200
            data = response.json()
            assert "model_id" in data
            assert "metrics" in data
            assert "feature_names" in data
            assert "feature_importance" in data

    def test_model_info_no_model(self, client):
        """Info sem modelo carregado."""
        clear_model_cache()
        with patch("app.routers.training.load_model", side_effect=FileNotFoundError):
            response = client.get("/model-info")
            assert response.status_code == 503


class TestFeatureImportanceEndpoint:
    """Testes para GET /feature-importance."""

    def test_feature_importance(self, client, trained_model):
        models_dir, _ = trained_model
        clear_model_cache()
        with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            response = client.get("/feature-importance?top_n=5")
            assert response.status_code == 200
            data = response.json()
            assert "features" in data
            assert len(data["features"]) <= 5


class TestMonitoringEndpoints:
    """Testes para endpoints de monitoramento."""

    def test_drift_endpoint(self, client):
        response = client.get("/monitoring/drift")
        assert response.status_code == 200

    def test_stats_endpoint(self, client):
        response = client.get("/monitoring/stats")
        assert response.status_code == 200


class TestModelsEndpoints:
    """Testes para endpoints de modelos."""

    def test_list_models_empty(self, client, tmp_path):
        models_dir = tmp_path / "models_empty"
        models_dir.mkdir()
        with patch("app.services.storage.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.storage.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            response = client.get("/models")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0

    def test_available_models(self, client):
        response = client.get("/models/available")
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) >= 1
        assert any(m["type"] == "xgboost" for m in data["models"])

    def test_available_features(self, client):
        response = client.get("/features/available")
        assert response.status_code == 200
        data = response.json()
        assert len(data["features"]) > 0
        assert all("name" in f and "category" in f for f in data["features"])
