"""
Testes da API FastAPI.
"""
import pytest
import joblib
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.main import app
from app.ml.train import train_model
from app.ml.feature_engineering import select_features
from app.ml.predict import clear_model_cache
from app.core.config import RANDOM_STATE, TEST_SIZE


@pytest.fixture
def client():
    """Retorna um TestClient FastAPI."""
    return TestClient(app)


@pytest.fixture
def trained_model(engineered_data, tmp_path):
    """Treina e salva modelo para testes de API."""
    X, y = select_features(engineered_data)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = train_model(X_train, y_train, optimize=False)
    feature_names = list(X_train.columns)

    model_path = tmp_path / "test_model.joblib"
    meta_path = tmp_path / "test_metadata.joblib"
    ref_path = tmp_path / "test_reference.joblib"

    joblib.dump(model, model_path)
    joblib.dump({
        "model_name": "test",
        "model_version": "0.1",
        "feature_names": feature_names,
        "feature_importance": [{"feature": f, "importance": 0.1} for f in feature_names],
        "metrics": {"f1_score": 0.8, "accuracy": 0.85, "precision": 0.75, "recall": 0.9, "auc_roc": 0.88},
        "n_training_samples": len(X_train),
        "confusion_matrix": {"true_negatives": 30, "false_positives": 3, "false_negatives": 2, "true_positives": 15},
    }, meta_path)

    reference_stats = {}
    for col in X_train.columns:
        col_data = X_train[col].dropna()
        if len(col_data) > 0:
            reference_stats[col] = {"mean": float(col_data.mean()), "std": float(col_data.std()), "min": float(col_data.min()), "max": float(col_data.max()), "median": float(col_data.median()), "q25": float(col_data.quantile(0.25)), "q75": float(col_data.quantile(0.75))}
    joblib.dump(reference_stats, ref_path)

    return model_path, meta_path, ref_path


class TestHealthEndpoint:
    """Testes para GET /health."""

    def test_health_no_model(self, client):
        """Health check sem modelo carregado."""
        clear_model_cache()
        with patch("app.api.routes.load_model", side_effect=FileNotFoundError("No model")):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is False

    def test_health_with_model(self, client, trained_model):
        """Health check com modelo carregado."""
        model_path, meta_path, _ = trained_model
        clear_model_cache()
        with patch("app.ml.predict.MODEL_PATH", model_path), \
             patch("app.ml.predict.TRAIN_METADATA_PATH", meta_path):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Testes para POST /predict."""

    def test_predict_success(self, client, trained_model, sample_student_input):
        """Predição com dados válidos."""
        model_path, meta_path, _ = trained_model
        clear_model_cache()
        with patch("app.ml.predict.MODEL_PATH", model_path), \
             patch("app.ml.predict.TRAIN_METADATA_PATH", meta_path):
            response = client.post("/predict", json=sample_student_input)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert data["prediction"] in [0, 1]

    def test_predict_no_model(self, client, sample_student_input):
        """Predição sem modelo carregado."""
        clear_model_cache()
        with patch("app.api.routes.predict", side_effect=FileNotFoundError("No model")):
            response = client.post("/predict", json=sample_student_input)
            assert response.status_code == 503

    def test_predict_invalid_input(self, client, trained_model):
        """Predição com dados inválidos."""
        model_path, meta_path, _ = trained_model
        clear_model_cache()
        with patch("app.ml.predict.MODEL_PATH", model_path), \
             patch("app.ml.predict.TRAIN_METADATA_PATH", meta_path):
            response = client.post("/predict", json={"invalid": "data"})
            assert response.status_code == 422  # Validation error


class TestBatchPredictEndpoint:
    """Testes para POST /predict/batch."""

    def test_batch_predict_success(self, client, trained_model, sample_student_input):
        """Predição em lote com dados válidos."""
        model_path, meta_path, _ = trained_model
        clear_model_cache()
        with patch("app.ml.predict.MODEL_PATH", model_path), \
             patch("app.ml.predict.TRAIN_METADATA_PATH", meta_path):
            response = client.post("/predict/batch", json={"students": [sample_student_input, sample_student_input]})
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert len(data["predictions"]) == 2


class TestModelInfoEndpoint:
    """Testes para GET /model-info."""

    def test_model_info_success(self, client, trained_model):
        """Informações do modelo."""
        model_path, meta_path, _ = trained_model
        clear_model_cache()
        with patch("app.ml.predict.MODEL_PATH", model_path), \
             patch("app.ml.predict.TRAIN_METADATA_PATH", meta_path):
            response = client.get("/model-info")
            assert response.status_code == 200
            data = response.json()
            assert "model_name" in data
            assert "metrics" in data
            assert "feature_names" in data
            assert "feature_importance" in data

    def test_model_info_no_model(self, client):
        """Info sem modelo carregado."""
        clear_model_cache()
        with patch("app.api.routes.load_model", side_effect=FileNotFoundError):
            response = client.get("/model-info")
            assert response.status_code == 503


class TestFeatureImportanceEndpoint:
    """Testes para GET /feature-importance."""

    def test_feature_importance(self, client, trained_model):
        model_path, meta_path, _ = trained_model
        clear_model_cache()
        with patch("app.ml.predict.MODEL_PATH", model_path), \
             patch("app.ml.predict.TRAIN_METADATA_PATH", meta_path):
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
