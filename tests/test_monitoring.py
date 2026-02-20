"""
Testes adicionais para aumentar cobertura.

Cobrindo: monitoring/drift.py, ml/model_storage (save), e ml/preprocessing.py (preprocess_dataset).
"""
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import patch

from app.services.drift_service import (
    log_prediction,
    get_prediction_log,
    get_prediction_stats,
    calculate_psi,
    check_drift,
    check_all_drift,
    _prediction_log,
)
from app.services.model_storage import save_trained_model, clear_cache as clear_storage_cache
from app.services.preprocessing import preprocess_dataset, load_dataset
from app.services.evaluate import log_evaluation_results


class TestDriftMonitoring:
    """Testes para o módulo de monitoramento de drift."""

    def setup_method(self):
        """Limpa o log antes de cada teste."""
        _prediction_log.clear()

    def test_log_prediction(self):
        log_prediction({"IAA": 5.0}, {"prediction": 0, "probability": 0.2})
        assert len(get_prediction_log()) == 1

    def test_log_multiple_predictions(self):
        for i in range(10):
            log_prediction({"IAA": float(i)}, {"prediction": i % 2, "probability": 0.5})
        assert len(get_prediction_log()) == 10

    def test_get_prediction_stats_empty(self):
        stats = get_prediction_stats()
        assert stats["total_predictions"] == 0
        assert stats["risk_rate"] == 0.0

    def test_get_prediction_stats_with_data(self):
        log_prediction({"IAA": 5.0}, {"prediction": 0, "probability": 0.2})
        log_prediction({"IAA": 3.0}, {"prediction": 1, "probability": 0.8})
        stats = get_prediction_stats()
        assert stats["total_predictions"] == 2
        assert stats["risk_rate"] == 0.5

    def test_calculate_psi_identical(self):
        ref = np.random.normal(5, 1, 1000)
        psi = calculate_psi(ref, ref)
        assert psi < 0.1

    def test_calculate_psi_different(self):
        ref = np.random.normal(5, 1, 1000)
        prod = np.random.normal(10, 2, 1000)
        psi = calculate_psi(ref, prod)
        assert psi > 0.1

    def test_check_drift_few_values(self):
        df_ref = pd.DataFrame({"IAA": [4.0, 5.0, 6.0, 5.0, 5.0]})
        result = check_drift("IAA", np.array([5.0]), df_ref)
        assert "warning" in result

    def test_check_drift_valid(self):
        df_ref = pd.DataFrame({"IAA": np.random.normal(5, 1, 100).tolist()})
        result = check_drift("IAA", np.random.normal(5, 1, 50), df_ref)
        assert "drift_status" in result
        assert result["drift_status"] in ["OK", "WARNING", "DRIFT_DETECTED"]

    def test_check_drift_wrong_feature(self):
        df_ref = pd.DataFrame({"IAA": [1, 2, 3]})
        result = check_drift("NONEXISTENT", np.array([5.0, 6.0]), df_ref)
        assert "error" in result

    def test_check_all_drift_no_data(self):
        result = check_all_drift()
        assert result["status"] == "NO_DATA"

    def test_check_all_drift_no_reference(self):
        log_prediction({"IAA": 5.0}, {"prediction": 0, "probability": 0.2})
        with patch("app.services.drift_service.load_reference_data", return_value=None):
            result = check_all_drift()
            assert result["status"] == "NO_REFERENCE"


class TestSaveModelArtifacts:
    """Testes para salvamento de artefatos via model_storage."""

    def test_save_model(self, engineered_data, tmp_path):
        from app.services.feature_engineering import select_features
        from app.services.train_service import train_model
        from sklearn.model_selection import train_test_split

        X, y = select_features(engineered_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = train_model(X_train, y_train, model_type="xgboost", optimize=False)

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with patch("app.services.model_storage.MODELS_DIR", models_dir), \
             patch("app.services.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            model_id = save_trained_model(
                model=model,
                metadata={
                    "metrics": {"f1_score": 0.9},
                    "confusion_matrix": {"tn": 10, "fp": 1, "fn": 2, "tp": 8},
                    "feature_importance": [{"feature": "IAA", "importance": 0.5}],
                    "n_training_samples": len(X_train),
                },
                model_type="xgboost",
                feature_names=list(X_train.columns),
                X_train=X_train,
            )

        model_dir = models_dir / model_id
        assert (model_dir / "model.joblib").exists()
        assert (model_dir / "metadata.joblib").exists()
        assert (model_dir / "reference.joblib").exists()

        metadata = joblib.load(model_dir / "metadata.joblib")
        assert "feature_names" in metadata
        assert "metrics" in metadata

        # Verificar se referência é DataFrame
        ref_sample = joblib.load(model_dir / "reference.joblib")
        assert isinstance(ref_sample, pd.DataFrame)
        assert len(ref_sample) <= 1000


class TestLoadDataset:
    """Testes para carregamento de dados."""

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_file.xlsx")


class TestLogEvaluation:
    """Testes para logging de avaliação."""

    def test_log_evaluation(self):
        """Verifica que log_evaluation_results não falha."""
        metrics = {"accuracy": 0.95, "f1_score": 0.90}
        report = "classification report text"
        confusion = {"true_negatives": 10, "false_positives": 1, "false_negatives": 2, "true_positives": 8}
        # Deve rodar sem exceção
        log_evaluation_results(metrics, report, confusion)
