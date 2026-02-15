"""
Testes adicionais para aumentar cobertura.

Cobrindo: monitoring/drift.py, ml/train.py (save), e ml/preprocessing.py (preprocess_dataset).
"""
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import patch

from app.monitoring.drift import (
    log_prediction,
    get_prediction_log,
    get_prediction_stats,
    calculate_psi,
    check_drift,
    check_all_drift,
    _prediction_log,
)
from app.ml.train import save_model_artifacts, run_training_pipeline
from app.ml.preprocessing import preprocess_dataset, load_dataset
from app.ml.evaluate import log_evaluation_results


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

    def test_check_drift_no_reference(self, tmp_path):
        with patch("app.monitoring.drift.REFERENCE_DIST_PATH", tmp_path / "nonexistent.joblib"):
            result = check_drift("IAA", np.array([5.0, 6.0, 7.0, 8.0, 9.0]))
            assert "error" in result

    def test_check_drift_few_values(self, tmp_path):
        ref_path = tmp_path / "ref.joblib"
        joblib.dump({"IAA": {"mean": 5.0, "std": 1.0, "min": 2.0, "max": 8.0, "median": 5.0, "q25": 4.0, "q75": 6.0}}, ref_path)
        with patch("app.monitoring.drift.REFERENCE_DIST_PATH", ref_path):
            result = check_drift("IAA", np.array([5.0]))
            assert "warning" in result

    def test_check_drift_valid(self, tmp_path):
        ref_path = tmp_path / "ref.joblib"
        joblib.dump({"IAA": {"mean": 5.0, "std": 1.0, "min": 2.0, "max": 8.0, "median": 5.0, "q25": 4.0, "q75": 6.0}}, ref_path)
        with patch("app.monitoring.drift.REFERENCE_DIST_PATH", ref_path):
            result = check_drift("IAA", np.random.normal(5, 1, 50))
            assert "drift_status" in result
            assert result["drift_status"] in ["OK", "WARNING", "DRIFT_DETECTED"]

    def test_check_drift_wrong_feature(self, tmp_path):
        ref_path = tmp_path / "ref.joblib"
        joblib.dump({"IAA": {"mean": 5.0, "std": 1.0}}, ref_path)
        with patch("app.monitoring.drift.REFERENCE_DIST_PATH", ref_path):
            result = check_drift("NONEXISTENT", np.array([5.0, 6.0, 7.0, 8.0, 9.0]))
            assert "error" in result

    def test_check_all_drift_no_data(self):
        result = check_all_drift()
        assert result["status"] == "NO_DATA"

    def test_check_all_drift_no_reference(self, tmp_path):
        log_prediction({"IAA": 5.0}, {"prediction": 0, "probability": 0.2})
        with patch("app.monitoring.drift.REFERENCE_DIST_PATH", tmp_path / "nonexistent.joblib"):
            result = check_all_drift()
            assert result["status"] == "NO_REFERENCE"


class TestSaveModelArtifacts:
    """Testes para salvamento de artefatos do modelo."""

    def test_save_model(self, engineered_data, tmp_path):
        from app.ml.feature_engineering import select_features
        from app.ml.train import train_model
        from sklearn.model_selection import train_test_split

        X, y = select_features(engineered_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = train_model(X_train, y_train, optimize=False)

        model_path = tmp_path / "model.joblib"
        meta_path = tmp_path / "metadata.joblib"
        ref_path = tmp_path / "reference.joblib"

        with patch("app.ml.train.MODEL_PATH", model_path), \
             patch("app.ml.train.TRAIN_METADATA_PATH", meta_path), \
             patch("app.ml.train.REFERENCE_DIST_PATH", ref_path):
            save_model_artifacts(
                model=model,
                feature_names=list(X_train.columns),
                metrics={"f1_score": 0.9},
                confusion={"tn": 10, "fp": 1, "fn": 2, "tp": 8},
                feature_importance=[{"feature": "IAA", "importance": 0.5}],
                X_train=X_train,
            )

        assert model_path.exists()
        assert meta_path.exists()
        assert ref_path.exists()

        metadata = joblib.load(meta_path)
        assert "feature_names" in metadata
        assert "metrics" in metadata


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
