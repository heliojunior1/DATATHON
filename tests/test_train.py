"""
Testes do módulo de treinamento.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.ml.train import train_model, get_xgb_param_grid
from app.ml.feature_engineering import select_features
from app.ml.evaluate import (
    calculate_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_feature_importance,
)
from app.core.config import RANDOM_STATE, TEST_SIZE


class TestGetXGBParamGrid:
    """Testes para o grid de hiperparâmetros."""

    def test_returns_dict(self):
        grid = get_xgb_param_grid()
        assert isinstance(grid, dict)

    def test_has_essential_params(self):
        grid = get_xgb_param_grid()
        essential_params = ["n_estimators", "max_depth", "learning_rate", "subsample"]
        for param in essential_params:
            assert param in grid

    def test_params_are_lists(self):
        grid = get_xgb_param_grid()
        for values in grid.values():
            assert isinstance(values, list)
            assert len(values) > 0


class TestTrainModel:
    """Testes para o treinamento do modelo."""

    def test_trains_without_optimization(self, engineered_data):
        """Verifica treinamento sem otimização."""
        X, y = select_features(engineered_data)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model = train_model(X_train, y_train, optimize=False)
        assert isinstance(model, XGBClassifier)

    def test_model_can_predict(self, engineered_data):
        """Verifica que o modelo treinado faz predições."""
        X, y = select_features(engineered_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model = train_model(X_train, y_train, optimize=False)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_model_predict_proba(self, engineered_data):
        """Verifica que o modelo retorna probabilidades."""
        X, y = select_features(engineered_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model = train_model(X_train, y_train, optimize=False)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert all(0 <= p <= 1 for p in proba[:, 1])


class TestCalculateMetrics:
    """Testes para cálculo de métricas."""

    def test_basic_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        metrics = calculate_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_metrics_range(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        metrics = calculate_metrics(y_true, y_pred)
        for value in metrics.values():
            assert 0 <= value <= 1

    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_with_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        assert "auc_roc" in metrics
        assert 0 <= metrics["auc_roc"] <= 1


class TestGetClassificationReport:
    """Testes para relatório de classificação."""

    def test_returns_string(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        report = get_classification_report(y_true, y_pred)
        assert isinstance(report, str)
        assert "Sem Risco" in report
        assert "Em Risco" in report


class TestGetConfusionMatrix:
    """Testes para confusion matrix."""

    def test_returns_dict(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm = get_confusion_matrix(y_true, y_pred)
        assert "true_negatives" in cm
        assert "false_positives" in cm
        assert "false_negatives" in cm
        assert "true_positives" in cm

    def test_values_sum(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm = get_confusion_matrix(y_true, y_pred)
        total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
        assert total == len(y_true)


class TestGetFeatureImportance:
    """Testes para importância de features."""

    def test_returns_sorted_list(self, engineered_data):
        X, y = select_features(engineered_data)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model = train_model(X_train, y_train, optimize=False)
        importance = get_feature_importance(model, list(X_train.columns))
        assert isinstance(importance, list)
        assert len(importance) == len(X_train.columns)
        # Verify sorted descending
        for i in range(len(importance) - 1):
            assert importance[i]["importance"] >= importance[i + 1]["importance"]

    def test_importance_has_feature_name(self, engineered_data):
        X, y = select_features(engineered_data)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model = train_model(X_train, y_train, optimize=False)
        importance = get_feature_importance(model, list(X_train.columns))
        for item in importance:
            assert "feature" in item
            assert "importance" in item
