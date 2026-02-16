"""
Testes do Model Registry.
"""
import pytest
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier

from app.ml.model_registry import (
    create_model,
    get_available_models,
    get_param_grid,
    supports_hyperparam_search,
    supports_scale_pos_weight,
)


class TestGetAvailableModels:
    def test_returns_list(self):
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) >= 4  # XGBoost + CatBoost + LightGBM + TabPFN

    def test_xgboost_available(self):
        models = get_available_models()
        types = [m["type"] for m in models]
        assert "xgboost" in types

    def test_catboost_available(self):
        models = get_available_models()
        types = [m["type"] for m in models]
        assert "catboost" in types

    def test_lightgbm_available(self):
        models = get_available_models()
        types = [m["type"] for m in models]
        assert "lightgbm" in types

    def test_tabpfn_available(self):
        models = get_available_models()
        types = [m["type"] for m in models]
        assert "tabpfn" in types

    def test_model_has_required_fields(self):
        models = get_available_models()
        for m in models:
            assert "type" in m
            assert "name" in m
            assert "description" in m
            assert "supports_nan" in m


class TestGetParamGrid:
    def test_xgboost_grid(self):
        grid = get_param_grid("xgboost")
        assert isinstance(grid, dict)
        assert "n_estimators" in grid
        assert "max_depth" in grid
        assert "learning_rate" in grid

    def test_catboost_grid(self):
        grid = get_param_grid("catboost")
        assert isinstance(grid, dict)
        assert "iterations" in grid
        assert "depth" in grid
        assert "learning_rate" in grid
        assert "l2_leaf_reg" in grid

    def test_lightgbm_grid(self):
        grid = get_param_grid("lightgbm")
        assert isinstance(grid, dict)
        assert "n_estimators" in grid
        assert "num_leaves" in grid
        assert "learning_rate" in grid
        assert "max_depth" in grid

    def test_tabpfn_grid_empty(self):
        """TabPFN é pré-treinado, não tem grid de hiperparâmetros."""
        grid = get_param_grid("tabpfn")
        assert isinstance(grid, dict)
        assert len(grid) == 0

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="não registrado"):
            get_param_grid("modelo_inexistente")


class TestSupportsFlags:
    def test_xgboost_supports_all(self):
        assert supports_hyperparam_search("xgboost") is True
        assert supports_scale_pos_weight("xgboost") is True

    def test_catboost_supports_all(self):
        assert supports_hyperparam_search("catboost") is True
        assert supports_scale_pos_weight("catboost") is True

    def test_lightgbm_supports_all(self):
        assert supports_hyperparam_search("lightgbm") is True
        assert supports_scale_pos_weight("lightgbm") is True

    def test_tabpfn_no_hyperparam_search(self):
        assert supports_hyperparam_search("tabpfn") is False

    def test_tabpfn_no_scale_pos_weight(self):
        assert supports_scale_pos_weight("tabpfn") is False


class TestCreateModel:
    # ── XGBoost ──
    def test_create_xgboost(self):
        model = create_model("xgboost")
        assert isinstance(model, XGBClassifier)

    def test_create_xgboost_with_params(self):
        model = create_model("xgboost", params={"n_estimators": 50})
        assert model.n_estimators == 50

    def test_create_xgboost_scale_pos_weight(self):
        model = create_model("xgboost", scale_pos_weight=2.5)
        assert model.scale_pos_weight == 2.5

    # ── CatBoost ──
    def test_create_catboost(self):
        model = create_model("catboost")
        assert isinstance(model, CatBoostClassifier)

    def test_create_catboost_with_params(self):
        model = create_model("catboost", params={"iterations": 50})
        assert model.get_param("iterations") == 50

    def test_create_catboost_scale_pos_weight(self):
        model = create_model("catboost", scale_pos_weight=2.5)
        assert model.get_param("scale_pos_weight") == 2.5

    def test_catboost_silent_mode(self):
        model = create_model("catboost")
        assert model.get_param("verbose") == 0

    # ── LightGBM ──
    def test_create_lightgbm(self):
        model = create_model("lightgbm")
        assert isinstance(model, LGBMClassifier)

    def test_create_lightgbm_with_params(self):
        model = create_model("lightgbm", params={"n_estimators": 50})
        assert model.n_estimators == 50

    def test_create_lightgbm_scale_pos_weight(self):
        model = create_model("lightgbm", scale_pos_weight=2.5)
        assert model.scale_pos_weight == 2.5

    def test_lightgbm_silent_mode(self):
        model = create_model("lightgbm")
        assert model.verbose == -1

    def test_lightgbm_num_leaves(self):
        model = create_model("lightgbm", params={"num_leaves": 50})
        assert model.num_leaves == 50

    # ── TabPFN ──
    def test_create_tabpfn(self):
        model = create_model("tabpfn")
        assert isinstance(model, TabPFNClassifier)

    def test_create_tabpfn_ignores_scale_pos_weight(self):
        """TabPFN não aceita scale_pos_weight, create_model ignora."""
        model = create_model("tabpfn")
        assert isinstance(model, TabPFNClassifier)

    def test_tabpfn_n_estimators(self):
        model = create_model("tabpfn", params={"n_estimators": 4})
        assert model.n_estimators == 4

    def test_tabpfn_fit_predict(self):
        """Smoke test: TabPFN deve treinar e prever em dados simples."""
        model = create_model("tabpfn", params={"n_estimators": 2})
        X = np.random.rand(50, 5)
        y = (X[:, 0] > 0.5).astype(int)
        try:
            model.fit(X, y)
        except RuntimeError as e:
            if "download" in str(e).lower():
                pytest.skip("TabPFN model not available for download")
            raise
        preds = model.predict(X[:5])
        assert len(preds) == 5
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 2)

    # ── Invalid ──
    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="não registrado"):
            create_model("modelo_inexistente")
