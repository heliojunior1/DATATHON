"""
Testes do Model Registry.
"""
import pytest
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from app.ml.model_registry import create_model, get_available_models, get_param_grid


class TestGetAvailableModels:
    def test_returns_list(self):
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) >= 2  # XGBoost + CatBoost

    def test_xgboost_available(self):
        models = get_available_models()
        types = [m["type"] for m in models]
        assert "xgboost" in types

    def test_catboost_available(self):
        models = get_available_models()
        types = [m["type"] for m in models]
        assert "catboost" in types

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

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="não registrado"):
            get_param_grid("modelo_inexistente")


class TestCreateModel:
    def test_create_xgboost(self):
        model = create_model("xgboost")
        assert isinstance(model, XGBClassifier)

    def test_create_xgboost_with_params(self):
        model = create_model("xgboost", params={"n_estimators": 50})
        assert model.n_estimators == 50

    def test_create_xgboost_scale_pos_weight(self):
        model = create_model("xgboost", scale_pos_weight=2.5)
        assert model.scale_pos_weight == 2.5

    def test_create_catboost(self):
        model = create_model("catboost")
        assert isinstance(model, CatBoostClassifier)

    def test_create_catboost_with_params(self):
        model = create_model("catboost", params={"iterations": 50})
        # CatBoost usa get_param para acessar parâmetros
        assert model.get_param("iterations") == 50

    def test_create_catboost_scale_pos_weight(self):
        model = create_model("catboost", scale_pos_weight=2.5)
        assert model.get_param("scale_pos_weight") == 2.5

    def test_catboost_silent_mode(self):
        """CatBoost deve ser criado com verbose=0 para não poluir logs."""
        model = create_model("catboost")
        assert model.get_param("verbose") == 0

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="não registrado"):
            create_model("modelo_inexistente")
