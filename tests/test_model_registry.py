"""
Testes do Model Registry.
"""
import pytest
from xgboost import XGBClassifier

from app.ml.model_registry import create_model, get_available_models, get_param_grid


class TestGetAvailableModels:
    def test_returns_list(self):
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) >= 1

    def test_xgboost_available(self):
        models = get_available_models()
        types = [m["type"] for m in models]
        assert "xgboost" in types

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

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="não registrado"):
            create_model("modelo_inexistente")
