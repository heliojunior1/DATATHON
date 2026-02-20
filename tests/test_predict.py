"""
Testes do módulo de predição.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
from sklearn.model_selection import train_test_split

from app.services.predict_service import prepare_input_features, predict, predict_batch, clear_model_cache
from app.services.train_service import train_model
from app.services.feature_engineering import select_features
from app.services.model_storage import save_trained_model, clear_cache as clear_storage_cache
from app.config import RANDOM_STATE, TEST_SIZE, SELECTED_FEATURES


class TestPrepareInputFeatures:
    """Testes para preparação de features de entrada."""

    def test_returns_dataframe(self, sample_student_input):
        features = SELECTED_FEATURES[:10]
        df = prepare_input_features(sample_student_input, features)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_correct_columns(self, sample_student_input):
        features = ["IAA", "IEG", "IPS", "IDA"]
        df = prepare_input_features(sample_student_input, features)
        assert list(df.columns) == features

    def test_genero_encoding(self):
        input_data = {"Genero": "Menina"}
        df = prepare_input_features(input_data, ["Genero_encoded"])
        assert df["Genero_encoded"].iloc[0] == 0

        input_data = {"Genero": "Menino"}
        df = prepare_input_features(input_data, ["Genero_encoded"])
        assert df["Genero_encoded"].iloc[0] == 1

    def test_escola_encoding(self):
        input_data = {"Instituicao_ensino": "Escola Pública"}
        df = prepare_input_features(input_data, ["Escola_encoded"])
        assert df["Escola_encoded"].iloc[0] == 0

    def test_anos_na_pm_from_ingresso(self):
        input_data = {"Ano_ingresso": 2018}
        df = prepare_input_features(input_data, ["Anos_na_PM"])
        assert df["Anos_na_PM"].iloc[0] == 4  # 2022 - 2018

    def test_pedra_encoding(self):
        input_data = {"Pedra_22": "Topázio"}
        df = prepare_input_features(input_data, ["Pedra_22_encoded"])
        assert df["Pedra_22_encoded"].iloc[0] == 4

    def test_boolean_flags_from_string(self):
        input_data = {"Atingiu_PV": "Sim", "Indicado": "Não"}
        df = prepare_input_features(input_data, ["Ponto_virada_flag", "Indicado_flag"])
        assert df["Ponto_virada_flag"].iloc[0] == 1
        assert df["Indicado_flag"].iloc[0] == 0

    def test_missing_feature_fills_nan(self, sample_student_input):
        features = ["IAA", "FEATURE_INEXISTENTE"]
        df = prepare_input_features(sample_student_input, features)
        assert pd.isna(df["FEATURE_INEXISTENTE"].iloc[0])

    def test_evolucao_pedra_calculation(self):
        input_data = {"Pedra_20": "Ágata", "Pedra_22": "Topázio"}
        df = prepare_input_features(input_data, ["Pedra_20_encoded", "Pedra_22_encoded", "Evolucao_pedra_20_22", "tinha_pedra_20"])
        assert df["Evolucao_pedra_20_22"].iloc[0] == 2  # 4 - 2
        assert df["tinha_pedra_20"].iloc[0] == 1

    def test_fase_encoding(self):
        input_data = {"Fase": "Fase 3"}
        df = prepare_input_features(input_data, ["Fase_encoded"])
        assert df["Fase_encoded"].iloc[0] == 3

    def test_tem_nota_ingles_inference(self):
        input_data = {"Inglês": 7.5}
        df = prepare_input_features(input_data, ["Tem_nota_ingles"])
        assert df["Tem_nota_ingles"].iloc[0] == 1

        input_data_sem = {}
        df2 = prepare_input_features(input_data_sem, ["Tem_nota_ingles"])
        assert df2["Tem_nota_ingles"].iloc[0] == 0

    def test_delta_neutro_no_pedra(self):
        """Sem Pedra 20, delta deve ser 0."""
        input_data = {"Pedra_22": "Topázio"}
        df = prepare_input_features(input_data, ["Pedra_20_encoded", "Pedra_22_encoded", "Evolucao_pedra_20_22", "tinha_pedra_20"])
        assert df["tinha_pedra_20"].iloc[0] == 0
        assert df["Evolucao_pedra_20_22"].iloc[0] == 0  # Delta neutro
        assert df["Pedra_20_encoded"].iloc[0] == 0  # Imputado com 0


class TestPredict:
    """Testes para predição (requer modelo treinado)."""

    @pytest.fixture
    def trained_model_setup(self, engineered_data, tmp_path):
        """Treina e salva um modelo usando model_storage."""
        X, y = select_features(engineered_data)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model = train_model(X_train, y_train, model_type="xgboost", optimize=False)
        feature_names = list(X_train.columns)

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with patch("app.ml.model_storage.MODELS_DIR", models_dir), \
             patch("app.ml.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            metadata = {
                "metrics": {"f1_score": 0.8},
                "feature_importance": [{"feature": f, "importance": 0.1} for f in feature_names],
                "n_training_samples": len(X_train),
            }
            model_id = save_trained_model(model, metadata, "xgboost", feature_names, X_train)
            yield models_dir, model_id, feature_names

    def test_predict_returns_dict(self, trained_model_setup, sample_student_input):
        models_dir, model_id, _ = trained_model_setup
        clear_model_cache()
        with patch("app.ml.model_storage.MODELS_DIR", models_dir), \
             patch("app.ml.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            result = predict(sample_student_input)
            assert isinstance(result, dict)
            assert "prediction" in result
            assert "probability" in result
            assert "risk_level" in result
            assert "label" in result

    def test_prediction_is_binary(self, trained_model_setup, sample_student_input):
        models_dir, model_id, _ = trained_model_setup
        clear_model_cache()
        with patch("app.ml.model_storage.MODELS_DIR", models_dir), \
             patch("app.ml.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            result = predict(sample_student_input)
            assert result["prediction"] in [0, 1]

    def test_probability_range(self, trained_model_setup, sample_student_input):
        models_dir, model_id, _ = trained_model_setup
        clear_model_cache()
        with patch("app.ml.model_storage.MODELS_DIR", models_dir), \
             patch("app.ml.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            result = predict(sample_student_input)
            assert 0 <= result["probability"] <= 1

    def test_risk_level_mapping(self, trained_model_setup, sample_student_input):
        models_dir, model_id, _ = trained_model_setup
        clear_model_cache()
        with patch("app.ml.model_storage.MODELS_DIR", models_dir), \
             patch("app.ml.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            result = predict(sample_student_input)
            valid_levels = ["Muito Baixo", "Baixo", "Moderado", "Alto", "Muito Alto"]
            assert result["risk_level"] in valid_levels

    def test_predict_with_model_id(self, trained_model_setup, sample_student_input):
        models_dir, model_id, _ = trained_model_setup
        clear_model_cache()
        with patch("app.ml.model_storage.MODELS_DIR", models_dir), \
             patch("app.ml.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            result = predict(sample_student_input, model_id=model_id)
            assert result["model_id"] == model_id

    def test_predict_batch(self, trained_model_setup, sample_student_input):
        models_dir, model_id, _ = trained_model_setup
        clear_model_cache()
        with patch("app.ml.model_storage.MODELS_DIR", models_dir), \
             patch("app.ml.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            results = predict_batch([sample_student_input, sample_student_input])
            assert len(results) == 2
            for result in results:
                assert "prediction" in result

    def test_model_not_found_raises(self, tmp_path):
        clear_model_cache()
        models_dir = tmp_path / "empty_models"
        models_dir.mkdir()
        with patch("app.ml.model_storage.MODELS_DIR", models_dir), \
             patch("app.ml.model_storage.INDEX_PATH", models_dir / "index.json"):
            clear_storage_cache()
            with pytest.raises(FileNotFoundError):
                predict({"IAA": 5.0})
