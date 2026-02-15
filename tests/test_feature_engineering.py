"""
Testes do módulo de feature engineering.
"""
import pytest
import pandas as pd
import numpy as np

from app.ml.feature_engineering import (
    create_derived_features,
    select_features,
    run_feature_engineering,
)
from app.core.config import SELECTED_FEATURES, SELECTED_FEATURES_NO_IAN


class TestCreateDerivedFeatures:
    """Testes para criação de features derivadas."""

    def test_anos_na_pm_created(self, preprocessed_data):
        """Verifica criação de Anos_na_PM."""
        df = create_derived_features(preprocessed_data)
        assert "Anos_na_PM" in df.columns
        assert all(df["Anos_na_PM"] >= 0)

    def test_anos_na_pm_calculation(self, preprocessed_data):
        """Verifica cálculo correto de Anos_na_PM."""
        df = create_derived_features(preprocessed_data)
        for _, row in df.head(10).iterrows():
            expected = 2022 - row["Ano ingresso"]
            assert row["Anos_na_PM"] == expected

    def test_evolucao_pedra_20_22(self, preprocessed_data):
        """Verifica criação de Evolucao_pedra_20_22."""
        df = create_derived_features(preprocessed_data)
        assert "Evolucao_pedra_20_22" in df.columns

    def test_evolucao_pedra_21_22(self, preprocessed_data):
        """Verifica criação de Evolucao_pedra_21_22."""
        df = create_derived_features(preprocessed_data)
        assert "Evolucao_pedra_21_22" in df.columns

    def test_evolucao_values_range(self, preprocessed_data):
        """Verifica que evolução está em range válido (-3 a +3)."""
        df = create_derived_features(preprocessed_data)
        for col in ["Evolucao_pedra_20_22", "Evolucao_pedra_21_22"]:
            valid = df[col].dropna()
            if len(valid) > 0:
                assert all(valid >= -3)
                assert all(valid <= 3)

    def test_preserves_columns(self, preprocessed_data):
        """Verifica que colunas originais são preservadas."""
        original_cols = set(preprocessed_data.columns)
        df = create_derived_features(preprocessed_data)
        assert original_cols.issubset(set(df.columns))


class TestSelectFeatures:
    """Testes para seleção de features."""

    def test_returns_x_and_y(self, engineered_data):
        """Verifica que retorna X e y."""
        X, y = select_features(engineered_data)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_y_is_binary(self, engineered_data):
        """Verifica que y é binário."""
        _, y = select_features(engineered_data)
        assert set(y.unique()).issubset({0, 1})

    def test_x_shape_matches_y(self, engineered_data):
        """Verifica que X e y têm o mesmo número de linhas."""
        X, y = select_features(engineered_data)
        assert len(X) == len(y)

    def test_exclude_ian(self, engineered_data):
        """Verifica que IAN é excluído quando include_ian=False."""
        X_with, _ = select_features(engineered_data, include_ian=True)
        X_without, _ = select_features(engineered_data, include_ian=False)
        if "IAN" in X_with.columns:
            assert "IAN" not in X_without.columns

    def test_include_ian(self, engineered_data):
        """Verifica que IAN é incluído quando include_ian=True."""
        X, _ = select_features(engineered_data, include_ian=True)
        assert "IAN" in X.columns


class TestRunFeatureEngineering:
    """Testes para a pipeline de feature engineering."""

    def test_runs_without_error(self, preprocessed_data):
        """Verifica que a pipeline executa sem erros."""
        df = run_feature_engineering(preprocessed_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(preprocessed_data)

    def test_adds_new_columns(self, preprocessed_data):
        """Verifica que novas colunas são adicionadas."""
        df = run_feature_engineering(preprocessed_data)
        assert len(df.columns) > len(preprocessed_data.columns)
