"""
Testes do módulo de feature engineering.
"""
import pytest
import pandas as pd
import numpy as np

from app.services.feature_engineering import (
    create_derived_features,
    select_features,
    run_feature_engineering,
)
from app.config import SELECTED_FEATURES, SELECTED_FEATURES_NO_IAN, INCLUDE_IAN


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

    def test_evolucao_delta_neutro(self, preprocessed_data):
        """Verifica que delta é 0 para alunos sem histórico de Pedra."""
        df = create_derived_features(preprocessed_data)
        if "tinha_pedra_20" in df.columns:
            sem_pedra = df[df["tinha_pedra_20"] == 0]
            if len(sem_pedra) > 0:
                assert all(sem_pedra["Evolucao_pedra_20_22"] == 0)

    def test_variancia_indicadores_created(self, preprocessed_data):
        """Verifica criação de Variancia_indicadores."""
        df = create_derived_features(preprocessed_data)
        assert "Variancia_indicadores" in df.columns
        assert all(df["Variancia_indicadores"] >= 0)

    def test_ratio_ida_ieg_created(self, preprocessed_data):
        """Verifica criação de Ratio_IDA_IEG."""
        df = create_derived_features(preprocessed_data)
        assert "Ratio_IDA_IEG" in df.columns
        assert all(df["Ratio_IDA_IEG"] >= 0)

    def test_mismatch_idade_fase_created(self, preprocessed_data):
        """Verifica criação de mismatch_idade_fase."""
        df = create_derived_features(preprocessed_data)
        assert "mismatch_idade_fase" in df.columns
        assert set(df["mismatch_idade_fase"].unique()).issubset({0, 1})

    def test_delta_idade_fase_created(self, preprocessed_data):
        """Verifica criação de delta_idade_fase."""
        df = create_derived_features(preprocessed_data)
        assert "delta_idade_fase" in df.columns
        # delta pode ser positivo ou negativo
        assert not df["delta_idade_fase"].isnull().any()

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

    def test_default_excludes_ian(self, engineered_data):
        """Verifica que por default (INCLUDE_IAN=False) IAN é excluído."""
        X, _ = select_features(engineered_data)
        assert INCLUDE_IAN is False
        assert "IAN" not in X.columns

    def test_include_ian_override(self, engineered_data):
        """Verifica que include_ian=True inclui a feature."""
        X_with, _ = select_features(engineered_data, include_ian=True)
        assert "IAN" in X_with.columns

    def test_exclude_ian_override(self, engineered_data):
        """Verifica que include_ian=False exclui a feature."""
        X_without, _ = select_features(engineered_data, include_ian=False)
        assert "IAN" not in X_without.columns

    def test_new_features_present(self, engineered_data):
        """Verifica que novas features derivadas estão no X."""
        X, _ = select_features(engineered_data)
        for feat in ["Variancia_indicadores", "Ratio_IDA_IEG",
                      "Tem_nota_ingles", "Fase_encoded",
                      "tinha_pedra_20", "tinha_pedra_21",
                      "tem_ranking_cf", "tem_ranking_ct"]:
            assert feat in X.columns, f"Feature '{feat}' deveria estar presente"
        # Verificar que features com leakage NÃO estão presentes
        for feat in ["mismatch_idade_fase", "delta_idade_fase"]:
            assert feat not in X.columns, f"Feature '{feat}' com leakage não deveria estar presente"


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
