"""
Testes do módulo de pré-processamento.
"""
import pytest
import pandas as pd
import numpy as np

from app.ml.preprocessing import (
    create_target_variable,
    encode_categorical_columns,
    handle_missing_values,
    _encode_rec_avaliador,
)
from app.core.config import DEFASAGEM_THRESHOLD


class TestCreateTargetVariable:
    """Testes para criação da variável target."""

    def test_creates_risco_column(self, sample_raw_data):
        """Verifica se a coluna risco_defasagem é criada."""
        df = create_target_variable(sample_raw_data)
        assert "risco_defasagem" in df.columns

    def test_binary_values(self, sample_raw_data):
        """Verifica se o target é binário (0 ou 1)."""
        df = create_target_variable(sample_raw_data)
        assert set(df["risco_defasagem"].unique()).issubset({0, 1})

    def test_threshold_logic(self, sample_raw_data):
        """Verifica se o threshold funciona corretamente."""
        df = create_target_variable(sample_raw_data, threshold=-2)
        # Alunos com Defas <= -2 devem ser 1 (em risco)
        for _, row in df.iterrows():
            if row["Defas"] <= -2:
                assert row["risco_defasagem"] == 1
            else:
                assert row["risco_defasagem"] == 0

    def test_custom_threshold(self, sample_raw_data):
        """Testa com threshold customizado."""
        df_strict = create_target_variable(sample_raw_data, threshold=-3)
        df_loose = create_target_variable(sample_raw_data, threshold=-1)
        assert df_strict["risco_defasagem"].sum() <= df_loose["risco_defasagem"].sum()

    def test_preserves_original_columns(self, sample_raw_data):
        """Verifica que não perde colunas originais."""
        original_cols = set(sample_raw_data.columns)
        df = create_target_variable(sample_raw_data)
        assert original_cols.issubset(set(df.columns))


class TestEncodeCategoricalColumns:
    """Testes para codificação de variáveis categóricas."""

    def test_genero_encoded(self, sample_raw_data):
        """Verifica codificação de Gênero."""
        df = create_target_variable(sample_raw_data)
        df = encode_categorical_columns(df)
        assert "Genero_encoded" in df.columns
        assert set(df["Genero_encoded"].unique()).issubset({0, 1})

    def test_escola_encoded(self, sample_raw_data):
        """Verifica codificação de Instituição de ensino."""
        df = create_target_variable(sample_raw_data)
        df = encode_categorical_columns(df)
        assert "Escola_encoded" in df.columns
        assert set(df["Escola_encoded"].unique()).issubset({0, 1, 2})

    def test_pedra_encoded(self, sample_raw_data):
        """Verifica codificação das Pedras."""
        df = create_target_variable(sample_raw_data)
        df = encode_categorical_columns(df)
        for year in ["20", "21", "22"]:
            col = f"Pedra_{year}_encoded"
            assert col in df.columns
            # Valores válidos: 1-4 (Quartzo a Topázio) ou NaN
            valid_values = df[col].dropna().unique()
            assert all(v in {1, 2, 3, 4} for v in valid_values)

    def test_rec_psico_encoded(self, sample_raw_data):
        """Verifica codificação de Rec Psicologia."""
        df = create_target_variable(sample_raw_data)
        df = encode_categorical_columns(df)
        assert "Rec_psico_encoded" in df.columns
        assert all(df["Rec_psico_encoded"] >= 0)
        assert all(df["Rec_psico_encoded"] <= 4)

    def test_boolean_flags(self, sample_raw_data):
        """Verifica flags booleanas (Ponto de Virada, Indicado, Destaques)."""
        df = create_target_variable(sample_raw_data)
        df = encode_categorical_columns(df)
        for flag in ["Ponto_virada_flag", "Indicado_flag", "Destaque_IEG_flag",
                      "Destaque_IDA_flag", "Destaque_IPV_flag"]:
            assert flag in df.columns
            assert set(df[flag].unique()).issubset({0, 1})

    def test_rec_avaliador_encoded(self, sample_raw_data):
        """Verifica codificação de recomendações de avaliadores."""
        df = create_target_variable(sample_raw_data)
        df = encode_categorical_columns(df)
        assert "Rec_av1_encoded" in df.columns
        assert "Rec_av2_encoded" in df.columns


class TestEncodeRecAvaliador:
    """Testes da função _encode_rec_avaliador."""

    def test_promovido(self):
        assert _encode_rec_avaliador("Promovido de fase") == 4

    def test_mantido(self):
        assert _encode_rec_avaliador("Mantido na mesma fase") == 3

    def test_mantido_ressalvas(self):
        assert _encode_rec_avaliador("Mantido na mesma fase com ressalvas") == 2

    def test_observacao(self):
        assert _encode_rec_avaliador("Em observação") == 1

    def test_sem_info(self):
        assert _encode_rec_avaliador("Sem informação") == 0

    def test_nan_value(self):
        assert _encode_rec_avaliador(None) == 0
        assert _encode_rec_avaliador(np.nan) == 0

    def test_unknown_value(self):
        assert _encode_rec_avaliador("Unknown") == 0


class TestHandleMissingValues:
    """Testes para tratamento de valores ausentes."""

    def test_matem_no_nulls(self, sample_raw_data):
        """Verifica que Matem não tem mais nulos."""
        # Adicionar nulos propositalmente
        df = sample_raw_data.copy()
        df.loc[0, "Matem"] = np.nan
        df = create_target_variable(df)
        df = encode_categorical_columns(df)
        df = handle_missing_values(df)
        assert df["Matem"].isnull().sum() == 0

    def test_portug_no_nulls(self, sample_raw_data):
        """Verifica que Portug não tem mais nulos."""
        df = sample_raw_data.copy()
        df.loc[0, "Portug"] = np.nan
        df = create_target_variable(df)
        df = encode_categorical_columns(df)
        df = handle_missing_values(df)
        assert df["Portug"].isnull().sum() == 0

    def test_ingles_preserves_nulls(self, sample_raw_data):
        """Verifica que Inglês mantém nulos (para XGBoost)."""
        df = create_target_variable(sample_raw_data)
        df = encode_categorical_columns(df)
        original_nulls = df["Inglês"].isnull().sum()
        df = handle_missing_values(df)
        assert df["Inglês"].isnull().sum() == original_nulls

    def test_preserves_data_shape(self, sample_raw_data):
        """Verifica que o número de linhas não muda."""
        df = create_target_variable(sample_raw_data)
        df = encode_categorical_columns(df)
        original_len = len(df)
        df = handle_missing_values(df)
        assert len(df) == original_len
