"""
Módulo de engenharia de features.

Cria features derivadas a partir dos dados pré-processados.
"""
import pandas as pd
import numpy as np

from app.core.config import (
    SELECTED_FEATURES,
    SELECTED_FEATURES_NO_IAN,
    INCLUDE_IAN,
    IDADE_MAX_ESPERADA,
)
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria todas as features derivadas a partir do dataset pré-processado.

    Features criadas:
    - Anos_na_PM: anos na Passos Mágicos
    - Evolucao_pedra_20_22: evolução da classificação Pedra entre 2020 e 2022
    - Evolucao_pedra_21_22: evolução da classificação Pedra entre 2021 e 2022
    - Variancia_indicadores: desvio padrão dos indicadores (desempenho irregular)
    - Ratio_IDA_IEG: razão desempenho vs esforço
    - mismatch_idade_fase: flag binária (idade > esperada para fase)
    - delta_idade_fase: diferença contínua (idade - esperada)

    Args:
        df: DataFrame pré-processado (com colunas codificadas).

    Returns:
        DataFrame com features derivadas adicionadas.
    """
    df = df.copy()

    # Anos na Passos Mágicos
    df["Anos_na_PM"] = 2022 - df["Ano ingresso"]
    logger.info(f"  Anos_na_PM: min={df['Anos_na_PM'].min()}, max={df['Anos_na_PM'].max()}")

    # Evolução das Pedras (delta ordinal entre anos)
    if "Pedra_20_encoded" in df.columns and "Pedra_22_encoded" in df.columns:
        df["Evolucao_pedra_20_22"] = df["Pedra_22_encoded"] - df["Pedra_20_encoded"]
        logger.info(
            f"  Evolucao_pedra_20_22: "
            f"disponível para {df['Evolucao_pedra_20_22'].notna().sum()} alunos"
        )

    if "Pedra_21_encoded" in df.columns and "Pedra_22_encoded" in df.columns:
        df["Evolucao_pedra_21_22"] = df["Pedra_22_encoded"] - df["Pedra_21_encoded"]
        logger.info(
            f"  Evolucao_pedra_21_22: "
            f"disponível para {df['Evolucao_pedra_21_22'].notna().sum()} alunos"
        )

    # Forçar delta neutro para alunos sem histórico (estratégia "Delta Neutro com Flag")
    if "tinha_pedra_20" in df.columns and "Evolucao_pedra_20_22" in df.columns:
        df.loc[df["tinha_pedra_20"] == 0, "Evolucao_pedra_20_22"] = 0
    if "tinha_pedra_21" in df.columns and "Evolucao_pedra_21_22" in df.columns:
        df.loc[df["tinha_pedra_21"] == 0, "Evolucao_pedra_21_22"] = 0

    # Variância dos indicadores — alunos com desempenho irregular
    indicator_cols = ["IAA", "IEG", "IPS", "IDA", "IPV"]
    available_indicators = [c for c in indicator_cols if c in df.columns]
    if len(available_indicators) >= 3:
        df["Variancia_indicadores"] = df[available_indicators].std(axis=1)
        logger.info(f"  Variancia_indicadores: μ={df['Variancia_indicadores'].mean():.2f}")

    # Ratio desempenho vs esforço
    if "IDA" in df.columns and "IEG" in df.columns:
        df["Ratio_IDA_IEG"] = df["IDA"] / (df["IEG"] + 0.01)
        logger.info(f"  Ratio_IDA_IEG: μ={df['Ratio_IDA_IEG'].mean():.2f}")

    # Mismatch idade-fase (indicador de atraso escolar)
    if "Idade 22" in df.columns and "Fase_encoded" in df.columns:
        idade_esperada = df["Fase_encoded"].map(IDADE_MAX_ESPERADA)
        df["delta_idade_fase"] = df["Idade 22"] - idade_esperada
        df["mismatch_idade_fase"] = (df["delta_idade_fase"] > 0).astype(int)
        n_mismatch = df["mismatch_idade_fase"].sum()
        logger.info(f"  mismatch_idade_fase: {n_mismatch} alunos com idade acima do esperado")

    logger.info("Features derivadas criadas com sucesso")
    return df


def select_features(
    df: pd.DataFrame, include_ian: bool | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Seleciona as features finais para o modelo.

    Args:
        df: DataFrame com todas as features criadas.
        include_ian: Se True, inclui IAN (data leakage). Se False, exclui.
                     Se None, usa o valor de config.INCLUDE_IAN (default: False).

    Returns:
        Tupla (X, y) com features selecionadas e target.
    """
    if include_ian is None:
        include_ian = INCLUDE_IAN

    features = SELECTED_FEATURES if include_ian else SELECTED_FEATURES_NO_IAN

    # Filtrar apenas as features que existem no DataFrame
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        logger.warning(f"Features ausentes no DataFrame: {missing_features}")

    X = df[available_features].copy()
    y = df["risco_defasagem"].copy()

    logger.info(f"Features selecionadas: {len(available_features)} (IAN={'incluído' if include_ian else 'excluído'})")
    logger.info(f"Shape final: X={X.shape}, y={y.shape}")

    return X, y


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completa de feature engineering.

    Args:
        df: DataFrame pré-processado.

    Returns:
        DataFrame com todas as features criadas.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO FEATURE ENGINEERING")
    logger.info("=" * 60)

    df = create_derived_features(df)

    logger.info(f"Feature engineering concluído: {df.shape[1]} colunas totais")
    return df
