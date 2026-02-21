"""
Módulo de pré-processamento de dados.

Responsável pelo carregamento, limpeza e transformação inicial do dataset PEDE.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from app.config import (
    DATASET_PATH,
    DEFASAGEM_THRESHOLD,
    GENERO_MAP,
    ESCOLA_MAP,
    PEDRA_ORDINAL_MAP,
    FASE_ORDINAL_MAP,
    REC_AVALIADOR_MAP,
)
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


def load_dataset(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Carrega o dataset PEDE a partir do arquivo Excel.

    Args:
        filepath: Caminho do arquivo. Se None, usa o padrão em config.

    Returns:
        DataFrame com os dados brutos.
    """
    path = Path(filepath) if filepath else DATASET_PATH
    logger.info(f"Carregando dataset de: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")

    df = pd.read_excel(path)
    logger.info(f"Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")
    return df


def create_target_variable(df: pd.DataFrame, threshold: int = DEFASAGEM_THRESHOLD) -> pd.DataFrame:
    """
    Cria a variável target binária de risco de defasagem.

    Args:
        df: DataFrame com a coluna 'Defas'.
        threshold: Limite para classificar como 'em risco'. Defas <= threshold => risco=1.

    Returns:
        DataFrame com a coluna 'risco_defasagem' adicionada.
    """
    df = df.copy()
    df["risco_defasagem"] = (df["Defas"] <= threshold).astype(int)
    n_risco = df["risco_defasagem"].sum()
    n_total = len(df)
    logger.info(
        f"Target criado: {n_risco} em risco ({n_risco/n_total*100:.1f}%), "
        f"{n_total - n_risco} sem risco ({(n_total - n_risco)/n_total*100:.1f}%)"
    )
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variáveis categóricas com mapeamentos ordinais.

    Colunas processadas:
    - Gênero → Genero_encoded
    - Instituição de ensino → Escola_encoded
    - Pedra 20/21/22 → Pedra_XX_encoded
    - Rec Psicologia → Rec_psico_encoded
    - Atingiu PV → Ponto_virada_flag
    - Indicado → Indicado_flag
    - Destaque IEG/IDA/IPV → Destaque_XXX_flag
    - Rec Av1/Av2 → Rec_avX_encoded

    Returns:
        DataFrame com colunas codificadas adicionadas.
    """
    df = df.copy()

    # Gênero
    df["Genero_encoded"] = df["Gênero"].map(GENERO_MAP).fillna(0).astype(int)

    # Instituição de ensino
    df["Escola_encoded"] = df["Instituição de ensino"].map(ESCOLA_MAP).fillna(0).astype(int)

    # Pedras (ordinal com NaN para ausentes)
    for year in ["20", "21", "22"]:
        col = f"Pedra {year}"
        new_col = f"Pedra_{year}_encoded"
        if col in df.columns:
            df[new_col] = df[col].map(PEDRA_ORDINAL_MAP)
            # Manter NaN para XGBoost lidar nativamente

    # Flags de presença para Pedras históricas (estratégia "Delta Neutro com Flag")
    for year in ["20", "21"]:
        col = f"Pedra {year}"
        flag_col = f"tinha_pedra_{year}"
        encoded_col = f"Pedra_{year}_encoded"
        if col in df.columns:
            df[flag_col] = df[col].notna().astype(int)
            # Imputar encoded com 0 (neutro) para alunos sem histórico
            if encoded_col in df.columns:
                df[encoded_col] = df[encoded_col].fillna(0)

    # Fase (ordinal: Alfa=0 ... Fase 8=8)
    if "Fase" in df.columns:
        # Tratar valores como "Fase 1", "Alfa", etc.
        df["Fase_encoded"] = df["Fase"].map(FASE_ORDINAL_MAP)
        # Para valores não mapeados, tentar extrair número
        mask_null = df["Fase_encoded"].isna()
        if mask_null.any():
            df.loc[mask_null, "Fase_encoded"] = pd.to_numeric(
                df.loc[mask_null, "Fase"], errors="coerce"
            ).fillna(0)
        df["Fase_encoded"] = df["Fase_encoded"].astype(int)

    # Rec Psicologia removida — redundante com IPS (já capturado pelo indicador IPS)

    # Ponto de Virada (booleano)
    df["Ponto_virada_flag"] = (df["Atingiu PV"] == "Sim").astype(int)

    # Indicado para Bolsa
    df["Indicado_flag"] = (df["Indicado"] == "Sim").astype(int)

    # Destaques (booleanos)
    for detalhe in ["IEG", "IDA", "IPV"]:
        col = f"Destaque {detalhe}"
        flag_col = f"Destaque_{detalhe}_flag"
        if col in df.columns:
            df[flag_col] = df[col].apply(
                lambda x: 1 if isinstance(x, str) and "Destaque" in x else 0
            )

    # Recomendações dos Avaliadores (apenas Av1 e Av2 — Av3/Av4 têm muitos nulos)
    for av_num in [1, 2]:
        col = f"Rec Av{av_num}"
        new_col = f"Rec_av{av_num}_encoded"
        if col in df.columns:
            df[new_col] = df[col].apply(_encode_rec_avaliador)

    # Flags de presença para Rankings (preparação para produção — alunos novos não têm ranking)
    for rank_col, flag_col in [("Cf", "tem_ranking_cf"), ("Ct", "tem_ranking_ct")]:
        if rank_col in df.columns:
            df[flag_col] = df[rank_col].notna().astype(int)

    logger.info("Variáveis categóricas codificadas com sucesso")
    return df


def _encode_rec_avaliador(value) -> int:
    """Mapeia recomendações de avaliadores para valores numéricos."""
    if pd.isna(value) or not isinstance(value, str):
        return 0
    value_lower = value.lower().strip()
    for key, val in REC_AVALIADOR_MAP:
        if key.lower() in value_lower:
            return val
    return 0


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores ausentes no dataset.

    Estratégia:
    - Colunas numéricas com poucos nulos (Matem, Portug): preencher com mediana
    - Colunas com muitos nulos (Inglês, Pedras 20/21): manter NaN para XGBoost
    - Colunas categóricas: já tratadas em encode_categorical_columns

    Returns:
        DataFrame com nulos tratados.
    """
    df = df.copy()

    # Matem e Portug — pouquíssimos nulos (2 cada), preenchemos com a mediana
    for col in ["Matem", "Portug"]:
        if col in df.columns:
            median_val = df[col].median()
            n_nulls = df[col].isnull().sum()
            if n_nulls > 0:
                df[col] = df[col].fillna(median_val)
                logger.info(f"  {col}: {n_nulls} nulos preenchidos com mediana ({median_val:.2f})")

    # Inglês — criar flag binária (tem/não tem nota) e remover coluna de nota
    if "Inglês" in df.columns:
        df["Tem_nota_ingles"] = df["Inglês"].notna().astype(int)
        n_tem = df["Tem_nota_ingles"].sum()
        logger.info(f"  Inglês: {n_tem} alunos com nota → Tem_nota_ingles criada, coluna de nota removida")
        df = df.drop(columns=["Inglês"])

    logger.info("Tratamento de valores ausentes concluído")
    return df


def preprocess_dataset(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Pipeline completa de pré-processamento.

    Args:
        filepath: Caminho do dataset. Se None, usa o padrão.

    Returns:
        DataFrame pré-processado e pronto para feature engineering.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PRÉ-PROCESSAMENTO")
    logger.info("=" * 60)

    df = load_dataset(filepath)
    df = create_target_variable(df)
    df = encode_categorical_columns(df)
    df = handle_missing_values(df)

    logger.info(f"Pré-processamento concluído: {df.shape[0]} registros, {df.shape[1]} colunas")
    return df
