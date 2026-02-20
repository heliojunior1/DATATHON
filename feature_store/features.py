"""
Feature Views do Feature Store.

Organiza features em Feature Views por categoria, cada um vinculado
a um FileSource (Parquet) e à entidade Aluno.
"""
from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float64, Int64, String

from feature_store.entities import aluno
from feature_store.data_sources import (
    indicadores_pede_source,
    demograficas_source,
    notas_source,
    evolucao_pedras_source,
    flags_source,
    avaliadores_source,
    rankings_source,
    derivadas_source,
)

# TTL: features válidas por 365 dias (atualização anual do PEDE)
_TTL = timedelta(days=365)


# ── Indicadores PEDE ─────────────────────────────────────────────────

indicadores_pede_fv = FeatureView(
    name="indicadores_pede",
    entities=[aluno],
    ttl=_TTL,
    schema=[
        Field(name="IAA", dtype=Float64),
        Field(name="IEG", dtype=Float64),
        Field(name="IPS", dtype=Float64),
        Field(name="IDA", dtype=Float64),
        Field(name="IPV", dtype=Float64),
        Field(name="IAN", dtype=Float64),
        Field(name="INDE_22", dtype=Float64),
    ],
    source=indicadores_pede_source,
    online=True,
    tags={"category": "indicadores", "owner": "datathon-team"},
)


# ── Demográficas ─────────────────────────────────────────────────────

demograficas_fv = FeatureView(
    name="demograficas",
    entities=[aluno],
    ttl=_TTL,
    schema=[
        Field(name="Idade_22", dtype=Int64),
        Field(name="Genero_encoded", dtype=Int64),
        Field(name="Escola_encoded", dtype=Int64),
        Field(name="Fase_encoded", dtype=Int64),
        Field(name="Anos_na_PM", dtype=Int64),
    ],
    source=demograficas_source,
    online=True,
    tags={"category": "demograficas", "owner": "datathon-team"},
)


# ── Notas ────────────────────────────────────────────────────────────

notas_fv = FeatureView(
    name="notas",
    entities=[aluno],
    ttl=_TTL,
    schema=[
        Field(name="Matem", dtype=Float64),
        Field(name="Portug", dtype=Float64),
        Field(name="Tem_nota_ingles", dtype=Int64),
    ],
    source=notas_source,
    online=True,
    tags={"category": "notas", "owner": "datathon-team"},
)


# ── Evolução Pedras ──────────────────────────────────────────────────

evolucao_pedras_fv = FeatureView(
    name="evolucao_pedras",
    entities=[aluno],
    ttl=_TTL,
    schema=[
        Field(name="Pedra_20_encoded", dtype=Int64),
        Field(name="Pedra_21_encoded", dtype=Int64),
        Field(name="Pedra_22_encoded", dtype=Int64),
        Field(name="Evolucao_pedra_20_22", dtype=Float64),
        Field(name="Evolucao_pedra_21_22", dtype=Float64),
        Field(name="tinha_pedra_20", dtype=Int64),
        Field(name="tinha_pedra_21", dtype=Int64),
    ],
    source=evolucao_pedras_source,
    online=True,
    tags={"category": "evolucao_pedras", "owner": "datathon-team"},
)


# ── Flags ────────────────────────────────────────────────────────────

flags_fv = FeatureView(
    name="flags",
    entities=[aluno],
    ttl=_TTL,
    schema=[
        Field(name="Ponto_virada_flag", dtype=Int64),
        Field(name="Indicado_flag", dtype=Int64),
        Field(name="Destaque_IEG_flag", dtype=Int64),
        Field(name="Destaque_IDA_flag", dtype=Int64),
        Field(name="Destaque_IPV_flag", dtype=Int64),
    ],
    source=flags_source,
    online=True,
    tags={"category": "flags", "owner": "datathon-team"},
)


# ── Avaliadores ──────────────────────────────────────────────────────

avaliadores_fv = FeatureView(
    name="avaliadores",
    entities=[aluno],
    ttl=_TTL,
    schema=[
        Field(name="Rec_av1_encoded", dtype=Int64),
        Field(name="Rec_av2_encoded", dtype=Int64),
    ],
    source=avaliadores_source,
    online=True,
    tags={"category": "avaliadores", "owner": "datathon-team"},
)


# ── Rankings ─────────────────────────────────────────────────────────

rankings_fv = FeatureView(
    name="rankings",
    entities=[aluno],
    ttl=_TTL,
    schema=[
        Field(name="Cf", dtype=Float64),
        Field(name="Ct", dtype=Float64),
        Field(name="N_Av", dtype=Int64),
        Field(name="tem_ranking_cf", dtype=Int64),
        Field(name="tem_ranking_ct", dtype=Int64),
    ],
    source=rankings_source,
    online=True,
    tags={"category": "rankings", "owner": "datathon-team"},
)


# ── Derivadas ────────────────────────────────────────────────────────

derivadas_fv = FeatureView(
    name="derivadas",
    entities=[aluno],
    ttl=_TTL,
    schema=[
        Field(name="Variancia_indicadores", dtype=Float64),
        Field(name="Ratio_IDA_IEG", dtype=Float64),
    ],
    source=derivadas_source,
    online=True,
    tags={"category": "derivadas", "owner": "datathon-team"},
)


# ── Lista de todas as Feature Views (para registro) ─────────────────

ALL_FEATURE_VIEWS = [
    indicadores_pede_fv,
    demograficas_fv,
    notas_fv,
    evolucao_pedras_fv,
    flags_fv,
    avaliadores_fv,
    rankings_fv,
    derivadas_fv,
]

# Mapeamento: nome da coluna no DataFrame original → nome no Feature Store
# (colunas com caracteres especiais são renomeadas)
COLUMN_RENAME_MAP = {
    "INDE 22": "INDE_22",
    "Idade 22": "Idade_22",
    "Nº Av": "N_Av",
}

# Inverso: Feature Store → DataFrame original
COLUMN_RENAME_INVERSE = {v: k for k, v in COLUMN_RENAME_MAP.items()}
