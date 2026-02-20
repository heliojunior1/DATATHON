"""
Data Sources para o Feature Store.

Define os FileSource (Parquet) para cada grupo de features.
"""
from pathlib import Path
from feast import FileSource

# Diretório base dos dados Parquet
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _parquet_source(name: str, timestamp_field: str = "event_timestamp") -> FileSource:
    """Cria um FileSource Parquet padronizado."""
    return FileSource(
        name=f"{name}_source",
        path=str(_DATA_DIR / f"{name}.parquet"),
        timestamp_field=timestamp_field,
    )


# ── Data Sources por categoria ──────────────────────────────────────

indicadores_pede_source = _parquet_source("indicadores_pede")
demograficas_source = _parquet_source("demograficas")
notas_source = _parquet_source("notas")
evolucao_pedras_source = _parquet_source("evolucao_pedras")
flags_source = _parquet_source("flags")
avaliadores_source = _parquet_source("avaliadores")
rankings_source = _parquet_source("rankings")
derivadas_source = _parquet_source("derivadas")
