"""
Gerenciador centralizado do Feature Store.

Fornece interface de alto nível para ingestão, materialização e
consulta de features via Feast.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Adicionar raiz do projeto ao path para imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from feast import FeatureStore
from feast.repo_config import RepoConfig

from feature_store.features import (
    ALL_FEATURE_VIEWS,
    COLUMN_RENAME_MAP,
    COLUMN_RENAME_INVERSE,
)
from feature_store.entities import aluno


class FeatureStoreManager:
    """
    Interface de alto nível para o Feature Store (Feast).

    Responsabilidades:
    - Ingestão de features em Parquet (offline store)
    - Registro de Feature Views no Feast registry
    - Materialização de features no online store (SQLite)
    - Consulta de features históricas (treinamento) e online (inferência)
    """

    # Mapeamento: nome do Feature View → colunas do DataFrame original
    _FV_COLUMNS = {
        "indicadores_pede": ["IAA", "IEG", "IPS", "IDA", "IPV", "IAN", "INDE 22"],
        "demograficas": [
            "Idade 22", "Genero_encoded", "Escola_encoded",
            "Fase_encoded", "Anos_na_PM",
        ],
        "notas": ["Matem", "Portug", "Tem_nota_ingles"],
        "evolucao_pedras": [
            "Pedra_20_encoded", "Pedra_21_encoded", "Pedra_22_encoded",
            "Evolucao_pedra_20_22", "Evolucao_pedra_21_22",
            "tinha_pedra_20", "tinha_pedra_21",
        ],
        "flags": [
            "Ponto_virada_flag", "Indicado_flag",
            "Destaque_IEG_flag", "Destaque_IDA_flag", "Destaque_IPV_flag",
        ],
        "avaliadores": ["Rec_av1_encoded", "Rec_av2_encoded"],
        "rankings": ["Cf", "Ct", "Nº Av", "tem_ranking_cf", "tem_ranking_ct"],
        "derivadas": ["Variancia_indicadores", "Ratio_IDA_IEG"],
    }

    def __init__(self, repo_path: str | Path | None = None):
        """
        Inicializa o gerenciador.

        Args:
            repo_path: Caminho do repositório Feast. Se None, usa o padrão.
        """
        self._repo_path = Path(repo_path) if repo_path else Path(__file__).resolve().parent
        self._data_dir = self._repo_path / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._store: FeatureStore | None = None

    @property
    def store(self) -> FeatureStore:
        """Retorna a instância do FeatureStore (lazy initialization)."""
        if self._store is None:
            self._store = FeatureStore(repo_path=str(self._repo_path))
        return self._store

    @property
    def data_dir(self) -> Path:
        """Retorna o diretório de dados Parquet."""
        return self._data_dir

    # ── Ingestão ─────────────────────────────────────────────────────

    def ingest_features(
        self,
        df: pd.DataFrame,
        event_timestamp: datetime | None = None,
    ) -> dict[str, Path]:
        """
        Converte o DataFrame (pós feature engineering) em arquivos Parquet
        por categoria e salva no diretório de dados do Feature Store.

        Args:
            df: DataFrame com features (após preprocessing + feature engineering).
            event_timestamp: Timestamp do evento. Se None, usa datetime atual.

        Returns:
            Dicionário {nome_fv: path_parquet}.
        """
        if event_timestamp is None:
            event_timestamp = datetime.now()

        # Extrair aluno_id do DataFrame (coluna RA)
        if "RA" not in df.columns:
            raise ValueError(
                "DataFrame precisa ter coluna 'RA' para identificar alunos. "
                "Verifique se o pré-processamento mantém esta coluna."
            )

        created_files = {}

        for fv_name, columns in self._FV_COLUMNS.items():
            # Filtrar apenas colunas disponíveis
            available_cols = [c for c in columns if c in df.columns]
            if not available_cols:
                continue

            # Montar DataFrame para o Parquet
            fv_df = df[["RA"] + available_cols].copy()
            fv_df = fv_df.rename(columns={"RA": "aluno_id"})

            # Renomear colunas com caracteres especiais
            fv_df = fv_df.rename(columns=COLUMN_RENAME_MAP)

            # Adicionar timestamp do evento
            fv_df["event_timestamp"] = pd.Timestamp(event_timestamp)

            # Garantir tipos numéricos (NaN → 0 para colunas inteiras)
            for col in fv_df.columns:
                if col in ("aluno_id", "event_timestamp"):
                    continue
                if fv_df[col].dtype == "object":
                    fv_df[col] = pd.to_numeric(fv_df[col], errors="coerce")
                if fv_df[col].isna().any():
                    fv_df[col] = fv_df[col].fillna(0)

            # Salvar Parquet
            parquet_path = self._data_dir / f"{fv_name}.parquet"
            fv_df.to_parquet(parquet_path, index=False)
            created_files[fv_name] = parquet_path

        return created_files

    # ── Registry ─────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        Registra entidades e Feature Views no registry do Feast.
        Equivalente a `feast apply`.

        Constrói Feature Views dinamicamente para usar os caminhos
        corretos dos arquivos Parquet (baseado no data_dir do manager).
        """
        from feast import FeatureView, FileSource

        dynamic_fvs = []
        for fv in ALL_FEATURE_VIEWS:
            # Criar FileSource apontando para o data_dir deste manager
            source = FileSource(
                name=f"{fv.name}_source",
                path=str(self._data_dir / f"{fv.name}.parquet"),
                timestamp_field="event_timestamp",
            )
            # Recriar FeatureView com o novo source e entidades originais
            new_fv = FeatureView(
                name=fv.name,
                entities=[aluno],
                ttl=fv.ttl,
                schema=list(fv.schema),
                source=source,
                online=fv.online,
                tags=dict(fv.tags) if fv.tags else None,
            )
            dynamic_fvs.append(new_fv)

        objects = [aluno] + dynamic_fvs
        self.store.apply(objects)

    # ── Materialização ───────────────────────────────────────────────

    def materialize(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> None:
        """
        Materializa features do offline store (Parquet) para o online store (SQLite).

        Args:
            start_date: Data de início. Se None, usa 1 ano atrás.
            end_date: Data de fim. Se None, usa agora.
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        self.store.materialize(
            start_date=start_date,
            end_date=end_date,
        )

    def materialize_incremental(self, end_date: datetime | None = None) -> None:
        """
        Materialização incremental (apenas dados novos desde a última materialização).
        """
        if end_date is None:
            end_date = datetime.now()
        self.store.materialize_incremental(end_date=end_date)

    # ── Consulta de Features ─────────────────────────────────────────

    def get_training_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Busca features históricas para treinamento.

        Args:
            entity_df: DataFrame com colunas ['aluno_id', 'event_timestamp'].
            feature_refs: Lista de referências de features no formato
                         'feature_view:feature'. Se None, busca todas.

        Returns:
            DataFrame unificado com todas as features solicitadas.
        """
        if feature_refs is None:
            feature_refs = self._get_all_feature_refs()

        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        ).to_df()

        # Reverter renomeação de colunas
        training_df = training_df.rename(columns=COLUMN_RENAME_INVERSE)

        return training_df

    def get_online_features(
        self,
        entity_ids: list[str],
        feature_refs: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Busca features online (versão mais recente) para inferência.

        Args:
            entity_ids: Lista de aluno_ids (RAs).
            feature_refs: Lista de referências de features. Se None, busca todas.

        Returns:
            DataFrame com features online.
        """
        if feature_refs is None:
            feature_refs = self._get_all_feature_refs()

        entity_rows = [{"aluno_id": aid} for aid in entity_ids]

        online_response = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )

        result_df = online_response.to_df()

        # Reverter renomeação de colunas
        result_df = result_df.rename(columns=COLUMN_RENAME_INVERSE)

        return result_df

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Retorna informações de status do Feature Store."""
        status = {
            "repo_path": str(self._repo_path),
            "data_dir": str(self._data_dir),
            "online_store_exists": (self._data_dir / "online_store.db").exists(),
            "registry_exists": (self._data_dir / "registry.db").exists(),
            "parquet_files": [],
            "feature_views": [],
        }

        # Listar Parquet files
        for pf in sorted(self._data_dir.glob("*.parquet")):
            status["parquet_files"].append({
                "name": pf.name,
                "size_bytes": pf.stat().st_size,
            })

        # Listar Feature Views registradas
        try:
            fvs = self.store.list_feature_views()
            for fv in fvs:
                status["feature_views"].append({
                    "name": fv.name,
                    "features": [f.name for f in fv.features],
                    "tags": dict(fv.tags) if fv.tags else {},
                })
        except Exception:
            status["feature_views_error"] = "Registry not initialized. Run 'apply()' first."

        return status

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_all_feature_refs(self) -> list[str]:
        """Retorna todas as referências de features de todas as Feature Views."""
        refs = []
        for fv in ALL_FEATURE_VIEWS:
            for feat in fv.features:
                refs.append(f"{fv.name}:{feat.name}")
        return refs

    def is_populated(self) -> bool:
        """Verifica se o Feature Store tem dados Parquet e registry."""
        has_parquet = any(self._data_dir.glob("*.parquet"))
        has_registry = (self._data_dir / "registry.db").exists()
        return has_parquet and has_registry
