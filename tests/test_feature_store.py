"""
Testes do Feature Store (Feast + SQLite).

Testa ingestão de features, registro no Feast, e consultas.
Usa um diretório temporário para isolamento.
"""
import sys
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Garantir que o diretório raiz está no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_engineered_df():
    """DataFrame simulando dados após preprocessing + feature engineering."""
    np.random.seed(42)
    n = 20

    return pd.DataFrame({
        "RA": [f"RA-{i:03d}" for i in range(n)],
        # Indicadores PEDE
        "IAA": np.round(np.random.uniform(3, 10, n), 2),
        "IEG": np.round(np.random.uniform(2, 10, n), 2),
        "IPS": np.round(np.random.uniform(4, 10, n), 2),
        "IDA": np.round(np.random.uniform(2, 10, n), 2),
        "IPV": np.round(np.random.uniform(1, 10, n), 2),
        "IAN": np.random.choice([0.0, 5.0, 10.0], n),
        "INDE 22": np.round(np.random.uniform(2.5, 9.5, n), 3),
        # Demográficas
        "Idade 22": np.random.randint(7, 20, n),
        "Genero_encoded": np.random.choice([0, 1], n),
        "Escola_encoded": np.random.choice([0, 1, 2], n),
        "Fase_encoded": np.random.randint(0, 9, n),
        "Anos_na_PM": np.random.randint(0, 7, n),
        # Notas
        "Matem": np.round(np.random.uniform(2, 10, n), 2),
        "Portug": np.round(np.random.uniform(2, 10, n), 2),
        "Tem_nota_ingles": np.random.choice([0, 1], n),
        # Evolução Pedras
        "Pedra_20_encoded": np.random.choice([0, 1, 2, 3, 4], n),
        "Pedra_21_encoded": np.random.choice([0, 1, 2, 3, 4], n),
        "Pedra_22_encoded": np.random.choice([1, 2, 3, 4], n),
        "Evolucao_pedra_20_22": np.random.randint(-2, 3, n).astype(float),
        "Evolucao_pedra_21_22": np.random.randint(-2, 3, n).astype(float),
        "tinha_pedra_20": np.random.choice([0, 1], n),
        "tinha_pedra_21": np.random.choice([0, 1], n),
        # Flags
        "Ponto_virada_flag": np.random.choice([0, 1], n),
        "Indicado_flag": np.random.choice([0, 1], n),
        "Destaque_IEG_flag": np.random.choice([0, 1], n),
        "Destaque_IDA_flag": np.random.choice([0, 1], n),
        "Destaque_IPV_flag": np.random.choice([0, 1], n),
        # Avaliadores
        "Rec_av1_encoded": np.random.choice([0, 1, 2, 3, 4], n),
        "Rec_av2_encoded": np.random.choice([0, 1, 2, 3, 4], n),
        # Rankings
        "Cf": np.random.randint(1, 193, n).astype(float),
        "Ct": np.random.randint(1, 19, n).astype(float),
        "Nº Av": np.random.choice([2, 3, 4], n),
        "tem_ranking_cf": np.ones(n, dtype=int),
        "tem_ranking_ct": np.ones(n, dtype=int),
        # Derivadas
        "Variancia_indicadores": np.round(np.random.uniform(0.5, 3.0, n), 2),
        "Ratio_IDA_IEG": np.round(np.random.uniform(0.2, 5.0, n), 2),
        # Target
        "risco_defasagem": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def temp_feature_store(tmp_path, sample_engineered_df):
    """Configura um Feature Store temporário para testes."""
    fs_dir = tmp_path / "feature_store"
    fs_dir.mkdir()
    data_dir = fs_dir / "data"
    data_dir.mkdir()

    # Copiar feature_store.yaml
    project_root = Path(__file__).resolve().parent.parent
    src_yaml = project_root / "feature_store" / "feature_store.yaml"
    if src_yaml.exists():
        shutil.copy(src_yaml, fs_dir / "feature_store.yaml")
    else:
        # Criar YAML mínimo
        (fs_dir / "feature_store.yaml").write_text(
            "project: datathon_pede_test\n"
            "provider: local\n"
            "registry:\n"
            "  path: data/registry.db\n"
            "online_store:\n"
            "  type: sqlite\n"
            "  path: data/online_store.db\n"
            "offline_store:\n"
            "  type: file\n"
            "entity_key_serialization_version: 3\n"
        )

    return fs_dir


class TestFeatureStoreManager:
    """Testes para o FeatureStoreManager."""

    def test_import(self):
        """Verifica que FeatureStoreManager pode ser importado."""
        from feature_store.feature_store_manager import FeatureStoreManager
        assert FeatureStoreManager is not None

    def test_initialization(self, temp_feature_store):
        """Verifica inicialização do manager."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)
        assert manager.data_dir.exists()

    def test_ingest_features(self, temp_feature_store, sample_engineered_df):
        """Verifica ingestão de features em Parquet."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        created_files = manager.ingest_features(sample_engineered_df)

        assert len(created_files) > 0
        for fv_name, path in created_files.items():
            assert path.exists(), f"Parquet {fv_name} não foi criado"
            assert path.suffix == ".parquet"

            # Verificar conteúdo
            df = pd.read_parquet(path)
            assert "aluno_id" in df.columns
            assert "event_timestamp" in df.columns
            assert len(df) == len(sample_engineered_df)

    def test_ingest_all_categories(self, temp_feature_store, sample_engineered_df):
        """Verifica que todas as 8 categorias são ingeridas."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        created_files = manager.ingest_features(sample_engineered_df)

        expected_categories = [
            "indicadores_pede", "demograficas", "notas",
            "evolucao_pedras", "flags", "avaliadores",
            "rankings", "derivadas",
        ]
        for cat in expected_categories:
            assert cat in created_files, f"Categoria '{cat}' não foi ingerida"

    def test_ingest_column_rename(self, temp_feature_store, sample_engineered_df):
        """Verifica que colunas com caracteres especiais são renomeadas."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        created_files = manager.ingest_features(sample_engineered_df)

        # INDE 22 → INDE_22
        ind_df = pd.read_parquet(created_files["indicadores_pede"])
        assert "INDE_22" in ind_df.columns
        assert "INDE 22" not in ind_df.columns

        # Idade 22 → Idade_22
        dem_df = pd.read_parquet(created_files["demograficas"])
        assert "Idade_22" in dem_df.columns

        # Nº Av → N_Av
        rank_df = pd.read_parquet(created_files["rankings"])
        assert "N_Av" in rank_df.columns

    def test_ingest_requires_ra_column(self, temp_feature_store):
        """Verifica que ingestão falha sem coluna RA."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        df_no_ra = pd.DataFrame({"IAA": [5.0], "IEG": [6.0]})
        with pytest.raises(ValueError, match="RA"):
            manager.ingest_features(df_no_ra)

    def test_apply_and_materialize(self, temp_feature_store, sample_engineered_df):
        """Verifica fluxo completo: ingest → apply → materialize."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        # Ingestar
        manager.ingest_features(sample_engineered_df)

        # Apply (registrar Feature Views)
        manager.apply()

        # Verificar registry criado
        registry_path = manager.data_dir / "registry.db"
        assert registry_path.exists()

        # Materializar
        manager.materialize()

        # Verificar online store criado
        online_store_path = manager.data_dir / "online_store.db"
        assert online_store_path.exists()

    def test_get_status_empty(self, temp_feature_store):
        """Verifica status com Feature Store vazio."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        status = manager.get_status()
        assert status["online_store_exists"] is False
        assert status["registry_exists"] is False
        assert len(status["parquet_files"]) == 0

    def test_get_status_after_ingest(self, temp_feature_store, sample_engineered_df):
        """Verifica status após ingestão."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        manager.ingest_features(sample_engineered_df)
        status = manager.get_status()

        assert len(status["parquet_files"]) == 8

    def test_is_populated_false(self, temp_feature_store):
        """Verifica is_populated retorna False quando vazio."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)
        assert manager.is_populated() is False

    def test_is_populated_true(self, temp_feature_store, sample_engineered_df):
        """Verifica is_populated retorna True após ingest + apply."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        manager.ingest_features(sample_engineered_df)
        manager.apply()

        assert manager.is_populated() is True

    def test_online_features_after_materialize(
        self, temp_feature_store, sample_engineered_df
    ):
        """Verifica que features online são retornadas após materialização."""
        from feature_store.feature_store_manager import FeatureStoreManager
        manager = FeatureStoreManager(repo_path=temp_feature_store)

        manager.ingest_features(sample_engineered_df)
        manager.apply()
        manager.materialize()

        # Buscar features do primeiro aluno
        aluno_id = "RA-000"
        features_df = manager.get_online_features([aluno_id])

        assert not features_df.empty
        assert "aluno_id" in features_df.columns


class TestFeatureStoreEntities:
    """Testes para definições de entidades."""

    def test_aluno_entity(self):
        """Verifica entidade Aluno."""
        from feature_store.entities import aluno
        assert aluno.name == "aluno"
        assert aluno.join_key == "aluno_id"


class TestFeatureViews:
    """Testes para definições de Feature Views."""

    def test_all_feature_views_defined(self):
        """Verifica que todas as Feature Views estão definidas."""
        from feature_store.features import ALL_FEATURE_VIEWS
        assert len(ALL_FEATURE_VIEWS) == 8

    def test_feature_view_names(self):
        """Verifica nomes das Feature Views."""
        from feature_store.features import ALL_FEATURE_VIEWS
        names = [fv.name for fv in ALL_FEATURE_VIEWS]
        expected = [
            "indicadores_pede", "demograficas", "notas",
            "evolucao_pedras", "flags", "avaliadores",
            "rankings", "derivadas",
        ]
        for expected_name in expected:
            assert expected_name in names, f"Feature View '{expected_name}' não encontrada"

    def test_column_rename_map(self):
        """Verifica mapeamento de renomeação de colunas."""
        from feature_store.features import COLUMN_RENAME_MAP, COLUMN_RENAME_INVERSE
        assert "INDE 22" in COLUMN_RENAME_MAP
        assert COLUMN_RENAME_MAP["INDE 22"] == "INDE_22"
        assert "INDE_22" in COLUMN_RENAME_INVERSE
        assert COLUMN_RENAME_INVERSE["INDE_22"] == "INDE 22"

    def test_feature_views_have_entity(self):
        """Verifica que todas as Feature Views referenciam a entidade aluno."""
        from feature_store.features import ALL_FEATURE_VIEWS
        for fv in ALL_FEATURE_VIEWS:
            # In Feast 0.60.0, entities are stored as strings
            entity_names = [
                e.name if hasattr(e, 'name') else str(e)
                for e in fv.entities
            ]
            assert "aluno" in entity_names, (
                f"Feature View '{fv.name}' não tem entidade 'aluno'"
            )

    def test_feature_views_online_enabled(self):
        """Verifica que todas as Feature Views têm online=True."""
        from feature_store.features import ALL_FEATURE_VIEWS
        for fv in ALL_FEATURE_VIEWS:
            assert fv.online is True, (
                f"Feature View '{fv.name}' deveria ter online=True"
            )


class TestDataSources:
    """Testes para definições de Data Sources."""

    def test_all_sources_defined(self):
        """Verifica que todos os Data Sources existem."""
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
        sources = [
            indicadores_pede_source, demograficas_source, notas_source,
            evolucao_pedras_source, flags_source, avaliadores_source,
            rankings_source, derivadas_source,
        ]
        assert len(sources) == 8
        for src in sources:
            assert src is not None


class TestConfig:
    """Testes para configuração do Feature Store."""

    def test_feature_store_config_exists(self):
        """Verifica que configuração do Feature Store existe em config.py."""
        from app.config import (
            FEATURE_STORE_DIR,
            FEATURE_STORE_DATA_DIR,
            FEATURE_STORE_ONLINE_DB,
            USE_FEATURE_STORE,
        )
        assert FEATURE_STORE_DIR is not None
        assert FEATURE_STORE_DATA_DIR is not None
        assert FEATURE_STORE_ONLINE_DB is not None
        assert isinstance(USE_FEATURE_STORE, bool)

    def test_use_feature_store_default_false(self):
        """Verifica que USE_FEATURE_STORE é False por padrão."""
        from app.config import USE_FEATURE_STORE
        # Padrão: False (ativado via env var USE_FEATURE_STORE=true)
        assert USE_FEATURE_STORE is False
