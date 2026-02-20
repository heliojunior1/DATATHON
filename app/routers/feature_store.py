"""
Router do Feature Store.

Endpoints:
- GET  /feature-store/status                  — Status do Feature Store
- POST /feature-store/materialize             — Materializar features
- GET  /feature-store/features/{aluno_id}     — Buscar features de um aluno
"""
from fastapi import APIRouter, HTTPException

from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get("/feature-store/status", tags=["Feature Store"])
async def feature_store_status():
    """Retorna o status do Feature Store (registry, online store, feature views)."""
    try:
        from feature_store.feature_store_manager import FeatureStoreManager
        from app.models.schemas import FeatureStoreStatusResponse

        manager = FeatureStoreManager()
        status = manager.get_status()
        return FeatureStoreStatusResponse(**status)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Feast não instalado. Execute: pip install feast",
        )
    except Exception as e:
        logger.error(f"Erro ao obter status do Feature Store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feature-store/materialize", tags=["Feature Store"])
async def feature_store_materialize():
    """
    Executa materialização completa: preprocess → feature engineering →
    ingestão Parquet → apply → materialize (SQLite).
    """
    try:
        from feature_store.feature_store_manager import FeatureStoreManager
        from app.services.preprocessing import preprocess_dataset
        from app.services.feature_engineering import run_feature_engineering
        from app.models.schemas import FeatureStoreMaterializeResponse

        # Pipeline completa
        df = preprocess_dataset()
        df = run_feature_engineering(df)

        manager = FeatureStoreManager()
        manager.ingest_features(df)
        manager.apply()
        manager.materialize()

        status = manager.get_status()
        return FeatureStoreMaterializeResponse(
            message="Materialização concluída com sucesso!",
            parquet_files=len(status.get("parquet_files", [])),
            feature_views=len(status.get("feature_views", [])),
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Feast não instalado. Execute: pip install feast",
        )
    except Exception as e:
        logger.error(f"Erro na materialização: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-store/features/{aluno_id}", tags=["Feature Store"])
async def feature_store_get_features(aluno_id: str):
    """Busca features de um aluno no online store (SQLite)."""
    try:
        from feature_store.feature_store_manager import FeatureStoreManager
        from app.models.schemas import FeatureStoreFeaturesResponse

        manager = FeatureStoreManager()
        if not manager.is_populated():
            raise HTTPException(
                status_code=404,
                detail="Feature Store não populado. Execute a materialização primeiro.",
            )

        features_df = manager.get_online_features([aluno_id])
        if features_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Aluno '{aluno_id}' não encontrado no Feature Store.",
            )

        features = features_df.iloc[0].to_dict()
        # Remover aluno_id do dict de features
        features.pop("aluno_id", None)

        return FeatureStoreFeaturesResponse(
            aluno_id=aluno_id,
            features=features,
        )
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Feast não instalado. Execute: pip install feast",
        )
    except Exception as e:
        logger.error(f"Erro ao buscar features: {e}")
        raise HTTPException(status_code=500, detail=str(e))
