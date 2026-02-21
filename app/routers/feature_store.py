"""
Router do Feature Store.

Endpoints:
- GET  /feature-store/status                  — Status do Feature Store
- POST /feature-store/materialize             — Materializar features
- GET  /feature-store/features/{aluno_id}     — Buscar features de um aluno
"""
from fastapi import APIRouter, HTTPException

from app.utils.helpers import setup_logger
from app.utils.error_handlers import handle_route_errors

logger = setup_logger(__name__)

router = APIRouter()


@router.get("/feature-store/status", tags=["Feature Store"])
@handle_route_errors(logger)
async def feature_store_status():
    """Retorna o status do Feature Store (registry, online store, feature views)."""
    from feature_store.feature_store_manager import FeatureStoreManager
    from app.models.feature_store import FeatureStoreStatusResponse

    manager = FeatureStoreManager()
    status = manager.get_status()
    return FeatureStoreStatusResponse(**status)


@router.post("/feature-store/materialize", tags=["Feature Store"])
@handle_route_errors(logger)
async def feature_store_materialize():
    """
    Executa materialização completa: preprocess → feature engineering →
    ingestão Parquet → apply → materialize (SQLite).
    """
    from feature_store.feature_store_manager import FeatureStoreManager
    from app.services.ml.preprocessing import preprocess_dataset
    from app.services.ml.feature_engineering import run_feature_engineering
    from app.models.feature_store import FeatureStoreMaterializeResponse

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


@router.get("/feature-store/features/{aluno_id}", tags=["Feature Store"])
@handle_route_errors(logger)
async def feature_store_get_features(aluno_id: str):
    """Busca features de um aluno no online store (SQLite)."""
    from feature_store.feature_store_manager import FeatureStoreManager
    from app.models.feature_store import FeatureStoreFeaturesResponse

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
    features.pop("aluno_id", None)

    return FeatureStoreFeaturesResponse(
        aluno_id=aluno_id,
        features=features,
    )
