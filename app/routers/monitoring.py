"""
Router de Monitoramento.

Endpoints:
- GET /monitoring/drift  — Status de data drift
- GET /monitoring/stats  — Estatísticas das predições
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import DriftResponse
from app.services.drift_service import check_all_drift, get_prediction_stats
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get("/monitoring/drift", response_model=DriftResponse, tags=["Monitoramento"])
async def drift_status():
    """Verifica status de data drift comparando dados de produção com dados de treinamento."""
    try:
        result = check_all_drift()
        return DriftResponse(**result)
    except Exception as e:
        logger.error(f"Erro no monitoramento de drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/stats", tags=["Monitoramento"])
async def prediction_stats():
    """Retorna estatísticas das predições realizadas desde o último restart."""
    return get_prediction_stats()
