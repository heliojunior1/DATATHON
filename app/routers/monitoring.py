"""
Router de Monitoramento.

Endpoints:
- GET /monitoring/drift    — Status de data drift (KS-test, PSI, KL Divergence)
- GET /monitoring/stats    — Estatísticas das predições
- GET /monitoring/latency  — Métricas de latência de inferência
- GET /monitoring/missing  — Taxa de valores ausentes por feature
- GET /monitoring/system   — Métricas de CPU e RAM
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import DriftResponse
from app.services.drift_service import (
    check_all_drift,
    get_prediction_stats,
    get_latency_stats,
    get_missing_values_stats,
    get_throughput,
)
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get("/monitoring/drift", response_model=DriftResponse, tags=["Monitoramento"])
async def drift_status():
    """Verifica status de data drift (KS-test, PSI, KL Divergence) por feature."""
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


@router.get("/monitoring/latency", tags=["Monitoramento"])
async def latency_stats():
    """Retorna estatísticas de latência de inferência (avg, p50, p95, p99 em ms)."""
    return get_latency_stats()


@router.get("/monitoring/throughput", tags=["Monitoramento"])
async def throughput_stats(window_minutes: int = 60):
    """Retorna throughput de requisições na janela de tempo especificada."""
    return get_throughput(window_minutes=window_minutes)


@router.get("/monitoring/missing", tags=["Monitoramento"])
async def missing_values_stats():
    """Retorna taxa de valores ausentes (None/NaN) por feature nas predições."""
    return get_missing_values_stats()


@router.get("/monitoring/system", tags=["Monitoramento"])
async def system_metrics():
    """Retorna métricas de uso de CPU e RAM do processo atual."""
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": round(process.memory_percent(), 2),
            "memory_used_mb": round(mem.rss / 1024 ** 2, 1),
            "memory_available_mb": round(psutil.virtual_memory().available / 1024 ** 2, 1),
        }
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="psutil não instalado. Execute: pip install psutil",
        )
    except Exception as e:
        logger.error(f"Erro ao coletar métricas do sistema: {e}")
        raise HTTPException(status_code=500, detail=str(e))
