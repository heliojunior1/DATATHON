"""
Router de Monitoramento.

Endpoints:
- GET /monitoring/drift           — Status de data drift (KS-test, PSI, KL Divergence)
- GET /monitoring/stats           — Estatísticas das predições
- GET /monitoring/latency         — Métricas de latência de inferência
- GET /monitoring/missing         — Taxa de valores ausentes por feature
- GET /monitoring/system          — Métricas de CPU e RAM
- GET /monitoring/feedback        — Lista predições para confirmação de outcome
- POST /monitoring/feedback       — Registra outcome real de uma predição
- GET /monitoring/concept-drift   — Detecta degradação de performance via feedback loop
"""
from fastapi import APIRouter, HTTPException

from app.models.monitoring import DriftResponse, FeedbackRequest, FeedbackResponse
from app.services.monitoring.drift_service import (
    check_all_drift,
    get_prediction_stats,
    get_latency_stats,
    get_missing_values_stats,
    get_throughput,
    submit_feedback,
    get_predictions_for_feedback,
    get_concept_drift_stats,
)
from app.utils.helpers import setup_logger
from app.utils.error_handlers import handle_route_errors

logger = setup_logger(__name__)

router = APIRouter()


@router.get("/monitoring/drift", response_model=DriftResponse, tags=["Monitoramento"])
@handle_route_errors(logger)
async def drift_status():
    """Verifica status de data drift (KS-test, PSI, KL Divergence) por feature."""
    result = check_all_drift()
    return DriftResponse(**result)


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
@handle_route_errors(logger)
async def system_metrics():
    """Retorna métricas de uso de CPU e RAM do processo atual."""
    import psutil
    process = psutil.Process()
    mem = process.memory_info()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": round(process.memory_percent(), 2),
        "memory_used_mb": round(mem.rss / 1024 ** 2, 1),
        "memory_available_mb": round(psutil.virtual_memory().available / 1024 ** 2, 1),
    }


# ── Concept Drift — Feedback Loop ────────────────────────────────────────────


@router.get("/monitoring/feedback", tags=["Concept Drift"])
async def list_predictions_for_feedback(limit: int = 50):
    """Lista predições recentes aguardando confirmação de outcome real pelo usuário."""
    return {"predictions": get_predictions_for_feedback(limit)}


@router.post("/monitoring/feedback", response_model=FeedbackResponse, tags=["Concept Drift"])
async def submit_prediction_feedback(request: FeedbackRequest):
    """
    Registra o outcome real de uma predição.

    Use este endpoint para informar se um aluno ficou defasado ou não,
    permitindo o cálculo de concept drift ao longo do tempo.
    """
    success = submit_feedback(request.prediction_id, request.actual_outcome)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"prediction_id '{request.prediction_id}' não encontrado.",
        )
    return FeedbackResponse(
        success=True,
        message="Feedback registrado com sucesso.",
        prediction_id=request.prediction_id,
    )


@router.get("/monitoring/concept-drift", tags=["Concept Drift"])
@handle_route_errors(logger)
async def concept_drift_status(window_size: int = 20):
    """
    Detecta concept drift comparando F1/Recall entre janelas temporais de predições confirmadas.

    Requer pelo menos 5 feedbacks confirmados via POST /monitoring/feedback.
    """
    return get_concept_drift_stats(window_size)
