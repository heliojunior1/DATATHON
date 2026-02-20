"""
Router de Predição.

Endpoints:
- GET  /health              — Health check
- POST /predict             — Previsão para um aluno
- POST /predict/batch       — Previsão em lote
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    StudentInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from app.services.predict_service import predict, predict_batch, load_model
from app.services.drift_service import log_prediction
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Verifica se a API e o modelo estão funcionando."""
    try:
        model, metadata = load_model()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=metadata.get("model_type", metadata.get("model_name")),
            model_version=metadata.get("model_id"),
        )
    except FileNotFoundError:
        return HealthResponse(
            status="model_not_loaded",
            model_loaded=False,
        )
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return HealthResponse(
            status="error",
            model_loaded=False,
        )


@router.post("/predict", response_model=PredictionResponse, tags=["Predição"])
async def predict_student(student: StudentInput, model_id: str | None = None):
    """
    Faz previsão de risco de defasagem para um aluno.

    Args:
        student: Dados do aluno.
        model_id: ID do modelo a usar (query param). Se None, usa o mais recente.
    """
    try:
        input_data = student.model_dump(by_alias=True)
        result = predict(input_data, model_id=model_id)

        # Registrar para monitoramento
        log_prediction(input_data, result)

        logger.info(
            f"Predição realizada: risco={result['prediction']}, "
            f"prob={result['probability']:.4f}, "
            f"nível={result['risk_level']}, "
            f"modelo={result.get('model_id')}"
        )

        return PredictionResponse(**result)

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro.",
        )
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predição"])
async def predict_students_batch(request: BatchPredictionRequest):
    """
    Faz previsão de risco para múltiplos alunos.

    Aceita até 100 alunos por requisição. O model_id pode ser especificado no body.
    """
    try:
        input_list = [s.model_dump(by_alias=True) for s in request.students]
        results = predict_batch(input_list, model_id=request.model_id)

        predictions = [PredictionResponse(**r) for r in results]
        risk_count = sum(1 for r in results if r["prediction"] == 1)

        # Registrar para monitoramento
        for inp, result in zip(input_list, results):
            log_prediction(inp, result)

        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            risk_count=risk_count,
            risk_rate=risk_count / len(predictions) if predictions else 0.0,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro.",
        )
    except Exception as e:
        logger.error(f"Erro na predição em lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))
