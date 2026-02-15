"""
Rotas da API FastAPI.

Endpoints:
- GET  /health           — Health check
- POST /predict          — Previsão para um aluno
- POST /predict/batch    — Previsão em lote
- GET  /model-info       — Informações do modelo
- GET  /feature-importance — Top features
- GET  /monitoring/drift  — Status de drift
- GET  /monitoring/stats  — Estatísticas de predições
"""
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    StudentInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    DriftResponse,
)
from app.ml.predict import predict, predict_batch, load_model
from app.monitoring.drift import log_prediction, check_all_drift, get_prediction_stats
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
            model_name=metadata.get("model_name"),
            model_version=metadata.get("model_version"),
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
async def predict_student(student: StudentInput):
    """
    Faz previsão de risco de defasagem para um aluno.

    Recebe os dados do aluno e retorna:
    - prediction: 0 (sem risco) ou 1 (em risco)
    - probability: probabilidade de risco
    - risk_level: nível de risco descritivo
    - top_factors: features mais influentes
    """
    try:
        input_data = student.model_dump(by_alias=True)
        result = predict(input_data)

        # Registrar para monitoramento
        log_prediction(input_data, result)

        logger.info(
            f"Predição realizada: risco={result['prediction']}, "
            f"prob={result['probability']:.4f}, "
            f"nível={result['risk_level']}"
        )

        return PredictionResponse(**result)

    except FileNotFoundError as e:
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

    Aceita até 100 alunos por requisição.
    """
    try:
        input_list = [s.model_dump(by_alias=True) for s in request.students]
        results = predict_batch(input_list)

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


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Modelo"])
async def model_info():
    """Retorna informações sobre o modelo em produção."""
    try:
        _, metadata = load_model()

        return ModelInfoResponse(
            model_name=metadata.get("model_name", "unknown"),
            model_version=metadata.get("model_version", "unknown"),
            metrics=metadata.get("metrics", {}),
            feature_names=metadata.get("feature_names", []),
            feature_importance=metadata.get("feature_importance", []),
            n_training_samples=metadata.get("n_training_samples", 0),
            confusion_matrix=metadata.get("confusion_matrix", {}),
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treinamento primeiro.",
        )
    except Exception as e:
        logger.error(f"Erro ao obter info do modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance", tags=["Modelo"])
async def feature_importance(top_n: int = 10):
    """Retorna as features mais importantes do modelo."""
    try:
        _, metadata = load_model()
        importance = metadata.get("feature_importance", [])
        return {"features": importance[:top_n]}

    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/drift", response_model=DriftResponse, tags=["Monitoramento"])
async def drift_status():
    """
    Verifica status de data drift comparando dados de produção com dados de treinamento.

    Retorna análise de drift por feature usando teste KS e shift de média.
    """
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
