"""
Rotas da API FastAPI.

Endpoints:
- GET  /health              — Health check
- POST /predict             — Previsão para um aluno
- POST /predict/batch       — Previsão em lote
- GET  /model-info          — Informações do modelo
- GET  /feature-importance  — Top features
- GET  /monitoring/drift    — Status de drift
- GET  /monitoring/stats    — Estatísticas de predições
- POST /train               — Treinar modelo
- GET  /models              — Listar modelos treinados
- GET  /models/available    — Tipos de modelo disponíveis
- GET  /features/available  — Features disponíveis
- DELETE /models/{model_id} — Deletar modelo
- GET  /learning-curve      — Gráfico de learning curves
"""
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.schemas import (
    StudentInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    DriftResponse,
    TrainRequest,
    TrainResponse,
    ModelListResponse,
    AvailableFeaturesResponse,
    AvailableModelsResponse,
)
from app.ml.predict import predict, predict_batch, load_model, clear_model_cache
from app.ml.model_registry import get_available_models
from app.ml.model_storage import list_trained_models, delete_model, get_latest_model_id
from app.monitoring.drift import log_prediction, check_all_drift, get_prediction_stats
from app.core.config import AVAILABLE_FEATURES
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Estado do treinamento em andamento
_training_status = {"running": False, "result": None, "error": None}


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


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Modelo"])
async def model_info(model_id: str | None = None):
    """Retorna informações sobre um modelo. Se model_id=None, usa o mais recente."""
    try:
        _, metadata = load_model(model_id)

        return ModelInfoResponse(
            model_id=metadata.get("model_id"),
            model_name=metadata.get("model_name"),
            model_type=metadata.get("model_type"),
            model_version=metadata.get("model_version"),
            metrics=metadata.get("metrics", {}),
            feature_names=metadata.get("feature_names", []),
            feature_importance=metadata.get("feature_importance", []),
            n_training_samples=metadata.get("n_training_samples", 0),
            confusion_matrix=metadata.get("confusion_matrix", {}),
            cv_results=metadata.get("cv_results"),
            trained_at=metadata.get("trained_at"),
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
async def feature_importance(top_n: int = 10, model_id: str | None = None):
    """Retorna as features mais importantes do modelo."""
    try:
        _, metadata = load_model(model_id)
        importance = metadata.get("feature_importance", [])
        return {"features": importance[:top_n]}

    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/learning-curve", tags=["Modelo"])
async def learning_curve_image(model_id: str | None = None):
    """Retorna o gráfico de learning curves do modelo."""
    try:
        _, metadata = load_model(model_id)
        lc_path = metadata.get("learning_curve_path")
        if lc_path and Path(lc_path).exists():
            return FileResponse(
                lc_path,
                media_type="image/png",
                filename="learning_curves.png",
            )
        raise HTTPException(
            status_code=404,
            detail="Learning curves não disponíveis. Treine com learning curves habilitadas.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao servir learning curves: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Novos Endpoints: Multi-Model ────────────────────────────────────────────


@router.post("/train", response_model=TrainResponse, tags=["Treinamento"])
async def train_model_endpoint(request: TrainRequest):
    """
    Treina um novo modelo com o tipo e features especificados.

    O treinamento roda de forma síncrona (pode demorar 10-60s dependendo das opções).
    """
    global _training_status

    if _training_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Já existe um treinamento em andamento. Aguarde a conclusão.",
        )

    _training_status = {"running": True, "result": None, "error": None}

    try:
        from app.ml.train import run_training_pipeline

        logger.info(
            f"Iniciando treinamento: type={request.model_type}, "
            f"features={len(request.features) if request.features else 'todas'}, "
            f"optimize={request.optimize}"
        )

        results = run_training_pipeline(
            model_type=request.model_type,
            selected_features=request.features,
            include_ian=request.include_ian,
            optimize=request.optimize,
            n_iter=request.n_iter,
            run_cv=request.run_cv,
            run_learning_curves=request.run_learning_curves,
        )

        # Limpar cache para forçar reload do novo modelo
        clear_model_cache()

        _training_status = {"running": False, "result": results, "error": None}

        return TrainResponse(
            model_id=results["model_id"],
            model_type=results["model_type"],
            metrics=results["metrics"],
            feature_names=results["feature_names"],
            n_train=results["n_train"],
            n_test=results["n_test"],
            cv_results=results.get("cv_results"),
            message=f"Modelo '{results['model_id']}' treinado com sucesso!",
        )

    except ValueError as e:
        _training_status = {"running": False, "result": None, "error": str(e)}
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _training_status = {"running": False, "result": None, "error": str(e)}
        logger.error(f"Erro no treinamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelListResponse, tags=["Treinamento"])
async def list_models():
    """Lista todos os modelos treinados (mais recente primeiro)."""
    models = list_trained_models()
    return ModelListResponse(models=models, total=len(models))


@router.get("/models/available", response_model=AvailableModelsResponse, tags=["Treinamento"])
async def available_models():
    """Lista os tipos de modelo disponíveis para treinamento."""
    models = get_available_models()
    return AvailableModelsResponse(models=models)


@router.get("/features/available", response_model=AvailableFeaturesResponse, tags=["Treinamento"])
async def available_features():
    """Lista as features disponíveis para seleção no treinamento."""
    features = [
        {
            "name": name,
            "category": meta["category"],
            "description": meta["description"],
            "default_selected": meta["default"],
        }
        for name, meta in AVAILABLE_FEATURES.items()
    ]
    return AvailableFeaturesResponse(features=features)


@router.delete("/models/{model_id}", tags=["Treinamento"])
async def delete_model_endpoint(model_id: str):
    """Deleta um modelo treinado."""
    success = delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_id}' não encontrado.")
    clear_model_cache()
    return {"message": f"Modelo '{model_id}' deletado com sucesso.", "model_id": model_id}


@router.get("/training/status", tags=["Treinamento"])
async def training_status():
    """Retorna o status do treinamento atual."""
    return _training_status
