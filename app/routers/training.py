"""
Router de Treinamento e Informações do Modelo.

Endpoints:
- POST /train               — Treinar modelo
- GET  /models              — Listar modelos treinados
- GET  /models/available    — Tipos de modelo disponíveis
- GET  /features/available  — Features disponíveis
- DELETE /models/{model_id} — Deletar modelo
- GET  /training/status     — Status do treinamento
- GET  /model-info          — Informações do modelo
- GET  /feature-importance  — Top features
- GET  /learning-curve      — Gráfico de learning curves
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.models.training import (
    ModelInfoResponse,
    TrainRequest,
    TrainResponse,
    ModelListResponse,
    AvailableFeaturesResponse,
    AvailableModelsResponse,
)
from app.services.prediction.predict_service import load_model, clear_model_cache
from app.services.training.model_registry import get_available_models
from app.repositories.model_repository import ModelRepository

_repo = ModelRepository()
from app.config import AVAILABLE_FEATURES
from app.utils.helpers import setup_logger
from app.utils.error_handlers import handle_route_errors

logger = setup_logger(__name__)

router = APIRouter()

# Estado do treinamento em andamento
_training_status = {"running": False, "result": None, "error": None}


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Modelo"])
@handle_route_errors(logger)
async def model_info(model_id: str | None = None):
    """Retorna informações sobre um modelo. Se model_id=None, usa o mais recente."""
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


@router.get("/feature-importance", tags=["Modelo"])
@handle_route_errors(logger)
async def feature_importance(top_n: int = 10, model_id: str | None = None):
    """Retorna as features mais importantes do modelo."""
    _, metadata = load_model(model_id)
    importance = metadata.get("feature_importance", [])
    return {"features": importance[:top_n]}


@router.get("/learning-curve", tags=["Modelo"])
@handle_route_errors(logger)
async def learning_curve_image(model_id: str | None = None):
    """Retorna o gráfico de learning curves do modelo."""
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
        from app.services.training.train_service import run_training_pipeline

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
    models = _repo.list()
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
    success = _repo.delete(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_id}' não encontrado.")
    clear_model_cache()
    return {"message": f"Modelo '{model_id}' deletado com sucesso.", "model_id": model_id}


@router.get("/training/status", tags=["Treinamento"])
async def training_status():
    """Retorna o status do treinamento atual."""
    return _training_status
