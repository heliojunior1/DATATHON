"""
FastAPI Application — Datathon Passos Mágicos

API para previsão de risco de defasagem escolar.
"""
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from app.routers.prediction import router as prediction_router
from app.routers.training import router as training_router
from app.routers.monitoring import router as monitoring_router
from app.routers.feature_store import router as feature_store_router
from app.config import MODEL_NAME, MODEL_VERSION
from app.utils.helpers import setup_logger

logger = setup_logger(__name__, log_file="api.log")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Contadores de requisições para monitoramento de error rate
_request_stats: dict = {"total": 0, "errors_4xx": 0, "errors_5xx": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: carrega modelo no startup."""
    logger.info("=" * 60)
    logger.info("  Iniciando API — Datathon Passos Mágicos")
    logger.info("=" * 60)

    try:
        from app.services.predict_service import load_model
        model, metadata = load_model()
        logger.info(f"Modelo carregado: {metadata.get('model_name')} v{metadata.get('model_version')}")
        logger.info(f"Métricas: {metadata.get('metrics', {})}")
    except FileNotFoundError:
        logger.warning(
            "Modelo não encontrado! Execute 'python train_pipeline.py' para treinar. "
            "A API funcionará, mas endpoints de predição retornarão erro 503."
        )
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")

    yield

    logger.info("API encerrada.")


app = FastAPI(
    title="Datathon Passos Mágicos — API de Previsão de Defasagem",
    description="""
## Previsão de Risco de Defasagem Escolar

API para o modelo preditivo de risco de defasagem escolar dos alunos
da Associação Passos Mágicos.

### Funcionalidades:
- **Predição individual e em lote** de risco de defasagem
- **Informações do modelo** e ranking de features
- **Monitoramento de drift** dos dados em produção

### Métricas utilizadas:
O modelo utiliza **XGBoost** otimizado com as seguintes métricas:
- F1-Score (métrica primária)
- AUC-ROC
- Recall (priorizado para não perder alunos em risco)
""",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def count_request_errors(request: Request, call_next):
    """Middleware para contagem de requisições e erros 4xx/5xx."""
    response = await call_next(request)
    _request_stats["total"] += 1
    if 400 <= response.status_code < 500:
        _request_stats["errors_4xx"] += 1
    elif response.status_code >= 500:
        _request_stats["errors_5xx"] += 1
    return response


@app.get("/monitoring/errors", tags=["Monitoramento"])
async def error_stats():
    """Retorna contagem de requisições e erros 4xx/5xx desde o último restart."""
    total = _request_stats["total"]
    errors = _request_stats["errors_4xx"] + _request_stats["errors_5xx"]
    return {
        "total_requests": total,
        "errors_4xx": _request_stats["errors_4xx"],
        "errors_5xx": _request_stats["errors_5xx"],
        "error_rate": round(errors / total, 4) if total > 0 else 0.0,
    }

# Registrar routers da API
app.include_router(prediction_router)
app.include_router(training_router)
app.include_router(monitoring_router)
app.include_router(feature_store_router)

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve o frontend dashboard."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
