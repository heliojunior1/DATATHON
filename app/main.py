"""
FastAPI Application — Datathon Passos Mágicos

API para previsão de risco de defasagem escolar.
"""
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.core.config import MODEL_NAME, MODEL_VERSION
from app.utils.helpers import setup_logger

logger = setup_logger(__name__, log_file="api.log")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: carrega modelo no startup."""
    logger.info("=" * 60)
    logger.info("  Iniciando API — Datathon Passos Mágicos")
    logger.info("=" * 60)

    try:
        from app.ml.predict import load_model
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

# Registrar rotas da API
app.include_router(router)

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve o frontend dashboard."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

