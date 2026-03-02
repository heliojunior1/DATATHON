# ============================================================
# Stage 1: builder — instala dependências e corrige tabpfn
# gcc é necessário para compilar alguns pacotes (ex: catboost)
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Patch: corrige tabpfn/layer.py para compatibilidade com PyTorch >= 2.0
# O patch edita fisicamente o arquivo instalado em /install antes de
# copiá-lo para a imagem final — a imagem runtime recebe o tabpfn já corrigido
COPY scripts/ ./scripts/
RUN PYTHONPATH=/install/lib/python3.11/site-packages python scripts/patch_tabpfn.py

# ============================================================
# Stage 2: runtime — imagem final leve (sem gcc)
# gcc e artefatos de compilação ficam apenas no builder (descartado)
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# libgomp1 é necessário em runtime para o lightgbm (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependências já compiladas e com tabpfn corrigido
COPY --from=builder /install /usr/local

# Copiar código e recursos da aplicação
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY feature_store/ ./feature_store/
COPY data/ ./data/

# Criar diretório de logs
RUN mkdir -p logs

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expor porta (Render injeta $PORT automaticamente)
EXPOSE 8000

# Render define $PORT; fallback para 8000 local
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
