FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Aplicar patch de compatibilidade: TabPFN v1 + PyTorch >= 2.0
COPY scripts/ ./scripts/
RUN python scripts/patch_tabpfn.py

# Copiar código da aplicação
COPY app/ ./app/

# Copiar dados e modelo (se existirem)
COPY data/ ./data/
COPY models/ ./models/

# Criar diretórios necessários
RUN mkdir -p logs models

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expor porta (Render injeta $PORT automaticamente)
EXPOSE 8000

# Comando padrão: iniciar a API
# Render define $PORT; fallback para 8000 local
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
