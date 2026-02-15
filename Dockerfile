FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY app/ ./app/
COPY static/ ./static/
COPY train_pipeline.py .

# Copiar dados e modelo (se existirem)
COPY data/ ./data/
COPY models/ ./models/

# Criar diretórios necessários
RUN mkdir -p logs models

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expor porta
EXPOSE 8000

# Comando padrão: iniciar a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
