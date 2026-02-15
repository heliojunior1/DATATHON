"""
Configurações centrais do projeto.
"""
import os
from pathlib import Path


# Diretório raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Diretório de dados
DATA_DIR = BASE_DIR / "data"

# Diretório de modelos serializados
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Arquivo do dataset
DATASET_FILENAME = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

# Nomes do modelo
MODEL_NAME = "xgboost_defasagem"
MODEL_VERSION = "1.0.0"
MODEL_FILENAME = f"{MODEL_NAME}_v{MODEL_VERSION}.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# Arquivo de metadados de treinamento
TRAIN_METADATA_FILENAME = f"{MODEL_NAME}_v{MODEL_VERSION}_metadata.joblib"
TRAIN_METADATA_PATH = MODELS_DIR / TRAIN_METADATA_FILENAME

# Arquivo de distribuição de referência (para drift)
REFERENCE_DIST_FILENAME = f"{MODEL_NAME}_v{MODEL_VERSION}_reference.joblib"
REFERENCE_DIST_PATH = MODELS_DIR / REFERENCE_DIST_FILENAME

# Limiar para classificação binária de risco
DEFASAGEM_THRESHOLD = -2  # Defas <= -2 => em risco

# Mapeamento ordinal das Pedras (classificação Passos Mágicos)
PEDRA_ORDINAL_MAP = {
    "Quartzo": 1,
    "Ágata": 2,
    "Ametista": 3,
    "Topázio": 4,
}

# Mapeamento de Gênero
GENERO_MAP = {
    "Menina": 0,
    "Menino": 1,
}

# Mapeamento de Instituição de Ensino
ESCOLA_MAP = {
    "Escola Pública": 0,
    "Rede Decisão": 1,
    "Escola JP II": 2,
}

# Mapeamento de Recomendação Psicológica (ordenado por severidade)
REC_PSICO_MAP = {
    "Não avaliado": 0,
    "Não atendido": 1,
    "Sem limitações": 2,
    "Não indicado": 3,
    "Requer avaliação": 4,
}

# Mapeamento de Recomendações de Avaliadores
# IMPORTANTE: ordered list — strings mais específicas primeiro para evitar match parcial
REC_AVALIADOR_MAP = [
    ("Mantido na mesma fase com ressalvas", 2),
    ("Promovido de fase", 4),
    ("Mantido na mesma fase", 3),
    ("Em observação", 1),
    ("Sem informação", 0),
]

# Colunas numéricas de indicadores PEDE (features principais)
INDICATOR_COLUMNS = [
    "IAA", "IEG", "IPS", "IDA", "IPV", "IAN", "INDE 22",
]

# Colunas de notas
GRADE_COLUMNS = ["Matem", "Portug", "Inglês"]

# Features selecionadas para o modelo (definidas após feature engineering)
SELECTED_FEATURES = [
    # Indicadores PEDE
    "IAA", "IEG", "IPS", "IDA", "IPV", "IAN", "INDE 22",
    # Notas
    "Matem", "Portug", "Inglês",
    # Demográficas
    "Idade 22", "Genero_encoded", "Escola_encoded", "Anos_na_PM",
    # Evolução
    "Pedra_20_encoded", "Pedra_21_encoded", "Pedra_22_encoded",
    "Evolucao_pedra_20_22", "Evolucao_pedra_21_22",
    # Categóricas transformadas
    "Rec_psico_encoded",
    # Flags
    "Ponto_virada_flag", "Indicado_flag",
    "Destaque_IEG_flag", "Destaque_IDA_flag", "Destaque_IPV_flag",
    # Avaliadores
    "Rec_av1_encoded", "Rec_av2_encoded",
    # Rankings
    "Cg", "Cf", "Ct", "Nº Av",
]

# Features SEM IAN (para modelo alternativo — evita data leakage)
SELECTED_FEATURES_NO_IAN = [f for f in SELECTED_FEATURES if f != "IAN"]

# Configurações de treinamento
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
