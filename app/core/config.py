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

# Limiar para classificação binária de risco (abordagem preventiva)
DEFASAGEM_THRESHOLD = -1  # Defas <= -1 (equivale a Defas < 0) => em risco

# Flag para incluir/excluir IAN (desativado por padrão — data leakage, corr 0.838 com target)
INCLUDE_IAN = False

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

# Mapeamento ordinal das Fases (progressão escolar)
FASE_ORDINAL_MAP = {
    "Alfa": 0, "Fase 1": 1, "Fase 2": 2, "Fase 3": 3,
    "Fase 4": 4, "Fase 5": 5, "Fase 6": 6, "Fase 7": 7, "Fase 8": 8,
}

# Idade máxima esperada por fase (para feature mismatch_idade_fase)
IDADE_MAX_ESPERADA = {
    0: 8,   # Alfa (2º/3º ano)
    1: 10,  # Fase 1 (4º ano)
    2: 12,  # Fase 2 (5º/6º ano)
    3: 14,  # Fase 3 (7º/8º ano)
    4: 15,  # Fase 4 (9º ano)
    5: 16,  # Fase 5 (1º EM)
    6: 17,  # Fase 6 (2º EM)
    7: 18,  # Fase 7 (3º EM)
    8: 19,  # Fase 8
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

# Colunas de notas (Inglês removido — 67% nulos, usar flag Tem_nota_ingles)
GRADE_COLUMNS = ["Matem", "Portug"]

# Features selecionadas para o modelo (definidas após feature engineering)
SELECTED_FEATURES = [
    # Indicadores PEDE (IAN incluído, controlado pela flag INCLUDE_IAN)
    "IAA", "IEG", "IPS", "IDA", "IPV", "IAN", "INDE 22",
    # Notas (sem Inglês)
    "Matem", "Portug",
    # Demográficas
    "Idade 22", "Genero_encoded", "Escola_encoded", "Anos_na_PM",
    # Fase (ordinal)
    "Fase_encoded",
    # Evolução Pedras (com flags de presença)
    "Pedra_20_encoded", "Pedra_21_encoded", "Pedra_22_encoded",
    "Evolucao_pedra_20_22", "Evolucao_pedra_21_22",
    "tinha_pedra_20", "tinha_pedra_21",
    # Flags
    "Ponto_virada_flag", "Indicado_flag",
    "Destaque_IEG_flag", "Destaque_IDA_flag", "Destaque_IPV_flag",
    "Tem_nota_ingles",
    # Avaliadores
    "Rec_av1_encoded", "Rec_av2_encoded",
    # Rankings (com flags de presença para produção)
    "Cf", "Ct", "Nº Av",
    "tem_ranking_cf", "tem_ranking_ct",
    # Features derivadas novas
    "Variancia_indicadores", "Ratio_IDA_IEG",
    # REMOVIDAS: delta_idade_fase (corr 0.708) e mismatch_idade_fase (corr 0.553)
    # são proxies diretos de Defas (target) → data leakage
]

# Features SEM IAN (para modelo padrão — evita data leakage)
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

# Features disponíveis para seleção pelo usuário (com metadata para UI)
AVAILABLE_FEATURES = {
    # Indicadores PEDE
    "IAA": {"category": "Indicadores PEDE", "description": "Auto Avaliação (0-10)", "default": True},
    "IEG": {"category": "Indicadores PEDE", "description": "Engajamento (0-10)", "default": True},
    "IPS": {"category": "Indicadores PEDE", "description": "Psicossocial (0-10)", "default": True},
    "IDA": {"category": "Indicadores PEDE", "description": "Aprendizagem (0-10)", "default": True},
    "IPV": {"category": "Indicadores PEDE", "description": "Ponto de Virada (0-10)", "default": True},
    "IAN": {"category": "Indicadores PEDE", "description": "Adequação ao Nível (⚠️ data leakage)", "default": False},
    "INDE 22": {"category": "Indicadores PEDE", "description": "Índice de Desenvolvimento Educacional", "default": True},
    # Notas
    "Matem": {"category": "Notas", "description": "Nota de Matemática", "default": True},
    "Portug": {"category": "Notas", "description": "Nota de Português", "default": True},
    "Tem_nota_ingles": {"category": "Notas", "description": "Flag: tem nota de Inglês", "default": True},
    # Demográficas
    "Idade 22": {"category": "Demográficas", "description": "Idade do aluno em 2022", "default": True},
    "Genero_encoded": {"category": "Demográficas", "description": "Gênero (codificado)", "default": True},
    "Escola_encoded": {"category": "Demográficas", "description": "Instituição de Ensino (codificada)", "default": True},
    "Anos_na_PM": {"category": "Demográficas", "description": "Anos na Passos Mágicos", "default": True},
    "Fase_encoded": {"category": "Demográficas", "description": "Fase escolar (ordinal)", "default": True},
    # Evolução Pedras
    "Pedra_20_encoded": {"category": "Evolução Pedras", "description": "Classificação Pedra 2020", "default": True},
    "Pedra_21_encoded": {"category": "Evolução Pedras", "description": "Classificação Pedra 2021", "default": True},
    "Pedra_22_encoded": {"category": "Evolução Pedras", "description": "Classificação Pedra 2022", "default": True},
    "Evolucao_pedra_20_22": {"category": "Evolução Pedras", "description": "Evolução 2020→2022", "default": True},
    "Evolucao_pedra_21_22": {"category": "Evolução Pedras", "description": "Evolução 2021→2022", "default": True},
    "tinha_pedra_20": {"category": "Evolução Pedras", "description": "Flag: tinha pedra 2020", "default": True},
    "tinha_pedra_21": {"category": "Evolução Pedras", "description": "Flag: tinha pedra 2021", "default": True},
    # Flags
    "Ponto_virada_flag": {"category": "Flags", "description": "Atingiu Ponto de Virada", "default": True},
    "Indicado_flag": {"category": "Flags", "description": "Indicado para Bolsa", "default": True},
    "Destaque_IEG_flag": {"category": "Flags", "description": "Destaque em IEG", "default": True},
    "Destaque_IDA_flag": {"category": "Flags", "description": "Destaque em IDA", "default": True},
    "Destaque_IPV_flag": {"category": "Flags", "description": "Destaque em IPV", "default": True},
    # Avaliadores
    "Rec_av1_encoded": {"category": "Avaliadores", "description": "Recomendação Avaliador 1", "default": True},
    "Rec_av2_encoded": {"category": "Avaliadores", "description": "Recomendação Avaliador 2", "default": True},
    # Rankings
    "Cf": {"category": "Rankings", "description": "Classificação na Fase", "default": True},
    "Ct": {"category": "Rankings", "description": "Classificação na Turma", "default": True},
    "Nº Av": {"category": "Rankings", "description": "Número de Avaliações", "default": True},
    "tem_ranking_cf": {"category": "Rankings", "description": "Flag: tem ranking Cf", "default": True},
    "tem_ranking_ct": {"category": "Rankings", "description": "Flag: tem ranking Ct", "default": True},
    # Features derivadas
    "Variancia_indicadores": {"category": "Derivadas", "description": "Desvio padrão dos indicadores", "default": True},
    "Ratio_IDA_IEG": {"category": "Derivadas", "description": "Razão desempenho/esforço", "default": True},
}

