"""
Fixtures compartilhadas para testes.
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Garantir que o diretório raiz está no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_raw_data():
    """Retorna um DataFrame simulando dados brutos do PEDE."""
    np.random.seed(42)
    n = 50

    data = {
        "RA": [f"RA-{i}" for i in range(n)],
        "Fase": np.random.choice(
            ["Alfa", "Fase 1", "Fase 2", "Fase 3", "Fase 4", "Fase 5", "Fase 6", "Fase 7", "Fase 8"],
            n,
        ),
        "Turma": [f"{np.random.randint(0,8)}{chr(65+np.random.randint(0,3))}" for _ in range(n)],
        "Nome": [f"Aluno-{i}" for i in range(n)],
        "Ano nasc": np.random.randint(2003, 2015, n),
        "Idade 22": np.random.randint(7, 20, n),
        "Gênero": np.random.choice(["Menina", "Menino"], n),
        "Ano ingresso": np.random.randint(2016, 2023, n),
        "Instituição de ensino": np.random.choice(
            ["Escola Pública", "Rede Decisão", "Escola JP II"],
            n,
            p=[0.85, 0.12, 0.03],
        ),
        "Pedra 20": np.where(
            np.random.random(n) > 0.6,
            np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
            None,
        ),
        "Pedra 21": np.where(
            np.random.random(n) > 0.45,
            np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
            None,
        ),
        "Pedra 22": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
        "INDE 22": np.round(np.random.uniform(2.5, 9.5, n), 3),
        "Cg": np.random.randint(1, 861, n),
        "Cf": np.random.randint(1, 193, n),
        "Ct": np.random.randint(1, 19, n),
        "Nº Av": np.random.choice([2, 3, 4], n),
        "Avaliador1": [f"Avaliador-{np.random.randint(1,7)}" for _ in range(n)],
        "Rec Av1": np.random.choice(
            ["Promovido de fase", "Mantido na mesma fase", "Mantido na mesma fase com ressalvas",
             "Em observação", "Sem informação"],
            n,
        ),
        "Avaliador2": [f"Avaliador-{np.random.randint(1,5)}" for _ in range(n)],
        "Rec Av2": np.random.choice(
            ["Promovido de fase", "Mantido na mesma fase", "Mantido na mesma fase com ressalvas",
             "Em observação", "Sem informação"],
            n,
        ),
        "Avaliador3": np.where(
            np.random.random(n) > 0.4,
            [f"Avaliador-{np.random.randint(1,7)}" for _ in range(n)],
            None,
        ),
        "Rec Av3": np.random.choice(
            ["Promovido de fase", "Mantido na mesma fase", "Sem informação"],
            n,
        ),
        "Avaliador4": np.where(np.random.random(n) > 0.65, "Avaliador-1", None),
        "Rec Av4": np.where(
            np.random.random(n) > 0.65,
            np.random.choice(["Promovido de fase", "Mantido na mesma fase"], n),
            None,
        ),
        "IAA": np.round(np.random.uniform(3, 10, n), 2),
        "IEG": np.round(np.random.uniform(2, 10, n), 2),
        "IPS": np.round(np.random.uniform(4, 10, n), 2),
        "Rec Psicologia": np.random.choice(
            ["Não atendido", "Sem limitações", "Requer avaliação", "Não indicado", "Não avaliado"],
            n,
        ),
        "IDA": np.round(np.random.uniform(2, 10, n), 2),
        "Matem": np.round(np.random.uniform(2, 10, n), 2),
        "Portug": np.round(np.random.uniform(2, 10, n), 2),
        "Inglês": np.where(np.random.random(n) > 0.67, np.round(np.random.uniform(2, 10, n), 2), np.nan),
        "Indicado": np.random.choice(["Sim", "Não"], n, p=[0.15, 0.85]),
        "Atingiu PV": np.random.choice(["Sim", "Não"], n, p=[0.13, 0.87]),
        "IPV": np.round(np.random.uniform(1, 10, n), 2),
        "IAN": np.random.choice([0.0, 5.0, 10.0], n),
        "Fase ideal": np.random.choice(
            ["ALFA  (2º e 3º ano)", "Fase 1 (4º ano)", "Fase 2 (5º e 6º ano)",
             "Fase 3 (7º e 8º ano)", "Fase 4 (9º ano)", "Fase 5 (1º EM)",
             "Fase 6 (2º EM)", "Fase 7 (3º EM)"],
            n,
        ),
        "Defas": np.random.choice([-5, -4, -3, -2, -1, 0, 1, 2], n, p=[0.01, 0.01, 0.03, 0.19, 0.47, 0.25, 0.02, 0.02]),
        "Destaque IEG": np.random.choice(["Destaque IEG", "Não"], n, p=[0.2, 0.8]),
        "Destaque IDA": np.random.choice(["Destaque IDA", "Não"], n, p=[0.2, 0.8]),
        "Destaque IPV": np.random.choice(["Destaque IPV", "Não"], n, p=[0.2, 0.8]),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_student_input():
    """Retorna um dicionário simulando dados de um aluno para predição."""
    return {
        "IAA": 7.5,
        "IEG": 8.0,
        "IPS": 6.5,
        "IDA": 7.0,
        "IPV": 5.5,
        "INDE 22": 7.2,
        "Matem": 7.5,
        "Portug": 6.8,
        "Tem_nota_ingles": 1,
        "Idade 22": 14,
        "Gênero": "Menina",
        "Instituição de ensino": "Escola Pública",
        "Ano ingresso": 2018,
        "Fase": "Fase 3",
        "Pedra 20": "Ágata",
        "Pedra 21": "Ametista",
        "Pedra 22": "Ametista",
        "Atingiu PV": "Não",
        "Indicado": "Não",
        "Cf": 50,
        "Ct": 5,
        "Nº Av": 3,
        "Destaque IEG": "Não",
        "Destaque IDA": "Não",
        "Destaque IPV": "Não",
        "Rec_av1_encoded": 3,
        "Rec_av2_encoded": 3,
    }


@pytest.fixture
def preprocessed_data(sample_raw_data):
    """Retorna dados pré-processados."""
    from app.services.preprocessing import create_target_variable, encode_categorical_columns, handle_missing_values
    df = create_target_variable(sample_raw_data)
    df = encode_categorical_columns(df)
    df = handle_missing_values(df)
    return df


@pytest.fixture
def engineered_data(preprocessed_data):
    """Retorna dados com feature engineering aplicada."""
    from app.services.feature_engineering import run_feature_engineering
    return run_feature_engineering(preprocessed_data)
