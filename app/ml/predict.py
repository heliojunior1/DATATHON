"""
Módulo de predição.

Carrega o modelo treinado e faz previsões para novos dados.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from app.core.config import (
    MODEL_PATH,
    TRAIN_METADATA_PATH,
    PEDRA_ORDINAL_MAP,
    GENERO_MAP,
    ESCOLA_MAP,
    REC_PSICO_MAP,
)
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# Cache do modelo carregado
_model_cache = {}
_metadata_cache = {}


def load_model(model_path: str | Path | None = None) -> tuple:
    """
    Carrega o modelo e metadados do disco.

    Args:
        model_path: Caminho do modelo. Se None, usa o padrão.

    Returns:
        Tupla (model, metadata).
    """
    path = Path(model_path) if model_path else MODEL_PATH
    meta_path = TRAIN_METADATA_PATH

    cache_key = str(path)

    if cache_key not in _model_cache:
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {path}. "
                "Execute 'python train_pipeline.py' primeiro."
            )
        _model_cache[cache_key] = joblib.load(path)
        logger.info(f"Modelo carregado de: {path}")

    if cache_key not in _metadata_cache:
        if meta_path.exists():
            _metadata_cache[cache_key] = joblib.load(meta_path)
            logger.info(f"Metadados carregados de: {meta_path}")
        else:
            _metadata_cache[cache_key] = {}

    return _model_cache[cache_key], _metadata_cache[cache_key]


def clear_model_cache() -> None:
    """Limpa o cache do modelo (útil para recarregar após re-treinamento)."""
    _model_cache.clear()
    _metadata_cache.clear()
    logger.info("Cache do modelo limpo")


def prepare_input_features(input_data: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Prepara os dados de entrada para predição.

    Recebe um dicionário com os dados do aluno e transforma nas features
    esperadas pelo modelo.

    Args:
        input_data: Dicionário com dados do aluno.
        feature_names: Lista de features esperadas pelo modelo.

    Returns:
        DataFrame com uma linha pronta para predição.
    """
    features = {}

    # Indicadores diretos
    direct_mapping = {
        "IAA": "IAA",
        "IEG": "IEG",
        "IPS": "IPS",
        "IDA": "IDA",
        "IPV": "IPV",
        "IAN": "IAN",
        "INDE 22": "INDE_22",
        "Matem": "Matem",
        "Portug": "Portug",
        "Inglês": "Ingles",
        "Idade 22": "Idade_22",
        "Cg": "Cg",
        "Cf": "Cf",
        "Ct": "Ct",
        "Nº Av": "N_Av",
    }

    for model_feat, input_key in direct_mapping.items():
        if model_feat in feature_names:
            # Tentar nome do modelo primeiro, depois input_key
            value = input_data.get(model_feat, input_data.get(input_key))
            features[model_feat] = value

    # Gênero
    if "Genero_encoded" in feature_names:
        genero = input_data.get("Genero_encoded", input_data.get("Genero"))
        if isinstance(genero, str):
            features["Genero_encoded"] = GENERO_MAP.get(genero, 0)
        else:
            features["Genero_encoded"] = genero if genero is not None else 0

    # Escola
    if "Escola_encoded" in feature_names:
        escola = input_data.get("Escola_encoded", input_data.get("Instituição de ensino", input_data.get("Instituicao_ensino")))
        if isinstance(escola, str):
            features["Escola_encoded"] = ESCOLA_MAP.get(escola, 0)
        else:
            features["Escola_encoded"] = escola if escola is not None else 0

    # Anos na PM
    if "Anos_na_PM" in feature_names:
        anos = input_data.get("Anos_na_PM", input_data.get("Ano ingresso", input_data.get("Ano_ingresso")))
        if anos is not None and isinstance(anos, (int, float)) and anos > 100:
            # Se recebeu Ano_ingresso, calcular
            features["Anos_na_PM"] = 2022 - int(anos)
        else:
            features["Anos_na_PM"] = anos

    # Pedras codificadas
    for year in ["20", "21", "22"]:
        col = f"Pedra_{year}_encoded"
        if col in feature_names:
            # Tentar "Pedra 20" (com espaço) e "Pedra_20" (com underscore)
            val = input_data.get(col, input_data.get(f"Pedra {year}", input_data.get(f"Pedra_{year}")))
            if isinstance(val, str):
                features[col] = PEDRA_ORDINAL_MAP.get(val)
            else:
                features[col] = val

    # Evolução das Pedras - Recalcular sempre se possível, pois input pode não ter
    if "Evolucao_pedra_20_22" in feature_names:
        p20 = features.get("Pedra_20_encoded")
        p22 = features.get("Pedra_22_encoded")
        if p20 is not None and p22 is not None:
            features["Evolucao_pedra_20_22"] = p22 - p20
        else:
            features["Evolucao_pedra_20_22"] = input_data.get("Evolucao_pedra_20_22")

    if "Evolucao_pedra_21_22" in feature_names:
        p21 = features.get("Pedra_21_encoded")
        p22 = features.get("Pedra_22_encoded")
        if p21 is not None and p22 is not None:
            features["Evolucao_pedra_21_22"] = p22 - p21
        else:
            features["Evolucao_pedra_21_22"] = input_data.get("Evolucao_pedra_21_22")

    # Rec Psicologia
    if "Rec_psico_encoded" in feature_names:
        rec = input_data.get("Rec_psico_encoded", input_data.get("Rec Psicologia", input_data.get("Rec_Psicologia")))
        if isinstance(rec, str):
            features["Rec_psico_encoded"] = REC_PSICO_MAP.get(rec, 0)
        else:
            features["Rec_psico_encoded"] = rec if rec is not None else 0

    # Flags booleanas
    bool_mapping = {
        "Ponto_virada_flag": ["Ponto_virada_flag", "Atingiu_PV"],
        "Indicado_flag": ["Indicado_flag", "Indicado"],
        "Destaque_IEG_flag": ["Destaque_IEG_flag", "Destaque_IEG"],
        "Destaque_IDA_flag": ["Destaque_IDA_flag", "Destaque_IDA"],
        "Destaque_IPV_flag": ["Destaque_IPV_flag", "Destaque_IPV"],
    }
    for model_feat, keys in bool_mapping.items():
        if model_feat in feature_names:
            value = None
            for key in keys:
                if key in input_data:
                    value = input_data[key]
                    break
            if isinstance(value, str):
                features[model_feat] = 1 if value.lower() in ("sim", "yes", "true", "1") else 0
            elif value is not None:
                features[model_feat] = int(value)
            else:
                features[model_feat] = 0

    # Recomendações dos avaliadores
    for av_num in [1, 2]:
        col = f"Rec_av{av_num}_encoded"
        if col in feature_names:
            features[col] = input_data.get(col, 0)

    # Criar DataFrame com as features na ordem correta
    df = pd.DataFrame([features])

    # Garantir que todas as features do modelo existam
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = np.nan

    df = df[feature_names]

    # Converter todas as colunas para float64 (XGBoost exige tipos numéricos)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def predict(input_data: dict) -> dict:
    """
    Faz previsão para um aluno.

    Args:
        input_data: Dicionário com dados do aluno.

    Returns:
        Dicionário com:
        - prediction: 0 (sem risco) ou 1 (em risco)
        - probability: probabilidade de risco (0 a 1)
        - risk_level: descrição textual do risco
        - top_factors: top 5 features mais influentes para este aluno
    """
    model, metadata = load_model()
    feature_names = metadata.get("feature_names", [])

    X = prepare_input_features(input_data, feature_names)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    # Determinar nível de risco
    if probability >= 0.8:
        risk_level = "Muito Alto"
    elif probability >= 0.6:
        risk_level = "Alto"
    elif probability >= 0.4:
        risk_level = "Moderado"
    elif probability >= 0.2:
        risk_level = "Baixo"
    else:
        risk_level = "Muito Baixo"

    # Top factors (feature importance geral do modelo)
    importance = metadata.get("feature_importance", [])
    top_factors = importance[:5] if importance else []

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "label": "Em Risco de Defasagem" if prediction == 1 else "Sem Risco",
        "top_factors": top_factors,
    }


def predict_batch(input_list: list[dict]) -> list[dict]:
    """Faz previsão para múltiplos alunos."""
    return [predict(data) for data in input_list]
