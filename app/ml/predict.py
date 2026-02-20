"""
Módulo de predição.

Carrega modelos treinados e faz previsões para novos dados.
Suporta múltiplos modelos via model_storage.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from app.core.config import (
    PEDRA_ORDINAL_MAP,
    GENERO_MAP,
    ESCOLA_MAP,
    FASE_ORDINAL_MAP,
    IDADE_MAX_ESPERADA,
)
from app.ml.model_storage import load_trained_model, clear_cache
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


def load_model(model_id: str | None = None) -> tuple:
    """
    Carrega o modelo e metadados.

    Args:
        model_id: ID do modelo. Se None, usa o mais recente.

    Returns:
        Tupla (model, metadata).
    """
    return load_trained_model(model_id)


def clear_model_cache() -> None:
    """Limpa o cache do modelo (útil para recarregar após re-treinamento)."""
    clear_cache()


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

    # Indicadores diretos (IAN incluído apenas se estiver no modelo treinado)
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
        "Idade 22": "Idade_22",
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

    # Fase (ordinal encoding)
    if "Fase_encoded" in feature_names:
        fase = input_data.get("Fase_encoded", input_data.get("Fase"))
        if isinstance(fase, str):
            features["Fase_encoded"] = FASE_ORDINAL_MAP.get(fase, 0)
        elif fase is not None:
            features["Fase_encoded"] = int(fase)
        else:
            features["Fase_encoded"] = 0

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

    # Flags de presença para Pedras históricas
    for year in ["20", "21"]:
        flag_col = f"tinha_pedra_{year}"
        if flag_col in feature_names:
            pedra_val = features.get(f"Pedra_{year}_encoded")
            features[flag_col] = 1 if pedra_val is not None and not (isinstance(pedra_val, float) and np.isnan(pedra_val)) else 0
            # Imputar encoded com 0 se não tem histórico
            if features[flag_col] == 0:
                features[f"Pedra_{year}_encoded"] = 0

    # Evolução das Pedras - Recalcular sempre se possível
    if "Evolucao_pedra_20_22" in feature_names:
        p20 = features.get("Pedra_20_encoded")
        p22 = features.get("Pedra_22_encoded")
        tinha_20 = features.get("tinha_pedra_20", 1)
        if tinha_20 == 0:
            features["Evolucao_pedra_20_22"] = 0  # Delta neutro
        elif p20 is not None and p22 is not None:
            features["Evolucao_pedra_20_22"] = p22 - p20
        else:
            features["Evolucao_pedra_20_22"] = input_data.get("Evolucao_pedra_20_22", 0)

    if "Evolucao_pedra_21_22" in feature_names:
        p21 = features.get("Pedra_21_encoded")
        p22 = features.get("Pedra_22_encoded")
        tinha_21 = features.get("tinha_pedra_21", 1)
        if tinha_21 == 0:
            features["Evolucao_pedra_21_22"] = 0  # Delta neutro
        elif p21 is not None and p22 is not None:
            features["Evolucao_pedra_21_22"] = p22 - p21
        else:
            features["Evolucao_pedra_21_22"] = input_data.get("Evolucao_pedra_21_22", 0)

    # Flags de presença para Rankings (fallback para alunos novos)
    for rank_col, flag_col in [("Cf", "tem_ranking_cf"), ("Ct", "tem_ranking_ct")]:
        if flag_col in feature_names:
            rank_val = features.get(rank_col)
            features[flag_col] = 1 if rank_val is not None else 0

    # Tem nota de inglês (flag binária)
    if "Tem_nota_ingles" in feature_names:
        tem = input_data.get("Tem_nota_ingles")
        if tem is None:
            # Inferir: se o input tem campo "Inglês" ou "Ingles" com valor não None
            ingles_val = input_data.get("Inglês", input_data.get("Ingles"))
            features["Tem_nota_ingles"] = 1 if ingles_val is not None else 0
        else:
            features["Tem_nota_ingles"] = int(tem)

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

    # Features derivadas — calcular se possível
    # Variância dos indicadores
    if "Variancia_indicadores" in feature_names:
        val = input_data.get("Variancia_indicadores")
        if val is not None:
            features["Variancia_indicadores"] = val
        else:
            ind_vals = [features.get(c) for c in ["IAA", "IEG", "IPS", "IDA", "IPV"] if features.get(c) is not None]
            if len(ind_vals) >= 3:
                features["Variancia_indicadores"] = float(np.std(ind_vals, ddof=1))
            else:
                features["Variancia_indicadores"] = 0.0

    # Ratio IDA/IEG
    if "Ratio_IDA_IEG" in feature_names:
        val = input_data.get("Ratio_IDA_IEG")
        if val is not None:
            features["Ratio_IDA_IEG"] = val
        else:
            ida = features.get("IDA", 0) or 0
            ieg = features.get("IEG", 0) or 0
            features["Ratio_IDA_IEG"] = ida / (ieg + 0.01)

    # Mismatch e delta idade-fase
    if "delta_idade_fase" in feature_names or "mismatch_idade_fase" in feature_names:
        delta = input_data.get("delta_idade_fase")
        if delta is None:
            idade = features.get("Idade 22") or input_data.get("Idade 22", input_data.get("Idade_22"))
            fase_enc = features.get("Fase_encoded", 0)
            if idade is not None and fase_enc is not None:
                idade_esp = IDADE_MAX_ESPERADA.get(int(fase_enc), 15)
                delta = float(idade) - idade_esp
            else:
                delta = 0.0
        if "delta_idade_fase" in feature_names:
            features["delta_idade_fase"] = delta
        if "mismatch_idade_fase" in feature_names:
            features["mismatch_idade_fase"] = 1 if delta > 0 else 0

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


def predict(input_data: dict, model_id: str | None = None) -> dict:
    """
    Faz previsão para um aluno.

    Args:
        input_data: Dicionário com dados do aluno.
        model_id: ID do modelo a usar. Se None, usa o mais recente.

    Returns:
        Dicionário com prediction, probability, risk_level, label, top_factors.
    """
    model, metadata = load_model(model_id)
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
        "model_id": metadata.get("model_id"),
    }


def predict_batch(input_list: list[dict], model_id: str | None = None) -> list[dict]:
    """Faz previsão para múltiplos alunos."""
    return [predict(data, model_id=model_id) for data in input_list]


def predict_from_store(aluno_id: str, model_id: str | None = None) -> dict:
    """
    Faz previsão buscando features diretamente do Feature Store (online).

    Ideal para cenários onde apenas o ID do aluno é conhecido e as features
    já foram materializadas no SQLite via `materialize_features.py`.

    Args:
        aluno_id: RA do aluno.
        model_id: ID do modelo. Se None, usa o mais recente.

    Returns:
        Dicionário com prediction, probability, risk_level, label, top_factors.

    Raises:
        RuntimeError: Se o Feature Store não está disponível ou populado.
    """
    try:
        from feature_store.feature_store_manager import FeatureStoreManager
    except ImportError:
        raise RuntimeError("Feast não instalado. Execute: pip install feast")

    manager = FeatureStoreManager()
    if not manager.is_populated():
        raise RuntimeError(
            "Feature Store não está populado. "
            "Execute: python scripts/materialize_features.py"
        )

    # Buscar features online
    features_df = manager.get_online_features([aluno_id])

    if features_df.empty:
        raise ValueError(f"Aluno '{aluno_id}' não encontrado no Feature Store.")

    # Converter para dicionário e fazer predição normal
    input_data = features_df.iloc[0].to_dict()
    return predict(input_data, model_id=model_id)
