"""
Módulo de predição.

Carrega modelos treinados e faz previsões para novos dados.
Suporta múltiplos modelos via ModelRepository (injetável para testes).
"""
import time
import pandas as pd
import numpy as np

from app.config import (
    PEDRA_ORDINAL_MAP,
    GENERO_MAP,
    ESCOLA_MAP,
    FASE_ORDINAL_MAP,
    IDADE_MAX_ESPERADA,
)
from app.domain.risk_level import classify_risk
from app.repositories.model_repository import ModelRepository
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# Instância padrão — usada quando nenhum repo é injetado explicitamente
_default_repo: ModelRepository = ModelRepository()


# ── Interface de carregamento ─────────────────────────────────────────────────


def load_model(
    model_id: str | None = None,
    repo: ModelRepository | None = None,
) -> tuple:
    """
    Carrega o modelo e metadados via repository.

    Args:
        model_id: ID do modelo. Se None, usa o mais recente.
        repo: Repository a usar. Se None, usa a instância padrão do módulo.

    Returns:
        Tupla (model, metadata).
    """
    return (repo or _default_repo).load(model_id)


def clear_model_cache(repo: ModelRepository | None = None) -> None:
    """
    Limpa o cache do modelo.

    Args:
        repo: Repository a usar. Se None, usa a instância padrão do módulo.
    """
    (repo or _default_repo).clear_cache()


# ── Helpers de mapeamento ─────────────────────────────────────────────────────


def _map_indicadores(input_data: dict, features: dict, feature_names: list[str]) -> None:
    """Mapeia indicadores PEDE e notas diretamente do input."""
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
            value = input_data.get(model_feat, input_data.get(input_key))
            features[model_feat] = value


def _map_categoricas(input_data: dict, features: dict, feature_names: list[str]) -> None:
    """Codifica variáveis categóricas: Gênero, Escola, Fase e Anos_na_PM."""
    if "Genero_encoded" in feature_names:
        genero = input_data.get("Genero_encoded", input_data.get("Genero"))
        if isinstance(genero, str):
            features["Genero_encoded"] = GENERO_MAP.get(genero, 0)
        else:
            features["Genero_encoded"] = genero if genero is not None else 0

    if "Escola_encoded" in feature_names:
        escola = input_data.get(
            "Escola_encoded",
            input_data.get("Instituição de ensino", input_data.get("Instituicao_ensino")),
        )
        if isinstance(escola, str):
            features["Escola_encoded"] = ESCOLA_MAP.get(escola, 0)
        else:
            features["Escola_encoded"] = escola if escola is not None else 0

    if "Fase_encoded" in feature_names:
        fase = input_data.get("Fase_encoded", input_data.get("Fase"))
        if isinstance(fase, str):
            features["Fase_encoded"] = FASE_ORDINAL_MAP.get(fase, 0)
        elif fase is not None:
            features["Fase_encoded"] = int(fase)
        else:
            features["Fase_encoded"] = 0

    if "Anos_na_PM" in feature_names:
        anos = input_data.get(
            "Anos_na_PM", input_data.get("Ano ingresso", input_data.get("Ano_ingresso"))
        )
        if anos is not None and isinstance(anos, (int, float)) and anos > 100:
            features["Anos_na_PM"] = 2022 - int(anos)
        else:
            features["Anos_na_PM"] = anos


def _map_pedras(input_data: dict, features: dict, feature_names: list[str]) -> None:
    """Codifica Pedras por ano, flags de presença e evolução ordinal."""
    for year in ["20", "21", "22"]:
        col = f"Pedra_{year}_encoded"
        if col in feature_names:
            val = input_data.get(
                col, input_data.get(f"Pedra {year}", input_data.get(f"Pedra_{year}"))
            )
            features[col] = PEDRA_ORDINAL_MAP.get(val) if isinstance(val, str) else val

    for year in ["20", "21"]:
        flag_col = f"tinha_pedra_{year}"
        if flag_col in feature_names:
            pedra_val = features.get(f"Pedra_{year}_encoded")
            features[flag_col] = (
                1
                if pedra_val is not None
                and not (isinstance(pedra_val, float) and np.isnan(pedra_val))
                else 0
            )
            if features[flag_col] == 0:
                features[f"Pedra_{year}_encoded"] = 0

    if "Evolucao_pedra_20_22" in feature_names:
        p20, p22 = features.get("Pedra_20_encoded"), features.get("Pedra_22_encoded")
        tinha_20 = features.get("tinha_pedra_20", 1)
        if tinha_20 == 0:
            features["Evolucao_pedra_20_22"] = 0
        elif p20 is not None and p22 is not None:
            features["Evolucao_pedra_20_22"] = p22 - p20
        else:
            features["Evolucao_pedra_20_22"] = input_data.get("Evolucao_pedra_20_22", 0)

    if "Evolucao_pedra_21_22" in feature_names:
        p21, p22 = features.get("Pedra_21_encoded"), features.get("Pedra_22_encoded")
        tinha_21 = features.get("tinha_pedra_21", 1)
        if tinha_21 == 0:
            features["Evolucao_pedra_21_22"] = 0
        elif p21 is not None and p22 is not None:
            features["Evolucao_pedra_21_22"] = p22 - p21
        else:
            features["Evolucao_pedra_21_22"] = input_data.get("Evolucao_pedra_21_22", 0)


def _map_flags(input_data: dict, features: dict, feature_names: list[str]) -> None:
    """Codifica flags booleanas, nota de inglês, rankings e recomendações."""
    for rank_col, flag_col in [("Cf", "tem_ranking_cf"), ("Ct", "tem_ranking_ct")]:
        if flag_col in feature_names:
            features[flag_col] = 1 if features.get(rank_col) is not None else 0

    if "Tem_nota_ingles" in feature_names:
        tem = input_data.get("Tem_nota_ingles")
        if tem is None:
            ingles_val = input_data.get("Inglês", input_data.get("Ingles"))
            features["Tem_nota_ingles"] = 1 if ingles_val is not None else 0
        else:
            features["Tem_nota_ingles"] = int(tem)

    bool_mapping = {
        "Ponto_virada_flag": ["Ponto_virada_flag", "Atingiu_PV"],
        "Indicado_flag": ["Indicado_flag", "Indicado"],
        "Destaque_IEG_flag": ["Destaque_IEG_flag", "Destaque_IEG"],
        "Destaque_IDA_flag": ["Destaque_IDA_flag", "Destaque_IDA"],
        "Destaque_IPV_flag": ["Destaque_IPV_flag", "Destaque_IPV"],
    }
    for model_feat, keys in bool_mapping.items():
        if model_feat in feature_names:
            value = next((input_data[k] for k in keys if k in input_data), None)
            if isinstance(value, str):
                features[model_feat] = 1 if value.lower() in ("sim", "yes", "true", "1") else 0
            elif value is not None:
                features[model_feat] = int(value)
            else:
                features[model_feat] = 0

    for av_num in [1, 2]:
        col = f"Rec_av{av_num}_encoded"
        if col in feature_names:
            features[col] = input_data.get(col, 0)


def _calc_derivadas(input_data: dict, features: dict, feature_names: list[str]) -> None:
    """Calcula features derivadas: variância dos indicadores, ratio IDA/IEG e mismatch de fase."""
    if "Variancia_indicadores" in feature_names:
        val = input_data.get("Variancia_indicadores")
        if val is not None:
            features["Variancia_indicadores"] = val
        else:
            ind_vals = [
                features.get(c)
                for c in ["IAA", "IEG", "IPS", "IDA", "IPV"]
                if features.get(c) is not None
            ]
            features["Variancia_indicadores"] = (
                float(np.std(ind_vals, ddof=1)) if len(ind_vals) >= 3 else 0.0
            )

    if "Ratio_IDA_IEG" in feature_names:
        val = input_data.get("Ratio_IDA_IEG")
        if val is not None:
            features["Ratio_IDA_IEG"] = val
        else:
            ida = features.get("IDA", 0) or 0
            ieg = features.get("IEG", 0) or 0
            features["Ratio_IDA_IEG"] = ida / (ieg + 0.01)

    if "delta_idade_fase" in feature_names or "mismatch_idade_fase" in feature_names:
        delta = input_data.get("delta_idade_fase")
        if delta is None:
            idade = features.get("Idade 22") or input_data.get(
                "Idade 22", input_data.get("Idade_22")
            )
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


def ensure_feature_order(features: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Alinha e converte o dicionário de features para DataFrame na ordem esperada pelo modelo.

    Colunas ausentes são preenchidas com NaN; todas são convertidas para float64.

    Args:
        features: Dicionário com valores das features calculadas.
        feature_names: Lista de features na ordem exata exigida pelo modelo.

    Returns:
        DataFrame com uma linha pronta para predição.
    """
    df = pd.DataFrame([features])
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = np.nan
    df = df[feature_names]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_input_features(input_data: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Prepara os dados de entrada para predição.

    Orquestra os helpers de mapeamento para transformar o dicionário do aluno
    nas features esperadas pelo modelo, na ordem correta.

    Args:
        input_data: Dicionário com dados do aluno.
        feature_names: Lista de features esperadas pelo modelo.

    Returns:
        DataFrame com uma linha pronta para predição.
    """
    features: dict = {}
    _map_indicadores(input_data, features, feature_names)
    _map_categoricas(input_data, features, feature_names)
    _map_pedras(input_data, features, feature_names)
    _map_flags(input_data, features, feature_names)
    _calc_derivadas(input_data, features, feature_names)
    return ensure_feature_order(features, feature_names)


def predict(
    input_data: dict,
    model_id: str | None = None,
    repo: ModelRepository | None = None,
) -> dict:
    """
    Faz previsão para um aluno.

    Args:
        input_data: Dicionário com dados do aluno.
        model_id: ID do modelo a usar. Se None, usa o mais recente.
        repo: Repository de modelos. Se None, usa a instância padrão do módulo.
              Passe um mock em testes para evitar acesso a disco.

    Returns:
        Dicionário com prediction, probability, risk_level, label, top_factors.
    """
    model, metadata = load_model(model_id, repo=repo)
    feature_names = metadata.get("feature_names", [])

    X = prepare_input_features(input_data, feature_names)

    t0 = time.perf_counter()
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    importance = metadata.get("feature_importance", [])
    top_factors = importance[:5] if importance else []

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": classify_risk(probability),
        "label": "Em Risco de Defasagem" if prediction == 1 else "Sem Risco",
        "top_factors": top_factors,
        "model_id": metadata.get("model_id"),
        "latency_ms": latency_ms,
    }


def predict_batch(
    input_list: list[dict],
    model_id: str | None = None,
    repo: ModelRepository | None = None,
) -> list[dict]:
    """
    Faz previsão para múltiplos alunos.

    Args:
        input_list: Lista de dicionários com dados dos alunos.
        model_id: ID do modelo. Se None, usa o mais recente.
        repo: Repository de modelos. Se None, usa a instância padrão do módulo.
    """
    return [predict(data, model_id=model_id, repo=repo) for data in input_list]


def predict_from_store(aluno_id: str, model_id: str | None = None) -> dict:
    """
    Faz previsão buscando features diretamente do Feature Store (online).

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

    features_df = manager.get_online_features([aluno_id])

    if features_df.empty:
        raise ValueError(f"Aluno '{aluno_id}' não encontrado no Feature Store.")

    input_data = features_df.iloc[0].to_dict()
    return predict(input_data, model_id=model_id)
