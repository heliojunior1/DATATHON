"""
Domínio: Classificação de nível de risco.

Centraliza a lógica de mapeamento probabilidade → nível de risco,
eliminando duplicação entre predict_service e drift_service.
"""
from app.config import RISK_THRESHOLDS


def classify_risk(probability: float) -> str:
    """
    Classifica o nível de risco com base na probabilidade prevista.

    Args:
        probability: Probabilidade de risco (0.0 a 1.0).

    Returns:
        Nível de risco: "Muito Alto", "Alto", "Moderado", "Baixo" ou "Muito Baixo".
    """
    if probability >= RISK_THRESHOLDS["muito_alto"]:
        return "Muito Alto"
    elif probability >= RISK_THRESHOLDS["alto"]:
        return "Alto"
    elif probability >= RISK_THRESHOLDS["moderado"]:
        return "Moderado"
    elif probability >= RISK_THRESHOLDS["baixo"]:
        return "Baixo"
    return "Muito Baixo"
