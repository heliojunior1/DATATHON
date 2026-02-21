"""
Facade de compatibilidade do serviço de monitoramento.

Este módulo re-exporta as funções dos submódulos especializados para
manter compatibilidade com imports existentes. A lógica está dividida em:

- monitoring/prediction_log_store.py  — log em memória
- monitoring/monitoring_stats_service.py — métricas operacionais
- monitoring/data_drift_service.py    — data drift (KS-test, PSI, KL)
- monitoring/concept_drift_service.py — concept drift (feedback loop)
"""
from app.services.monitoring.prediction_log_store import (
    log_prediction,
    get_prediction_log,
    _prediction_log,
)
from app.services.monitoring.monitoring_stats_service import (
    get_prediction_stats,
    get_latency_stats,
    get_throughput,
    get_missing_values_stats,
)
from app.services.monitoring.data_drift_service import (
    calculate_psi,
    calculate_kl_divergence,
    check_drift,
    check_all_drift,
)
from app.services.monitoring.concept_drift_service import (
    submit_feedback,
    get_predictions_for_feedback,
    get_concept_drift_stats,
)

__all__ = [
    "log_prediction",
    "get_prediction_log",
    "_prediction_log",
    "get_prediction_stats",
    "get_latency_stats",
    "get_throughput",
    "get_missing_values_stats",
    "calculate_psi",
    "calculate_kl_divergence",
    "check_drift",
    "check_all_drift",
    "submit_feedback",
    "get_predictions_for_feedback",
    "get_concept_drift_stats",
]
