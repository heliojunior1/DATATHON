"""
Implementação concreta do Repository de Modelos (backend: disco + joblib).

Implementa ModelRepositoryProtocol delegando ao model_storage.
Os services devem depender do protocolo, não desta classe diretamente.
"""
import pandas as pd

from app.repositories.protocols import ModelRepositoryProtocol
from app.services.storage.model_storage import (
    save_trained_model,
    load_trained_model,
    list_trained_models,
    delete_model,
    load_reference_data,
    get_latest_model_id,
    clear_cache,
)
from app.utils.helpers import setup_logger

logger = setup_logger(__name__)


class ModelRepository(ModelRepositoryProtocol):
    """
    Interface de acesso a modelos treinados persistidos em disco.

    Encapsula as operações de CRUD sobre modelos, permitindo que
    os services sejam testados com mocks sem dependência de disco.
    """

    def save(
        self,
        model: object,
        metadata: dict,
        model_type: str,
        feature_names: list[str],
        X_train: pd.DataFrame,
    ) -> str:
        """
        Persiste um modelo treinado com seus artefatos.

        Args:
            model: Modelo treinado (sklearn-compatible).
            metadata: Metadados (métricas, confusion_matrix, etc.).
            model_type: Tipo do modelo ("xgboost", etc.).
            feature_names: Nomes das features usadas.
            X_train: Dados de treinamento (amostra para drift detection).

        Returns:
            model_id gerado (ex: "xgb_20260221_101345").
        """
        return save_trained_model(model, metadata, model_type, feature_names, X_train)

    def load(self, model_id: str | None = None) -> tuple[object, dict]:
        """
        Carrega um modelo e seus metadados do disco.

        Args:
            model_id: ID do modelo. Se None, carrega o mais recente.

        Returns:
            Tupla (model, metadata).

        Raises:
            FileNotFoundError: Se o modelo não existe.
        """
        return load_trained_model(model_id)

    def list(self) -> list[dict]:
        """
        Lista todos os modelos treinados.

        Returns:
            Lista de dicts ordenada por data (mais recente primeiro).
        """
        return list_trained_models()

    def delete(self, model_id: str) -> bool:
        """
        Remove um modelo do disco e do índice.

        Args:
            model_id: ID do modelo a remover.

        Returns:
            True se removido com sucesso, False se não encontrado.
        """
        return delete_model(model_id)

    def load_reference(self, model_id: str | None = None) -> pd.DataFrame | None:
        """
        Carrega a amostra de referência do modelo para drift detection.

        Args:
            model_id: ID do modelo. Se None, usa o mais recente.

        Returns:
            DataFrame com amostra de referência ou None.
        """
        return load_reference_data(model_id)

    def get_latest_id(self) -> str | None:
        """Retorna o model_id mais recente ou None se não há modelos."""
        return get_latest_model_id()

    def clear_cache(self) -> None:
        """Limpa o cache de modelos em memória."""
        clear_cache()
