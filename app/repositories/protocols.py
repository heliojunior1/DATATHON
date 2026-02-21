"""
Protocolo (interface) do Repository de Modelos.

Define o contrato que qualquer implementação de repositório deve cumprir,
permitindo troca de backend (disco, banco, cloud) sem alterar os services.
"""
from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class ModelRepositoryProtocol(Protocol):
    """Interface de acesso a modelos treinados."""

    def save(
        self,
        model: object,
        metadata: dict,
        model_type: str,
        feature_names: list[str],
        X_train: pd.DataFrame,
    ) -> str:
        """Persiste um modelo e retorna o model_id gerado."""
        ...

    def load(self, model_id: str | None = None) -> tuple[object, dict]:
        """Carrega (model, metadata). Raises FileNotFoundError se não existe."""
        ...

    def list(self) -> list[dict]:
        """Lista todos os modelos (mais recente primeiro)."""
        ...

    def delete(self, model_id: str) -> bool:
        """Remove um modelo. Retorna True se deletado, False se não encontrado."""
        ...

    def load_reference(self, model_id: str | None = None) -> pd.DataFrame | None:
        """Carrega amostra de referência para drift detection."""
        ...

    def get_latest_id(self) -> str | None:
        """Retorna o model_id mais recente ou None."""
        ...

    def clear_cache(self) -> None:
        """Limpa o cache de modelos em memória."""
        ...
