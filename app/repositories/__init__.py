"""
Camada de acesso a dados.

- protocols.py        → ModelRepositoryProtocol (interface / contrato)
- model_repository.py → ModelRepository (implementação disco + joblib)
"""
from app.repositories.protocols import ModelRepositoryProtocol
from app.repositories.model_repository import ModelRepository
