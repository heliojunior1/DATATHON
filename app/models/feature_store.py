"""
Schemas Pydantic para o Feature Store.
"""
from pydantic import BaseModel, Field
from typing import Optional


class FeatureStoreStatusResponse(BaseModel):
    """Status do Feature Store."""
    repo_path: str
    data_dir: str
    online_store_exists: bool
    registry_exists: bool
    parquet_files: list[dict] = Field(default_factory=list)
    feature_views: list[dict] = Field(default_factory=list)
    feature_views_error: Optional[str] = None


class FeatureStoreFeaturesResponse(BaseModel):
    """Features de um aluno retornadas pelo Feature Store."""
    aluno_id: str
    features: dict = Field(default_factory=dict)
    source: str = "online_store"


class FeatureStoreMaterializeResponse(BaseModel):
    """Resposta da materialização do Feature Store."""
    message: str
    parquet_files: int = 0
    feature_views: int = 0
