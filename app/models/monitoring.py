"""
Schemas Pydantic para monitoramento de drift e feedback.
"""
from pydantic import BaseModel, Field
from typing import Optional


class DriftResponse(BaseModel):
    """Resposta do monitoramento de drift."""
    status: str
    total_features_checked: int = 0
    drift_detected: int = 0
    warnings: int = 0
    details: dict = Field(default_factory=dict)
    prediction_stats: dict = Field(default_factory=dict)
    message: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request para registrar o outcome real de uma predição."""
    prediction_id: str = Field(..., description="UUID da predição")
    actual_outcome: int = Field(..., ge=0, le=1, description="0 = Sem risco real, 1 = Ficou defasado")


class FeedbackResponse(BaseModel):
    """Resposta do registro de feedback."""
    success: bool
    message: str
    prediction_id: str
