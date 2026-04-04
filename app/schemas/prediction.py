"""
Prediction-related Pydantic schemas

Decisions:
  - PredictData.timestamp: string or int (accept both), normalized to int internally
  - PredictResponse.model_version: string ("v1", "v2", etc) for better readability in API responses
"""

from pydantic import BaseModel, Field


class PredictData(BaseModel):
    timestamp: str | int = Field(
        ...,
        description="Timestamp of the point (string or unix int, both accepted)",
    )
    value: float


class PredictResponse(BaseModel):
    anomaly: bool
    model_version: str
