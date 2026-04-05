"""
Prediction-related Pydantic schemas

Decisions:
  - PredictData.timestamp: string in unix timestamp format for better readability in API requests
  - PredictResponse.model_version: string ("v1", "v2", etc) for better readability in API responses
"""

from pydantic import BaseModel, Field


class PredictData(BaseModel):
    timestamp: str = Field(
        ...,
        description="Timestamp should be in the unix timestamp format",
    )
    value: float


class PredictResponse(BaseModel):
    anomaly: bool
    model_version: str
