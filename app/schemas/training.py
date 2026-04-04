"""
Training-related Pydantic schemas

Decisions:
  - TrainData.timestamps: int64 unix timestamp (as specified)

"""

from __future__ import annotations
from pydantic import BaseModel, Field, model_validator


class TrainData(BaseModel):
    timestamps: list[int] = Field(
        ...,
        description="Unix timestamps in seconds, strictly increasing",
    )
    values: list[float] = Field(
        ...,
        description="Measured values corresponding to each timestamp",
    )

    @model_validator(mode="after")
    def same_length(self) -> TrainData:
        if len(self.timestamps) != len(self.values):
            raise ValueError(
                f"timestamps and values must have the same length, "
                f"got {len(self.timestamps)} vs {len(self.values)}"
            )
        return self


class TrainResponse(BaseModel):
    series_id: str
    version: str
    points_used: int