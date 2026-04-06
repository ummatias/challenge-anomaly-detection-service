from typing import Optional
from pydantic import BaseModel


class LatencyMetrics(BaseModel):
    avg: Optional[float] = None
    p95: Optional[float] = None


class HealthCheckResponse(BaseModel):
    series_trained: int
    inference_latency_ms: LatencyMetrics
    training_latency_ms: LatencyMetrics
