"""
LatencyTracker.

Uses a deque with maxlen instead of list to avoid popping or memory growth
p95 is computed over the rolling window on demand, not approximated.

Thread-safe for reads; the service layer owns write.
"""

from collections import deque
from prometheus_client import Gauge
import numpy as np


"""
    Prometheus metric to track how many series have trained models, set
    at startup and after training new models.
"""
series_trained_gauge = Gauge(
    "anomaly_series_trained_total",
    "Total number of series with trained models",
)
class LatencyTracker:

    def __init__(self, maxlen: int = 1000) -> None:
        self._samples: deque[float] = deque(maxlen=maxlen)

    def record(self, ms: float) -> None:
        self._samples.append(ms)

    @property
    def is_empty(self) -> bool:
        return len(self._samples) == 0

    def avg(self) -> float | None:
        if self.is_empty:
            return None
        return round(float(np.mean(self._samples)), 3)

    def percentile(self, p: float) -> float | None:
        if self.is_empty:
            return None
        return round(float(np.percentile(list(self._samples), p)), 3)

    def p95(self) -> float | None:
        return self.percentile(95)
