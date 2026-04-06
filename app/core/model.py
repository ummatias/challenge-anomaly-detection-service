"""
AnomalyDetectionModel: exactly as specified in the challenge.

Limitations:
    - The threshold is one-sided (upper only).
    - Time-series agnostic: it does not consider temporal patterns, only the distribution of values.
    - Does not handle seasonality, drift, or concept shift.
    - Can be evolved to incorporate incorporate time windows, seasonality, drift detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ModelParams:
    mean: float
    std: float
    n_points: int


class AnomalyDetectionModel:

    def __init__(self) -> None:
        self._params: ModelParams | None = None

    def fit(self, values: Sequence[float]) -> ModelParams:
        arr = np.array(values, dtype=np.float64)
        self._params = ModelParams(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            n_points=len(arr),
        )
        return self._params

    def predict(self, value: float) -> bool:
        if self._params is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return value > self._params.mean + 3 * self._params.std

    def load_params(self, params: ModelParams) -> None:
        self._params = params

    @property
    def params(self) -> ModelParams | None:
        return self._params
