"""
AnomalyDetectionModel: exactly as specified in the challenge.

Limitation: the threshold is one-sided (upper only).
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
        #Restore a previously fitted model from persisted params.
        self._params = params

    @property
    def params(self) -> ModelParams | None:
        return self._params
