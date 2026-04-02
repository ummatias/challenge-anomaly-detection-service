"""
Preflight validation

The validate() runs all checks and raises ValidationError
Separated checks for unit tests

- Minimum 30 data points (configurable via env)
- No NaN or infinite values
- Non-constant values (std=0 would make the model meaningless)
- Monotonic timestamps (strictly increasing, no duplicates)
"""

import math
import os
from typing import Sequence

MIN_POINTS = int(os.getenv("MIN_TRAIN_POINTS", 30))


class ValidationError(ValueError):
    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"[{rule}] {detail}")


def check_minimum_points(values: Sequence[float]) -> None:
    if len(values) < MIN_POINTS:
        raise ValidationError(
            rule="minimum_points",
            detail=f"Need at least {MIN_POINTS} data points, got {len(values)}. "
                   f"With fewer points, the sample std is unreliable as an estimator."
                   f"Override with MIN_TRAIN_POINTS on .env",
        )


def check_no_invalid_values(values: Sequence[float]) -> None:
    for i, v in enumerate(values):
        if math.isnan(v):
            raise ValidationError(
                rule="no_nan",
                detail=f"NaN found at index {i}.",
            )
        if math.isinf(v):
            raise ValidationError(
                rule="no_inf",
                detail=f"Infinite value found at index {i}.",
            )


def check_non_constant(values: Sequence[float]) -> None:
    if len(set(values)) == 1:
        raise ValidationError(
            rule="non_constant",
            detail="All values are identical. std=0 means the model would flag "
                   "every point above the mean as anomalous, which is meaningless.",
        )


def check_monotonic_timestamps(timestamps: Sequence[int]) -> None:
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            raise ValidationError(
                rule="monotonic_timestamps",
                detail=f"Timestamp at index {i} ({timestamps[i]}) is not strictly "
                       f"greater than previous ({timestamps[i - 1]}). "
                       f"Timestamps must be strictly increasing (unix epoch, no duplicates).",
            )


def validate(timestamps: Sequence[int], values: Sequence[float]) -> None:
    """
    Run all preflight checks. 
    Raises ValidationError on first failure.
    """
    check_minimum_points(values)
    check_no_invalid_values(values)
    check_non_constant(values)
    check_monotonic_timestamps(timestamps)
