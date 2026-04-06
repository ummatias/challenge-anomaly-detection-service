"""
Unit tests for preflight validation logic in app.core.validation.
"""

import pytest
from app.core.validation import (
    ValidationError,
    check_minimum_points,
    check_no_invalid_values,
    check_non_constant,
    check_monotonic_timestamps,
    validate,
    MIN_POINTS,
)

GOOD_VALUES = [float(i) for i in range(MIN_POINTS)]
GOOD_TIMESTAMPS = list(range(MIN_POINTS))


class TestMinimumPoints:
    def test_exact_minimum_passes(self):
        check_minimum_points(GOOD_VALUES)

    def test_below_minimum_fails(self):
        with pytest.raises(ValidationError) as exc:
            check_minimum_points([1.0] * (MIN_POINTS - 1))
        assert exc.value.rule == "minimum_points"

    def test_empty_fails(self):
        with pytest.raises(ValidationError) as exc:
            check_minimum_points([])
        assert exc.value.rule == "minimum_points"


class TestNoInvalidValues:
    def test_clean_data_passes(self):
        check_no_invalid_values([1.0, 2.0, 3.0])

    def test_nan_fails(self):
        import math

        with pytest.raises(ValidationError) as exc:
            check_no_invalid_values([1.0, math.nan, 3.0])
        assert exc.value.rule == "no_nan"

    def test_inf_fails(self):
        with pytest.raises(ValidationError) as exc:
            check_no_invalid_values([1.0, float("inf"), 3.0])
        assert exc.value.rule == "no_inf"

    def test_negative_inf_fails(self):
        with pytest.raises(ValidationError) as exc:
            check_no_invalid_values([float("-inf"), 1.0])
        assert exc.value.rule == "no_inf"


class TestNonConstant:
    def test_varying_data_passes(self):
        check_non_constant([1.0, 2.0, 3.0])

    def test_constant_data_fails(self):
        with pytest.raises(ValidationError) as exc:
            check_non_constant([5.0] * 50)
        assert exc.value.rule == "non_constant"

    def test_two_unique_values_passes(self):
        check_non_constant([1.0, 2.0, 1.0, 2.0])


class TestMonotonicTimestamps:
    def test_strictly_increasing_passes(self):
        check_monotonic_timestamps([1, 2, 3, 4, 5])

    def test_duplicate_timestamp_fails(self):
        with pytest.raises(ValidationError) as exc:
            check_monotonic_timestamps([1, 2, 2, 4])
        assert exc.value.rule == "monotonic_timestamps"

    def test_decreasing_fails(self):
        with pytest.raises(ValidationError) as exc:
            check_monotonic_timestamps([5, 4, 3])
        assert exc.value.rule == "monotonic_timestamps"


class TestValidateEntryPoint:
    def test_valid_data_passes(self):
        validate(GOOD_TIMESTAMPS, GOOD_VALUES)  # No exception

    def test_error_message_is_structured(self):
        with pytest.raises(ValidationError) as exc:
            validate(GOOD_TIMESTAMPS, [1.0] * (MIN_POINTS - 1))
        assert exc.value.rule == "minimum_points"
        assert exc.value.detail is not None
