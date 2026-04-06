"""
Unit tests for LatencyTracker class in app.core.metrics.
"""

import pytest
from app.core.metrics import LatencyTracker


class TestLatencyTrackerEmpty:
    def test_avg_is_none_when_empty(self):
        lt = LatencyTracker()
        assert lt.avg() is None

    def test_p95_is_none_when_empty(self):
        lt = LatencyTracker()
        assert lt.p95() is None

    def test_is_empty_true(self):
        assert LatencyTracker().is_empty is True


class TestLatencyTrackerWithData:
    def test_avg_is_correct(self):
        lt = LatencyTracker()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            lt.record(v)
        assert abs(lt.avg() - 3.0) < 0.01  # avg of 1..5 is 3.0

    def test_p95_is_computed(self):
        lt = LatencyTracker()
        for v in range(1, 101):
            lt.record(float(v))
        assert lt.p95() is not None
        assert lt.p95() >= 94.0  # p95 of 1..100 ≈ 95

    def test_is_empty_false_after_record(self):
        lt = LatencyTracker()
        lt.record(1.0)
        assert lt.is_empty is False


class TestLatencyTrackerRollingWindow:
    def test_respects_maxlen(self):
        lt = LatencyTracker(maxlen=5)
        for v in range(100):
            lt.record(float(v))
        assert lt.avg() > 90.0  # Only the last 5 values should be in the window

    def test_old_values_evicted(self):
        lt = LatencyTracker(maxlen=3)
        for v in [1.0, 2.0, 3.0]:
            lt.record(v)
        lt.record(100.0)  # This should evict 1.0
        assert abs(lt.avg() - 35.0) < 0.01
