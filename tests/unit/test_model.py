"""
Unit tests for AnomalyDetectionModel in app.core.model.
"""

import pytest
import numpy as np
from app.core.model import AnomalyDetectionModel


def make_model(values):
    m = AnomalyDetectionModel()
    m.fit(values)
    return m


class TestFit:
    def test_mean_and_std_are_correct(self):
        values = list(range(1, 101))  # 1..100
        m = make_model(values)
        assert abs(m.params.mean - np.mean(values)) < 1e-9
        assert abs(m.params.std - np.std(values)) < 1e-9

    def test_n_points_recorded(self):
        values = [1.0] * 50 + [2.0] * 50
        m = make_model(values)
        assert m.params.n_points == 100

    def test_returns_params(self):
        values = [float(i) for i in range(30)]
        m = AnomalyDetectionModel()
        params = m.fit(values)
        assert params is not None
        assert params.mean is not None


class TestPredict:
    def test_normal_point_not_anomaly(self):
        values = [float(i) for i in range(100)]
        m = make_model(values)
        assert m.predict(49.5) is False  # close to mean, should not be anomaly

    def test_extreme_high_value_is_anomaly(self):
        values = [float(i) for i in range(100)]
        m = make_model(values)
        assert m.predict(10_000.0) is True  # far above mean, should be anomaly

    def test_boundary_exactly_at_threshold_is_not_anomaly(self):
        values = [float(i) for i in range(100)]
        m = make_model(values)
        threshold = m.params.mean + 3 * m.params.std
        assert (
            m.predict(threshold) is False
        )  # exactly at threshold, should not be anomaly

    def test_known_limitation_lower_outlier_not_anomaly(self):
        """
        A value far BELOW the mean is not  by design (spec limitation).
        """
        values = [float(i) for i in range(100)]
        m = make_model(values)
        assert m.predict(-10_000.0) is False  # NOT an anomaly by design

    def test_predict_before_fit_raises(self):
        m = AnomalyDetectionModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict(1.0)


class TestLoadParams:
    def test_round_trip(self):
        values = [float(i) for i in range(50)]
        m1 = AnomalyDetectionModel()
        params = m1.fit(values)

        m2 = AnomalyDetectionModel()
        m2.load_params(params)

        assert m2.predict(1000.0) == m1.predict(
            1000.0
        )  # Both should agree on this extreme value
        assert m2.predict(25.0) == m1.predict(
            25.0
        )  # Both should agree on this normal value
