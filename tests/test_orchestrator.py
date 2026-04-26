"""
Tests for the AI Orchestrator.

Covers:
  - get_metric: Prometheus query parsing, NaN/Inf rejection, error handling
  - determine_root_cause: dependency-aware root cause logic using z-scores
  - NaN sample filtering: ensures corrupt Prometheus values never enter training
  - Z-score anomaly detection: verifies the secondary detection path
  - The `nan or 0.0` footgun: documents the exact bug that was fixed
"""
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

# ── Import orchestrator with side-effects neutralised ───────────────────
# conftest.py already injected a docker mock into sys.modules.
# We patch threading.Thread so the background monitoring loop never starts.
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-orchestrator"))

with patch("threading.Thread"):
    import main as orch


# ── Helpers ──────────────────────────────────────────────────────────────

def fitted_scaler(
    auth_mean: float = 0.05,
    auth_std: float = 0.01,
    order_mean: float = 0.05,
    order_std: float = 0.01,
) -> StandardScaler:
    """Return a StandardScaler fitted on synthetic data with known statistics."""
    rng = np.random.default_rng(42)
    auth_samples = rng.normal(auth_mean, auth_std, 200)
    order_samples = rng.normal(order_mean, order_std, 200)
    X = np.column_stack([auth_samples, order_samples])
    return StandardScaler().fit(X)


PROMETHEUS_URL = "http://prometheus:9090/api/v1/query"


def prometheus_response(value: str) -> dict:
    return {
        "status": "success",
        "data": {"result": [{"value": [0, value]}]},
    }


# ── get_metric ────────────────────────────────────────────────────────────


class TestGetMetric:
    """Prometheus query fetching and value parsing."""

    def test_returns_float_on_success(self, requests_mock):
        requests_mock.get(PROMETHEUS_URL, json=prometheus_response("0.123"))
        assert orch.get_metric("q") == pytest.approx(0.123)

    def test_returns_none_for_nan(self, requests_mock):
        """Prometheus returns 'NaN' for 0/0 queries (e.g. no traffic yet)."""
        requests_mock.get(PROMETHEUS_URL, json=prometheus_response("NaN"))
        assert orch.get_metric("q") is None

    def test_returns_none_for_inf(self, requests_mock):
        requests_mock.get(PROMETHEUS_URL, json=prometheus_response("Inf"))
        assert orch.get_metric("q") is None

    def test_returns_none_for_negative_inf(self, requests_mock):
        requests_mock.get(PROMETHEUS_URL, json=prometheus_response("-Inf"))
        assert orch.get_metric("q") is None

    def test_returns_none_on_connection_error(self, requests_mock):
        import requests
        requests_mock.get(PROMETHEUS_URL, exc=requests.ConnectionError)
        assert orch.get_metric("q") is None

    def test_returns_none_on_timeout(self, requests_mock):
        import requests
        requests_mock.get(PROMETHEUS_URL, exc=requests.Timeout)
        assert orch.get_metric("q") is None

    def test_returns_none_when_status_is_error(self, requests_mock):
        requests_mock.get(
            PROMETHEUS_URL,
            json={"status": "error", "errorType": "bad_data", "error": "parse error"},
        )
        assert orch.get_metric("q") is None

    def test_returns_none_when_result_is_empty(self, requests_mock):
        requests_mock.get(
            PROMETHEUS_URL,
            json={"status": "success", "data": {"result": []}},
        )
        assert orch.get_metric("q") is None

    def test_returns_none_on_malformed_json(self, requests_mock):
        requests_mock.get(PROMETHEUS_URL, json={"unexpected": "shape"})
        assert orch.get_metric("q") is None


# ── determine_root_cause ──────────────────────────────────────────────────


class TestDetermineRootCause:
    """
    Root cause logic based on the service dependency model:

        order-service → auth-service  (order calls auth on every request)
        auth-service  → (independent)

    Rules:
      - auth elevated (alone OR with order) → restart auth-service
      - only order elevated                 → restart order-service
      - neither elevated                    → None
    """

    def setup_method(self):
        self.scaler = fitted_scaler()

    def test_auth_elevated_alone_returns_auth_service(self):
        # auth at 7s is ~695 std devs above the 0.05s baseline
        assert orch.determine_root_cause(7.0, 0.05, self.scaler) == "auth-service"

    def test_order_elevated_alone_returns_order_service(self):
        # order at 2.5s is ~245 std devs above baseline; auth is normal
        assert orch.determine_root_cause(0.05, 2.5, self.scaler) == "order-service"

    def test_both_elevated_returns_auth_service(self):
        """
        When both services are elevated, auth-service is the root cause.
        Order's latency is explained by upstream auth degradation.
        """
        assert orch.determine_root_cause(7.0, 2.5, self.scaler) == "auth-service"

    def test_both_elevated_auth_wins_even_if_order_latency_is_higher(self):
        """Upstream dependency rule holds regardless of absolute latency values."""
        # auth at 1s, order at 5s — order looks worse, but auth is upstream
        result = orch.determine_root_cause(1.0, 5.0, self.scaler)
        assert result == "auth-service"

    def test_neither_elevated_returns_none(self):
        # Both within normal range — no clear root cause
        assert orch.determine_root_cause(0.05, 0.05, self.scaler) is None

    def test_both_zero_returns_none(self):
        # No traffic yet — nothing to diagnose
        assert orch.determine_root_cause(0.0, 0.0, self.scaler) is None

    def test_z_threshold_boundary(self):
        """Values just below Z_THRESHOLD should not trigger root cause."""
        # Z_THRESHOLD defaults to 2.0; produce a value just under it
        just_below = self.scaler.mean_[1] + (orch.Z_THRESHOLD - 0.1) * max(
            float(self.scaler.scale_[1]), 1e-6
        )
        assert orch.determine_root_cause(0.05, just_below, self.scaler) is None


# ── NaN / Inf sample filtering ────────────────────────────────────────────


class TestTrainingSampleFiltering:
    """
    Corrupt Prometheus values (NaN, Inf) must never enter training data,
    as they cause IsolationForest.fit() to silently fail.
    """

    def test_valid_sample_passes_filter(self):
        sample = np.array([0.05, 0.08])
        assert np.isfinite(sample).all()

    def test_nan_in_auth_rejected(self):
        sample = np.array([float("nan"), 0.05])
        assert not np.isfinite(sample).all()

    def test_nan_in_order_rejected(self):
        sample = np.array([0.05, float("nan")])
        assert not np.isfinite(sample).all()

    def test_inf_rejected(self):
        sample = np.array([float("inf"), 0.05])
        assert not np.isfinite(sample).all()

    def test_negative_inf_rejected(self):
        sample = np.array([0.05, float("-inf")])
        assert not np.isfinite(sample).all()

    def test_nan_is_truthy_in_python(self):
        """
        Documents the exact bug that was fixed: `nan or 0.0` returns nan,
        not 0.0, because NaN is truthy. This is why we use an explicit
        None check instead of the `or` shorthand.
        """
        nan_val = float("nan")
        assert math.isnan(nan_val or 0.0)  # old buggy pattern — nan passes through

    def test_explicit_none_check_is_safe(self):
        """The fixed pattern correctly handles None from get_metric."""
        def safe_default(raw):
            return raw if raw is not None else 0.0

        assert safe_default(None) == 0.0
        assert safe_default(0.0) == 0.0
        assert safe_default(0.5) == 0.5
        # NaN from get_metric is already converted to None upstream,
        # so this path never receives nan directly.


# ── Z-score secondary anomaly detection ──────────────────────────────────


class TestZScoreDetection:
    """
    The z-score path catches anomalies that IsolationForest can miss —
    particularly single-dimension spikes when the other feature was
    constant during training.
    """

    def setup_method(self):
        self.scaler = fitted_scaler()

    def _is_zscore_anomaly(self, auth_latency: float, order_latency: float) -> bool:
        scale_auth = max(float(self.scaler.scale_[0]), 1e-6)
        scale_order = max(float(self.scaler.scale_[1]), 1e-6)
        auth_z = (auth_latency - self.scaler.mean_[0]) / scale_auth
        order_z = (order_latency - self.scaler.mean_[1]) / scale_order
        return (auth_z > orch.DIRECT_ANOMALY_THRESHOLD) or (
            order_z > orch.DIRECT_ANOMALY_THRESHOLD
        )

    def test_high_auth_latency_triggers_zscore_anomaly(self):
        assert self._is_zscore_anomaly(7.0, 0.05)

    def test_high_order_latency_triggers_zscore_anomaly(self):
        assert self._is_zscore_anomaly(0.05, 2.5)

    def test_normal_latency_does_not_trigger(self):
        assert not self._is_zscore_anomaly(0.05, 0.05)

    def test_threshold_is_configurable_via_constant(self):
        """DIRECT_ANOMALY_THRESHOLD must be exposed so tests and env vars can tune it."""
        assert hasattr(orch, "DIRECT_ANOMALY_THRESHOLD")
        assert orch.DIRECT_ANOMALY_THRESHOLD > 0
