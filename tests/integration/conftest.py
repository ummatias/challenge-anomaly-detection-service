"""
Shared fixtures for integration tests.

Fixtures:
  - isolated_storage: move STORAGE_ROOT to a tmp dir per test
  - client: FastAPI TestClient with isolated storage
  - make_train_payload: factory function for generating training payloads
"""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """
    Redirect STORAGE_ROOT to a temp directory for each test.

    autouse=True means this runs for every test in the directory
    """
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import importlib
    import app.core.persistence as p
    importlib.reload(p)
    yield


@pytest.fixture
def client(isolated_storage):
    from app.main import app
    return TestClient(app)


@pytest.fixture
def make_train_payload():
    """
    Factory to generate training payloads with customizable parameters.

    Usage:
        def test_foo(client, make_train_payload):
            r = client.post("/fit/s1", json=make_train_payload(n=100))
    """
    def _make(n: int = 50, multiplier: float = 1.0) -> dict:
        return {
            "timestamps": list(range(n)),
            "values": [float(i) * multiplier for i in range(n)],
        }
    return _make