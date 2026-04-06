"""
Unit tests for versioning logic in app.core.versioning.
"""

from app.core.versioning import (
    next_version,
    latest_version,
    append_version,
    version_exists,
    get_version_entry,
)
from app.core.model import ModelParams


def make_params(n=50):
    return ModelParams(mean=10.0, std=2.0, n_points=n)


class TestNextVersion:
    def test_empty_manifest_gives_v1(self):
        manifest = {"series_id": "s1", "versions": []}
        assert next_version(manifest) == "v1"

    def test_one_version_gives_v2(self):
        manifest = {"versions": [{"version": "v1"}]}
        assert next_version(manifest) == "v2"

    def test_monotonic_increment(self):
        manifest = {"versions": [{"version": f"v{i}"} for i in range(1, 6)]}
        assert next_version(manifest) == "v6"


class TestLatestVersion:
    def test_empty_returns_none(self):
        assert latest_version({"versions": []}) is None

    def test_returns_last_entry(self):
        manifest = {
            "versions": [
                {"version": "v1"},
                {"version": "v2"},
                {"version": "v3"},
            ]
        }
        assert latest_version(manifest) == "v3"


class TestAppendVersion:
    def test_appends_entry(self):
        manifest = {"series_id": "s1", "versions": []}
        params = make_params()
        updated = append_version(manifest, "v1", params)
        assert len(updated["versions"]) == 1
        assert updated["versions"][0]["version"] == "v1"

    def test_does_not_mutate_original(self):
        manifest = {"versions": []}
        append_version(manifest, "v1", make_params())
        assert (
            len(manifest["versions"]) == 0
        )  # Original manifest should remain unchanged

    def test_entry_has_required_fields(self):
        manifest = {"versions": []}
        updated = append_version(manifest, "v1", make_params(n=42))
        entry = updated["versions"][0]
        assert entry["n_points"] == 42
        assert "trained_at" in entry
        assert "mean" in entry
        assert "std" in entry


class TestVersionExists:
    def test_existing_version(self):
        manifest = {"versions": [{"version": "v1"}, {"version": "v2"}]}
        assert version_exists(manifest, "v1") is True
        assert version_exists(manifest, "v2") is True

    def test_missing_version(self):
        manifest = {"versions": [{"version": "v1"}]}
        assert version_exists(manifest, "v99") is False
