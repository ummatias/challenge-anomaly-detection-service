"""
Versioning

read and write model version metadata
Version format: "v1", "v2", ...

Metadata includes:
    - version
    - trained_at (ISO timestamp)
    - n_points
    - mean
    - std
"""

from datetime import datetime, timezone
from app.core.model import ModelParams


def next_version(manifest: dict) -> str:
    versions = manifest.get("versions", [])
    return f"v{len(versions) + 1}"


def latest_version(manifest: dict) -> str | None:
    versions = manifest.get("versions", [])
    if not versions:
        return None
    return versions[-1]["version"]


def append_version(manifest: dict, version: str, params: ModelParams) -> dict:
    """
    Return an updated copy of manifest with the new version appended.
    Caller is responsible for saving.
    """
    entry = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_points": params.n_points,
        "mean": params.mean,
        "std": params.std,
    }
    updated = dict(manifest)
    updated["versions"] = list(manifest.get("versions", [])) + [entry]
    return updated


def get_version_entry(manifest: dict, version: str) -> dict | None:
    for entry in manifest.get("versions", []):
        if entry["version"] == version:
            return entry
    return None


def version_exists(manifest: dict, version: str) -> bool:
    return get_version_entry(manifest, version) is not None
