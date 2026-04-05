"""
Persistence.

Joblib save/load with path resolution.

Storage layout:
storage/                
    <series_id>/                # one directory per series_id
        manifest.json           # version list + metadata / version
        v1/                     # one directory / version
            model.joblib        # serialized object
            training_data.json  # raw training data used for plotting and validation
        v2/
        ...

Joblib is preferred over pickle for numpy-heavy objects:
it uses memory-mapped files and compresses arrays efficiently.
"""

import json
import os
from pathlib import Path
import joblib
from app.core.model import ModelParams

STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "storage"))


def _series_dir(series_id: str) -> Path:
    return STORAGE_ROOT / series_id

def _version_dir(series_id: str, version: str) -> Path:
    return _series_dir(series_id) / version

def _manifest_path(series_id: str) -> Path:
    return _series_dir(series_id) / "manifest.json"

def load_manifest(series_id: str) -> dict:
    path = _manifest_path(series_id)
    if not path.exists():
        return {"series_id": series_id, "versions": []}
    with open(path) as f:
        return json.load(f)

def save_manifest(series_id: str, manifest: dict) -> None:
    path = _manifest_path(series_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

def save_model(series_id: str, version: str, params: ModelParams) -> None:
    vdir = _version_dir(series_id, version)
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(params, vdir / "model.joblib")


def load_model(series_id: str, version: str) -> ModelParams:
    path = _version_dir(series_id, version) / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No model found for series='{series_id}' version='{version}'. "
            f"Train the model first via POST /fit/{series_id}."
        )
    return joblib.load(path)

def save_training_data(series_id: str, version: str, timestamps: list[int], values: list[float]) -> None:
    vdir = _version_dir(series_id, version)
    vdir.mkdir(parents=True, exist_ok=True)
    with open(vdir / "training_data.json", "w") as f:
        json.dump({"timestamps": timestamps, "values": values}, f)


def load_training_data(series_id: str, version: str) -> dict:
    path = _version_dir(series_id, version) / "training_data.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No training data found for series='{series_id}' version='{version}'. "
            f"This version may have been trained before plot support was added."
        )
    with open(path) as f:
        return json.load(f)

def series_exists(series_id: str) -> bool:
    return _manifest_path(series_id).exists()

def list_series() -> list[str]:
    if not STORAGE_ROOT.exists():
        return []
    return [d.name for d in STORAGE_ROOT.iterdir() if d.is_dir()]
