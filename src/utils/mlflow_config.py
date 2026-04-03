from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_EXPERIMENT_NAME = "Telco Churn"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_ARTIFACT_DIRNAME = "mlartifacts"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_tracking_uri(tracking_uri: Optional[str] = None) -> str:
    return tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def resolve_artifact_root(artifact_root: Optional[str] = None) -> str:
    raw_value = artifact_root or os.getenv("MLFLOW_ARTIFACT_ROOT")
    if raw_value:
        if "://" in raw_value:
            return raw_value
        return Path(raw_value).resolve().as_uri()

    return (get_project_root() / DEFAULT_ARTIFACT_DIRNAME).resolve().as_uri()


def configure_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    artifact_root: Optional[str] = None,
) -> str:
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri)
    resolved_artifact_root = resolve_artifact_root(artifact_root)

    mlflow.set_tracking_uri(resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=resolved_artifact_root,
        )

    mlflow.set_experiment(experiment_name)
    return resolved_tracking_uri


def build_model_uri(run_id: str, artifact_path: str = "model") -> str:
    return f"runs:/{run_id}/{artifact_path}"
