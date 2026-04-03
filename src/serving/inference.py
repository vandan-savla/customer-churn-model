"""Inference helpers for loading the latest churn model bundle."""

import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.utils.mlflow_config import DEFAULT_EXPERIMENT_NAME, configure_mlflow, get_project_root

PROJECT_ROOT = get_project_root()
LOCAL_SERVING_DIR = PROJECT_ROOT / "src" / "serving" / "model"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(LOCAL_SERVING_DIR / "bundle")))
FEATURE_METADATA_PATH = Path(os.getenv("FEATURE_METADATA_PATH", str(LOCAL_SERVING_DIR / "metadata.json")))


def _load_local_bundle():
    model = mlflow.pyfunc.load_model(str(MODEL_DIR))
    if FEATURE_METADATA_PATH.exists():
        metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
        feature_cols = metadata["feature_columns"]
    else:
        feature_cols = json.loads((LOCAL_SERVING_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    return model, feature_cols


def _load_latest_mlflow_bundle():
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT_NAME)
    configure_mlflow(experiment_name=experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' does not exist")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No runs found for experiment '{experiment_name}'")

    run_id = runs[0].info.run_id
    metadata_path = client.download_artifacts(run_id, "deployment_metadata.json")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    return model, metadata["feature_columns"]


try:
    model, FEATURE_COLS = _load_local_bundle()
except Exception:
    model, FEATURE_COLS = _load_latest_mlflow_bundle()

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for column in NUMERIC_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column, mapping in BINARY_MAP.items():
        if column in df.columns:
            df[column] = (
                df[column].astype(str).str.strip().map(mapping).astype("Int64").fillna(0).astype(int)
            )

    object_columns = [column for column in df.select_dtypes(include=["object"]).columns]
    if object_columns:
        df = pd.get_dummies(df, columns=object_columns, drop_first=True)

    bool_columns = df.select_dtypes(include=["bool"]).columns
    if len(bool_columns) > 0:
        df[bool_columns] = df[bool_columns].astype(int)

    return df.reindex(columns=FEATURE_COLS, fill_value=0)


def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)
    preds = model.predict(df_enc)

    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    result = preds[0] if isinstance(preds, (list, tuple)) else preds
    return "Likely to churn" if result == 1 else "Not likely to churn"
