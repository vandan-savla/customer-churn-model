#!/usr/bin/env python3
"""Train the churn pipeline end to end and track it with MLflow."""

import argparse
import json
import sys
import time
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_model
from src.models.train import train_model
from src.utils.mlflow_config import build_model_uri, configure_mlflow, get_project_root
from src.utils.validate_data import validate_telco_data


def export_serving_bundle(model, feature_cols, target, project_root: Path, experiment_name: str, run_id: str) -> None:
    serving_dir = project_root / "src" / "serving" / "model"
    serving_dir.mkdir(parents=True, exist_ok=True)

    feature_columns_path = serving_dir / "feature_columns.json"
    preprocessing_path = serving_dir / "preprocessing.pkl"
    metadata_path = serving_dir / "metadata.json"
    model_path = serving_dir / "bundle"

    feature_columns_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    joblib.dump({"feature_columns": feature_cols, "target": target}, preprocessing_path)
    metadata_path.write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "run_id": run_id,
                "model_uri": build_model_uri(run_id),
                "target": target,
                "feature_columns": feature_cols,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if model_path.exists():
        import shutil

        shutil.rmtree(model_path)
    mlflow.sklearn.save_model(sk_model=model, path=str(model_path))


def save_and_log_model_bundle(model, project_root: Path) -> None:
    model_bundle_dir = project_root / "artifacts" / "mlflow_model"
    if model_bundle_dir.exists():
        import shutil

        shutil.rmtree(model_bundle_dir)
    mlflow.sklearn.save_model(sk_model=model, path=str(model_bundle_dir))
    mlflow.log_artifacts(str(model_bundle_dir), artifact_path="model")


def log_json_artifact(payload, output_path: Path, artifact_subdir: str | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if artifact_subdir:
        mlflow.log_artifact(str(output_path), artifact_path=artifact_subdir)
    else:
        mlflow.log_artifact(str(output_path))


def main(args):
    project_root = get_project_root()
    temp_dir = project_root / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    configure_mlflow(
        experiment_name=args.experiment,
        tracking_uri=args.mlflow_uri,
        artifact_root=args.mlflow_artifact_root,
    )

    with mlflow.start_run():
        run = mlflow.active_run()
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "model": "xgboost",
                "threshold": args.threshold,
                "test_size": args.test_size,
                "target": args.target,
                "input_path": args.input,
            }
        )

        print("Loading data...")
        df = load_data(args.input)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        print("Validating data quality...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:
            log_json_artifact(failed, project_root / "artifacts" / "failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")

        print("Preprocessing data...")
        df = preprocess_data(df, target_col=args.target)

        processed_path = project_root / "data" / "processed" / "telco_churn_processed.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        print("Building features...")
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in data")

        df_enc = build_features(df, target_col=args.target)
        bool_cols = df_enc.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            df_enc[bool_cols] = df_enc[bool_cols].astype(int)

        feature_cols = list(df_enc.drop(columns=[args.target]).columns)
        artifacts_dir = project_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        feature_columns_path = artifacts_dir / "feature_columns.json"
        feature_columns_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
        preprocessing_path = artifacts_dir / "preprocessing.pkl"
        joblib.dump({"feature_columns": feature_cols, "target": args.target}, preprocessing_path)

        feature_columns_text_path = artifacts_dir / "feature_columns.txt"
        feature_columns_text_path.write_text("\n".join(feature_cols), encoding="utf-8")
        mlflow.log_artifact(str(feature_columns_text_path))
        mlflow.log_artifact(str(preprocessing_path))

        X = df_enc.drop(columns=[args.target])
        y = df_enc[args.target]

        print("Splitting train/test data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=42,
        )

        positive_count = int((y_train == 1).sum())
        if positive_count == 0:
            raise ValueError("Training split has no positive samples; cannot train churn classifier.")
        scale_pos_weight = float((y_train == 0).sum() / positive_count)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        print("Training XGBoost model...")
        model, train_time, training_params = train_model(
            X_train=X_train,
            y_train=y_train,
            scale_pos_weight=scale_pos_weight,
        )
        mlflow.log_metric("train_time", train_time)
        mlflow.log_params(training_params)

        print("Evaluating model...")
        metrics, report, confusion = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=args.threshold,
        )
        mlflow.log_metrics(metrics)
        log_json_artifact(
            confusion.tolist(),
            project_root / "artifacts" / "confusion_matrix.json",
        )

        print("Saving model to MLflow...")
        save_and_log_model_bundle(model, project_root)
        log_json_artifact(
            {
                "run_id": run_id,
                "model_uri": build_model_uri(run_id),
                "experiment_name": args.experiment,
                "target": args.target,
                "feature_columns": feature_cols,
            },
            project_root / "artifacts" / "deployment_metadata.json",
        )

        export_serving_bundle(
            model=model,
            feature_cols=feature_cols,
            target=args.target,
            project_root=project_root,
            experiment_name=args.experiment,
            run_id=run_id,
        )

        print("Training complete.")
        print(f"Run ID: {run_id}")
        print(f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1']:.3f} | ROC AUC: {metrics['roc_auc']:.3f}")
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", type=str, default="Churn")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--experiment", type=str, default="Telco Churn")
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Defaults to sqlite:///mlflow.db or MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--mlflow_artifact_root",
        type=str,
        default=None,
        help="Artifact root for MLflow. Defaults to ./mlartifacts or MLFLOW_ARTIFACT_ROOT.",
    )

    main(parser.parse_args())
