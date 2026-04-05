# Telco Customer Churn ML

End-to-end churn prediction project that starts with notebook experimentation and evolves into a modular training, tracking, serving, and deployment workflow.

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Problem Statement

The objective is to predict whether a telecom customer is likely to churn so the business can identify high-risk customers early and design targeted retention strategies.

This project was intentionally built in stages:

1. Explore the dataset and test ideas in [notebooks/EDA.ipynb](/e:/Codes/ML/customer-churn-model/notebooks/EDA.ipynb)
2. Move stable logic into reusable modules under `src/`
3. Build a repeatable training pipeline in [scripts/run_pipeline.py](/e:/Codes/ML/customer-churn-model/scripts/run_pipeline.py)
4. Track experiments with MLflow
5. Export a serving-ready model bundle for inference
6. Deploy the application with Docker and CI/CD

That progression matters because notebooks are good for experimentation, but reusable modules and pipelines are what make an ML system maintainable and deployable.

## Architecture Diagram

![Telco Churn Architecture](./assets/Telco%20Churn.drawio.png)

The architecture follows the same journey from experimentation to production:

- [notebooks/EDA.ipynb](/e:/Codes/ML/customer-churn-model/notebooks/EDA.ipynb) was used for data understanding, feature exploration, and early model experiments
- [src/data/load_data.py](/e:/Codes/ML/customer-churn-model/src/data/load_data.py), [src/data/preprocess.py](/e:/Codes/ML/customer-churn-model/src/data/preprocess.py), and [src/features/build_features.py](/e:/Codes/ML/customer-churn-model/src/features/build_features.py) now own the reusable data pipeline
- [src/models/train.py](/e:/Codes/ML/customer-churn-model/src/models/train.py), [src/models/evaluate.py](/e:/Codes/ML/customer-churn-model/src/models/evaluate.py), and [src/models/tune.py](/e:/Codes/ML/customer-churn-model/src/models/tune.py) keep modeling concerns modular
- [scripts/run_pipeline.py](/e:/Codes/ML/customer-churn-model/scripts/run_pipeline.py) orchestrates the full workflow: validation, preprocessing, feature engineering, training, evaluation, MLflow logging, and model promotion
- [src/serving/inference.py](/e:/Codes/ML/customer-churn-model/src/serving/inference.py) handles prediction-time transformations and model loading
- [src/app/main.py](/e:/Codes/ML/customer-churn-model/src/app/main.py) exposes the model through FastAPI and a mounted Gradio UI

## Why The Code Was Modularized

After experimentation, the code was split into modules so each step had one clear responsibility:

- data loading and cleaning
- feature engineering
- training and evaluation
- validation
- inference
- application serving

Benefits of this structure:

- easier to retrain consistently
- easier to debug and extend
- cleaner train/serve separation
- easier deployment
- simpler collaboration compared to notebook-only code

## Training Workflow

The main entrypoint is [scripts/run_pipeline.py](/e:/Codes/ML/customer-churn-model/scripts/run_pipeline.py).

At a high level it:

1. loads raw customer data
2. validates required schema and business rules
3. preprocesses the dataset
4. builds model features
5. trains the XGBoost classifier
6. evaluates metrics
7. logs the run to MLflow
8. exports the promoted serving bundle to [src/serving/model](/e:/Codes/ML/customer-churn-model/src/serving/model)

This makes the training flow reproducible instead of depending on notebook cells or manual steps.

## Inference And Serving

Inference is kept separate from training.

- [src/serving/inference.py](/e:/Codes/ML/customer-churn-model/src/serving/inference.py) loads the promoted model bundle
- [src/serving/model](/e:/Codes/ML/customer-churn-model/src/serving/model) stores the model and metadata used at runtime

This is important because the application should serve a stable promoted model, not rebuild or search experiment artifacts dynamically.

The serving bundle includes:

- the serialized model
- feature metadata
- preprocessing metadata
- run linkage back to MLflow

That keeps training traceable and inference predictable.

## MLflow

MLflow is used for experiment tracking, not as the primary runtime dependency for serving.

It stores:

- experiment history
- run parameters
- evaluation metrics
- logged artifacts
- model artifacts for each run

Project MLflow assets:

- tracking database: [mlflow.db](/e:/Codes/ML/customer-churn-model/mlflow.db)
- artifact store: [mlartifacts](/e:/Codes/ML/customer-churn-model/mlartifacts)

## Deployment

The project includes:

- [Dockerfile](/e:/Codes/ML/customer-churn-model/Dockerfile) for the FastAPI app
- [Dockerfile.mlflow](/e:/Codes/ML/customer-churn-model/Dockerfile.mlflow) for MLflow
- [docker-compose.yml](/e:/Codes/ML/customer-churn-model/docker-compose.yml) for running both services together

CI/CD is defined in [.github/workflows/workflow.yaml](/e:/Codes/ML/customer-churn-model/.github/workflows/workflow.yaml).

On pushes to `main`, the workflow:

1. checks out the code
2. authenticates with AWS
3. logs in to Amazon ECR
4. builds the application image
5. pushes the image to ECR

This makes the repo a deployable ML application rather than only a local experiment.

## Important Files

- [notebooks/EDA.ipynb](/e:/Codes/ML/customer-churn-model/notebooks/EDA.ipynb): initial experimentation
- [scripts/run_pipeline.py](/e:/Codes/ML/customer-churn-model/scripts/run_pipeline.py): end-to-end training pipeline
- [src/models/train.py](/e:/Codes/ML/customer-churn-model/src/models/train.py): model training logic
- [src/models/evaluate.py](/e:/Codes/ML/customer-churn-model/src/models/evaluate.py): evaluation logic
- [src/serving/inference.py](/e:/Codes/ML/customer-churn-model/src/serving/inference.py): runtime inference
- [src/app/main.py](/e:/Codes/ML/customer-churn-model/src/app/main.py): FastAPI + Gradio app
- [.github/workflows/workflow.yaml](/e:/Codes/ML/customer-churn-model/.github/workflows/workflow.yaml): build and push pipeline

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### Train The Model

```bash
python scripts/run_pipeline.py --input data/raw/telco_customer_churn.csv --target Churn
```

### Start MLflow UI

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Start The App

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

### Run With Docker

```bash
docker compose up --build
```

## Summary

This project demonstrates a practical ML lifecycle:

- experiment in notebooks
- modularize reusable code
- train through a repeatable pipeline
- track runs with MLflow
- promote a serving-ready model bundle
- expose predictions through FastAPI and Gradio
- package and deploy through Docker and CI/CD
