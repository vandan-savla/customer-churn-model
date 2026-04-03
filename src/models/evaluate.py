from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X_test, y_test, threshold: float = 0.35) -> Tuple[Dict[str, float], str, np.ndarray]:
    start_time = time.time()
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    pred_time = time.time() - start_time

    metrics = {
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pred_time": pred_time,
    }
    report = classification_report(y_test, predictions, digits=3, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)
    return metrics, report, matrix
