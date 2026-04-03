from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from xgboost import XGBClassifier


DEFAULT_XGBOOST_PARAMS: Dict[str, float | int | str] = {
    "n_estimators": 301,
    "learning_rate": 0.034,
    "max_depth": 7,
    "subsample": 0.95,
    "colsample_bytree": 0.98,
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "logloss",
}


def build_model(scale_pos_weight: float, overrides: Optional[Dict[str, float | int | str]] = None) -> XGBClassifier:
    params = {**DEFAULT_XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}
    if overrides:
        params.update(overrides)
    return XGBClassifier(**params)


def train_model(
    X_train,
    y_train,
    scale_pos_weight: float,
    overrides: Optional[Dict[str, float | int | str]] = None,
) -> Tuple[XGBClassifier, float, Dict[str, float | int | str]]:
    params = {**DEFAULT_XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}
    if overrides:
        params.update(overrides)

    model = XGBClassifier(**params)
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    return model, train_time, params
