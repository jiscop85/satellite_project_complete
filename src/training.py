cfrom __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import ProjectConfig
from .features import build_feature_stack, flatten_features, flatten_labels, preprocess_image
from .io_raster import read_label, read_raster


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_names: List[str]
    config: Dict
    metrics: Dict


def prepare_samples(feature_stack: np.ndarray, labels: np.ndarray, cfg: ProjectConfig):
    X = flatten_features(feature_stack)
    y = flatten_labels(labels).astype(np.int32)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y) & (y >= 0)
    X = X[valid]
    y = y[valid]
    if len(y) == 0:
        raise ValueError("No valid labeled pixels found.")

    rng = np.random.default_rng(cfg.random_state)
    classes = np.unique(y)
    idx_all = []

    if cfg.class_balance:
        per_class = max(1, cfg.max_samples_total // max(len(classes), 1))
        per_class = min(per_class, cfg.max_samples_per_class)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            if len(idx) > per_class:
                idx = rng.choice(idx, size=per_class, replace=False)
            idx_all.extend(idx.tolist())
    else:
        total = min(cfg.max_samples_total, len(y))
        idx_all = rng.choice(len(y), size=total, replace=False).tolist()

    idx_all = np.array(idx_all, dtype=int)
    rng.shuffle(idx_all)
    Xs = X[idx_all]
    ys = y[idx_all]
    return Xs, ys


def build_pipeline(cfg: ProjectConfig) -> Pipeline:
    model = ExtraTreesClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        class_weight="balanced",
        bootstrap=False,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def train_model(
    image_path: str | Path,
    labels_path: str | Path,
    model_out: str | Path,
    metrics_out: Optional[str | Path] = None,
    cfg: Optional[ProjectConfig] = None,
):
    cfg = cfg or ProjectConfig()

    image, meta = read_raster(image_path)
    labels, _ = read_label(labels_path, resample_to=(meta["height"], meta["width"]))
    image = preprocess_image(image, cfg)
    feature_stack, feature_names = build_feature_stack(image, cfg)
    X, y = prepare_samples(feature_stack, labels, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = evaluate(y_test, preds)

    if cfg.cv_folds >= 2 and len(np.unique(y_train)) > 1:
        min_count = int(np.min(np.unique(y_train, return_counts=True)[1]))
        n_splits = max(2, min(cfg.cv_folds, min_count))
        if n_splits >= 2:
            folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
            accs, f1s = [], []
            for tr_idx, val_idx in folds.split(X_train, y_train):
                fold_pipe = build_pipeline(cfg)
                fold_pipe.fit(X_train[tr_idx], y_train[tr_idx])
                pred = fold_pipe.predict(X_train[val_idx])
                accs.append(accuracy_score(y_train[val_idx], pred))
                f1s.append(f1_score(y_train[val_idx], pred, average="weighted", zero_division=0))
            metrics.update(
                {
                    "cv_accuracy_mean": float(np.mean(accs)),
                    "cv_accuracy_std": float(np.std(accs)),
                    "cv_f1_weighted_mean": float(np.mean(f1s)),
                    "cv_f1_weighted_std": float(np.std(f1s)),
                }
            )

    bundle = ModelBundle(
        pipeline=pipeline,
        feature_names=feature_names,
        config=cfg.to_dict(),
        metrics=metrics,
    )

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)

    if metrics_out is not None:
        metrics_out = Path(metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "feature_names": feature_names, "config": cfg.to_dict()}, f, ensure_ascii=False, indent=2)

    return bundle


def load_bundle(model_path: str | Path) -> ModelBundle:
    obj = joblib.load(model_path)
    if isinstance(obj, ModelBundle):
        return obj
    if isinstance(obj, dict) and {"pipeline", "feature_names", "config", "metrics"}.issubset(obj.keys()):
        return ModelBundle(**obj)
    raise TypeError("Unsupported model bundle format")
