"""
XGBoost Snapshot Scorer — Stage 2 of the ML pipeline.

Takes a 27-dimensional feature vector (from FeatureExtractor) at a single
game snapshot and outputs P(performed_poorly), a calibrated probability.

=== WHY XGBOOST OVER LOGISTIC REGRESSION? ===

Logistic regression is a good baseline but assumes that features contribute
independently (additively) to the score. In practice, signals interact:

  - death_accel_flag AND repeat_deaths_norm together → pride-tilt (much stronger
    than either alone)
  - cs_drop_flag AND obj_absence_rate together → doom-tilt (disengaged, not
    just having a bad farm game)
  - wrong_build_flag ALONE → could be a pocket pick, not necessarily tilt

XGBoost builds decision trees that naturally capture these "AND" relationships.
It also handles the fact that Group 1 z-score features are continuous while
Group 2 signal flags are binary — trees don't care about feature scale.

=== SHAP VALUES ===

SHAP (SHapley Additive exPlanations) decomposes the model's output for a
single prediction into per-feature contributions:

  P(tilted) = base_rate + SHAP(cs_z) + SHAP(repeat_deaths_norm) + ...

This is mathematically rigorous (Shapley values from cooperative game theory)
and gives us interpretability — we know not just "this player is tilted" but
"cs drop contributed +0.12 and item sell contributed +0.18 to the score."

=== CALIBRATION ===

XGBoost probabilities are often overconfident (too close to 0 or 1).
We apply Platt scaling (logistic regression on top of raw XGBoost output)
to map raw predictions to calibrated probabilities.

After calibration, P=0.60 should mean "~60% of players in this situation
actually performed poorly."
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional heavy deps — warn at import time if unavailable
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed — SnapshotScorer unavailable. Run: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ml.features.feature_extractor import FEATURE_NAMES, FEATURE_DIM


class SnapshotScorer:
    """
    Binary classifier: does this snapshot feature vector predict poor performance?

    Workflow:
        scorer = SnapshotScorer()
        scorer.train(X_train, y_train, X_val, y_val)
        proba = scorer.predict_proba(X_test)   # calibrated probability
        scorer.save("experiments/snapshot_scorer.pkl")

        # Later:
        scorer = SnapshotScorer.load("experiments/snapshot_scorer.pkl")
        p = scorer.predict_proba(fv.vector.reshape(1, -1))
    """

    # XGBoost hyperparameters (tuned for small datasets ~200–2000 samples)
    DEFAULT_PARAMS = {
        "n_estimators":     200,
        "max_depth":        4,        # shallow trees → less overfitting on small data
        "learning_rate":    0.05,
        "subsample":        0.80,     # stochastic gradient boosting
        "colsample_bytree": 0.80,     # feature subsampling per tree
        "min_child_weight": 3,        # require ≥3 samples in leaf → regularization
        "reg_alpha":        0.1,      # L1 regularization
        "reg_lambda":       1.0,      # L2 regularization
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
    }

    def __init__(self, params: dict | None = None):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost is required. Run: pip install xgboost")
        self._model: Any = None
        self._calibrator: Any = None   # Platt scaling LogisticRegression
        self._params = params or self.DEFAULT_PARAMS
        self.feature_names = FEATURE_NAMES
        self.is_fitted = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val:  np.ndarray | None = None,
        X_cal:  np.ndarray | None = None,
        y_cal:  np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Train the XGBoost model + Platt scaling calibrator.

        Args:
            X_train: shape (n_train, FEATURE_DIM) — training features
            y_train: shape (n_train,) — binary labels (1 = performed poorly)
            X_val:   shape (n_val, FEATURE_DIM) — validation set (optional)
            y_val:   shape (n_val,) — validation labels (optional)
            X_cal:   shape (n_cal, FEATURE_DIM) — calibration set for Platt scaling.
                     Should be DIFFERENT from X_val (which is used for early stopping).
                     If None, falls back to X_val, then X_train (less ideal).

        Returns:
            dict with training metrics: {"train_logloss", "val_logloss" (if val provided)}
        """
        from xgboost import XGBClassifier

        self._validate_input(X_train, y_train)

        pos_rate = y_train.mean()
        scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-9)
        logger.info(
            f"Training SnapshotScorer: {len(X_train)} samples, "
            f"{y_train.sum():.0f} positive ({pos_rate:.1%} base rate), "
            f"scale_pos_weight={scale_pos_weight:.2f}"
        )

        params = dict(self._params)
        params["scale_pos_weight"] = scale_pos_weight

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self._model = XGBClassifier(**params)
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set or None,
            verbose=False,
        )

        # ── Platt scaling calibration ─────────────────────────────────────────
        # IMPORTANT: use a SEPARATE calibration set, NOT the val set used for
        # early stopping. Using the same set for both early stopping and
        # calibration creates a subtle data leakage: the model's n_estimators
        # was chosen to minimize val loss, so the val set is not an unbiased
        # estimate of raw probability quality. A separate cal set is unbiased.
        from sklearn.linear_model import LogisticRegression

        if X_cal is not None and y_cal is not None:
            cal_X, cal_y = X_cal, y_cal
            logger.info(f"Platt scaling on dedicated calibration set ({len(cal_y)} samples)")
        elif X_val is not None and y_val is not None:
            cal_X, cal_y = X_val, y_val
            logger.warning(
                "Platt scaling is using the val set (same set used for early stopping). "
                "Pass X_cal/y_cal for an unbiased calibration set."
            )
        else:
            cal_X, cal_y = X_train, y_train
            logger.warning("Platt scaling on training set — calibration will be overconfident.")

        raw_proba = self._model.predict_proba(cal_X)[:, 1].reshape(-1, 1)
        self._calibrator = LogisticRegression(C=10.0, solver="lbfgs")
        self._calibrator.fit(raw_proba, cal_y)
        logger.info("Platt scaling calibrator fitted.")

        self.is_fitted = True

        # ── Return metrics ────────────────────────────────────────────────────
        metrics: dict[str, float] = {}
        train_raw = self._model.predict_proba(X_train)[:, 1]
        metrics["train_logloss"] = float(_log_loss(y_train, train_raw))
        metrics["train_brier"]   = float(np.mean((train_raw - y_train) ** 2))

        if X_val is not None and y_val is not None:
            val_raw = self._model.predict_proba(X_val)[:, 1]
            metrics["val_logloss"] = float(_log_loss(y_val, val_raw))
            metrics["val_brier"]   = float(np.mean((val_raw - y_val) ** 2))

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return calibrated P(performed_poorly) for each sample.

        Args:
            X: shape (n_samples, FEATURE_DIM) or (FEATURE_DIM,) for single sample

        Returns:
            np.ndarray of shape (n_samples,) with values in [0, 1]
        """
        self._check_fitted()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self._model.predict_proba(X)[:, 1].reshape(-1, 1)
        calibrated = self._calibrator.predict_proba(raw)[:, 1]
        return calibrated

    def predict(self, X: np.ndarray, threshold: float = 0.55) -> np.ndarray:
        """Return binary predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importances(self) -> dict[str, float]:
        """
        XGBoost gain-based feature importances (mean gain per split).

        Gain importance = how much a feature reduces the loss when it is
        used as a split point, averaged across all trees. More reliable
        than split count ("weight") for comparing features of different scales.
        """
        self._check_fitted()
        gains = self._model.get_booster().get_score(importance_type="gain")
        # Map f0, f1, ... → feature names
        named = {}
        for k, v in gains.items():
            try:
                idx = int(k[1:])
                named[FEATURE_NAMES[idx]] = round(v, 4)
            except (ValueError, IndexError):
                named[k] = round(v, 4)
        # Sort by importance descending
        return dict(sorted(named.items(), key=lambda x: -x[1]))

    def shap_values(self, X: np.ndarray) -> np.ndarray | None:
        """
        Compute SHAP values for a feature matrix.

        Returns shape (n_samples, FEATURE_DIM) — each row sums to
        (prediction - base_rate), giving per-feature contribution.

        Returns None if SHAP is not installed.
        """
        if not SHAP_AVAILABLE:
            logger.warning("shap not installed — run: pip install shap")
            return None
        self._check_fitted()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        explainer   = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)
        # XGBoost binary classifier returns a 2D array for class 1
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def save(self, path: str | Path) -> None:
        """Serialize model + calibrator to a pickle file."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":        self._model,
                "calibrator":   self._calibrator,
                "params":       self._params,
                "feature_names": self.feature_names,
            }, f)
        logger.info(f"SnapshotScorer saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotScorer":
        """Deserialize model from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        scorer = cls(params=data["params"])
        scorer._model       = data["model"]
        scorer._calibrator  = data["calibrator"]
        scorer.feature_names = data.get("feature_names", FEATURE_NAMES)
        scorer.is_fitted    = True
        return scorer

    def format_importances(self) -> str:
        """Human-readable feature importance table."""
        imp = self.feature_importances()
        total = max(sum(imp.values()), 1e-9)
        lines = ["Feature importances (XGBoost gain):", f"{'Feature':<30} {'Gain':>10} {'%':>7}"]
        lines.append("-" * 50)
        for name, gain in list(imp.items())[:20]:  # top 20
            bar = "█" * int(gain / total * 30)
            lines.append(f"{name:<30} {gain:>10.2f} {gain/total:>6.1%} {bar}")
        return "\n".join(lines)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("SnapshotScorer is not fitted. Call .train() first.")

    @staticmethod
    def _validate_input(X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} features, got {X.shape[1]}")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(X) < 10:
            raise ValueError("Need at least 10 samples to train")


def _log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
