"""
Training entry point for TiltRadar ML models.

Usage:
  # Signal calibration study (understand which signals are predictive):
  conda run -n tilt-radar python -m ml.training.train --mode calibrate

  # Train XGBoost snapshot scorer (needs ~200+ games):
  conda run -n tilt-radar python -m ml.training.train --mode snapshot_scorer

  # Train GRU temporal model (needs ~500+ games + snapshot_sequence column):
  conda run -n tilt-radar python -m ml.training.train --mode temporal

  # Full pipeline: calibrate + train both models:
  conda run -n tilt-radar python -m ml.training.train --mode all

Output:
  experiments/signal_calibration.json
  experiments/snapshot_scorer.pkl
  experiments/temporal_model.pt
  experiments/evaluation_report.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


def main():
    parser = argparse.ArgumentParser(description="TiltRadar ML training")
    parser.add_argument(
        "--mode",
        choices=["calibrate", "snapshot_scorer", "temporal", "all"],
        default="all",
        help="Which step to run",
    )
    parser.add_argument(
        "--feature-type",
        choices=["signal_binary", "full_feature_vector"],
        default="signal_binary",
        help="Feature set for snapshot_scorer. Use full_feature_vector after migration.",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=30,
        help="Minimum TiltPredictionLog records required before training",
    )
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading TiltPredictionLog records from DB...")
    from ml.training.dataset import (
        load_prediction_logs,
        build_signal_feature_matrix,
        build_full_feature_matrix,
        temporal_split,
    )

    try:
        records = load_prediction_logs(min_records=args.min_records)
    except Exception as e:
        logger.error(f"Failed to load records: {e}")
        sys.exit(1)

    if not records:
        logger.error("No records found. Play games to collect training data.")
        sys.exit(1)

    n_poor = sum(1 for r in records if r.get("performed_poorly", False))
    logger.info(
        f"Dataset: {len(records)} records, "
        f"{n_poor} positive ({n_poor/max(len(records),1):.1%} base rate)"
    )

    # ── Signal calibration ────────────────────────────────────────────────────
    if args.mode in ("calibrate", "all"):
        run_calibration(records)

    # ── XGBoost snapshot scorer ───────────────────────────────────────────────
    if args.mode in ("snapshot_scorer", "all"):
        if len(records) < 50:
            logger.warning(
                f"Only {len(records)} records — XGBoost may overfit. "
                f"Collect 200+ for reliable results."
            )

        # Build feature matrix
        if args.feature_type == "full_feature_vector":
            try:
                dataset = build_full_feature_matrix(records)
            except ValueError as e:
                logger.warning(f"{e} — falling back to signal_binary features.")
                dataset = build_signal_feature_matrix(records)
        else:
            dataset = build_signal_feature_matrix(records)

        run_snapshot_scorer(dataset)

    # ── GRU temporal model ────────────────────────────────────────────────────
    if args.mode in ("temporal", "all"):
        run_temporal_model(records)


def run_calibration(records: list[dict]) -> None:
    """Signal calibration study."""
    from ml.evaluation.signal_calibration import SignalCalibration

    logger.info("=" * 60)
    logger.info("RUNNING SIGNAL CALIBRATION STUDY")
    logger.info("=" * 60)

    calibrator = SignalCalibration()
    report     = calibrator.compute(records)

    print("\n" + calibrator.format_report(report))

    # Save to disk
    out_path = EXPERIMENTS_DIR / "signal_calibration.json"
    out_path.write_text(calibrator.to_json(report))
    logger.info(f"Calibration report saved to {out_path}")

    # Print recommended weights dict
    logger.info("\nRecommended engine.py WEIGHTS (based on empirical lift):")
    for k, w in sorted(report.weight_dict.items(), key=lambda x: -x[1]):
        logger.info(f"  {k:<35}: {w:.3f}")


def run_snapshot_scorer(dataset) -> None:
    """Train and evaluate the XGBoost snapshot scorer."""
    from ml.training.dataset import temporal_split
    from ml.models.snapshot_scorer import SnapshotScorer
    from ml.evaluation.evaluator import Evaluator

    logger.info("=" * 60)
    logger.info(f"TRAINING SNAPSHOT SCORER ({dataset.feature_type})")
    logger.info("=" * 60)

    try:
        split = temporal_split(dataset)
    except ValueError as e:
        logger.error(f"Split failed: {e}")
        return

    logger.info(split.summary() if hasattr(split, 'summary') else "Split complete")

    # Carve off the last ~14% of training data as a dedicated Platt calibration set.
    # This prevents calibration from reusing the val set (which influenced n_estimators
    # via early stopping) — that would make calibrated probabilities overconfident.
    n_train_total = split.train.n_samples
    n_cal = max(1, int(n_train_total * 0.143))  # ~10% of all data
    n_train_actual = n_train_total - n_cal

    X_train_actual = split.train.X[:n_train_actual]
    y_train_actual = split.train.y[:n_train_actual]
    X_cal = split.train.X[n_train_actual:]
    y_cal = split.train.y[n_train_actual:]

    logger.info(
        f"Data split: train={n_train_actual}, cal={n_cal}, "
        f"val={split.val.n_samples}, test={split.test.n_samples}"
    )

    try:
        scorer = SnapshotScorer()
    except ImportError as e:
        logger.error(f"{e}")
        return

    # Train
    metrics = scorer.train(
        X_train_actual, y_train_actual,
        split.val.X,    split.val.y,
        X_cal,          y_cal,
    )
    logger.info(f"Training metrics: {metrics}")

    # Evaluate on test set
    evaluator = Evaluator()
    y_pred    = scorer.predict_proba(split.test.X)
    report    = evaluator.evaluate(split.test.y, y_pred)

    print("\n" + evaluator.format_report(report))

    # Save model
    model_path = EXPERIMENTS_DIR / "snapshot_scorer.pkl"
    scorer.save(model_path)

    # Save timestamped version for rollback
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = EXPERIMENTS_DIR / f"snapshot_scorer_{ts}.pkl"
    shutil.copy(model_path, versioned_path)
    logger.info(f"Versioned model saved to {versioned_path}")

    # Save report
    report_path = EXPERIMENTS_DIR / "snapshot_scorer_eval.txt"
    report_path.write_text(evaluator.format_report(report))
    logger.info(f"Evaluation saved to {report_path}")

    # Feature importances
    logger.info("\n" + scorer.format_importances())


def run_temporal_model(records: list[dict]) -> None:
    """Train and evaluate the GRU temporal model."""
    from ml.training.dataset import build_sequence_dataset

    logger.info("=" * 60)
    logger.info("TRAINING TEMPORAL GRU MODEL")
    logger.info("=" * 60)

    try:
        sequences, label_sequences = build_sequence_dataset(records)
    except NotImplementedError as e:
        logger.warning(str(e))
        return
    except Exception as e:
        logger.error(f"Failed to build sequences: {e}")
        return

    if len(sequences) < 20:
        logger.warning(
            f"Only {len(sequences)} game sequences. "
            f"GRU training is not reliable below 200. Skipping."
        )
        return

    from ml.models.temporal_model import TemporalTiltModel
    from ml.evaluation.evaluator import Evaluator

    # Split
    n        = len(sequences)
    n_test   = max(1, int(n * 0.15))
    n_val    = max(1, int(n * 0.15))
    n_train  = n - n_val - n_test

    train_seq = sequences[:n_train]
    val_seq   = sequences[n_train:n_train + n_val]
    test_seq  = sequences[n_train + n_val:]
    train_lbl = label_sequences[:n_train]
    val_lbl   = label_sequences[n_train:n_train + n_val]
    test_lbl  = label_sequences[n_train + n_val:]

    model   = TemporalTiltModel()
    history = model.train(train_seq, train_lbl, val_seq, val_lbl)

    logger.info(
        f"Training complete. "
        f"Final train_loss={history['train_loss'][-1]:.4f}"
        + (f", val_loss={history['val_loss'][-1]:.4f}" if history.get("val_loss") else "")
    )

    # Evaluate on test set: last-step prediction vs last-step label
    # (last label is 1.0 for tilt games at/after peak, 0.0 for clean games)
    y_pred = np.array([
        model.predict_proba_sequence(seq)[-1]
        for seq in test_seq
    ])
    y_true = np.array([lbl[-1] for lbl in test_lbl], dtype=np.float32)

    evaluator = Evaluator()
    report    = evaluator.evaluate(y_true, y_pred)
    print("\n" + evaluator.format_report(report))

    # Save
    model_path = EXPERIMENTS_DIR / "temporal_model.pt"
    model.save(model_path)

    # Save timestamped version for rollback
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = EXPERIMENTS_DIR / f"temporal_model_{ts}.pt"
    shutil.copy(model_path, versioned_path)
    logger.info(f"Versioned model saved to {versioned_path}")

    report_path = EXPERIMENTS_DIR / "temporal_model_eval.txt"
    report_path.write_text(evaluator.format_report(report))
    logger.info(f"GRU model saved to {model_path}")


if __name__ == "__main__":
    main()
