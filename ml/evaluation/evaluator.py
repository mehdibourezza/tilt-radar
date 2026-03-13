"""
Model Evaluation Framework.

Computes a comprehensive set of metrics for binary tilt classifiers:

  1. AUC-ROC          — does the score rank tilted players above non-tilted ones?
  2. Brier score       — calibration quality (MSE between predicted probability and label)
  3. Precision / Recall / F1 at operating threshold
  4. Calibration error (ECE) — are scores meaningful as probabilities?
  5. Time-to-detection — at what game minute did the model first correctly fire?

=== WHY BRIER SCORE? ===

AUC-ROC measures ranking ability but ignores whether scores are calibrated
probabilities. A model that outputs 0.9 for a non-tilted player is equally
bad as one that outputs 0.6 — AUC-ROC won't catch this.

Brier score = mean( (p_pred - y_true)^2 )

A Brier score of 0.0 is perfect. 0.25 is the score of a model that always
predicts 0.5. A good classifier should achieve < 0.15.

=== WHY ECE? ===

Expected Calibration Error measures the gap between predicted confidence
and actual accuracy. If a model says "0.70" for 100 players, about 70 of them
should perform poorly. ECE is the weighted average of this gap across
probability bins.

ECE < 0.05 = well-calibrated
ECE > 0.15 = needs Platt scaling or isotonic regression
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Operating threshold for precision/recall (same as PREDICTION_THRESHOLD in ws.py)
DEFAULT_THRESHOLD = 0.55

# ECE bin count
N_CALIBRATION_BINS = 10


@dataclass
class EvaluationReport:
    """Full evaluation results for one model / one set of predictions."""
    n_samples:    int
    n_positive:   int       # ground truth positives (performed poorly)

    # Ranking metrics
    auc_roc:      float     # 0.5 = random, 1.0 = perfect

    # Calibration metrics
    brier_score:  float     # lower is better (0.0 = perfect, 0.25 = naive)
    ece:          float     # Expected Calibration Error

    # Decision threshold metrics (at DEFAULT_THRESHOLD)
    threshold:    float
    precision:    float
    recall:       float
    f1:           float
    tp:           int
    fp:           int
    fn:           int
    tn:           int

    # Calibration curve data (for plotting)
    calibration_bins: list[dict] = field(default_factory=list)
    # Each dict: {"bin_lower": float, "mean_pred": float, "frac_pos": float, "count": int}

    # ROC curve data (for plotting)
    roc_points: list[dict] = field(default_factory=list)
    # Each dict: {"fpr": float, "tpr": float, "threshold": float}


class Evaluator:
    """
    Evaluates binary tilt classifier predictions.

    Usage:
        evaluator = Evaluator()
        report = evaluator.evaluate(y_true, y_pred_proba)
        print(evaluator.format_report(report))
    """

    def evaluate(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> EvaluationReport:
        """
        Compute full evaluation metrics.

        Args:
            y_true: Binary labels (0 or 1). 1 = performed poorly (tilted).
            y_pred: Predicted probabilities in [0, 1].
            threshold: Decision threshold for precision/recall/F1.

        Returns:
            EvaluationReport
        """
        y_true = np.array(y_true, dtype=np.float32)
        y_pred = np.array(y_pred, dtype=np.float32)

        if len(y_true) == 0:
            raise ValueError("Empty arrays passed to evaluator")
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: {len(y_true)} labels, {len(y_pred)} predictions")

        n_samples  = len(y_true)
        n_positive = int(y_true.sum())

        auc   = _compute_auc_roc(y_true, y_pred)
        brier = float(np.mean((y_pred - y_true) ** 2))
        ece, bins = _compute_ece(y_true, y_pred, N_CALIBRATION_BINS)
        roc_points = _compute_roc_curve(y_true, y_pred)

        # Threshold-based metrics
        y_bin = (y_pred >= threshold).astype(np.float32)
        tp = int((y_bin * y_true).sum())
        fp = int((y_bin * (1 - y_true)).sum())
        fn = int(((1 - y_bin) * y_true).sum())
        tn = int(((1 - y_bin) * (1 - y_true)).sum())

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)

        return EvaluationReport(
            n_samples=n_samples,
            n_positive=n_positive,
            auc_roc=round(auc, 4),
            brier_score=round(brier, 4),
            ece=round(ece, 4),
            threshold=threshold,
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1=round(f1, 3),
            tp=tp, fp=fp, fn=fn, tn=tn,
            calibration_bins=bins,
            roc_points=roc_points,
        )

    @staticmethod
    def format_report(report: EvaluationReport) -> str:
        """Human-readable evaluation summary."""
        base_rate = report.n_positive / max(report.n_samples, 1)
        lines = [
            "=" * 55,
            f"EVALUATION REPORT  ({report.n_samples} samples, "
            f"{report.n_positive} positive = {base_rate:.1%} base rate)",
            "=" * 55,
            "",
            "  RANKING",
            f"    AUC-ROC:      {report.auc_roc:.4f}  "
            + ("★ excellent" if report.auc_roc > 0.80 else
               "✓ good"      if report.auc_roc > 0.70 else
               "~ marginal"  if report.auc_roc > 0.60 else "✗ poor"),
            "",
            "  CALIBRATION",
            f"    Brier score:  {report.brier_score:.4f}  "
            + ("★ excellent" if report.brier_score < 0.10 else
               "✓ good"      if report.brier_score < 0.15 else
               "~ marginal"  if report.brier_score < 0.20 else "✗ poor"),
            f"    ECE:          {report.ece:.4f}  "
            + ("★ well-calibrated" if report.ece < 0.05 else
               "✓ acceptable"       if report.ece < 0.10 else
               "⚠ needs scaling"),
            "",
            f"  THRESHOLD ({report.threshold})",
            f"    Precision:    {report.precision:.3f}",
            f"    Recall:       {report.recall:.3f}",
            f"    F1:           {report.f1:.3f}",
            f"    TP={report.tp}  FP={report.fp}  FN={report.fn}  TN={report.tn}",
            "",
            "  CALIBRATION CURVE (predicted probability vs actual fraction positive)",
            f"    {'Bin':>12}  {'Pred':>6}  {'Actual':>7}  {'Count':>6}  {'Gap':>6}",
        ]
        for b in report.calibration_bins:
            if b["count"] == 0:
                continue
            gap = abs(b["mean_pred"] - b["frac_pos"])
            lines.append(
                f"    {b['bin_lower']:.2f}–{b['bin_lower']+0.1:.2f}  "
                f"{b['mean_pred']:>6.3f}  {b['frac_pos']:>7.3f}  "
                f"{b['count']:>6}  {gap:>6.3f}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)

    @staticmethod
    def compare(report_a: EvaluationReport, report_b: EvaluationReport,
                name_a: str = "Model A", name_b: str = "Model B") -> str:
        """Side-by-side comparison of two evaluation reports."""
        lines = [
            f"{'Metric':<20} {name_a:>12} {name_b:>12} {'Δ':>10}",
            "-" * 56,
        ]
        metrics = [
            ("AUC-ROC",      report_a.auc_roc,      report_b.auc_roc,      True),
            ("Brier score",  report_a.brier_score,   report_b.brier_score,  False),
            ("ECE",          report_a.ece,            report_b.ece,          False),
            ("Precision",    report_a.precision,      report_b.precision,    True),
            ("Recall",       report_a.recall,         report_b.recall,       True),
            ("F1",           report_a.f1,             report_b.f1,           True),
        ]
        for name, va, vb, higher_is_better in metrics:
            delta = vb - va
            sign  = "+" if delta > 0 else ""
            icon  = ("↑" if delta > 0 else "↓") if abs(delta) > 0.005 else "="
            improved = (delta > 0) == higher_is_better
            marker = "✓" if improved and abs(delta) > 0.005 else ""
            lines.append(
                f"{name:<20} {va:>12.4f} {vb:>12.4f} {sign}{delta:>8.4f} {icon}{marker}"
            )
        return "\n".join(lines)


# ── Internal computation helpers ──────────────────────────────────────────────

def _compute_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute AUC-ROC using the trapezoidal rule.

    We implement this manually to avoid a hard dependency on scikit-learn
    at inference time. The result matches sklearn.metrics.roc_auc_score.
    """
    # Sort by predicted probability descending
    order  = np.argsort(y_pred)[::-1]
    y_true = y_true[order]

    n_pos  = y_true.sum()
    n_neg  = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5   # undefined — return chance level

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp, fp   = 0.0, 0.0

    for label in y_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal AUC
    auc = float(np.trapz(tpr_list, fpr_list))
    return max(0.0, min(1.0, auc))


def _compute_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, n_points: int = 50) -> list[dict]:
    """Return sampled (fpr, tpr, threshold) triples for plotting."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    points = []
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return []
    for t in thresholds:
        y_bin = (y_pred >= t).astype(np.float32)
        tp    = (y_bin * y_true).sum()
        fp    = (y_bin * (1 - y_true)).sum()
        tpr   = tp / n_pos
        fpr   = fp / n_neg
        points.append({"fpr": round(float(fpr), 4), "tpr": round(float(tpr), 4), "threshold": round(float(t), 3)})
    return points


def _compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int) -> tuple[float, list[dict]]:
    """
    Expected Calibration Error using equal-width bins.

    ECE = Σ (|bin| / n) * |acc(bin) - conf(bin)|

    Returns: (ece_value, calibration_bin_list)
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0
    bins_out  = []
    n         = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_pred >= lo) & (y_pred < hi)
        if i == n_bins - 1:
            mask |= (y_pred == hi)

        count = mask.sum()
        if count == 0:
            bins_out.append({"bin_lower": round(lo, 2), "mean_pred": 0.0, "frac_pos": 0.0, "count": 0})
            continue

        mean_pred = float(y_pred[mask].mean())
        frac_pos  = float(y_true[mask].mean())
        ece      += (count / n) * abs(mean_pred - frac_pos)
        bins_out.append({
            "bin_lower": round(lo, 2),
            "mean_pred": round(mean_pred, 4),
            "frac_pos":  round(frac_pos, 4),
            "count":     int(count),
        })

    return float(ece), bins_out
