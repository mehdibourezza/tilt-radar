"""
Signal Calibration Study.

Answers the question: "Which of our 12 rule-based signals actually predict
poor performance, and how strongly?"

This runs AFTER collecting TiltPredictionLog records (needs ~200+ games).
It produces:
  1. Per-signal precision, recall, lift, and empirical weight
  2. A calibrated weight dict that can replace the hand-tuned WEIGHTS in engine.py
  3. A signal correlation matrix (to detect redundant co-firing signals)

=== WHAT IS LIFT? ===

Lift = P(poor | signal active) / P(poor overall)

A lift of 2.0 means: when this signal fires, the player is 2× more likely to
perform poorly than a random player in our dataset.

Lift > 1.5 → signal is useful
Lift > 2.5 → signal is highly predictive
Lift < 1.0 → signal fires MORE on good players (we have the sign wrong or it's noise)

=== SIGNAL PATTERN MATCHING ===

TiltPredictionLog stores peak_signals as a list of human-readable strings like:
  ["died_to_same_enemy_3x", "death_rate_accelerating", "cs_down_35pct_vs_baseline"]

We map each of the 12 signal keys to a substring pattern to detect presence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Signal key → substring pattern for peak_signals list ─────────────────────
# A signal is considered "active" if any element in peak_signals contains this substring.
SIGNAL_PATTERNS: dict[str, str] = {
    "repeat_deaths_same_enemy": "same_enemy",
    "death_acceleration":        "accelerating",
    "cs_drop_vs_baseline":       "cs_down",
    "early_death_cluster":       "deaths_before",
    "kill_participation_drop":   "kp_dropped",
    "vision_score_low":          "vision_very_low",
    "objective_absence":         "absent_",
    "item_sell":                 "sold_",
    "level_deficit":             "level_deficit",
    "gold_deficit_per_role":     "gold_deficit",
    "wrong_build":               "wrong_build",
    "respawn_timer_trend":       "respawn",
}


@dataclass
class SignalStats:
    """Calibration statistics for one signal."""
    signal_key:    str
    n_active:      int      # number of records where this signal fired
    n_total:       int      # total records in dataset
    base_rate:     float    # P(poor) overall
    precision:     float    # P(poor | signal active) — when it fires, how often right?
    recall:        float    # P(signal active | poor) — how many real cases does it catch?
    f1:            float    # harmonic mean of precision and recall
    lift:          float    # precision / base_rate — how much better than chance?
    empirical_weight: float  # recommended weight for engine.py WEIGHTS dict


@dataclass
class CalibrationReport:
    """Full calibration study output."""
    n_records:      int
    n_poor:         int
    base_rate:      float           # P(performed_poorly) across all records
    signal_stats:   list[SignalStats]
    weight_dict:    dict[str, float]   # drop-in replacement for engine.py WEIGHTS
    correlation_matrix: list[list[float]] | None  # 12×12 signal co-occurrence correlations


class SignalCalibration:
    """
    Runs the calibration study on TiltPredictionLog records.

    Usage:
        # Load records from DB (async context needed — call outside this class)
        records = await load_prediction_logs()

        calibrator = SignalCalibration()
        report = calibrator.compute(records)
        print(calibrator.format_report(report))

        # Use learned weights in the engine:
        engine.WEIGHTS = report.weight_dict
    """

    SIGNAL_KEYS = list(SIGNAL_PATTERNS.keys())

    def compute(self, records: list[dict]) -> CalibrationReport:
        """
        Compute calibration statistics from a list of TiltPredictionLog dicts.

        Each record must have:
          - peak_signals: list[str]   — signals active at peak tilt
          - performed_poorly: bool    — the ground-truth label

        Args:
            records: list of dicts from TiltPredictionLog (can be ORM objects
                     converted to dicts or raw dicts from DB query)

        Returns:
            CalibrationReport with per-signal stats and recommended weights
        """
        if len(records) < 30:
            logger.warning(
                f"Only {len(records)} records — calibration unreliable. "
                f"Collect at least 200 games for meaningful results."
            )

        n_total = len(records)
        n_poor  = sum(1 for r in records if _get_label(r))
        base_rate = n_poor / max(n_total, 1)

        # ── Build binary signal matrix: shape (n_records, 12) ────────────────
        signal_matrix = np.zeros((n_total, len(self.SIGNAL_KEYS)), dtype=np.float32)
        labels        = np.array([_get_label(r) for r in records], dtype=np.float32)

        for i, record in enumerate(records):
            signals = _get_signals(record)
            for j, key in enumerate(self.SIGNAL_KEYS):
                pattern = SIGNAL_PATTERNS[key]
                if any(pattern in s for s in signals):
                    signal_matrix[i, j] = 1.0

        # ── Per-signal statistics ─────────────────────────────────────────────
        signal_stats = []
        for j, key in enumerate(self.SIGNAL_KEYS):
            col       = signal_matrix[:, j]
            active    = col.sum()
            if active == 0:
                # Signal never fired in this dataset — assign minimal weight
                signal_stats.append(SignalStats(
                    signal_key=key, n_active=0, n_total=n_total,
                    base_rate=base_rate, precision=base_rate, recall=0.0,
                    f1=0.0, lift=1.0, empirical_weight=0.05,
                ))
                continue

            # TP: signal fired AND performed poorly
            tp = float((col * labels).sum())
            fp = float((col * (1 - labels)).sum())
            fn = float(((1 - col) * labels).sum())

            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-9)
            lift      = precision / max(base_rate, 1e-9)

            # Empirical weight: scale lift to [0.05, 0.40] range
            # Lift 1.0 → weight 0.05 (noise level)
            # Lift 3.0 → weight 0.35 (strongly predictive)
            empirical_weight = float(np.clip((lift - 1.0) / 2.0 * 0.35 + 0.05, 0.05, 0.40))

            signal_stats.append(SignalStats(
                signal_key=key,
                n_active=int(active),
                n_total=n_total,
                base_rate=round(base_rate, 3),
                precision=round(precision, 3),
                recall=round(recall, 3),
                f1=round(f1, 3),
                lift=round(lift, 3),
                empirical_weight=round(empirical_weight, 3),
            ))

        # ── Signal correlation matrix (Pearson on binary columns) ─────────────
        corr_matrix = None
        if n_total >= 20:
            try:
                corr = np.corrcoef(signal_matrix.T)
                corr_matrix = [[round(float(v), 3) for v in row] for row in corr]
            except Exception:
                pass   # degenerate case (all-zero column) — skip silently

        # ── Weight dict (drop-in replacement for engine.WEIGHTS) ─────────────
        weight_dict = {
            s.signal_key: s.empirical_weight
            for s in signal_stats
        }

        return CalibrationReport(
            n_records=n_total,
            n_poor=n_poor,
            base_rate=round(base_rate, 3),
            signal_stats=sorted(signal_stats, key=lambda s: s.lift, reverse=True),
            weight_dict=weight_dict,
            correlation_matrix=corr_matrix,
        )

    @staticmethod
    def format_report(report: CalibrationReport) -> str:
        """Human-readable calibration report."""
        lines = [
            "=" * 70,
            f"SIGNAL CALIBRATION REPORT  ({report.n_records} records)",
            f"Base rate (% poor): {report.base_rate:.1%}  "
            f"({report.n_poor} / {report.n_records})",
            "=" * 70,
            f"{'Signal':<30} {'Active':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} "
            f"{'Lift':>6} {'Weight':>7}",
            "-" * 70,
        ]
        for s in report.signal_stats:
            bar = "▓" * int(s.lift * 5) if s.lift > 0 else ""
            lines.append(
                f"{s.signal_key:<30} {s.n_active:>6} {s.precision:>6.2f} "
                f"{s.recall:>6.2f} {s.f1:>6.2f} {s.lift:>6.2f}  "
                f"{s.empirical_weight:>5.2f} {bar}"
            )

        lines += [
            "-" * 70,
            "",
            "Recommended WEIGHTS dict for engine.py:",
            "  WEIGHTS = {",
        ]
        for key, w in sorted(report.weight_dict.items()):
            lines.append(f'    "{key}": {w},')
        lines += ["  }", "=" * 70]

        # Highlight highly correlated signal pairs (possible redundancy)
        if report.correlation_matrix:
            keys = list(SIGNAL_PATTERNS.keys())
            high_corr = []
            n = len(keys)
            for i in range(n):
                for j in range(i + 1, n):
                    c = report.correlation_matrix[i][j]
                    if c > 0.60:
                        high_corr.append((keys[i], keys[j], c))
            if high_corr:
                lines.append("\nHighly correlated signal pairs (>0.60 — possible redundancy):")
                for a, b, c in sorted(high_corr, key=lambda x: -x[2]):
                    lines.append(f"  {a} ↔ {b}  r={c:.2f}")

        return "\n".join(lines)

    @staticmethod
    def to_json(report: CalibrationReport) -> str:
        """Serialize report to JSON for storage / sharing."""
        return json.dumps({
            "n_records":    report.n_records,
            "n_poor":       report.n_poor,
            "base_rate":    report.base_rate,
            "signal_stats": [
                {
                    "signal_key": s.signal_key,
                    "n_active": s.n_active,
                    "precision": s.precision,
                    "recall":    s.recall,
                    "f1":        s.f1,
                    "lift":      s.lift,
                    "empirical_weight": s.empirical_weight,
                }
                for s in report.signal_stats
            ],
            "weight_dict": report.weight_dict,
        }, indent=2)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_label(record) -> bool:
    """Extract performed_poorly from an ORM object or dict."""
    if isinstance(record, dict):
        return bool(record.get("performed_poorly", False))
    return bool(getattr(record, "performed_poorly", False))


def _get_signals(record) -> list[str]:
    """Extract peak_signals from an ORM object or dict."""
    if isinstance(record, dict):
        v = record.get("peak_signals", [])
    else:
        v = getattr(record, "peak_signals", [])
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    return v or []
