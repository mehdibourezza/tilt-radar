"""
Dataset loading and preparation for TiltRadar ML training.

Two data sources:

1. TiltPredictionLog (DB table) — labeled prediction records.
   One record = one player's peak tilt score + final outcome per game.
   Features: the 12 binary signal indicators at peak + meta features.
   Label:    performed_poorly (bool)
   Used for: SnapshotScorer training + SignalCalibration study

2. feature_vector_at_peak (new JSON column in TiltPredictionLog) — populated
   after integrating FeatureExtractor into the WebSocket handler.
   Features: full 27-dimensional normalized feature vector.
   Label:    performed_poorly (bool)
   Used for: SnapshotScorer training (better than signal binary features)

=== TIME-BASED TRAIN/VAL/TEST SPLIT ===

We MUST NOT use random split for time-series data. If we shuffle and split:
  - Train set contains games from the future
  - Model appears accurate but fails in deployment (it "saw the future")

Correct approach: sort records by recorded_at, then split chronologically.

  [oldest ←───────────────────────────────────→ newest]
  [──────── 70% train ──────────][── 15% val ──][── 15% test ──]

The model never sees val/test timestamps during training. This simulates
real deployment: train on past games, evaluate on future games.

=== SIGNAL BINARY FEATURES (12 dims, available immediately) ===

When feature_vector_at_peak is not available (pre-migration records),
we fall back to encoding the active signals as binary features:

  X[i] = [has_repeat_deaths, has_death_accel, has_cs_drop, has_early_deaths,
           has_kp_drop, has_vision_low, has_obj_absence, has_item_sell,
           has_level_deficit, has_gold_deficit, has_wrong_build, has_respawn,
           n_signals_active_norm,  # 0–1
           peak_tilt_score]        # engine output as meta-feature

  Total: 14 features (sufficient for signal calibration and a basic XGBoost)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# ── Signal pattern matching (same as signal_calibration.py) ──────────────────
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

SIGNAL_KEYS = list(SIGNAL_PATTERNS.keys())
SIGNAL_FEATURE_DIM = len(SIGNAL_KEYS) + 2   # 12 binary + n_signals_norm + peak_score


@dataclass
class TiltDataset:
    """
    A prepared dataset for training or evaluation.

    X:       feature matrix, shape (n_samples, n_features)
    y:       binary labels, shape (n_samples,)  — 1 = performed_poorly
    meta:    list of dicts with record metadata (player_name, recorded_at, etc.)
    feature_type: "signal_binary" or "full_feature_vector"
    """
    X: np.ndarray
    y: np.ndarray
    meta: list[dict]
    feature_type: str

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def base_rate(self) -> float:
        return float(self.y.mean())


@dataclass
class DataSplit:
    """Train / val / test split, time-ordered."""
    train: TiltDataset
    val:   TiltDataset
    test:  TiltDataset

    def summary(self) -> str:
        return (
            f"DataSplit: train={self.train.n_samples} "
            f"(+{self.train.y.sum():.0f}/{self.train.n_samples}, "
            f"{self.train.base_rate:.1%}), "
            f"val={self.val.n_samples} ({self.val.base_rate:.1%}), "
            f"test={self.test.n_samples} ({self.test.base_rate:.1%})"
        )


# ── Loading from DB ───────────────────────────────────────────────────────────

async def load_prediction_logs_async(min_records: int = 30) -> list[dict]:
    """
    Load all TiltPredictionLog records from the database.

    Returns a list of dicts (serialized ORM objects), sorted oldest → newest
    by recorded_at.

    Requires the DB to be running and tables to exist.
    """
    from data.db.session import get_session
    from data.db.models import TiltPredictionLog
    from sqlalchemy import select, asc

    records = []
    async for session in get_session():
        result = await session.execute(
            select(TiltPredictionLog).order_by(asc(TiltPredictionLog.recorded_at))
        )
        rows = result.scalars().all()
        for row in rows:
            records.append({
                "id":                   row.id,
                "player_name":          row.player_name,
                "player_type":          row.player_type,
                "champion_name":        row.champion_name,
                "peak_tilt_score":      row.peak_tilt_score,
                "peak_tilt_type":       row.peak_tilt_type,
                "peak_signals":         row.peak_signals or [],
                "final_cs_per_min":     row.final_cs_per_min,
                "final_death_rate":     row.final_death_rate,
                "final_kda":            row.final_kda,
                "game_duration_min":    row.game_duration_min,
                "performed_poorly":     row.performed_poorly,
                "verdict":              row.verdict,
                "recorded_at":          row.recorded_at.isoformat() if row.recorded_at else None,
                # New columns (None if not yet populated)
                "role":                 getattr(row, "role", None),
                "game_time_at_peak":    getattr(row, "game_time_at_peak", None),
                "feature_vector_at_peak": getattr(row, "feature_vector_at_peak", None),
                "n_signals_active":     getattr(row, "n_signals_active", None),
            })

    if len(records) < min_records:
        logger.warning(
            f"Only {len(records)} records in TiltPredictionLog. "
            f"Need at least {min_records} for meaningful training. "
            f"Play more games!"
        )

    logger.info(f"Loaded {len(records)} prediction log records from DB")
    return records


def load_prediction_logs(min_records: int = 30) -> list[dict]:
    """Synchronous wrapper for load_prediction_logs_async."""
    return asyncio.run(load_prediction_logs_async(min_records))


# ── Feature building ──────────────────────────────────────────────────────────

def build_signal_feature_matrix(records: list[dict]) -> TiltDataset:
    """
    Build feature matrix from the 12 signal indicators + meta features.

    Available even before the FeatureExtractor migration — works on all
    historical records as long as peak_signals is populated.

    Feature dimensions:
      [0:12]  — binary signal indicators (1.0 if signal was active at peak)
      [12]    — number of signals active (normalized to [0,1])
      [13]    — peak_tilt_score (engine's output, as meta-feature)
    """
    n = len(records)
    X = np.zeros((n, SIGNAL_FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    meta = []

    for i, record in enumerate(records):
        signals = _parse_signals(record)

        for j, key in enumerate(SIGNAL_KEYS):
            pattern = SIGNAL_PATTERNS[key]
            if any(pattern in s for s in signals):
                X[i, j] = 1.0

        X[i, 12] = min(len(signals) / 12.0, 1.0)
        X[i, 13] = float(record.get("peak_tilt_score", 0.0))

        y[i] = 1.0 if record.get("performed_poorly", False) else 0.0
        meta.append({
            "player_name":   record.get("player_name", ""),
            "player_type":   record.get("player_type", ""),
            "champion_name": record.get("champion_name", ""),
            "verdict":       record.get("verdict", ""),
            "recorded_at":   record.get("recorded_at"),
        })

    return TiltDataset(X=X, y=y, meta=meta, feature_type="signal_binary")


def build_full_feature_matrix(records: list[dict]) -> TiltDataset:
    """
    Build feature matrix from the full 27-dimensional feature_vector_at_peak.

    Only available for records collected after the FeatureExtractor migration.
    Filters out records where feature_vector_at_peak is None.
    """
    from ml.features.feature_extractor import FEATURE_DIM

    eligible = [r for r in records if r.get("feature_vector_at_peak") is not None]
    if not eligible:
        raise ValueError(
            "No records have feature_vector_at_peak. "
            "Play games after the FeatureExtractor migration and run again."
        )

    n = len(eligible)
    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    meta = []

    for i, record in enumerate(eligible):
        fv = record["feature_vector_at_peak"]
        if isinstance(fv, str):
            fv = json.loads(fv)
        X[i] = np.array(fv, dtype=np.float32)
        y[i] = 1.0 if record.get("performed_poorly", False) else 0.0
        meta.append({
            "player_name":   record.get("player_name", ""),
            "player_type":   record.get("player_type", ""),
            "champion_name": record.get("champion_name", ""),
            "verdict":       record.get("verdict", ""),
            "recorded_at":   record.get("recorded_at"),
        })

    logger.info(f"Built full feature matrix: {n} records with feature_vector_at_peak")
    return TiltDataset(X=X, y=y, meta=meta, feature_type="full_feature_vector")


# ── Train / val / test split ──────────────────────────────────────────────────

def temporal_split(
    dataset: TiltDataset,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
) -> DataSplit:
    """
    Time-based chronological split.

    Records are assumed to be in temporal order (load_prediction_logs sorts by
    recorded_at). We split by index (oldest → newest) rather than shuffling.

    Args:
        dataset:    TiltDataset with records in temporal order
        val_ratio:  fraction of data for validation (default 15%)
        test_ratio: fraction of data for test (default 15%)

    Returns:
        DataSplit with non-overlapping, time-ordered train/val/test subsets
    """
    n = dataset.n_samples
    if n < 20:
        raise ValueError(f"Too few samples ({n}) for a meaningful split. Need at least 20.")

    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    if n_train < 10:
        raise ValueError(
            f"Training set too small ({n_train} samples) after split. "
            f"Collect more games before training."
        )

    def _slice(start: int, end: int) -> TiltDataset:
        return TiltDataset(
            X=dataset.X[start:end],
            y=dataset.y[start:end],
            meta=dataset.meta[start:end],
            feature_type=dataset.feature_type,
        )

    split = DataSplit(
        train=_slice(0,              n_train),
        val=  _slice(n_train,        n_train + n_val),
        test= _slice(n_train + n_val, n),
    )

    logger.info(
        f"Temporal split: train={split.train.n_samples} "
        f"({split.train.base_rate:.1%} positive), "
        f"val={split.val.n_samples} ({split.val.base_rate:.1%}), "
        f"test={split.test.n_samples} ({split.test.base_rate:.1%})"
    )
    return split


# ── Sequence building (for GRU temporal model) ────────────────────────────────

def build_sequence_dataset(
    records: list[dict],
    group_by: str = "game_session",
    ramp_start_fraction: float = 0.60,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build per-game sequences of feature vectors for GRU training, with
    per-timestep ramp labels derived from game_time_at_peak (Option B labeling).

    NOTE: This requires the snapshot_sequence column (Phase D) which is not
    yet populated. This function is a placeholder that will be activated
    when that column is available.

    === OPTION B RAMP LABELS ===

    Last-state supervision (the naive approach) labels EVERY snapshot in a game
    with the final game outcome. So a snapshot at minute 1 with no evidence of
    tilt gets labeled 1.0 just because the game ended badly. This adds noise and
    makes the GRU predict tilt before any evidence exists.

    Option B uses game_time_at_peak as a timing anchor:

      t < 0.6 * peak_time  →  label = 0.0   (definitely not tilted yet)
      t in [60%, 100%] of peak_time  →  linear ramp 0.0 → 1.0
      t >= peak_time  →  label = 1.0   (peak tilt confirmed)

    For non-tilt games (performed_poorly=False): all labels = 0.0
    For tilt games with no peak_time recorded: falls back to last-state (label 1.0
    at final timestep only), which is conservative but unambiguous.

    snapshot_sequence format expected per record:
      [{"feature_vector": [f0, f1, ...], "game_time": 42.5}, ...]

    Args:
        records: TiltPredictionLog records
        group_by: reserved for future use (currently all records treated individually)
        ramp_start_fraction: when tilt starts building as a fraction of peak_time.
                             Default 0.60 is conservative but arbitrary — validate
                             empirically once enough data with timestamped tilt peaks
                             is available.

    Returns:
        (sequences, label_sequences)
          sequences[i]:       np.ndarray shape (T_i, FEATURE_DIM)
          label_sequences[i]: np.ndarray shape (T_i,) with values in [0.0, 1.0]
    """
    eligible = [r for r in records if r.get("snapshot_sequence") is not None]
    if not eligible:
        raise NotImplementedError(
            "GRU training requires snapshot_sequence data which is not yet collected. "
            "This column will be populated after the Phase D migration. "
            "Use SnapshotScorer (XGBoost) in the meantime."
        )

    from ml.features.feature_extractor import FEATURE_DIM
    sequences:       list[np.ndarray] = []
    label_sequences: list[np.ndarray] = []

    for record in eligible:
        seq_data = record["snapshot_sequence"]
        if isinstance(seq_data, str):
            seq_data = json.loads(seq_data)

        # seq_data is a list of dicts: [{feature_vector: [...], game_time: float}, ...]
        # Support both the new dict format and the old flat array format for compatibility.
        if seq_data and isinstance(seq_data[0], dict):
            feature_vecs = [s["feature_vector"] for s in seq_data]
            timestamps   = [float(s.get("game_time", i * 5)) for i, s in enumerate(seq_data)]
        else:
            # Legacy: flat list of feature vectors (no timestamps) — use 5s intervals
            feature_vecs = seq_data
            timestamps   = [i * 5.0 for i in range(len(seq_data))]

        seq = np.array(feature_vecs, dtype=np.float32)
        if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
            logger.warning(f"Skipping record {record.get('id')}: shape mismatch {seq.shape}")
            continue

        performed_poorly = bool(record.get("performed_poorly", False))
        peak_time        = record.get("game_time_at_peak")
        ramp             = _ramp_labels(timestamps, peak_time, performed_poorly, ramp_start_fraction)

        sequences.append(seq)
        label_sequences.append(ramp)

    n_tilt = sum(1 for lbl in label_sequences if lbl[-1] > 0.5)
    logger.info(
        f"Built {len(sequences)} game sequences for GRU training "
        f"({n_tilt} tilt, {len(sequences) - n_tilt} non-tilt)"
    )
    return sequences, label_sequences


def _ramp_labels(
    timestamps: list[float],
    peak_time: float | None,
    performed_poorly: bool,
    ramp_start_fraction: float = 0.60,
) -> np.ndarray:
    """
    Compute per-timestep ramp labels for one game sequence.

    For non-tilt games: all zeros.
    For tilt games with a known peak_time:
      - Before 60% of peak_time: 0.0
      - 60–100% of peak_time: linear ramp 0.0 → 1.0
      - At or after peak_time: 1.0
    For tilt games with no peak_time: conservative last-step label only.
    """
    T      = len(timestamps)
    labels = np.zeros(T, dtype=np.float32)

    if not performed_poorly:
        return labels  # all zeros — clean game

    if peak_time is None or peak_time <= 0:
        # Fallback: we know the game went badly but not when peak tilt was.
        # Put label 1.0 only at the final snapshot — conservative, unambiguous.
        labels[-1] = 1.0
        return labels

    ramp_start = peak_time * ramp_start_fraction
    ramp_width = peak_time - ramp_start  # always > 0 since peak_time > 0

    for i, t in enumerate(timestamps):
        if t >= peak_time:
            labels[i] = 1.0
        elif t >= ramp_start:
            labels[i] = (t - ramp_start) / ramp_width
        # else: 0.0 (before tilt evidence starts)

    return labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_signals(record: dict) -> list[str]:
    v = record.get("peak_signals", [])
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    return v or []
