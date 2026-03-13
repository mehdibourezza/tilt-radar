"""
GameSequenceRecorder — accumulates per-snapshot feature vectors throughout a game.

=== WHY THIS IS SEPARATE FROM SnapshotBuffer ===

SnapshotBuffer (max_len=6) stores the last 30 seconds of raw player dicts.
Its purpose is narrow: give the FeatureExtractor enough history to compute
temporal delta features (delta_cs_per_min, delta_death_rate, etc.).

GameSequenceRecorder stores the FULL game trajectory of FEATURE VECTORS —
one 26-dim vector per 5-second poll, from game start to game end. It has no
size limit and is what the GRU temporal model trains on.

At game end, call .to_json() to serialize, then save to the snapshot_sequence
column in TiltPredictionLog.

=== INTEGRATION POINT IN ws.py ===

  # At connection start (one recorder per tracked player):
  recorder = GameSequenceRecorder(player_name=name)

  # Inside the 5-second polling loop, after extracting feature vector:
  fv = extractor.extract(player, game_time, ...)
  recorder.record(fv.vector, game_time)

  # At game_over, before saving TiltPredictionLog:
  log_entry.snapshot_sequence = recorder.to_json()

=== MEMORY ESTIMATE ===

A 40-minute game at 5-second intervals = 480 snapshots.
Each snapshot = 26 float32 values = 104 bytes.
Total per player: ~50KB. 10 players simultaneously: ~500KB — negligible.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from ml.features.feature_extractor import FEATURE_DIM

logger = logging.getLogger(__name__)


class GameSequenceRecorder:
    """
    Accumulates feature vectors for one player throughout a game.

    One instance per player per game. Call .clear() at game_over
    (or create a new instance for the next game).
    """

    def __init__(self, player_name: str = "") -> None:
        self.player_name  = player_name
        self._vectors:    list[np.ndarray] = []
        self._timestamps: list[float]      = []

    # ── Recording ──────────────────────────────────────────────────────────────

    def record(self, feature_vector: np.ndarray, game_time: float) -> None:
        """
        Record one snapshot's feature vector.

        Args:
            feature_vector: shape (FEATURE_DIM,) — output of FeatureExtractor.extract()
            game_time:      seconds since game start
        """
        if feature_vector.shape != (FEATURE_DIM,):
            logger.warning(
                f"GameSequenceRecorder: expected shape ({FEATURE_DIM},), "
                f"got {feature_vector.shape}. Skipping."
            )
            return
        self._vectors.append(feature_vector.copy())
        self._timestamps.append(float(game_time))

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def get_sequence(self) -> np.ndarray:
        """
        Return full game trajectory as a matrix of shape (T, FEATURE_DIM).
        Returns empty array of shape (0, FEATURE_DIM) if nothing recorded.
        """
        if not self._vectors:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        return np.stack(self._vectors, axis=0).astype(np.float32)

    def get_timestamps(self) -> list[float]:
        """Game_time value (seconds) for each recorded snapshot."""
        return list(self._timestamps)

    @property
    def n_snapshots(self) -> int:
        return len(self._vectors)

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_json(self) -> str | None:
        """
        Serialize to JSON for the snapshot_sequence DB column.

        Format: list of {"feature_vector": [...26 floats...], "game_time": float}

        Returns None if fewer than 3 snapshots were recorded.
        """
        if self.n_snapshots < 3:
            return None

        entries = [
            {
                "feature_vector": [round(float(v), 5) for v in vec],
                "game_time":      round(ts, 1),
            }
            for vec, ts in zip(self._vectors, self._timestamps)
        ]
        return json.dumps(entries)

    @classmethod
    def from_json(cls, data: str, player_name: str = "") -> "GameSequenceRecorder":
        """Reconstruct from a serialized snapshot_sequence column value."""
        recorder = cls(player_name=player_name)
        entries  = json.loads(data) if isinstance(data, str) else data
        for entry in entries:
            fv = np.array(entry["feature_vector"], dtype=np.float32)
            recorder._vectors.append(fv)
            recorder._timestamps.append(float(entry.get("game_time", 0.0)))
        return recorder

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset for the next game."""
        self._vectors.clear()
        self._timestamps.clear()
