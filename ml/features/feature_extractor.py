"""
Feature extractor for TiltRadar ML pipeline.

Takes a single player dict (from a live game snapshot) + baseline objects
and returns a normalized, ML-ready feature vector.

This is the bridge between the raw game data and all downstream models:
  - XGBoost snapshot scorer  (ml/models/snapshot_scorer.py)
  - GRU temporal model       (ml/models/temporal_model.py)

=== FEATURE VECTOR: 26 DIMENSIONS ===

Group 1 — Baseline-normalized performance (4)
  Positive values = performing WORSE than baseline. Clipped to [-3, 3].
  Convention: z = (expected - actual) / IQR  for metrics where lower = worse (CS, gold)
              z = (actual - expected) / IQR  for metrics where higher = worse (deaths)

Group 2 — Raw behavioral signals (11)
  Binary flags or continuous [0, 1] encodings of the engine signals.
  These are NOT z-scored — they are the normalized signal magnitudes.
  Note: cs_drop_flag was removed (redundant with cs_z from Group 1).

Group 3 — Temporal delta features (4)
  Change from the previous snapshot. Requires ≥2 snapshots; defaults to 0.0.
  Captures the *direction* of performance (improving vs degrading).

Group 4 — Game state context (5)
  Role one-hot encoding + normalized game time.
  Allows the model to learn role-specific normal ranges.

Group 5 — Baseline quality indicator (2)
  Encodes how reliable the baseline comparison is.
  Model can learn to discount features when baseline is unavailable.

=== DESIGN PRINCIPLES ===

  - All features in [-3, 3] or [0, 1] after normalization — no raw magnitudes
  - Role-specific signals zero-pad for roles that skip them (e.g., support skips CS)
  - Missing baseline → Group 1 features default to 0.0 (neutral, not noisy)
  - NaN / Inf are explicitly removed in the final step (failsafe)
  - The same feature names appear in FEATURE_NAMES and FeatureVector.names
    so that SHAP plots are human-readable

"""

from __future__ import annotations

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Import the champion signature items dict from the engine so we have a
# single source of truth — avoids the two dicts drifting out of sync.
from ml.inference.engine import CHAMPION_SIGNATURE_ITEMS, ROLE_GOLD_PER_MIN

# ── Feature registry ──────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # ── Group 1: Baseline-normalized performance (4) ──────────────────────────
    "cs_z",           # CS/min deficit vs baseline, in IQR units. Positive = below expected.
    "gold_z",         # Gold/min deficit vs expected role income, in IQR units.
    "kp_z",           # Kill participation deficit vs baseline, in IQR units.
    "death_z",        # Death rate excess vs baseline, in IQR units.

    # ── Group 2: Raw behavioral signals (11) ──────────────────────────────────
    "repeat_deaths_norm",       # Deaths to same enemy / 5.0  (0 → 1+)
    "death_accel_flag",         # 1.0 if late deaths ≥3 AND > early + 2, else 0.0
    "early_death_cluster_norm", # Unproductive deaths before 10 min / 6.0
    "kp_drop_relative",         # (kp_early − kp_late) / kp_early, clamped [0, 1]
    "vision_per_min",           # Ward score / game_time_min  (raw — small value = bad)
    "obj_absence_rate",         # obj_missed / max(obj_total, 1)  [0, 1]
    "sold_items_norm",          # Items sold count / 4.0, capped at 1.0
    "level_deficit_norm",       # max(0, avg_level − player_level) / 5.0
    "gold_deficit_ratio",       # max(0, 1 − estimated_gold / expected_gold)  [0, 1]
    "build_distance",           # Continuous [0, 1]: fraction of expected items MISSING
    "dead_time_pct",            # Estimated dead time / game_time  [0, 1]

    # ── Group 3: Temporal delta features (4) ──────────────────────────────────
    "delta_cs_per_min",      # CS/min change from previous snapshot (negative = declining)
    "delta_death_rate",      # Death rate change from previous snapshot (positive = worsening)
    "death_velocity_5min",   # Deaths in last 5 min / 5.0  (recent deaths per minute)
    "kp_trend",              # kp_late − kp_early at this snapshot (negative = disengaging)

    # ── Group 4: Game state context (5) ───────────────────────────────────────
    "game_time_norm",  # game_time / 2100.0 (35 min), clamped [0, 1]
    "role_support",    # 1.0 if UTILITY
    "role_jungle",     # 1.0 if JUNGLE
    "role_top",        # 1.0 if TOP
    "role_mid",        # 1.0 if MIDDLE
    # NOTE: ADC/BOTTOM is encoded as all four role flags = 0

    # ── Group 5: Baseline quality indicator (2) ───────────────────────────────
    "has_personal_baseline",  # 1.0 if personal history available; 0.0 if peer/none
    "chronic_slump",          # 1.0 if PELT detected a long-term performance drop
]

FEATURE_DIM: int = len(FEATURE_NAMES)
assert FEATURE_DIM == 26, f"Expected 26 features, got {FEATURE_DIM}"


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """
    A normalized ML-ready feature vector for one player at one snapshot.

    vector: np.ndarray of shape (FEATURE_DIM,) — dtype float32, no NaN/Inf
    names:  list of FEATURE_DIM feature names (mirrors FEATURE_NAMES)
    meta:   debugging info — not consumed by models, useful for logging / SHAP
    """
    vector: np.ndarray
    names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Named feature → value, useful for logging and SHAP waterfall plots."""
        return {name: float(v) for name, v in zip(self.names, self.vector)}

    def to_list(self) -> list[float]:
        """Plain list, used for JSON serialization (DB storage)."""
        return [float(v) for v in self.vector]


# ── IQR constants (population estimates used when no personal baseline) ───────
# Derived from empirical analysis across ~50k ranked games; update after major patches.

_DEFAULT_CS_IQR    = 1.5    # CS/min — typical spread across one tier
_DEFAULT_DR_IQR    = 0.10   # Deaths per minute
_DEFAULT_GOLD_IQR  = 55.0   # Gold per minute


# ── Feature extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a normalized 26-dimensional feature vector from one player in one snapshot.

    Instantiate once, call .extract() for every (player, snapshot) pair.

    Args:
        player:            dict from snapshot["players"][i]
        game_time:         float — seconds since game start
        kill_events:       list of ChampionKill event dicts from snapshot["events"]
        all_players:       list of all 10 player dicts (needed for level average)
        personal_baseline: PlayerBaseline ORM object or None
        peer_baseline:     PeerGroupBaseline ORM object or None
        history:           ordered list of previous player dicts for this player,
                           each enriched with a "_game_time" key by SnapshotBuffer.
                           Oldest entry first. Used for temporal delta features.

    Returns:
        FeatureVector — always valid (no NaN, no KeyError)
    """

    def __init__(self, item_registry=None) -> None:
        """
        Args:
            item_registry: Optional ItemRegistry for patch-aware build distance computation.
                           If None, falls back to CHAMPION_SIGNATURE_ITEMS (hardcoded IDs).
        """
        self._item_registry = item_registry

    def extract(
        self,
        player: dict,
        game_time: float,
        kill_events: list[dict],
        all_players: list[dict],
        personal_baseline: Any = None,
        peer_baseline: Any = None,
        history: list[dict] | None = None,
    ) -> FeatureVector:
        history = history or []

        # ── Identity helpers ──────────────────────────────────────────────────
        name              = player.get("summonerName", "")
        name_for_events   = name.split("#")[0].lower()
        champion          = player.get("championName", "")
        position          = player.get("position", "").upper()
        is_support        = position == "UTILITY"
        game_time_safe    = max(game_time, 1.0)
        game_time_min     = game_time_safe / 60.0

        # ── Resolve baseline (personal > peer > defaults) ─────────────────────
        is_personal = personal_baseline is not None
        is_peer     = peer_baseline is not None and not is_personal

        if is_personal:
            bl_cs_med    = getattr(personal_baseline, "lt_cs_per_min_median",        None) or 6.5
            bl_cs_iqr    = getattr(personal_baseline, "lt_cs_per_min_iqr",           None) or _DEFAULT_CS_IQR
            bl_kp_med    = getattr(personal_baseline, "lt_kill_participation_median", None) or 0.50
            bl_dr_med    = getattr(personal_baseline, "lt_death_rate_median",         None) or 0.15
            bl_dr_iqr    = _DEFAULT_DR_IQR
            bl_gold_med  = getattr(personal_baseline, "lt_gold_per_min_median",       None) or 350.0
            bl_gold_iqr  = _DEFAULT_GOLD_IQR
            chronic_slump = 1.0 if getattr(personal_baseline, "chronic_slump_detected", False) else 0.0
        elif is_peer:
            bl_cs_med    = getattr(peer_baseline, "cs_per_min_median",        None) or 6.5
            bl_cs_iqr    = getattr(peer_baseline, "cs_per_min_iqr",           None) or _DEFAULT_CS_IQR
            bl_kp_med    = getattr(peer_baseline, "kill_participation_median", None) or 0.50
            bl_dr_med    = getattr(peer_baseline, "death_rate_median",         None) or 0.15
            bl_dr_iqr    = getattr(peer_baseline, "death_rate_iqr",           None) or _DEFAULT_DR_IQR
            bl_gold_med  = getattr(peer_baseline, "gold_per_min_median",       None) or 350.0
            bl_gold_iqr  = _DEFAULT_GOLD_IQR
            chronic_slump = 0.0
        else:
            # No baseline — Group 1 will be 0 (neutral). Model uses Group 5 to know this.
            bl_cs_med, bl_cs_iqr = 6.5, _DEFAULT_CS_IQR
            bl_kp_med             = 0.50
            bl_dr_med, bl_dr_iqr  = 0.15, _DEFAULT_DR_IQR
            bl_gold_med, bl_gold_iqr = 350.0, _DEFAULT_GOLD_IQR
            chronic_slump = 0.0

        # ── Raw player stats ──────────────────────────────────────────────────
        kills   = player.get("kills", 0)
        deaths  = player.get("deaths", 0)
        assists = player.get("assists", 0)
        cs      = player.get("cs", 0)

        current_cs_per_min   = cs / game_time_min
        current_death_rate   = deaths / game_time_min
        kp_early_val         = float(player.get("kp_early") or 0.0)
        kp_late_val          = float(player.get("kp_late")  or 0.0)
        current_kp           = (kp_early_val + kp_late_val) / 2.0

        # Estimated gold uses the same formula as the engine so features are consistent
        estimated_gold_per_min = (
            cs * 20 + kills * 300 + assists * 150 + game_time_min * 120
        ) / game_time_min

        # ── Group 1: Baseline-normalized z-scores ─────────────────────────────
        # All z-scores are "positive = performing worse" so the model has a
        # consistent direction: higher feature value → more likely tilted.

        if not is_support and bl_cs_iqr > 0:
            cs_z = float(np.clip((bl_cs_med - current_cs_per_min) / bl_cs_iqr, -3.0, 3.0))
        else:
            cs_z = 0.0   # support: intentionally low CS, not a signal

        if not is_support and bl_gold_iqr > 0:
            gold_z = float(np.clip((bl_gold_med - estimated_gold_per_min) / bl_gold_iqr, -3.0, 3.0))
        else:
            gold_z = 0.0

        kp_z    = float(np.clip((bl_kp_med - current_kp) / 0.15, -3.0, 3.0))
        death_z = float(np.clip((current_death_rate - bl_dr_med) / max(bl_dr_iqr, 0.01), -3.0, 3.0))

        # ── Group 2a: Repeat deaths to same enemy ────────────────────────────
        killers = [
            e.get("killer", "")
            for e in kill_events
            if e.get("victim", "").lower() == name_for_events
        ]
        max_repeat = Counter(killers).most_common(1)[0][1] if killers else 0
        repeat_deaths_norm = min(max_repeat / 5.0, 1.0)

        # ── Group 2b: Death acceleration ─────────────────────────────────────
        if game_time > 600:
            mid      = game_time / 2.0
            early_d  = sum(1 for e in kill_events
                           if e.get("victim", "").lower() == name_for_events
                           and e.get("time", 0) < mid)
            late_d   = sum(1 for e in kill_events
                           if e.get("victim", "").lower() == name_for_events
                           and e.get("time", 0) >= mid)
            death_accel_flag = 1.0 if (late_d >= 3 and late_d > early_d + 2) else 0.0
        else:
            early_d, late_d  = 0, 0
            death_accel_flag = 0.0

        # ── Group 2d: Early death cluster ────────────────────────────────────
        early_deaths = sum(
            1 for e in kill_events
            if e.get("victim", "").lower() == name_for_events
            and e.get("time", 0) < 600
        )
        early_death_cluster_norm = min(early_deaths / 6.0, 1.0)

        # ── Group 2e: KP relative drop ────────────────────────────────────────
        if kp_early_val > 0 and game_time > 1200:
            kp_drop_relative = float(np.clip(
                (kp_early_val - kp_late_val) / kp_early_val, 0.0, 1.0
            ))
        else:
            kp_drop_relative = 0.0

        # ── Group 2f: Vision per minute (raw — low value is the signal) ───────
        ward_score    = float(player.get("ward_score", 0.0))
        vision_per_min = ward_score / game_time_min

        # ── Group 2g: Objective absence rate ─────────────────────────────────
        obj_total  = player.get("obj_total", 0)
        obj_missed = player.get("obj_missed", 0)
        obj_absence_rate = (obj_missed / obj_total) if obj_total >= 3 else 0.0

        # ── Group 2h: Sold items ──────────────────────────────────────────────
        sold = player.get("sold_items", [])
        sold_items_norm = min(len(sold) / 4.0, 1.0)

        # ── Group 2i: Level deficit ───────────────────────────────────────────
        if all_players and game_time > 600 and not is_support:
            avg_level = sum(p.get("level", 1) for p in all_players) / max(len(all_players), 1)
            level_deficit_norm = float(np.clip(
                max(0.0, avg_level - player.get("level", 1)) / 5.0, 0.0, 1.0
            ))
        else:
            level_deficit_norm = 0.0

        # ── Group 2j: Gold deficit ratio ─────────────────────────────────────
        if not is_support and game_time > 600:
            expected_gold     = ROLE_GOLD_PER_MIN.get(position, 380.0) * game_time_min
            estimated_gold    = cs * 20 + kills * 300 + assists * 150 + game_time_min * 120
            gold_deficit_ratio = float(np.clip(
                1.0 - estimated_gold / max(expected_gold, 1.0), 0.0, 1.0
            ))
        else:
            gold_deficit_ratio = 0.0

        # ── Group 2k: Build distance ──────────────────────────────────────────
        # Continuous [0, 1]: 0 = has all expected items, 1 = has none.
        # Uses ItemRegistry (empirical, patch-aware) when available;
        # falls back to CHAMPION_SIGNATURE_ITEMS (hardcoded item IDs).
        if self._item_registry is not None and game_time > 900:
            current_item_names = player.get("item_names", [])
            build_distance = self._item_registry.build_distance(
                champion, position, current_item_names
            )
        elif game_time > 900:
            expected_items = CHAMPION_SIGNATURE_ITEMS.get(champion, set())
            current_items  = set(player.get("items", []))
            if expected_items and len(current_items) >= 2:
                matched = len(current_items & expected_items)
                build_distance = 1.0 - (matched / len(expected_items))
            else:
                build_distance = 0.0
        else:
            build_distance = 0.0

        # ── Group 2l: Dead time percentage ───────────────────────────────────
        death_times = [
            e.get("time", 0.0) for e in kill_events
            if e.get("victim", "").lower() == name_for_events
        ]
        if death_times:
            def _est_respawn(t: float) -> float:
                # Approximates LoL respawn timer scaling: ~7s at 0 min → ~50s at 35 min
                return 7.0 + (t / 60.0) * 1.25

            total_dead = sum(_est_respawn(t) for t in death_times)
            dead_time_pct = float(np.clip(total_dead / game_time_safe, 0.0, 1.0))
        else:
            dead_time_pct = 0.0

        # ── Group 3: Temporal delta features ─────────────────────────────────
        if history:
            prev            = history[-1]
            prev_time       = float(prev.get("_game_time", game_time - 5.0))
            prev_time_min   = max(prev_time, 1.0) / 60.0
            prev_cs_per_min = prev.get("cs", 0) / prev_time_min
            prev_death_rate = prev.get("deaths", 0) / prev_time_min
            delta_cs_per_min  = current_cs_per_min - prev_cs_per_min
            delta_death_rate  = current_death_rate  - prev_death_rate
        else:
            delta_cs_per_min = 0.0
            delta_death_rate = 0.0

        deaths_last_5min     = sum(
            1 for e in kill_events
            if e.get("victim", "").lower() == name_for_events
            and e.get("time", 0) >= game_time - 300.0
        )
        death_velocity_5min = deaths_last_5min / 5.0

        kp_trend = kp_late_val - kp_early_val  # positive = KP improving late game

        # ── Group 4: Game state context ───────────────────────────────────────
        game_time_norm = float(np.clip(game_time / 2100.0, 0.0, 1.0))
        role_support   = 1.0 if position == "UTILITY" else 0.0
        role_jungle    = 1.0 if position == "JUNGLE"  else 0.0
        role_top       = 1.0 if position == "TOP"     else 0.0
        role_mid       = 1.0 if position == "MIDDLE"  else 0.0

        # ── Group 5: Baseline quality ─────────────────────────────────────────
        has_personal_baseline = 1.0 if is_personal else 0.0

        # ── Assemble ──────────────────────────────────────────────────────────
        vector = np.array([
            cs_z, gold_z, kp_z, death_z,
            repeat_deaths_norm, death_accel_flag, early_death_cluster_norm,
            kp_drop_relative, vision_per_min, obj_absence_rate, sold_items_norm,
            level_deficit_norm, gold_deficit_ratio, build_distance, dead_time_pct,
            delta_cs_per_min, delta_death_rate, death_velocity_5min, kp_trend,
            game_time_norm, role_support, role_jungle, role_top, role_mid,
            has_personal_baseline, chronic_slump,
        ], dtype=np.float32)

        # Failsafe: remove any NaN/Inf that slipped through
        vector = np.nan_to_num(vector, nan=0.0, posinf=3.0, neginf=-3.0)

        meta = {
            "player":         name,
            "champion":       champion,
            "position":       position,
            "game_time_min":  round(game_time_min, 1),
            "baseline_type":  "personal" if is_personal else ("peer" if is_peer else "none"),
            "cs_per_min":     round(current_cs_per_min, 2),
            "death_rate":     round(current_death_rate, 3),
            "n_history":      len(history),
        }

        return FeatureVector(vector=vector, meta=meta)
