"""
Change Point Detection for player behavioral baselines.

=== WHAT IS CHANGE POINT DETECTION? ===

A time series has a "change point" when its statistical properties suddenly shift.
For a player over 100 games, their CS/min might look like:

  Games 1-75:   [7.1, 6.9, 7.3, 7.0, 7.2, ...]  ← consistent, centered around 7.1
  Games 76-100: [4.2, 3.8, 4.1, 4.5, 3.9, ...]  ← sudden drop, new distribution

The change point is at game 76. Before it: normal play. After it: something changed.

Without CPD, if you naively average all 100 games:
  mean = 6.47  → this represents neither period accurately
  And in a live game where this player does 7.0 CS/min, your model
  would think they're ABOVE baseline when they're actually at their historical best.

With CPD, you know:
  True baseline = 7.1 (games 1-75)
  Current slump started at game 76
  In-game 7.0 CS/min = normal (not tilting now, was already in slump before game)

=== TWO ALGORITHMS WE USE ===

1. PELT (Pruned Exact Linear Time) — for offline/historical analysis
   - Finds the globally optimal set of change points
   - O(n log n) — fast enough for 100-game histories
   - From the `ruptures` library
   - Minimizes: sum of within-segment variance + penalty * num_change_points

2. BOCPD (Bayesian Online Change Point Detection) — conceptual foundation
   - For online/streaming use: detects change points as new data arrives
   - Maintains probability: P(change point at time t | data so far)
   - We implement a simplified version manually

=== OUR STRATEGY ===

For baseline computation (offline, after ingestion):
  → Use PELT on each metric's game history
  → Identify segments, use only the LATEST stable segment as the true baseline
  → Flag if that segment started recently (chronic slump indicator)

For live game scoring:
  → Compare current in-game stats to the "true baseline segment" only
  → Not to the full history (which may be contaminated)
"""

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
import logging

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logging.warning("ruptures not installed — change point detection disabled")

logger = logging.getLogger(__name__)


class ChangePoint(NamedTuple):
    game_index: int       # which game index the change occurred at
    metric: str           # which metric changed (e.g. "cs_per_min")
    before_mean: float    # mean of the metric before the change point
    after_mean: float     # mean of the metric after the change point
    magnitude: float      # how large the shift is (in standard deviations of the before segment)
    direction: str        # "drop" or "spike"


@dataclass
class BaselineResult:
    """
    Result of baseline computation for a single player.

    true_baseline_start: game index where the current stable period begins
    If this is close to the end of the history (e.g., game 80 out of 100),
    the player recently entered a new behavioral regime — could be chronic slump.
    """
    games_analyzed: int
    change_points: list[ChangePoint]
    true_baseline_start: int        # index of first game in the current stable period
    chronic_slump_detected: bool    # True if current period is significantly below historical best

    # Robust stats computed on the TRUE BASELINE segment only
    cs_per_min_median: float | None
    cs_per_min_iqr: float | None
    kill_participation_median: float | None
    death_rate_median: float | None
    gold_per_min_median: float | None
    solo_death_rate_median: float | None


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    """Return (median, IQR) — robust to outliers."""
    if len(values) == 0:
        return 0.0, 0.0
    q25, q75 = np.percentile(values, [25, 75])
    return float(np.median(values)), float(q75 - q25)


def detect_change_points(
    series: np.ndarray,
    metric_name: str,
    penalty: float = 3.0,       # higher = fewer change points detected (avoid overfitting)
    min_segment_len: int = 5,   # don't detect change points in segments shorter than 5 games
) -> list[ChangePoint]:
    """
    Run PELT on a 1D time series (one metric over N games).

    penalty (also called 'pen' in ruptures) controls sensitivity:
      - Low penalty → finds many change points (risk: noise)
      - High penalty → finds only large, obvious shifts
      - We default to 3.0 which corresponds roughly to requiring a 3-sigma shift

    Returns a list of ChangePoint objects sorted by game index.
    """
    if not RUPTURES_AVAILABLE:
        return []

    n = len(series)
    if n < min_segment_len * 2:
        return []   # not enough data

    # PELT with RBF (Radial Basis Function) cost — good for detecting mean+variance shifts
    algo = rpt.Pelt(model="rbf", min_size=min_segment_len).fit(series.reshape(-1, 1))
    breakpoints = algo.predict(pen=penalty)
    # ruptures returns breakpoints as end indices (exclusive), last one is always n
    # Remove the last one (it's always len(series))
    breakpoints = breakpoints[:-1]

    if not breakpoints:
        return []

    # Build segment boundaries: [(0, bp1), (bp1, bp2), ..., (bpN, n)]
    boundaries = [0] + breakpoints + [n]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    change_points = []
    for i, bp in enumerate(breakpoints):
        before_seg = series[segments[i][0]:segments[i][1]]
        after_seg = series[segments[i + 1][0]:segments[i + 1][1]]

        before_mean = float(np.mean(before_seg))
        after_mean = float(np.mean(after_seg))
        before_std = float(np.std(before_seg)) or 1.0

        magnitude = abs(after_mean - before_mean) / before_std
        direction = "drop" if after_mean < before_mean else "spike"

        change_points.append(ChangePoint(
            game_index=bp,
            metric=metric_name,
            before_mean=before_mean,
            after_mean=after_mean,
            magnitude=magnitude,
            direction=direction,
        ))

    return change_points


def compute_player_baseline(
    game_stats: list[dict],
    min_games: int = 15,
    slump_threshold_sigma: float = 1.5,
) -> BaselineResult | None:
    """
    Compute a robust behavioral baseline for a player from their game history.

    game_stats: list of dicts, each with keys:
        cs_per_min, kill_participation, deaths, game_duration_min,
        gold_per_min, solo_deaths

    Ordered from oldest to most recent.

    Steps:
    1. Run PELT on each key metric independently
    2. Find the most recent stable segment (our "true baseline")
    3. Check if current segment is significantly worse than historical best
       (chronic slump detection)
    4. Compute robust stats (median/IQR) on the true baseline segment only
    """
    n = len(game_stats)
    if n < min_games:
        logger.info(f"Not enough games ({n} < {min_games}) to compute baseline")
        return None

    # Extract metric arrays (oldest → most recent)
    cs_series = np.array([g["cs_per_min"] for g in game_stats], dtype=float)
    kp_series = np.array([g["kill_participation"] for g in game_stats], dtype=float)
    dr_series = np.array([g.get("deaths", 0) / max(g.get("game_duration_min", 30), 1) for g in game_stats], dtype=float)
    gold_series = np.array([g["gold_per_min"] for g in game_stats], dtype=float)
    solo_dr_series = np.array([g.get("solo_deaths", 0) / max(g.get("game_duration_min", 30), 1) for g in game_stats], dtype=float)

    # Run CPD on each metric
    all_change_points = []
    for series, name in [
        (cs_series, "cs_per_min"),
        (kp_series, "kill_participation"),
        (dr_series, "death_rate"),
        (gold_series, "gold_per_min"),
    ]:
        cps = detect_change_points(series, name)
        all_change_points.extend(cps)

    # Find the true baseline start: the latest change point across all metrics
    # This is the start of the player's current behavioral regime
    if all_change_points:
        true_baseline_start = max(cp.game_index for cp in all_change_points)
    else:
        true_baseline_start = 0   # no change points → all games are one stable period

    # Chronic slump detection:
    # Compare the current segment's CS median to the historical best segment median.
    # If current is > slump_threshold_sigma standard deviations below best → slump.
    chronic_slump = False
    if true_baseline_start > 0:
        historical_cs_median = float(np.median(cs_series[:true_baseline_start]))
        current_cs_median = float(np.median(cs_series[true_baseline_start:]))
        historical_std = float(np.std(cs_series[:true_baseline_start])) or 1.0
        sigma_drop = (historical_cs_median - current_cs_median) / historical_std
        chronic_slump = sigma_drop > slump_threshold_sigma
        if chronic_slump:
            logger.info(
                f"Chronic slump detected: CS dropped {sigma_drop:.1f}σ "
                f"({historical_cs_median:.1f} → {current_cs_median:.1f})"
            )

    # Compute robust stats on TRUE BASELINE segment only
    baseline_games = game_stats[true_baseline_start:]
    baseline_cs = cs_series[true_baseline_start:]
    baseline_kp = kp_series[true_baseline_start:]
    baseline_dr = dr_series[true_baseline_start:]
    baseline_gold = gold_series[true_baseline_start:]
    baseline_solo_dr = solo_dr_series[true_baseline_start:]

    cs_median, cs_iqr = _robust_stats(baseline_cs)
    kp_median, _ = _robust_stats(baseline_kp)
    dr_median, _ = _robust_stats(baseline_dr)
    gold_median, _ = _robust_stats(baseline_gold)
    solo_dr_median, _ = _robust_stats(baseline_solo_dr)

    return BaselineResult(
        games_analyzed=n,
        change_points=all_change_points,
        true_baseline_start=true_baseline_start,
        chronic_slump_detected=chronic_slump,
        cs_per_min_median=cs_median,
        cs_per_min_iqr=cs_iqr,
        kill_participation_median=kp_median,
        death_rate_median=dr_median,
        gold_per_min_median=gold_median,
        solo_death_rate_median=solo_dr_median,
    )
