"""
Unit tests for change point detection and baseline computation.

These test the statistical core of the system — the PELT algorithm
and the BaselineResult computation logic.

No DB, no network — pure numpy + ruptures.
"""
import pytest
import numpy as np
from ml.features.change_point import detect_change_points, compute_player_baseline


# ===========================================================================
# Helpers
# ===========================================================================

def flat_game_history(cs: float = 6.5, n: int = 20) -> list[dict]:
    """N games all with the same stats — no change points expected."""
    return [
        {
            "cs_per_min": cs,
            "kill_participation": 0.55,
            "deaths": 5,
            "game_duration_min": 30,
            "gold_per_min": 380,
            "solo_deaths": 2,
            "won": True,
        }
        for _ in range(n)
    ]


def step_game_history(cs_before: float, cs_after: float, n_before: int = 15, n_after: int = 15) -> list[dict]:
    """Step-function history: n_before games at cs_before, then n_after at cs_after."""
    games_before = [
        {
            "cs_per_min": cs_before,
            "kill_participation": 0.55,
            "deaths": 5,
            "game_duration_min": 30,
            "gold_per_min": 380,
            "solo_deaths": 2,
            "won": True,
        }
        for _ in range(n_before)
    ]
    games_after = [
        {
            "cs_per_min": cs_after,
            "kill_participation": 0.55,
            "deaths": 5,
            "game_duration_min": 30,
            "gold_per_min": 380,
            "solo_deaths": 2,
            "won": True,
        }
        for _ in range(n_after)
    ]
    return games_before + games_after


# ===========================================================================
# detect_change_points
# ===========================================================================

class TestDetectChangePoints:

    def test_flat_series_has_no_change_points(self):
        series = np.full(30, 6.5)
        cps = detect_change_points(series, "cs_per_min")
        assert cps == []

    def test_obvious_step_down_detected(self):
        """A sharp drop from 7.0 to 3.5 should produce exactly one change point."""
        series = np.concatenate([np.full(15, 7.0), np.full(15, 3.5)])
        cps = detect_change_points(series, "cs_per_min")
        assert len(cps) >= 1
        # The change point should be around index 15
        assert any(10 <= cp.game_index <= 20 for cp in cps)

    def test_change_point_direction_is_drop(self):
        series = np.concatenate([np.full(15, 7.0), np.full(15, 3.5)])
        cps = detect_change_points(series, "cs_per_min")
        assert cps[0].direction == "drop"
        assert cps[0].before_mean > cps[0].after_mean

    def test_change_point_direction_is_spike(self):
        series = np.concatenate([np.full(15, 4.0), np.full(15, 7.5)])
        cps = detect_change_points(series, "cs_per_min")
        assert cps[0].direction == "spike"
        assert cps[0].after_mean > cps[0].before_mean

    def test_too_short_series_returns_empty(self):
        """Less than 2 * min_segment_len games → skip."""
        series = np.array([6.5, 3.0, 6.5, 3.0])
        cps = detect_change_points(series, "cs_per_min", min_segment_len=5)
        assert cps == []

    def test_magnitude_is_positive(self):
        series = np.concatenate([np.full(15, 7.0), np.full(15, 3.5)])
        cps = detect_change_points(series, "cs_per_min")
        for cp in cps:
            assert cp.magnitude > 0

    def test_metric_name_stored_correctly(self):
        series = np.concatenate([np.full(15, 7.0), np.full(15, 3.5)])
        cps = detect_change_points(series, "gold_per_min")
        assert all(cp.metric == "gold_per_min" for cp in cps)


# ===========================================================================
# compute_player_baseline
# ===========================================================================

class TestComputePlayerBaseline:

    def test_too_few_games_returns_none(self):
        games = flat_game_history(n=5)
        result = compute_player_baseline(games, min_games=15)
        assert result is None

    def test_stable_history_no_change_points(self):
        games = flat_game_history(n=20)
        result = compute_player_baseline(games, min_games=15)
        assert result is not None
        assert result.change_points == []
        assert result.true_baseline_start == 0
        assert result.chronic_slump_detected is False

    def test_stable_history_baseline_stats_correct(self):
        games = flat_game_history(cs=6.5, n=20)
        result = compute_player_baseline(games, min_games=15)
        assert result is not None
        assert result.cs_per_min_median == pytest.approx(6.5, abs=0.1)
        assert result.cs_per_min_iqr == pytest.approx(0.0, abs=0.1)

    def test_games_analyzed_reflects_input(self):
        games = flat_game_history(n=20)
        result = compute_player_baseline(games, min_games=15)
        assert result.games_analyzed == 20

    def test_step_drop_moves_baseline_start(self):
        """After a sharp drop, true_baseline_start should be > 0."""
        games = step_game_history(cs_before=7.0, cs_after=3.5, n_before=15, n_after=15)
        result = compute_player_baseline(games, min_games=15)
        assert result is not None
        assert result.true_baseline_start > 0

    def test_step_drop_baseline_stats_reflect_recent_period(self):
        """Stats should reflect the CURRENT (lower) period, not the full history."""
        games = step_game_history(cs_before=7.0, cs_after=3.5, n_before=15, n_after=15)
        result = compute_player_baseline(games, min_games=15)
        assert result is not None
        # Current period CS/min should be close to 3.5, not to the naive mean of 5.25
        assert result.cs_per_min_median < 5.0

    def test_chronic_slump_detected_on_large_drop(self):
        """A big drop (>1.5 sigma) in CS should flag chronic_slump_detected."""
        games = step_game_history(cs_before=7.0, cs_after=3.0, n_before=20, n_after=20)
        result = compute_player_baseline(games, min_games=15)
        assert result is not None
        assert result.chronic_slump_detected is True

    def test_chronic_slump_not_detected_on_improvement(self):
        """A step UP should never flag as a slump."""
        games = step_game_history(cs_before=4.0, cs_after=7.0, n_before=15, n_after=15)
        result = compute_player_baseline(games, min_games=15)
        assert result is not None
        assert result.chronic_slump_detected is False

    def test_chronic_slump_not_detected_on_small_drop(self):
        """A drop within normal variance shouldn't trigger the slump flag."""
        games = step_game_history(cs_before=6.5, cs_after=6.0, n_before=15, n_after=15)
        result = compute_player_baseline(games, min_games=15)
        # May or may not have change points, but should not flag as slump
        if result is not None:
            assert result.chronic_slump_detected is False
