"""
Unit tests for the post-game tilt prediction evaluation (_evaluate_outcome).

This is the self-judgment system: after each game, we compare what we predicted
(peak tilt score) against what actually happened (final KDA/CS).

Four verdicts to test:
  - true_positive:  predicted tilt AND final stats were bad
  - false_positive: predicted tilt BUT final stats were fine
  - true_negative:  did NOT predict tilt AND final stats were fine
  - false_negative: did NOT predict tilt BUT final stats were bad
"""
import pytest
from types import SimpleNamespace
from api.routers.ws import _evaluate_outcome, PREDICTION_THRESHOLD


# ===========================================================================
# Helpers
# ===========================================================================

def make_peak(score: float, tilt_type: str = "rage", signals: list | None = None) -> dict:
    return {
        "score": score,
        "tilt_type": tilt_type,
        "signals": signals or ["death_rate_accelerating"],
        "champion": "Jinx",
        "player_type": "enemy",
    }


def make_final_player(kills: int = 2, deaths: int = 8, assists: int = 3, cs: int = 100) -> dict:
    return {
        "summonerName": "TestEnemy",
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "cs": cs,
    }


def silver_peer_baseline():
    """Peer baseline: CS/min=6.0, death_rate=0.3/min."""
    return SimpleNamespace(
        cs_per_min_median=6.0,
        cs_per_min_iqr=1.2,
        death_rate_median=0.30,
        death_rate_iqr=0.15,
    )


# ===========================================================================
# Verdict: true_positive
# ===========================================================================

class TestTruePositive:

    def test_high_tilt_score_and_bad_final_stats_is_true_positive(self):
        """High peak + poor performance = we were right."""
        peak = make_peak(score=0.75)
        # 10 deaths in 30 min = 0.33/min (above 0.30 * 1.5 = 0.45) and low CS
        final = make_final_player(kills=1, deaths=10, assists=2, cs=80)   # 2.67 CS/min < 6.0*0.75
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=silver_peer_baseline(),
        )
        assert result["verdict"] == "true_positive"
        assert result["predicted_tilted"] is True
        assert result["performed_poorly"] is True

    def test_verdict_true_positive_when_only_death_rate_high(self):
        """CS is fine but deaths are very high → still performed poorly."""
        peak = make_peak(score=0.65)
        # 15 deaths in 30 min = 0.5/min, well above 0.45 threshold (0.30 * 1.5)
        final = make_final_player(kills=3, deaths=15, assists=5, cs=200)
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=silver_peer_baseline(),
        )
        assert result["verdict"] == "true_positive"


# ===========================================================================
# Verdict: false_positive
# ===========================================================================

class TestFalsePositive:

    def test_high_tilt_score_but_good_final_stats_is_false_positive(self):
        """High peak but player recovered — we over-triggered."""
        peak = make_peak(score=0.72)
        # Good CS (200 in 30 min = 6.67/min, above 6.0) and moderate deaths
        final = make_final_player(kills=5, deaths=4, assists=8, cs=200)
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=silver_peer_baseline(),
        )
        assert result["verdict"] == "false_positive"
        assert result["predicted_tilted"] is True
        assert result["performed_poorly"] is False

    def test_false_positive_fields_are_logged(self):
        peak = make_peak(score=0.80, tilt_type="pride", signals=["died_to_same_enemy_3x"])
        final = make_final_player(kills=8, deaths=3, assists=10, cs=210)
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=silver_peer_baseline(),
        )
        assert result["peak_tilt_score"] == pytest.approx(0.80)
        assert result["peak_tilt_type"] == "pride"
        assert result["verdict"] == "false_positive"


# ===========================================================================
# Verdict: true_negative
# ===========================================================================

class TestTrueNegative:

    def test_low_score_and_good_stats_is_true_negative(self):
        """We correctly stayed silent and the player performed fine."""
        peak = make_peak(score=0.20, tilt_type="none", signals=[])
        final = make_final_player(kills=4, deaths=3, assists=7, cs=195)
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=silver_peer_baseline(),
        )
        assert result["verdict"] == "true_negative"
        assert result["predicted_tilted"] is False
        assert result["performed_poorly"] is False


# ===========================================================================
# Verdict: false_negative
# ===========================================================================

class TestFalseNegative:

    def test_low_score_but_bad_stats_is_false_negative(self):
        """We missed a tilt that was real — player performed poorly despite low score."""
        peak = make_peak(score=0.25, tilt_type="none", signals=[])
        # 12 deaths in 30 min = 0.4/min, CS only 70 in 30 min = 2.33/min
        final = make_final_player(kills=1, deaths=12, assists=2, cs=70)
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=silver_peer_baseline(),
        )
        assert result["verdict"] == "false_negative"
        assert result["predicted_tilted"] is False
        assert result["performed_poorly"] is True


# ===========================================================================
# Edge cases and output format
# ===========================================================================

class TestEvaluateOutcomeEdgeCases:

    def test_no_peer_baseline_uses_heuristics(self):
        """When peer baseline is None, fall back to KDA+deaths heuristics."""
        peak = make_peak(score=0.75)
        # KDA < 1.0 and deaths >= 5 → performed poorly
        final = make_final_player(kills=1, deaths=8, assists=2, cs=100)
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=None,
        )
        assert result["verdict"] in ("true_positive", "false_positive", "true_negative", "false_negative")
        assert result["peer_cs_per_min_median"] is None
        assert result["peer_death_rate_median"] is None

    def test_no_peer_baseline_heuristic_kda_above_1_is_not_poor(self):
        """KDA >= 1.0 or deaths < 5 → not performing poorly."""
        peak = make_peak(score=0.75)
        final = make_final_player(kills=5, deaths=3, assists=6, cs=120)
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=None,
        )
        assert result["performed_poorly"] is False

    def test_output_contains_all_expected_keys(self):
        required_keys = {
            "player_name", "player_type", "champion_name",
            "peak_tilt_score", "peak_tilt_type", "peak_signals",
            "final_kills", "final_deaths", "final_assists", "final_cs",
            "game_duration_min", "final_cs_per_min", "final_death_rate", "final_kda",
            "peer_cs_per_min_median", "peer_death_rate_median",
            "predicted_tilted", "performed_poorly", "verdict",
        }
        peak = make_peak(score=0.70)
        final = make_final_player()
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final, 30.0, silver_peer_baseline()
        )
        assert required_keys.issubset(result.keys())

    def test_derived_rates_are_computed_correctly(self):
        peak = make_peak(score=0.50)
        final = make_final_player(kills=3, deaths=6, assists=9, cs=120)
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=30.0,
            peer_baseline=silver_peer_baseline(),
        )
        assert result["final_cs_per_min"] == pytest.approx(120 / 30, abs=0.01)
        assert result["final_death_rate"] == pytest.approx(6 / 30, abs=0.01)
        assert result["final_kda"] == pytest.approx((3 + 9) / 6, abs=0.01)

    def test_prediction_threshold_boundary(self):
        """Score exactly at PREDICTION_THRESHOLD should be predicted_tilted=True."""
        peak = make_peak(score=PREDICTION_THRESHOLD)
        final = make_final_player()
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final, 30.0, silver_peer_baseline()
        )
        assert result["predicted_tilted"] is True

    def test_score_just_below_threshold_is_not_predicted_tilted(self):
        peak = make_peak(score=PREDICTION_THRESHOLD - 0.01)
        final = make_final_player()
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final, 30.0, silver_peer_baseline()
        )
        assert result["predicted_tilted"] is False

    def test_zero_game_duration_does_not_crash(self):
        """Guard against division by zero if game_duration_min is 0."""
        peak = make_peak(score=0.60)
        final = make_final_player(deaths=5, cs=50)
        # Should not raise
        result = _evaluate_outcome(
            "TestEnemy", "enemy", "Jinx",
            peak, final,
            game_duration_min=0.0,
            peer_baseline=silver_peer_baseline(),
        )
        assert "verdict" in result

    def test_self_player_type_stored_correctly(self):
        peak = {**make_peak(score=0.65), "player_type": "self"}
        final = make_final_player()
        result = _evaluate_outcome(
            "Me", "self", "Jinx",
            peak, final, 30.0, silver_peer_baseline()
        )
        assert result["player_type"] == "self"
