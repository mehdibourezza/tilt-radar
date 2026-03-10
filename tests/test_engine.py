"""
Unit tests for the TiltInferenceEngine.

We test each signal independently, then combined scoring,
confidence computation, exploit recommendations, and the full score() pipeline.

No DB, no network — pure logic.
"""
import pytest
from tests.conftest import make_snapshot, make_player, make_kill_event


# ===========================================================================
# Signal 1 — Repeat deaths to the same enemy (pride-tilt)
# ===========================================================================

class TestRepeatDeathsSignal:

    def test_three_deaths_to_same_enemy_triggers_full_signal(self, engine):
        player = make_player("Jinx", deaths=3)
        events = [
            make_kill_event("Caitlyn", "Jinx", 200),
            make_kill_event("Caitlyn", "Jinx", 500),
            make_kill_event("Caitlyn", "Jinx", 800),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=900)
        assert any("same_enemy_3x" in s for s in signals)
        assert weights["repeat_deaths_same_enemy"] == pytest.approx(0.30)

    def test_two_deaths_to_same_enemy_triggers_half_weight(self, engine):
        player = make_player("Jinx", deaths=2)
        events = [
            make_kill_event("Caitlyn", "Jinx", 200),
            make_kill_event("Caitlyn", "Jinx", 500),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=900)
        assert any("same_enemy_2x" in s for s in signals)
        assert weights["repeat_deaths_same_enemy"] == pytest.approx(0.15)

    def test_one_death_to_same_enemy_does_not_trigger(self, engine):
        player = make_player("Jinx", deaths=1)
        events = [make_kill_event("Caitlyn", "Jinx", 300)]
        signals, weights = engine._extract_signals(player, events, game_time=900)
        assert not any("same_enemy" in s for s in signals)
        assert "repeat_deaths_same_enemy" not in weights

    def test_deaths_to_different_enemies_do_not_trigger(self, engine):
        player = make_player("Jinx", deaths=3)
        events = [
            make_kill_event("Caitlyn", "Jinx", 200),
            make_kill_event("Ezreal", "Jinx", 500),
            make_kill_event("Lux", "Jinx", 800),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=900)
        assert not any("same_enemy" in s for s in signals)

    def test_killer_name_case_insensitive_matching(self, engine):
        """Victim name matching should be case-insensitive."""
        player = make_player("jinx", deaths=3)   # lowercase
        events = [
            make_kill_event("Caitlyn", "Jinx", 200),   # different case
            make_kill_event("Caitlyn", "Jinx", 500),
            make_kill_event("Caitlyn", "Jinx", 800),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=900)
        assert any("same_enemy_3x" in s for s in signals)


# ===========================================================================
# Signal 2 — Death acceleration
# ===========================================================================

class TestDeathAccelerationSignal:

    def test_more_deaths_in_second_half_triggers_signal(self, engine):
        player = make_player("Jinx")
        # 0 early, 3 late — in a 20-min game (midpoint = 600s)
        events = [
            make_kill_event("Caitlyn", "Jinx", 700),
            make_kill_event("Caitlyn", "Jinx", 900),
            make_kill_event("Caitlyn", "Jinx", 1100),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=1200)
        assert "death_rate_accelerating" in signals
        assert "death_acceleration" in weights

    def test_equal_deaths_early_late_does_not_trigger(self, engine):
        player = make_player("Jinx")
        # 2 early, 2 late
        events = [
            make_kill_event("Caitlyn", "Jinx", 100),
            make_kill_event("Caitlyn", "Jinx", 300),
            make_kill_event("Caitlyn", "Jinx", 700),
            make_kill_event("Caitlyn", "Jinx", 900),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=1200)
        assert "death_rate_accelerating" not in signals

    def test_game_under_10min_skips_signal(self, engine):
        """Death acceleration needs at least 10 minutes of data."""
        player = make_player("Jinx")
        events = [
            make_kill_event("Caitlyn", "Jinx", 400),
            make_kill_event("Caitlyn", "Jinx", 500),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=550)
        assert "death_rate_accelerating" not in signals


# ===========================================================================
# Signal 3 — CS drop vs baseline
# ===========================================================================

class TestCSDropSignal:

    def test_large_cs_drop_vs_peer_baseline_triggers_signal(self, engine, peer_baseline):
        """CS/min = 3.0 when peer median is 6.0, IQR=1.2 → deviation = (6.0-3.0)/1.2 = 2.5 > 1.5."""
        player = make_player("Jinx", cs=45)   # 45 CS at 15 min = 3.0 CS/min
        signals, weights = engine._extract_signals(
            player, [], game_time=900, baseline=peer_baseline, is_peer=True
        )
        assert any("cs_down" in s for s in signals)
        assert "cs_drop_vs_baseline" in weights
        assert weights["cs_drop_vs_baseline"] == pytest.approx(0.25)

    def test_slight_cs_drop_triggers_partial_signal(self, engine, peer_baseline):
        """CS/min = 4.8 → deviation = (6.0-4.8)/1.2 = 1.0, between 0.8 and 1.5 → partial."""
        player = make_player("Jinx", cs=72)   # 72 CS at 15 min = 4.8 CS/min
        signals, weights = engine._extract_signals(
            player, [], game_time=900, baseline=peer_baseline, is_peer=True
        )
        assert any("cs_slightly_below" in s for s in signals)
        assert weights["cs_drop_vs_baseline"] == pytest.approx(0.25 * 0.4)

    def test_normal_cs_does_not_trigger(self, engine, peer_baseline):
        """CS/min = 6.5 → above peer median → no signal."""
        player = make_player("Jinx", cs=97)   # 97 CS at 15 min ≈ 6.5 CS/min
        signals, weights = engine._extract_signals(
            player, [], game_time=900, baseline=peer_baseline, is_peer=True
        )
        assert not any("cs" in s for s in signals)

    def test_no_baseline_means_no_cs_signal(self, engine):
        player = make_player("Jinx", cs=30)
        signals, weights = engine._extract_signals(
            player, [], game_time=900, baseline=None
        )
        assert "cs_drop_vs_baseline" not in weights

    def test_cs_signal_skipped_before_5_minutes(self, engine, peer_baseline):
        """Not enough game time to judge CS."""
        player = make_player("Jinx", cs=10)
        signals, weights = engine._extract_signals(
            player, [], game_time=240, baseline=peer_baseline, is_peer=True
        )
        assert "cs_drop_vs_baseline" not in weights

    def test_cs_label_suffix_differs_for_personal_vs_peer(self, engine, personal_baseline, peer_baseline):
        """Signal string should include 'vs_baseline' for personal, 'vs_tier_avg' for peer."""
        player = make_player("Jinx", cs=30)   # very low — will trigger both
        personal_signals, _ = engine._extract_signals(
            player, [], game_time=900, baseline=personal_baseline, is_peer=False
        )
        peer_signals, _ = engine._extract_signals(
            player, [], game_time=900, baseline=peer_baseline, is_peer=True
        )
        assert any("vs_baseline" in s for s in personal_signals)
        assert any("vs_tier_avg" in s for s in peer_signals)


# ===========================================================================
# Signal 4 — Early death cluster
# ===========================================================================

class TestEarlyDeathClusterSignal:

    def test_three_deaths_before_10min_triggers(self, engine):
        player = make_player("Jinx")
        events = [
            make_kill_event("Caitlyn", "Jinx", 200),
            make_kill_event("Caitlyn", "Jinx", 350),
            make_kill_event("Caitlyn", "Jinx", 550),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=900)
        assert any("deaths_before_10min" in s for s in signals)
        assert "early_death_cluster" in weights

    def test_two_deaths_before_10min_does_not_trigger(self, engine):
        player = make_player("Jinx")
        events = [
            make_kill_event("Caitlyn", "Jinx", 200),
            make_kill_event("Caitlyn", "Jinx", 500),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=900)
        assert "early_death_cluster" not in weights

    def test_deaths_after_10min_not_counted(self, engine):
        player = make_player("Jinx")
        events = [
            make_kill_event("Caitlyn", "Jinx", 700),
            make_kill_event("Caitlyn", "Jinx", 800),
            make_kill_event("Caitlyn", "Jinx", 900),
        ]
        signals, weights = engine._extract_signals(player, events, game_time=1200)
        assert "early_death_cluster" not in weights


# ===========================================================================
# Tilt type classification
# ===========================================================================

class TestTiltClassification:

    def test_same_enemy_signal_classifies_as_pride(self, engine):
        assert engine._classify_type(["died_to_same_enemy_3x"]) == "pride"

    def test_cs_drop_and_death_classifies_as_doom(self, engine):
        assert engine._classify_type(["cs_down_30pct_vs_tier_avg", "3_deaths_before_10min"]) == "doom"

    def test_accelerating_deaths_classifies_as_rage(self, engine):
        assert engine._classify_type(["death_rate_accelerating"]) == "rage"

    def test_empty_signals_classifies_as_none(self, engine):
        assert engine._classify_type([]) == "none"

    def test_death_only_signals_default_to_rage(self, engine):
        assert engine._classify_type(["3_deaths_before_10min"]) == "rage"


# ===========================================================================
# Confidence computation
# ===========================================================================

class TestConfidenceComputation:

    def test_personal_baseline_gives_full_confidence(self, engine, personal_baseline):
        conf = engine._compute_confidence(personal_baseline, None, game_time=900)
        assert conf == pytest.approx(1.0)

    def test_peer_baseline_only_gives_70pct_confidence(self, engine, peer_baseline):
        conf = engine._compute_confidence(None, peer_baseline, game_time=900)
        assert conf == pytest.approx(0.70)

    def test_no_baseline_gives_40pct_confidence(self, engine):
        conf = engine._compute_confidence(None, None, game_time=900)
        assert conf == pytest.approx(0.40)

    def test_personal_baseline_scaled_down_under_5min(self, engine, personal_baseline):
        conf = engine._compute_confidence(personal_baseline, None, game_time=240)
        assert conf == pytest.approx(1.0 * 0.4)

    def test_personal_baseline_partially_scaled_5_to_10min(self, engine, personal_baseline):
        conf = engine._compute_confidence(personal_baseline, None, game_time=480)
        assert conf == pytest.approx(1.0 * 0.7)

    def test_personal_baseline_takes_priority_over_peer(self, engine, personal_baseline, peer_baseline):
        conf = engine._compute_confidence(personal_baseline, peer_baseline, game_time=900)
        assert conf == pytest.approx(1.0)   # personal wins


# ===========================================================================
# Exploit / advice recommendations
# ===========================================================================

class TestRecommendations:

    def test_enemy_pride_tilt_exploit(self, engine):
        result = engine._recommend_exploit("pride", 0.70, "Jinx", "enemy")
        assert result is not None
        assert "Jinx" in result
        assert "overcommit" in result.lower()

    def test_self_pride_tilt_advice(self, engine):
        result = engine._recommend_exploit("pride", 0.70, "Jinx", "self")
        assert result is not None
        assert "disengage" in result.lower() or "dying" in result.lower()

    def test_ally_doom_tilt_advice(self, engine):
        result = engine._recommend_exploit("doom", 0.70, "Thresh", "ally")
        assert result is not None
        assert "Thresh" in result
        assert "ally" in result.lower()

    def test_below_threshold_returns_none(self, engine):
        assert engine._recommend_exploit("rage", 0.30, "Jinx", "enemy") is None

    def test_tilt_type_none_returns_none_for_all_player_types(self, engine):
        for pt in ("enemy", "self", "ally"):
            assert engine._recommend_exploit("none", 0.70, "Jinx", pt) is None


# ===========================================================================
# Full score() pipeline
# ===========================================================================

class TestFullScorePipeline:

    def test_score_returns_players_and_enemies_keys(self, engine):
        snapshot = make_snapshot([make_player("Jinx")])
        report = engine.score(snapshot, {}, {})
        assert "players" in report
        assert "enemies" in report
        assert "game_time" in report
        assert "game_time_fmt" in report

    def test_enemies_is_subset_of_players(self, engine):
        ally = make_player("Thresh", team="ORDER", is_enemy=False, is_self=False)
        enemy = make_player("Jinx", team="CHAOS", is_enemy=True)
        self_player = make_player("Me", team="ORDER", is_enemy=False, is_self=True)
        snapshot = make_snapshot([ally, enemy, self_player])
        report = engine.score(snapshot, {}, {})
        enemy_names = {r["summonerName"] for r in report["enemies"]}
        assert "Jinx" in enemy_names
        assert "Thresh" not in enemy_names
        assert "Me" not in enemy_names

    def test_player_type_assigned_correctly(self, engine):
        ally = make_player("Thresh", team="ORDER", is_enemy=False, is_self=False)
        enemy = make_player("Jinx", team="CHAOS", is_enemy=True)
        self_p = make_player("Me", team="ORDER", is_enemy=False, is_self=True)
        snapshot = make_snapshot([ally, enemy, self_p])
        report = engine.score(snapshot, {}, {})
        types_by_name = {r["summonerName"]: r["player_type"] for r in report["players"]}
        assert types_by_name["Me"] == "self"
        assert types_by_name["Thresh"] == "ally"
        assert types_by_name["Jinx"] == "enemy"

    def test_tilt_score_in_range(self, engine, peer_baseline):
        # A player with very bad stats to ensure non-zero score
        player = make_player("Jinx", deaths=5, cs=30)
        events = [
            make_kill_event("Caitlyn", "Jinx", 100),
            make_kill_event("Caitlyn", "Jinx", 200),
            make_kill_event("Caitlyn", "Jinx", 300),
            make_kill_event("Caitlyn", "Jinx", 400),
            make_kill_event("Caitlyn", "Jinx", 500),
        ]
        snapshot = make_snapshot([player], events=events)
        report = engine.score(snapshot, {}, {"Jinx": peer_baseline})
        score = report["players"][0]["tilt_score"]
        assert 0.0 <= score <= 1.0

    def test_no_signals_in_clean_game(self, engine, peer_baseline):
        """A player with perfect CS and no deaths should score 0."""
        player = make_player("Jinx", deaths=0, cs=100)  # 6.67 CS/min, above baseline
        snapshot = make_snapshot([player], events=[])
        report = engine.score(snapshot, {}, {"Jinx": peer_baseline})
        score = report["players"][0]["tilt_score"]
        assert score == 0.0

    def test_exploit_hidden_below_medium_threshold(self, engine, peer_baseline):
        """Exploit should be None when score is below TILT_MEDIUM."""
        player = make_player("Jinx", deaths=0, cs=100)
        snapshot = make_snapshot([player], events=[])
        report = engine.score(snapshot, {}, {"Jinx": peer_baseline})
        assert report["players"][0]["exploit"] is None
