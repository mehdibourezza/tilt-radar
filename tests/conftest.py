"""
Shared fixtures and helpers for TiltRadar tests.
"""
import sys
import pytest
from types import SimpleNamespace

sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# Mock baseline objects
# The real ORM objects (PlayerBaseline, PeerGroupBaseline) are accessed via
# getattr() in the engine, so SimpleNamespace is a perfect stand-in.
# ---------------------------------------------------------------------------

@pytest.fixture
def personal_baseline():
    """A mock PlayerBaseline with typical Silver player stats."""
    return SimpleNamespace(
        lt_cs_per_min_median=6.5,
        lt_cs_per_min_iqr=1.0,
        lt_kill_participation_median=0.55,
        lt_death_rate_median=0.30,
        lt_gold_per_min_median=380.0,
        lt_solo_death_rate_median=0.15,
    )


@pytest.fixture
def peer_baseline():
    """A mock PeerGroupBaseline for Silver IV."""
    return SimpleNamespace(
        cs_per_min_median=6.0,
        cs_per_min_iqr=1.2,
        kill_participation_median=0.50,
        death_rate_median=0.32,
        death_rate_iqr=0.15,
        gold_per_min_median=370.0,
        solo_death_rate_median=0.16,
    )


@pytest.fixture
def engine():
    """A fresh TiltInferenceEngine for each test."""
    from ml.inference.engine import TiltInferenceEngine
    return TiltInferenceEngine()


def make_snapshot(
    players: list[dict],
    events: list[dict] | None = None,
    game_time: float = 900.0,   # 15 minutes default
) -> dict:
    """Build a minimal snapshot dict as the agent would send it."""
    return {
        "game_time": game_time,
        "your_summoner": "TestPlayer",
        "players": players,
        "events": events or [],
    }


def make_player(
    name: str = "TestEnemy",
    champion: str = "Jinx",
    team: str = "CHAOS",
    is_enemy: bool = True,
    is_self: bool = False,
    kills: int = 0,
    deaths: int = 0,
    assists: int = 0,
    cs: int = 90,   # 6.0 CS/min at 15 min
) -> dict:
    return {
        "summonerName": name,
        "championName": champion,
        "team": team,
        "is_enemy": is_enemy,
        "is_self": is_self,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "cs": cs,
    }


def make_kill_event(killer: str, victim: str, time: float = 300.0) -> dict:
    return {"type": "ChampionKill", "killer": killer, "victim": victim, "time": time}
