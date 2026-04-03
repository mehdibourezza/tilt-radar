"""
Pydantic schemas for strict input validation across all API endpoints.

All user-facing inputs (path params, WebSocket messages) are validated here
to enforce type checks, length limits, and reject unexpected fields.
"""

import re
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ---------------------------------------------------------------------------
# Path parameter validation
# ---------------------------------------------------------------------------

# Riot summoner names: 3-16 chars, alphanumeric + spaces + some unicode.
# Tags: 2-5 alphanumeric chars (e.g. "EUW", "007").
SUMMONER_NAME_RE = re.compile(r"^[\w\s\-\.]{1,30}$", re.UNICODE)
TAG_RE = re.compile(r"^[a-zA-Z0-9]{1,8}$")


def validate_summoner_name(name: str) -> str:
    """Validate a Riot summoner name against allowed characters and length."""
    name = name.strip()
    if not name or len(name) > 30:
        raise ValueError("Summoner name must be 1-30 characters")
    if not SUMMONER_NAME_RE.match(name):
        raise ValueError("Summoner name contains invalid characters")
    return name


def validate_tag(tag: str) -> str:
    """Validate a Riot tag line (e.g. 'EUW', '007')."""
    tag = tag.strip()
    if not TAG_RE.match(tag):
        raise ValueError("Tag must be 1-8 alphanumeric characters")
    return tag


# ---------------------------------------------------------------------------
# WebSocket snapshot schemas — strict validation, reject unknown fields
# ---------------------------------------------------------------------------

class PlayerSnapshot(BaseModel):
    """Single player state within a game snapshot."""
    model_config = ConfigDict(extra="ignore")  # silently drop unexpected fields

    summonerName: str = Field(..., min_length=1, max_length=64)
    championName: str = Field(default="", max_length=64)
    team: str = Field(default="", max_length=16)
    kills: int = Field(default=0, ge=0, le=200)
    deaths: int = Field(default=0, ge=0, le=200)
    assists: int = Field(default=0, ge=0, le=200)
    cs: int = Field(default=0, ge=0, le=5000)
    ward_score: float = Field(default=0.0, ge=0, le=1000)
    position: str = Field(default="", max_length=32)
    level: int = Field(default=1, ge=1, le=30)
    items: list[str] = Field(default_factory=list, max_length=10)
    sold_items: list[str] = Field(default_factory=list, max_length=20)
    is_dead: bool = Field(default=False)
    is_self: bool = Field(default=False)
    is_enemy: bool = Field(default=False)
    puuid: str = Field(default="", max_length=128)
    obj_total: int = Field(default=0, ge=0, le=100)
    obj_missed: int = Field(default=0, ge=0, le=100)


class GameEvent(BaseModel):
    """A single in-game event (kill, objective, etc.)."""
    model_config = ConfigDict(extra="ignore")

    type: str = Field(..., max_length=64)
    time: float = Field(default=0.0, ge=0, le=7200)
    killer: str = Field(default="", max_length=64)
    victim: str = Field(default="", max_length=64)
    assisters: list[str] = Field(default_factory=list, max_length=10)


class GameSnapshot(BaseModel):
    """
    Full game snapshot sent by the local agent every ~5 seconds.
    Validated on arrival to reject malformed or oversized payloads.
    """
    model_config = ConfigDict(extra="ignore")

    game_time: float = Field(..., ge=0, le=7200)  # max 2 hours
    players: list[PlayerSnapshot] = Field(..., max_length=10)
    events: list[GameEvent] = Field(default_factory=list, max_length=2000)

    @field_validator("players")
    @classmethod
    def limit_player_count(cls, v: list) -> list:
        if len(v) > 10:
            raise ValueError("A game has at most 10 players")
        return v


class GameOverSnapshot(BaseModel):
    """Payload sent when the game ends."""
    model_config = ConfigDict(extra="ignore")

    event: str = Field(...)
    final_players: list[PlayerSnapshot] = Field(default_factory=list, max_length=10)
    game_duration_min: float = Field(default=30.0, ge=0, le=120)

    @field_validator("event")
    @classmethod
    def must_be_game_over(cls, v: str) -> str:
        if v != "game_over":
            raise ValueError("Expected event='game_over'")
        return v
