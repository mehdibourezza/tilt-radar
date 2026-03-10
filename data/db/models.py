"""
SQLAlchemy ORM models.

Schema design philosophy:
  - players: identity and current rank
  - matches: one row per game
  - player_match_stats: one row per player per game (the feature source)
  - player_baselines: computed baseline profiles per player (updated periodically)
  - tilt_events: recorded tilt episodes with type and signals
"""

from datetime import datetime
from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, String, JSON, Text, Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    puuid: Mapped[str] = mapped_column(String(78), unique=True, nullable=False)
    game_name: Mapped[str] = mapped_column(String(64), nullable=False)
    tag_line: Mapped[str] = mapped_column(String(16), nullable=False)
    platform: Mapped[str] = mapped_column(String(8), nullable=False)  # euw1, na1, ...

    # Current rank (refreshed periodically)
    tier: Mapped[str | None] = mapped_column(String(16))    # IRON, GOLD, DIAMOND, ...
    division: Mapped[str | None] = mapped_column(String(4)) # I, II, III, IV
    lp: Mapped[int | None] = mapped_column(Integer)

    # Ingestion state
    last_ingested_at: Mapped[datetime | None] = mapped_column(DateTime)
    total_games_ingested: Mapped[int] = mapped_column(Integer, default=0)
    baseline_computed_at: Mapped[datetime | None] = mapped_column(DateTime)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    stats: Mapped[list["PlayerMatchStats"]] = relationship(back_populates="player")
    baseline: Mapped["PlayerBaseline | None"] = relationship(back_populates="player", uselist=False)
    tilt_events: Mapped[list["TiltEvent"]] = relationship(back_populates="player")


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)  # EUW1_7123456789
    platform: Mapped[str] = mapped_column(String(8), nullable=False)
    queue_id: Mapped[int] = mapped_column(Integer)        # 420 = ranked solo
    game_duration: Mapped[int] = mapped_column(Integer)   # seconds
    game_start: Mapped[datetime] = mapped_column(DateTime)
    patch: Mapped[str] = mapped_column(String(8))         # "14.18"

    # Raw timeline stored as JSON — expensive to recompute
    timeline_raw: Mapped[dict | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    player_stats: Mapped[list["PlayerMatchStats"]] = relationship(back_populates="match")


class PlayerMatchStats(Base):
    """
    Per-player per-game statistics. This is our feature table.
    One row = one player's performance in one game.
    """
    __tablename__ = "player_match_stats"
    __table_args__ = (
        Index("ix_pms_player_game_start", "player_id", "game_start"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    game_start: Mapped[datetime] = mapped_column(DateTime)  # denormalized for fast queries

    # Champion
    champion_id: Mapped[int] = mapped_column(Integer)
    champion_name: Mapped[str] = mapped_column(String(32))
    role: Mapped[str] = mapped_column(String(16))     # TOP, JUNGLE, MID, BOTTOM, SUPPORT
    won: Mapped[bool] = mapped_column(Boolean)

    # Core stats
    kills: Mapped[int] = mapped_column(Integer)
    deaths: Mapped[int] = mapped_column(Integer)
    assists: Mapped[int] = mapped_column(Integer)
    cs: Mapped[int] = mapped_column(Integer)               # total minions + jungle CS
    vision_score: Mapped[int] = mapped_column(Integer)
    damage_dealt: Mapped[int] = mapped_column(Integer)
    gold_earned: Mapped[int] = mapped_column(Integer)

    # Derived rates (computed at ingestion time — per minute)
    cs_per_min: Mapped[float] = mapped_column(Float)
    gold_per_min: Mapped[float] = mapped_column(Float)
    kill_participation: Mapped[float] = mapped_column(Float)  # (kills+assists) / team_kills

    # Behavioral signals (extracted from timeline)
    deaths_pre_15: Mapped[int] = mapped_column(Integer, default=0)
    deaths_post_15: Mapped[int] = mapped_column(Integer, default=0)
    avg_death_distance_from_base: Mapped[float | None] = mapped_column(Float)
    solo_deaths: Mapped[int] = mapped_column(Integer, default=0)   # died without allies nearby
    repeat_deaths_same_enemy: Mapped[int] = mapped_column(Integer, default=0)  # died 2+ times to same enemy
    objective_contests_when_behind: Mapped[int] = mapped_column(Integer, default=0)
    cs_trajectory: Mapped[list | None] = mapped_column(JSON)  # [cs@5min, cs@10min, ..., cs@25min]

    player: Mapped["Player"] = relationship(back_populates="stats")
    match: Mapped["Match"] = relationship(back_populates="player_stats")


class PlayerBaseline(Base):
    """
    Computed baseline profile for a player.
    Updated after each ingestion batch via the change point detector.

    Stores three windows (long/medium/short) + robust statistics.
    The difference between windows is what we compare against in-game.
    """
    __tablename__ = "player_baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), unique=True, nullable=False)

    games_used: Mapped[int] = mapped_column(Integer)

    # Long-term robust baseline (median/IQR over up to 100 games)
    lt_cs_per_min_median: Mapped[float | None] = mapped_column(Float)
    lt_cs_per_min_iqr: Mapped[float | None] = mapped_column(Float)
    lt_kill_participation_median: Mapped[float | None] = mapped_column(Float)
    lt_death_rate_median: Mapped[float | None] = mapped_column(Float)
    lt_gold_per_min_median: Mapped[float | None] = mapped_column(Float)
    lt_solo_death_rate_median: Mapped[float | None] = mapped_column(Float)

    # Medium-term window (last 20 games)
    mt_cs_per_min_median: Mapped[float | None] = mapped_column(Float)
    mt_kill_participation_median: Mapped[float | None] = mapped_column(Float)
    mt_death_rate_median: Mapped[float | None] = mapped_column(Float)

    # Change point detection results
    # List of dicts: [{game_index, metric, magnitude, detected_at}, ...]
    change_points: Mapped[list | None] = mapped_column(JSON)
    last_change_point_at: Mapped[datetime | None] = mapped_column(DateTime)
    chronic_slump_detected: Mapped[bool] = mapped_column(Boolean, default=False)

    # Peer group anchor (filled from rank-normalized stats)
    peer_cs_per_min_median: Mapped[float | None] = mapped_column(Float)
    peer_death_rate_median: Mapped[float | None] = mapped_column(Float)
    peer_kill_participation_median: Mapped[float | None] = mapped_column(Float)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    player: Mapped["Player"] = relationship(back_populates="baseline")


class PeerGroupBaseline(Base):
    """
    Aggregate stats for players at a given rank tier.
    Used as fallback baseline when a player has no personal history in our DB.

    One row per (queue, tier, division) combination.
    Example: RANKED_SOLO_5x5 / GOLD / II → median CS/min = 6.1, etc.

    Populated by scripts/populate_peer_groups.py
    Updated periodically (patch changes affect these numbers).
    """
    __tablename__ = "peer_group_baselines"
    __table_args__ = (
        Index("ix_pgb_tier_division", "tier", "division", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    queue: Mapped[str] = mapped_column(String(32), default="RANKED_SOLO_5x5")
    tier: Mapped[str] = mapped_column(String(16))      # IRON, BRONZE, ..., CHALLENGER
    division: Mapped[str] = mapped_column(String(4))   # I, II, III, IV

    sample_size: Mapped[int] = mapped_column(Integer)  # how many players sampled

    # Core performance medians
    cs_per_min_median: Mapped[float] = mapped_column(Float)
    cs_per_min_iqr: Mapped[float] = mapped_column(Float)
    kill_participation_median: Mapped[float] = mapped_column(Float)
    death_rate_median: Mapped[float] = mapped_column(Float)   # deaths per minute
    death_rate_iqr: Mapped[float] = mapped_column(Float)
    gold_per_min_median: Mapped[float] = mapped_column(Float)
    solo_death_rate_median: Mapped[float] = mapped_column(Float)
    win_rate_median: Mapped[float] = mapped_column(Float)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TiltEvent(Base):
    """
    A recorded tilt episode — either historical (from batch analysis)
    or real-time (detected during a live game analysis request).
    """
    __tablename__ = "tilt_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"))  # null for live analysis

    tilt_score: Mapped[float] = mapped_column(Float)      # 0.0 - 1.0
    tilt_type: Mapped[str] = mapped_column(String(32))    # rage | doom | pride | blame
    confidence: Mapped[float] = mapped_column(Float)

    # Signals that triggered the detection
    key_signals: Mapped[list] = mapped_column(JSON)        # ["died_to_same_enemy_3x", ...]
    exploit_recommendation: Mapped[str | None] = mapped_column(Text)

    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    player: Mapped["Player"] = relationship(back_populates="tilt_events")


class TiltPredictionLog(Base):
    """
    Post-game evaluation of tilt predictions vs actual outcomes.

    This is the labeled training data for the future ML model.
    After every game, we compare the peak tilt score we predicted with what
    actually happened at the final scoreboard — so the model can learn from mistakes.

    Verdict taxonomy:
      - true_positive:  predicted tilt ≥ 0.55 AND final performance was poor
      - false_positive: predicted tilt ≥ 0.55 BUT final performance was fine
      - true_negative:  predicted no tilt AND final performance was fine
      - false_negative: predicted no tilt BUT final performance was actually poor

    These records are the foundation for:
      1. Recalibrating signal weights (which signals correlate with real outcomes?)
      2. Training the Temporal Transformer to replace rule-based v1
    """
    __tablename__ = "tilt_prediction_logs"
    __table_args__ = (
        Index("ix_tpl_recorded_at", "recorded_at"),
        Index("ix_tpl_player_name", "player_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Who was predicted (we don't always have a DB player_id for enemies)
    player_name: Mapped[str] = mapped_column(String(64))
    player_type: Mapped[str] = mapped_column(String(8))   # self | ally | enemy
    champion_name: Mapped[str] = mapped_column(String(32))

    # What we predicted at peak
    peak_tilt_score: Mapped[float] = mapped_column(Float)
    peak_tilt_type: Mapped[str] = mapped_column(String(32))  # rage | doom | pride | none
    peak_signals: Mapped[list] = mapped_column(JSON)         # signals active at peak

    # Final scoreboard (from last snapshot before game ended)
    final_kills: Mapped[int] = mapped_column(Integer, default=0)
    final_deaths: Mapped[int] = mapped_column(Integer, default=0)
    final_assists: Mapped[int] = mapped_column(Integer, default=0)
    final_cs: Mapped[int] = mapped_column(Integer, default=0)
    game_duration_min: Mapped[float] = mapped_column(Float)
    final_cs_per_min: Mapped[float] = mapped_column(Float)
    final_death_rate: Mapped[float] = mapped_column(Float)    # deaths per minute
    final_kda: Mapped[float] = mapped_column(Float)           # (kills+assists)/max(deaths,1)

    # Reference baseline (what we compared against — snapshot for reproducibility)
    peer_cs_per_min_median: Mapped[float | None] = mapped_column(Float)
    peer_death_rate_median: Mapped[float | None] = mapped_column(Float)

    # Verdict
    predicted_tilted: Mapped[bool] = mapped_column(Boolean)   # peak >= 0.55
    performed_poorly: Mapped[bool] = mapped_column(Boolean)   # final stats confirm tilt
    verdict: Mapped[str] = mapped_column(String(16))          # true_positive | false_positive | true_negative | false_negative

    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
