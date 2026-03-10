"""
Repository layer — all database access goes through here.

Why a repository pattern?
  - Keeps SQL/ORM logic out of business logic (inference engine, workers, routers)
  - Easy to test: mock the repository instead of the DB
  - Single place to optimize queries later (add indexes, caching, etc.)

PlayerRepository handles everything related to players, their stats, and baselines.
"""

import logging
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from data.db.models import Player, Match, PlayerMatchStats, PlayerBaseline, TiltEvent, PeerGroupBaseline, TiltPredictionLog
from ml.features.change_point import BaselineResult

logger = logging.getLogger(__name__)


class PlayerRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    # -------------------------------------------------------------------------
    # Player
    # -------------------------------------------------------------------------

    async def get_player_by_puuid(self, puuid: str) -> Player | None:
        result = await self.session.execute(
            select(Player).where(Player.puuid == puuid)
        )
        return result.scalar_one_or_none()

    async def get_or_create_player(
        self, puuid: str, game_name: str, tag_line: str, platform: str
    ) -> Player:
        player = await self.get_player_by_puuid(puuid)
        if player:
            return player

        player = Player(
            puuid=puuid,
            game_name=game_name,
            tag_line=tag_line,
            platform=platform,
        )
        self.session.add(player)
        await self.session.flush()   # flush to get the generated ID without committing
        logger.info(f"Created new player: {game_name}#{tag_line}")
        return player

    async def update_rank(self, puuid: str, tier: str, division: str, lp: int):
        await self.session.execute(
            update(Player)
            .where(Player.puuid == puuid)
            .values(tier=tier, division=division, lp=lp, updated_at=datetime.utcnow())
        )

    # -------------------------------------------------------------------------
    # Match stats
    # -------------------------------------------------------------------------

    async def match_exists(self, match_id: str) -> bool:
        result = await self.session.execute(
            select(Match.id).where(Match.match_id == match_id)
        )
        return result.scalar_one_or_none() is not None

    async def upsert_player_stats(self, puuid: str, game_stats: list[dict]):
        """
        Insert PlayerMatchStats rows for a player.
        Skips games that are already in the DB (idempotent).

        game_stats: list of dicts as returned by workers/tasks._parse_match()
        Each dict must include a 'match_id' key for deduplication.
        """
        player = await self.get_player_by_puuid(puuid)
        if not player:
            logger.warning(f"Player {puuid} not found — cannot insert stats")
            return

        inserted = 0
        for stats in game_stats:
            match_id_str = stats.get("match_id")
            if not match_id_str:
                continue

            # Skip if already stored
            if await self.match_exists(match_id_str):
                continue

            match = Match(
                match_id=match_id_str,
                platform=stats.get("platform", "unknown"),
                queue_id=stats.get("queue_id", 420),
                game_duration=int(stats.get("game_duration_min", 0) * 60),
                game_start=stats.get("game_start", datetime.utcnow()),
                patch=stats.get("patch", "unknown"),
            )
            self.session.add(match)
            await self.session.flush()

            row = PlayerMatchStats(
                player_id=player.id,
                match_id=match.id,
                game_start=match.game_start,
                champion_id=stats.get("champion_id", 0),
                champion_name=stats.get("champion_name", "unknown"),
                role=stats.get("role", "unknown"),
                won=stats.get("won", False),
                kills=stats.get("kills", 0),
                deaths=stats.get("deaths", 0),
                assists=stats.get("assists", 0),
                cs=stats.get("cs", 0),
                vision_score=stats.get("vision_score", 0),
                damage_dealt=stats.get("damage_dealt", 0),
                gold_earned=stats.get("gold_earned", 0),
                cs_per_min=stats.get("cs_per_min", 0.0),
                gold_per_min=stats.get("gold_per_min", 0.0),
                kill_participation=stats.get("kill_participation", 0.0),
                deaths_pre_15=stats.get("deaths_pre_15", 0),
                deaths_post_15=stats.get("deaths_post_15", 0),
                solo_deaths=stats.get("solo_deaths", 0),
                repeat_deaths_same_enemy=stats.get("repeat_deaths_same_enemy", 0),
            )
            self.session.add(row)
            inserted += 1

        # Update ingestion metadata on player
        await self.session.execute(
            update(Player)
            .where(Player.puuid == puuid)
            .values(
                last_ingested_at=datetime.utcnow(),
                total_games_ingested=Player.total_games_ingested + inserted,
            )
        )
        logger.info(f"Inserted {inserted} new game records for {puuid}")

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------

    async def get_baseline(self, puuid: str) -> PlayerBaseline | None:
        """
        Fetch the computed baseline for a player.
        Returns None if player doesn't exist or baseline hasn't been computed yet.
        """
        result = await self.session.execute(
            select(PlayerBaseline)
            .join(Player, Player.id == PlayerBaseline.player_id)
            .where(Player.puuid == puuid)
        )
        return result.scalar_one_or_none()

    async def upsert_baseline(self, puuid: str, baseline: BaselineResult):
        """
        Insert or update the PlayerBaseline for a player.
        Uses PostgreSQL's ON CONFLICT DO UPDATE (upsert) for atomicity.
        """
        player = await self.get_player_by_puuid(puuid)
        if not player:
            logger.warning(f"Cannot upsert baseline — player {puuid} not found")
            return

        # Serialize change points for JSON storage
        change_points_json = [
            {
                "game_index": cp.game_index,
                "metric": cp.metric,
                "before_mean": cp.before_mean,
                "after_mean": cp.after_mean,
                "magnitude": cp.magnitude,
                "direction": cp.direction,
            }
            for cp in baseline.change_points
        ]

        # Determine last change point timestamp (approximate from game index)
        last_cp_at = datetime.utcnow() if baseline.change_points else None

        values = {
            "player_id": player.id,
            "games_used": baseline.games_analyzed,
            "lt_cs_per_min_median": baseline.cs_per_min_median,
            "lt_cs_per_min_iqr": baseline.cs_per_min_iqr,
            "lt_kill_participation_median": baseline.kill_participation_median,
            "lt_death_rate_median": baseline.death_rate_median,
            "lt_gold_per_min_median": baseline.gold_per_min_median,
            "lt_solo_death_rate_median": baseline.solo_death_rate_median,
            "change_points": change_points_json,
            "last_change_point_at": last_cp_at,
            "chronic_slump_detected": baseline.chronic_slump_detected,
            "computed_at": datetime.utcnow(),
        }

        # PostgreSQL upsert: insert, or update if player_id already has a baseline
        stmt = pg_insert(PlayerBaseline).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["player_id"],
            set_={k: v for k, v in values.items() if k != "player_id"},
        )
        await self.session.execute(stmt)
        logger.info(
            f"Baseline upserted for {puuid} — "
            f"{baseline.games_analyzed} games, "
            f"slump={baseline.chronic_slump_detected}, "
            f"change_points={len(baseline.change_points)}"
        )

    # -------------------------------------------------------------------------
    # Tilt events
    # -------------------------------------------------------------------------

    async def save_tilt_event(
        self,
        puuid: str,
        tilt_score: float,
        tilt_type: str,
        confidence: float,
        key_signals: list[str],
        exploit: str | None,
        match_id: int | None = None,
    ):
        """Log a tilt detection event for later analysis and ML training data."""
        player = await self.get_player_by_puuid(puuid)
        if not player:
            return

        event = TiltEvent(
            player_id=player.id,
            match_id=match_id,
            tilt_score=tilt_score,
            tilt_type=tilt_type,
            confidence=confidence,
            key_signals=key_signals,
            exploit_recommendation=exploit,
        )
        self.session.add(event)

    async def get_recent_game_stats(self, puuid: str, limit: int = 100) -> list[PlayerMatchStats]:
        """
        Fetch the N most recent game stats for a player, ordered newest first.
        Used by the baseline recomputation job.
        """
        result = await self.session.execute(
            select(PlayerMatchStats)
            .join(Player, Player.id == PlayerMatchStats.player_id)
            .where(Player.puuid == puuid)
            .order_by(PlayerMatchStats.game_start.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        # Return in chronological order (oldest first) for change point detection
        return list(reversed(rows))

    # -------------------------------------------------------------------------
    # Peer group baselines
    # -------------------------------------------------------------------------

    # When a tier's baselines aren't populated, fall back to the nearest tier
    # that IS likely to be in the DB.  Order matters: first match wins.
    _TIER_FALLBACKS: dict[str, list[tuple[str, str]]] = {
        # IRON is very close to Bronze — round up to Bronze IV
        "IRON":          [("BRONZE", "IV"), ("BRONZE", "III"), ("SILVER", "IV")],
        # GOLD is close to Silver — round down to Silver I
        "GOLD":          [("SILVER", "I"),  ("SILVER", "II"),  ("SILVER", "III")],
        # Higher tiers fall back down the ladder if not yet populated
        "PLATINUM":      [("GOLD", "I"),    ("SILVER", "I")],
        "EMERALD":       [("PLATINUM", "I"), ("GOLD", "I")],
        "DIAMOND":       [("EMERALD", "I"),  ("PLATINUM", "I")],
        "MASTER":        [("DIAMOND", "I"),  ("EMERALD", "I")],
        "GRANDMASTER":   [("MASTER", "I"),   ("DIAMOND", "I")],
        "CHALLENGER":    [("GRANDMASTER", "I"), ("MASTER", "I")],
    }

    async def get_peer_baseline(self, tier: str, division: str) -> PeerGroupBaseline | None:
        """
        Fetch the aggregate baseline for a given rank tier.
        If the exact tier/division isn't in the DB, automatically falls back to
        the nearest populated tier (e.g. IRON → BRONZE IV, GOLD → SILVER I).
        """
        candidates = [(tier.upper(), division.upper())] + self._TIER_FALLBACKS.get(tier.upper(), [])
        for t, d in candidates:
            result = await self.session.execute(
                select(PeerGroupBaseline).where(
                    PeerGroupBaseline.tier == t,
                    PeerGroupBaseline.division == d,
                )
            )
            row = result.scalar_one_or_none()
            if row is not None:
                if (t, d) != (tier.upper(), division.upper()):
                    logger.info(f"Peer baseline fallback: {tier} {division} → {t} {d}")
                return row
        return None

    async def log_tilt_prediction(self, entry: dict):
        """
        Store a post-game prediction evaluation record.
        entry keys: player_name, player_type, champion_name, peak_tilt_score, peak_tilt_type,
                    peak_signals, final_kills, final_deaths, final_assists, final_cs,
                    game_duration_min, final_cs_per_min, final_death_rate, final_kda,
                    peer_cs_per_min_median, peer_death_rate_median,
                    predicted_tilted, performed_poorly, verdict
        """
        row = TiltPredictionLog(**entry)
        self.session.add(row)

    async def upsert_peer_baseline(self, tier: str, division: str, stats: dict):
        """
        Insert or update peer group stats for a rank tier.
        stats dict keys: cs_per_min_median, cs_per_min_iqr, kill_participation_median,
                         death_rate_median, death_rate_iqr, gold_per_min_median,
                         solo_death_rate_median, win_rate_median, sample_size
        """
        stmt = pg_insert(PeerGroupBaseline).values(
            tier=tier.upper(),
            division=division.upper(),
            queue="RANKED_SOLO_5x5",
            computed_at=datetime.utcnow(),
            **stats,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["tier", "division"],
            set_={k: v for k, v in stats.items()},
        )
        await self.session.execute(stmt)
        logger.info(f"Peer group baseline upserted for {tier} {division} (n={stats.get('sample_size')})")
