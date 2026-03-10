"""
Celery tasks for the data ingestion pipeline.

Main tasks:
  ingest_player   → fetch last N games for a player, store in DB, compute baseline
  ingest_enemies  → triggered at game start: ingest all 5 enemies in parallel

These run asynchronously so the WebSocket endpoint can start immediately
while data is being fetched in the background.
"""

import asyncio
import logging
from workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def ingest_player(self, puuid: str, platform: str = "euw1", max_games: int = 100):
    """
    Fetch and store the last N ranked games for a player, then compute their baseline.

    Steps:
      1. Fetch match IDs from Riot API
      2. For each match: fetch match data + timeline
      3. Parse into PlayerMatchStats rows
      4. Compute baseline via change point detection
      5. Store everything in PostgreSQL

    Called:
      - Manually to pre-populate data for known players
      - Automatically when a new player is encountered (lazy ingestion)
    """
    try:
        run_async(_ingest_player_async(puuid, platform, max_games))
    except Exception as exc:
        logger.exception(f"ingest_player failed for {puuid}: {exc}")
        raise self.retry(exc=exc)


@celery_app.task
def ingest_enemies(enemy_puuids: list[str], platform: str = "euw1"):
    """
    Ingest all 5 enemies in parallel at game start.

    Spawns one ingest_player task per enemy.
    Each task runs independently — if one fails the others continue.

    Called by the WebSocket handler when a game is detected,
    so baselines are ready (or refreshed) before the game reaches 10 minutes.
    """
    logger.info(f"Triggering ingestion for {len(enemy_puuids)} enemies")
    for puuid in enemy_puuids:
        ingest_player.delay(puuid, platform)


async def _ingest_player_async(puuid: str, platform: str, max_games: int):
    """Async implementation of player ingestion."""
    from data.riot.client import RiotClient, PLATFORM_TO_REGION
    from data.db.repository import PlayerRepository
    from data.db.session import get_session
    from ml.features.change_point import compute_player_baseline

    region = PLATFORM_TO_REGION.get(platform, "europe")

    async with RiotClient() as riot:
        # Fetch match IDs
        match_ids = await riot.get_match_ids(puuid, count=max_games, region=region)
        logger.info(f"Fetched {len(match_ids)} match IDs for {puuid}")

        if not match_ids:
            logger.warning(f"No matches found for {puuid}")
            return

        # Fetch and parse each match
        game_stats = []
        for match_id in match_ids:
            try:
                match = await riot.get_match(match_id, region=region)
                timeline = await riot.get_match_timeline(match_id, region=region)
                if match and timeline:
                    stats = _parse_match(match, timeline, puuid)
                    if stats:
                        game_stats.append(stats)
            except Exception as e:
                logger.warning(f"Failed to parse match {match_id}: {e}")
                continue

        logger.info(f"Parsed {len(game_stats)} games for {puuid}")

        # Compute baseline
        baseline = compute_player_baseline(game_stats)

        # Store in DB
        async with get_session() as session:
            repo = PlayerRepository(session)
            await repo.upsert_player_stats(puuid, game_stats)
            if baseline:
                await repo.upsert_baseline(puuid, baseline)
            await session.commit()

        logger.info(f"Ingestion complete for {puuid} — baseline computed: {baseline is not None}")


def _parse_match(match: dict, timeline: dict, target_puuid: str) -> dict | None:
    """
    Extract per-player stats from raw Riot match + timeline JSON.

    Returns a flat dict of features for one player in one game,
    or None if the player is not found in this match.
    """
    info = match.get("info", {})
    participants = info.get("participants", [])

    # Find our target player in this match
    player_data = next(
        (p for p in participants if p.get("puuid") == target_puuid), None
    )
    if not player_data:
        return None

    game_duration_sec = info.get("gameDuration", 1)
    game_duration_min = game_duration_sec / 60

    kills = player_data.get("kills", 0)
    deaths = player_data.get("deaths", 0)
    assists = player_data.get("assists", 0)
    cs = player_data.get("totalMinionsKilled", 0) + player_data.get("neutralMinionsKilled", 0)

    # Team kills for kill participation
    team_id = player_data.get("teamId")
    team_kills = sum(
        p.get("kills", 0) for p in participants if p.get("teamId") == team_id
    ) or 1  # avoid division by zero

    # Extract behavioral signals from timeline
    timeline_signals = _extract_timeline_signals(timeline, target_puuid, participants)

    return {
        "cs_per_min": cs / game_duration_min,
        "kill_participation": (kills + assists) / team_kills,
        "deaths": deaths,
        "game_duration_min": game_duration_min,
        "gold_per_min": player_data.get("goldEarned", 0) / game_duration_min,
        "solo_deaths": timeline_signals.get("solo_deaths", 0),
        "repeat_deaths_same_enemy": timeline_signals.get("repeat_deaths_same_enemy", 0),
        "won": player_data.get("win", False),
    }


def _extract_timeline_signals(timeline: dict, target_puuid: str, participants: list) -> dict:
    """
    Parse the raw timeline JSON to extract behavioral signals.

    The timeline contains frames (every 60s) with:
      - participantFrames: gold, XP, CS, position for each player at that minute
      - events: kills, item purchases, ward placements, etc.

    We extract: solo deaths, repeat deaths to same enemy.
    """
    # Build puuid → participantId map (Riot uses 1-indexed participant IDs in timeline)
    puuid_to_pid = {p.get("puuid"): p.get("participantId") for p in participants}
    target_pid = puuid_to_pid.get(target_puuid)

    if not target_pid:
        return {}

    frames = timeline.get("info", {}).get("frames", [])
    kill_events = []
    for frame in frames:
        for event in frame.get("events", []):
            if event.get("type") == "CHAMPION_KILL":
                kill_events.append(event)

    # Solo deaths: died and no assist from teammate within ~500 units
    solo_deaths = 0
    for event in kill_events:
        if event.get("victimId") == target_pid:
            assisters = event.get("assistingParticipantIds", [])
            if not assisters:
                solo_deaths += 1

    # Repeat deaths to same enemy
    from collections import Counter
    killer_ids = [
        e.get("killerId") for e in kill_events
        if e.get("victimId") == target_pid
    ]
    max_repeat = Counter(killer_ids).most_common(1)[0][1] if killer_ids else 0

    return {
        "solo_deaths": solo_deaths,
        "repeat_deaths_same_enemy": max_repeat,
    }
