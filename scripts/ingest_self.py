"""
One-time (and periodic) ingestion of your own match history.

Fetches your last 20 ranked games from Riot, computes your personal baseline
using change point detection, and stores it in the DB.

Run this BEFORE starting a gaming session so the backend has your baseline ready.
Re-run it after ~10 games to keep the baseline fresh.

Usage:
    conda run -n tilt-radar python scripts/ingest_self.py

Takes ~2-3 minutes (API rate limiting). You only need to do it once before each session.
"""

import asyncio
import sys
import logging
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GAME_NAME = "MarreDesNoobsNul"
TAG_LINE  = "007"
PLATFORM  = "euw1"
MAX_GAMES = 20


async def main():
    from data.riot.client import RiotClient, PLATFORM_TO_REGION
    from data.db.repository import PlayerRepository
    from data.db.session import get_session
    from ml.features.change_point import compute_player_baseline
    from workers.tasks import _parse_match

    region = PLATFORM_TO_REGION.get(PLATFORM, "europe")

    async with RiotClient() as riot:
        # Resolve PUUID
        puuid = await riot.get_puuid(GAME_NAME, TAG_LINE)
        if not puuid:
            logger.error(f"Could not resolve PUUID for {GAME_NAME}#{TAG_LINE}")
            return
        logger.info(f"Resolved PUUID: {puuid[:20]}...")

        # Check rank
        rank = await riot.get_rank(puuid)
        if rank:
            logger.info(f"Rank: {rank['tier']} {rank['rank']} — {rank['leaguePoints']} LP")
        else:
            logger.info("Unranked")

        # Fetch last N match IDs
        logger.info(f"Fetching last {MAX_GAMES} match IDs from Riot API...")
        match_ids = await riot.get_match_ids(puuid, count=MAX_GAMES, region=region)
        logger.info(f"Found {len(match_ids)} matches")

        if not match_ids:
            logger.error("No matches found — nothing to ingest")
            return

        # Parse each match
        game_stats = []
        for i, match_id in enumerate(match_ids, 1):
            try:
                logger.info(f"  [{i}/{len(match_ids)}] Parsing {match_id}...")
                match    = await riot.get_match(match_id, region=region)
                timeline = await riot.get_match_timeline(match_id, region=region)
                if match and timeline:
                    stats = _parse_match(match, timeline, puuid)
                    if stats:
                        stats["match_id"] = match_id   # needed for DB deduplication
                        game_stats.append(stats)
            except Exception as e:
                logger.warning(f"  Failed {match_id}: {e}")
                continue

        logger.info(f"Parsed {len(game_stats)} valid games")

        if len(game_stats) < 5:
            logger.error("Not enough games to compute a reliable baseline (need at least 5)")
            return

        # Compute baseline via change point detection
        logger.info("Computing personal baseline (change point detection)...")
        baseline = compute_player_baseline(game_stats)

        if not baseline:
            logger.error("Baseline computation failed — not enough data")
            return

        logger.info(
            f"Baseline computed from {baseline.games_analyzed} games:\n"
            f"  CS/min median:      {baseline.cs_per_min_median:.2f}  (IQR {baseline.cs_per_min_iqr:.2f})\n"
            f"  Kill participation: {baseline.kill_participation_median:.2%}\n"
            f"  Death rate:         {baseline.death_rate_median:.3f}/min\n"
            f"  Chronic slump:      {baseline.chronic_slump_detected}\n"
            f"  Change points found:{len(baseline.change_points)}"
        )

        # Store in DB
        async with get_session() as session:
            repo = PlayerRepository(session)

            # Ensure player row exists
            await repo.get_or_create_player(puuid, GAME_NAME, TAG_LINE, PLATFORM)

            # Store individual game stats
            await repo.upsert_player_stats(puuid, game_stats)

            # Store computed baseline
            await repo.upsert_baseline(puuid, baseline)

            await session.commit()

        logger.info("Done — personal baseline saved to DB.")
        logger.info("The backend will now use your personal history for self-tilt detection.")

        if baseline.chronic_slump_detected:
            logger.warning(
                "CHRONIC SLUMP DETECTED in your recent games — "
                "the baseline reflects your latest stable period, not your all-time average."
            )


if __name__ == "__main__":
    asyncio.run(main())
