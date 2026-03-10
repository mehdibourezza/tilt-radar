"""
Populate peer group baselines for each rank tier.

Fetches a sample of ranked players per tier, pulls their last 20 games,
computes aggregate (median) stats, and stores them in the peer_group_baselines table.

This gives the inference engine a reference point when scoring an enemy
who has no personal history in our DB:
  "This Gold II player's CS/min is 4.1 — the Gold II median is 6.2 — that's a drop signal"

Usage:
    conda run -n tilt-radar python scripts/populate_peer_groups.py --tiers GOLD PLATINUM
    conda run -n tilt-radar python scripts/populate_peer_groups.py --tiers ALL

Runtime: ~5-10 min per tier (API rate limiting). Run once, refresh monthly.

With a dev API key (100 req/2min), we sample 15 players per tier to stay safe.
With a production key, increase --sample to 50+.
"""

import asyncio
import sys
import argparse
import numpy as np
import logging
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
DIVISIONS = ["I", "II", "III", "IV"]
GAMES_PER_PLAYER = 20
DEFAULT_SAMPLE = 15   # players per tier — safe for dev key


async def fetch_players_for_tier(riot, tier: str, division: str, count: int) -> list[str]:
    """Fetch PUUIDs for a given rank tier using the league endpoint."""
    platform = "euw1"
    url = (
        f"https://{platform}.api.riotgames.com/lol/league/v4/entries"
        f"/RANKED_SOLO_5x5/{tier}/{division}"
    )
    data = await riot._request(url, params={"page": 1})
    if not data:
        return []
    # League entries now include puuid directly — no summoner lookup needed
    puuids = [
        entry["puuid"]
        for entry in data[:count]
        if entry.get("puuid")
    ]
    logger.info(f"Found {len(puuids)} players for {tier} {division}")
    return puuids


async def compute_tier_stats(riot, puuids: list[str], tier: str, division: str) -> dict | None:
    """Fetch last N games for each player and compute aggregate stats."""
    from workers.tasks import _parse_match

    all_stats = []
    for i, puuid in enumerate(puuids):
        logger.info(f"  [{i+1}/{len(puuids)}] Fetching {GAMES_PER_PLAYER} games for {puuid[:12]}...")
        try:
            match_ids = await riot.get_match_ids(puuid, count=GAMES_PER_PLAYER, region="europe")
            for match_id in match_ids[:GAMES_PER_PLAYER]:
                match = await riot.get_match(match_id, region="europe")
                timeline = await riot.get_match_timeline(match_id, region="europe")
                if match and timeline:
                    stats = _parse_match(match, timeline, puuid)
                    if stats:
                        all_stats.append(stats)
        except Exception as e:
            logger.warning(f"  Failed for {puuid[:12]}: {e}")
            continue

    if len(all_stats) < 10:
        logger.warning(f"Not enough data for {tier} {division} ({len(all_stats)} games) — skipping")
        return None

    cs = np.array([s["cs_per_min"] for s in all_stats])
    kp = np.array([s["kill_participation"] for s in all_stats])
    dr = np.array([s["deaths"] / max(s["game_duration_min"], 1) for s in all_stats])
    gold = np.array([s["gold_per_min"] for s in all_stats])
    solo_dr = np.array([s.get("solo_deaths", 0) / max(s["game_duration_min"], 1) for s in all_stats])
    wins = np.array([1.0 if s.get("won") else 0.0 for s in all_stats])

    logger.info(
        f"  {tier} {division}: {len(all_stats)} games from {len(puuids)} players — "
        f"CS/min median={np.median(cs):.2f}, KP median={np.median(kp):.2f}"
    )

    return {
        "sample_size": len(puuids),
        "cs_per_min_median": float(np.median(cs)),
        "cs_per_min_iqr": float(np.percentile(cs, 75) - np.percentile(cs, 25)),
        "kill_participation_median": float(np.median(kp)),
        "death_rate_median": float(np.median(dr)),
        "death_rate_iqr": float(np.percentile(dr, 75) - np.percentile(dr, 25)),
        "gold_per_min_median": float(np.median(gold)),
        "solo_death_rate_median": float(np.median(solo_dr)),
        "win_rate_median": float(np.median(wins)),
    }


async def main(tiers: list[str], sample: int):
    from data.riot.client import RiotClient
    from data.db.session import get_session
    from data.db.repository import PlayerRepository

    async with RiotClient() as riot:
        async with get_session() as session:
            repo = PlayerRepository(session)

            for tier in tiers:
                for division in DIVISIONS:
                    # Challenger/GM/Master have no divisions — skip IV/III/II
                    if tier in ("CHALLENGER", "GRANDMASTER", "MASTER") and division != "I":
                        continue

                    logger.info(f"\n{'='*40}")
                    logger.info(f"Processing {tier} {division} (sample={sample} players)")
                    logger.info(f"{'='*40}")

                    puuids = await fetch_players_for_tier(riot, tier, division, sample)
                    if not puuids:
                        logger.warning(f"No players found for {tier} {division}")
                        continue

                    stats = await compute_tier_stats(riot, puuids, tier, division)
                    if stats:
                        await repo.upsert_peer_baseline(tier, division, stats)
                        await session.commit()
                        logger.info(f"Saved peer baseline for {tier} {division}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiers", nargs="+",
        default=["GOLD"],
        help="Tiers to populate (e.g. GOLD PLATINUM) or ALL"
    )
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="Players to sample per tier")
    args = parser.parse_args()

    tiers = ALL_TIERS if "ALL" in args.tiers else [t.upper() for t in args.tiers]
    logger.info(f"Populating peer groups for: {tiers}")
    asyncio.run(main(tiers, args.sample))
