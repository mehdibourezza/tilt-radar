"""
Diverse Player Match Scraper — fixes ELO bias in training data.

The core problem: all TiltPredictionLog records come from one player's live
games (one ELO bracket, one champion pool, one play style). The model learns
what tilt looks like FOR THAT PLAYER, not in general.

This scraper pulls historical match data from the Riot Match v5 API across
multiple ELO tiers and produces:

  1. Training records — rows compatible with TiltPredictionLog, labeling
     each player's game as performed_poorly=True/False based on final stats.
     These are written to the DB to expand the training set.

  2. ItemRegistry data — item build frequencies per (champion, role) pair,
     used to compute expected builds and build_distance features.

=== LABEL COMPUTATION (performed_poorly) ===

Without running the live engine, we can't compute peak_tilt_score.
But we CAN compute performed_poorly from final match stats:

  performed_poorly = True IF:
    - KDA < 1.0  (died more than killed+assisted)  AND
    - at least one of:
        - deaths > 8
        - cs_per_min < 3.0 (for non-support roles)
        - death_rate > 0.50 per minute for the last third of the game

This is an approximation. It won't catch "tilted but performed okay" cases
(the main label flaw discussed in KNOWN_LIMITATIONS.md), but it's consistent
with how ws.py defines performed_poorly for live games.

=== SIGNAL APPROXIMATION ===

For the 12 binary signals, we approximate from match timeline data:
  - repeat_deaths_same_enemy: check killer_id in kill events
  - death_acceleration: compare early vs late deaths by timestamp
  - cs_drop_vs_baseline: compare player CS to tier average
  - early_death_cluster: deaths before 10 min
  - item_sell: events of type ITEM_SOLD in timeline

Other signals (kp_drop, vision_low, objective_absence) require more timeline
detail and are set to None in scraped records (they won't be used as features,
they'll still be useful as metadata for the calibration study).

=== RATE LIMITS ===

Riot development key: 20 requests/second, 100/2 minutes.
This scraper uses 0.07s sleep between calls (~14 req/s) with exponential
backoff on 429 responses.

Riot production key: up to 500 req/second — adjust SLEEP_BETWEEN_CALLS.

=== USAGE ===

  from ml.data.scraper import MatchScraper

  scraper = MatchScraper(api_key="RGAPI-...", region="europe")

  # Scrape 500 games from Gold–Diamond spread across roles
  records = scraper.scrape(
      tiers=["GOLD", "PLATINUM", "DIAMOND"],
      games_per_tier=200,
      update_item_registry=True,
  )

  # Save to DB (adds to TiltPredictionLog)
  scraper.save_to_db(records)

  # Or just get the records for offline analysis
  import json
  with open("experiments/scraped_records.json", "w") as f:
      json.dump(records, f)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Riot API rate limiting
SLEEP_BETWEEN_CALLS = 0.07     # ~14 req/s — under the 20/s dev key limit
MAX_RETRIES = 4

# performed_poorly thresholds (matches ws.py logic)
MIN_KDA_THRESHOLD    = 1.0
MAX_DEATHS_THRESHOLD = 8
MAX_DEATH_RATE_POOR  = 0.50    # deaths per minute in last third of game
MAX_CS_POOR_NON_SUP  = 3.0    # cs/min (non-support only)


class MatchScraper:
    """
    Scrapes match data from Riot API to produce diverse training records.

    Also updates an ItemRegistry with item build frequencies per champion+role.

    Args:
        api_key: Riot API key (RGAPI-...)
        region:  Riot regional routing value: "europe", "americas", "asia"
    """

    def __init__(self, api_key: str, region: str = "europe") -> None:
        self.api_key = api_key
        self.region  = region
        self._platform = _region_to_platform(region)

    def scrape(
        self,
        tiers: list[str] | None = None,
        games_per_tier: int = 200,
        update_item_registry: bool = True,
        item_registry_path: str | Path = "experiments/item_registry.json",
    ) -> list[dict]:
        """
        Scrape match data from diverse ELO tiers.

        Args:
            tiers:                ELO tiers to sample from. Defaults to a spread from
                                  low to high: ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]
            games_per_tier:       Number of games to analyze per tier.
            update_item_registry: If True, update ItemRegistry with item build data.
            item_registry_path:   Path to load/save ItemRegistry JSON.

        Returns:
            List of record dicts compatible with TiltPredictionLog.
        """
        tiers = tiers or ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]

        # Item frequency tracking: (champion, role) -> {item_name: count}
        item_counts:     dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        win_game_counts: dict[tuple, int]             = defaultdict(int)

        all_records: list[dict] = []

        for tier in tiers:
            logger.info(f"Scraping tier {tier} ({games_per_tier} games)...")
            summoner_ids = self._get_summoner_ids(tier, count=50)
            games_scraped = 0

            for summoner_id in summoner_ids:
                if games_scraped >= games_per_tier:
                    break
                try:
                    puuid = self._get_puuid(summoner_id)
                    if not puuid:
                        continue
                    match_ids = self._get_match_ids(puuid, count=10)

                    for match_id in match_ids:
                        if games_scraped >= games_per_tier:
                            break
                        try:
                            records = self._process_match(
                                match_id, tier, item_counts, win_game_counts
                            )
                            all_records.extend(records)
                            games_scraped += 1
                            logger.debug(f"  {match_id}: {len(records)} participants scraped")
                        except Exception as e:
                            logger.debug(f"  Skipped match {match_id}: {e}")
                            continue

                except Exception as e:
                    logger.debug(f"Skipped summoner {summoner_id}: {e}")
                    continue

            logger.info(f"  {tier}: {games_scraped} games scraped, {sum(1 for r in all_records if r.get('elo_tier') == tier)} participant records")

        if update_item_registry:
            self._update_item_registry(item_counts, win_game_counts, item_registry_path)

        logger.info(f"Scraping complete: {len(all_records)} total participant records")
        return all_records

    def save_to_db(self, records: list[dict]) -> int:
        """
        Save scraped records to the TiltPredictionLog table.

        Returns the number of records inserted.
        """
        import asyncio
        return asyncio.run(self._save_to_db_async(records))

    async def _save_to_db_async(self, records: list[dict]) -> int:
        from data.db.session import get_session
        from data.db.models import TiltPredictionLog
        from sqlalchemy import select

        inserted = 0
        async for session in get_session():
            for r in records:
                # Skip if no performed_poorly label
                if r.get("performed_poorly") is None:
                    continue

                log = TiltPredictionLog(
                    player_name       = r.get("player_name", "scraped"),
                    player_type       = "scraped",
                    champion_name     = r.get("champion_name", ""),
                    peak_tilt_score   = 0.0,   # unknown for scraped records
                    peak_tilt_type    = "scraped",
                    peak_signals      = json.dumps(r.get("active_signals", [])),
                    final_cs_per_min  = r.get("final_cs_per_min"),
                    final_death_rate  = r.get("final_death_rate"),
                    final_kda         = r.get("final_kda"),
                    game_duration_min = r.get("game_duration_min"),
                    performed_poorly  = r.get("performed_poorly", False),
                    verdict           = "scraped",
                    recorded_at       = datetime.fromisoformat(r["recorded_at"]) if r.get("recorded_at") else datetime.now(timezone.utc),
                    role              = r.get("role"),
                )
                session.add(log)
                inserted += 1
            await session.commit()

        logger.info(f"Saved {inserted} scraped records to TiltPredictionLog")
        return inserted

    # ── Match processing ───────────────────────────────────────────────────────

    def _process_match(
        self,
        match_id: str,
        tier: str,
        item_counts: dict,
        win_game_counts: dict,
    ) -> list[dict]:
        """
        Fetch and process one match. Returns participant records + updates item counts.
        """
        data       = self._api_get(
            f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        )
        match_data = json.loads(data)

        info         = match_data["info"]
        game_dur_sec = info.get("gameDuration", 0)
        game_dur_min = game_dur_sec / 60.0

        if game_dur_min < 15:
            return []  # remake / too short to be meaningful

        records = []
        for p in info["participants"]:
            record = self._participant_to_record(p, match_id, tier, game_dur_min)
            records.append(record)

            # Update item frequency for winning players
            role    = p.get("teamPosition", "").upper()
            champ   = p.get("championName", "")
            key     = (champ, role)
            if p.get("win") and champ and role:
                win_game_counts[key] += 1
                for slot in range(7):
                    item_id = p.get(f"item{slot}", 0)
                    if item_id > 0:
                        item_name = str(item_id)  # use ID as name until registry loads names
                        item_counts[key][item_name] += 1

        return records

    def _participant_to_record(
        self, p: dict, match_id: str, tier: str, game_dur_min: float
    ) -> dict:
        """Convert a participant dict from the Match API to a TiltPredictionLog-compatible record."""
        kills   = p.get("kills", 0)
        deaths  = p.get("deaths", 0)
        assists = p.get("assists", 0)
        cs      = p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)
        role    = p.get("teamPosition", "").upper()
        champ   = p.get("championName", "")

        kda           = (kills + assists) / max(deaths, 1)
        cs_per_min    = cs / max(game_dur_min, 1)
        death_rate    = deaths / max(game_dur_min, 1)
        is_support    = role == "UTILITY"

        # performed_poorly heuristic (approximation — see module docstring)
        performed_poorly = (
            kda < MIN_KDA_THRESHOLD
            and (
                deaths > MAX_DEATHS_THRESHOLD
                or (not is_support and cs_per_min < MAX_CS_POOR_NON_SUP)
                or death_rate > MAX_DEATH_RATE_POOR
            )
        )

        # Approximate active signals from available data
        active_signals: list[str] = []
        if deaths > MAX_DEATHS_THRESHOLD:
            active_signals.append("deaths_before_10")
        if not is_support and cs_per_min < 4.0:
            active_signals.append("cs_down")
        if kda < 1.0:
            active_signals.append("kp_dropped")

        return {
            "player_name":      p.get("summonerName", ""),
            "champion_name":    champ,
            "role":             role,
            "elo_tier":         tier,
            "match_id":         match_id,
            "final_kda":        round(kda, 3),
            "final_cs_per_min": round(cs_per_min, 2),
            "final_death_rate": round(death_rate, 3),
            "game_duration_min": round(game_dur_min, 1),
            "performed_poorly": performed_poorly,
            "active_signals":   active_signals,
            "recorded_at":      datetime.now(timezone.utc).isoformat(),
        }

    # ── ItemRegistry update ────────────────────────────────────────────────────

    def _update_item_registry(
        self,
        item_counts: dict,
        win_game_counts: dict,
        path: str | Path,
        frequency_threshold: float = 0.40,
    ) -> None:
        """
        Write item frequency data to the ItemRegistry JSON.

        Items appearing in ≥ frequency_threshold of winning games for a
        champion+role are added to the registry as expected items.
        """
        from ml.data.item_registry import ItemRegistry

        path = Path(path)
        if path.exists():
            registry = ItemRegistry.load(path)
        else:
            registry = ItemRegistry()

        # Load item name lookup (Community Dragon)
        registry._load_item_names()
        id_to_name = registry._item_id_to_name

        for (champ, role), counts in item_counts.items():
            win_games = win_game_counts.get((champ, role), 0)
            if win_games < 10:
                continue  # not enough data

            expected = [
                id_to_name.get(int(item_id_str), item_id_str)
                for item_id_str, count in counts.items()
                if count / win_games >= frequency_threshold
                and int(item_id_str) not in _EXCLUDED_ITEM_IDS
            ]
            if expected:
                registry._builds[f"{champ}|{role}"] = sorted(
                    expected, key=lambda x: -counts.get(str(next(
                        (k for k, v in id_to_name.items() if v == x), 0
                    )), 0)
                )

        registry.save(path)
        logger.info(f"ItemRegistry updated: {len(registry._builds)} champion+role entries")

    # ── Riot API helpers ───────────────────────────────────────────────────────

    def _get_summoner_ids(self, tier: str, count: int = 50) -> list[str]:
        data    = self._api_get(
            f"https://{self._platform}.api.riotgames.com/lol/league/v4/entries/"
            f"RANKED_SOLO_5x5/{tier}/I?page=1"
        )
        entries = json.loads(data)
        return [e["summonerId"] for e in entries[:count]]

    def _get_puuid(self, summoner_id: str) -> str | None:
        try:
            data = self._api_get(
                f"https://{self._platform}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
            )
            return json.loads(data).get("puuid")
        except Exception:
            return None

    def _get_match_ids(self, puuid: str, count: int = 10) -> list[str]:
        data = self._api_get(
            f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/"
            f"by-puuid/{puuid}/ids?queue=420&count={count}"
        )
        return json.loads(data)

    def _api_get(self, url: str) -> bytes:
        """Rate-limited GET with exponential backoff on 429."""
        req = urllib.request.Request(url, headers={"X-Riot-Token": self.api_key})
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(SLEEP_BETWEEN_CALLS)
                with urllib.request.urlopen(req, timeout=12) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    retry_after = int(e.headers.get("Retry-After", "5"))
                    wait = retry_after + (2 ** attempt)
                    logger.warning(f"Rate limited (429). Waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                elif e.code in (404, 403):
                    raise
                else:
                    raise
        raise RuntimeError(f"All {MAX_RETRIES} retries failed for: {url}")


# ── Constants ──────────────────────────────────────────────────────────────────

_EXCLUDED_ITEM_IDS = {
    2003, 2031, 2055, 3364, 3363, 3340, 2139, 2138, 2140,
}


def _region_to_platform(region: str) -> str:
    return {"americas": "na1", "europe": "euw1", "asia": "kr"}.get(region.lower(), "euw1")
