"""
Empirical Item Registry — data-driven "expected build" per champion+role.

Instead of hard-coding which items are correct (which goes stale every patch),
we compute it from real match data:

  1. Pull the last N games for a sample of players across ELO tiers
     using the Riot Match v5 API.
  2. For each (champion, role, win) triple, record which items they finished with.
  3. An item is "expected" for a champion+role if it appears in ≥ FREQUENCY_THRESHOLD
     fraction of winning games (default 40%).
  4. Persist the result as a JSON file so we don't re-query every game.

=== WHY NOT JUST USE A TIER LIST SITE? ===

Tier list sites (u.gg, op.gg) show aggregated stats, but we can't query them
programmatically without scraping. Riot's own Match API is the authoritative,
stable, rate-limited source we already have access to.

=== PATCH AWARENESS ===

Each computed registry is tagged with the patch version it was computed from.
The engine uses the registry that matches the current patch, falling back to
the most recent available one if the current patch isn't cached yet.

=== USAGE ===

  # Build the registry (run once per patch, or when meta shifts):
  registry = ItemRegistry()
  registry.build(
      champion="Jinx",
      role="BOTTOM",
      sample_size=200,      # games to analyze per champion+role
      tier_mix=["GOLD", "PLATINUM", "DIAMOND"],
  )
  registry.save("experiments/item_registry.json")

  # In engine.py / feature_extractor.py:
  registry = ItemRegistry.load("experiments/item_registry.json")
  expected = registry.get_expected_items("Jinx", "BOTTOM", patch="14.21")
  wrong_build = not any(item in expected for item in player_current_items)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# An item is "expected" if it appears in this fraction of winning games
FREQUENCY_THRESHOLD = 0.40

# Riot item IDs that are consumables / non-build components — exclude from analysis
EXCLUDED_ITEM_IDS = {
    2003,   # Health Potion
    2031,   # Refillable Potion
    2055,   # Control Ward
    3364,   # Oracle Lens (trinket)
    3363,   # Farsight Alteration (trinket)
    3340,   # Stealth Ward (trinket)
    2139,   # Elixir of Sorcery
    2138,   # Elixir of Iron
    2140,   # Elixir of Wrath
}


class ItemRegistry:
    """
    Stores and queries empirical expected item builds per champion+role.

    Internal structure:
        {
            "patch": "14.21",
            "builds": {
                "Jinx|BOTTOM": ["Kraken Slayer", "Runaan's Hurricane", ...],
                "Lux|UTILITY": ["Shard of True Ice", "Zhonya's Hourglass", ...],
                ...
            }
        }

    Item names are used (not IDs) for human readability and cross-version stability.
    IDs are resolved via the Community Dragon item data (fetched once, cached).
    """

    def __init__(self) -> None:
        self._patch: str = "unknown"
        # champion|role -> list of expected item names
        self._builds: dict[str, list[str]] = {}
        # id -> name mapping, loaded from Community Dragon
        self._item_id_to_name: dict[int, str] = {}

    # ── Query interface ───────────────────────────────────────────────────────

    def get_expected_items(
        self,
        champion: str,
        role: str,
        patch: str | None = None,
    ) -> list[str]:
        """
        Return the list of expected items for a champion+role combination.

        Args:
            champion: Champion name, e.g. "Jinx"
            role:     Player role: TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
            patch:    Ignored for now (single-patch registry). Reserved for
                      future multi-patch support.

        Returns:
            List of item names (empty if champion+role not in registry).
        """
        key = f"{champion}|{role}"
        return self._builds.get(key, [])

    def build_distance(
        self,
        champion: str,
        role: str,
        current_items: list[str],
        min_completed_items: int = 2,
    ) -> float:
        """
        Compute continuous build distance: fraction of expected items the player is MISSING.

        Returns:
            0.0 = player has all expected items (perfect build)
            1.0 = player has none of the expected items (completely wrong build)
            0.0 = not enough items yet OR no registry entry for this champion+role

        This is strictly more informative than the binary is_wrong_build():
        a player with 1 out of 4 expected items has distance 0.75 (progressing
        but wrong), while a player with 0 out of 4 has distance 1.0.
        """
        expected = self.get_expected_items(champion, role)
        if not expected:
            return 0.0

        completed = [item for item in current_items if item and item not in {
            "Health Potion", "Refillable Potion", "Control Ward",
            "Oracle Lens", "Stealth Ward", "Farsight Alteration",
            "Elixir of Sorcery", "Elixir of Iron", "Elixir of Wrath",
        }]

        if len(completed) < min_completed_items:
            return 0.0

        matched = sum(1 for item in completed if item in expected)
        return float(1.0 - matched / len(expected))

    def is_wrong_build(
        self,
        champion: str,
        role: str,
        current_items: list[str],
        min_completed_items: int = 2,
    ) -> bool:
        """
        Returns True if the player has ≥ min_completed_items and none of their
        completed items match the expected build for this champion+role.

        Args:
            champion:            e.g. "Jinx"
            role:                e.g. "BOTTOM"
            current_items:       list of item names the player currently holds
            min_completed_items: only fire the signal if the player has built
                                 enough items (avoids false positives early game)
        """
        expected = self.get_expected_items(champion, role)
        if not expected:
            return False  # no data for this champion+role

        # Filter out consumables and empty slots from current items
        completed = [item for item in current_items if item and item not in {
            "Health Potion", "Refillable Potion", "Control Ward",
            "Oracle Lens", "Stealth Ward", "Farsight Alteration",
            "Elixir of Sorcery", "Elixir of Iron", "Elixir of Wrath",
        }]

        if len(completed) < min_completed_items:
            return False  # too early to judge

        return not any(item in expected for item in completed)

    # ── Build from Riot API ───────────────────────────────────────────────────

    def build(
        self,
        champion: str,
        role: str,
        api_key: str,
        region: str = "europe",
        sample_size: int = 200,
        tier_mix: list[str] | None = None,
        patch: str | None = None,
    ) -> None:
        """
        Compute the expected item build for one champion+role by sampling
        real games from the Riot Match v5 API.

        Args:
            champion:    Champion name, e.g. "Jinx"
            role:        Role string, e.g. "BOTTOM"
            api_key:     Riot API key (RGAPI-...)
            region:      Riot regional routing: "europe", "americas", "asia"
            sample_size: Number of games to analyze
            tier_mix:    ELO tiers to sample from, e.g. ["GOLD", "DIAMOND"]
            patch:       Current patch string (auto-detected if None)
        """
        import urllib.request

        tier_mix = tier_mix or ["GOLD", "PLATINUM", "DIAMOND"]

        logger.info(
            f"Building item registry for {champion}+{role} "
            f"from {sample_size} games across {tier_mix}..."
        )

        # Load item name lookup
        self._load_item_names()

        # Collect match IDs from players in specified tiers
        match_ids = self._collect_match_ids(
            champion, role, api_key, region, tier_mix, sample_size
        )

        if not match_ids:
            logger.warning(f"No match IDs found for {champion}+{role}. Registry not updated.")
            return

        # Count item frequencies across winning games
        item_counts: dict[str, int] = defaultdict(int)
        win_games = 0

        for match_id in match_ids:
            try:
                participant = self._fetch_participant(
                    match_id, champion, role, api_key, region
                )
                if participant is None:
                    continue
                if not participant.get("win", False):
                    continue

                win_games += 1
                for slot_key in [f"item{i}" for i in range(7)]:
                    item_id = participant.get(slot_key, 0)
                    if item_id and item_id not in EXCLUDED_ITEM_IDS:
                        item_name = self._item_id_to_name.get(item_id)
                        if item_name:
                            item_counts[item_name] += 1

            except Exception as e:
                logger.debug(f"Skipped match {match_id}: {e}")
                continue

        if win_games == 0:
            logger.warning(f"No winning games found for {champion}+{role}")
            return

        # Items present in ≥ FREQUENCY_THRESHOLD of winning games
        expected = [
            item for item, count in item_counts.items()
            if count / win_games >= FREQUENCY_THRESHOLD
        ]
        expected.sort(key=lambda x: -item_counts[x])  # sort by frequency desc

        key = f"{champion}|{role}"
        self._builds[key] = expected
        if patch:
            self._patch = patch

        logger.info(
            f"Registry for {champion}+{role}: {len(expected)} expected items "
            f"(from {win_games} winning games out of {len(match_ids)} analyzed). "
            f"Items: {expected}"
        )

    def build_all(
        self,
        champions_roles: list[tuple[str, str]],
        api_key: str,
        region: str = "europe",
        sample_size: int = 200,
        tier_mix: list[str] | None = None,
    ) -> None:
        """
        Build registry for multiple champion+role pairs in sequence.

        Args:
            champions_roles: list of (champion, role) tuples
            ...same other args as build()
        """
        for champion, role in champions_roles:
            try:
                self.build(champion, role, api_key, region, sample_size, tier_mix)
            except Exception as e:
                logger.error(f"Failed to build registry for {champion}+{role}: {e}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"patch": self._patch, "builds": self._builds}
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"ItemRegistry saved to {path} ({len(self._builds)} champion+role entries)")

    @classmethod
    def load(cls, path: str | Path) -> "ItemRegistry":
        registry = cls()
        data = json.loads(Path(path).read_text())
        registry._patch  = data.get("patch", "unknown")
        registry._builds = data.get("builds", {})
        logger.info(
            f"ItemRegistry loaded from {path}: "
            f"patch={registry._patch}, {len(registry._builds)} entries"
        )
        return registry

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _api_get(self, url: str, api_key: str, max_retries: int = 3) -> bytes:
        """
        Make a rate-limited GET request to the Riot API.
        Retries with exponential backoff on 429 (rate limit) responses.
        """
        import urllib.request
        import urllib.error

        req = urllib.request.Request(url, headers={"X-Riot-Token": api_key})
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    retry_after = int(e.headers.get("Retry-After", "5"))
                    wait = retry_after + (2 ** attempt)
                    logger.warning(f"Rate limited (429). Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)
                elif e.code in (404, 403):
                    raise  # don't retry — resource not found or forbidden
                else:
                    raise
        raise RuntimeError(f"Failed after {max_retries} retries: {url}")

    def _load_item_names(self) -> None:
        """Fetch item id→name mapping from Community Dragon (free CDN, no API key)."""
        if self._item_id_to_name:
            return  # already loaded

        import urllib.request

        url = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/items.json"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                items = json.loads(resp.read().decode())
            for item in items:
                self._item_id_to_name[item["id"]] = item["name"]
            logger.info(f"Loaded {len(self._item_id_to_name)} item names from Community Dragon")
        except Exception as e:
            logger.warning(f"Could not load item names from Community Dragon: {e}")

    def _collect_match_ids(
        self,
        champion: str,
        role: str,
        api_key: str,
        region: str,
        tiers: list[str],
        target_count: int,
    ) -> list[str]:
        """Collect match IDs containing the target champion+role from Riot API."""
        import urllib.request

        # Get champion ID from champion name via Community Dragon
        champion_id = self._get_champion_id(champion, api_key, region)
        if champion_id is None:
            logger.warning(f"Could not resolve champion ID for '{champion}'")
            return []

        match_ids: list[str] = []
        per_tier = max(1, target_count // len(tiers))

        headers = {"X-Riot-Token": api_key}

        for tier in tiers:
            summoner_ids = self._get_summoner_ids_from_tier(tier, api_key, region)
            for summoner_id in summoner_ids:
                if len(match_ids) >= target_count:
                    break
                try:
                    puuid = self._get_puuid(summoner_id, api_key, region)
                    if puuid is None:
                        continue
                    ids = self._get_match_ids_for_puuid(puuid, champion_id, api_key, region, count=5)
                    time.sleep(0.07)  # ~14 req/s — stays under 20/s dev key limit
                    match_ids.extend(ids)
                except Exception:
                    continue

        return list(dict.fromkeys(match_ids))[:target_count]  # deduplicate, cap

    def _get_champion_id(self, champion_name: str, api_key: str, region: str) -> int | None:
        """Resolve champion name to Riot champion ID integer via Data Dragon."""
        import urllib.request
        url = "https://ddragon.leagueoflegends.com/api/versions.json"
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                versions = json.loads(r.read())
            latest = versions[0]
            champ_url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json"
            with urllib.request.urlopen(champ_url, timeout=8) as r:
                champ_data = json.loads(r.read())
            for key, val in champ_data["data"].items():
                if val["name"] == champion_name or key == champion_name:
                    return int(val["key"])
        except Exception as e:
            logger.warning(f"Champion ID lookup failed: {e}")
        return None

    def _get_summoner_ids_from_tier(
        self, tier: str, api_key: str, region: str, count: int = 20
    ) -> list[str]:
        """Get a sample of summoner IDs from a given tier using Riot League API."""
        platform = _region_to_platform(region)
        division = "I"
        url = (
            f"https://{platform}.api.riotgames.com/lol/league/v4/entries/"
            f"RANKED_SOLO_5x5/{tier}/{division}?page=1"
        )
        try:
            data = self._api_get(url, api_key)
            entries = json.loads(data)
            return [e["summonerId"] for e in entries[:count]]
        except Exception as e:
            logger.debug(f"Could not fetch summoner IDs for tier {tier}: {e}")
            return []

    def _get_puuid(self, summoner_id: str, api_key: str, region: str) -> str | None:
        platform = _region_to_platform(region)
        url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
        try:
            data = self._api_get(url, api_key)
            return json.loads(data).get("puuid")
        except Exception:
            return None

    def _get_match_ids_for_puuid(
        self, puuid: str, champion_id: int, api_key: str, region: str, count: int = 5
    ) -> list[str]:
        url = (
            f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/"
            f"{puuid}/ids?queue=420&count={count}"
        )
        try:
            data = self._api_get(url, api_key)
            return json.loads(data)
        except Exception:
            return []

    def _fetch_participant(
        self, match_id: str, champion: str, role: str, api_key: str, region: str
    ) -> dict | None:
        """Fetch match data and return the participant dict for the target champion+role."""
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        data = self._api_get(url, api_key)
        match_data = json.loads(data)
        for p in match_data["info"]["participants"]:
            if (
                p.get("championName") == champion
                and p.get("teamPosition", "").upper() == role.upper()
            ):
                return p
        return None


def _region_to_platform(region: str) -> str:
    """Map regional routing (americas/europe/asia) to platform routing (na1/euw1/kr)."""
    mapping = {
        "americas": "na1",
        "europe":   "euw1",
        "asia":     "kr",
    }
    return mapping.get(region.lower(), "euw1")
