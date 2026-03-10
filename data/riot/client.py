"""
Riot Games API client.

Two routing layers (this is a Riot-specific thing):
  - Platform routing: na1, euw1, kr, ... → champion/summoner/live-game endpoints
  - Regional routing: americas, europe, asia → match history/timeline endpoints

Rate limiting: dev keys = 20 req/s, 100 req/2min.
We use a token bucket approach to stay within limits.
"""

import asyncio
import time
import logging
from typing import Any
import httpx

from configs.config import get_settings

logger = logging.getLogger(__name__)

PLATFORM_TO_REGION = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "kr": "asia",
    "jp1": "asia",
}


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: int, period: float):
        self.rate = rate
        self.period = period
        self._tokens = rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill tokens proportionally to time elapsed
            self._tokens = min(
                self.rate,
                self._tokens + elapsed * (self.rate / self.period),
            )
            self._last_refill = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) * (self.period / self.rate)
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


class RiotClient:
    """
    Async Riot Games API client with automatic rate limiting and retries.

    Usage:
        async with RiotClient() as client:
            puuid = await client.get_puuid("Faker", "T1", platform="kr")
            matches = await client.get_match_ids(puuid, count=50)
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.riot_api_key
        self.default_platform = settings.riot_default_platform
        self.default_region = PLATFORM_TO_REGION[settings.riot_default_platform]

        # Two rate limiters: per-second and per-2-minutes
        self._limiter_per_second = RateLimiter(
            rate=settings.riot_rate_limit_per_second, period=1.0
        )
        self._limiter_per_two_minutes = RateLimiter(
            rate=settings.riot_rate_limit_per_two_minutes, period=120.0
        )
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers={"X-Riot-Token": self.api_key},
            timeout=10.0,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _request(self, url: str, params: dict | None = None) -> Any:
        """Core request method with rate limiting and retry on 429."""
        await self._limiter_per_second.acquire()
        await self._limiter_per_two_minutes.acquire()

        for attempt in range(3):
            response = await self._client.get(url, params=params)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning(f"429 rate limited — waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                continue

            if response.status_code == 404:
                return None  # Caller decides how to handle not found

            response.raise_for_status()

        raise RuntimeError(f"Request failed after 3 attempts: {url}")

    # -------------------------------------------------------------------------
    # Summoner / Account endpoints
    # -------------------------------------------------------------------------

    async def get_puuid(self, game_name: str, tag_line: str, region: str | None = None) -> str | None:
        """
        Convert Riot ID (GameName#TAG) → PUUID.
        PUUID is the stable cross-region player identifier.
        """
        region = region or self.default_region
        url = f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        data = await self._request(url)
        return data["puuid"] if data else None

    async def get_summoner_by_puuid(self, puuid: str, platform: str | None = None) -> dict | None:
        """Get summoner details (level, icon, etc.) from PUUID."""
        platform = platform or self.default_platform
        url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return await self._request(url)

    async def get_rank(self, puuid: str, platform: str | None = None) -> dict | None:
        """Get current rank (tier, division, LP) for a summoner by PUUID."""
        platform = platform or self.default_platform
        url = f"https://{platform}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
        data = await self._request(url)
        if not data:
            return None
        # Return solo queue rank specifically
        for entry in data:
            if entry.get("queueType") == "RANKED_SOLO_5x5":
                return entry
        return None

    # -------------------------------------------------------------------------
    # Match history endpoints (uses regional routing)
    # -------------------------------------------------------------------------

    async def get_match_ids(
        self,
        puuid: str,
        count: int = 100,
        queue: int = 420,          # 420 = ranked solo/duo
        region: str | None = None,
    ) -> list[str]:
        """
        Get list of match IDs for a player.
        queue=420 → ranked solo, queue=400 → normal draft
        """
        region = region or self.default_region
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        data = await self._request(url, params={"count": count, "queue": queue})
        return data or []

    async def get_match(self, match_id: str, region: str | None = None) -> dict | None:
        """
        Get full post-game match data.
        Contains: participants, stats, items, KDA, CS, etc.
        """
        region = region or self.default_region
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return await self._request(url)

    async def get_match_timeline(self, match_id: str, region: str | None = None) -> dict | None:
        """
        Get minute-by-minute event timeline for a match.

        This is the richest data source for us:
        - Every kill/death with position (x, y) and timestamp
        - Item purchases (with exact minute)
        - Gold/XP/CS snapshots every minute for each player
        - Ward placements and destructions
        - Objective events (dragon, baron, turret)

        This is NOT real-time — it's available after the game ends.
        """
        region = region or self.default_region
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        return await self._request(url)

    # -------------------------------------------------------------------------
    # Live game endpoint (uses platform routing)
    # -------------------------------------------------------------------------

    async def get_live_game(self, puuid: str, platform: str | None = None) -> dict | None:
        """
        Get current live game state for a player (if they are in a game).

        What you get:
        - All 10 participants (champion, summoner spells, runes)
        - Game mode, duration
        - Banned champions
        - Perks/augments

        What you DON'T get:
        - Real-time kills, deaths, gold, positions
        - These are only available post-game via timeline

        Polling this every ~30s gives you participant list but not live stats.
        """
        platform = platform or self.default_platform
        url = f"https://{platform}.api.riotgames.com/lol/spectator/v5/active-games/by-summoner/{puuid}"
        return await self._request(url)
