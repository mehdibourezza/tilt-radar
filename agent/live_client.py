"""
Wrapper for the Riot Live Client Data API.

This API runs locally on the player's machine at port 2999 while a game is active.
It uses a self-signed SSL certificate — we disable SSL verification intentionally.

Key endpoints:
  /liveclientdata/allgamedata     → full snapshot: all players, all events, game time
  /liveclientdata/eventdata       → just the event list (kills, objectives, etc.)
  /liveclientdata/playerlist      → all 10 players with live stats
  /liveclientdata/gamestats       → game time, mode, map

Availability:
  - Returns 404 / connection refused when not in a game
  - Becomes available ~30s after game loading screen starts
  - Disappears when the game ends
"""

import httpx
import logging

logger = logging.getLogger(__name__)

BASE_URL = "https://127.0.0.1:2999/liveclientdata"


class LiveClientAPI:
    """
    Polls the local Live Client Data API.

    SSL verification is disabled because Riot uses a self-signed cert on localhost.
    This is safe — we are only ever talking to localhost.
    """

    def __init__(self):
        # verify=False: Riot's local API uses a self-signed cert
        self._client = httpx.AsyncClient(verify=False, timeout=3.0)

    async def close(self):
        await self._client.aclose()

    async def is_in_game(self) -> bool:
        """Return True if a game is currently active."""
        try:
            r = await self._client.get(f"{BASE_URL}/gamestats")
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def get_all_data(self) -> dict | None:
        """
        Full game snapshot. Returns None if not in a game.

        Structure:
        {
          "activePlayer": { ...your player stats... },
          "allPlayers": [
            {
              "summonerName": "Faker",
              "championName": "Azir",
              "team": "ORDER" | "CHAOS",
              "position": "MIDDLE",
              "scores": {
                "kills": 3,
                "deaths": 1,
                "assists": 2,
                "creepScore": 187,
              },
              "items": [...],
              "summonerSpells": {...},
            },
            ...  (10 players total)
          ],
          "events": {
            "Events": [
              { "EventName": "ChampionKill", "KillerName": "Faker",
                "VictimName": "Caps", "EventTime": 312.5 },
              { "EventName": "DragonKill", "KillerName": "...", "EventTime": 480.2 },
              ...
            ]
          },
          "gameData": {
            "gameTime": 623.1,   ← seconds since game start
            "gameMode": "CLASSIC",
          }
        }
        """
        try:
            r = await self._client.get(f"{BASE_URL}/allgamedata")
            if r.status_code == 200:
                return r.json()
            return None
        except (httpx.ConnectError, httpx.TimeoutException):
            return None

    async def get_events(self) -> list[dict]:
        """
        All game events so far (kills, objectives, turrets, etc.).
        Returns empty list if not in a game.
        """
        try:
            r = await self._client.get(f"{BASE_URL}/eventdata")
            if r.status_code == 200:
                return r.json().get("Events", [])
            return []
        except (httpx.ConnectError, httpx.TimeoutException):
            return []

    async def get_player_list(self) -> list[dict]:
        """All 10 players with live stats."""
        try:
            r = await self._client.get(f"{BASE_URL}/playerlist")
            if r.status_code == 200:
                return r.json()
            return []
        except (httpx.ConnectError, httpx.TimeoutException):
            return []
