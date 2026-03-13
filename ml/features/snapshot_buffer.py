"""
SnapshotBuffer — rolling per-player history window for temporal features.

The feature extractor needs access to the last N snapshots for each player
to compute delta features (e.g., Δ CS/min, death velocity, KP trend).

This class is owned by the WebSocket handler and lives for the duration of
one game session. At game_over it is discarded.

=== USAGE ===

    buffer = SnapshotBuffer(max_len=6)

    # Inside the snapshot receive loop:
    for player in snapshot["players"]:
        buffer.update(snapshot["game_time"], player)

    # Then when extracting features for that player:
    history = buffer.get_history(player["summonerName"])  # excludes current
    fv = extractor.extract(player, game_time, ..., history=history)

=== WHY max_len=6? ===

At a 5-second polling interval, 6 snapshots cover 30 seconds of game time.
Temporal features we care about (death velocity, delta CS rate) are most
meaningful over a 15–30 second window — enough to capture a "rage spiral"
(3 deaths in 30 seconds) while ignoring older, no-longer-relevant history.

For the GRU temporal model, we will use a longer window (up to 24 snapshots
= 2 minutes). The buffer supports this by setting max_len accordingly.
"""

from __future__ import annotations

from collections import deque


class SnapshotBuffer:
    """
    Per-player rolling window of raw snapshot player dicts.

    Each stored entry is the original player dict enriched with a
    "_game_time" key so the feature extractor can compute per-minute rates
    for historical snapshots.

    Thread safety: not thread-safe. The WebSocket handler is async/single-threaded,
    so this is fine for our use case.
    """

    def __init__(self, max_len: int = 6):
        """
        Args:
            max_len: Maximum number of snapshots to retain per player.
                     Older snapshots are automatically evicted (deque behavior).
        """
        self.max_len   = max_len
        self._buffers: dict[str, deque[dict]] = {}

    def update(self, game_time: float, player: dict) -> None:
        """
        Add a player's snapshot to their rolling buffer.

        Injects "_game_time" into a shallow copy of the player dict so the
        feature extractor can compute historical CS/min and death rates
        without needing access to the snapshot-level game_time.

        Args:
            game_time: Seconds since game start (from snapshot["game_time"])
            player:    Player dict from snapshot["players"][i]
        """
        name = player.get("summonerName", "")
        if not name:
            return

        if name not in self._buffers:
            self._buffers[name] = deque(maxlen=self.max_len)

        entry = dict(player)   # shallow copy — avoid mutating the live snapshot
        entry["_game_time"] = game_time
        self._buffers[name].append(entry)

    def get_history(self, player_name: str) -> list[dict]:
        """
        Returns all buffered snapshots for a player, excluding the most recent
        (which is the current snapshot being processed).

        By convention the caller has already called update() before calling
        get_history(), so the deque's last element is the current snapshot.
        Passing everything except the last entry as "history" gives the
        feature extractor the preceding context.

        Returns: list of player dicts (oldest first), possibly empty.
        """
        buf = self._buffers.get(player_name)
        if not buf or len(buf) < 2:
            return []
        return list(buf)[:-1]   # all but the most recent

    def get_full_sequence(self, player_name: str) -> list[dict]:
        """
        Returns all buffered snapshots including the most recent.
        Used when building training sequences for the GRU model.
        """
        return list(self._buffers.get(player_name, []))

    def has_player(self, player_name: str) -> bool:
        return player_name in self._buffers

    def player_count(self) -> int:
        return len(self._buffers)

    def clear(self) -> None:
        """
        Reset the buffer. Call this at game_over to free memory
        before the next game session starts.
        """
        self._buffers.clear()

    def __repr__(self) -> str:
        lengths = {name: len(buf) for name, buf in self._buffers.items()}
        return f"SnapshotBuffer(max_len={self.max_len}, players={lengths})"
