"""
TiltRadar Local Agent.

Runs on the player's machine alongside League of Legends.
Responsibilities:
  1. Wait for a game to start (polls Live Client API until it responds)
  2. Open a WebSocket connection to the TiltRadar backend
  3. Every POLL_INTERVAL seconds: send a game snapshot to the backend
  4. Receive tilt scores back and print them to the console
  5. Detect game end and cleanly shut down

Usage:
  python -m agent.local_agent --summoner "Faker" --tag "T1" --server ws://localhost:8000

The --summoner and --tag are YOUR Riot ID (not the enemies).
The backend uses it to identify which game you are in via the Spectator API.
"""

import asyncio
import argparse
import json
import logging
import websockets
from datetime import datetime, timezone

from agent.live_client import LiveClientAPI
from agent.overlay import TiltOverlay, Notification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

POLL_INTERVAL = 5       # seconds between snapshots sent to backend
GAME_WAIT_INTERVAL = 3  # seconds between checks when waiting for game to start


def extract_snapshot(
    raw: dict,
    your_summoner: str,
    prev_items: dict[str, list[int]] | None = None,
) -> dict:
    """
    Extract the relevant fields from a raw Live Client API response.
    We strip unnecessary data before sending to reduce WebSocket payload size.

    prev_items: {summonerName → [itemIDs]} from the previous snapshot.
                Used to detect item sells between snapshots.

    Returns a clean snapshot dict ready to be JSON-serialized and sent.
    """
    all_players = raw.get("allPlayers", [])
    raw_events  = raw.get("events", {}).get("Events", [])
    game_time   = raw.get("gameData", {}).get("gameTime", 0)

    if prev_items is None:
        prev_items = {}

    # Identify your team so we know which 5 are enemies.
    # summonerName from the Live Client API includes the Riot tag ("MarreDesNoobsNul#007"),
    # but --summoner arg is just the game name ("MarreDesNoobsNul"). Strip before comparing.
    your_team = None
    for p in all_players:
        api_name = p.get("summonerName", "").split("#")[0]
        if api_name.lower() == your_summoner.lower():
            your_team = p.get("team")
            break

    enemy_team = "CHAOS" if your_team == "ORDER" else "ORDER"

    # Build name→team lookup (tag-stripped, lowercase) for participation calcs.
    name_to_team: dict[str, str] = {
        p.get("summonerName", "").split("#")[0].lower(): p.get("team", "")
        for p in all_players
    }

    # Raw kill + objective events — used below for KP and objective presence.
    kill_events_raw = [e for e in raw_events if e.get("EventName") == "ChampionKill"]
    obj_event_names = {"DragonKill", "BaronKill", "RiftHeraldKill"}
    obj_events_raw  = [e for e in raw_events if e.get("EventName") in obj_event_names]

    # KP time split: first half vs second half of current game time.
    midpoint = game_time / 2 if game_time > 0 else 1e9

    players_snapshot = []
    for p in all_players:
        summoner_name = p.get("summonerName", "")
        name_clean    = summoner_name.split("#")[0].lower()
        team          = p.get("team", "")
        is_self       = summoner_name.split("#")[0].lower() == your_summoner.lower()

        # --- Kill Participation (early vs late) ---
        # A "team kill" is any ChampionKill where the killer is on this player's team.
        def _is_team_kill(e: dict) -> bool:
            return name_to_team.get(e.get("KillerName", "").lower()) == team

        def _player_participated(e: dict) -> bool:
            return (
                e.get("KillerName", "").lower() == name_clean
                or name_clean in [a.lower() for a in e.get("Assisters", [])]
            )

        team_kills_early  = sum(1 for e in kill_events_raw if _is_team_kill(e)  and e.get("EventTime", 0) <  midpoint)
        team_kills_late   = sum(1 for e in kill_events_raw if _is_team_kill(e)  and e.get("EventTime", 0) >= midpoint)
        player_part_early = sum(1 for e in kill_events_raw if _is_team_kill(e)  and _player_participated(e) and e.get("EventTime", 0) <  midpoint)
        player_part_late  = sum(1 for e in kill_events_raw if _is_team_kill(e)  and _player_participated(e) and e.get("EventTime", 0) >= midpoint)

        # Only meaningful once each window has ≥3 team kills.
        kp_early = round(player_part_early / team_kills_early, 3) if team_kills_early >= 3 else None
        kp_late  = round(player_part_late  / team_kills_late,  3) if team_kills_late  >= 3 else None

        # --- Objective Absence ---
        # Team objectives = drake/baron where a teammate was the killer.
        team_obj_events = [e for e in obj_events_raw if name_to_team.get(e.get("KillerName", "").lower()) == team]
        obj_total  = len(team_obj_events)
        obj_missed = sum(1 for e in team_obj_events if not _player_participated(e))

        # --- Item Sells ---
        prev = prev_items.get(summoner_name, [])
        current_items = [i.get("itemID") for i in p.get("items", []) if i.get("itemID")]
        sold_items = [item for item in prev if item and item not in current_items]

        players_snapshot.append({
            "summonerName": summoner_name,
            "championName": p.get("championName"),
            "team":         team,
            "position":     p.get("position", ""),
            "level":        p.get("level", 1),
            "is_self":      is_self,
            "is_enemy":     team == enemy_team,
            "kills":        p.get("scores", {}).get("kills", 0),
            "deaths":       p.get("scores", {}).get("deaths", 0),
            "assists":      p.get("scores", {}).get("assists", 0),
            "cs":           p.get("scores", {}).get("creepScore", 0),
            "ward_score":   p.get("scores", {}).get("wardScore", 0.0),
            "items":         current_items,
            "sold_items":    sold_items,
            "kp_early":      kp_early,
            "kp_late":       kp_late,
            "obj_total":     obj_total,
            "obj_missed":    obj_missed,
            "is_dead":       p.get("isDead", False),
            "respawn_timer": p.get("respawnTimer", 0.0),
        })

    # Only keep kill events and objective events — what we need for tilt signals
    relevant_event_names = {
        "ChampionKill", "DragonKill", "BaronKill",
        "TurretKilled", "InhibKilled", "FirstBlood",
    }
    filtered_events = [
        {
            "type":      e.get("EventName"),
            "time":      e.get("EventTime"),
            "killer":    e.get("KillerName"),
            "victim":    e.get("VictimName"),
            "assisters": e.get("Assisters", []),
        }
        for e in raw_events
        if e.get("EventName") in relevant_event_names
    ]

    return {
        "game_time":    game_time,
        "your_summoner": your_summoner,
        "players":      players_snapshot,
        "events":       filtered_events,
        "snapshot_at":  datetime.now(timezone.utc).isoformat(),
    }


def _save_post_game_report(report_data: dict, your_summoner: str) -> None:
    """Format the post-game report as readable text, log it, and save to reports/."""
    import os
    import textwrap

    os.makedirs("reports", exist_ok=True)
    now       = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d_%H-%M")
    filename  = f"reports/{timestamp}_{your_summoner.replace('#', '_')}.txt"

    W = 80  # line width

    def bar(char: str = "═") -> str:
        return char * W

    lines: list[str] = []
    lines += [
        bar(),
        "TILT RADAR  —  POST-GAME REPORT",
        f"Duration : {report_data.get('game_time_fmt', '?')}   |   "
        f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        bar(),
    ]

    players = report_data.get("players", [])

    for section_label, ptype_key in [("ENEMIES", "enemy"), ("ALLIES", "ally"), ("YOU", "self")]:
        group = sorted(
            [p for p in players if p.get("player_type") == ptype_key],
            key=lambda p: p.get("tilt_score", 0.0),
            reverse=True,
        )
        if not group:
            continue

        lines += ["", f"  {section_label}  " + bar("─")[len(section_label) + 4:], ""]

        for p in group:
            score_pct  = int(p.get("tilt_score", 0.0) * 100)
            tilt_label = p.get("tilt_type", "none").upper()
            champion   = p.get("champion", "?")
            summoner   = p.get("summoner", "?")

            lines.append(f"  [{score_pct:>3}%  {tilt_label:<10}]  {champion} — {summoner}")

            s = p.get("final_stats", {})
            lines.append(
                f"   Stats : {s.get('kills',0)}/{s.get('deaths',0)}/{s.get('assists',0)}"
                f"  |  CS {s.get('cs',0)} ({s.get('cs_per_min',0):.1f}/min)"
                f"  |  Wards {s.get('ward_score',0):.0f} ({s.get('ward_per_min',0):.2f}/min)"
                f"  |  KDA {s.get('kda',0):.2f}"
            )

            if p.get("key_signals"):
                lines.append(f"   Signals: {', '.join(p['key_signals'])}")

            obj_total  = p.get("obj_total", 0)
            obj_missed = p.get("obj_missed", 0)
            if obj_total > 0:
                lines.append(f"   Objectives missed: {obj_missed}/{obj_total}")

            timeline = p.get("timeline", [])
            if timeline:
                lines.append("   Timeline:")
                for entry in timeline:
                    icon = "⚠" if entry["severity"] == "high" else ("→" if entry["severity"] == "medium" else " ")
                    lines.append(f"     {entry['time_fmt']:>6}  {icon}  {entry['detail']}")

            assessment = p.get("assessment", "")
            if assessment:
                lines.append("   Assessment:")
                for wrapped_line in textwrap.wrap(assessment, width=W - 6):
                    lines.append(f"     {wrapped_line}")

            lines += ["", "  " + "─" * (W - 2), ""]

    lines += [bar(), ""]
    text = "\n".join(lines)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info("\n" + text)
    logger.info(f"Post-game report saved → {filename}")


def process_tilt_report(report: dict, overlay: TiltOverlay, prev_scores: dict) -> dict:
    """
    Push overlay notifications for all players whose tilt changed meaningfully.
    Returns updated prev_scores dict.
    """
    # "players" covers all 10; fall back to "enemies" for backward compat
    all_players = report.get("players") or report.get("enemies", [])
    updated_scores = dict(prev_scores)

    # Feed the persistent HUD with the full player list on every report
    # Skip empty lists (e.g. the initial "status: ready" message from the backend)
    if all_players:
        overlay.update_hud(all_players)

    for player in all_players:
        score = player.get("tilt_score", 0.0)
        summoner = player.get("summonerName", "unknown")
        champion = player.get("championName", "?")
        tilt_type = player.get("tilt_type", "unknown")
        exploit = player.get("exploit")
        signals = player.get("key_signals", [])
        player_type = player.get("player_type", "enemy")

        # Detect new signals since last report
        prev_signals = prev_scores.get(f"{summoner}_signals", [])
        new_signals = [s for s in signals if s not in prev_signals]

        if overlay.should_notify(summoner, score, new_signals):
            overlay.notify(Notification(
                champion=champion,
                summoner=summoner,
                tilt_score=score,
                tilt_type=tilt_type,
                exploit=exploit,
                new_signals=new_signals,
                player_type=player_type,
            ))
            overlay._last_notified[summoner] = score

        updated_scores[summoner] = score
        updated_scores[f"{summoner}_signals"] = signals

    return updated_scores


async def run_agent(summoner: str, tag: str, server: str, overlay: TiltOverlay):
    """
    WebSocket + polling loop. Runs in a background thread (asyncio loop) so that
    tkinter can own the main thread (required on Windows for correct rendering).
    """
    live = LiveClientAPI()
    prev_scores: dict = {}

    try:
        # Step 1 — wait for a game to start
        logger.info(f"Waiting for a game to start ({summoner}#{tag})...")
        while not await live.is_in_game():
            await asyncio.sleep(GAME_WAIT_INTERVAL)
        logger.info("Game detected! Connecting to TiltRadar backend...")

        ws_url = f"{server}/ws/game/{summoner}/{tag}"

        async with websockets.connect(ws_url) as ws:
            logger.info(f"Connected to backend: {ws_url}")

            consecutive_failures = 0
            last_snapshot: dict | None = None

            while True:
                raw = await live.get_all_data()

                if raw is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        logger.info("Game ended — sending final state to backend...")
                        game_over_msg = {
                            "event": "game_over",
                            "summoner": summoner,
                            "tag": tag,
                            "final_players": last_snapshot["players"] if last_snapshot else [],
                            "game_duration_min": last_snapshot["game_time"] / 60 if last_snapshot else 0,
                        }
                        await ws.send(json.dumps(game_over_msg))

                        # Receive the post-game report the backend generates
                        try:
                            raw_report = await asyncio.wait_for(ws.recv(), timeout=15.0)
                            report_data = json.loads(raw_report)
                            if report_data.get("type") == "post_game_report":
                                _save_post_game_report(report_data, summoner)
                        except asyncio.TimeoutError:
                            logger.warning("Post-game report timed out — backend may still be processing.")
                        except Exception:
                            logger.exception("Failed to receive post-game report.")
                        break
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                consecutive_failures = 0
                prev_items = {
                    p["summonerName"]: p["items"]
                    for p in last_snapshot["players"]
                } if last_snapshot else {}
                snapshot = extract_snapshot(raw, summoner, prev_items)
                last_snapshot = snapshot

                await ws.send(json.dumps(snapshot))
                logger.debug(f"Snapshot sent at game time {snapshot['game_time']:.0f}s")

                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=4.0)
                    report = json.loads(response)
                    prev_scores = process_tilt_report(report, overlay, prev_scores)
                except asyncio.TimeoutError:
                    logger.warning("Backend did not respond in time — skipping")

                await asyncio.sleep(POLL_INTERVAL)

    finally:
        overlay.stop()
        await live.close()
        logger.info("Agent stopped.")


def main():
    parser = argparse.ArgumentParser(description="TiltRadar Local Agent")
    parser.add_argument("--summoner", required=True, help="Your Riot summoner name")
    parser.add_argument("--tag", required=True, help="Your Riot tag (e.g. EUW, T1)")
    parser.add_argument("--server", default="ws://localhost:8000", help="Backend WebSocket URL")
    args = parser.parse_args()

    overlay = TiltOverlay()
    logger.info("Overlay ready — set LoL to Borderless Windowed mode if not already done.")

    # asyncio runs in a background thread; tkinter MUST own the main thread on Windows
    loop = asyncio.new_event_loop()

    def run_loop():
        loop.run_until_complete(run_agent(args.summoner, args.tag, args.server, overlay))

    import threading
    agent_thread = threading.Thread(target=run_loop, daemon=True)
    agent_thread.start()

    overlay._run()   # blocks on main thread until game ends and overlay.stop() is called


if __name__ == "__main__":
    main()
