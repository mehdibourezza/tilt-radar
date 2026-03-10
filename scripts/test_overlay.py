"""
Quick smoke test for the TiltRadar overlay.

Runs without LoL. Simulates:
  1. HUD appearing with 10 fake players (all at 0%)
  2. After 3s: scores update to realistic mid-game values
  3. After 6s: enemy spikes → pop-up notification fires
  4. After 15s: program exits

Run with:
  python -m scripts.test_overlay
"""
import time
import threading
from agent.overlay import TiltOverlay, Notification

# --- Fake players (YOU + 4 allies + 5 enemies) ---
PLAYERS_INITIAL = [
    {"summonerName": "MarreDesNoobsNul#007", "championName": "Jinx",       "player_type": "self",  "tilt_score": 0.0},
    {"summonerName": "Thresh#EUW",           "championName": "Thresh",     "player_type": "ally",  "tilt_score": 0.0},
    {"summonerName": "Lux#1234",             "championName": "Lux",        "player_type": "ally",  "tilt_score": 0.0},
    {"summonerName": "Garen#EUW",            "championName": "Garen",      "player_type": "ally",  "tilt_score": 0.0},
    {"summonerName": "Ahri#EUW",             "championName": "Ahri",       "player_type": "ally",  "tilt_score": 0.0},
    {"summonerName": "Caitlyn#AXIS",         "championName": "Caitlyn",    "player_type": "enemy", "tilt_score": 0.0},
    {"summonerName": "Blitzcrank#EUW",       "championName": "Blitzcrank", "player_type": "enemy", "tilt_score": 0.0},
    {"summonerName": "Zed#KR1",              "championName": "Zed",        "player_type": "enemy", "tilt_score": 0.0},
    {"summonerName": "Fizz#EUW",             "championName": "Fizz",       "player_type": "enemy", "tilt_score": 0.0},
    {"summonerName": "Yasuo#FF",             "championName": "Yasuo",      "player_type": "enemy", "tilt_score": 0.0},
]

PLAYERS_MID_GAME = [
    {"summonerName": "MarreDesNoobsNul#007", "championName": "Jinx",       "player_type": "self",  "tilt_score": 0.30},
    {"summonerName": "Thresh#EUW",           "championName": "Thresh",     "player_type": "ally",  "tilt_score": 0.10},
    {"summonerName": "Lux#1234",             "championName": "Lux",        "player_type": "ally",  "tilt_score": 0.20},
    {"summonerName": "Garen#EUW",            "championName": "Garen",      "player_type": "ally",  "tilt_score": 0.0},
    {"summonerName": "Ahri#EUW",             "championName": "Ahri",       "player_type": "ally",  "tilt_score": 0.45},
    {"summonerName": "Caitlyn#AXIS",         "championName": "Caitlyn",    "player_type": "enemy", "tilt_score": 0.50},
    {"summonerName": "Blitzcrank#EUW",       "championName": "Blitzcrank", "player_type": "enemy", "tilt_score": 0.25},
    {"summonerName": "Zed#KR1",              "championName": "Zed",        "player_type": "enemy", "tilt_score": 0.15},
    {"summonerName": "Fizz#EUW",             "championName": "Fizz",       "player_type": "enemy", "tilt_score": 0.35},
    {"summonerName": "Yasuo#FF",             "championName": "Yasuo",      "player_type": "enemy", "tilt_score": 0.70},
]

PLAYERS_LATE_GAME = [
    {"summonerName": "MarreDesNoobsNul#007", "championName": "Jinx",       "player_type": "self",  "tilt_score": 0.40},
    {"summonerName": "Thresh#EUW",           "championName": "Thresh",     "player_type": "ally",  "tilt_score": 0.10},
    {"summonerName": "Lux#1234",             "championName": "Lux",        "player_type": "ally",  "tilt_score": 0.20},
    {"summonerName": "Garen#EUW",            "championName": "Garen",      "player_type": "ally",  "tilt_score": 0.0},
    {"summonerName": "Ahri#EUW",             "championName": "Ahri",       "player_type": "ally",  "tilt_score": 0.55},
    {"summonerName": "Caitlyn#AXIS",         "championName": "Caitlyn",    "player_type": "enemy", "tilt_score": 0.80},
    {"summonerName": "Blitzcrank#EUW",       "championName": "Blitzcrank", "player_type": "enemy", "tilt_score": 0.25},
    {"summonerName": "Zed#KR1",             "championName": "Zed",        "player_type": "enemy", "tilt_score": 0.15},
    {"summonerName": "Fizz#EUW",             "championName": "Fizz",       "player_type": "enemy", "tilt_score": 0.50},
    {"summonerName": "Yasuo#FF",             "championName": "Yasuo",      "player_type": "enemy", "tilt_score": 0.90},
]


def simulate(overlay: TiltOverlay):
    # Step 1 — initial state (all zeros, HUD builds from this)
    print("[test] Step 1: HUD initializing with 10 players at 0%...")
    overlay.update_hud(PLAYERS_INITIAL)
    time.sleep(3)

    # Step 2 — mid-game scores update
    print("[test] Step 2: Mid-game scores (Yasuo at 70%, Caitlyn at 50%)...")
    overlay.update_hud(PLAYERS_MID_GAME)
    time.sleep(3)

    # Step 3 — Yasuo spikes → pop-up notification
    print("[test] Step 3: Yasuo spikes → pop-up card fires...")
    overlay.notify(Notification(
        champion="Yasuo",
        summoner="Yasuo#FF",
        tilt_score=0.90,
        tilt_type="rage",
        exploit="Yasuo is playing emotionally. Bait them with an overextend then punish.",
        new_signals=["death_rate_accelerating", "3_deaths_before_10min"],
        player_type="enemy",
    ))
    overlay.update_hud(PLAYERS_LATE_GAME)
    time.sleep(3)

    # Step 4 — Ahri ally pop-up (cyan card)
    print("[test] Step 4: Ahri (ally) crosses threshold → cyan pop-up...")
    overlay.notify(Notification(
        champion="Ahri",
        summoner="Ahri#EUW",
        tilt_score=0.55,
        tilt_type="doom",
        exploit="Ahri (ally) looks checked out — don't rely on them for carries this fight.",
        new_signals=["cs_down_25pct_vs_baseline"],
        player_type="ally",
    ))
    time.sleep(6)

    print("[test] Done — closing overlay.")
    overlay.stop()


if __name__ == "__main__":
    overlay = TiltOverlay()

    # Simulation runs in a background thread.
    # Tkinter MUST run on the main thread on Windows — so we call _run() directly here.
    sim_thread = threading.Thread(target=simulate, args=(overlay,), daemon=True)
    sim_thread.start()

    overlay._run()   # blocks until overlay.stop() is called from simulate()
