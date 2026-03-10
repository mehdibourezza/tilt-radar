"""
Show your personal progress over your last N games.

Usage:
    conda run -n tilt-radar python scripts/my_progress.py --summoner "YourName" --tag "EUW"

Shows:
  - CS/min trend (last 20 games)
  - KDA trend
  - Death pattern improvement (solo deaths, repeat deaths)
  - Win rate trend
  - Change points in your own performance (using our same PELT algorithm)
"""

import asyncio
import sys
import argparse
sys.path.insert(0, ".")


async def main(summoner: str, tag: str):
    from data.db.session import get_session
    from data.db.repository import PlayerRepository
    from data.riot.client import RiotClient

    # Resolve PUUID
    async with RiotClient() as riot:
        puuid = await riot.get_puuid(summoner, tag)
        if not puuid:
            print(f"Player {summoner}#{tag} not found.")
            return

    async with get_session() as session:
        repo = PlayerRepository(session)
        stats = await repo.get_recent_game_stats(puuid, limit=20)
        baseline = await repo.get_baseline(puuid)

    if not stats:
        print(f"No games found for {summoner}#{tag} yet.")
        print("Play a game first — your stats are captured automatically at game end.")
        return

    print(f"\n{'='*52}")
    print(f"  TiltRadar — Personal Progress: {summoner}#{tag}")
    print(f"{'='*52}")
    print(f"  Games tracked: {len(stats)}")

    # CS/min trend
    cs_values = [s.cs_per_min for s in stats]
    cs_avg_first_half = _avg(cs_values[:len(cs_values)//2])
    cs_avg_second_half = _avg(cs_values[len(cs_values)//2:])
    cs_delta = cs_avg_second_half - cs_avg_first_half
    cs_arrow = "↑" if cs_delta > 0.1 else ("↓" if cs_delta < -0.1 else "→")

    print(f"\n  CS/min")
    print(f"  Early games avg : {cs_avg_first_half:.1f}")
    print(f"  Recent games avg: {cs_avg_second_half:.1f}  {cs_arrow} ({cs_delta:+.1f})")
    _print_sparkline([s.cs_per_min for s in stats], label="  ")

    # KDA trend
    kdas = [
        (s.kills + s.assists) / max(s.deaths, 1)
        for s in stats
    ]
    kda_first = _avg(kdas[:len(kdas)//2])
    kda_recent = _avg(kdas[len(kdas)//2:])
    kda_delta = kda_recent - kda_first
    kda_arrow = "↑" if kda_delta > 0.1 else ("↓" if kda_delta < -0.1 else "→")

    print(f"\n  KDA")
    print(f"  Early games avg : {kda_first:.2f}")
    print(f"  Recent games avg: {kda_recent:.2f}  {kda_arrow} ({kda_delta:+.2f})")

    # Death patterns
    solo_deaths_avg_early  = _avg([s.solo_deaths for s in stats[:len(stats)//2]])
    solo_deaths_avg_recent = _avg([s.solo_deaths for s in stats[len(stats)//2:]])
    sd_delta = solo_deaths_avg_recent - solo_deaths_avg_early
    sd_arrow = "↑" if sd_delta > 0.1 else ("↓" if sd_delta < -0.1 else "→")

    print(f"\n  Solo deaths per game (lower = better)")
    print(f"  Early games avg : {solo_deaths_avg_early:.1f}")
    print(f"  Recent games avg: {solo_deaths_avg_recent:.1f}  {sd_arrow} ({sd_delta:+.1f})")

    # Win rate
    wins_early  = sum(1 for s in stats[:len(stats)//2] if s.won)
    wins_recent = sum(1 for s in stats[len(stats)//2:] if s.won)
    total_early  = max(len(stats)//2, 1)
    total_recent = max(len(stats) - len(stats)//2, 1)
    wr_early  = wins_early / total_early
    wr_recent = wins_recent / total_recent
    wr_arrow = "↑" if wr_recent > wr_early + 0.05 else ("↓" if wr_recent < wr_early - 0.05 else "→")

    print(f"\n  Win rate")
    print(f"  Early games : {wr_early:.0%}")
    print(f"  Recent games: {wr_recent:.0%}  {wr_arrow}")

    # Change point detection on own performance
    if baseline and baseline.change_points:
        print(f"\n  Performance shifts detected:")
        for cp in baseline.change_points[-3:]:   # show last 3
            direction = "dropped" if cp["direction"] == "drop" else "improved"
            print(f"  · Your {cp['metric']} {direction} around game {cp['game_index']} "
                  f"({cp['before_mean']:.1f} → {cp['after_mean']:.1f})")

    if baseline and baseline.chronic_slump_detected:
        print(f"\n  ⚠  Slump detected in your recent games.")
        print(f"     Your recent performance is significantly below your historical level.")
        print(f"     Consider taking a break or reviewing your replays.")

    print(f"\n{'='*52}\n")


def _avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_sparkline(values: list[float], label: str = ""):
    """Print a simple ASCII sparkline."""
    if not values:
        return
    mn, mx = min(values), max(values)
    span = mx - mn or 1
    bars = " ▁▂▃▄▅▆▇█"
    line = "".join(bars[int((v - mn) / span * 8)] for v in values)
    print(f"{label}Trend: {line}  ({mn:.1f} – {mx:.1f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summoner", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.summoner, args.tag))
