"""
WebSocket endpoint for real-time tilt analysis.

Flow:
  1. Agent connects: ws/game/{summoner}/{tag}
  2. Backend looks up the active game via Riot Spectator API
     → identifies all 10 participants → fetches baselines from DB
  3. For each snapshot the agent sends:
     → run tilt inference on all 10 players
     → send back a tilt report JSON
     → track peak tilt scores per player for post-game evaluation
  4. On game_over:
     → compare peak predictions against final scoreboard outcomes
     → log TiltPredictionLog records (labeled training data for the ML model)
     → queue Celery ingestion for self + all 9 others
"""

import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from data.riot.client import RiotClient
from data.db.repository import PlayerRepository
from ml.inference.engine import TiltInferenceEngine
from ml.features.feature_extractor import FeatureExtractor
from ml.features.snapshot_buffer import SnapshotBuffer
from api.dependencies.db import get_db_session
from workers.tasks import ingest_player

logger = logging.getLogger(__name__)
router = APIRouter()

# Tilt threshold for "predicted tilted" in the post-game verdict
PREDICTION_THRESHOLD = 0.55


def _evaluate_outcome(
    player_name: str,
    player_type: str,
    champion_name: str,
    peak: dict,               # {score, tilt_type, signals}
    final_player: dict,       # from last snapshot
    game_duration_min: float,
    peer_baseline,            # PeerGroupBaseline ORM object or None
) -> dict:
    """
    Compare what we predicted (peak tilt score) against how the player actually ended.

    Returns a dict ready to be passed to repo.log_tilt_prediction().

    Verdict logic:
      - "performed_poorly" = final stats significantly below peer group baseline
        (if peer baseline available) or rough heuristics (if not)
      - "predicted_tilted" = peak_score >= PREDICTION_THRESHOLD

      true_positive:  predicted AND performed poorly  → signal was real
      false_positive: predicted BUT performed fine    → we over-triggered
      true_negative:  not predicted AND performed fine → correct silence
      false_negative: not predicted BUT performed poorly → we missed it
    """
    deaths = final_player.get("deaths", 0)
    cs = final_player.get("cs", 0)
    kills = final_player.get("kills", 0)
    assists = final_player.get("assists", 0)

    game_duration_min = max(game_duration_min, 1.0)
    final_death_rate = deaths / game_duration_min
    final_cs_per_min = cs / game_duration_min
    final_kda = (kills + assists) / max(deaths, 1)

    peer_cs   = getattr(peer_baseline, "cs_per_min_median", None)
    peer_dr   = getattr(peer_baseline, "death_rate_median", None)

    if peer_cs and peer_dr:
        # Concrete underperformance vs rank average
        cs_dropped   = final_cs_per_min  < peer_cs * 0.75   # >25% below rank median
        deaths_high  = final_death_rate  > peer_dr  * 1.50  # >50% above rank median
        performed_poorly = cs_dropped or deaths_high
    else:
        # Rough heuristics when no peer baseline available
        performed_poorly = final_kda < 1.0 and deaths >= 5

    predicted_tilted = peak["score"] >= PREDICTION_THRESHOLD

    if predicted_tilted and performed_poorly:
        verdict = "true_positive"
    elif predicted_tilted and not performed_poorly:
        verdict = "false_positive"
    elif not predicted_tilted and performed_poorly:
        verdict = "false_negative"
    else:
        verdict = "true_negative"

    return {
        "player_name": player_name,
        "player_type": player_type,
        "champion_name": champion_name,
        "peak_tilt_score": round(peak["score"], 3),
        "peak_tilt_type": peak["tilt_type"],
        "peak_signals": peak["signals"],
        "final_kills": kills,
        "final_deaths": deaths,
        "final_assists": assists,
        "final_cs": cs,
        "game_duration_min": round(game_duration_min, 2),
        "final_cs_per_min": round(final_cs_per_min, 3),
        "final_death_rate": round(final_death_rate, 3),
        "final_kda": round(final_kda, 2),
        "peer_cs_per_min_median": peer_cs,
        "peer_death_rate_median": peer_dr,
        "predicted_tilted": predicted_tilted,
        "performed_poorly": performed_poorly,
        "verdict": verdict,
    }


@router.websocket("/ws/game/{summoner}/{tag}")
async def game_websocket(websocket: WebSocket, summoner: str, tag: str):
    await websocket.accept()
    logger.info(f"Agent connected for {summoner}#{tag}")

    engine    = TiltInferenceEngine()
    extractor = FeatureExtractor()
    buffer    = SnapshotBuffer(max_len=6)

    try:
        async with RiotClient() as riot:
            # Step 1 — resolve player PUUID
            puuid = await riot.get_puuid(summoner, tag)
            if not puuid:
                await websocket.send_json({"error": f"Summoner {summoner}#{tag} not found"})
                await websocket.close()
                return

            # Step 2 — fetch active game to identify all participants
            summoner_data = await riot.get_summoner_by_puuid(puuid)
            if not summoner_data:
                await websocket.send_json({"error": "Could not fetch summoner data"})
                await websocket.close()
                return

            live_game = await riot.get_live_game(puuid)
            if not live_game:
                await websocket.send_json({"error": "No active game found"})
                await websocket.close()
                return

            # Identify teams
            your_team_id = None
            for participant in live_game.get("participants", []):
                if participant.get("puuid") == puuid:
                    your_team_id = participant.get("teamId")
                    break

            all_participants = live_game.get("participants", [])
            enemy_puuids = [
                p["puuid"]
                for p in all_participants
                if p.get("teamId") != your_team_id and p.get("puuid")
            ]
            ally_puuids = [
                p["puuid"]
                for p in all_participants
                if p.get("teamId") == your_team_id
                and p.get("puuid") and p.get("puuid") != puuid
            ]
            non_self_puuids = enemy_puuids + ally_puuids

            # Step 3 — fetch baselines for all 10 players
            async for session in get_db_session():
                repo = PlayerRepository(session)
                personal_baselines = {}   # puuid → PlayerBaseline  (self only, if ingested)
                peer_baselines = {}       # puuid → PeerGroupBaseline (all 9 non-self + self fallback)

                # Self: personal baseline first, peer as fallback
                self_baseline = await repo.get_baseline(puuid)
                if self_baseline:
                    personal_baselines[puuid] = self_baseline
                    logger.info(f"Personal baseline loaded for self ({puuid[:12]})")
                else:
                    logger.warning("No personal baseline for self — peer group will be used as fallback")
                    self_rank = await riot.get_rank(puuid)
                    if self_rank:
                        tier = self_rank.get("tier", "GOLD")
                        division = self_rank.get("rank", "II")
                        peer = await repo.get_peer_baseline(tier, division)
                        if peer:
                            peer_baselines[puuid] = peer

                # All 9 others: peer group baseline by rank
                for other_puuid in non_self_puuids:
                    if not other_puuid:
                        continue
                    rank = await riot.get_rank(other_puuid)
                    if rank:
                        tier = rank.get("tier", "GOLD")
                        division = rank.get("rank", "II")
                        peer = await repo.get_peer_baseline(tier, division)
                        if peer:
                            peer_baselines[other_puuid] = peer
                            logger.info(f"Peer baseline loaded: {tier} {division} for {other_puuid[:12]}")
                        else:
                            logger.warning(f"No peer baseline for {tier} {division} — run populate_peer_groups.py")
                    else:
                        logger.warning(f"Could not fetch rank for {other_puuid[:12]}")

            personal_count = len(personal_baselines)
            peer_count = len(peer_baselines)
            logger.info(
                f"Ready — personal baselines: {personal_count}/1, "
                f"peer baselines: {peer_count}/10 (9 others + self fallback)"
            )
            await websocket.send_json({
                "status": "ready",
                "personal_baselines": personal_count,
                "peer_baselines": peer_count,
            })

            # Step 4 — main loop: receive snapshots, score, track peaks
            # peak_scores: {summoner_name → {score, tilt_type, signals, champion, role,
            #                                game_time_at_peak, feature_vector}}
            peak_scores: dict[str, dict] = {}
            # Retained for post-game report: last full snapshot + item sell timeline
            backend_last_snapshot: dict | None = None
            item_sell_timeline: dict[str, list] = {}   # {summoner → [{time, items}]}

            while True:
                raw = await websocket.receive_text()
                snapshot = json.loads(raw)

                # Game over — build report, run evaluation, queue ingestion
                if snapshot.get("event") == "game_over":
                    logger.info(f"Game over for {summoner}#{tag}")
                    final_players = snapshot.get("final_players", [])
                    game_duration_min = snapshot.get("game_duration_min", 30.0)

                    # Send post-game report to agent before any async DB work
                    post_game_report = _build_post_game_report(
                        peak_scores, backend_last_snapshot,
                        item_sell_timeline, game_duration_min,
                    )
                    await websocket.send_json(post_game_report)
                    logger.info("Post-game report sent to agent.")

                    await _run_post_game_evaluation(
                        peak_scores, final_players, game_duration_min,
                        peer_baselines,
                    )

                    # Reset snapshot buffer for next game session
                    buffer.clear()

                    # Queue async ingestion for self + all others
                    ingest_player.delay(puuid, platform="euw1")
                    for ep in non_self_puuids:
                        ingest_player.delay(ep, platform="euw1")
                    break

                # One-shot debug: on first snapshot that has kill events, log raw names
                # to check for mismatches between player names and kill event names
                kill_events_raw = [e for e in snapshot.get("events", []) if e.get("type") == "ChampionKill"]
                if kill_events_raw and not peak_scores.get("__debug_logged__"):
                    peak_scores["__debug_logged__"] = True
                    player_names = [p["summonerName"] for p in snapshot.get("players", [])]
                    sample = [{"killer": e.get("killer"), "victim": e.get("victim")} for e in kill_events_raw[:3]]
                    logger.info(f"DEBUG names in players list: {player_names}")
                    logger.info(f"DEBUG kill event names (sample): {sample}")

                report    = engine.score(snapshot, personal_baselines, peer_baselines)
                game_time = snapshot.get("game_time", 0)
                all_players = snapshot.get("players", [])
                kill_events = [e for e in snapshot.get("events", []) if e.get("type") == "ChampionKill"]

                # ── Feature extraction (Phase A ML pipeline) ──────────────────
                # Update buffer first so history is available for delta features.
                # Then extract the feature vector for each player.
                for player in all_players:
                    buffer.update(game_time, player)

                player_features: dict[str, list[float]] = {}
                for player in all_players:
                    pname     = player.get("summonerName", "")
                    puuid_key = player.get("puuid")
                    personal  = personal_baselines.get(puuid_key) or personal_baselines.get(pname)
                    peer      = peer_baselines.get(puuid_key) or peer_baselines.get(pname)
                    history   = buffer.get_history(pname)
                    try:
                        fv = extractor.extract(
                            player=player,
                            game_time=game_time,
                            kill_events=kill_events,
                            all_players=all_players,
                            personal_baseline=personal,
                            peer_baseline=peer,
                            history=history,
                        )
                        player_features[pname] = fv.to_list()
                    except Exception as exc:
                        logger.warning(f"Feature extraction failed for {pname}: {exc}")

                # Log a compact score summary every snapshot so we can debug in real time
                game_fmt = report.get("game_time_fmt", "?")
                tilted = [
                    f"{r['summonerName']}({r['player_type'][0].upper()})={r['tilt_score']:.2f}"
                    for r in report.get("players", [])
                    if r["tilt_score"] > 0
                ]
                if tilted:
                    logger.info(f"[{game_fmt}] Tilt scores: {', '.join(tilted)}")
                else:
                    logger.info(f"[{game_fmt}] All scores 0.00 — no signals triggered")

                # Update peak tilt scores — now also captures role, game_time, feature_vector
                for player_result in report.get("players", []):
                    name  = player_result["summonerName"]
                    score = player_result["tilt_score"]
                    if score > peak_scores.get(name, {}).get("score", 0.0):
                        # Find position from the raw player dict
                        raw_player = next(
                            (p for p in all_players if p.get("summonerName") == name), {}
                        )
                        peak_scores[name] = {
                            "score":            score,
                            "tilt_type":        player_result["tilt_type"],
                            "signals":          player_result["key_signals"],
                            "champion":         player_result["championName"],
                            "player_type":      player_result["player_type"],
                            "role":             raw_player.get("position", ""),
                            "game_time_at_peak": game_time,
                            "feature_vector":   player_features.get(name),
                            "n_signals_active": len(player_result["key_signals"]),
                        }

                await websocket.send_json(report)

                # Retain last snapshot for post-game report; track item sell timing
                backend_last_snapshot = snapshot
                for player in snapshot.get("players", []):
                    sold = player.get("sold_items", [])
                    if sold:
                        item_sell_timeline.setdefault(player["summonerName"], []).append({
                            "time": snapshot["game_time"],
                            "items": sold,
                        })

    except WebSocketDisconnect:
        logger.info(f"Agent disconnected for {summoner}#{tag}")
    except Exception as e:
        logger.exception(f"Error in WebSocket handler: {e}")
        await websocket.close()


async def _run_post_game_evaluation(
    peak_scores: dict,
    final_players: list[dict],
    game_duration_min: float,
    peer_baselines: dict,
):
    """
    Compare tilt predictions against final scoreboard.
    Logs a TiltPredictionLog record for every player, then prints a summary.
    """
    if not final_players or not peak_scores:
        logger.info("Post-game evaluation skipped — no data available")
        return

    # Index final player stats by summoner name (case-insensitive)
    final_by_name = {p["summonerName"].lower(): p for p in final_players}

    entries = []
    for summoner_name, peak in peak_scores.items():
        final = final_by_name.get(summoner_name.lower())
        if not final:
            continue

        # Find peer baseline for this player (by name since we may not have puuid in snapshot)
        peer = None
        for bl in peer_baselines.values():
            peer = bl
            break   # use any available as approximate; good enough for self/ally fallback

        entry = _evaluate_outcome(
            player_name=summoner_name,
            player_type=peak["player_type"],
            champion_name=peak.get("champion", "?"),
            peak=peak,
            final_player=final,
            game_duration_min=game_duration_min,
            peer_baseline=peer,
        )
        # Attach ML pipeline columns captured during the game
        entry["role"]                 = peak.get("role", "")
        entry["game_time_at_peak"]    = peak.get("game_time_at_peak")
        entry["feature_vector_at_peak"] = peak.get("feature_vector")
        entry["n_signals_active"]     = peak.get("n_signals_active", 0)
        entries.append(entry)

    if not entries:
        logger.info("Post-game evaluation: no matching entries")
        return

    # Store all records in one DB session
    async for session in get_db_session():
        repo = PlayerRepository(session)
        for entry in entries:
            await repo.log_tilt_prediction(entry)
        await session.commit()

    # Log a human-readable game summary
    tp = sum(1 for e in entries if e["verdict"] == "true_positive")
    fp = sum(1 for e in entries if e["verdict"] == "false_positive")
    fn = sum(1 for e in entries if e["verdict"] == "false_negative")
    tn = sum(1 for e in entries if e["verdict"] == "true_negative")
    total = len(entries)

    logger.info("=" * 50)
    logger.info(f"POST-GAME TILT PREDICTION ACCURACY ({total} players)")
    logger.info(f"  True positives  (caught real tilt): {tp}")
    logger.info(f"  False positives (over-triggered):   {fp}")
    logger.info(f"  False negatives (missed tilt):      {fn}")
    logger.info(f"  True negatives  (correct silence):  {tn}")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        logger.info(f"  Precision: {precision:.0%}  ({tp}/{tp+fp} tilt alerts were real)")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        logger.info(f"  Recall:    {recall:.0%}  ({tp}/{tp+fn} real tilts were caught)")

    # Highlight the most interesting cases
    for e in sorted(entries, key=lambda x: x["peak_tilt_score"], reverse=True):
        if e["verdict"] in ("true_positive", "false_positive"):
            icon = "✓" if e["verdict"] == "true_positive" else "✗"
            logger.info(
                f"  [{icon}] {e['player_type'].upper()} {e['player_name']} ({e['champion_name']}) — "
                f"peak={e['peak_tilt_score']:.0%}, {e['verdict']}, "
                f"final KDA={e['final_kda']:.1f}, CS/min={e['final_cs_per_min']:.1f}"
            )
    logger.info("=" * 50)


def _build_post_game_report(
    peak_scores: dict,
    last_snapshot: dict | None,
    item_sell_timeline: dict,
    game_duration_min: float,
) -> dict:
    """
    Build a structured post-game report for every player.

    For each player we produce:
      - final stats (KDA, CS/min, ward score, etc.)
      - a chronological timeline of notable bad plays
      - a plain-language assessment paragraph

    The agent receives this, formats it, and saves it to disk.
    """
    if not last_snapshot:
        return {"type": "post_game_report", "players": [], "error": "No snapshot data available"}

    game_time      = last_snapshot.get("game_time", game_duration_min * 60)
    game_time_safe = max(game_time, 1.0)
    events         = last_snapshot.get("events", [])
    players        = last_snapshot.get("players", [])

    # Name sets for team membership checks (tag-stripped, lower-case)
    ally_names  = {p["summonerName"].split("#")[0].lower() for p in players if not p.get("is_enemy")}
    enemy_names = {p["summonerName"].split("#")[0].lower() for p in players if p.get("is_enemy")}

    player_reports = []
    for player in players:
        summoner   = player.get("summonerName", "?")
        name_clean = summoner.split("#")[0].lower()
        champion   = player.get("championName", "?")
        is_self    = player.get("is_self", False)
        is_enemy   = player.get("is_enemy", False)
        ptype      = "self" if is_self else ("enemy" if is_enemy else "ally")
        team_names = enemy_names if is_enemy else ally_names

        # Peak tilt info (exclude the debug sentinel key "__debug_logged__")
        peak        = peak_scores.get(summoner) if isinstance(peak_scores.get(summoner), dict) else None
        tilt_score  = peak["score"]     if peak else 0.0
        tilt_type   = peak["tilt_type"] if peak else "none"
        key_signals = peak["signals"]   if peak else []

        # ── Final stats ────────────────────────────────────────────────────────
        kills      = player.get("kills", 0)
        deaths     = player.get("deaths", 0)
        assists    = player.get("assists", 0)
        cs         = player.get("cs", 0)
        ward_score = player.get("ward_score", 0.0)
        final_stats = {
            "kills":        kills,
            "deaths":       deaths,
            "assists":      assists,
            "cs":           cs,
            "cs_per_min":   round(cs         / (game_time_safe / 60), 1),
            "ward_score":   round(ward_score, 1),
            "ward_per_min": round(ward_score / (game_time_safe / 60), 2),
            "kda":          round((kills + assists) / max(deaths, 1), 2),
        }

        # ── Timeline ───────────────────────────────────────────────────────────
        timeline = []

        # Deaths — annotated with repeat-killer and death-streak detection
        deaths_list = sorted(
            [e for e in events
             if e.get("type") == "ChampionKill"
             and e.get("victim", "").lower() == name_clean],
            key=lambda e: e.get("time", 0),
        )
        killer_counts: dict[str, int] = {}
        for i, ev in enumerate(deaths_list, 1):
            t      = ev.get("time", 0.0)
            killer = ev.get("killer", "Unknown")
            killer_counts[killer] = killer_counts.get(killer, 0) + 1
            count  = killer_counts[killer]

            severity = "low"
            note     = ""
            if count == 2:
                note     = " — 2nd death to same enemy"
                severity = "medium"
            elif count == 3:
                note     = " — 3rd death to same enemy ⚠ PRIDE SIGNAL"
                severity = "high"
            elif count > 3:
                note     = f" — {count}th death to same enemy ⚠"
                severity = "high"

            if i >= 2 and t - deaths_list[i - 2].get("time", 0) <= 180:
                note    += " | death streak"
                severity = "high"

            timeline.append({
                "time":     t,
                "time_fmt": f"{int(t // 60)}:{int(t % 60):02d}",
                "type":     "death",
                "severity": severity,
                "detail":   f"Death #{i} — killed by {killer}{note}",
            })

        # Item sells (timestamped as we received each snapshot)
        for sell in item_sell_timeline.get(summoner, []):
            t    = sell["time"]
            sold = sell["items"]
            timeline.append({
                "time":     t,
                "time_fmt": f"{int(t // 60)}:{int(t % 60):02d}",
                "type":     "item_sell",
                "severity": "high" if len(sold) >= 2 else "medium",
                "detail":   f"Sold {len(sold)} item(s) ⚠ tilt behavior",
            })

        # Objective absence — each drake/baron the team took without this player
        for ev in events:
            if ev.get("type") not in ("DragonKill", "BaronKill"):
                continue
            killer = ev.get("killer", "").lower()
            if killer not in team_names:
                continue   # other team's objective
            participated = (
                killer == name_clean
                or name_clean in [a.lower() for a in ev.get("assisters", [])]
            )
            if not participated:
                t        = ev.get("time", 0.0)
                obj_name = "Dragon" if ev.get("type") == "DragonKill" else "Baron"
                timeline.append({
                    "time":     t,
                    "time_fmt": f"{int(t // 60)}:{int(t % 60):02d}",
                    "type":     "objective_miss",
                    "severity": "medium",
                    "detail":   f"Team took {obj_name} — not present",
                })

        timeline.sort(key=lambda x: x["time"])

        player_reports.append({
            "summoner":    summoner,
            "champion":    champion,
            "player_type": ptype,
            "tilt_score":  round(tilt_score, 3),
            "tilt_type":   tilt_type,
            "final_stats": final_stats,
            "key_signals": key_signals,
            "obj_total":   player.get("obj_total", 0),
            "obj_missed":  player.get("obj_missed", 0),
            "timeline":    timeline,
            "assessment":  _build_assessment(tilt_type, key_signals, tilt_score, champion, ptype),
        })

    gm, gs = int(game_time // 60), int(game_time % 60)
    return {
        "type":              "post_game_report",
        "game_time_fmt":     f"{gm}:{gs:02d}",
        "game_duration_min": round(game_duration_min, 2),
        "players":           player_reports,
    }


def _build_assessment(
    tilt_type: str,
    signals: list[str],
    score: float,
    champion: str,
    player_type: str,
) -> str:
    """One-to-two sentence plain-language summary of what went wrong."""
    if score < 0.20:
        if player_type == "self":
            return "No significant tilt signals detected — you played with a clear head."
        return f"{champion} showed no significant tilt indicators this game."

    parts = []

    if any("same_enemy" in s for s in signals):
        n = next((s.split("same_enemy_")[1].rstrip("x") for s in signals if "same_enemy_" in s), None)
        target = f"{n} times" if n and n.isdigit() else "repeatedly"
        if player_type == "self":
            parts.append(f"You died to the same enemy {target} — consider changing your approach.")
        else:
            parts.append(f"Died to the same enemy {target} — emotionally fixated on that matchup.")

    if any("accelerating" in s for s in signals):
        parts.append(
            "Your death rate escalated as the game progressed." if player_type == "self"
            else "Death rate accelerated as the game progressed."
        )

    if any("cs_down" in s for s in signals):
        sig = next(s for s in signals if "cs_down" in s)
        pct = next((p for p in sig.split("_") if p.isdigit()), None)
        drop = f" by {pct}%" if pct else ""
        parts.append(
            f"CS dropped{drop} below expectations — focus on wave management." if player_type == "self"
            else f"CS performance dropped{drop} below rank average."
        )

    if any("kp_drop" in s or "kp_dropping" in s for s in signals):
        parts.append(
            "Kill participation fell in the second half — you disengaged from team fights." if player_type == "self"
            else "Kill participation dropped significantly in the second half."
        )

    if any("sold" in s for s in signals):
        parts.append(
            "Sold an item mid-game — a common stress response, try to recognize it next time." if player_type == "self"
            else "Sold item(s) mid-game — strong indicator of mental state collapse."
        )

    if any("absent" in s or "objective" in s for s in signals):
        parts.append(
            "Missed several team objectives — possible disengagement from the game." if player_type == "self"
            else "Was absent for most team objectives — likely disengaged."
        )

    if not parts:
        type_label = tilt_type if tilt_type not in ("none", "unknown") else "behavioral"
        return f"Tilt score reached {score:.0%} — {type_label} pattern detected."

    return " ".join(parts)
