"""
Tilt Inference Engine.

Takes a live game snapshot (from the local agent) + enemy baselines (from DB)
and produces a tilt report for each enemy player.

This is the rule-based v1. Later this will be replaced/augmented by the
trained Temporal Transformer model. Starting rule-based lets us:
  - Validate the pipeline end-to-end immediately
  - Generate labeled data for ML training (we log every score + outcome)
  - Ship something useful before the ML model is trained
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tilt score thresholds
TILT_HIGH   = 0.70
TILT_MEDIUM = 0.45

# Expected total gold income per minute by role.
# Accounts for base passive income (~120g/min) + typical CS + kill/assist share.
# Used to estimate whether a player is gold-starved relative to their role.
ROLE_GOLD_PER_MIN: dict[str, int] = {
    "TOP":     400,
    "JUNGLE":  360,
    "MIDDLE":  430,
    "BOTTOM":  450,
    "UTILITY": 270,   # support — low CS, mainly passive + assists
    "":        380,   # unknown role fallback
}

# Champion → set of item IDs that represent a correct core build.
# Signal fires when a player has NONE of these after 15 min with ≥2 items built.
# IDs target Season 14/15 items — comment out and recheck after major item reworks.
CHAMPION_SIGNATURE_ITEMS: dict[str, set[int]] = {
    # ── ADC ──────────────────────────────────────────────────────────────────
    "Jinx":         {3031, 3085, 3046, 6675},
    "Caitlyn":      {6675, 3031, 3094},
    "Jhin":         {3094, 3095, 3142},
    "Ashe":         {6675, 3031, 3085},
    "Ezreal":       {3003, 3078, 3153},
    "Lucian":       {6672, 3031},
    "Samira":       {6672, 3031, 3036},
    "Xayah":        {6675, 3031, 3046},
    "Miss Fortune": {6672, 3031, 3036},
    "Sivir":        {3031, 3085, 3094},
    "Kog'Maw":      {3085, 3031},
    "Vayne":        {3153, 3031, 6675},
    "Tristana":     {3031, 3085, 6675},
    # ── Mid ──────────────────────────────────────────────────────────────────
    "Ahri":         {3165, 3157},
    "Zed":          {3142, 3814, 3147},
    "Syndra":       {3165, 3089},
    "Lux":          {3157, 3135},
    "Viktor":       {3089, 3135},
    "Orianna":      {3165, 3089},
    "Veigar":       {3165, 3089},
    "Yasuo":        {3031, 3046},
    "Yone":         {6675, 3031, 3046},
    "Katarina":     {3165, 3089},
    "Akali":        {3814, 3142},
    "LeBlanc":      {3165, 3089},
    "Cassiopeia":   {3165, 3089, 3135},
    # ── Jungle ───────────────────────────────────────────────────────────────
    "Lee Sin":      {3142, 3814},
    "Hecarim":      {3078, 3742, 3153},
    "Graves":       {3142, 3147},
    "Vi":           {3153, 3748},
    "Kha'Zix":      {3142, 3814, 3147},
    "Amumu":        {3083, 3190},
    "Warwick":      {3153, 3748, 3078},
    "Jarvan IV":    {3071, 3742, 3083},
    # ── Top ──────────────────────────────────────────────────────────────────
    "Darius":       {3071, 3078, 3748},
    "Garen":        {3742, 3748, 3083},
    "Fiora":        {3078, 3742, 3153},
    "Jax":          {3078, 3153},
    "Malphite":     {3083, 3190, 3742},
    "Nasus":        {3083, 3742},
    "Sett":         {3748, 3071, 3083},
    "Renekton":     {3078, 3071, 3742},
    "Urgot":        {3071, 3748},
    # ── Support ──────────────────────────────────────────────────────────────
    "Thresh":       {3109, 3050, 3190},
    "Lulu":         {3504, 2065, 3050},
    "Nautilus":     {3109, 3190, 3083},
    "Blitzcrank":   {3109, 3190},
    "Leona":        {3190, 3109, 3083},
    "Soraka":       {2065, 3504},
    "Nami":         {3504, 2065},
    "Janna":        {3504, 2065, 3190},
    "Morgana":      {3165, 3190, 3109},
    "Sona":         {3504, 2065, 3089},
}


@dataclass
class PlayerTiltScore:
    summoner_name: str
    champion_name: str
    player_type: str            # self | ally | enemy
    tilt_score: float           # 0.0 – 1.0
    tilt_type: str              # rage | doom | pride | unknown
    confidence: float           # 0.0 – 1.0  (lower when no baseline)
    key_signals: list[str]
    exploit: str | None


class TiltInferenceEngine:
    """
    Scores each enemy player's tilt level from a game snapshot.

    Signal extraction:
      - Repeat deaths to same enemy (pride-tilt indicator)
      - Death acceleration (rate increasing over game time)
      - CS drop vs baseline (doom-tilt indicator)
      - Objective desperation (contesting when behind)

    Each signal contributes a weighted score. Final score is clipped to [0, 1].

    Wrong build signal uses ItemRegistry (empirical, patch-aware) when available,
    falls back to hardcoded CHAMPION_SIGNATURE_ITEMS otherwise.
    Pass item_registry=ItemRegistry.load(...) to the constructor to enable it.
    """

    # Signal weights — tuned heuristically, will be learned by the ML model later
    WEIGHTS = {
        "repeat_deaths_same_enemy": 0.30,
        "death_acceleration":       0.25,
        "cs_drop_vs_baseline":      0.25,
        "early_death_cluster":      0.20,
        "kill_participation_drop":  0.20,
        "vision_score_low":         0.15,
        "objective_absence":        0.20,
        "item_sell":                0.35,
        # New signals
        "level_deficit":            0.15,
        "gold_deficit_per_role":    0.20,
        "wrong_build":              0.25,
        "respawn_timer_trend":      0.20,
    }

    def __init__(self, item_registry=None) -> None:
        """
        Args:
            item_registry: Optional ItemRegistry instance for patch-aware wrong build
                           detection. If None, falls back to CHAMPION_SIGNATURE_ITEMS.
        """
        self._item_registry = item_registry

    def score(self, snapshot: dict, personal_baselines: dict, peer_baselines: dict = None) -> dict:
        """
        Score all 10 players from a single snapshot.

        snapshot: dict from local agent (players, events, game_time)
          Each player in snapshot["players"] has is_self and is_enemy flags.
        personal_baselines: {puuid_or_name: PlayerBaseline} — for self (own history)
        peer_baselines: {puuid_or_name: PeerGroupBaseline} — rank-tier fallback for all 9 others

        Baseline priority:
          1. Personal baseline (most accurate — based on their own history)
          2. Peer group baseline (fallback — compares against rank average)
          3. Signal-only (no baseline — lowest confidence)

        Returns a tilt report dict ready to be JSON-serialized and sent back.
        """
        if peer_baselines is None:
            peer_baselines = {}

        game_time = snapshot.get("game_time", 0)
        events = snapshot.get("events", [])
        players = snapshot.get("players", [])
        kill_events = [e for e in events if e.get("type") == "ChampionKill"]

        results = []
        for player in players:
            name = player.get("summonerName", "unknown")
            champion = player.get("championName", "unknown")
            puuid = player.get("puuid")

            is_self = player.get("is_self", False)
            is_enemy = player.get("is_enemy", False)
            player_type = "self" if is_self else ("enemy" if is_enemy else "ally")

            personal = personal_baselines.get(puuid) or personal_baselines.get(name)
            peer = peer_baselines.get(puuid) or peer_baselines.get(name)
            active_baseline = personal or peer

            signals, signal_weights = self._extract_signals(
                player, kill_events, game_time, active_baseline,
                is_peer=(personal is None and peer is not None),
                all_players=players,
                all_events=events,
            )

            tilt_score = min(1.0, sum(signal_weights.values()))
            tilt_type = self._classify_type(signals)
            confidence = self._compute_confidence(personal, peer, game_time)
            exploit = self._recommend_exploit(tilt_type, tilt_score, champion, player_type)

            results.append({
                "summonerName": name,
                "championName": champion,
                "player_type": player_type,
                "tilt_score": round(tilt_score, 3),
                "tilt_type": tilt_type,
                "confidence": round(confidence, 3),
                "key_signals": signals,
                "exploit": exploit if tilt_score >= TILT_MEDIUM else None,
            })

        game_time_min = int(game_time // 60)
        game_time_sec = int(game_time % 60)

        # Keep "enemies" key for backward compat, add "players" for full picture
        enemy_results = [r for r in results if r["player_type"] == "enemy"]
        return {
            "game_time": game_time,
            "game_time_fmt": f"{game_time_min}:{game_time_sec:02d}",
            "players": results,
            "enemies": enemy_results,
        }

    @staticmethod
    def _is_productive_death(
        death_time: float,
        dead_team: str,
        all_events: list,
        name_to_team: dict,
        window: float = 30.0,
    ) -> bool:
        """
        Returns True if a death generated value for the dead player's team within `window` seconds.

        A death is productive if shortly after it, the dead player's team:
          - Got a champion kill (trade / dive)
          - Killed a turret (cross-map play)
          - Secured an objective (drake, baron, herald, inhibitor)

        This prevents penalizing dives, sacrifices, and cross-map rotations.
        Example: top laner dives and dies, team takes bot turret + double kill → productive.

        # TODO (flaw 5): The productive death filter currently can't distinguish
        # 1v1 deaths from 1v2/1v3 deaths. A 1v1 death is NEVER productive.
        # A 1v2 or 1v3 death that leads to a cross-map objective within ~15s IS productive.
        # Fix requires: tracking participant counts at each kill event using the event
        # timeline + champion position data. The Riot Live Client Data API does not
        # expose participant counts directly — this requires position inference from
        # the allGameData endpoint and cross-referencing with kill events.
        # Until implemented, all deaths to the same enemy count against the player
        # regardless of how many enemies were present.
        """
        productive_types = {
            "ChampionKill", "TurretKilled", "DragonKill",
            "BaronKill", "RiftHeraldKill", "InhibKilled",
        }
        for event in all_events:
            t = event.get("time", 0)
            if not (death_time < t <= death_time + window):
                continue
            if event.get("type") not in productive_types:
                continue
            killer = event.get("killer", "").lower()
            if name_to_team.get(killer) == dead_team:
                return True
        return False

    def _extract_signals(
        self,
        enemy: dict,
        kill_events: list[dict],
        game_time: float,
        baseline,           # PlayerBaseline or PeerGroupBaseline ORM object or None
        is_peer: bool = False,
        all_players: list[dict] | None = None,
        all_events: list[dict] | None = None,
    ) -> tuple[list[str], dict[str, float]]:
        """Extract tilt signals for one enemy. Returns (signal names, weighted scores)."""
        name = enemy.get("summonerName", "")
        cs = enemy.get("cs", 0)
        position = enemy.get("position", "").upper()
        is_support = position == "UTILITY"
        is_jungle  = position == "JUNGLE"
        dead_team  = enemy.get("team", "")

        signals = []
        weights = {}
        _all_events = all_events or []

        # Kill events use game name only (no #tag), but summonerName includes the tag.
        # Strip the tag for event comparisons: "Fenixazz#AXIS" → "Fenixazz"
        name_for_events = name.split("#")[0]

        # Build tag-stripped name → team lookup for productive death checks.
        name_to_team: dict[str, str] = {
            p.get("summonerName", "").split("#")[0].lower(): p.get("team", "")
            for p in (all_players or [])
        }

        def _productive(e: dict) -> bool:
            return self._is_productive_death(
                e.get("time", 0), dead_team, _all_events, name_to_team
            )

        # --- Signal 1: Repeat deaths to the same enemy (pride-tilt) ---
        # Productive deaths (trade/dive that got team value) are excluded — dying to the
        # same person 3 times while enabling cross-map plays is not pride, it's strategy.
        death_sources = [
            e.get("killer") for e in kill_events
            if e.get("victim", "").lower() == name_for_events.lower()
            and not _productive(e)
        ]
        if death_sources:
            from collections import Counter
            max_deaths_to_one_enemy = Counter(death_sources).most_common(1)[0][1]
            if max_deaths_to_one_enemy >= 3:
                signals.append(f"died_to_same_enemy_{max_deaths_to_one_enemy}x")
                weights["repeat_deaths_same_enemy"] = self.WEIGHTS["repeat_deaths_same_enemy"]
            elif max_deaths_to_one_enemy == 2:
                signals.append("died_to_same_enemy_2x")
                weights["repeat_deaths_same_enemy"] = self.WEIGHTS["repeat_deaths_same_enemy"] * 0.5

        # --- Signal 2: Death acceleration ---
        # Compare unproductive deaths in first half vs second half of game so far.
        # Requires late deaths to exceed early by at least 2 (not just 1) to avoid noise.
        if game_time > 600:
            midpoint = game_time / 2
            early_deaths = sum(
                1 for e in kill_events
                if e.get("victim", "").lower() == name_for_events.lower()
                and e.get("time", 0) < midpoint
                and not _productive(e)
            )
            late_deaths = sum(
                1 for e in kill_events
                if e.get("victim", "").lower() == name_for_events.lower()
                and e.get("time", 0) >= midpoint
                and not _productive(e)
            )
            if late_deaths >= 3 and late_deaths > early_deaths + 2:
                signals.append("death_rate_accelerating")
                weights["death_acceleration"] = self.WEIGHTS["death_acceleration"]

        # --- Signal 3: CS drop vs baseline (personal or peer group) ---
        # Skipped for supports — their CS is intentionally low and not a tilt signal.
        # Only fires on a clear deviation (>1.5 IQR) — the "slightly below" variant was too noisy.
        if not is_support and baseline and game_time > 420:
            current_cs_per_min = cs / (game_time / 60) if game_time > 0 else 0

            # Attribute names differ between PersonalBaseline and PeerGroupBaseline
            if is_peer:
                expected = getattr(baseline, "cs_per_min_median", None)
                iqr = getattr(baseline, "cs_per_min_iqr", None) or 1.0
                label_suffix = "vs_tier_avg"
            else:
                expected = getattr(baseline, "lt_cs_per_min_median", None)
                iqr = getattr(baseline, "lt_cs_per_min_iqr", None) or 1.0
                label_suffix = "vs_baseline"

            if expected:
                deviation = (expected - current_cs_per_min) / iqr if iqr > 0 else 0
                if deviation > 1.5:
                    drop_pct = int((1 - current_cs_per_min / expected) * 100)
                    signals.append(f"cs_down_{drop_pct}pct_{label_suffix}")
                    weights["cs_drop_vs_baseline"] = self.WEIGHTS["cs_drop_vs_baseline"]

        # --- Signal 4: Early death cluster (4+ unproductive deaths before 10 min) ---
        # Raised from 3 to 4: 3 deaths in 10 min can happen to a support or jungler in a bad
        # but non-tilted game. 4+ is a clearer sign of something wrong.
        # Productive deaths (dives, trades) are excluded.
        early_deaths_count = sum(
            1 for e in kill_events
            if e.get("victim", "").lower() == name_for_events.lower()
            and e.get("time", 0) < 600
            and not _productive(e)
        )
        if early_deaths_count >= 4:
            signals.append(f"{early_deaths_count}_deaths_before_10min")
            weights["early_death_cluster"] = self.WEIGHTS["early_death_cluster"]

        # --- Signal 5: Kill participation drop (early half → late half) ---
        # Requires 20 min of game time to have stable KP windows.
        kp_early = enemy.get("kp_early")
        kp_late  = enemy.get("kp_late")
        if kp_early is not None and kp_late is not None and game_time > 1200:
            if kp_early >= 0.35 and kp_late < kp_early * 0.5:
                drop_pct = int((1 - kp_late / kp_early) * 100)
                signals.append(f"kp_dropped_{drop_pct}pct")
                weights["kill_participation_drop"] = self.WEIGHTS["kill_participation_drop"]

        # --- Signal 6: Vision score rate very low ---
        # Only fires after 20 min (ward habits take time to establish) and only on clear outliers.
        # Removed the weak "vision_low" variant — it fired on almost everyone before 20 min.
        ward_score = enemy.get("ward_score", 0.0)
        if game_time > 1200:
            vision_per_min = ward_score / (game_time / 60)
            if vision_per_min < 0.20:
                signals.append(f"vision_very_low_{vision_per_min:.1f}_per_min")
                weights["vision_score_low"] = self.WEIGHTS["vision_score_low"]

        # --- Signal 7: Objective absence (team took drake/baron without them) ---
        # Requires 3+ objectives so a single missed drake doesn't flag (could be recall timing).
        # Jungler gets a 2× weight multiplier — missing objectives is literally their primary job.
        obj_total  = enemy.get("obj_total", 0)
        obj_missed = enemy.get("obj_missed", 0)
        if obj_total >= 3:
            absence_rate = obj_missed / obj_total
            if absence_rate >= 0.67:
                signals.append(f"absent_{obj_missed}of{obj_total}_objectives")
                obj_weight = self.WEIGHTS["objective_absence"] * (2.0 if is_jungle else 1.0)
                weights["objective_absence"] = obj_weight

        # --- Signal 8: Item sell ---
        # Selling a completed item mid-game is rare and almost always emotional.
        # Selling 1 = weak signal; 2+ = very strong (rage-quitting items).
        sold_items = enemy.get("sold_items", [])
        if len(sold_items) >= 2:
            signals.append(f"sold_{len(sold_items)}_items")
            weights["item_sell"] = self.WEIGHTS["item_sell"]
        elif len(sold_items) == 1:
            signals.append("sold_1_item")
            weights["item_sell"] = self.WEIGHTS["item_sell"] * 0.55

        # --- Signal 9: Level deficit ---
        # Skipped for supports — they intentionally cede XP to scaling carries.
        # Raised threshold: 3+ levels behind avg (was 2.5) to avoid flagging normal losing laners.
        if not is_support and all_players and game_time > 600:
            avg_level = sum(p.get("level", 1) for p in all_players) / max(len(all_players), 1)
            level = enemy.get("level", 1)
            deficit = avg_level - level
            if deficit >= 3.0:
                signals.append(f"level_deficit_{deficit:.1f}")
                weights["level_deficit"] = self.WEIGHTS["level_deficit"]

        # --- Signal 10: Estimated gold deficit per role ---
        # Skipped for supports — their gold comes from support items/assists which we can't
        # accurately estimate without the full gold API. KDA signals cover them instead.
        # Raised strong threshold to 0.55 (was 0.65) and removed the weak variant.
        if not is_support and game_time > 600:
            game_time_min  = game_time / 60
            kills_g        = enemy.get("kills", 0)
            assists_g      = enemy.get("assists", 0)
            cs_g           = enemy.get("cs", 0)
            estimated_gold = cs_g * 20 + kills_g * 300 + assists_g * 150 + game_time_min * 120
            expected_gold  = ROLE_GOLD_PER_MIN.get(position, 380) * game_time_min
            if expected_gold > 0:
                gold_ratio = estimated_gold / expected_gold
                if gold_ratio < 0.55:
                    deficit_pct = int((1 - gold_ratio) * 100)
                    signals.append(f"gold_deficit_{deficit_pct}pct_vs_{position.lower() or 'role'}")
                    weights["gold_deficit_per_role"] = self.WEIGHTS["gold_deficit_per_role"]

        # --- Signal 11: Wrong build ---
        # If a player has NONE of their champion's expected items after 15 min with ≥2 items built,
        # they are either trolling, panicking, or improvising under stress.
        # Uses empirical ItemRegistry (patch-aware) if available; falls back to hardcoded map.
        champion_name = enemy.get("championName", "")
        position      = enemy.get("position", "")
        if game_time > 900:
            current_item_set = set(enemy.get("items", []))
            if len(current_item_set) >= 2:
                wrong = False
                if self._item_registry is not None:
                    # Empirical path: item names from registry
                    current_item_names = enemy.get("item_names", [])
                    wrong = self._item_registry.is_wrong_build(
                        champion_name, position, current_item_names
                    )
                else:
                    # Fallback: hardcoded item ID set
                    expected_items = CHAMPION_SIGNATURE_ITEMS.get(champion_name, set())
                    wrong = bool(expected_items) and not bool(current_item_set & expected_items)

                if wrong:
                    signals.append(f"wrong_build_{champion_name.lower().replace(' ', '_')}")
                    weights["wrong_build"] = self.WEIGHTS["wrong_build"]

        # --- Signal 12: Respawn timer trend ---
        # Estimate total time spent dead from death event timestamps.
        # LoL respawn timer grows with game time: roughly 7s at 0 min → 50s at 35 min.
        # If a player is spending an escalating share of game time dead, it signals
        # desperate overplaying or giving up (dying on purpose to end faster).
        # Only count unproductive deaths for respawn time calculations —
        # a player who died diving for a baron shouldn't be penalised for "time spent dead".
        player_death_times = [
            e.get("time", 0.0) for e in kill_events
            if e.get("victim", "").lower() == name_for_events.lower()
            and not _productive(e)
        ]
        if player_death_times and game_time > 600:
            def _est_respawn(t: float) -> float:
                return 7.0 + (t / 60.0) * 1.25   # rough scaling

            total_dead_time = sum(_est_respawn(t) for t in player_death_times)
            dead_pct        = total_dead_time / game_time

            if game_time > 1200:
                # Compare first half vs second half dead time %
                early_dead = sum(_est_respawn(t) for t in player_death_times if t <  game_time / 2)
                late_dead  = sum(_est_respawn(t) for t in player_death_times if t >= game_time / 2)
                half_dur   = game_time / 2
                early_pct  = early_dead / half_dur
                late_pct   = late_dead  / half_dur
                if late_pct > 0.25 and late_pct > early_pct * 2.0:
                    signals.append(f"respawn_escalating_{late_pct:.0%}_dead_2nd_half")
                    weights["respawn_timer_trend"] = self.WEIGHTS["respawn_timer_trend"]
                    dead_pct = -1.0   # prevent double-count below

            if dead_pct > 0.22:
                signals.append(f"spending_{dead_pct:.0%}_of_game_dead")
                weights["respawn_timer_trend"] = self.WEIGHTS["respawn_timer_trend"] * 0.65

        # --- Gate: require at least one meaningful signal ---
        # If every signal is a weak variant (no single signal contributes ≥ 0.20),
        # this is likely noise — cap the score to prevent false positives from stacking.
        if weights and max(weights.values()) < 0.20:
            # All signals are weak — halve all contributions
            weights = {k: v * 0.5 for k, v in weights.items()}

        return signals, weights

    def _classify_type(self, signals: list[str]) -> str:
        """Classify tilt type from active signals."""
        if not signals:
            return "none"
        # Surrender: sold items + checked-out behavior → mentally given up
        sold     = any("sold" in s for s in signals)
        absent   = any("objective" in s or "absent" in s for s in signals)
        kp_drop  = any("kp_drop" in s or "kp_dropping" in s for s in signals)
        if sold and (absent or kp_drop):
            return "surrender"
        if any("same_enemy" in s for s in signals):
            return "pride"       # locked onto one target emotionally
        if sold or (kp_drop and absent):
            return "doom"        # checked out, passive/disengaged
        if any("cs_down" in s for s in signals) and any("death" in s for s in signals):
            return "doom"
        if any("accelerating" in s for s in signals):
            return "rage"        # escalating aggression
        return "rage"            # default — more likely than doom with death signals

    def _compute_confidence(self, personal_baseline, peer_baseline, game_time: float) -> float:
        """
        Confidence in the tilt score.
        Personal baseline > peer group baseline > no baseline.
        """
        if personal_baseline is not None:
            confidence = 1.0        # best case: compared against their own history
        elif peer_baseline is not None:
            confidence = 0.70       # good case: compared against rank average
        else:
            confidence = 0.40       # worst case: signal-only, no reference

        if game_time < 300:
            confidence *= 0.4       # less than 5 minutes — too early
        elif game_time < 600:
            confidence *= 0.7       # 5-10 min — building picture

        return round(confidence, 2)

    def _recommend_exploit(self, tilt_type: str, score: float, champion: str, player_type: str = "enemy") -> str | None:
        if score < TILT_MEDIUM:
            return None
        if player_type == "self":
            advice = {
                "pride":     "You keep dying to the same player — disengage and reset. Take a different angle.",
                "doom":      "Your CS is dropping — focus on wave clear and skip risky fights for now.",
                "rage":      "Death rate is climbing — slow down, play safe for 2 waves, then re-engage.",
                "surrender": "You're showing signs of giving up (sold item, missing objectives). Take a breath — one good fight can reset everything.",
                "none":      None,
            }
            return advice.get(tilt_type)
        if player_type == "ally":
            advice = {
                "pride":     f"{champion} (ally) is fixated on one enemy — they may overcommit. Peel for them.",
                "doom":      f"{champion} (ally) looks checked out — don't rely on them for carries this fight.",
                "rage":      f"{champion} (ally) is escalating. Keep them on a short leash, avoid coinflip plays.",
                "surrender": f"{champion} (ally) has mentally checked out. Play around the other 3 for now.",
                "none":      None,
            }
            return advice.get(tilt_type)
        # Enemy
        exploits = {
            "pride":     f"Keep sending {champion} back to the same matchup — they will overcommit to fight back.",
            "doom":      f"{champion} is checked out. Force skirmishes near them — they won't respond.",
            "rage":      f"{champion} is playing emotionally. Bait them with an overextend then punish.",
            "surrender": f"{champion} has mentally given up. Invade their jungle, contest vision — they won't contest.",
            "none":      None,
        }
        return exploits.get(tilt_type)
