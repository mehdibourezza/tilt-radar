# TiltRadar — Known Limitations

This file documents fundamental flaws that **cannot be fixed** with the current
data and infrastructure. Each entry explains the problem, why it can't be fixed,
and what would be required to address it in the future.

---

## 1. The Label Problem (Critical)

**What it is**: `performed_poorly` is computed from *final* game stats (KDA,
CS, death rate averaged over the full game). A player who tilts at minute 5
but recovers by minute 20 and finishes 5/3/10 is labeled "not tilted."
The model is partially trained on noise.

**Why it can't be fixed**: We have no ground truth for *when* a player tilted.
Riot does not expose behavioral annotations. The only path forward is humans
annotating replays — not scalable.

**What would help**: Riot releasing behavioral data, or player self-report
("I was tilted from minute 5 to 15"). The TiltRadar overlay itself could
crowdsource this: after the tilt alert fires, ask "were you actually tilting
right now?" — one click creates a labeled training sample.

---

## 2. Gold Estimation is Invented

**What it is**: `ROLE_GOLD_PER_MIN` in engine.py is a hand-crafted table.
Real gold income varies by champion, itemization, jungle path, and game pace.
The gold deficit signal compares against a fictional baseline.

**Why it can't be fixed**: The Riot Live Client Data API does not expose
current gold directly. The Match v5 API has post-game `goldEarned` but no
per-minute curves.

**What would help**: Riot expanding the Live Client Data API to include current
gold. Alternatively, approximating gold from item buy/sell events in the
timeline (item sell events contain gold amounts).

---

## 3. Productive Death Filter — 1v1 vs 1v2/1v3

**What it is**: The engine can't distinguish a 1v1 death (never productive)
from a 1v2 or 1v3 death that bought cross-map value (productive). A TODO
comment exists in engine.py at the filter site.

**Why it can't be fixed in real-time**: The Live Client Data API does not
expose participant counts at kill events. Inferring them requires tracking all
10 player positions at each tick and cross-referencing with kill timestamps —
the API doesn't provide tick-resolution positional data.

**What would help**: Riot's full match timeline (post-game only) has
participant frames with positions. This could retroactively label productive
deaths in historical data, but can't work live.

---

## 4. Selection Bias in the Calibration Study

**What it is**: `SignalCalibration` only runs on games where the tilt engine
fired. Games where tilt never reached the threshold never enter
`TiltPredictionLog`. The calibration study answers "which signals are predictive
*given the engine already flagged something?*" — not "which signals predict
poor performance in general." This is survivorship bias.

**Why it can't be fully fixed**: A full fix requires logging every game
regardless of score. Currently the engine only runs during active TiltRadar
sessions.

**Partial mitigation**: Lower the logging threshold to 0.3. Use
`scraper.py` to supplement with scraped data from diverse players.
Stratify calibration results by `peak_tilt_score` bins.

---

## 5. Ramp Label Start Fraction is Arbitrary

**What it is**: `ramp_start_fraction=0.60` in `_ramp_labels()` assumes tilt
starts building from 60% of `game_time_at_peak`. This is a reasonable guess
but has no empirical basis. The parameter is now configurable in
`build_sequence_dataset()` but we still can't validate the default.

**Why it can't be validated**: Validating the ramp start requires labeled
timestamps of *when* tilt actually began — which is the same label problem
as issue 1.

**What would help**: Player self-report annotations, or physiological signals
(heart rate, GSR) as a proxy for tilt onset.

---

## 6. ELO Bias in Live Baseline Normalization — DATA ONLY (code is done)

**What it is**: Each player's rank IS detected at game start via
`/lol/league/v4/entries/by-summoner/{summonerId}` and their tier-appropriate
`PeerGroupBaseline` IS loaded in `ws.py`. The code is correct.

**The actual blocker**: `PeerGroupBaseline` rows must be populated by running
`scripts/populate_peer_groups.py`. Until that script runs, the table is empty
and `ws.py` logs `"No peer baseline for {tier} {division} — run populate_peer_groups.py"`,
falling back to rough KDA heuristics for all non-self players.

**What to do**: Run `populate_peer_groups.py` once to seed the table, then
re-run it after major patches (patch numbers change CS/gold norms).
