# TiltRadar — Project Report

> Internal engineering document. Single source of truth for architecture,
> components, metrics, decisions, and next steps.

---

## 1. Project Overview

**TiltRadar** is a real-time emotional state detection system for competitive
gaming (League of Legends). It detects when players become emotionally frustrated
("tilted") during live matches by extracting behavioral signals every 5 seconds,
scoring frustration probability through a multi-stage ML pipeline, and surfacing
actionable insights through an in-game overlay.

The system runs a local agent alongside the game, streams snapshots over WebSocket
to a backend, scores all 10 players (self, allies, enemies), and returns tilt
reports with exploit recommendations. A self-labeling pipeline generates training
data from every game: post-game outcomes retroactively label whether tilt predictions
were correct, bootstrapping the ML models without manual annotation.

**Tech stack:** Python, FastAPI, WebSockets, PyTorch (GRU), XGBoost, scikit-learn,
ruptures (PELT change-point detection), PostgreSQL (async via asyncpg), SQLAlchemy 2.0,
Alembic, Redis + Celery, tkinter (overlay), Riot Games API.

**Repository:** https://github.com/mehdibourezza/tilt-radar

---

## 2. Architecture Map

```
┌─────────────────────────────────────────────────────────────────────┐
│  LOCAL AGENT (agent/)                                               │
│                                                                     │
│  Riot Live Client API (localhost:2999)                              │
│       |                                                             │
│       v                                                             │
│  local_agent.py ──> snapshot every 5s ──> WebSocket ──> Backend     │
│       ^                                                             │
│       |                                                             │
│  overlay.py <── tilt reports <── WebSocket <── Backend              │
│  (tkinter HUD: notification cards, color-coded by player type)     │
└─────────────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────────────┐
│  BACKEND API (api/)                                                 │
│                                                                     │
│  ws.py (WebSocket endpoint)                                        │
│    |                                                                │
│    ├── Fetch player baselines from DB                               │
│    ├── Fetch peer group baselines (rank-tier fallback)              │
│    |                                                                │
│    v                                                                │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  ML PIPELINE (ml/)                                        │      │
│  │                                                           │      │
│  │  Stage 1: TiltInferenceEngine (rule-based, 12 signals)   │      │
│  │       |                                                   │      │
│  │       v                                                   │      │
│  │  Stage 2: FeatureExtractor -> 26-dim normalized vector    │      │
│  │       |                                                   │      │
│  │       v                                                   │      │
│  │  Stage 3: SnapshotScorer (XGBoost + Platt calibration)   │      │
│  │       |                                                   │      │
│  │       v                                                   │      │
│  │  Stage 4: TemporalTiltModel (GRU, stateful per player)   │      │
│  │       |                                                   │      │
│  │       v                                                   │      │
│  │  Tilt score + type + signals + exploit recommendation     │      │
│  └───────────────────────────────────────────────────────────┘      │
│    |                                                                │
│    ├── Post-game: self-labeling pipeline (TiltPredictionLog)       │
│    └── Background: Celery workers ingest match history (Riot API)  │
└─────────────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────────────┐
│  DATA LAYER (data/)                                                 │
│                                                                     │
│  PostgreSQL:                                                        │
│    players, matches, player_match_stats, player_baselines,         │
│    peer_group_baselines, tilt_events, tilt_prediction_logs          │
│                                                                     │
│  Riot API Client (token-bucket rate limiter):                       │
│    Match v5, Spectator, Live Client Data, League entries            │
│                                                                     │
│  Redis + Celery:                                                    │
│    Background ingestion of match histories for all 10 players      │
└─────────────────────────────────────────────────────────────────────┘

LEGEND: All components [BUILT]. ML models not yet trained (need game data).
```

---

## 3. Component Registry

### 3.1 Tilt Inference Engine (`ml/inference/engine.py`)

- **Purpose:** Rule-based v1 tilt scoring. Extracts 12 behavioral signals from a live game snapshot and produces a weighted tilt score [0, 1] per player.
- **Inputs:** Game snapshot dict (players, events, game_time) + personal/peer baselines.
- **Outputs:** Per-player: tilt_score, tilt_type (rage/doom/pride/surrender), confidence, key_signals[], exploit recommendation.
- **12 Signals:**
  1. Repeat deaths to same enemy (pride-tilt)
  2. Death rate acceleration (late > early)
  3. CS drop vs baseline (personal or peer group)
  4. Early death cluster (4+ deaths before 10 min)
  5. Kill participation drop (early vs late half)
  6. Vision score rate very low
  7. Objective absence (missed 2/3+ team objectives)
  8. Item sell (1 = weak, 2+ = very strong rage signal)
  9. Level deficit (3+ below game average)
  10. Gold deficit per role (estimated vs expected by position)
  11. Wrong build (no signature items after 15 min)
  12. Respawn timer trend (escalating time spent dead)
- **Key decisions:**
  - Productive death filter: deaths that generated team value (turret, objective, trade kill within 30s) are excluded from signal scoring. Prevents penalizing dives and sacrifices.
  - Support/jungle exceptions: CS signals skip supports (intentionally low CS), objective signals get 2x weight for junglers.
  - Weak signal gate: if no single signal contributes >= 0.20, all weights are halved to prevent false positive stacking.
  - ItemRegistry (patch-aware, empirical) is preferred over hardcoded CHAMPION_SIGNATURE_ITEMS when available.
- **Files:** `engine.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.2 Feature Extractor (`ml/features/feature_extractor.py`)

- **Purpose:** Transform raw game snapshot + baselines into a 26-dimensional normalized feature vector for ML models.
- **Inputs:** Player dict, game_time, kill events, all 10 players, personal/peer baseline, snapshot history.
- **Outputs:** FeatureVector — numpy array of shape (26,), dtype float32, no NaN/Inf.
- **Feature groups:**
  - Group 1 (4 features): Baseline-normalized z-scores — CS, gold, KP, death rate deviation from personal or peer baseline. Clipped to [-3, 3].
  - Group 2 (11 features): Raw behavioral signals — repeat deaths, death acceleration, early deaths, KP drop, vision rate, objective absence, item sells, level deficit, gold deficit, build distance, dead time percentage. All in [0, 1].
  - Group 3 (4 features): Temporal deltas — change from previous snapshot (CS rate, death rate, 5-min death velocity, KP trend). Captures direction of performance.
  - Group 4 (5 features): Game context — normalized game time + role one-hot encoding (support, jungle, top, mid; ADC = all zeros).
  - Group 5 (2 features): Baseline quality indicator — has_personal_baseline flag + chronic_slump flag from PELT.
- **Key decisions:**
  - "Positive = worse" convention for z-scores: higher value always means more likely tilted, giving the model a consistent direction.
  - Missing baseline -> Group 1 defaults to 0.0 (neutral, not noisy), Group 5 tells the model to discount these features.
  - NaN/Inf failsafe at assembly: `np.nan_to_num()` ensures no garbage reaches the model.
- **Files:** `feature_extractor.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.3 Change-Point Detection (`ml/features/change_point.py`)

- **Purpose:** Detect shifts in a player's behavioral baseline using PELT (Pruned Exact Linear Time) algorithm from the `ruptures` library. Identifies when a player's performance regime changed (e.g., went from 7.1 CS/min to 4.2 CS/min at game 76 of 100).
- **Inputs:** Game history (list of per-game stats), ordered oldest to newest.
- **Outputs:** BaselineResult — true baseline start index, change points list, chronic slump flag, robust stats (median/IQR) computed on the current stable segment only.
- **Key decisions:**
  - Uses PELT with RBF (Radial Basis Function) cost model — detects both mean and variance shifts.
  - Penalty=3.0 (roughly requires a 3-sigma shift to trigger), min_segment_len=5 games.
  - True baseline = latest stable segment only (not full history average, which gets contaminated by old data).
  - Chronic slump = current segment CS median > 1.5 standard deviations below historical best.
- **Files:** `change_point.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.4 Snapshot Scorer — XGBoost (`ml/models/snapshot_scorer.py`)

- **Purpose:** Stage 2 of ML pipeline. Binary classifier: does this 26-dim feature vector predict poor performance? Uses XGBoost (gradient boosted decision trees).
- **Inputs:** Feature vector of shape (26,) from FeatureExtractor.
- **Outputs:** Calibrated P(performed_poorly) in [0, 1].
- **Architecture:** XGBoost (200 trees, max_depth=4, lr=0.05) + Platt scaling calibration (logistic regression on top of raw XGBoost output to calibrate probabilities).
- **Key decisions:**
  - XGBoost over logistic regression: captures signal interactions (e.g., death_accel AND repeat_deaths together = pride-tilt, much stronger than either alone). Trees naturally handle mixed feature types (continuous z-scores + binary flags).
  - Platt scaling on separate calibration set (not val set): avoids subtle data leakage from using the same set for early stopping and calibration.
  - SHAP (SHapley Additive exPlanations) values for interpretability: per-feature contribution to each prediction.
  - Hyperparams tuned for small datasets (200-2000 samples): shallow trees, aggressive regularization (L1/L2), stochastic subsampling.
- **Files:** `snapshot_scorer.py`
- **Status:** Built. Not yet trained (needs TiltPredictionLog data from games).
- **Last updated:** 2026-04-03

---

### 3.5 Temporal Model — GRU (`ml/models/temporal_model.py`)

- **Purpose:** Stage 3 of ML pipeline. Processes sequences of feature vectors (one per 5-second snapshot) and outputs P(tilted) as a real-time posterior that updates with each new snapshot.
- **Inputs:** Sequence of feature vectors, shape (T, 26) where T = 6-24 snapshots (30s to 2 min).
- **Outputs:** P(tilted) at each timestep; maintains per-player hidden state for O(1) live inference.
- **Architecture:** TiltGRU — 2-layer GRU (input=26, hidden=64) + LayerNorm + Linear(64->32) + ReLU + Dropout(0.3) + Linear(32->1) + Sigmoid. ~35k parameters.
- **Key decisions:**
  - GRU over LSTM or Transformer: short sequences (6-24 steps), small dataset (500-2000 labeled sequences), need <10ms CPU inference. GRU is lighter than LSTM with comparable performance. Transformer needs more data for attention to shine.
  - Per-timestep ramp labels (not last-state supervision): loss computed at every timestep, with a ramp from 0.0 (before tilt builds) to 1.0 (at peak). Forces the GRU to learn temporal dynamics, not just endpoint prediction.
  - Stateful inference: hidden state h_t retained between 5-second calls. O(1) per snapshot, ~0.2ms on CPU.
  - Training: AdamW, ReduceLROnPlateau, early stopping (patience=15), gradient clipping, BCEWithLogitsLoss with pos_weight for class imbalance.
- **Files:** `temporal_model.py`
- **Status:** Built. Not yet trained.
- **Last updated:** 2026-04-03

---

### 3.6 Training Pipeline (`ml/training/`)

- **Purpose:** Three-stage training: signal calibration -> XGBoost snapshot scorer -> GRU temporal model.
- **Data source:** TiltPredictionLog table — self-labeled after every game (peak tilt prediction vs final scoreboard outcome).
- **Sequence dataset builder:** Converts TiltPredictionLog entries with snapshot_sequence into numpy arrays + ramp labels for GRU training.
- **Key decisions:**
  - Ramp label construction: `ramp_start_fraction=0.60` assumes tilt starts building from 60% of game_time_at_peak. Configurable but not empirically validated (see Known Limitations).
  - Train/val/test split: grouped by date (no player leakage across splits).
- **Files:** `train.py`, `dataset.py`
- **Status:** Built. Not yet run (needs accumulated game data).
- **Last updated:** 2026-04-03

---

### 3.7 Evaluation (`ml/evaluation/`)

- **Purpose:** Comprehensive evaluation framework for tilt classifiers. Computes AUC-ROC, Brier score, ECE (Expected Calibration Error), precision/recall/F1 at operating threshold, time-to-detection, and per-signal accuracy.
- **Key metrics:**
  - Brier score: MSE between predicted probability and label (0.0 = perfect, 0.25 = always predicting 0.5). Catches calibration issues that AUC-ROC misses.
  - ECE: Expected Calibration Error — weighted average gap between predicted confidence and actual accuracy across probability bins. <0.05 = well-calibrated.
  - Signal calibration: evaluates which of the 12 engine signals actually correlate with poor outcomes (precision, recall, lift per signal).
- **Files:** `evaluator.py`, `signal_calibration.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.8 Local Agent (`agent/`)

- **Purpose:** Client-side process that runs alongside the game. Reads game state every 5 seconds from Riot's Live Client Data API (localhost:2999), streams snapshots over WebSocket to the backend, receives tilt reports, displays overlay.
- **Components:**
  - `local_agent.py` — main loop: poll -> extract snapshot -> send via WebSocket -> receive report -> update overlay. Handles reconnection, game start/end detection.
  - `live_client.py` — wrapper around Riot's Local Client API endpoints.
  - `overlay.py` — tkinter-based transparent overlay. Shows notification cards when tilt score crosses 50% (with 20% jump or new signals as anti-spam). Color-coded: enemies (orange/red), allies (cyan), self (purple). Auto-dismisses after 7 seconds.
- **Files:** `local_agent.py`, `live_client.py`, `overlay.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.9 Backend API (`api/`)

- **Purpose:** FastAPI WebSocket server. Receives snapshots, runs ML pipeline, returns tilt reports. Handles post-game self-labeling.
- **Endpoints:**
  - `ws://host:8001/ws/game` — WebSocket for live game analysis
- **Security:**
  - IP-based rate limiting (slowapi)
  - Strict Pydantic schema validation on all WebSocket inputs (type checks, length limits, bounded values)
  - Message size limits (100 KB), idle connection timeouts
  - CORS restricted to known origins, OWASP headers
  - Production validators in config (reject default secrets, empty API keys)
- **Files:** `main.py`, `routers/ws.py`, `schemas/validation.py`, `dependencies/db.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.10 Data Layer (`data/`)

- **Purpose:** PostgreSQL database (async via asyncpg) + Riot API client.
- **Database schema (7 tables):**
  - `players` — identity, rank, ingestion state
  - `matches` — one row per game, stores raw timeline JSON
  - `player_match_stats` — per-player per-game feature table (core stats + behavioral signals)
  - `player_baselines` — computed baseline profiles with change-point detection results
  - `peer_group_baselines` — aggregate stats per rank tier (fallback for unknown players)
  - `tilt_events` — recorded tilt episodes
  - `tilt_prediction_logs` — post-game labeled data for ML training (self-labeling pipeline)
- **Riot API client:** Token-bucket rate limiter (20/sec, 100/2min for dev keys). Endpoints: Match v5, Spectator, Live Client Data, League entries.
- **Files:** `db/models.py`, `db/repository.py`, `db/session.py`, `riot/client.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.11 Workers (`workers/`)

- **Purpose:** Celery async tasks for background match history ingestion. After a game starts, ingests the last 100 games for all 10 players via Riot API to build/update baselines.
- **Files:** `celery_app.py`, `tasks.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.12 Snapshot Buffer & Game Sequence Recorder (`ml/features/`)

- **Purpose:** SnapshotBuffer maintains rolling window of player snapshots for temporal delta features. GameSequenceRecorder records full game trajectories (feature vectors at each tick) for GRU training data stored in TiltPredictionLog.snapshot_sequence.
- **Files:** `snapshot_buffer.py`, `game_sequence_recorder.py`
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.13 Item Registry (`ml/data/item_registry.py`)

- **Purpose:** Patch-aware empirical item build data. Tracks which items are commonly built on each champion in each role. Used for "wrong build" signal detection (replaces hardcoded CHAMPION_SIGNATURE_ITEMS with data-driven detection).
- **Status:** Built.
- **Last updated:** 2026-04-03

---

### 3.14 Scripts (`scripts/`)

- `populate_peer_groups.py` — Seed PeerGroupBaseline table with rank-tier stats
- `ingest_self.py` — Ingest your own match history
- `check_rank.py`, `verify_summoner.py`, `debug_league.py` — Riot API debugging
- `test_overlay.py` — Test the overlay UI without a live game
- `create_tables.py` — Initialize database tables
- `my_progress.py` — Show your ingestion progress
- **Status:** Built.

---

### 3.15 Tests (`tests/`)

- `test_engine.py` — TiltInferenceEngine signal extraction tests
- `test_change_point.py` — PELT change-point detection tests
- `test_post_game_eval.py` — Post-game self-labeling pipeline tests
- **Framework:** pytest + pytest-asyncio
- **Status:** Built.
- **Last updated:** 2026-04-03

---

## 4. Evaluation & Metrics

### ML Models — Not Yet Trained

No model training has been performed yet. Training requires accumulated game data
in the `tilt_prediction_logs` table (from running the system during live games).

The self-labeling pipeline generates training data automatically from every game:
peak tilt predictions are compared against final scoreboard outcomes to produce
labeled (true_positive, false_positive, true_negative, false_negative) records.

### Target Metrics (from evaluation framework)

| Metric | Target | Notes |
|--------|--------|-------|
| AUC-ROC | > 0.70 | Ranking quality |
| Brier Score | < 0.15 | Calibration (0.25 = always predict 0.5) |
| ECE | < 0.10 | Expected Calibration Error |
| Precision @ 0.55 threshold | > 0.60 | Acceptable false positive rate |
| Recall @ 0.55 threshold | > 0.50 | Catch at least half of true tilts |

---

## 5. Technical Decisions Log

```
[2026-04] GRU over LSTM and Transformer for temporal model
  CHOSE: 2-layer GRU (64 hidden, ~35k params)
  OVER: LSTM (more params, no benefit on short sequences), Transformer (needs more data, sequences are 6-24 steps)
  BECAUSE: Short sequences, small expected dataset (500-2000), need <10ms CPU inference. GRU is the right balance of capacity and efficiency.

[2026-04] XGBoost over logistic regression for snapshot scoring
  CHOSE: XGBoost (200 trees, depth 4)
  OVER: Logistic regression (additive only), random forest (less regularization control)
  BECAUSE: Signals interact (death_accel + repeat_deaths = pride-tilt). Trees capture AND relationships naturally. Mixed feature types (z-scores + binary flags) handled without scaling.

[2026-04] PELT change-point detection for baselines, not rolling average
  CHOSE: PELT (ruptures library) on 100-game history per metric
  OVER: Rolling 20-game average, exponential moving average
  BECAUSE: Rolling averages are contaminated by regime changes. A player who tilts at game 76 and never recovers has their "baseline" dragged down gradually. PELT finds the exact breakpoint and isolates the current regime.

[2026-04] Self-labeling pipeline (no manual annotation)
  CHOSE: Post-game comparison of peak tilt score vs final scoreboard outcome
  OVER: Manual replay annotation, crowd-sourced labeling
  BECAUSE: Scalable — every game generates labeled data automatically. Tradeoff: labels are tied to final stats (not moment of tilt — see Known Limitations).

[2026-04] Platt scaling for probability calibration
  CHOSE: Logistic regression on separate calibration set (not val set)
  OVER: Isotonic regression, temperature scaling, no calibration
  BECAUSE: XGBoost probabilities are often overconfident. Separate cal set avoids leakage from using val set for both early stopping and calibration.

[2026-04] Per-timestep ramp labels for GRU training
  CHOSE: Smooth ramp from 0.0 (before 60% of game_time_at_peak) to 1.0 (at peak)
  OVER: Last-state supervision (loss only at final timestep), hard 0/1 step label
  BECAUSE: Last-state supervision lets the GRU ignore the trajectory. Ramp forces learning temporal dynamics. Ramp (not step) is honest encoding of uncertainty about exact tilt onset.

[2026-04] Productive death filter in signal extraction
  CHOSE: Exclude deaths that generated team value (kill, turret, objective within 30s)
  OVER: Count all deaths equally
  BECAUSE: Diving and dying for a turret is strategy, not tilt. Without this filter, aggressive playstyles generate false positives.
```

---

## 6. Tech Debt & Known Issues

1. **Labels tied to final stats, not moment of tilt (CRITICAL).** A player who tilts at minute 5 but recovers by minute 20 is labeled "not tilted." The model partially trains on noise. Fix: player self-report via overlay ("were you tilted right now?") or physiological signals.

2. **Gold estimation is invented.** `ROLE_GOLD_PER_MIN` is a hand-crafted table. Real gold varies by champion, items, game pace. Riot Live Client Data API doesn't expose current gold. Fix: Riot API expansion, or inference from item buy/sell events.

3. **1v1 vs 1v2 death ambiguity.** Productive death filter can't distinguish solo deaths from multi-enemy deaths without positional data. Fix: post-game timeline retroactive labeling.

4. **Selection bias in calibration.** Signal calibration only runs on games where tilt fired — survivorship bias. Fix: lower logging threshold to 0.3, supplement with scraped data.

5. **Ramp label start fraction is arbitrary.** `ramp_start_fraction=0.60` has no empirical basis. Fix: player self-report annotations or physiological signals for tilt onset.

6. **Peer baselines require manual seeding.** PeerGroupBaseline table must be populated by running `scripts/populate_peer_groups.py` after each patch. Without it, rank-tier fallback uses rough heuristics.

7. **Champion signature items are hardcoded.** CHAMPION_SIGNATURE_ITEMS covers ~40 champions and becomes stale after item reworks. ItemRegistry (empirical, patch-aware) is the fix, but needs data to populate.

8. **No CI/CD pipeline.** No automated testing or deployment.

---

## 7. What's Next

1. **Accumulate game data** — Run the system during live games to build up TiltPredictionLog entries for ML training. Need ~200+ games minimum.
2. **Train XGBoost snapshot scorer** — First ML model to train (needs data from step 1).
3. **Train GRU temporal model** — Second ML model (needs more data, ~500+ game sequences).
4. **Signal calibration study** — Evaluate which of the 12 signals actually predict poor outcomes.
5. **Populate peer group baselines** — Run `populate_peer_groups.py` to seed rank-tier stats.
6. **Docker deployment** — Use `docker/docker-compose.yml` for reproducible setup.
7. **W&B experiment tracking** — Log training runs and signal calibration results.
8. **README polish** — Architecture diagrams, demo screenshots for GitHub.

---

## 8. Environment & Setup

### Prerequisites
- Python 3.11+
- Docker (for PostgreSQL + Redis) or local installs
- Riot Games API key (https://developer.riotgames.com)

### Setup
```bash
cd C:/Users/boure/tilt-radar
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set RIOT_API_KEY, region, platform

# Start infrastructure
docker compose -f docker/docker-compose.yml up postgres redis -d

# Create tables
python scripts/create_tables.py
```

### Run (3 terminals)
```bash
# Terminal 1 — Backend API
uvicorn api.main:app --host 0.0.0.0 --port 8001

# Terminal 2 — Celery worker (background match ingestion)
celery -A workers.celery_app worker --loglevel=info

# Terminal 3 — Local agent (start before or during a game)
python -m agent.local_agent --summoner "YourName" --tag "EUW" --server ws://localhost:8001
```

### Train Models
```bash
# After accumulating game data:
python -m ml.training.train --mode all
# Outputs: experiments/signal_calibration.json, snapshot_scorer.pkl, temporal_model.pt
```

### Run Tests
```bash
pytest tests/ -v
```
