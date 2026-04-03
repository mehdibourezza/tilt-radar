# TiltRadar

**Real-time emotional state detection in competitive gaming.**

TiltRadar is an ML-powered system that detects when players in online competitive games (League of Legends) become emotionally frustrated during live matches. It extracts behavioral signals every 5 seconds, scores frustration probability through a multi-stage ML pipeline, and surfaces actionable insights through a real-time in-game overlay.

---

## How It Works

```
LOCAL MACHINE                                  BACKEND SERVER
                                              
League of Legends                             FastAPI + WebSocket
      |                                             |
  Live Client API (port 2999)                 Tilt Inference Engine
      |                                        |            |
  Local Agent ---- WebSocket (5s) ------->  12 Behavioral   26D Feature
      |                                     Signals         Extraction
  In-Game Overlay                              |            |
  (tilt alerts +                           Tilt Score    XGBoost +
   exploit tips)                           [0 - 1]      GRU Temporal
      |                                        |            |
  Post-Game Report  <---- JSON -----------  Evaluation + Labeled Data
                                                    |
                                              PostgreSQL + Redis/Celery
                                              (baselines, ingestion)
```

### Detection Pipeline

1. **Signal Extraction** -- 12 composite behavioral indicators scored every 5 seconds:
   - Repeat deaths to the same opponent (pride/fixation)
   - Death rate acceleration over time
   - CS (farm) performance drop vs personal baseline
   - Kill participation decline (disengagement)
   - Item sells mid-game (rage indicator)
   - Objective absence, vision neglect, level/gold deficits, and more

2. **ML Scoring** -- 3-stage pipeline:
   - **Feature Extractor**: converts raw game state + baselines into a 26-dimensional normalized vector
   - **XGBoost Snapshot Scorer**: classifies single-snapshot tilt likelihood
   - **GRU Temporal Model** (PyTorch): processes sequences of 6-24 snapshots with stateful hidden state, runs inference in <10ms on CPU

3. **Baseline Normalization** -- compares each player against their own history:
   - **PELT change-point detection** on 100-game histories to find the latest stable performance segment
   - Falls back to rank-tier peer group aggregates when personal data is unavailable

4. **Self-Labeling Loop** -- no manual annotation needed:
   - Post-game: compares peak in-game tilt predictions against final scoreboard outcomes
   - Generates `true_positive` / `false_positive` / `true_negative` / `false_negative` labels automatically
   - Each game produces labeled training data for the next model iteration

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI, WebSockets, Uvicorn |
| **ML Models** | PyTorch (GRU), scikit-learn (XGBoost) |
| **Feature Engineering** | NumPy, SciPy, ruptures (PELT change-point detection) |
| **Database** | PostgreSQL (async via asyncpg), SQLAlchemy 2.0, Alembic |
| **Task Queue** | Redis + Celery |
| **Local Agent** | Python asyncio, httpx, tkinter overlay |
| **External APIs** | Riot Games API (Match v5, Spectator, Live Client Data) |
| **Monitoring** | Prometheus, Weights & Biases |
| **Security** | slowapi rate limiting, Pydantic schema validation, OWASP headers |

---

## Project Structure

```
tilt-radar/
|-- api/                    # FastAPI backend
|   |-- main.py             # App setup, CORS, rate limiting, security headers
|   |-- routers/ws.py       # WebSocket endpoint (core game-phase logic)
|   |-- schemas/             # Pydantic input validation schemas
|-- agent/
|   |-- local_agent.py      # Client-side agent (snapshot extraction, WebSocket)
|   |-- overlay.py          # In-game overlay (HUD + notification cards)
|-- ml/
|   |-- inference/engine.py # 12-signal rule-based tilt scoring
|   |-- features/           # Feature extraction, change-point baselines, buffers
|   |-- models/             # GRU temporal model architecture
|   |-- training/           # Training pipeline (calibrate / XGBoost / GRU)
|   |-- evaluation/         # Post-game accuracy evaluation
|-- data/
|   |-- db/                 # SQLAlchemy models, async repository, session
|   |-- riot/client.py      # Riot API client with token-bucket rate limiter
|-- workers/                # Celery async ingestion tasks
|-- configs/config.py       # Pydantic settings with production validators
|-- docker/                 # Docker Compose (PostgreSQL, Redis, API, worker)
|-- tests/                  # pytest + pytest-asyncio
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL + Redis) or local installs
- A [Riot Games API key](https://developer.riotgames.com)

### Setup

```bash
# Clone and install
git clone https://github.com/mehdibourezza/tilt-radar.git
cd tilt-radar
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set RIOT_API_KEY and your region/platform

# Start infrastructure
docker compose -f docker/docker-compose.yml up postgres redis -d

# Create tables
python -c "import asyncio; from data.db.session import create_all_tables; asyncio.run(create_all_tables())"
```

### Run (3 terminals)

```bash
# Terminal 1 — Backend
uvicorn api.main:app --host 0.0.0.0 --port 8001

# Terminal 2 — Worker
celery -A workers.celery_app worker --loglevel=info

# Terminal 3 — Agent (start before or during a game)
python -m agent.local_agent --summoner "YourName" --tag "EUW" --server ws://localhost:8001
```

### Train Models

```bash
python -m ml.training.train --mode all
# Outputs: experiments/signal_calibration.json, snapshot_scorer.pkl, temporal_model.pt
```

---

## In-Game Overlay

When an enemy's tilt score crosses 50%, a notification card appears:

```
+--------------------------------------+
| Azir            RAGE          73%    |
| ||||||||||||..........               |
| -> Invade their jungle               |
+--------------------------------------+
```

- Auto-dismisses after 7 seconds
- Anti-spam: only fires if score jumped 20%+ or new signals detected
- Color-coded: enemies (orange/red), allies (cyan), self (purple)

---

## Known Limitations & Future Work

This project documents its own fundamental constraints transparently (see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)):

| Limitation | Root Cause | Path Forward |
|---|---|---|
| **Labels tied to final stats, not moment of tilt** | No ground truth for *when* frustration occurred | Player self-report via overlay, or physiological signals (heart rate) |
| **Gold estimation is approximate** | Live Client API doesn't expose current gold | Riot API expansion, or inference from item buy/sell events |
| **1v1 vs 1v2 death ambiguity** | No real-time positional data for all players | Post-game timeline retroactive labeling |
| **Selection bias in calibration** | Only games where tilt fired enter training set | Lower logging threshold, supplement with scraped data |
| **Peer baselines require seeding** | `PeerGroupBaseline` table must be populated per patch | Run `scripts/populate_peer_groups.py` after each patch |

These limitations are solvable with richer data sources, opening a path toward a general-purpose behavioral prediction platform applicable beyond gaming.

---

## Security

- IP-based rate limiting on all endpoints (slowapi)
- Strict Pydantic schema validation on all WebSocket inputs (type checks, length limits, bounded values)
- WebSocket message size limits (100 KB) and idle connection timeouts
- Per-IP connection rate limiting to prevent resource exhaustion
- CORS restricted to known origins, OWASP security headers
- No hardcoded secrets -- all credentials via environment variables with production validators
- API docs disabled in production mode

---

## License

See [LICENSE](LICENSE).
