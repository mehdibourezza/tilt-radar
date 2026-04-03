# TiltRadar

**Real-time emotional state detection in competitive gaming.**

TiltRadar is an ML-powered system that detects when players in online competitive games (League of Legends) become emotionally frustrated during live matches. It extracts behavioral signals every 5 seconds, scores frustration probability through a multi-stage ML pipeline, and surfaces actionable insights through a real-time in-game overlay.

---

## How It Works

A **local agent** runs alongside the game and reads its state every 5 seconds through the game's local API. Each snapshot (player stats, kills, items, objectives) is streamed over a **WebSocket** to the backend server.

The backend runs each snapshot through four stages:

> **1. Signal Extraction** -- 12 behavioral indicators are scored in real time: repeat deaths to the same opponent (fixation), death rate acceleration, farm performance dropping below personal baseline, declining team participation, rage item sells, objective absence, and more.
>
> **2. Feature Engineering** -- Raw signals + player baselines are converted into a **26-dimensional normalized vector**. Baselines are computed using **PELT change-point detection** on the player's last 100 games, isolating their latest stable performance segment rather than averaging over stale history.
>
> **3. ML Scoring** -- The feature vector is scored by an **XGBoost snapshot classifier**, then fed into a **stateful GRU temporal model** (PyTorch) that tracks sequences of 6-24 snapshots. The GRU maintains hidden state across snapshots, so inference is O(1) per tick and runs in **<10ms on CPU**.
>
> **4. Self-Labeling** -- After the game ends, peak predictions are compared against final scoreboard outcomes to automatically generate labeled training data (true/false positives and negatives). No manual annotation is needed -- every game improves the next model.

Tilt scores and recommendations are sent back to the agent, which displays them through a **real-time in-game overlay**. In the background, **Celery workers** ingest match histories for all 10 players via the Riot API, building the baseline database for future games.

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
