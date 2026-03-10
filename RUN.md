# How to Run TiltRadar

## One-time setup

### 1. Get a Riot API key
Go to https://developer.riotgames.com → sign in → copy your Development API key
Add it to `.env`:
```
RIOT_API_KEY=RGAPI-xxxx-your-key-here
RIOT_DEFAULT_PLATFORM=euw1     # change to na1, kr, etc. for your region
```

### 2. Set LoL to Borderless Windowed mode
In-game: Settings → Video → Window Mode → Borderless

---

## Every session (3 terminals)

### Terminal 1 — Start the database and backend services
```bash
cd C:\Users\boure\tilt-radar
docker compose -f docker/docker-compose.yml up postgres redis -d
conda run -n tilt-radar python -c "import asyncio; from data.db.session import create_all_tables; asyncio.run(create_all_tables())"
conda run -n tilt-radar uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### Terminal 2 — Start the Celery worker (data ingestion)
```bash
cd C:\Users\boure\tilt-radar
conda run -n tilt-radar celery -A workers.celery_app worker --loglevel=info
```

### Terminal 3 — Start the local agent (run BEFORE or AFTER entering champion select)
```bash
cd C:\Users\boure\tilt-radar
conda run -n tilt-radar python -m agent.local_agent --summoner "MarreDesNoobsNul" --tag "007" --server ws://localhost:8001
```

The agent waits for a game to start automatically.
Once detected, the overlay appears top-right — semi-transparent, click-through.

---

## What you'll see

Nothing until an enemy reaches tilt score > 50%.
When they do, a card appears top-right (like the kill feed, amber/orange):

```
┌──────────────────────────────────────┐
│ Azir            RAGE          73%   │
│ ████████████░░░░░░░░               │
│ → Invade their jungle               │
└──────────────────────────────────────┘
```

Card auto-dismisses after 7 seconds.
New card only appears if score jumps +20% or a new signal is detected.

---

## Docker not installed?

Install PostgreSQL and Redis manually:
- PostgreSQL 16: https://www.postgresql.org/download/windows/
- Redis for Windows: https://github.com/tporadowski/redis/releases

Then create the database:
```sql
CREATE DATABASE "tiltRadar";
```
