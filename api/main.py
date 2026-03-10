import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from api.routers import ws

logging.basicConfig(level="INFO")

app = FastAPI(
    title="TiltRadar API",
    description="Real-time enemy tilt detection for League of Legends",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

# Routers
app.include_router(ws.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
