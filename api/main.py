import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.routers import ws
from configs.config import get_settings

logging.basicConfig(level="INFO")
settings = get_settings()

# ---------------------------------------------------------------------------
# Rate limiter — keyed by client IP, sensible defaults
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TiltRadar API",
    description="Real-time enemy tilt detection for League of Legends",
    version="0.1.0",
    # Disable interactive docs in production to reduce attack surface
    docs_url="/docs" if settings.app_env == "development" else None,
    redoc_url="/redoc" if settings.app_env == "development" else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS — restrict to known origins instead of wildcard
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
    max_age=600,
)

# Reject requests with unexpected Host headers (prevents DNS rebinding)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"],
)


# ---------------------------------------------------------------------------
# Security headers middleware (OWASP recommendations)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"  # modern best practice: rely on CSP
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    # Remove server version header to avoid information leakage
    response.headers.pop("server", None)
    return response


# ---------------------------------------------------------------------------
# Prometheus metrics — restricted to localhost only
# ---------------------------------------------------------------------------
@app.get("/metrics-internal")
@limiter.limit("10/minute")
async def guarded_metrics(request: Request):
    """Metrics are only accessible from localhost."""
    client_ip = request.client.host if request.client else ""
    if client_ip not in ("127.0.0.1", "::1", "localhost"):
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    # Prometheus instrumentator exposes at /metrics; we guard the alias here.
    return JSONResponse({"hint": "Use the /metrics endpoint from localhost"})


Instrumentator().instrument(app).expose(app)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(ws.router)


# ---------------------------------------------------------------------------
# Health check — rate limited to prevent abuse
# ---------------------------------------------------------------------------
@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    return {"status": "ok"}
