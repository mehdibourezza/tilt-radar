"""
SQLAlchemy async session factory.

We use async SQLAlchemy with asyncpg (PostgreSQL async driver).
All DB operations in this project are async — no blocking calls.

Two usage patterns:

  1. FastAPI dependency (via api/dependencies/db.py):
       async for session in get_session():
           repo = PlayerRepository(session)
           ...

  2. Celery tasks (sync context — we wrap with run_async):
       async with get_session() as session:
           ...
"""

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from configs.config import get_settings

settings = get_settings()

# Engine: one per process, shared across all requests
# pool_size: max concurrent DB connections
# echo: set True to log all SQL (useful for debugging, disable in production)
engine = create_async_engine(
    settings.postgres_url,
    pool_size=10,
    max_overflow=20,
    echo=settings.app_env == "development",
)

# Session factory — creates AsyncSession instances
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # don't expire objects after commit (safer for async)
    autoflush=False,
)


@asynccontextmanager
async def get_session() -> AsyncSession:
    """
    Async context manager for DB sessions.

    Usage in Celery tasks:
        async with get_session() as session:
            repo = PlayerRepository(session)
            await repo.upsert_baseline(...)
            await session.commit()

    Automatically rolls back on exception, closes session on exit.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_all_tables():
    """
    Create all tables defined in models.py.
    Used for development/testing — in production use Alembic migrations.
    """
    from data.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
