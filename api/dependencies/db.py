"""
FastAPI dependency for database sessions.

FastAPI's dependency injection system calls get_db_session() for every
request that declares it. It handles session lifecycle automatically:
  - Opens a session before the request handler runs
  - Commits on success
  - Rolls back on exception
  - Always closes the session after

Usage in a router:
    from api.dependencies.db import get_db_session
    from sqlalchemy.ext.asyncio import AsyncSession

    @router.get("/player/{puuid}")
    async def get_player(puuid: str, session: AsyncSession = Depends(get_db_session)):
        repo = PlayerRepository(session)
        return await repo.get_player_by_puuid(puuid)
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from data.db.session import AsyncSessionLocal


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
