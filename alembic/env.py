"""
Alembic environment — configured for async SQLAlchemy.

Why async Alembic?
  Our engine uses asyncpg (async PostgreSQL driver). Alembic by default is
  synchronous, but it supports async via run_async_migrations(). We use that here.

How to use:
  Create a new migration after changing models.py:
    alembic revision --autogenerate -m "add tilt_events table"

  Apply pending migrations:
    alembic upgrade head

  Roll back one step:
    alembic downgrade -1
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# Import our models so Alembic can detect schema changes automatically
from data.db.models import Base
from configs.config import get_settings

config = context.config
settings = get_settings()

# Override the sqlalchemy.url from alembic.ini with our settings
config.set_main_option("sqlalchemy.url", settings.postgres_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations with an async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
