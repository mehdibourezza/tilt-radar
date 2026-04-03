import secrets

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Riot API
    riot_api_key: str
    riot_default_region: str = "europe"
    riot_default_platform: str = "euw1"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "tiltRadar"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # App
    app_env: str = "development"
    secret_key: str = "change-me"
    log_level: str = "INFO"

    # W&B
    wandb_api_key: str = ""
    wandb_project: str = "tilt-radar"

    # Baseline computation
    baseline_min_games: int = 30        # minimum games to build a reliable baseline
    baseline_long_window: int = 100     # long-term window (games)
    baseline_medium_window: int = 20    # medium-term window
    baseline_short_window: int = 5      # short-term (current form)

    # Riot API rate limits (development key defaults)
    riot_rate_limit_per_second: int = 20
    riot_rate_limit_per_two_minutes: int = 100

    # -----------------------------------------------------------------------
    # Validators — reject insecure defaults in production
    # -----------------------------------------------------------------------

    @field_validator("riot_api_key")
    @classmethod
    def riot_key_must_be_set(cls, v: str) -> str:
        placeholder = {"", "your-key-here", "RGAPI-your-key-here"}
        if v.strip() in placeholder:
            raise ValueError(
                "RIOT_API_KEY must be set to a real Riot API key in .env"
            )
        return v.strip()

    @field_validator("secret_key")
    @classmethod
    def secret_key_not_default_in_prod(cls, v: str, info) -> str:
        env = info.data.get("app_env", "development")
        if env == "production" and v in ("change-me", "change-me-in-production", ""):
            raise ValueError(
                "SECRET_KEY must be changed from its default in production. "
                f"Suggestion: SECRET_KEY={secrets.token_urlsafe(32)}"
            )
        return v

    @field_validator("postgres_password")
    @classmethod
    def db_password_not_default_in_prod(cls, v: str, info) -> str:
        env = info.data.get("app_env", "development")
        if env == "production" and v == "postgres":
            raise ValueError(
                "POSTGRES_PASSWORD must be changed from default 'postgres' in production"
            )
        return v

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
