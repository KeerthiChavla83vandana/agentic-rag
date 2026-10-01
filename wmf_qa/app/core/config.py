"""
Application settings loaded from environment variables via pydantic-settings.

All settings are read from the `.env` file (or actual environment variables).
Sensitive values like API keys and DB passwords are never hardcoded here.

Usage:
    from app.core.config import settings
    print(settings.database_url)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration sourced from environment variables.

    Environment variables are matched case-insensitively.
    See `.env.example` for the full list of required and optional variables.

    Attributes:
        database_host:        PostgreSQL hostname (e.g. 'wmf_qa_db' inside Docker).
        database_port:        PostgreSQL port (default 5432).
        database_name:        Database name (e.g. 'wmf_ip_qa').
        database_user:        Database username.
        database_password:    Database password.
        backend_name:         Human-readable service name for health check responses.
        backend_version:      Semantic version string (e.g. '0.1.0').
        gemini_api_key:       Google Gemini API key. Required for LLM operations
                              (router, grader, rewriter, synthesis). None disables LLM.
        nvidia_api_key:       Optional NVIDIA NIM API key (reserved for future use,
                              e.g. adding image/vision summarization via LLaMA Vision).
        embedding_model_name: HuggingFace model identifier for sentence-transformers.
                              Default: 'BAAI/bge-m3' (1024-dim, multilingual, English + Kannada).
    """

    # ---------------------------------------------------------------------------
    # Database connection parameters
    # ---------------------------------------------------------------------------
    database_host: str
    database_port: int
    database_name: str
    database_user: str
    database_password: str

    # ---------------------------------------------------------------------------
    # Service identity
    # ---------------------------------------------------------------------------
    backend_name: str
    backend_version: str

    # ---------------------------------------------------------------------------
    # LLM & Embedding settings
    # ---------------------------------------------------------------------------
    llm_provider: str = "nvidia"
    """LLM provider to use: 'nvidia' or 'gemini'."""

    gemini_api_key: str | None = None
    """Google Gemini API key."""

    nvidia_api_key: str | None = None
    """NVIDIA NIM API key."""

    nvidia_model_name: str = "meta/llama-3.2-11b-vision-instruct"
    """NVIDIA NIM Model ID (e.g. meta/llama-3.2-11b-vision-instruct, meta/llama-3.2-90b-vision-instruct)."""

    embedding_model_name: str = "BAAI/bge-m3"
    """Sentence-Transformers model name. BAAI/bge-m3 produces 1024-dim vectors."""

    # ---------------------------------------------------------------------------
    # Pydantic-settings config
    # ---------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars silently
    )

    # ---------------------------------------------------------------------------
    # Derived connection URL properties
    # ---------------------------------------------------------------------------

    @property
    def database_url(self) -> str:
        """Synchronous SQLAlchemy PostgreSQL connection URL using psycopg (v3).

        Used by: Alembic migrations, SQLAlchemy sync engine (legacy tooling).
        Format: postgresql+psycopg://<user>:<password>@<host>:<port>/<db>
        """
        return (
            f"postgresql+psycopg://"
            f"{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}"
            f"/{self.database_name}"
        )

    @property
    def async_database_url(self) -> str:
        """Async SQLAlchemy PostgreSQL connection URL using asyncpg.

        Used by: All FastAPI async route handlers and pipeline functions.
        Format: postgresql+asyncpg://<user>:<password>@<host>:<port>/<db>

        asyncpg is a pure-Python async PostgreSQL driver that integrates with
        Python's asyncio event loop and avoids blocking the thread pool.
        """
        return (
            f"postgresql+asyncpg://"
            f"{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}"
            f"/{self.database_name}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    Uses functools.lru_cache so the .env file is read only once at startup,
    not on every request. This is the recommended pydantic-settings pattern.
    """
    return Settings()


# ---------------------------------------------------------------------------
# Module-level singleton — import this directly across the codebase.
# ---------------------------------------------------------------------------
settings = get_settings()
