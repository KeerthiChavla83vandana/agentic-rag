"""
Async SQLAlchemy database engine, session factory, and dependency provider.

Architecture Notes:
- We use SQLAlchemy's asyncio extension (`create_async_engine`) backed by `asyncpg`.
- asyncpg is a pure-Python async PostgreSQL driver that is fully non-blocking
  and compatible with Python's asyncio event loop used by FastAPI/uvicorn.
- Every FastAPI route and pipeline function receives an `AsyncSession` via
  the `get_db()` dependency injector.

Why async?
- Prevents DB I/O from blocking the uvicorn event loop thread.
- Allows multiple concurrent requests to share the same event loop efficiently.
- Enables asyncio.gather() patterns (e.g. parallel chunk grading).

Connection Pooling:
- pool_pre_ping=True sends a lightweight SELECT 1 before each connection checkout
  to detect and recycle stale connections (important after Docker restarts).
- pool_size=10, max_overflow=20 are reasonable defaults for a single-node deployment.
  Adjust based on expected concurrent users.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models.

    All models (Document, DocumentChunk) inherit from this class.
    It provides the shared metadata registry used by create_all().
    """
    pass


# ---------------------------------------------------------------------------
# Async Engine
# ---------------------------------------------------------------------------
# Uses asyncpg driver: postgresql+asyncpg://
# pool_pre_ping ensures stale connections don't cause 500 errors after DB restarts.
engine = create_async_engine(
    settings.async_database_url,
    pool_pre_ping=True,
    pool_size=10,        # max persistent connections in the pool
    max_overflow=20,     # max additional connections beyond pool_size under load
    echo=False,          # set True temporarily to log all SQL queries for debugging
)

# ---------------------------------------------------------------------------
# Async Session Factory
# ---------------------------------------------------------------------------
# async_sessionmaker creates AsyncSession instances on demand.
# expire_on_commit=False means ORM attributes remain accessible after commit
# without triggering an additional SELECT (important in async context where
# lazy loading would fail outside a session).
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# FastAPI Dependency: get_db()
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session as a FastAPI dependency.

    Usage in route handlers:
        @app.post("/some/route")
        async def my_route(db: AsyncSession = Depends(get_db)):
            result = await db.execute(...)

    The session is automatically closed (and rolled back if an exception
    was raised) after the request completes, even if the handler raises.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            # Roll back any in-progress transaction on error to keep DB consistent.
            await session.rollback()
            raise
        # Session is closed automatically via the context manager on exit.
