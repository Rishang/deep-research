"""
Cache models for Docling client.
"""

import datetime
from typing import Optional

from pydantic import BaseModel
from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, create_engine, select


class CacheConfig(BaseModel):
    """Configuration for the cache system."""

    enabled: bool = True
    ttl_seconds: int = 86400  # Default TTL: 1 day
    db_url: str = "sqlite:///docling_cache.sqlite3"  # Default to SQLite
    create_tables: bool = True


class CacheEntry(SQLModel, table=True):
    """Generic cache entry model that can store any pickled data."""

    __tablename__ = "cache_entries"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    key: str = SQLField(index=True)  # Hash-based cache key
    function_name: str  # Name of the cached function
    data: bytes  # Pickled data
    created_at: datetime.datetime = SQLField(default_factory=datetime.datetime.now)
    expires_at: datetime.datetime


# Global engine reference
_engine = None


def get_engine(db_url: str):
    """Get or create a database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(db_url)
    return _engine


def create_db_and_tables(db_url: str):
    """Create database tables if they don't exist."""
    engine = get_engine(db_url)
    SQLModel.metadata.create_all(engine)


def is_cache_expired(expires_at: datetime.datetime) -> bool:
    """Check if a cache entry has expired."""
    return datetime.datetime.now() > expires_at


def clear_cache(db_url: str, function_name: Optional[str] = None):
    """
    Clear cache entries, optionally filtering by function name.

    Args:
        db_url: Database URL
        function_name: If provided, only clear cache for this function
    """
    engine = get_engine(db_url)
    with Session(engine) as session:
        if function_name:
            statement = select(CacheEntry).where(
                CacheEntry.function_name == function_name
            )
            for entry in session.exec(statement):
                session.delete(entry)
        else:
            # Clear all cache entries
            statement = select(CacheEntry)
            for entry in session.exec(statement):
                session.delete(entry)
        session.commit()


def clear_expired_cache(db_url: str):
    """Clear expired cache entries."""
    engine = get_engine(db_url)
    now = datetime.datetime.now()
    with Session(engine) as session:
        statement = select(CacheEntry).where(CacheEntry.expires_at < now)
        for entry in session.exec(statement):
            session.delete(entry)
        session.commit()
