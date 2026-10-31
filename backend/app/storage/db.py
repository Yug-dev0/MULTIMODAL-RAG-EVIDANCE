import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

DATABASE_URL = os.environ.get("POSTGRES_URL", "postgresql+asyncpg://ragdb:ragdb@localhost:5432/ragdb")
# NullPool: a fresh low-level connection per checkout, so tests running each
# async test in its own event loop don't reuse a connection bound to a stale loop.
engine = create_async_engine(DATABASE_URL, echo=False, poolclass=NullPool)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

@asynccontextmanager
async def get_session():
    async with SessionLocal() as session:
        yield session
