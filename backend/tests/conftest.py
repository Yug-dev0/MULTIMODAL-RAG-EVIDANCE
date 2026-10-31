import os

# Must run before `app.api.auth` is imported by anything (including the
# lines below): MOCK_AUTH is computed once at module import time from this
# env var, not re-checked per request. A belt-and-suspenders default for
# anything that touches auth.py directly rather than through a TestClient
# with the dependency override below (e.g. calling get_current_user in
# isolation) — not setdefault, since a developer running tests with the
# real .env sourced into their shell (FIREBASE_AUTH_MOCK=false there) would
# otherwise have that value already present in os.environ before Python
# even starts, silently skipping the override.
os.environ["FIREBASE_AUTH_MOCK"] = "true"

import re
from urllib.parse import urlparse

import pytest
from sqlalchemy import text
from app.storage.db import get_session
from app.storage.graph import get_driver
from app.main import app
from app.api.auth import User, get_current_user

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Must match app.storage.db's fallback so the guard reports the URL actually
# in use when POSTGRES_URL is unset.
DEFAULT_POSTGRES_URL = "postgresql+asyncpg://ragdb:ragdb@localhost:5432/ragdb"
_TEST_DB_PATTERN = re.compile(r"(^|[_-])test([_-]|$)")


@pytest.fixture
def auth_headers() -> dict:
    """Bearer header accepted by get_current_user in mock auth mode."""
    return {"Authorization": "Bearer mock-token-test@example.com"}


async def _fake_current_user() -> User:
    return User(uid="test-uid", email="test@example.com")


@pytest.fixture(autouse=True)
def _override_auth():
    # Routes require a verified Firebase user; tests exercise route logic,
    # not the auth layer (that's covered by tests/api/test_auth.py), so stub
    # the dependency for every TestClient request rather than requiring every
    # test to construct and pass a real/mock bearer token.
    app.dependency_overrides[get_current_user] = _fake_current_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
async def _clean_postgres(request):
    if "integration" in request.keywords:
        # TRUNCATE destroys every row in the corpus. The Neo4j fixture below
        # already refuses to run against a non-local host; this one did not,
        # so running the suite with POSTGRES_URL still pointed at a working
        # dev database silently wiped real ingested evidence. Require a
        # database that is both local AND named as a test database -- a dev
        # database on localhost is exactly the case that got destroyed.
        url = os.environ.get("POSTGRES_URL", DEFAULT_POSTGRES_URL)
        parsed = urlparse(url)
        host = parsed.hostname or ""
        db_name = (parsed.path or "").lstrip("/")
        if host not in _LOCAL_HOSTS:
            raise RuntimeError(
                f"Refusing destructive Postgres test cleanup against non-local host {host!r}. "
                "Point POSTGRES_URL at a local test database to run integration tests."
            )
        if not _TEST_DB_PATTERN.search(db_name):
            raise RuntimeError(
                f"Refusing to TRUNCATE database {db_name!r}: its name doesn't mark it as a test "
                "database, and this fixture deletes all evidence, entities and events.\n"
                f"Create one and point POSTGRES_URL at it, e.g.:\n"
                f"  createdb {db_name}_test && psql -d {db_name}_test -c 'CREATE EXTENSION IF NOT EXISTS vector'\n"
                f"  POSTGRES_URL=postgresql+asyncpg://.../{db_name}_test alembic upgrade head\n"
                f"  POSTGRES_URL=postgresql+asyncpg://.../{db_name}_test pytest"
            )
        async with get_session() as session:
            # chat_sessions/chat_messages/session_sources are included because
            # they were not before: chat history survived between runs, so a
            # test asserting on "the questions this user asked" saw questions
            # left behind by earlier runs and passed or failed on stale rows.
            await session.execute(text(
                "TRUNCATE evidence_units, entities, knowledge_events, relationships, baseline_chunks, "
                "chat_sessions, chat_messages, session_sources CASCADE"
            ))
            await session.commit()
    yield


@pytest.fixture(autouse=True)
async def _clean_neo4j(request):
    # Tests that touch the graph (via app.storage.graph) leave nodes/edges
    # behind otherwise — Neo4j had no equivalent of the Postgres TRUNCATE
    # above, so every run of the suite against a shared dev instance
    # accumulated stale test nodes (e.g. hardcoded fixture ids) into the
    # real graph data that ingestion/demo queries read from.
    if "integration" in request.keywords:
        # DETACH DELETE wipes the entire graph — refuse to run it against
        # anything that isn't obviously a local/test instance, so pointing
        # NEO4J_URI at a shared or production instance can't silently
        # destroy real data by running the test suite.
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        host = urlparse(uri).hostname or ""
        if host not in _LOCAL_HOSTS:
            raise RuntimeError(
                f"Refusing destructive Neo4j test cleanup against non-local host {host!r}. "
                "Point NEO4J_URI at a local instance to run integration tests."
            )
        driver = get_driver()
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        await driver.close()
    yield
