import pytest
from unittest.mock import patch
from openai import RateLimitError, APIStatusError
import httpx
from app.concurrency import call_with_backoff

pytestmark = pytest.mark.asyncio


def _rate_limit_error():
    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com"))
    return RateLimitError("rate limited", response=resp, body=None)


def _server_error(status_code: int):
    resp = httpx.Response(status_code, request=httpx.Request("POST", "https://api.openai.com"))
    return APIStatusError("server error", response=resp, body=None)


async def test_retries_on_rate_limit_and_eventually_succeeds():
    # Regression test: a transient 429 previously killed the whole
    # ingestion job outright instead of retrying, even though OpenAI's own
    # error message says to retry within milliseconds.
    calls = []

    def flaky(x):
        calls.append(x)
        if len(calls) < 3:
            raise _rate_limit_error()
        return "ok"

    with patch("app.concurrency.asyncio.sleep") as mock_sleep:
        result = await call_with_backoff(flaky, "arg")

    assert result == "ok"
    assert len(calls) == 3
    assert mock_sleep.call_count == 2


async def test_retries_on_5xx_but_not_4xx_client_errors():
    calls = []

    def flaky(x):
        calls.append(x)
        if len(calls) < 2:
            raise _server_error(503)
        return "ok"

    with patch("app.concurrency.asyncio.sleep"):
        result = await call_with_backoff(flaky, "arg")
    assert result == "ok"
    assert len(calls) == 2


async def test_does_not_retry_non_retryable_client_errors():
    def always_bad(x):
        raise _server_error(400)

    with pytest.raises(APIStatusError):
        await call_with_backoff(always_bad, "arg")


async def test_raises_after_exhausting_all_retries():
    with patch("app.concurrency.asyncio.sleep"):
        with pytest.raises(RateLimitError):
            await call_with_backoff(lambda: (_ for _ in ()).throw(_rate_limit_error()))
