import asyncio
import logging
from openai import RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

# Process-wide cap on concurrent OpenAI calls. Ingestion/linking/retrieval
# previously issued these one at a time in `for` loops, so wall-clock time
# was dominated by serialized network round-trips rather than actual work.
# Running them concurrently collapses that latency, but must stay bounded —
# unbounded concurrency doesn't reduce total tokens consumed, it just burns
# through the account's tokens-per-minute ceiling faster and produces more
# 429s. Every low-level OpenAI-calling function acquires this same instance,
# so the cap holds even when multiple ingestion jobs run at once.
LLM_SEMAPHORE = asyncio.Semaphore(5)

MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1.0


async def call_with_backoff(fn, *args, **kwargs):
    """Runs a blocking OpenAI SDK call in a thread under LLM_SEMAPHORE,
    retrying with exponential backoff on rate limits (429) and transient
    server errors (5xx).

    Concurrency alone doesn't prevent hitting a tokens-per-minute ceiling --
    a long video can still fire enough LLM calls in a short window to trip
    one, and the API itself says these are worth retrying ("try again in
    103ms"). Without this, a single transient 429 failed the entire
    ingestion job outright, discarding everything already extracted.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            async with LLM_SEMAPHORE:
                return await asyncio.to_thread(fn, *args, **kwargs)
        except RateLimitError as e:
            last_exc = e
        except APIStatusError as e:
            if e.status_code < 500:
                raise
            last_exc = e

        if attempt < MAX_RETRIES - 1:
            delay = BASE_DELAY_SECONDS * (2 ** attempt)
            logger.warning(
                "OpenAI call failed (%s), retrying in %.1fs [attempt %d/%d]",
                type(last_exc).__name__, delay, attempt + 1, MAX_RETRIES,
            )
            await asyncio.sleep(delay)
    raise last_exc
