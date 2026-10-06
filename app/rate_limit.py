import logging
import threading
import time

import redis

from app.redis_client import redis_client

logger = logging.getLogger(__name__)

RATE_LIMIT_PER_MINUTE = 20

# Per-process fallback used only while Redis is unreachable. Failing closed
# would turn a Redis outage into a full chatbot outage; failing fully open
# would remove the LLM-cost protection. Per-process counting is looser (each
# worker counts separately) but still bounds abuse.
_local_counts: dict[tuple[str, int], int] = {}
_local_lock = threading.Lock()


def _check_local(client_id: str, window: int) -> bool:
    with _local_lock:
        for key in [k for k in _local_counts if k[1] < window]:
            del _local_counts[key]
        count = _local_counts.get((client_id, window), 0) + 1
        _local_counts[(client_id, window)] = count
    return count <= RATE_LIMIT_PER_MINUTE


def check_rate_limit(client_id: str) -> bool:
    window = int(time.time() // 60)
    key = f"ratelimit:{client_id}:{window}"
    try:
        count = redis_client.incr(key)
        if count == 1:
            redis_client.expire(key, 60)
    except redis.RedisError as e:
        logger.warning(f"Redis unavailable, using in-process rate limit: {e}")
        return _check_local(client_id, window)
    return count <= RATE_LIMIT_PER_MINUTE
