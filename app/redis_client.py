import os
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Short timeouts: Redis backs optional features (cache, history, rate limit).
# Without them, an unreachable host (dropped packets rather than a refused
# connection) would stall every request instead of failing fast and degrading.
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)
