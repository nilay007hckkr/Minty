import time
from app.redis_client import redis_client

RATE_LIMIT_PER_MINUTE = 20


def check_rate_limit(client_id: str) -> bool:
    window = int(time.time() // 60)
    key = f"ratelimit:{client_id}:{window}"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, 60)
    return count <= RATE_LIMIT_PER_MINUTE
