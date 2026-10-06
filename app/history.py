import json
import logging

import redis

from app.redis_client import redis_client

logger = logging.getLogger(__name__)

HISTORY_TTL_SECONDS = 60 * 60 * 24


# History is an optional feature: on a Redis failure, the chat continues as
# single-turn rather than failing the request.
def append_message(session_id: str, role: str, content: str) -> None:
    key = f"history:{session_id}"
    message = json.dumps({"role": role, "content": content})
    try:
        redis_client.rpush(key, message)
        redis_client.expire(key, HISTORY_TTL_SECONDS)
    except redis.RedisError as e:
        logger.warning(f"History write skipped, Redis unavailable: {e}")


def get_history(session_id: str, limit: int = 6) -> list[dict]:
    """Most recent `limit` messages, oldest first. Empty if Redis is down."""
    key = f"history:{session_id}"
    try:
        raw_messages = redis_client.lrange(key, -limit, -1)
    except redis.RedisError as e:
        logger.warning(f"History read skipped, Redis unavailable: {e}")
        return []
    return [json.loads(msg) for msg in raw_messages]
