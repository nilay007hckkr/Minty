import json
from app.redis_client import redis_client

HISTORY_TTL_SECONDS = 60 * 60 * 24


def append_message(session_id: str, role: str, content: str) -> None:
    key = f"history:{session_id}"
    message = json.dumps({"role": role, "content": content})
    redis_client.rpush(key, message)
    redis_client.expire(key, HISTORY_TTL_SECONDS)


def get_history(session_id: str) -> list[dict]:
    key = f"history:{session_id}"
    raw_messages = redis_client.lrange(key, 0, -1)
    return [json.loads(msg) for msg in raw_messages]
