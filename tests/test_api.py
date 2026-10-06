from unittest.mock import patch

import pytest
import redis
from fastapi.testclient import TestClient

from app.cache import _cache_key, normalize_query


# --- cache keying: different questions must never share an entry ----------

@pytest.mark.parametrize(
    "a, b",
    [
        # 0.970 cosine under the old semantic cache -> served the wrong answer
        ("Is there a minimum age to open an account?",
         "Is there a maximum age to open an account?"),
        ("What is the fee for an outgoing domestic wire transfer?",
         "What is the fee for an outgoing international wire transfer?"),
        ("What is the daily ATM withdrawal limit?",
         "What is the daily ATM withdrawal limit for premium accounts?"),
        ("Can I transfer from checking to savings?",
         "Can I transfer from savings to checking?"),
        ("Does the card charge foreign transaction fees?",
         "Does the card not charge foreign transaction fees?"),
    ],
)
def test_cache_keys_differ_for_different_questions(a, b):
    assert _cache_key(a) != _cache_key(b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("What is the daily ATM withdrawal limit?",
         "what's the daily ATM withdrawal limit"),
        ("Can I request a credit limit increase?",
         "How do I request a credit limit increase"),
        ("Are there any fees for using an out-of-network ATM?",
         "fees for using an out of network ATM?"),
    ],
)
def test_cache_keys_match_for_rephrasings_of_same_question(a, b):
    assert _cache_key(a) == _cache_key(b)


def test_cache_skips_queries_that_are_all_filler():
    assert normalize_query("What is it?") == "it"
    assert _cache_key("what is the") is None


# --- Redis outage: optional features degrade, chat keeps working -----------

def _redis_down(*args, **kwargs):
    raise redis.ConnectionError("Connection refused")


@pytest.fixture
def redis_offline():
    methods = ["get", "set", "incr", "expire", "rpush", "lrange", "ping"]
    patches = [
        patch(f"app.redis_client.redis_client.{m}", side_effect=_redis_down)
        for m in methods
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


@pytest.fixture
def client():
    from app.main import app

    # No `with`: skips the lifespan, which would load models and index.
    return TestClient(app)


def test_chat_answers_when_redis_is_down(redis_offline, client):
    graph_result = {"answer": "The limit is $1,000.", "sources": ["sample_data/FAQ.md"]}
    with patch("app.main.app_graph.invoke", return_value=graph_result) as invoke, \
            patch("app.main.condense_query", side_effect=lambda h, q: q):
        resp = client.post("/chat", json={"query": "ATM limit?", "session_id": "s1"})

    assert resp.status_code == 200
    assert resp.json() == {**graph_result, "cached": False}
    invoke.assert_called_once()


def test_rate_limit_still_enforced_in_process_when_redis_is_down(redis_offline):
    from app.rate_limit import RATE_LIMIT_PER_MINUTE, check_rate_limit

    results = [check_rate_limit("10.0.0.99") for _ in range(RATE_LIMIT_PER_MINUTE + 1)]
    assert all(results[:-1])
    assert results[-1] is False


def test_history_is_empty_when_redis_is_down(redis_offline):
    from app.history import append_message, get_history

    append_message("s1", "user", "hi")  # must not raise
    assert get_history("s1") == []


def test_health_reports_degraded_when_redis_is_down(redis_offline, client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "degraded", "redis": "unavailable"}
