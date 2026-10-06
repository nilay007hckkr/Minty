"""Response cache keyed on a normalized form of the (standalone) question.

This used to be a semantic cache (MiniLM cosine >= 0.92). That served wrong
answers: "minimum age to open an account" vs "maximum age ..." scored 0.970,
so a question the KB can't answer got the "18 or older" answer. Embedding
similarity can't see antonyms, negation, or one-word qualifiers like
domestic/international, and no threshold separates those from paraphrases.

Now a hit requires the same content words in the same order, ignoring case,
punctuation, and a small set of filler words. That trades some paraphrase hit
rate for never answering a different question. Follow-ups are condensed into
standalone questions before they get here, which keeps keys fairly canonical.
"""

import hashlib
import json
import logging
import re
import unicodedata

import redis

from app.redis_client import redis_client

logger = logging.getLogger(__name__)

# New prefix so entries from the old semantic cache are never read; they
# expire on their own within CACHE_TTL_SECONDS.
CACHE_PREFIX = "qcache:"
CACHE_TTL_SECONDS = 60 * 60

# Words that don't change what's being asked. Deliberately excludes negations
# ("not", "no") and anything directional ("to", "from"), and order is kept,
# so "checking to savings" != "savings to checking".
_FILLER = frozenset(
    "a an the is are what whats how do does can could i my me please there any".split()
)


def normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", query).lower()
    text = text.replace("'", "").replace("’", "")
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(t for t in tokens if t not in _FILLER)


def _cache_key(query: str) -> str | None:
    normalized = normalize_query(query)
    if not normalized:
        return None
    # sha256, not hash(): Python's str hash is randomized per process.
    return CACHE_PREFIX + hashlib.sha256(normalized.encode()).hexdigest()[:32]


def check_cache(query: str) -> dict | None:
    key = _cache_key(query)
    if key is None:
        return None
    try:
        raw = redis_client.get(key)
    except redis.RedisError as e:
        logger.warning(f"Cache read skipped, Redis unavailable: {e}")
        return None
    return json.loads(raw) if raw else None


def write_cache(query: str, response: dict) -> None:
    key = _cache_key(query)
    if key is None:
        return
    try:
        redis_client.set(key, json.dumps(response), ex=CACHE_TTL_SECONDS)
    except redis.RedisError as e:
        logger.warning(f"Cache write skipped, Redis unavailable: {e}")


def _delete_matching(pattern: str) -> int:
    deleted = 0
    for key in redis_client.scan_iter(match=pattern, count=500):
        deleted += redis_client.delete(key)
    return deleted


def clear_cache() -> int:
    """Drops every cached answer. Called when the indexed content changes,
    since cached answers would otherwise keep quoting the old content for up
    to CACHE_TTL_SECONDS."""
    try:
        return _delete_matching(f"{CACHE_PREFIX}*")
    except redis.RedisError as e:
        logger.warning(f"Cache clear skipped, Redis unavailable: {e}")
        return 0


def purge_legacy_cache() -> int:
    """Removes keys left by the old semantic cache. Its semcache:index SET had
    no TTL, so it would otherwise stay in Redis forever."""
    try:
        return _delete_matching("semcache:*")
    except redis.RedisError as e:
        logger.warning(f"Legacy cache purge skipped, Redis unavailable: {e}")
        return 0
