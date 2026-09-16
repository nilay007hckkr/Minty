import json
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from app.redis_client import redis_client

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CACHE_PREFIX = "semcache:"
CACHE_INDEX_KEY = "semcache:index"
SIMILARITY_THRESHOLD = 0.92
CACHE_TTL_SECONDS = 60 * 60


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def check_cache(query: str) -> dict | None:
    query_embedding = _embeddings.embed_query(query)
    keys = redis_client.smembers(CACHE_INDEX_KEY)

    best_score, best_response = 0.0, None
    for key in keys:
        raw = redis_client.get(key)
        if not raw:
            redis_client.srem(CACHE_INDEX_KEY, key)
            continue
        entry = json.loads(raw)
        score = _cosine_similarity(query_embedding, entry["embedding"])
        if score > best_score:
            best_score, best_response = score, entry["response"]

    if best_response and best_score >= SIMILARITY_THRESHOLD:
        return best_response
    return None


def write_cache(query: str, response: dict) -> None:
    query_embedding = _embeddings.embed_query(query)
    key = f"{CACHE_PREFIX}{abs(hash(query))}"
    payload = json.dumps({"embedding": query_embedding, "response": response})
    redis_client.set(key, payload, ex=CACHE_TTL_SECONDS)
    redis_client.sadd(CACHE_INDEX_KEY, key)
