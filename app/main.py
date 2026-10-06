from dotenv import load_dotenv

load_dotenv()

import logging
import os
from contextlib import asynccontextmanager
import redis
from fastapi import FastAPI, HTTPException, Request
from typing import Annotated
from pydantic import BaseModel, StringConstraints
from app.graph import app_graph, RECURSION_LIMIT
from app.nodes import condense_query
from app.vectorstore import get_vectorstore, index_documents
from app.history import append_message, get_history
from app.cache import check_cache, write_cache
from app.rate_limit import check_rate_limit
from app.redis_client import redis_client
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A fresh `docker compose up` starts with an empty ./data volume; without
    # this, every query would silently fall back. Also picks up edits to
    # sample_data. No-op if the index is already in sync.
    count = index_documents(get_vectorstore())
    logger.info(f"Vector store ready with {count} chunks")
    yield


app = FastAPI(title="Minty", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


MAX_QUERY_CHARS = 500


class ChatRequest(BaseModel):
    # Bounded because each query fans out into up to ~10 LLM calls; an empty
    # or 100 KB query would otherwise go straight to Groq. Violations -> 422.
    query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS),
    ]
    # Becomes part of a Redis key; the UI sends crypto.randomUUID().
    session_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


@app.get("/health")
def health_check():
    # Redis only backs optional features, so its loss is "degraded", not down.
    try:
        redis_client.ping()
        redis_status = "ok"
    except redis.RedisError:
        redis_status = "unavailable"
    return {
        "status": "ok" if redis_status == "ok" else "degraded",
        "redis": redis_status,
    }


@app.post("/chat")
def chat(request: ChatRequest, http_request: Request):
    # Keyed on client IP, not session_id: session_id is client-supplied, so
    # rotating it would reset the limit. Behind a reverse proxy, set
    # FORWARDED_ALLOW_IPS to the proxy's address so uvicorn trusts its
    # X-Forwarded-For header; otherwise every client shares the proxy's IP.
    client_ip = http_request.client.host if http_request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Please slow down."
        )

    # Follow-ups like "what about premium accounts?" are rewritten into a
    # standalone question before caching and retrieval see them.
    query = condense_query(get_history(request.session_id), request.query)
    if query != request.query:
        logger.info(f"Condensed follow-up: '{request.query}' -> '{query}'")

    cached = check_cache(query)
    if cached:
        append_message(request.session_id, "user", request.query)
        append_message(request.session_id, "assistant", cached["answer"])
        return {**cached, "cached": True}

    initial_state = {
        "original_query": query,
        "query": query,
        "refinement_count": 0,
    }
    try:
        result = app_graph.invoke(
            initial_state,
            config={
                "recursion_limit": RECURSION_LIMIT,
                "run_name": f"chat-{request.session_id}",
                "metadata": {
                    "session_id": request.session_id,
                    "query": request.query,
                    "standalone_query": query,
                },
                "tags": ["chat-endpoint"],
            },
        )
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="The chat pipeline encountered an unexpected error.",
        )

    response = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }

    if result.get("sources"):
        write_cache(query, response)

    append_message(request.session_id, "user", request.query)
    append_message(request.session_id, "assistant", response["answer"])

    return {**response, "cached": False}
