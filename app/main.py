from dotenv import load_dotenv

load_dotenv()

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from app.graph import app_graph
from app.nodes import vector_store
from app.vectorstore import index_documents
from app.history import append_message
from app.cache import check_cache, write_cache
from app.rate_limit import check_rate_limit
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A fresh `docker compose up` starts with an empty ./data volume; without
    # this, every query would silently fall back. No-op if already indexed.
    count = index_documents(vector_store)
    logger.info(f"Vector store ready with {count} chunks")
    yield


app = FastAPI(title="Minty", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    query: str
    session_id: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest, http_request: Request):
    # Keyed on client IP, not session_id: session_id is client-supplied, so
    # rotating it would reset the limit. Behind a reverse proxy this needs
    # uvicorn's --proxy-headers so client.host is the real client.
    client_ip = http_request.client.host if http_request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Please slow down."
        )

    cached = check_cache(request.query)
    if cached:
        append_message(request.session_id, "user", request.query)
        append_message(request.session_id, "assistant", cached["answer"])
        return {**cached, "cached": True}

    initial_state = {
        "original_query": request.query,
        "query": request.query,
        "refinement_count": 0,
    }
    try:
        result = app_graph.invoke(
            initial_state,
            config={
                "recursion_limit": 15,
                "run_name": f"chat-{request.session_id}",
                "metadata": {"session_id": request.session_id, "query": request.query},
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
        write_cache(request.query, response)

    append_message(request.session_id, "user", request.query)
    append_message(request.session_id, "assistant", response["answer"])

    return {**response, "cached": False}
