from dotenv import load_dotenv

load_dotenv()

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.graph import app_graph
from app.history import append_message
from app.cache import check_cache, write_cache
from app.rate_limit import check_rate_limit
from fastapi.staticfiles import StaticFiles

print(f"### main.py LOADED FROM: {__file__}", flush=True)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Minty")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    query: str
    session_id: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    if not check_rate_limit(request.session_id):
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
        result = app_graph.invoke(initial_state, config={"recursion_limit": 15})
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
