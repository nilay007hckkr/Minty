FROM python:3.12-slim
# Pinned so image builds don't change under us when uv releases.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
# --no-dev: pytest and other dev tools don't belong in the runtime image.
RUN uv sync --frozen --no-dev

# Bake both models into the image. Otherwise every fresh container downloads
# them from Hugging Face on startup (slow cold start, fails without network).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
# Models are local now; never reach out to the Hub at runtime.
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY sample_data ./sample_data
COPY static ./static

# Run as an unprivileged user. /app/data must exist and be owned by it so a
# fresh named volume mounted there inherits that ownership.
RUN useradd --create-home --uid 1000 minty \
    && mkdir -p /app/data \
    && chown -R minty:minty /app/data /app/.cache
USER minty

EXPOSE 8000
# The venv's uvicorn directly, not `uv run`: uv run re-syncs the project on
# start, which would reinstall the dev group and needs write access to .venv.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
