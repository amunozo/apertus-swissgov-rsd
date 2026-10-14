# CPU only. The head needs numpy, scipy, scikit-learn, nlpstats (no torch).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"

# Dependencies first (cached layer), then the project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

# Official evaluation code, dev gold and dev texts are fetched at build time,
# so the dev run needs no network. (The test branch of `make run` downloads the
# test texts and gold when it runs, see rsd/pipeline.py.)
RUN python -m official_eval.fetch && chmod -R a+rwX /app/cache  # the run may be non-root and writes derived gold files

CMD ["python", "-m", "rsd.pipeline", "--out", "outputs"]
