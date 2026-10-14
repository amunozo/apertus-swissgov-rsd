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

# Official evaluation code, dev and test texts and gold are fetched at build
# time (pinned commits), so `make run` needs no network. --test only downloads
# files; the test split is still loaded and scored only by the final run.
RUN python -m official_eval.fetch --test && chmod -R a+rwX /app/cache  # the run may be non-root and writes derived gold files

CMD ["python", "-m", "rsd.pipeline", "--out", "outputs"]
