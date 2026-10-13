.PHONY: run eval-reference

# Replace with the command that starts the project in a Docker container,
# e.g. `docker compose up`. It must work from a clean checkout.
run:
	@echo "make run is not implemented yet — edit this Makefile"; exit 1

# Data and evaluation layer, dev split only. Needs uv and network access on
# the first run. Checks that the official script and src/rsd/metrics.py
# reproduce the paper's released DiffAlign Qwen3-Embedding-8B dev scores, then
# writes docs/reference_dev_scores.md.
eval-reference:
	uv run pytest tests/test_eval.py
	uv run python -m official_eval.reference
