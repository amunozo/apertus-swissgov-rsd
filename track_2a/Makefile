.PHONY: run eval-reference test

IMAGE ?= swissgov-rsd

# Builds the CPU Docker image (fetches the official evaluation code at build
# time) and runs rsd.pipeline in it. Results appear in ./outputs/.
run:
	docker build -t $(IMAGE) .
	mkdir -p outputs
	docker run --rm --user $$(id -u):$$(id -g) -v "$(CURDIR)/outputs:/app/outputs" $(IMAGE)

test:
	uv run pytest

# Data and evaluation layer, dev split only. Needs uv and network access on
# the first run. Checks that the official script and src/rsd/metrics.py
# reproduce the paper's released DiffAlign Qwen3-Embedding-8B dev scores, then
# writes docs/reference_dev_scores.md.
eval-reference:
	uv run pytest tests/test_eval.py
	uv run python -m official_eval.reference
