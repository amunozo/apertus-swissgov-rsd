# Final run (test split, once)

The test split is used once, in this run, from a frozen and tagged commit.
Nothing below may be run before the code is frozen. All commands run from
`track_2a/`.

## Same code as the dev run

`kaggle/01_extract_dev.py` (dev) and `kaggle/04_extract_test.py` (test) are
thin wrappers around the same function, `rsd.extract.main`:

| | dev run | test run |
|---|---|---|
| call | `main(split="dev", final=False, state_layers=[8, 16, 24, 32])` | `main(split="test", final=True, state_layers=[])` |
| model | `andreasmartin/apertus-v1.5-8b-text` at commit `c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01` | same (pinned in `rsd.extract.MODEL_SHA`) |
| frames, fp16 patches, pooling, `features.npz` format | `rsd.extract` + `rsd.apertus` | same |
| hidden-state dumps | yes (for the states head) | no (the scalar head needs only `features.npz`) |

The refactor into `rsd.extract` was checked on the CPU with a tiny random
Apertus (32 layers, hidden size 64, real tokenizer, fp16 patches) on the first
2 dev documents per pair: the original `01_extract_dev.py`, the refactored one
and `04_extract_test.py` in dry-run mode (split forced to dev) produce
byte-identical `features.npz` and `index.tsv`, and the original and the
refactored dev script write byte-identical state files (12 files: 4 layers x 3
pairs). `tests/test_extract.py` checks that both scripts
call `main` and differ only in these three arguments.

`meta.json` records `code_sha256` of `rsd/apertus.py`, `rsd/extract.py` and
`rsd/data.py` as bundled into the Kaggle notebook. (The dev run predates this
field.)

## Expected GPU time

The test split has 56 documents per pair (dev: 168). The test files are about
37% of the dev files' size, and the dev run took 40 min of passes. Expect
about 15 min of forward passes plus about 5 min for the pip install and model
load: **about 20 min on 2x T4**. Plan a 45-minute slot.

## Checklist before running

- [ ] All dev work is done: head configuration chosen on dev only
      (`rsd.head` `FINAL_SET`, `FINAL_PER_PAIR`, `FINAL_LAMBDA`).
- [ ] `uv run pytest` passes.
- [ ] Optional: CPU dry run of the test script (needs a Python with torch and
      `transformers==5.19.0`; it uses dev, never test):
      `RSD_DRY_RUN=1 RSD_OUT=cache/dry_run/test PYTHONPATH=src python kaggle/04_extract_test.py`
- [ ] Commit everything, working tree clean (`git status`).
- [ ] Tag the frozen commit and record the sha:

```bash
git tag -a final-run -m "Frozen code for the single test run"
git rev-parse HEAD          # write this sha into the technical report
git push origin final-run
```

## Commands, in order

### (a) Test extraction on Kaggle

```bash
uv run python kaggle/run_on_kaggle.py kaggle/04_extract_test.py \
    --kernel amunozo/rsd-extract-test \
    --files '^(features\.npz|index\.tsv|meta\.json)$'
```

This pushes, waits, and downloads into `kaggle/outputs/04_extract_test/`.

### (b) Download (only if the wait in (a) was interrupted) and check

```bash
uv run python kaggle/run_on_kaggle.py kaggle/04_extract_test.py \
    --kernel amunozo/rsd-extract-test --download-only \
    --files '^(features\.npz|index\.tsv|meta\.json)$'
```

Check `kaggle/outputs/04_extract_test/meta.json`:

- `"split": "test"`, `"final": true`, `"dry_run": false`;
- `"n_docs"` is 56 for de, fr and it;
- `"model_sha"` is `c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01`, the dev run's
  commit (the script loads that revision and asserts it);
- `"code_sha256"` matches `sha256sum src/rsd/apertus.py src/rsd/extract.py src/rsd/data.py`;
- `"torch"` and `"transformers"` (5.19.0) match the dev run's `meta.json`.

Export the compact test features. This is required: `make run` predicts
test from this file, so the final run and the judges' run use the same input.

```bash
uv run python -m rsd.features kaggle/outputs/04_extract_test/features.npz data/features_test_head.npz
```

### (c) Fit the head on all of dev, predict test

These are the same arguments `make run` uses (`src/rsd/pipeline.py`). The
default head config is shared (e), lambda 1e-3; pass `--per-pair` / `--lam`
only if Alberto chose otherwise. The prefix must not contain "llm".

```bash
uv run python -m rsd.head fit-predict \
    --train-features data/features_dev_head.npz --train-split dev \
    --predict-features data/features_test_head.npz \
    --predict-split test --final \
    --out data/predictions_test/apertus_head_
```

Afterwards, commit `data/features_test_head.npz` and `data/predictions_test/`,
then run `make run` from a clean clone: its TEST line must match (d).

### (d) Official evaluation on test

```bash
uv run python -m official_eval.run data/predictions_test/apertus_head_ --final
```

### (e) Bootstrap confidence intervals (document-level, 1000 resamples)

```bash
uv run python - <<'EOF'
from rsd.data import load_all
from rsd.metrics import bootstrap_ci, evaluate
from rsd.predictions import read_predictions

test = load_all("test", final=True)
prefix = "data/predictions_test/apertus_head_"
scores = {lang: read_predictions(prefix, lang) for lang in test}
print({k: round(100 * v, 1) for k, v in evaluate(test, scores).items()})
print({k: (round(100 * lo, 1), round(100 * hi, 1)) for k, (lo, hi) in bootstrap_ci(test, scores).items()})
EOF
```

The local Spearman in (e) should match the official numbers in (d).

## After the run

- Record in the report: the tag, the sha, the Kaggle kernel version,
  `meta.json` (model sha, versions, timing), and the scores with CIs, whatever
  they are.
- Do not change the code or the head and re-run on test. If a run fails
  before producing scores (for example out of memory on a long test pair),
  fix only the failure, re-tag, re-run, and say so in the report.
