# Test results (single final run)

Spearman x 100 on the SwissGov-RSD **test** split (56 documents per pair),
global per pair. **mean** is the macro average over EN-DE, EN-FR and EN-IT of
the unrounded local scores. The test split was used once, for this run. Nothing
was changed after the scores were seen: not the code, the head, the features
or the predictions.

## What was run

Commands are those of `docs/final_run.md`, from `track_2a/`:

```
uv run python -m official_eval.run data/predictions_test/apertus_head_ --final   # (d) official script
# (e) bootstrap snippet from docs/final_run.md (rsd.metrics.bootstrap_ci)
```

The predictions in `data/predictions_test/apertus_head_*` were frozen and
committed before they were scored. The head is scalar head (e), shared across
the three pairs, 96 features, lambda 1e-3, fitted on all 504 dev documents
(`apertus_head_config.json`).

## Official scores

Official script (`scripts/evaluate_predictions_admin.py`, test gold). It prints
3 decimals of rho.

| | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| Spearman (official, rho) | 0.428 | 0.234 | 0.460 | |
| Spearman x 100 (official) | 42.8 | 23.4 | 46.0 | 37.4 |
| Spearman x 100 (local, `rsd.metrics`) | 42.8 | 23.4 | 46.0 | 37.4 |
| Kendall tau-b (official, for reference) | 0.349 | 0.190 | 0.374 | |

The local metric matches the official script on every pair at the printed
precision (unrounded local: 0.4282, 0.2343, 0.4600). The official script read
56 documents per pair, none were dropped, and it printed no warnings.

## Bootstrap confidence intervals

Document-level bootstrap, 1000 resamples, 95% percentile interval
(`rsd.metrics.bootstrap_ci`), Spearman x 100.

| | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| Score | 42.8 | 23.4 | 46.0 | 37.4 |
| 95% CI | [33.3, 51.7] | [18.2, 28.2] | [37.1, 55.2] | [31.9, 42.4] |

The CIs resample documents and keep the fitted head fixed. They do not include
the uncertainty from fitting the head on dev, so they are too narrow (see
`docs/scalar_head_dev_results.md`).

## Released baselines on test

The released DiffAlign predictions of the official repo (commit
`1807a42100e742ed03d337c54c4b9ea86995f565`, directory
`data/evaluation/encoder_predictions/full/`) were scored on the test rows with
the same official script (`final=True`) and the same bootstrap, by a one-off
script. Nothing was added to the repo. Every system had all 56 test documents
per pair, ordered as the gold. These are other people's systems, so these
numbers cannot feed back into our choices. Official and local scores matched
for all seven systems. Sorted by mean.

| System | EN-DE | EN-FR | EN-IT | mean [95% CI] |
|---|---|---|---|---|
| **Apertus scalar head (e), ours** | 42.8 | 23.4 | 46.0 | **37.4** [31.9, 42.4] |
| DiffAlign Qwen3-Embedding-8B | 19.7 | 11.2 | 20.4 | 17.1 [13.7, 20.5] |
| DiffAlign EuroBERT 610m | 18.3 | 10.3 | 20.5 | 16.4 [13.2, 19.3] |
| DiffAlign Qwen3-Embedding-4B | 18.6 | 9.8 | 20.6 | 16.3 [12.8, 20.1] |
| DiffAlign bge-m3 | 17.2 | 9.8 | 21.0 | 16.0 [9.2, 21.4] |
| DiffAlign EuroBERT 2.1B | 13.8 | 9.1 | 18.8 | 13.9 [10.4, 17.1] |
| DiffAlign embeddinggemma-300m | 15.1 | 10.0 | 16.4 | 13.9 [10.2, 17.3] |
| DiffAlign mmBERT-base | 13.2 | 8.0 | 16.5 | 12.6 [9.8, 15.7] |

The baselines are training-free, so they are not directly comparable to a
head trained on 504 labelled dev documents. The dev table in
`docs/ablation_dev.md` shows the same order of baseline scores (16.4 to 17.5
for the three best on dev).

## Context: dev numbers

From `docs/scalar_head_dev_results.md` and `docs/ablation_dev.md`, for the same
configuration (scalar head (e), shared).

| | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| Nested 5-fold CV on all dev (168 docs per pair) | 43.2 | 29.1 | 47.5 | 39.9 |
| Fit on dev/train, scored on dev/val (34 docs per pair) | 38.6 | 31.8 | 43.7 | 38.0 |
| **Test (fit on all dev, 56 docs per pair)** | 42.8 | 23.4 | 46.0 | 37.4 |

The per-pair (e) head had a nested-CV mean of 40.0 (42.9 / 29.5 / 47.5). That
is not the configuration that was run. The shared head's nested-CV mean is 39.9.

## How test compares to dev

The test mean (37.4) is 0.6 below dev/val (38.0) and 2.5 below the nested CV
(39.9). The test CI [31.9, 42.4] contains both. EN-DE and EN-IT are close to
their CV values (42.8 vs 43.2, 46.0 vs 47.5). EN-FR is lower on test (23.4)
than in CV (29.1) and on dev/val (31.8), and its test CI [18.2, 28.2] does not
reach the CV value. With 56 documents per pair, one pair's score moves by
several points depending on which documents are drawn, so I do not read this as
more than noise plus a possible small dev optimism: the configuration was
chosen on dev, and the CV did not nest that choice. The released baselines
also score a bit lower on test than on dev for some pairs and higher for others,
so test is not uniformly harder. The result is consistent with the dev
estimate of roughly 38-40, at the lower end.

## Run metadata

| Item | Value |
|---|---|
| Tag | `final-run` (annotated, "Frozen code for the single test run") |
| Tag commit (`git rev-parse final-run^{commit}`) | `4c804d40e8f528613e2237204f37a277141a3342` |
| HEAD when scored | `57ab49b8c563e1dbd74a852a66af413b5d85f079` |
| Model | `andreasmartin/apertus-v1.5-8b-text` |
| Model sha | `c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01` (same as the dev run) |
| `meta.json` flags | `split: test`, `final: true`, `dry_run: false`, `n_docs`: 56 / 56 / 56 (de / fr / it) |
| Versions | torch 2.11.0+cu128, transformers 5.19.0 |
| `code_sha256` | `apertus.py` 6a5128cc..., `extract.py` af1500d7..., `data.py` eec035f2... |
| Words | 150434 |
| Timing | alone: 336 passes, 205733 tokens, 268.5 s; cond: 336 passes, 411466 tokens, 525.5 s; wall 899.5 s |
| Kaggle kernel | `amunozo/rsd-extract-test` (kernel version number not recorded in the output files) |
| Head config | feature set e, shared (`per_pair: false`), lambda 1e-3, clip 5.0, soft-label BCE with L2, L-BFGS, 96 features, trained on 504 dev documents |

Full `meta.json`: `kaggle/outputs/04_extract_test/meta.json`. Head config:
`data/predictions_test/apertus_head_config.json`.
