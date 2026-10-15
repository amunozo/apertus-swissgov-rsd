# UZH Challenge: Token-Level Cross-Lingual Semantic Differences (SwissGov-RSD)

Submission for Track 2A of Hack Apertus (Online Stage, October 2026) addressing the **University of Zurich (UZH)** challenge: *Detecting Cross-Lingual Semantic Differences in Swiss Government Websites*.

## Overview

Given a pair of parallel web pages from `admin.ch` in English and German, French, or Italian, the task is to predict a token-level score in `[0, 1]` for every whitespace token on both sides, indicating how much its meaning differs from the other document. Systems are evaluated by global token-level Spearman rank correlation ($\rho \times 100$) per language pair, macro-averaged across `EN-DE`, `EN-FR`, and `EN-IT`.

We use **Apertus v1.5 8B** (`andreasmartin/apertus-v1.5-8b-text`) frozen as a **reader rather than a text generator**. For each document pair, four forward passes (`A` alone, `B` alone, `B` followed by `A`, and `A` followed by `B`) extract two families of per-word signals:
1. **Middle-layer DiffAlign** ($1 - \max \cos$ against the other document's subword states, raw and mean-centred, at layers 12, 14, 16, and 18).
2. **Contrastive surprisal gain** ($\text{gain} = \log p(\text{word} \mid \text{other doc, prefix}) - \log p(\text{word} \mid \text{prefix})$), measuring how much seeing the other document increases each word's log-probability; we use $-\text{gain}$ as the feature so that higher values indicate stronger semantic difference.

A regularised logistic regression head with 96 scalar features (multi-window moving averages and document-level aggregates, trained with soft-label binary cross-entropy on the 504 labelled dev documents) combines and calibrates these signals across documents.

## Headline Results

The held-out test split was touched once, from a frozen and tagged commit (`final-run`, `4c804d4`). Because the SwissGov-RSD paper's Table 3 evaluates on the full dataset (dev + test combined), we compare against the paper's released baseline predictions rescored with the official evaluation script on the exact same split.

### Test Split (Single Final Run, 56 pages / pair)

| System | EN-DE | EN-FR | EN-IT | Mean [95% CI] |
|---|---:|---:|---:|---:|
| **Ours: Apertus v1.5 8B scalar head (e), shared** | **42.8** | **23.4** | **46.0** | **37.4** [31.9, 42.4] |
| DiffAlign Qwen3-Embedding-8B (best of 7 rescored baselines) | 19.7 | 11.2 | 20.4 | 17.1 [13.7, 20.5] |
| DiffAlign EuroBERT 610m (released) | 18.3 | 10.3 | 20.5 | 16.4 [13.2, 19.3] |
| DiffAlign Qwen3-Embedding-4B (released) | 18.6 | 9.8 | 20.6 | 16.3 [12.8, 20.1] |

*(Note: we rescored the seven full-document DiffAlign baselines in `data/evaluation/encoder_predictions/full/`; the paper's segmented `LaBSE` and `XLM-R + SimCSE` runs were not rescored.)*

### Development Split Summary

| System | Split / Subset | EN-DE | EN-FR | EN-IT | Mean |
|---|---|---:|---:|---:|---:|
| Apertus few-shot prompting baseline | 30-doc short-page subset | 16.3 | -10.2 | 3.6 | 3.2 |
| Apertus DiffAlign alone raw L32 (paper's last-layer recipe) | Full dev (168 / pair) | 5.4 | 5.4 | 9.8 | 6.9 |
| Released DiffAlign Qwen3-Embedding-8B | Full dev (168 / pair) | 19.1 | 12.4 | 17.9 | 16.4 |
| Released DiffAlign Qwen3-Embedding-4B (best of 7 rescored on dev) | Full dev (168 / pair) | 20.6 | 11.7 | 20.2 | 17.5 |
| Controlled Qwen3-Embedding-8B alone raw L21 (best layer) | Full dev (168 / pair) | 30.6 | 19.7 | 33.0 | 27.8 |
| Apertus DiffAlign alone raw L14 (best layer) | Full dev (168 / pair) | 33.0 | 21.4 | 35.7 | 30.0 |
| Apertus DiffAlign cond raw L14 | Full dev (168 / pair) | 35.6 | 23.2 | 38.3 | 32.4 |
| Apertus DiffAlign cond raw L14 + 7-word smoothing (unsupervised) | Full dev (168 / pair) | 39.8 | 26.3 | 43.2 | 36.4 |
| **Ours: Apertus scalar head (e), shared (dev/train $\to$ dev/val)** | **dev/val (34 / pair)** | **38.6** | **31.8** | **43.7** | **38.0** |
| **Ours: Apertus scalar head (e), shared (nested 5-fold grouped CV)** | **Full dev (168 / pair, OOF)** | **43.2** | **29.1** | **47.5** | **39.9** |

## Run It

From `track_2a/` on a clean checkout:

```bash
make run
```

Requirements: `Docker (CPU only, x86-64, ~1 GB disk); network access during docker build only (downloads pinned GitHub/Hugging Face evaluation scripts and data, so the container needs no network at run time); ~1 minute runtime; no GPU, no model weights, no API keys, and no LLM_NAME / LLM_BASE_URL / LLM_API_KEY endpoint required.`

### What `make run` does

1. **Builds the CPU Docker image (`swissgov-rsd`):** installs locked Python dependencies (`uv.lock`), copies `src/` and `data/`, and runs `python -m official_eval.fetch --test` to cache the pinned official evaluation code (`ZurichNLP/SwissGov-RSD` @ `1807a42`) and the pinned dev and test texts and gold labels (`ZurichNLP/SwissGov-RSD` @ `f16aa05`).
2. **Fits the head and evaluates on test:** runs `python -m rsd.pipeline --out outputs`. Because `data/features_test_head.npz` is committed, `make run` executes the test branch:
   - Fits the shared 96-feature logistic regression head ($\lambda = 10^{-3}$) on all 504 dev documents (`data/features_dev_head.npz`).
   - Predicts all 168 test documents (`data/features_test_head.npz`) and writes official-format JSONL files to `outputs/test/head_{de,fr,it}.jsonl.jsonl`, along with `outputs/test/head_model.npz` and `outputs/test/head_config.json`.
   - Scores the predictions with the official evaluation script and writes `outputs/scores.json`.
   - Prints two summary lines:
     ```text
     TEST (official script, Spearman x100)
       EN-DE 42.8  EN-FR 23.4  EN-IT 46.0  mean 37.4
     ```

Because `make run` executes the same deterministic `rsd.head fit-predict` command on the same committed feature archives, it reproduces the frozen predictions in `data/predictions_test/`. To run the `dev/val` check (`38.6 / 31.8 / 43.7`, mean `38.0`) instead, run locally with `uv`:

```bash
uv run python -m rsd.head fit-predict --train-split train --predict-split val --out cache/head_check/devval_
```

or override the container command so `rsd.pipeline` runs without the test features file:

```bash
docker run --rm swissgov-rsd sh -c "rm data/features_test_head.npz && python -m rsd.pipeline --out /tmp/out"
```

Other Makefile targets (local with `uv`):
- `make test` — runs the `pytest` suite (metric agreement, test-split gates, fold grouping, feature/standardiser checks).
- `make eval-reference` — verifies the official evaluation script against the paper's released baseline predictions on dev and writes `docs/reference_dev_scores.md`.

### GPU Feature Extraction (Kaggle)

Forward passes with Apertus v1.5 8B require a GPU and are not part of `make run`. They were executed on Kaggle (2x NVIDIA T4 16 GB, ~45 min for dev, ~15 min for test) using the scripts in `kaggle/` (`01_extract_dev.py` and `04_extract_test.py`, calling `src/rsd/extract.py`). The exact commands to regenerate the `.npz` feature archives from scratch are in [`docs/final_run.md`](docs/final_run.md), [`docs/unsupervised_dev_results.md`](docs/unsupervised_dev_results.md), and Section 7 of [`technical_report.md`](technical_report.md).

## Data (`data/`, 41.8 MB total, limit 100 MB)

- `data/features_dev_head.npz` (27.5 MB): compact float32 per-word Apertus features for the 504 dev document pairs (DiffAlign at layers 12, 14, 16, 18 and word log-probabilities; no gold labels, no hidden-state vectors).
- `data/features_test_head.npz` (10.2 MB): compact float32 per-word Apertus features for the 168 test document pairs.
- `data/predictions_test/` (4.0 MB): the frozen, scored test predictions (`apertus_head_{de,fr,it}.jsonl.jsonl`), the fitted head weights (`apertus_head_model.npz`, 14 KB), and `apertus_head_config.json`.

**Licence and third-party text notice:** SwissGov-RSD annotations are licensed under CC BY 4.0 (Wastl et al., ACL 2026). Copyright of the underlying `admin.ch` texts remains with the Swiss federal authorities. Because the official evaluation format requires `text_a` and `text_b` in every prediction row, `data/predictions_test/*.jsonl.jsonl` and the dev visualisation pages in `docs/heatmaps/*.html` contain tokenised `admin.ch` text under that copyright notice. The official evaluation script is fetched at build time rather than redistributed in the repository.

## Role of Other Models

- **Apertus v1.5 8B** (`andreasmartin/apertus-v1.5-8b-text` @ `c0e7eeb`) is the sole model used in the submitted system.
- **Qwen3-Embedding-8B** (`Qwen/Qwen3-Embedding-8B` @ `1d8ad4c`) was run on the dev split only (`kaggle/02_qwen_control.py`) as an experimental control to separate the effect of layer depth from the choice of backbone.
- **Released DiffAlign predictions** from the SwissGov-RSD repository (Qwen3-Embedding 4B/8B, EuroBERT 610m/2.1B, mmBERT-base, embeddinggemma-300m, bge-m3) are rescored on dev and test as baselines.

## Repository Map

| Path | Description |
|---|---|
| [`technical_report.md`](technical_report.md) | Full technical report: architecture, fp16 fixes, ablations, error analysis, limitations |
| `src/rsd/` | Core package (`apertus.py`, `extract.py`, `features.py`, `head.py`, `metrics.py`, `pipeline.py`, analysis modules) |
| `src/official_eval/` | Downloader and wrapper for the pinned official SwissGov-RSD evaluation script |
| `kaggle/` | Kaggle 2x T4 runner and extraction scripts (`00_smoke_test.py` to `04_extract_test.py`) |
| `docs/test_results.md` | Single-run test split results and baseline comparisons |
| `docs/final_run.md` | Frozen single-run test procedure and exact CLI commands |
| `docs/final_run_meta.json` | Metadata (model SHA, code hashes, timings) recorded by the test extraction run |
| `docs/reference_dev_scores.md` | Released DiffAlign baselines rescored on dev with official and local metrics |
| `docs/ablation_dev.md` | Summary dev ablation table and EN-DE double-annotation analysis |
| `docs/unsupervised_dev_results.md` | Layer curves and unsupervised signal ablations on dev |
| `docs/qwen_control_dev_results.md` | Controlled layer-by-layer comparison with Qwen3-Embedding-8B on dev |
| `docs/prompting_baseline_dev_results.md` | Few-shot prompting baseline with Apertus on dev |
| `docs/scalar_head_dev_results.md` | Supervised logistic head protocols, feature ablations, and controls |
| `docs/heatmaps/index.html` | Interactive token-level heatmaps on 4 dev document pairs |
| `tests/` | Automated unit and hygiene tests (`pytest`) |

## License

Code in this repository is licensed under the Apache License 2.0 (see root `LICENSE`). Documentation and [`technical_report.md`](technical_report.md) are licensed under Creative Commons Attribution 4.0 (CC-BY-4.0).
