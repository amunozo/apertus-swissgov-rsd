# Supervised scalar head on dev

Spearman x 100 on the SwissGov-RSD **dev** split, global per language pair,
local metric (`rsd.metrics`). **mean** is the macro average over EN-DE, EN-FR,
and EN-IT. The test split was not used.

The baseline is the best training-free signal from
`docs/unsupervised_dev_results.md`: DiffAlign, conditioned passes, raw, layer
14, 7-word smoothing. It scores 36.4 on all of dev and 34.4 on dev/val.

## Setup

- **Input.** `kaggle/outputs/01_extract_dev/features.npz`: one row per
  whitespace word with DiffAlign at all 33 hidden states (alone/cond x
  raw/centred) and the log-probs from the alone and cond passes. No hidden
  states are used here.
- **Model.** Logistic regression on per-word features:
  - trained with soft-label binary cross-entropy (gold in [0, 1] is the
    target, so 0.4 counts as a soft target);
  - tokens with gold -1 (punctuation) are left out of training;
  - L2 penalty on the weights, not on the intercept;
  - fitted with L-BFGS (scipy).
  The raw score (logit) is the prediction, mapped to [0, 1] with
  `unit_interval`. There is no clipping and no smoothing of the output.
- **Standardisation.** Each feature is z-scored with the mean and std of the
  training rows only, then clipped to [-5, 5]. Clipping the *features* tames a
  few extreme log-prob values. It is not clipping of predictions.
- **Shared vs per-pair.** "Shared" is one model for all three pairs.
  "Per-pair" is one model per pair, each with its own standardisation. Pair
  indicator features were not added: an intercept shift per pair cannot
  change a ranking inside one pair, and the metric ranks within each pair.
- **Gold.** Gold labels are used only as the training target in training
  folds, and for scoring. No feature uses gold. `rsd.features.build` takes
  only the features.npz arrays (this is tested).

## Features (`src/rsd/features.py`)

All signals are oriented so that higher means more different.

- **Windows.** Every per-word signal enters at windows 1 (raw), 3, 7, and
  15. Each window is a centred moving average inside one side of one
  document, and never crosses a document or a side.
- **Document aggregates of a signal.** For each word:
  - the mean of the signal over its document side ("side mean");
  - the mean over the other side ("other mean");
  - the mean over both sides ("pair mean").
- **Layout features** (no model signal):
  - log length of the side;
  - log length ratio of the two sides;
  - side indicator (B = de/fr/it side);
  - relative position in the side.
  The log length of the other side is left out because it equals log length
  minus log ratio, an exact collinearity.

Feature sets for the ablation (each is a column subset of "e"):

| Set | Content | n features |
|---|---|---|
| (b) | DiffAlign cond raw L14 at 4 windows; its 3 doc aggregates; 4 layout features | 11 |
| (c) | (b) + -gain sum, -gain first, -log p_cond sum, -log p_alone sum at 4 windows; doc aggregates of -gain sum | 30 |
| (d) | (b) + DiffAlign {alone, cond} x {raw, cent} x L{12, 14, 16, 18} at 4 windows; doc aggregates of cond cent L14 and alone raw L14 | 77 |
| (e) | everything in (c) and (d) | 96 |

## Protocol

1. **5-fold CV over all of dev, grouped by `page_en`.**
   - The documents of all three pairs go into one list, and
     `rsd.data.group_folds` (scikit-learn GroupKFold) splits it. GroupKFold
     is deterministic, so no seed is needed.
   - A page is in the evaluation fold for all three pairs at once, and never
     in training at the same time (this is tested).
   - Inside each training fold, the L2 strength lambda is chosen by an inner
     4-fold grouped CV from {1e-5, 1e-4, 1e-3, 1e-2}. Lambda is on the mean
     loss.
   - The out-of-fold predictions of all 5 folds are pooled and scored once,
     per pair.
2. **Official split.** Fit on dev/train (134 pages per pair), with lambda
   chosen by an inner 4-fold grouped CV on dev/train. Score on dev/val (34
   pages).

**What is nested and what is not.**
- **Nested:** only lambda is chosen inside each outer training fold.
- **Not nested: the configuration** (feature set, shared vs per-pair). It
  was chosen by the inner-CV score on dev/train. dev/train is not inside the
  outer CV: it overlaps about 80% of each outer training fold, and also part
  of each outer evaluation fold. So the CV number of the chosen configuration
  is slightly optimistic.
- **The clean number for the chosen configuration is dev/val.** Both the
  configuration and lambda were chosen on dev/train only, and dev/val was
  never used for any choice (but see the post-hoc switch under "Chosen final
  configuration").
- **Fixed in advance:** the feature windows (1/3/7/15) and layers (12-18).
  They come from the unsupervised layer curve, which was read on all of dev.
  The head inherits this mild optimism from the baseline.

**Grid boundary check.** For the shared (b) and (c) models, lambda landed at
the top of the grid (1e-2). Extending the grid upward (non-nested 5-fold CV on
all dev) shows the curve is flat there and drops after it:

| | 1e-3 | 1e-2 | 1e-1 | 1 |
|---|---|---|---|---|
| (b) shared | 37.5 | 37.5 | 37.3 | 36.6 |
| (c) shared | 39.3 | 39.3 | 39.2 | 38.7 |

So the grid was wide enough. The chosen final configuration picks an interior
lambda in any case.

## Results

### Ablation, protocol 1: nested 5-fold grouped CV over all dev (out-of-fold)

| System | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| (a) baseline: DiffAlign cond raw L14 +smooth7 (no training) | 39.8 | 26.3 | 43.2 | 36.4 |
| (b) L14 cond signal + doc aggregates, shared | 40.9 | 26.9 | 44.7 | 37.5 |
| (b) L14 cond signal + doc aggregates, per-pair | 40.3 | 26.8 | 44.6 | 37.2 |
| (c) = (b) + log-prob/gain, shared | 42.7 | 28.1 | 47.1 | 39.3 |
| (c) = (b) + log-prob/gain, per-pair | 42.3 | 28.0 | 46.8 | 39.0 |
| (d) = (b) + DiffAlign L12-18, shared | 41.8 | 28.3 | 45.7 | 38.6 |
| (d) = (b) + DiffAlign L12-18, per-pair | 41.3 | 29.0 | 46.0 | 38.8 |
| **(e) all features, shared (recommended)** | **43.2** | **29.1** | **47.5** | **39.9** |
| (e) all features, per-pair (rule's pick) | 42.9 | 29.5 | 47.5 | 40.0 |

### Ablation, protocol 2: fit on dev/train, score dev/val

| System | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| (a) baseline (no training) | 33.9 | 29.3 | 40.1 | 34.4 |
| (b) shared | 35.0 | 29.9 | 41.5 | 35.5 |
| (b) per-pair | 34.6 | 29.9 | 40.5 | 35.0 |
| (c) shared | 36.7 | 31.2 | 43.4 | 37.1 |
| (c) per-pair | 36.7 | 30.7 | 42.5 | 36.6 |
| (d) shared | 37.9 | 30.9 | 42.6 | 37.1 |
| (d) per-pair | 37.5 | 30.6 | 41.8 | 36.6 |
| **(e) shared (recommended)** | **38.6** | **31.8** | **43.7** | **38.0** |
| (e) per-pair (rule's pick) | 38.6 | 31.5 | 42.9 | 37.6 |

### Selection inputs (dev/train only)

| System | n features | inner 4-fold CV on dev/train (mean) | lambda on dev/train | lambdas in the 5 outer folds |
|---|---|---|---|---|
| (b) shared | 11 | 37.9 | 1e-2 | 1e-2 x5 |
| (b) per-pair | 11 | 38.0 | 1e-4 | 1e-2, 1e-5, 1e-5, 1e-5, 1e-2 |
| (c) shared | 30 | 39.7 | 1e-2 | 1e-2 x5 |
| (c) per-pair | 30 | 39.8 | 1e-4 | 1e-2, 1e-5, 1e-4, 1e-4, 1e-2 |
| (d) shared | 77 | 38.9 | 1e-4 | 1e-4 x5 |
| (d) per-pair | 77 | 39.1 | 1e-4 | 1e-4 x5 |
| (e) shared | 96 | 40.4 | 1e-3 | 1e-3 x5 |
| (e) per-pair | 96 | 40.5 | 1e-3 | 1e-3, 1e-4, 1e-4, 1e-4, 1e-4 |

### Paired bootstrap vs the baseline

1000 page resamples, seed 0, `rsd.metrics.paired_bootstrap`. The head and the
baseline are scored on the same documents in each resample. Difference x 100
[95% CI].

**These CIs understate the uncertainty.** They resample pages with the
predictions held fixed. They do not include the variation that comes from
retraining the head on other pages.

A more honest spread is the gain per outer CV fold, for the (e) head, as
recomputed by an independent reviewer. The folds differ by more than the
bootstrap CI suggests:

| Fold | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| gain vs baseline (mean) | +3.9 | +3.4 | +4.5 | +2.2 | +4.3 |

Every fold gains, but the range is about 2 to 4.5 points.

All of dev (out-of-fold predictions vs baseline):

| Comparison | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| (b) shared | +1.1 [+0.8, +1.3] | +0.6 [+0.3, +0.9] | +1.5 [+1.2, +1.8] | +1.1 [+0.9, +1.3] |
| (b) per-pair | +0.5 [+0.0, +1.0] | +0.6 [-0.1, +1.2] | +1.4 [+0.8, +2.0] | +0.8 [+0.5, +1.2] |
| (c) shared | +2.9 [+2.3, +3.5] | +1.8 [+1.2, +2.5] | +3.8 [+3.2, +4.5] | +2.8 [+2.5, +3.3] |
| (c) per-pair | +2.5 [+1.8, +3.1] | +1.7 [+0.9, +2.5] | +3.6 [+2.8, +4.4] | +2.6 [+2.2, +3.1] |
| (d) shared | +1.9 [+1.1, +2.6] | +2.0 [+1.0, +2.9] | +2.5 [+1.8, +3.2] | +2.1 [+1.7, +2.6] |
| (d) per-pair | +1.5 [+0.7, +2.3] | +2.7 [+1.4, +4.5] | +2.8 [+2.1, +3.5] | +2.3 [+1.7, +3.0] |
| **(e) shared (recommended)** | +3.3 [+2.5, +4.1] | +2.8 [+2.0, +3.7] | +4.2 [+3.5, +5.0] | **+3.5 [+3.0, +3.9]** |
| (e) per-pair (rule's pick) | +3.1 [+2.1, +4.1] | +3.3 [+2.1, +4.5] | +4.2 [+3.4, +5.1] | +3.5 [+3.0, +4.1] |

dev/val (fit on dev/train) vs baseline on dev/val:

| Comparison | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| (b) shared | +1.1 [+0.3, +1.9] | +0.6 [+0.1, +1.2] | +1.4 [+0.9, +2.2] | +1.0 [+0.6, +1.6] |
| (b) per-pair | +0.6 [-0.9, +2.1] | +0.6 [-0.8, +2.0] | +0.4 [-1.1, +2.5] | +0.5 [-0.4, +1.8] |
| (c) shared | +2.8 [+1.3, +4.4] | +1.9 [+0.7, +3.5] | +3.3 [+2.2, +4.5] | +2.7 [+1.9, +3.7] |
| (c) per-pair | +2.7 [+0.6, +4.6] | +1.3 [-0.8, +3.8] | +2.4 [+0.7, +4.4] | +2.2 [+0.9, +3.7] |
| (d) shared | +3.9 [+1.9, +5.6] | +1.6 [-0.7, +3.9] | +2.5 [+1.4, +3.7] | +2.7 [+1.4, +3.9] |
| (d) per-pair | +3.6 [+1.0, +6.2] | +1.3 [-1.5, +4.0] | +1.7 [+0.2, +3.5] | +2.2 [+0.6, +4.1] |
| **(e) shared (recommended)** | +4.6 [+2.2, +7.3] | +2.5 [+0.6, +5.0] | +3.6 [+2.4, +4.9] | **+3.6 [+2.4, +5.0]** |
| (e) per-pair (rule's pick) | +4.6 [+1.4, +7.8] | +2.1 [-0.5, +5.1] | +2.8 [+1.2, +4.6] | +3.2 [+1.6, +5.2] |

### Official script check

The out-of-fold predictions of (e) per-pair (the configuration picked by the
rule at the time of the run) were written in the official format (`rsd.predictions.write_predictions`) to
`cache/scalar_head_preds/head_oof_*.jsonl.jsonl` (gitignored). The official
script (`src/official_eval`) gives the same numbers as the local metric:

| | EN-DE | EN-FR | EN-IT |
|---|---|---|---|
| official | 42.9 | 29.5 | 47.5 |
| local | 42.9 | 29.5 | 47.5 |

## What the gain comes from

**By feature group** (ablation, CV means):

- **Doc aggregates and calibration of the one signal**, (b) vs (a): +1.1. Small,
  but the CI excludes 0.
- **Adding the log-prob / gain signals**, (c) vs (b): +1.8.
- **Adding the other DiffAlign layers and variants**, (d) vs (b): +1.1 to
  +1.6.
- **Both together**, (e): +3.5 over the baseline.
- **The two groups are only partly additive.** With shared models, (c) adds
  +1.8 over (b) and (d) adds +1.1, but (e) adds only +2.4 over (b), not
  +2.9. They overlap, but each carries some information the other lacks.

**Within documents vs across documents.** The same split as in the
unsupervised results:

| | mean per-document Spearman (DE / FR / IT, mean) | side-mean only (DE / FR / IT, mean) |
|---|---|---|
| baseline | 31.4 / 23.6 / 35.3, 30.1 | 22.6 / 12.6 / 21.0, 18.7 |
| head (e) per-pair, OOF | 34.1 / 27.4 / 40.4, 34.0 | 24.3 / 13.1 / 22.3, 19.9 |

"Side-mean only" gives every word its document side's mean score.

- Most of the gain is in the ranking *within* documents: +3.9 on the
  per-document Spearman.
- The document-level part improves only a little: +1.2.
- This is not what we expected. We thought the head would mainly help by
  calibrating scores across documents.

**Coefficients.** Fit on all of dev; features are standardised, so the
weights are comparable. Weights of correlated features (the same signal at
nearby layers or windows) often come in +/- pairs. Such a pair is a contrast
between two features, not an effect of either one alone. So single weights
should not be read one by one. Two patterns are stable across pairs and
models:

- **The 15-word window gets the most weight.** In the shared (e) model
  (lambda 1e-3), the sums of absolute weights by window are:

  | Window | sum of abs weights |
  |---|---|
  | 15 | 3.8 |
  | 7 | 2.6 |
  | 3 | 2.2 |
  | unsmoothed | 1.8 |
  | doc aggregates (all) | 2.4 |

  Wider context helps. This again exploits the fact that differences come in
  spans.
- **The baseline signal stays the largest single weight**: DiffAlign cond raw
  L14 at window 15. It gets +1.70 (DE), +1.37 (FR), and +1.73 (IT) in the
  per-pair final models, and +0.73 in the shared model.

Top weights, shared (e) model, lambda 1e-3:

| feature | weight |
|---|---|
| da cond raw L14 s15 | +0.730 |
| da cond raw L14 s7 | +0.416 |
| da alone raw L14 pair_mean | -0.397 |
| da cond raw L14 side_mean | +0.388 |
| da cond cent L12 s15 | -0.367 |
| da alone raw L12 s15 | -0.364 |
| da cond cent L14 pair_mean | -0.327 |
| da cond cent L12 s7 | -0.325 |
| da cond cent L18 s15 | -0.323 |
| da cond raw L14 s3 | +0.305 |
| da cond cent L16 s15 | +0.276 |
| da alone cent L14 s15 | +0.273 |
| da cond cent L14 other_mean | +0.250 |
| -gain sum pair_mean | -0.245 |
| da cond cent L12 s3 | -0.243 |

The per-pair final-model tables are printed by the command below.

## Sanity controls

An independent reviewer reran the head with its own code: shared (e), lambda
1e-3, non-nested 5-fold grouped CV over all dev. It recomputed the CV mean of
(e) per-pair as 39.96, which matches the 40.0 above. Controls (mean
Spearman x 100):

| Control | mean | What it shows |
|---|---|---|
| Train on shuffled gold labels | 2.3 | The score needs the real training labels. The pipeline itself (features, folds, scoring) does not create signal. A later re-run with a different shuffle (seed 0) gives 3.2, so the exact value depends on the shuffle; the conclusion is the same. |
| Train on document-side mean labels (each word's training label is the mean gold of its document side; scored against the real gold) | 21.0 | A head that can only learn "which documents differ more" scores about 21, far below 39.9. So the head learns from word-level labels. A later re-run of this control gives 21.0 exactly. |
| "Head predictions circularly shifted within each document side" | 19.4 | **Not reproducible from the repo.** The reviewer's code is not in the repo, and an earlier note described it as shifted training labels rather than shifted predictions. Later re-runs: training labels shifted by half the side length give 6.9; predictions shifted by 15 words give 20.4 and by 30 words give 16.1. So the value depends on the protocol, and it is not used in any conclusion. The shuffled and document-mean controls carry the argument. |

Added after the fact (not part of the reviewer's controls): the gold document-side
mean oracle, where every word gets the true mean gold label of its document side
and no model is involved, scores 31.2 / 26.2 / 33.3, mean 30.3 on dev. The 21.0
above is a different quantity: it is a head trained on those mean labels, not
the oracle itself. An earlier version of this table labelled 21.0 as the gold
oracle and compared it with the head's side-mean-only score of 19.9; that was
wrong, and the comparison is dropped.

## Chosen final configuration

**Final configuration: feature set (e) (all 96 features), one shared model for the
three pairs, lambda 1e-3.**

The pre-stated rule (best inner-CV score on dev/train) picked (e) per-pair,
by 40.5 vs 40.4. We switched to shared after seeing the results. This is a
post-hoc switch on a tie, and we say so.

Reasons for the switch:
- **The two tie on CV:** 39.9 (shared) vs 40.0 (per-pair).
- **Shared is better on dev/val:** 38.0 vs 37.6. Its EN-FR CI also excludes
  0; per-pair's does not.
- **Shared's lambda is stable.** Lambda 1e-3 was chosen in all 5 outer folds
  and on dev/train. For (e) per-pair, lambda moves between 1e-3 and 1e-4 across
  folds (1e-3 once, 1e-4 four times; 1e-2 to 1e-5 for sets b and c).
- **Shared has a third of the parameters.**

Lambda: we keep 1e-3, the value every selection step picked. Because of the
switch, dev/val is no longer perfectly clean for this choice either.

| Numbers for shared (e) | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| nested CV (all dev; configuration choice not nested) | 43.2 | 29.1 | 47.5 | 39.9 |
| dev/val (fit on dev/train) | 38.6 | 31.8 | 43.7 | **38.0** |
| gain vs baseline, CV | +3.3 | +2.8 | +4.2 | +3.5 [+3.0, +3.9] |
| gain vs baseline, dev/val | +4.6 | +2.5 | +3.6 | +3.6 [+2.4, +5.0] |

Expect roughly 38-40 on dev-like data. The per-fold spread above (+2.2 to
+4.5) is a fairer picture of the uncertainty than the bootstrap CIs.

**Final-run path** (`python -m rsd.head fit-predict`):
- **No search.** The config is fixed and set by CLI flags (default: shared,
  (e), lambda 1e-3).
- **Fit:** on a features file (default: all of dev from
  `data/features_dev_head.npz`).
- **Predict:** another features file, with `unit_interval` per pair.
- **Outputs:**
  - predictions in the official format;
  - `<prefix>model.npz` (standardiser means and stds, weights, intercept,
    feature names);
  - `<prefix>config.json`.
- **Test needs `--final`.** Predicting test is refused without it (run once for
  the final test evaluation; see `docs/test_results.md`).
- **Checked on dev:** fit on dev/train, predict dev/val. It reproduces this
  document's dev/val row exactly (38.6 / 31.8 / 43.7, mean 38.0). A test
  checks this.

Final-run commands (after the test features exist; see `docs/final_run.md`):

```
uv run python -m rsd.features <test features.npz> data/features_test_head.npz
uv run python -m rsd.head fit-predict --final --predict-split test \
    --predict-features data/features_test_head.npz --out <prefix without "llm">
```

## Caveats

- **Shared vs per-pair is a tie.** The rule picked per-pair by a 0.1 margin
  on dev/train, which is noise. The recommendation of shared is a post-hoc
  switch (see above).
- **Only lambda is nested.** The configuration was chosen on dev/train, which
  overlaps the outer CV folds. Use dev/val (38.0) as the clean number.
- **The bootstrap CIs are too narrow.** They hold the trained models fixed;
  see the per-fold gains.
- **The layers and windows were fixed using all of dev.** The layer range
  12-18 and the windows come from the unsupervised layer curve, which was
  read on all of dev. The nested CV covers lambda only, not this. The effect should be small, because the curve is flat
  around L14. Still, the CV numbers are slightly optimistic.
- **dev/val is small** (34 pages per pair). Its CIs are wide; for EN-FR the
  per-pair model's CI includes 0.
- **The per-pair lambda varies** across outer folds (1e-3 to 1e-4 for set e;
  1e-2 to 1e-5 for sets b and c). The
  CV curve is flat, so this does not matter much for the score. It does mean
  the per-pair models are not well identified.
- **The within-document vs document-level split and the official-script
  check were run on the (e) per-pair predictions.** Shared (e) scores within
  0.1 of it, so we expect the same picture. We did not recompute these for
  shared.
- **The gain is moderate.** The head adds about 3.5 points on top of a strong
  training-free signal. Most of it comes from combining features that already
  exist (more layers, gain, wider windows), not from new information.
- **The baseline is not a fixed reference.** Its layer and smoothing were
  chosen on all of dev, so it is slightly optimistic too. The comparison is
  between two dev-tuned systems, with the head evaluated out-of-fold.
- **Not tried:** hidden states as features (the "states head") and non-linear
  models.

## Commands and cost

From `track_2a/`, CPU only:

```
# compact features for the head (27.5 MB, float32, compressed; bit-identical features)
uv run python -m rsd.features kaggle/outputs/01_extract_dev/features.npz data/features_dev_head.npz
# full evaluation: ablation, nested CV, dev/val, bootstrap, official check
uv run python -m rsd.head data/features_dev_head.npz \
    --official cache/scalar_head_preds --n-boot 1000
# fixed-config path, dev-only check (about 15 s)
uv run python -m rsd.head fit-predict --train-split train --predict-split val --out cache/head_check/devval_
```

- **Full evaluation:** about 54 minutes on a 12-core laptop. Most of that is
  the nested CV for the 96-feature sets and the 16 bootstraps. Use
  `--sets bc` for a quick run.
- **Compact file:** it keeps only the arrays the head needs (ids, four
  log-prob arrays, DiffAlign at layers 12/14/16/18), and no gold. The
  fit-predict output from it is byte-identical to that from the 250 MB
  `features.npz`.
- **Tests:** `uv run pytest` (`tests/test_head.py`).
