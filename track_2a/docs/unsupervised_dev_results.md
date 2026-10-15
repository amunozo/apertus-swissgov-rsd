# Unsupervised Apertus signals on dev

Spearman x 100 on the SwissGov-RSD **dev** split (168 documents per pair),
global per pair, local metric (`rsd.metrics`). **mean** is the macro average.
Higher score means "more different". Scores are mapped to [0, 1] with
`unit_interval` (no clipping). The test split was not used.

## Setup

Model: `andreasmartin/apertus-v1.5-8b-text` at commit `c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01`,
fp16 on Kaggle 2x T4 with the overflow patches in `src/rsd/apertus.py`.
For each pair and each side X (Y is the other side), four passes:

- alone_X = `[BOS] {NameX}:\n X`
- cond_X = `[BOS] {NameY}:\n Y \n\n{NameX}:\n X`

X's subword ids are the same in both passes, and only X's positions are read.

Signals per whitespace word:

- **DiffAlign** at every hidden state (0 = embeddings, 1-31 = decoder layers,
  32 = final norm output). For each subword: 1 - max cosine to the other
  document's subwords, then the mean over the word's subwords. *alone* uses X
  alone against Y alone (the paper's baseline). *cond* uses X from cond_X
  against Y from cond_Y. *cent* subtracts, per layer, the mean vector over all
  document subwords of both sides of the pair.
- **-gain**: gain = log p(word | Y) - log p(word), where the first term comes
  from the cond pass and the second from the alone pass. *sum* adds up the
  word's subwords; *first* uses the first subword only. The tables also show
  -log p_cond and -log p_alone on their own.
- **+smooth7**: a centered 7-word moving average within one side of a
  document. The window was fixed in advance, not tuned.

## Main results (dev)

| Variant | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| DiffAlign Qwen3-Embedding-8B (released, reference) | 19.1 | 12.4 | 17.9 | 16.4 |
| DiffAlign Qwen3-Embedding-8B (released) +smooth7 | 23.7 | 15.4 | 22.5 | 20.5 |
| DiffAlign alone raw L32 (last layer, as in the paper) | 5.4 | 5.4 | 9.8 | 6.9 |
| DiffAlign alone cent L32 | 4.4 | 3.5 | 8.2 | 5.4 |
| DiffAlign alone raw L14 | 33.0 | 21.4 | 35.7 | 30.0 |
| DiffAlign alone raw L14 +smooth7 | 37.5 | 24.8 | 41.0 | 34.4 |
| DiffAlign alone cent L15 | 33.0 | 21.1 | 36.5 | 30.2 |
| DiffAlign alone cent L14 +smooth7 | 37.0 | 24.1 | 41.2 | 34.1 |
| DiffAlign cond raw L14 | 35.6 | 23.2 | 38.3 | 32.4 |
| **DiffAlign cond raw L14 +smooth7** | **39.8** | **26.3** | **43.2** | **36.4** |
| DiffAlign cond cent L17 | 34.6 | 22.0 | 37.9 | 31.5 |
| DiffAlign cond cent L15 +smooth7 | 38.4 | 24.6 | 42.2 | 35.1 |
| -gain sum | 22.2 | 13.4 | 23.0 | 19.6 |
| -gain sum +smooth7 | 31.4 | 18.1 | 32.0 | 27.2 |
| -gain first | 21.3 | 12.7 | 21.8 | 18.6 |
| -gain first +smooth7 | 31.1 | 17.7 | 31.1 | 26.7 |
| -log p_cond sum | 20.7 | 11.6 | 20.2 | 17.5 |
| -log p_cond sum +smooth7 | 29.7 | 18.1 | 29.9 | 25.9 |
| -log p_cond first | 18.8 | 10.3 | 19.3 | 16.1 |
| -log p_cond first +smooth7 | 28.4 | 17.0 | 29.1 | 24.8 |
| -log p_alone sum (control: no access to Y) | 0.8 | -0.2 | -0.5 | 0.0 |
| -log p_alone first (control) | 0.4 | -0.3 | 0.5 | 0.2 |
| control: subwords per word | 4.6 | 2.0 | 1.3 | 2.6 |
| control: word position | 4.0 | -3.9 | 2.2 | 0.8 |

The DiffAlign rows use the best layer of each family on all of dev (see the
selection caveat below).

## DiffAlign layer curve (dev mean, no smoothing)

| Layer | alone raw | alone cent | cond raw | cond cent |
|---|---|---|---|---|
| 0 | 9.1 | 9.2 | 9.1 | 9.2 |
| 2 | 11.4 | 13.9 | 11.3 | 13.2 |
| 4 | 17.5 | 18.8 | 17.0 | 18.0 |
| 6 | 20.6 | 21.4 | 20.1 | 20.6 |
| 8 | 24.8 | 24.8 | 24.3 | 23.9 |
| 10 | 26.6 | 26.9 | 27.4 | 26.3 |
| 12 | 28.9 | 28.6 | 30.2 | 28.7 |
| 14 | 30.0 | 30.2 | 32.4 | 31.4 |
| 16 | 29.2 | 29.9 | 31.6 | 31.1 |
| 18 | 28.0 | 29.1 | 30.7 | 30.6 |
| 20 | 26.2 | 27.8 | 28.8 | 29.5 |
| 22 | 23.8 | 26.0 | 25.7 | 27.9 |
| 24 | 20.9 | 23.8 | 22.8 | 25.6 |
| 26 | 18.2 | 21.8 | 18.0 | 23.3 |
| 28 | 14.4 | 18.7 | 12.0 | 19.1 |
| 30 | 12.1 | 15.9 | 10.1 | 15.6 |
| 31 | 10.2 | 14.1 | 8.3 | 13.7 |
| 32 | 6.9 | 5.4 | 4.4 | 5.4 |

The full curve (every layer, with and without smoothing) is printed by the
command below. The middle layers (12-18) are best. The final layer is
the worst after the embeddings. Centering makes little difference at the best
layers but helps a lot from layer 22 on.

## Selection caveat

The best layer, the best family, and the decision to smooth were all picked by
looking at all of dev, so the best dev number (36.4) is optimistic. As a check,
the selection was repeated on the official dev/train part (134 pages per pair)
and scored on dev/val (34 pages):

| Family | layer chosen on dev/train | dev/train mean | dev/val mean |
|---|---|---|---|
| DiffAlign alone raw | L14 | 30.5 | 27.4 |
| DiffAlign alone cent | L15 | 30.6 | 28.1 |
| DiffAlign cond raw | L14 | 32.8 | 30.3 |
| DiffAlign cond raw +smooth7 | L14 | 36.8 | 34.4 |
| DiffAlign cond cent +smooth7 | L15 | 35.4 | 33.1 |

Over all 276 variants, the one chosen on dev/train is the same as the one chosen on
all dev (cond raw L14 +smooth7): 36.8 on dev/train, 34.4 on dev/val. dev/val is
small (34 pages), so its scores are noisy. The layer choice is stable (L14-L15
in every family), and the curve is flat around it, so the selection effect
looks small. Still, these numbers are not a held-out estimate.

## Comparison with the released DiffAlign Qwen3-Embedding-8B (dev)

Paired bootstrap over pages (`rsd.metrics.paired_bootstrap`, 1000 resamples,
seed 0). Difference x 100 [95% CI]:

| Comparison | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| cond raw L14 +smooth7 - Qwen8B | +20.8 [+18.4, +23.3] | +13.9 [+11.3, +16.5] | +25.3 [+22.9, +28.1] | +20.0 [+18.3, +21.7] |
| cond raw L14 +smooth7 - Qwen8B +smooth7 | +16.2 [+14.0, +18.2] | +10.9 [+8.8, +13.1] | +20.7 [+18.5, +22.9] | +15.9 [+14.4, +17.4] |
| cond raw L14 (no smoothing) - Qwen8B | +16.5 [+14.4, +18.7] | +10.9 [+8.9, +12.9] | +20.4 [+18.4, +22.7] | +15.9 [+14.6, +17.5] |

Official script spot-check (`scripts/evaluate_predictions_admin.py`, dev),
which matches the local metric:

| Variant | EN-DE | EN-FR | EN-IT |
|---|---|---|---|
| DiffAlign cond raw L14 +smooth7 | 39.8 | 26.3 | 43.2 |
| DiffAlign cond raw L15 +smooth7 | 39.5 | 26.2 | 42.9 |
| DiffAlign cond raw L16 +smooth7 | 39.2 | 25.7 | 42.5 |

## What this shows, and what it does not

- Apertus DiffAlign at a middle layer beats the released Qwen3-Embedding-8B
  DiffAlign on dev by about 14 points without smoothing (30.0 vs 16.4 for the
  plain alone-pass baseline). For the cond variant, the bootstrap CIs above
  exclude 0 for every pair. At the
  final layer, which is what the paper used, Apertus is much worse (6.9).
  Layer choice matters more than anything else here.
- So the comparison here is "middle-layer Apertus vs last-layer Qwen". See
  `docs/qwen_control_dev_results.md` for the controlled experiment that runs
  Qwen3-Embedding-8B across all 37 layers with the same recipe.
- Part of the score is document-level. Giving every word its document side's
  mean score keeps 18.7 of the 36.4. Shuffling scores within each document
  side keeps only 9.0, and breaking the score-document alignment drops it to
  about 0. Within-document ranking alone (mean per-document Spearman 31.4 /
  23.6 / 35.3) still clearly beats Qwen (14.8 / 9.7 / 13.3).
- Conditioning on the other document helps a little: cond vs alone is +2.4 at
  L14 without smoothing.
- The log-prob gain is a usable signal on its own (19.6, or 27.2 smoothed).
  Note that -log p_alone scores about 0, so the gain signal does not come from
  word rarity. The trivial controls also score near 0.
- Smoothing adds about 4 points to every DiffAlign variant, and about 4 to
  Qwen too. It uses the fact that gold differences come in spans of
  neighbouring words. That is a property of the labels, not of the model.
- The best DiffAlign layer was chosen on dev (see the caveat). The paper's
  Table 3 numbers are on the whole dataset (dev and test together), not the dev
  split, so they are not directly comparable to these dev numbers; compare with the released predictions
  rescored on dev, as above.
- Massive activations: at document positions, max |state| is about 5.7k (layer 31).
  The ~119k values seen in the smoke test sit at the BOS or header positions,
  which are never read.

## Commands and cost

From `track_2a/`:

```
# GPU (Kaggle 2x T4): features + states, kernel amunozo/rsd-extract-dev
python kaggle/run_on_kaggle.py kaggle/01_extract_dev.py --kernel amunozo/rsd-extract-dev \
    --files '^(features\.npz|meta\.json)$'
# CPU: all tables above (about 6 minutes), plus the official spot-check
uv run python -m rsd.unsupervised kaggle/outputs/01_extract_dev/features.npz \
    --official cache/unsupervised_preds
```

GPU time: one Kaggle run of about 45 minutes in session (model load 148 s,
extraction 2,378 s wall, including 2,117 s in the 2,016 forward passes, which processed
1.68M tokens at about 790 tok/s). Outputs: 13.55 GB on
Kaggle (12 word-pooled state files of 1.0-1.2 GB at layers 8/16/24/32, plus
`features.npz` 250 MB). Only `features.npz` and `meta.json` were downloaded.
