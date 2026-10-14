# Qwen control: is the gain from the layer or from Apertus?

Spearman x 100 on the SwissGov-RSD **dev** split (168 documents per pair),
global per pair, local metric (`rsd.metrics`). **mean** is the macro average.
The test split was not used.

## Question

Apertus DiffAlign at layer 14 scores 30.0 on dev. The paper's released
DiffAlign with Qwen3-Embedding-8B scores 16.4, but it uses the last layer. So
our comparison was "middle-layer Apertus vs last-layer Qwen". This control
runs Qwen3-Embedding-8B through DiffAlign at every layer, to separate the
effect of the layer from the effect of the model.

## Setup

Model: `Qwen/Qwen3-Embedding-8B` at commit `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`,
script `kaggle/02_qwen_control.py`. We followed the official recipe
(ZurichNLP/SwissGov-RSD at the pinned commit: `rsd/recognizers/diff_align.py`,
`feature_based.py`, `scripts/predict_encoder.py --type diffalign`):

- Each document is encoded alone, as plain text with no prompt. The input is
  `tok(text, padding=True, truncation=True)`, so the whitespace-tokenized text
  followed by `<|endoftext|>`, which the tokenizer appends. No document is
  truncated (the model's max length is 131k).
- Subwords are grouped into whitespace words the official way for this
  tokenizer: a token that starts with `Ġ` starts a new word. Special tokens
  are skipped. On all 1,008 document sides this grouping gives the gold word
  count and matches our own offset-based grouping exactly.
- Score per subword: 1 - max cosine to every position of the other
  document's sequence (its `<|endoftext|>` included, as in the official code).
  A word's score is the mean over its subwords.
- *cent* first subtracts, per layer, the mean state over the document
  subwords of both sides (special tokens excluded), like the Apertus
  features.
- Layers: 0 = embeddings, 1-35 = decoder layers, 36 = final norm output.
  Layer 36 is the official `layer=-1`.
- *+smooth7*: the same centred 7-word moving average within one document
  side as in `unsupervised_dev_results.md`.

Deviations from the official run:
- fp16 weights, with the residual stream and the MLP activation in fp32 (the
  same patches as for Apertus). The official run used fp32 weights.
- Every layer is scored, not only the last.

At document positions, max |state| is 15.2k (layers 17-34), below the fp16
limit of 65,504. No NaN or inf appeared.

## Reproduction check (last layer)

Our layer-36 scores against the released dev predictions, over every
whitespace token (punctuation included):

| Pair | tokens | Pearson | Spearman | max abs diff | mean abs diff |
|---|---|---|---|---|---|
| EN-DE | 126,421 | 1.0000 | 1.0000 | 0.058 | 0.00024 |
| EN-FR | 142,905 | 1.0000 | 1.0000 | 0.159 | 0.00025 |
| EN-IT | 136,328 | 1.0000 | 1.0000 | 0.080 | 0.00025 |

| Variant | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| Released Qwen3-Embedding-8B DiffAlign | 19.1 | 12.4 | 17.9 | 16.4 |
| Ours, layer 36 raw (official recipe, fp16) | 19.1 | 12.4 | 17.9 | 16.4 |

The pipeline reproduces the released system to the first decimal. The small
per-token differences come from fp16 vs fp32 (and the released 5-decimal
rounding). So the middle-layer numbers below come from the same pipeline that
produced the paper's number.

## Layer curve (alone passes, dev mean)

| Layer | raw | centred | raw +smooth7 | centred +smooth7 |
|---|---|---|---|---|
| 0 | 7.9 | 7.9 | 7.9 | 7.8 |
| 2 | 9.2 | 11.0 | 10.5 | 12.6 |
| 4 | 11.9 | 13.0 | 15.0 | 16.1 |
| 6 | 13.6 | 15.0 | 17.6 | 19.0 |
| 8 | 17.6 | 16.3 | 22.6 | 19.4 |
| 10 | 19.9 | 19.1 | 25.6 | 23.4 |
| 12 | 22.0 | 20.8 | 28.0 | 25.4 |
| 14 | 24.1 | 22.5 | 30.2 | 27.2 |
| 16 | 25.3 | 24.2 | 31.4 | 28.9 |
| 18 | 26.9 | 23.8 | 32.6 | 27.5 |
| 19 | 27.3 | 24.6 | 33.2 | 28.3 |
| 20 | 27.5 | 25.2 | 33.4 | 29.2 |
| **21** | **27.8** | 25.6 | **33.5** | 29.9 |
| 22 | 27.4 | 25.5 | 33.1 | 30.4 |
| 24 | 25.6 | 24.7 | 32.4 | 31.2 |
| 26 | 23.2 | 22.7 | 30.4 | 30.3 |
| 28 | 21.7 | 22.2 | 28.3 | 29.6 |
| 30 | 20.0 | 21.7 | 25.4 | 28.6 |
| 32 | 17.5 | 20.1 | 21.0 | 25.8 |
| 34 | 15.0 | 18.7 | 17.3 | 23.4 |
| 35 | 13.9 | 18.2 | 16.1 | 22.9 |
| 36 (official) | 16.4 | 16.8 | 20.5 | 22.0 |

The full curve (all 37 layers) is printed by the command below. The
curve has the same shape as for Apertus: it rises to a peak in the middle and
falls towards the end. Qwen peaks at layer 21 of 36 (58% depth), Apertus at
layer 14 of 32 (44% depth). Unlike Apertus, mean-centring hurts Qwen at the
best layers. It only helps in the last third.

Best layer per family, chosen on all dev, and chosen on the official
dev/train part then scored on dev/val:

| Family | best L on all dev | dev mean | best L on dev/train | dev/train mean | its dev/val mean |
|---|---|---|---|---|---|
| Qwen alone raw | L21 | 27.8 | L21 | 28.2 | 25.3 |
| Qwen alone raw +smooth7 | L21 | 33.5 | L21 | 34.0 | 30.8 |
| Qwen alone cent | L21 | 25.6 | L21 | 26.1 | 22.9 |
| Qwen alone cent +smooth7 | L24 | 31.2 | L24 | 31.5 | 29.3 |

The layer choice is the same on dev/train as on all dev. For comparison,
Apertus alone raw L14 scored 27.4 on dev/val when chosen on dev/train.

## Qwen vs Apertus, same recipe (dev)

| Variant | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| Qwen alone raw L36 (paper recipe) | 19.1 | 12.4 | 17.9 | 16.4 |
| Qwen alone raw L21 | 30.6 | 19.7 | 33.0 | 27.8 |
| Qwen alone raw L21 +smooth7 | 36.4 | 23.9 | 40.1 | 33.5 |
| Apertus alone raw L32 (paper recipe) | 5.4 | 5.4 | 9.8 | 6.9 |
| Apertus alone raw L14 | 33.0 | 21.4 | 35.7 | 30.0 |
| Apertus alone raw L14 +smooth7 | 37.5 | 24.8 | 41.0 | 34.4 |
| Apertus cond raw L14 +smooth7 (our best) | 39.8 | 26.3 | 43.2 | 36.4 |

Paired bootstrap over pages (`rsd.metrics.paired_bootstrap`, 1000 resamples,
seed 0). Difference x 100 [95% CI]:

| Comparison | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| Apertus alone raw L14 - Qwen alone raw L21 | +2.3 [+1.7, +2.9] | +1.7 [+1.1, +2.3] | +2.6 [+2.0, +3.2] | +2.2 [+1.8, +2.6] |
| Apertus alone raw L14 +smooth7 - Qwen alone raw L21 +smooth7 | +1.1 [+0.3, +1.8] | +0.9 [+0.1, +1.7] | +0.9 [+0.2, +1.6] | +1.0 [+0.5, +1.4] |
| Apertus cond raw L14 +smooth7 - Qwen alone raw L21 +smooth7 | +3.4 [+2.7, +4.2] | +2.4 [+1.8, +3.1] | +3.1 [+2.4, +3.8] | +3.0 [+2.5, +3.5] |
| Qwen alone raw L21 - Qwen L36 (released) | +11.6 [+9.7, +13.7] | +7.3 [+5.7, +9.1] | +15.1 [+13.4, +17.2] | +11.3 [+10.1, +12.7] |
| Qwen alone raw L21 +smooth7 - Qwen L36 (released) +smooth7 | +12.8 [+10.6, +14.8] | +8.5 [+6.6, +10.4] | +17.6 [+15.6, +19.7] | +12.9 [+11.6, +14.3] |

## Conclusion

- **Most of the gain comes from the layer, not from Apertus.** Going from
  Qwen's last layer to its best middle layer adds 11.4 points (16.4 to 27.8).
  Apertus at its best layer is only 2.2 points above Qwen at its best
  layer (30.0 vs 27.8). Of the 13.6-point gap between "Apertus L14" and "the
  paper's Qwen", about 11 points are the layer and about 2 are the model.
- **Apertus is still a little better, and the difference is consistent.**
  With the same recipe (alone passes, raw, best layer each), Apertus is +2.2
  [+1.8, +2.6]. The interval excludes 0 for every pair. With smoothing, the gap
  shrinks to +1.0 [+0.5, +1.4]. Our best variant (cond L14 +smooth7) is +3.0
  [+2.5, +3.5] above Qwen's best smoothed layer. Part of that comes from the
  conditioned passes, which only a causal LM with both documents in context
  can produce.
- Both best layers were chosen on dev, but the choice is stable on
  dev/train (L21 for Qwen, L14 for Apertus), so selection bias should be
  small and similar for both.
- What this does and does not show: an 8B embedding model and an 8B
  generative LLM end up close once the layer is chosen. The paper's
  last-layer recipe undersells both. It hurts the LLM much more (Apertus last
  layer: 6.9) than the embedding model (Qwen last layer: 16.4), because the
  embedding model is trained to make its last layer semantic. "Apertus beats
  the paper's DiffAlign by 14 points" is true, but most of that would also
  hold for Qwen. The fair claim is: layer choice is the main finding, and
  Apertus is slightly but consistently ahead at the best layer.

## Commands and cost

From `track_2a/`:

```
# GPU (Kaggle 2x T4), kernel amunozo/rsd-qwen-control
python kaggle/run_on_kaggle.py kaggle/02_qwen_control.py --kernel amunozo/rsd-qwen-control
# CPU: all tables above (about 5 minutes)
uv run python -m rsd.qwen_control kaggle/outputs/02_qwen_control/features.npz
```

GPU time:
- one smoke run (2 docs per pair, including the longest) of about 4 minutes;
- one full run of about 16 minutes in session: model load 102 s, then
  835 s wall, of which 630 s were the 1,008 forward passes (606k tokens at
  about 960 tok/s).

Peak memory was 10.3 / 8.2 GiB on the two GPUs. The output is
`features.npz` (148 MB, per-word scores for all 37 layers, raw and centred),
kept locally under `kaggle/outputs/02_qwen_control/` (gitignored).
