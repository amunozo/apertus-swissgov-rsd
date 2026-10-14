# Dev ablation: one table for the report

Spearman x 100 on the SwissGov-RSD **dev** split (168 documents per pair),
global per pair (`rsd.metrics`). **mean** is the macro average over the three
pairs. Every number is copied from the doc named in the last column. Nothing
here was re-computed, except the annotator section at the end
(`uv run python -m rsd.annotators`). The test split was not used.

## Main table

| Step | EN-DE | EN-FR | EN-IT | mean | Source |
|---|---|---|---|---|---|
| Paper baseline: DiffAlign Qwen3-Embedding-8B (released) | 19.1 | 12.4 | 17.9 | 16.4 | `reference_dev_scores.md` |
| Paper baseline: DiffAlign Qwen3-Embedding-4B (released) | 20.6 | 11.7 | 20.2 | 17.5 | `reference_dev_scores.md` |
| Paper baseline: DiffAlign EuroBERT 610m (released) | 18.6 | 11.8 | 18.0 | 16.1 | `reference_dev_scores.md` |
| Apertus DiffAlign, last layer (L32, paper recipe, alone, raw) | 5.4 | 5.4 | 9.8 | 6.9 | `unsupervised_dev_results.md` |
| Apertus DiffAlign, L14 alone, raw | 33.0 | 21.4 | 35.7 | 30.0 | `unsupervised_dev_results.md` |
| + conditioning on the other document (cond, raw, L14) | 35.6 | 23.2 | 38.3 | 32.4 | `unsupervised_dev_results.md` |
| + smooth7 (cond raw L14 +smooth7) = **unsupervised baseline** | 39.8 | 26.3 | 43.2 | 36.4 | `unsupervised_dev_results.md` |
| Log-prob gain alone (-gain sum, no DiffAlign) | 22.2 | 13.4 | 23.0 | 19.6 | `unsupervised_dev_results.md` |
| Log-prob gain alone, +smooth7 | 31.4 | 18.1 | 32.0 | 27.2 | `unsupervised_dev_results.md` |
| Control: Qwen3-Embedding-8B, best middle layer (L21, alone, raw) | 30.6 | 19.7 | 33.0 | 27.8 | `qwen_control_dev_results.md` |
| Control: Qwen best middle layer, +smooth7 | 36.4 | 23.9 | 40.1 | 33.5 | `qwen_control_dev_results.md` |
| Scalar head (e), shared, nested 5-fold CV on all dev (out-of-fold) | 43.2 | 29.1 | 47.5 | 39.9 | `scalar_head_dev_results.md` |
| **Scalar head (e), shared, fit on dev/train, scored on dev/val (recommended config)** | 38.6 | 31.8 | 43.7 | 38.0 | `scalar_head_dev_results.md` |
| For comparison: baseline on dev/val (no training) | 33.9 | 29.3 | 40.1 | 34.4 | `scalar_head_dev_results.md` |

"Scalar head (e) shared" is the recommended configuration, with all 96 features
and one model for the three pairs. Both of its rows belong to it. The dev/val
row has only 34 pages per pair, so it is noisy. The "-gain" rows use the log-prob
gain signal on its own. The head also uses this signal as a feature.

## Key paired-bootstrap differences

Paired bootstrap over pages (`rsd.metrics.paired_bootstrap`, 1000 resamples,
seed 0). Difference x 100 [95% CI]. The mean column is given here; per-pair
values are in the source docs.

| Comparison | mean difference | Source |
|---|---|---|
| Apertus cond raw L14 +smooth7 - released Qwen3-Embedding-8B | +20.0 [+18.3, +21.7] | `unsupervised_dev_results.md` |
| Apertus cond raw L14 +smooth7 - released Qwen +smooth7 | +15.9 [+14.4, +17.4] | `unsupervised_dev_results.md` |
| Qwen L21 - Qwen L36 (layer effect only) | +11.3 [+10.1, +12.7] | `qwen_control_dev_results.md` |
| Apertus alone raw L14 - Qwen alone raw L21 (model effect, same recipe) | +2.2 [+1.8, +2.6] | `qwen_control_dev_results.md` |
| same, both +smooth7 | +1.0 [+0.5, +1.4] | `qwen_control_dev_results.md` |
| Apertus cond raw L14 +smooth7 - Qwen L21 +smooth7 | +3.0 [+2.5, +3.5] | `qwen_control_dev_results.md` |
| Scalar head (e) shared - baseline, CV | +3.5 [+3.0, +3.9] | `scalar_head_dev_results.md` |
| Scalar head (e) shared - baseline, dev/val | +3.6 [+2.4, +5.0] | `scalar_head_dev_results.md` |

## What each step adds

- The layer matters most. Moving Apertus DiffAlign from the last layer (6.9) to
  layer 14 (30.0) adds about 23 points. The same move for Qwen adds 11.3
  points (16.4 to 27.8), so most of the gap to the paper's baseline is the layer.
- At the same recipe, Apertus is slightly ahead of Qwen (+2.2 raw, +1.0 with
  smoothing). Conditioning on the other document adds 2.4 points, and
  smoothing adds about 4 (cond: 32.4 to 36.4). Together these give +3.0 over
  Qwen's best smoothed layer.
- The scalar head adds about 3.5 points over the unsupervised baseline, in CV
  and on dev/val. It uses the log-prob gain, other layers (12-18), and
  document-level means as features.
- The prompting baseline with Apertus scores near zero (see below), as GPT-4o
  and Llama do in the paper.

## How to read this

- **CV vs dev/val.** The CV row is out-of-fold over all dev, but the feature set
  and the configuration were chosen on dev, so it is optimistic. The dev/val row
  fits on dev/train (134 pages) and scores on 34 pages. It is cleaner but noisy
  (CI of about +-1.3 on the gain). Neither is a test estimate. Use the dev/val row as
  the honest dev number for the head (38.0).
- **Smoothing gain is label structure.** The gold differences come in spans of
  neighbouring words, so averaging over 7 words helps every system, Qwen
  included. It says nothing about the model.
- **Layer chosen on dev.** The best layer (Apertus L14, Qwen L21) and the decision
  to smooth were picked on all dev. The choice was the same when repeated on
  dev/train, so the selection effect looks small, but the unsupervised 36.4 is
  still not a held-out number.
- The paper's tables report test, so they are not comparable to these dev rows.
  The paper baselines above are the released predictions rescored on dev.

## Prompting baseline (30-document subset, not comparable)

10 documents per pair, short pages only (at most 300 words per pair), so these
scores are not comparable to the full-dev table. The DiffAlign rows are the same
predictions as above, restricted to these 30 documents. Source:
`prompting_baseline_dev_results.md`.

| System | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| **Apertus prompting (paper prompt, greedy)** | 16.3 | -10.2 | 3.6 | 3.2 |
| Apertus DiffAlign alone raw L32 (paper recipe) | 12.3 | 11.7 | -5.2 | 6.3 |
| Qwen3-Embedding-8B DiffAlign (released) | 28.5 | 13.2 | 12.0 | 17.9 |
| Qwen3-Embedding-8B DiffAlign (released) +smooth7 | 34.7 | 18.1 | 13.3 | 22.0 |
| Apertus DiffAlign alone raw L14 | 43.9 | 32.2 | 32.0 | 36.0 |
| Apertus DiffAlign cond raw L14 +smooth7 | 50.5 | 39.5 | 37.6 | 42.5 |

10 of the 30 prompting outputs did not parse and count as all-zero predictions.

## Human agreement on EN-DE dev

EN-DE dev is double-annotated. The main dev gold
(`gold_labels/dev/gold_admin_de.jsonl`, field `annotator_tag`) uses annotator 2
for 90 documents and annotator 1 for 78. The official dev/train and dev/val
files hold annotator 1's labels for all documents. On 89 of the 90
annotator-2 documents the two label sets differ, and these 89 documents are the
set used here (60,300 tokens compared, same tokens in every row; a position is
kept only if neither annotator marks it -1). Metric: global token-level Spearman
as in `rsd.metrics`, with a bootstrap over documents (1000 resamples) for the 95% CI.
Reproduce with `uv run python -m rsd.annotators`.

| Comparison (89 documents) | Spearman x100 [95% CI] |
|---|---|
| annotator 1 vs annotator 2 (human ceiling) | 18.9 [10.6, 28.3] |
| DiffAlign cond raw L14 +smooth7 vs annotator 1 | 20.5 [15.5, 25.6] |
| DiffAlign cond raw L14 +smooth7 vs annotator 2 (the main dev gold) | 38.9 [33.5, 44.2] |
| Scalar head (e), out-of-fold, vs annotator 1 | 21.5 [16.2, 26.8] |
| Scalar head (e), out-of-fold, vs annotator 2 (the main dev gold) | 41.5 [35.9, 47.1] |

The head predictions in `cache/scalar_head_preds/` are the per-pair (e)
configuration (the rule's pick, 40.0 CV mean), not the shared one. Its numbers
on these 89 documents differ slightly from the 42.9 on all 168 EN-DE documents
because the document set is different.

What this shows:
- The two annotators agree only at 18.9 on these documents. Our systems score
  about twice as high against annotator 2 as against annotator 1 (38.9 vs 20.5
  baseline, 41.5 vs 21.5 head). The score depends strongly on which annotator
  made the gold. The 89 documents are the same, so the gap is the annotator.
- Against annotator 1 our systems score at or above the human-human number (20.5
  and 21.5 vs 18.9), and against annotator 2 they score far above it. Both
  comparisons have wide CIs for the human number (10.6 to 28.3), so "above the
  human ceiling" would be an overstatement. A better reading is: on this set the
  gold is too noisy for annotator 1 to be a useful ceiling, and our systems
  look more like annotator 2 than annotator 1.
- The head gains the same amount over the baseline with either annotator (+1.0
  vs annotator 1, +2.6 vs annotator 2). The first is small relative to its CI,
  so the head's gain is clearest on the main gold.

Comparison with the paper (Table 2, token-level Spearman between annotators,
25 overlapping documents per pair): 56.14 for EN-DE, 43.61 for EN-FR, 65.09
for EN-IT. This is not directly comparable to our 18.9:
- different documents (25 overlapping main-phase documents in the paper, 89 dev
  documents here), and we do not know how the paper masks tokens;
- the paper's footnote 5 says the first EN-DE annotation run was below average
  (Spearman 9.4 between annotators) and was repeated with a new annotator, which
  raised the agreement to 56.14. Our 18.9 is between those two numbers. One
  possible explanation is that the official dev/train and dev/val labels come
  from the first run, but nothing in the repo confirms that, so treat it as a
  guess. The EN-DE German subset was also re-annotated in March/April 2026
  (see `notes/uzh/swissgov_rsd_paper.md`).
- So we should not quote 18.9 as the human agreement of the dataset. The
  paper's 56.14 is the figure for the released annotations. Our number is
  agreement between the released annotation (annotator 2) and the other label
  set shipped in the official dev/train and dev/val files.
