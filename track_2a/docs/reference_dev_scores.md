# Reference dev scores of the released systems

Spearman x 100 on the SwissGov-RSD **dev** split (168 documents per pair),
global per pair. Predictions come from the
official repo https://github.com/ZurichNLP/SwissGov-RSD at commit `1807a42100e742ed03d337c54c4b9ea86995f565`
(dev rows only, see `src/official_eval/fetch.py`).

Commands (from the project root):

```
uv run python -m official_eval.fetch --predictions   # download code, dev data, released predictions
uv run python -m official_eval.run "<prefix>"        # official script on one system (dev)
uv run python -m official_eval.reference             # all systems below, writes this file
make eval-reference                                  # the acceptance check + this table
```

Columns: "official" is `scripts/evaluate_predictions_admin.py` (it prints 3
decimals of rho, so one decimal here), "local" is `rsd.metrics`. **mean** is
the mean of the three unrounded local per-pair scores. The official script's
+- values use a bootstrap with `n_resamples=1`, so they are not shown.

Notes:
- The paper scores the whole SwissGov-RSD dataset (dev and test together, paper sections 5.4 and 6), so its numbers are not directly comparable to these dev numbers. Compare with the released predictions rescored on the same split, as here.
- DiffAlign Qwen3-Embedding-8B is the check row: 19.1 / 12.4 / 17.9.

| System | official EN-DE | official EN-FR | official EN-IT | local EN-DE | local EN-FR | local EN-IT | mean |
|---|---|---|---|---|---|---|---|
| DiffAlign Qwen3-Embedding-4B | 20.6 | 11.7 | 20.2 | 20.6 | 11.7 | 20.2 | 17.5 |
| DiffAlign Qwen3-Embedding-8B | 19.1 | 12.4 | 17.9 | 19.1 | 12.4 | 17.9 | 16.4 |
| DiffAlign EuroBERT 610m | 18.6 | 11.8 | 18.0 | 18.6 | 11.8 | 18.0 | 16.1 |
| DiffAlign EuroBERT 2.1B | 13.6 | 8.8 | 16.8 | 13.6 | 8.8 | 16.8 | 13.1 |
| DiffAlign mmBERT-base | 14.7 | 8.8 | 15.3 | 14.7 | 8.8 | 15.3 | 12.9 |
| DiffAlign embeddinggemma-300m | 14.6 | 9.7 | 13.4 | 14.6 | 9.7 | 13.4 | 12.6 |
| DiffAlign bge-m3 | 13.8 | 9.8 | 10.4 | 13.8 | 9.8 | 10.4 | 11.4 |
