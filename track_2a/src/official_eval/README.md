# Official evaluation (fetched, not vendored)

Source: https://github.com/ZurichNLP/SwissGov-RSD, branch `master`, pinned to
commit `1807a42100e742ed03d337c54c4b9ea86995f565`. Dev data comes from the
Hugging Face dataset ZurichNLP/SwissGov-RSD at revision
`f16aa0536811b8c0cf975fede21eec5996e4b3c1` (labels CC BY 4.0, text copyright
with the Swiss federal authorities).

## Why the code is not copied here

The repo has a LICENSE file (MIT, University of Zurich) only inside `rsd/`.
`scripts/evaluate_predictions_admin.py`, `evaluation/` and `list_to_drop.txt`
have no license of their own, so we cannot assume they may be redistributed.
`fetch.py` downloads them at the pinned commit into `cache/official/` (gitignored)
instead. The only code we write is a 10-line stand-in for
`rsd.recognizers.utils.DifferenceSample`, because the upstream `rsd` package
imports torch.

## Use

```
uv run python -m official_eval.fetch                  # code + dev data (enough for dev evaluation)
uv run python -m official_eval.run "<prefix>"         # official script, dev only
uv run python -m official_eval.reference              # all released systems
```

`<prefix>` + `de|fr|it` + `.jsonl.jsonl` is the prediction file (see
`rsd/predictions.py`).

## Behaviour of the official script, checked

- File names must be `<prefix><lang>.jsonl.jsonl` .
- Predictions shorter than the gold are padded with `list + tuple`, which raises
  TypeError unless the prefix contains the string "Zurich". Write full-length
  predictions.
- Its bootstrap uses `n_resamples=1`: ignore any +- values. `run.py` never prints them.
- `list_to_drop.txt` is read from the working directory, which `run.py` sets.
  It is applied whenever the string "llm" is in the prediction path, and the
  script then crashes. `rsd.predictions.prediction_path` refuses such paths.
- It pairs predictions with gold by line position. `run.py` first checks that
  prediction ids equal gold ids in the same order.
- If the number of predictions differs from the gold, it reads the full
  (dev + test) gold file. We never download that file, so this path fails
  with FileNotFoundError, as intended.
- `run.py` calls `main()` with `dev`. The `test` split is scored only with
  `final=True` (uses the test gold, downloading it if it is not in `cache/`),
  for the final run.

## Test-split hygiene

Development (without `--test`) only downloads dev gold files. Released prediction files contain
dev and test rows together: `fetch.py` downloads them into memory and parses
every row with `json.loads` to read its id. Only dev rows are kept and
written; the test rows are discarded without being used. The only
test-related file written to disk is the official `list_to_drop.txt` (ids only).
With `fetch --test` (used by the Dockerfile, so that `make run` works offline)
the test texts and the test gold are also saved to `cache/`. This only saves
files; nothing is loaded or scored until the final run (`final=True`), which is
also what `make run` does when `data/features_test_head.npz` exists.
