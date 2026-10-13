"""Score the released DiffAlign systems of the official repo on DEV and write a table.

    uv run python -m official_eval.reference            # writes docs/reference_dev_scores.md

Official script for the per-pair scores, our local metric as a cross-check.
"""
import sys

from official_eval.fetch import COMMIT, REFERENCE_SYSTEMS, ROOT, fetch_system
from official_eval.run import official_scores
from rsd.data import load_all
from rsd.metrics import evaluate
from rsd.predictions import read_predictions

OUT = ROOT / "docs" / "reference_dev_scores.md"
HEADER = f"""# Reference dev scores of the released systems

Spearman x 100 on the SwissGov-RSD **dev** split (168 documents per pair),
global per pair. Predictions come from the
official repo https://github.com/ZurichNLP/SwissGov-RSD at commit `{COMMIT}`
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
- The paper's tables report the test split, so these numbers differ from it.
- DiffAlign Qwen3-Embedding-8B is the check row: 19.1 / 12.4 / 17.9.

| System | official EN-DE | official EN-FR | official EN-IT | local EN-DE | local EN-FR | local EN-IT | mean |
|---|---|---|---|---|---|---|---|
"""


def fmt(v: float) -> str:
    return f"{v:.1f}"


def main() -> None:
    docs = load_all("dev")
    rows = []
    for label, name in REFERENCE_SYSTEMS:
        prefix = fetch_system(name)
        off = official_scores(prefix)
        loc = evaluate(docs, {lang: read_predictions(prefix, lang) for lang in docs})
        rows.append((loc["macro"], f"| {label} | " + " | ".join(fmt(off[l]) for l in ("de", "fr", "it"))
                     + " | " + " | ".join(fmt(100 * loc[l]) for l in ("de", "fr", "it"))
                     + f" | {fmt(100 * loc['macro'])} |"))
        print(rows[-1][1], file=sys.stderr)
    rows.sort(key=lambda r: -r[0])
    OUT.write_text(HEADER + "\n".join(r[1] for r in rows) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
