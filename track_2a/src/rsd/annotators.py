"""Human agreement on EN-DE dev (two annotators) and how our systems score against each.

    uv run python -m rsd.annotators [features.npz] [--n-boot 1000]

The main dev gold file (gold_labels/dev/gold_admin_de.jsonl) has an
`annotator_tag` per document: 90 documents use annotator 2, 78 use annotator 1.
The official dev/train + dev/val files hold annotator 1's labels for all 168
documents. On 89 of the 90 annotator-2 documents the two label sets differ (one
is identical); those 89 documents are the double-annotated set used here.

Metric: global token-level Spearman as in rsd.metrics (side A always, side B
only if its labels are not all -1, positions with -1 dropped, everything
pooled). Positions are kept where BOTH annotators are not -1, so every row
uses the same tokens. Dev only; the test split is never loaded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from official_eval.fetch import GOLD_DIR, OFFICIAL, fetch_official
from rsd.data import load_docs
from rsd.predictions import read_predictions
from rsd.unsupervised import load_features, to_scores


def _read(path) -> dict[str, dict]:
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def double_annotated() -> tuple[dict, dict]:
    """(annotator 1 labels, annotator 2 labels) of the 89 documents: {id: (labels_a, labels_b)}."""
    fetch_official()
    root = OFFICIAL / GOLD_DIR
    main = _read(root / "gold_admin_de.jsonl")
    a1 = {**_read(root / "train" / "gold_admin_de.jsonl"), **_read(root / "val" / "gold_admin_de.jsonl")}
    ids = [i for i, r in main.items() if r["annotator_tag"] == 2
           and (r["labels_a"], r["labels_b"]) != (a1[i]["labels_a"], a1[i]["labels_b"])]
    for i in ids:
        assert main[i]["text_a"] == a1[i]["text_a"] and main[i]["text_b"] == a1[i]["text_b"]
    return ({i: (a1[i]["labels_a"], a1[i]["labels_b"]) for i in ids},
            {i: (main[i]["labels_a"], main[i]["labels_b"]) for i in ids})


def doc_pairs(x: tuple, y: tuple, ref: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Aligned values of x and y on one document; mask = positions where both annotators in `ref` are not -1."""
    xs, ys = [np.asarray(x[0], float)], [np.asarray(y[0], float)]
    masks = [(np.asarray(ref[0][0]) != -1) & (np.asarray(ref[1][0]) != -1)]
    if not all(v == -1 for v in ref[0][1]) and not all(v == -1 for v in ref[1][1]):
        xs.append(np.asarray(x[1], float)); ys.append(np.asarray(y[1], float))
        masks.append((np.asarray(ref[0][1]) != -1) & (np.asarray(ref[1][1]) != -1))
    m = np.concatenate(masks)
    return np.concatenate(xs)[m], np.concatenate(ys)[m]


def rho(chunks: list[tuple[np.ndarray, np.ndarray]]) -> float:
    return float(spearmanr(np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])).statistic)


def boot(chunks, n: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    r = [rho([chunks[i] for i in rng.integers(0, len(chunks), len(chunks))]) for _ in range(n)]
    return tuple(np.percentile(r, [2.5, 97.5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("features", nargs="?", default="kaggle/outputs/01_extract_dev/features.npz")
    p.add_argument("--head", default="cache/scalar_head_preds/head_oof_")
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()

    g1, g2 = double_annotated()
    ids = list(g1)
    ref = {i: (g1[i], g2[i]) for i in ids}
    print(f"EN-DE dev, {len(ids)} double-annotated documents (annotator 2 in the main dev gold)")

    systems = {}
    feats = load_features(a.features)
    base = to_scores(feats, feats["da_cond_raw"][:, 14], smoothing=True)["de"]
    systems["DiffAlign cond raw L14 +smooth7 (baseline)"] = base
    if Path(f"{a.head}de.jsonl.jsonl").exists():
        systems["Scalar head, OOF (per-pair (e))"] = read_predictions(a.head, "de")
    else:
        print("no scalar head predictions found; skipped")

    def table(name, x, y):
        chunks = [doc_pairs(x[i], y[i], ref[i]) for i in ids]
        lo, hi = boot(chunks, a.n_boot)
        print(f"| {name} | {100 * rho(chunks):.1f} [{100 * lo:.1f}, {100 * hi:.1f}] |")
        return chunks

    print("\nSpearman x100 [95% CI, bootstrap over documents], same tokens in every row\n")
    print("| Pair | EN-DE |\n|---|---|")
    chunks = table("annotator 1 vs annotator 2 (human ceiling)", g1, g2)
    print(f"\ntokens compared: {sum(len(c[0]) for c in chunks)}")
    print()
    print("| System | vs annotator 1 | vs annotator 2 |\n|---|---|---|")
    for name, s in systems.items():
        cells = []
        for g in (g1, g2):
            ch = [doc_pairs(s[i], g[i], ref[i]) for i in ids]
            lo, hi = boot(ch, a.n_boot)
            cells.append(f"{100 * rho(ch):.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
        print(f"| {name} | {cells[0]} | {cells[1]} |")

    # Check against the main-gold score on all 168 dev documents (annotator 2 on 89+1 of them).
    docs = load_docs("de")
    print(f"\n(dev docs loaded: {len(docs)}; main-gold labels match annotator 2 on the {len(ids)} documents)")
    for d in docs:
        if d.id in g2:
            assert d.labels_a == list(g2[d.id][0]) and d.labels_b == list(g2[d.id][1])


if __name__ == "__main__":
    main()
