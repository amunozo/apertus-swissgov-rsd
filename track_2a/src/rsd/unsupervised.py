"""Unsupervised scores from the Apertus dev features (kaggle/01_extract_dev.py).

    uv run python -m rsd.unsupervised kaggle/outputs/01_extract_dev/features.npz

Prints markdown tables (Spearman x100 on dev) for:
- DiffAlign at every layer: alone passes (X alone vs Y alone) and cond passes
  (X after Y vs Y after X), raw and mean-centered;
- log-prob signals: -gain (gain = log p(word | Y, header) - log p(word | header)),
  summed over subwords or first subword only, and -log p_cond itself;
- each optionally smoothed with a 7-word moving average within a document side.
Higher score = more different. Scores are mapped to [0, 1] with unit_interval
(monotone, no clipping). Dev only.
"""
import argparse

import numpy as np

from official_eval.fetch import fetch_system
from rsd.data import LANGS, load_all
from rsd.metrics import evaluate, paired_bootstrap
from rsd.predictions import Scores, read_predictions, unit_interval

N_STATES = 33
QWEN8B = "DiffAlign(model=Qwen_Qwen3-Embedding-8B, layer=-1_admin_"


def load_features(path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def smooth(x: np.ndarray, k: int = 7) -> np.ndarray:
    """Centered moving average; windows are cut at the document edges."""
    w = np.ones(k)
    return np.convolve(x, w, "same") / np.convolve(np.ones_like(x), w, "same")


def to_scores(feats: dict, values: np.ndarray, smoothing: bool = False) -> dict[str, Scores]:
    """Per-word values (one per feature row) -> {lang: {doc_id: (scores_a, scores_b)}} in [0, 1]."""
    out = {lang: {} for lang in LANGS}
    key = np.char.add(np.char.add(feats["doc_id"], "|"), feats["side"])
    bounds = np.flatnonzero(key[1:] != key[:-1]) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(key)]):
        lang, doc_id, side = str(feats["lang"][lo]), str(feats["doc_id"][lo]), str(feats["side"][lo])
        assert (feats["word"][lo:hi] == np.arange(hi - lo)).all()
        v = values[lo:hi].astype(np.float64)
        v = smooth(v) if smoothing else v
        a, b = out[lang].get(doc_id, (None, None))
        out[lang][doc_id] = (v.tolist(), b) if side == "a" else (a, v.tolist())
    return {lang: unit_interval(s) for lang, s in out.items()}


def variants(feats: dict) -> dict[str, np.ndarray]:
    """Name -> per-word raw score (higher = more different)."""
    v = {}
    for kind in ("alone", "cond"):
        for c in ("raw", "cent"):
            for layer in range(N_STATES):
                v[f"DiffAlign {kind} {c} L{layer}"] = feats[f"da_{kind}_{c}"][:, layer]
    for agg in ("sum", "first"):
        v[f"-gain {agg}"] = -(feats[f"lp_cond_{agg}"] - feats[f"lp_alone_{agg}"])
        v[f"-logp_cond {agg}"] = -feats[f"lp_cond_{agg}"]
        v[f"-logp_alone {agg}"] = -feats[f"lp_alone_{agg}"]
    # Controls: trivial signals that use no information from the other document.
    v["control: n_subwords"] = feats["n_sub"].astype(np.float64)
    v["control: word position"] = feats["word"].astype(np.float64)
    return v


def smooth_scores(scores_by_lang: dict[str, Scores]) -> dict[str, Scores]:
    return {lang: unit_interval({k: (smooth(np.asarray(a)).tolist(), smooth(np.asarray(b)).tolist())
                                 for k, (a, b) in s.items()}) for lang, s in scores_by_lang.items()}


def bootstrap_row(name, pb):
    return f"| {name} | " + " | ".join(
        f"{100 * pb[k][0]:+.1f} [{100 * pb[k][1]:+.1f}, {100 * pb[k][2]:+.1f}]" for k in (*LANGS, "macro")) + " |"


def subset(docs_by_lang, scores_by_lang, ids):
    docs = {lang: [d for d in docs_by_lang[lang] if d.id in ids] for lang in LANGS}
    return docs, {lang: {d.id: scores_by_lang[lang][d.id] for d in docs[lang]} for lang in LANGS}


def row(name, res):
    return f"| {name} | " + " | ".join(f"{100 * res[k]:.1f}" for k in (*LANGS, "macro")) + " |"


HEAD = "| Variant | EN-DE | EN-FR | EN-IT | mean |\n|---|---|---|---|---|"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("features")
    p.add_argument("--official", help="directory: write the top-3 variants' dev predictions there "
                   "and score them with the official script")
    a = p.parse_args()
    feats = load_features(a.features)
    dev = load_all("dev")
    split_ids = {s: {d.id for docs in load_all(s).values() for d in docs} for s in ("train", "val")}

    results = {}  # (name, smoothed) -> {"dev": res, "train": res, "val": res}
    for name, values in variants(feats).items():
        for sm in (False, True):
            scores = to_scores(feats, values, sm)
            r = {"dev": evaluate(dev, scores)}
            for s, ids in split_ids.items():
                r[s] = evaluate(*subset(dev, scores, ids))
            results[name, sm] = r

    label = lambda n, sm: n + (" +smooth7" if sm else "")
    print("## Log-prob signals (dev)\n\n" + HEAD)
    for (n, sm), r in results.items():
        if not n.startswith(("DiffAlign", "control")):
            print(row(label(n, sm), r["dev"]))

    print("\n## Controls (dev)\n\n" + HEAD)
    for (n, sm), r in results.items():
        if n.startswith("control"):
            print(row(label(n, sm), r["dev"]))

    for kind in ("alone", "cond"):
        print(f"\n## DiffAlign {kind}: layer curve (dev mean x100)\n")
        print("| Layer | raw | centered | raw +smooth7 | centered +smooth7 |\n|---|---|---|---|---|")
        for layer in range(N_STATES):
            vals = [results[f"DiffAlign {kind} {c} L{layer}", sm]["dev"]["macro"]
                    for sm in (False, True) for c in ("raw", "cent")]
            print(f"| {layer} | " + " | ".join(f"{100 * x:.1f}" for x in (vals[0], vals[1], vals[2], vals[3])) + " |")

    print("\n## Best layer per DiffAlign family\n")
    print("| Family | best L on all dev | dev mean | best L on dev/train | dev/train mean | its dev/val mean |"
          "\n|---|---|---|---|---|---|")
    best_rows = []
    for kind in ("alone", "cond"):
        for c in ("raw", "cent"):
            for sm in (False, True):
                fam = {layer: results[f"DiffAlign {kind} {c} L{layer}", sm] for layer in range(N_STATES)}
                ld = max(fam, key=lambda l: fam[l]["dev"]["macro"])
                lt = max(fam, key=lambda l: fam[l]["train"]["macro"])
                print(f"| DiffAlign {kind} {c}{' +smooth7' if sm else ''} | L{ld} | {100 * fam[ld]['dev']['macro']:.1f} "
                      f"| L{lt} | {100 * fam[lt]['train']['macro']:.1f} | {100 * fam[lt]['val']['macro']:.1f} |")
                best_rows.append((label(f"DiffAlign {kind} {c} L{ld}", sm), fam[ld]["dev"]))

    print("\n## Best layer per family, per pair (dev)\n\n" + HEAD)
    for n, r in best_rows:
        print(row(n, r))

    real = {k: r for k, r in results.items() if not k[0].startswith("control")}
    (bn, bsm), br = max(real.items(), key=lambda kv: kv[1]["dev"]["macro"])
    (tn, tsm), tr = max(real.items(), key=lambda kv: kv[1]["train"]["macro"])
    (un, _), ur = max(((k, r) for k, r in real.items() if not k[1]), key=lambda kv: kv[1]["dev"]["macro"])
    print(f"\nBest variant on all dev: {label(bn, bsm)} (dev mean {100 * br['dev']['macro']:.1f})")
    print(f"Best variant on dev/train: {label(tn, tsm)} (dev/train mean {100 * tr['train']['macro']:.1f}, "
          f"dev/val mean {100 * tr['val']['macro']:.1f}, dev mean {100 * tr['dev']['macro']:.1f})")
    print(f"Best unsmoothed variant on all dev: {un} (dev mean {100 * ur['dev']['macro']:.1f})")

    vs = variants(feats)
    best = to_scores(feats, vs[bn], bsm)
    best_raw = to_scores(feats, vs[un], False)
    qwen = {lang: read_predictions(fetch_system(QWEN8B), lang) for lang in LANGS}
    qwen_sm = smooth_scores(qwen)
    print("\n## DiffAlign Qwen3-Embedding-8B, released dev predictions (dev)\n\n" + HEAD)
    print(row("Qwen3-Embedding-8B (released)", evaluate(dev, qwen)))
    print(row("Qwen3-Embedding-8B (released) +smooth7", evaluate(dev, qwen_sm)))

    print("\n## Paired bootstrap vs DiffAlign Qwen3-Embedding-8B (dev, 1000 page resamples)\n")
    print("Difference x100 [95% CI].\n\n| Comparison | EN-DE | EN-FR | EN-IT | mean |\n|---|---|---|---|---|")
    print(bootstrap_row(f"{label(bn, bsm)} - Qwen8B", paired_bootstrap(dev, best, qwen, 1000, 0)))
    print(bootstrap_row(f"{label(bn, bsm)} - Qwen8B +smooth7", paired_bootstrap(dev, best, qwen_sm, 1000, 0)))
    print(bootstrap_row(f"{un} - Qwen8B", paired_bootstrap(dev, best_raw, qwen, 1000, 0)))

    if a.official:
        from pathlib import Path

        from official_eval.run import official_scores
        from rsd.predictions import write_predictions

        print("\n## Official script spot-check (dev)\n\n| Variant | EN-DE | EN-FR | EN-IT | local mean |\n|---|---|---|---|---|")
        top = sorted(real.items(), key=lambda kv: -kv[1]["dev"]["macro"])[:3]
        for k, ((n, sm), r) in enumerate(top):
            scores = best if k == 0 else to_scores(feats, vs[n], sm)
            prefix = Path(a.official) / f"top{k + 1}_"
            for lang in LANGS:
                write_predictions(prefix, dev[lang], scores[lang])
            off = official_scores(prefix)
            print(f"| {label(n, sm)} | " + " | ".join(f"{off[l]:.1f}" for l in LANGS) + f" | {100 * r['dev']['macro']:.1f} |")


if __name__ == "__main__":
    main()
