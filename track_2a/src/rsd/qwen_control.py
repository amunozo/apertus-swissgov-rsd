"""Qwen control: DiffAlign with Qwen3-Embedding-8B at every layer (kaggle/02_qwen_control.py).

    uv run python -m rsd.qwen_control kaggle/outputs/02_qwen_control/features.npz

Prints markdown tables (Spearman x100, dev):
- reproduction check: our last-layer (index 36) scores against the released
  DiffAlign Qwen3-Embedding-8B dev predictions (per-token correlation and score);
- the layer curve (alone passes, raw and mean-centred, with and without +smooth7);
- best layer per family, chosen on all dev and on dev/train (scored on dev/val);
- paired bootstrap of the best Qwen layer against Apertus alone raw L14, raw and
  smoothed, plus against the best Apertus variant (cond raw L14 +smooth7).
If the features cover only some documents (smoke run), everything is computed
on those documents. Dev only.
"""
import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr

from official_eval.fetch import fetch_system
from rsd.data import LANGS, load_all
from rsd.metrics import evaluate, paired_bootstrap
from rsd.predictions import read_predictions
from rsd.unsupervised import HEAD, QWEN8B, bootstrap_row, load_features, row, smooth_scores, subset, to_scores

N_QWEN = 37
LAST = N_QWEN - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("features")
    p.add_argument("--apertus", default="kaggle/outputs/01_extract_dev/features.npz")
    p.add_argument("--resamples", type=int, default=1000)
    a = p.parse_args()
    feats = load_features(a.features)
    ids = set(feats["doc_id"].tolist())
    dev_all = load_all("dev")
    dev = {lang: [d for d in docs if d.id in ids] for lang, docs in dev_all.items()}
    print(f"Documents: {', '.join(f'{lang} {len(dev[lang])}' for lang in LANGS)}\n")
    released = {lang: read_predictions(fetch_system(QWEN8B), lang) for lang in LANGS}
    released = {lang: {d.id: released[lang][d.id] for d in dev[lang]} for lang in LANGS}

    # Reproduction check, every whitespace token (punctuation included).
    print("## Reproduction check: our last layer vs the released predictions\n")
    print("| Pair | tokens | Pearson | Spearman | max abs diff | mean abs diff |\n|---|---|---|---|---|---|")
    raw_last = feats["da_alone_raw"][:, LAST].astype(np.float64)
    ours_raw = {lang: {} for lang in LANGS}  # unmapped scores, to compare values
    for i, (lang, doc_id, side) in enumerate(zip(feats["lang"], feats["doc_id"], feats["side"])):
        ours_raw[str(lang)].setdefault((str(doc_id), str(side)), []).append(raw_last[i])
    for lang in LANGS:
        x, y = [], []
        for d in dev[lang]:
            for k, s in enumerate("ab"):
                x += ours_raw[lang][d.id, s]
                y += released[lang][d.id][k]
        x, y = np.array(x), np.array(y)
        print(f"| EN-{lang.upper()} | {len(x)} | {pearsonr(x, y).statistic:.4f} | {spearmanr(x, y).statistic:.4f} "
              f"| {np.abs(x - y).max():.4f} | {np.abs(x - y).mean():.5f} |")

    print("\n" + HEAD)
    print(row("Released Qwen3-Embedding-8B DiffAlign", evaluate(dev, released)))
    print(row(f"Ours, L{LAST} raw (official recipe, fp16)", evaluate(dev, to_scores(feats, raw_last))))

    split_ids = {s: {d.id for docs in load_all(s).values() for d in docs} for s in ("train", "val")}
    results = {}
    for c in ("raw", "cent"):
        for layer in range(N_QWEN):
            for sm in (False, True):
                scores = to_scores(feats, feats[f"da_alone_{c}"][:, layer], sm)
                r = {"dev": evaluate(dev, scores)}
                for s, sids in split_ids.items():
                    docs_s, scores_s = subset(dev, scores, sids)
                    r[s] = evaluate(docs_s, scores_s) if all(docs_s.values()) else None
                results[c, layer, sm] = r

    print("\n## Layer curve, alone passes (dev mean x100)\n")
    print("| Layer | raw | centred | raw +smooth7 | centred +smooth7 |\n|---|---|---|---|---|")
    for layer in range(N_QWEN):
        vals = [results[c, layer, sm]["dev"]["macro"] for sm in (False, True) for c in ("raw", "cent")]
        print(f"| {layer} | " + " | ".join(f"{100 * x:.1f}" for x in vals) + " |")

    print("\n## Best layer per family\n")
    print("| Family | best L on all dev | dev mean | best L on dev/train | dev/train mean | its dev/val mean |"
          "\n|---|---|---|---|---|---|")
    best = {}
    for c in ("raw", "cent"):
        for sm in (False, True):
            fam = {layer: results[c, layer, sm] for layer in range(N_QWEN)}
            ld = max(fam, key=lambda l: fam[l]["dev"]["macro"])
            best[c, sm] = ld
            if fam[0]["train"] is not None and fam[0]["val"] is not None:
                lt = max(fam, key=lambda l: fam[l]["train"]["macro"])
                tr = f"L{lt} | {100 * fam[lt]['train']['macro']:.1f} | {100 * fam[lt]['val']['macro']:.1f}"
            else:
                tr = "- | - | -"
            print(f"| Qwen alone {c}{' +smooth7' if sm else ''} | L{ld} | {100 * fam[ld]['dev']['macro']:.1f} | {tr} |")

    print("\n## Per pair (dev)\n\n" + HEAD)
    for c, sm in best:
        ld = best[c, sm]
        print(row(f"Qwen alone {c} L{ld}{' +smooth7' if sm else ''}", results[c, ld, sm]["dev"]))
    for c in ("raw", "cent"):
        print(row(f"Qwen alone {c} L{LAST}", results[c, LAST, False]["dev"]))

    # Apertus, same documents.
    ap = load_features(a.apertus)
    keep = np.isin(ap["doc_id"], list(ids))
    ap = {k: v[keep] for k, v in ap.items()}
    ap_alone = to_scores(ap, ap["da_alone_raw"][:, 14])
    ap_alone_sm = to_scores(ap, ap["da_alone_raw"][:, 14], True)
    ap_best = to_scores(ap, ap["da_cond_raw"][:, 14], True)
    print(row("Apertus alone raw L14", evaluate(dev, ap_alone)))
    print(row("Apertus alone raw L14 +smooth7", evaluate(dev, ap_alone_sm)))
    print(row("Apertus cond raw L14 +smooth7", evaluate(dev, ap_best)))

    lr, ls = best["raw", False], best["raw", True]
    q = to_scores(feats, feats["da_alone_raw"][:, lr])
    q_sm = to_scores(feats, feats["da_alone_raw"][:, ls], True)
    print(f"\n## Paired bootstrap (dev, {a.resamples} page resamples, seed 0)\n")
    print("Difference x100 [95% CI].\n\n| Comparison | EN-DE | EN-FR | EN-IT | mean |\n|---|---|---|---|---|")
    print(bootstrap_row(f"Apertus alone raw L14 - Qwen alone raw L{lr}",
                        paired_bootstrap(dev, ap_alone, q, a.resamples, 0)))
    print(bootstrap_row(f"Apertus alone raw L14 +smooth7 - Qwen alone raw L{ls} +smooth7",
                        paired_bootstrap(dev, ap_alone_sm, q_sm, a.resamples, 0)))
    print(bootstrap_row(f"Apertus cond raw L14 +smooth7 - Qwen alone raw L{ls} +smooth7",
                        paired_bootstrap(dev, ap_best, q_sm, a.resamples, 0)))
    print(bootstrap_row(f"Qwen alone raw L{lr} - released Qwen (last layer)",
                        paired_bootstrap(dev, q, released, a.resamples, 0)))
    print(bootstrap_row(f"Qwen alone raw L{ls} +smooth7 - released Qwen +smooth7",
                        paired_bootstrap(dev, q_sm, smooth_scores(released), a.resamples, 0)))


if __name__ == "__main__":
    main()
