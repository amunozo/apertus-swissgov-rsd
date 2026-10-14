"""Static HTML heatmaps of gold vs two unsupervised signals for a few dev documents.

    uv run python -m rsd.heatmap kaggle/outputs/01_extract_dev/features.npz

Writes docs/heatmaps/<doc_id>.html and docs/heatmaps/index.html. Per token:
gold label, DiffAlign cond raw L14 +smooth7, and -gain sum +smooth7.
Model colours are percentiles within the language pair (over all scored dev
tokens, as in the metric), because the metric is rank-based. Dev only.
"""
import argparse
import html
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from rsd.data import LANGS, load_all
from rsd.unsupervised import load_features, to_scores, variants

SIGNALS = {"diff": "DiffAlign cond raw L14 +smooth7", "gain": "-gain sum +smooth7"}
OUT = Path("docs/heatmaps")
MAX_WORDS = 800
NAMES = {"de": "German", "fr": "French", "it": "Italian"}


def scored_mask(doc):
    """Boolean masks (a, b) of tokens the metric scores."""
    a = np.asarray(doc.labels_a) != -1
    b = np.asarray(doc.labels_b) != -1 if not all(x == -1 for x in doc.labels_b) else np.zeros(len(doc.labels_b), bool)
    return a, b


def percentiles(docs, raw):
    """Per pair: percentile (0..1, average ranks) of every token among all scored tokens."""
    pct = {}
    pool = np.concatenate([np.asarray(s)[m] for d in docs for s, m in zip(raw[d.id], scored_mask(d))])
    srt = np.sort(pool)
    n = len(pool)
    for d in docs:
        # average-rank percentile via left/right insertion points
        f = lambda s: (np.searchsorted(srt, s, "left") + np.searchsorted(srt, s, "right")) / (2 * n)
        pct[d.id] = tuple(f(np.asarray(s)) for s in raw[d.id])
    return pct


def doc_rho(d, sig_pct):
    ma, mb = scored_mask(d)
    g = np.concatenate([np.asarray(d.labels_a)[ma], np.asarray(d.labels_b)[mb]])
    p = np.concatenate([sig_pct[0][ma], sig_pct[1][mb]])
    if len(g) < 3 or np.ptp(g) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(spearmanr(p, g).statistic)


def gold_mean(d, side):
    g = np.asarray(d.labels_a if side == "a" else d.labels_b)
    g = g[g != -1]
    return float(g.mean()) if len(g) else float("nan")


def gold_all_mean(d):
    ma, mb = scored_mask(d)
    return float(np.concatenate([np.asarray(d.labels_a)[ma], np.asarray(d.labels_b)[mb]]).mean())


def longest_run(labels):
    best = run = 0
    for x in labels:
        run = run + 1 if x > 0 else 0
        best = max(best, run)
    return best


def select(rows):
    """rows: dicts with id, lang, n, gmean, run, rho_diff. Deterministic choice of 4 docs."""
    ok = [r for r in rows if r["n"] <= MAX_WORDS and not np.isnan(r["rho_diff"])]
    chosen = []

    def pick(cands, key, why):
        used = {c[0]["lang"] for c in chosen}
        cands = [r for r in cands if r["id"] not in {c[0]["id"] for c in chosen}]
        # prefer a pair not yet used, then best key (ties by id)
        cands.sort(key=lambda r: (r["lang"] in used, key(r), r["id"]))
        chosen.append((cands[0], why))

    pick(ok, lambda r: -r["gmean"], "high gold mean")
    pick(ok, lambda r: r["gmean"], "gold mean near zero")
    pick([r for r in ok if r["run"] >= 30], lambda r: -r["run"], "long contiguous gold span")
    pick([r for r in ok if r["gmean"] >= 0.15], lambda r: r["rho_diff"], "DiffAlign does badly despite non-trivial gold")
    return chosen


def colour(v, kind):
    """CSS colour for a value in [0,1]; kind 'gold' or 'model'. Alpha encodes intensity."""
    if v is None:
        return "var(--grey)"
    h = "var(--gold)" if kind == "gold" else "var(--model)"
    return f"color-mix(in srgb, {h} {round(100 * v)}%, var(--empty))"


def side_html(d, side, pcts, raws, label):
    toks = d.tokens_a if side == "a" else d.tokens_b
    gold = d.labels_a if side == "a" else d.labels_b
    i = 0 if side == "a" else 1
    out = [f'<h2>{html.escape(label)}</h2><p class="mean">gold mean (scored tokens): {gold_mean(d, side):.3f}'
           f' &middot; mean percentile: ' + ", ".join(
               f"{n} {np.mean(pcts[k][i][np.asarray(gold) != -1]):.2f}" if (np.asarray(gold) != -1).any() else f"{n} n/a"
               for k, n in (("diff", "DiffAlign"), ("gain", "-gain"))) + '</p><div class="text">']
    for j, (t, g) in enumerate(zip(toks, gold)):
        gc = colour(None if g == -1 else g, "gold")
        bars = [f'<i style="background:{gc}"></i>']
        for k in ("diff", "gain"):
            p = pcts[k][i][j]
            bars.append(f'<i style="background:{"var(--grey)" if g == -1 else colour(p, "model")}"></i>')
        tip = (f"#{j} {t} | gold {g:g} | DiffAlign {raws['diff'][i][j]:.3f} (pct {pcts['diff'][i][j]:.2f})"
               f" | -gain {raws['gain'][i][j]:.3f} (pct {pcts['gain'][i][j]:.2f})")
        out.append(f'<span class="t" title="{html.escape(tip)}"><b>{html.escape(t)}</b>{"".join(bars)}</span>')
    out.append("</div>")
    return "".join(out)


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--gold:#d9480f;--model:#1c7ed6;--empty:#f1f3f5;--grey:#adb5bd;--line:#dee2e6}
@media (prefers-color-scheme:dark){:root{--bg:#16181d;--fg:#e6e6e6;--muted:#9aa0a6;--gold:#ff922b;--model:#4dabf7;--empty:#2a2e36;--grey:#555b66;--line:#333842}}
body{background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.3rem}h2{font-size:1.1rem;margin-top:1.6em}.mean,.legend,.note{color:var(--muted);font-size:.9rem}
table{border-collapse:collapse}td,th{padding:3px 10px;border-bottom:1px solid var(--line);text-align:left}
.text{display:flex;flex-wrap:wrap;gap:6px 3px}.t{display:inline-block;min-width:8px}
.t b{display:block;font-weight:400;font-size:.9rem;white-space:nowrap}.t i{display:block;height:5px;margin-top:1px;border-radius:1px}
.sw{display:inline-block;width:3.5em;height:.8em;vertical-align:middle;border:1px solid var(--line)}
a{color:var(--model)}
"""


def page(d, info, pcts, raws, why):
    rho = info["rho"]
    other = NAMES[d.lang]
    legend = (
        '<p class="legend">Under each word, three bars: <b>1</b> gold, <b>2</b> DiffAlign cond raw L14 +smooth7, '
        '<b>3</b> -gain sum +smooth7. Grey = gold -1 (punctuation, ignored by the metric). '
        'Gold bar: 0 (empty) to 1 (orange). Model bars show the <b>percentile</b> of the score among all scored dev tokens '
        f'of the EN-{d.lang.upper()} pair (not min-max, not within this document), because the metric is rank-based: '
        'blue = high percentile = predicted more different. '
        '<span class="sw" style="background:linear-gradient(90deg,var(--empty),var(--gold))"></span> gold '
        '<span class="sw" style="background:linear-gradient(90deg,var(--empty),var(--model))"></span> percentile. '
        'Hover a word for exact values.</p>')
    ma, mb = scored_mask(d)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{d.id}</title><style>{CSS}</style></head><body>'
            f'<p><a href="index.html">&larr; index</a></p><h1>{d.id} (EN-{d.lang.upper()})</h1>'
            f'<p class="note">Dev document, selected as: {html.escape(why)}. page_en: {html.escape(d.page_en)}</p>'
            f'<table><tr><th></th><th>Spearman vs gold (this doc)</th><th>mean percentile</th></tr>'
            f'<tr><td>DiffAlign cond raw L14 +smooth7</td><td>{rho["diff"]:.3f}</td><td>{info["mp"]["diff"]:.2f}</td></tr>'
            f'<tr><td>-gain sum +smooth7</td><td>{rho["gain"]:.3f}</td><td>{info["mp"]["gain"]:.2f}</td></tr></table>'
            f'<p class="note">Spearman over the scored tokens of both sides together ({int(ma.sum() + mb.sum())} tokens). '
            f'Gold mean: English {gold_mean(d, "a"):.3f}, {other} {gold_mean(d, "b"):.3f}.</p>'
            f'{legend}{side_html(d, "a", pcts, raws, "English")}{side_html(d, "b", pcts, raws, other)}</body></html>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feats = load_features(a.features)
    dev = load_all("dev")  # dev only
    vs = variants(feats)
    scores = {k: to_scores(feats, vs[n.removesuffix(" +smooth7")], True)
              for k, n in (("diff", "DiffAlign cond raw L14"), ("gain", "-gain sum"))}
    pcts, rows = {}, []
    for lang in LANGS:
        pc = {k: percentiles(dev[lang], scores[k][lang]) for k in SIGNALS}
        for d in dev[lang]:
            p = {k: pc[k][d.id] for k in SIGNALS}
            pcts[d.id] = p
            ma, mb = scored_mask(d)
            rows.append({"id": d.id, "lang": lang, "n": max(len(d.tokens_a), len(d.tokens_b)), "gmean": gold_all_mean(d),
                         "run": max(longest_run(d.labels_a), longest_run(d.labels_b)),
                         "rho_diff": doc_rho(d, p["diff"]), "rho_gain": doc_rho(d, p["gain"]),
                         "mp": {k: float(np.concatenate([p[k][0][ma], p[k][1][mb]]).mean()) for k in SIGNALS}})
    byid = {r["id"]: r for r in rows}
    docs = {d.id: d for lang in LANGS for d in dev[lang]}
    index = []
    for r, why in select(rows):
        d = docs[r["id"]]
        raws = {k: tuple(np.asarray(s) for s in scores[k][d.lang][d.id]) for k in SIGNALS}
        info = {"rho": {"diff": r["rho_diff"], "gain": r["rho_gain"]}, "mp": r["mp"]}
        (out / f"{d.id}.html").write_text(page(d, info, pcts[d.id], raws, why))
        index.append((d, r, why))
        print(f"{d.id}: {why}; n={r['n']} gold mean={r['gmean']:.3f} run={r['run']} "
              f"rho diff={r['rho_diff']:.3f} gain={r['rho_gain']:.3f}")
    rowsh = "".join(
        f'<tr><td><a href="{d.id}.html">{d.id}</a></td><td>EN-{d.lang.upper()}</td><td>{len(d.tokens_a)} / {len(d.tokens_b)}</td>'
        f'<td>{r["gmean"]:.3f}</td><td>{html.escape(why)}</td></tr>' for d, r, why in index)
    (out / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Heatmaps</title><style>{CSS}</style></head><body><h1>Dev heatmaps</h1>'
        f'<p class="note">Gold vs DiffAlign cond raw L14 +smooth7 vs -gain sum +smooth7, SwissGov-RSD dev only.</p>'
        f'<table><tr><th>doc</th><th>pair</th><th>words (EN / other)</th><th>gold mean</th><th>why selected</th></tr>{rowsh}</table></body></html>')


if __name__ == "__main__":
    main()
