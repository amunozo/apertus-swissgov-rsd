"""Supervised scalar head on the Apertus features. CPU only.

Evaluation (dev only; nested CV, dev/train -> dev/val, ablation, bootstrap):

    uv run python -m rsd.head data/features_dev_head.npz \
        [--official cache/scalar_head_preds] [--n-boot 1000]

Fixed-config fit and predict (no search). Default: shared model, feature set
"e", lambda 1e-3, trained on all dev, predicting dev. Dev-only check of the
path: train on dev/train, predict dev/val (should give the dev/val row of
docs/scalar_head_dev_results.md, 38.0 for shared (e)):

    uv run python -m rsd.head fit-predict --train-split train --predict-split val

Final run only (test): add --final --predict-split test --predict-features <test file>.

Feature files: the full features.npz from Kaggle, or the compact export
(python -m rsd.features <features.npz> data/features_dev_head.npz).

Model: logistic regression on per-word features (rsd.features), trained with
soft-label binary cross-entropy (gold in [0, 1] is the target; -1 tokens are
left out of training), L2 penalty on the weights (not the intercept).
Features are standardised with the training rows' statistics only.

Gold labels are used only as training targets inside training folds, and for
scoring. They never enter a feature.

Protocols (Spearman x100, global per pair, macro mean):
1. 5-fold CV over all of dev, grouped by page_en across the three pairs
   (rsd.data.group_folds, deterministic). The L2 strength is chosen inside each
   training fold by an inner 4-fold grouped CV. Out-of-fold predictions are
   pooled and scored once.
2. Official split: fit on dev/train (L2 chosen by inner 4-fold grouped CV on
   dev/train), score dev/val.
The final configuration (feature set, shared vs per-pair) is chosen by the
inner-CV score on dev/train only.
That rule picked feature set (e) per-pair (40.5 vs 40.4 for shared). After
seeing the dev results, the final configuration was switched to shared, lambda
1e-3 (FINAL_PER_PAIR = False). This was a post hoc choice on a tie; see
docs/scalar_head_dev_results.md.

Baseline (no training): DiffAlign cond raw L14 +smooth7 (rsd.unsupervised).
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

from rsd.data import LANGS, Doc, group_folds, load_all
from rsd.features import FEATURE_SETS, Standardiser, build, segments
from rsd.metrics import evaluate, paired_bootstrap
from rsd.predictions import Scores, unit_interval
from rsd.unsupervised import to_scores

LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2)  # L2 strength on the mean loss (1 / (C * n) in sklearn terms)
N_OUTER, N_INNER = 5, 4


# ---------------------------------------------------------------- data glue

class Index:
    """Row bookkeeping: which feature rows belong to which document, and the
    gold vector aligned to the rows (used only as a target and for scoring)."""

    def __init__(self, feats: dict, docs: list[Doc]):
        self.docs = docs
        starts, ends = segments(feats)
        by_doc: dict[str, list[np.ndarray]] = {}
        self.y = np.full(len(feats["doc_id"]), np.nan)
        gold = {d.id: d for d in docs}
        for lo, hi in zip(starts, ends):
            doc_id, side = str(feats["doc_id"][lo]), str(feats["side"][lo])
            d = gold[doc_id]
            labels = d.labels_a if side == "a" else d.labels_b
            assert len(labels) == hi - lo, f"{doc_id} {side}: {hi - lo} rows vs {len(labels)} gold labels"
            self.y[lo:hi] = labels
            by_doc.setdefault(doc_id, []).append(np.arange(lo, hi))
        assert not np.isnan(self.y).any() and set(by_doc) == set(gold)
        self.rows = {k: np.concatenate(v) for k, v in by_doc.items()}
        self.lang = feats["lang"]
        self.blocks = block_list(feats)

    def rows_of(self, doc_idx) -> np.ndarray:
        return np.concatenate([self.rows[self.docs[i].id] for i in doc_idx])

    def scores(self, pred: np.ndarray, doc_idx) -> dict[str, Scores]:
        """Row predictions -> {lang: {doc_id: (a, b)}} mapped to [0, 1], for the
        documents doc_idx only (unit_interval per pair over those documents)."""
        return rows_to_scores(self.blocks, pred, {self.docs[i].id for i in doc_idx})


def block_list(feats: dict) -> list[tuple[int, int, str, str, str]]:
    """(start, end, doc_id, lang, side) of each (doc_id, side) block. No gold."""
    starts, ends = segments(feats)
    return [(int(lo), int(hi), str(feats["doc_id"][lo]), str(feats["lang"][lo]), str(feats["side"][lo]))
            for lo, hi in zip(starts, ends)]


def rows_to_scores(blocks, pred: np.ndarray, keep: set[str]) -> dict[str, Scores]:
    """Row predictions -> {lang: {doc_id: (a, b)}}, mapped to [0, 1] with
    unit_interval per pair over the kept documents. No gold."""
    out = {lang: {} for lang in LANGS}
    for lo, hi, doc_id, lang, side in blocks:
        if doc_id not in keep:
            continue
        a, b = out[lang].get(doc_id, (None, None))
        v = pred[lo:hi].tolist()
        out[lang][doc_id] = (v, b) if side == "a" else (a, v)
    return {lang: unit_interval(s) for lang, s in out.items() if s}


def fast_macro(pred: np.ndarray, y: np.ndarray, lang: np.ndarray, rows: np.ndarray) -> float:
    """Global Spearman macro over pairs on the given rows (gold -1 dropped).
    Same as rsd.metrics.evaluate (side B that is all -1 has no scored rows)."""
    r = rows[y[rows] != -1]
    return float(np.mean([spearmanr(pred[r[lang[r] == l]], y[r[lang[r] == l]]).statistic for l in LANGS]))


# ---------------------------------------------------------------- model

def fit_lr(X: np.ndarray, y: np.ndarray, lam: float, w0=None) -> np.ndarray:
    """Soft-label logistic regression. Returns [w..., b]. X: standardised."""
    n, d = X.shape

    def f(theta):
        w, b = theta[:-1], theta[-1]
        z = X @ w + b
        loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w
        r = (expit(z) - y) / n
        return loss, np.r_[X.T @ r + lam * w, r.sum()]

    theta0 = np.zeros(d + 1) if w0 is None else w0
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


def _groups(lang_rows, per_pair):
    return [(l, lang_rows == l) for l in LANGS] if per_pair else [(None, np.ones(len(lang_rows), bool))]


def fit_path(X, y, lang, per_pair: bool, lams=LAMBDAS) -> dict:
    """Fit one model per lambda (descending, warm-started), shared over pairs or
    one per pair. Returns {lam: {group: (standardiser, theta)}}. Rows with
    y == -1 are left out of training; standardisation uses training rows only."""
    out = {lam: {} for lam in lams}
    train = y != -1
    for l, m in _groups(lang, per_pair):
        m = m & train
        st = Standardiser().fit(X[m])
        Xs, theta = st.transform(X[m]), None
        for lam in sorted(lams, reverse=True):
            theta = fit_lr(Xs, y[m], lam, theta)
            out[lam][l] = (st, theta)
    return out


def predict(models: dict, X, lang, per_pair: bool) -> np.ndarray:
    out = np.empty(len(X))
    for l, m in _groups(lang, per_pair):
        st, theta = models[l]
        out[m] = st.transform(X[m]) @ theta[:-1] + theta[-1]
    return out


def oof_path(X, idx: "Index", doc_idx: list[int], n_splits: int, per_pair: bool, lams=LAMBDAS) -> dict:
    """{lam: out-of-fold predictions (row-aligned, NaN outside doc_idx)} over
    doc_idx, folds grouped by page_en."""
    pred = {lam: np.full(len(X), np.nan) for lam in lams}
    sub = [idx.docs[i] for i in doc_idx]
    for tr, va in group_folds(sub, n_splits):
        rtr, rva = idx.rows_of([doc_idx[i] for i in tr]), idx.rows_of([doc_idx[i] for i in va])
        path = fit_path(X[rtr], idx.y[rtr], idx.lang[rtr], per_pair, lams)
        for lam in lams:
            pred[lam][rva] = predict(path[lam], X[rva], idx.lang[rva], per_pair)
    return pred


def select_lambda(X, idx: "Index", doc_idx: list[int], per_pair: bool, n_splits: int = N_INNER):
    """Grouped CV on doc_idx: {lambda: macro Spearman}; returns (best, scores)."""
    rows = idx.rows_of(doc_idx)
    res = {lam: fast_macro(p, idx.y, idx.lang, rows) for lam, p in oof_path(X, idx, doc_idx, n_splits, per_pair).items()}
    return max(res, key=res.get), res


def train_and_predict(X, idx: "Index", tr: list[int], te: list[int], per_pair: bool):
    """Choose lambda by inner grouped CV on tr, fit on tr, predict te rows.
    Returns (rows of te, predictions, chosen lambda, inner CV scores)."""
    lam, inner = select_lambda(X, idx, tr, per_pair)
    rtr, rte = idx.rows_of(tr), idx.rows_of(te)
    models = fit_path(X[rtr], idx.y[rtr], idx.lang[rtr], per_pair, [lam])[lam]
    return rte, predict(models, X[rte], idx.lang[rte], per_pair), lam, inner


def run_protocols(X, idx: "Index", tr_idx: list[int], va_idx: list[int], per_pair: bool, n_jobs: int = 6):
    """Protocol 1 (nested 5-fold grouped CV over all dev) and protocol 2
    (dev/train -> dev/val), as 6 parallel jobs."""
    from joblib import Parallel, delayed
    jobs = group_folds(idx.docs, N_OUTER) + [(tr_idx, va_idx)]
    res = Parallel(n_jobs=n_jobs)(delayed(train_and_predict)(X, idx, tr, te, per_pair) for tr, te in jobs)
    cv = np.full(len(X), np.nan)
    for rte, p, _, _ in res[:-1]:
        cv[rte] = p
    assert not np.isnan(cv).any()
    val = np.full(len(X), np.nan)
    rte, p, lam_t, inner = res[-1]
    val[rte] = p
    return cv, [r[2] for r in res[:-1]], val, lam_t, inner


# ---------------------------------------------------------------- reporting

def row(name, res):
    return f"| {name} | " + " | ".join(f"{100 * res[k]:.1f}" for k in (*LANGS, "macro")) + " |"


def brow(name, pb):
    return f"| {name} | " + " | ".join(
        f"{100 * pb[k][0]:+.1f} [{100 * pb[k][1]:+.1f}, {100 * pb[k][2]:+.1f}]" for k in (*LANGS, "macro")) + " |"


HEAD = "| System | EN-DE | EN-FR | EN-IT | mean |\n|---|---|---|---|---|"
SET_LABEL = {"b": "(b) L14 cond signal + doc aggregates", "c": "(c) = (b) + log-prob/gain",
             "d": "(d) = (b) + DiffAlign L12-18 alone/cond raw/cent", "e": "(e) all features"}


# ---------------------------------------------------------------- final run

FINAL_SET, FINAL_PER_PAIR, FINAL_LAMBDA = "e", False, 1e-3
DEFAULT_FEATURES = "data/features_dev_head.npz"


def _rows_for(feats: dict, ids: set[str]) -> dict:
    """Keep the rows of the given documents (whole documents, so the
    per-document features are unchanged)."""
    m = np.isin(feats["doc_id"], sorted(ids))
    found = set(np.unique(feats["doc_id"][m]).tolist())
    if found != ids:
        raise ValueError(f"{len(ids - found)} documents have no feature rows")
    return {k: v[m] for k, v in feats.items()}


def fit_predict(train_features, predict_features, out_prefix, feature_set: str = FINAL_SET,
                per_pair: bool = FINAL_PER_PAIR, lam: float = FINAL_LAMBDA, train_split: str = "dev",
                predict_split: str = "dev", final: bool = False) -> dict[str, Scores]:
    """Fit one head with a fixed configuration (no CV, no search) and predict.

    train_split: dev documents to train on ("dev" = all 168 pages, "train" =
    official dev/train). predict_split: "dev", "train", "val", or "test"; test
    needs final=True and is for the single final run only. Writes official-format
    predictions <out_prefix><lang>.jsonl.jsonl, plus <out_prefix>model.npz
    (standardiser stats and weights) and <out_prefix>config.json.
    """
    import json

    from rsd.features import load
    from rsd.predictions import prediction_path, write_predictions

    if train_split not in ("dev", "train", "val"):
        raise ValueError(f"train_split {train_split!r}: training uses dev documents only")
    if predict_split == "test" and not final:
        raise PermissionError("Predicting test is only for the final run: pass final=True (--final).")
    prediction_path(out_prefix, "de")  # refuses prefixes containing "llm"

    train_docs = [d for docs in load_all(train_split).values() for d in docs]
    tf = _rows_for(load(train_features), {d.id for d in train_docs})
    X, names = build(tf, feature_set)
    idx = Index(tf, train_docs)
    models = fit_path(X, idx.y, idx.lang, per_pair, [lam])[lam]

    pred_docs = load_all(predict_split, final=final)
    pf = _rows_for(load(predict_features), {d.id for docs in pred_docs.values() for d in docs})
    Xp, names_p = build(pf, feature_set)
    assert names_p == names
    pred = predict(models, Xp, pf["lang"], per_pair)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite predictions (a needed feature is missing?)")
    scores = rows_to_scores(block_list(pf), pred, {d.id for docs in pred_docs.values() for d in docs})

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        write_predictions(out_prefix, pred_docs[lang], scores[lang], final=final)
    arrays = {"feature_names": np.array(names)}
    for g, (st, theta) in models.items():
        g = g or "shared"
        arrays.update({f"{g}_mean": st.mean, f"{g}_std": st.std, f"{g}_weights": theta[:-1],
                       f"{g}_intercept": theta[-1:]})
    np.savez(f"{out_prefix}model.npz", **arrays)
    config = {"feature_set": feature_set, "per_pair": per_pair, "lambda": lam, "clip": Standardiser().clip,
              "loss": "soft-label BCE, L2 on weights (mean loss), L-BFGS",
              "train_features": str(train_features), "train_split": train_split,
              "n_train_docs": len(train_docs), "predict_features": str(predict_features),
              "predict_split": predict_split, "n_features": len(names)}
    Path(f"{out_prefix}config.json").write_text(json.dumps(config, indent=2) + "\n")
    return scores


def fit_predict_main(argv):
    p = argparse.ArgumentParser(prog="python -m rsd.head fit-predict",
                                description="Fit the head with a fixed config and write predictions.")
    p.add_argument("--train-features", default=DEFAULT_FEATURES)
    p.add_argument("--predict-features", default=DEFAULT_FEATURES)
    p.add_argument("--train-split", default="dev", choices=("dev", "train", "val"))
    p.add_argument("--predict-split", default="dev", choices=("dev", "train", "val", "test"))
    p.add_argument("--out", default="cache/head_final/head_", help="output prefix (must not contain 'llm')")
    p.add_argument("--feature-set", default=FINAL_SET, choices=FEATURE_SETS)
    p.add_argument("--per-pair", action="store_true", help="one model per pair (default: shared)")
    p.add_argument("--lam", type=float, default=FINAL_LAMBDA)
    p.add_argument("--final", action="store_true", help="allow --predict-split test (final run only)")
    a = p.parse_args(argv)
    scores = fit_predict(a.train_features, a.predict_features, a.out, a.feature_set, a.per_pair, a.lam,
                         a.train_split, a.predict_split, a.final)
    print(f"wrote {a.out}{{de,fr,it}}.jsonl.jsonl, {a.out}model.npz, {a.out}config.json")
    if a.predict_split != "test":
        dev = load_all(a.predict_split)
        print(HEAD)
        print(row(f"head {a.feature_set} {'per-pair' if a.per_pair else 'shared'} lambda {a.lam:g}, "
                  f"trained on dev/{a.train_split}, scored on dev/{a.predict_split}", evaluate(dev, scores)))


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fit-predict":
        return fit_predict_main(sys.argv[2:])
    p = argparse.ArgumentParser()
    p.add_argument("features")
    p.add_argument("--official", default="cache/scalar_head_preds",
                   help="directory for the official-format OOF predictions of the chosen config")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--sets", default="".join(FEATURE_SETS))
    a = p.parse_args()
    t0 = time.time()
    log = lambda *m: print(f"[{time.time() - t0:6.0f}s]", *m, flush=True)

    from rsd.features import load
    feats = load(a.features)
    dev = load_all("dev")
    docs = [d for lang in LANGS for d in dev[lang]]
    idx = Index(feats, docs)
    all_idx = list(range(len(docs)))
    is_train = {d.id for v in load_all("train").values() for d in v}
    tr_idx = [i for i, d in enumerate(docs) if d.id in is_train]
    va_idx = [i for i, d in enumerate(docs) if d.id not in is_train]
    dev_val = {lang: [d for d in dev[lang] if d.id not in is_train] for lang in LANGS}

    base = to_scores(feats, feats["da_cond_raw"][:, 14], True)
    base_val = {lang: unit_interval({d.id: base[lang][d.id] for d in dev_val[lang]}) for lang in LANGS}
    out = {"(a) baseline: DiffAlign cond raw L14 +smooth7 (no training)":
           {"cv": evaluate(dev, base), "val": evaluate(dev_val, base_val), "inner_train": None}}
    preds = {}
    for fs in a.sets:
        X, names = build(feats, fs)
        log(f"feature set {fs}: {X.shape[1]} features")
        for per_pair in (False, True):
            name = f"{SET_LABEL[fs]}, {'per-pair' if per_pair else 'shared'}"
            pred_cv, lams, pred_val, lam_t, inner = run_protocols(X, idx, tr_idx, va_idx, per_pair)
            cv_scores = idx.scores(pred_cv, all_idx)
            val_scores = idx.scores(pred_val, va_idx)
            out[name] = {"cv": evaluate(dev, cv_scores), "val": evaluate(dev_val, val_scores),
                         "inner_train": inner[lam_t], "lams": lams, "lam_train": lam_t,
                         "fs": fs, "per_pair": per_pair, "n_feat": X.shape[1]}
            preds[name] = (cv_scores, val_scores)
            log(f"{name}: cv {100 * out[name]['cv']['macro']:.1f}, val {100 * out[name]['val']['macro']:.1f}, "
                f"inner(dev/train) {100 * inner[lam_t]:.1f}, outer lambdas {lams}, dev/train lambda {lam_t}")

    print("\n## Protocol 1: 5-fold grouped CV over all dev (out-of-fold, lambda chosen inside)\n\n" + HEAD)
    for n, r in out.items():
        print(row(n, r["cv"]))
    print("\n## Protocol 2: fit on dev/train, score dev/val\n\n" + HEAD)
    for n, r in out.items():
        print(row(n, r["val"]))
    print("\n## Selection inputs (dev/train only)\n\n| System | n features | inner 4-fold CV on dev/train (mean) "
          "| lambda (dev/train) | lambdas (outer folds) |\n|---|---|---|---|---|")
    for n, r in out.items():
        if r["inner_train"] is not None:
            print(f"| {n} | {r['n_feat']} | {100 * r['inner_train']:.1f} | {r['lam_train']:g} | "
                  + ", ".join(f"{x:g}" for x in r["lams"]) + " |")
    heads = {n: r for n, r in out.items() if r["inner_train"] is not None}
    best = max(heads, key=lambda n: heads[n]["inner_train"])
    print(f"\nChosen on dev/train inner CV: {best}")

    print(f"\n## Paired bootstrap vs baseline ({a.n_boot} page resamples, seed 0)\n\nDifference x100 [95% CI].\n\n"
          "| Comparison | EN-DE | EN-FR | EN-IT | mean |\n|---|---|---|---|---|")
    for n in heads:
        cv_s, val_s = preds[n]
        print(brow(f"CV OOF: {n} - baseline", paired_bootstrap(dev, cv_s, base, a.n_boot, 0)))
    for n in heads:
        cv_s, val_s = preds[n]
        print(brow(f"dev/val: {n} - baseline", paired_bootstrap(dev_val, val_s, base_val, a.n_boot, 0)))

    # Final config: lambda by plain 5-fold grouped CV on all dev, fit on all dev,
    # coefficient magnitudes (features are standardised, so they are comparable).
    r = heads[best]
    X, names = build(feats, r["fs"])
    lam, cvl = select_lambda(X, idx, all_idx, r["per_pair"], N_OUTER)
    print("\n## Final config: lambda on all dev (5-fold grouped CV, not nested)\n\n| lambda | CV mean |\n|---|---|")
    for k, v in cvl.items():
        print(f"| {k:g} | {100 * v:.1f} |")
    models = fit_path(X, idx.y, idx.lang, r["per_pair"], [lam])[lam]
    print(f"\n## Top coefficients (fit on all dev, lambda {lam:g}; standardised features; + = more different)\n")
    for l, (st, theta) in models.items():
        w = theta[:-1]
        order = np.argsort(-np.abs(w))[:15]
        print(f"\n{'shared' if l is None else l}: intercept {theta[-1]:+.2f}\n\n| feature | weight |\n|---|---|")
        for j in order:
            print(f"| {names[j]} | {w[j]:+.3f} |")

    # Official script on the chosen config's OOF predictions.
    from official_eval.run import official_scores
    from rsd.predictions import write_predictions
    prefix = Path(a.official) / "head_oof_"
    cv_s = preds[best][0]
    for lang in LANGS:
        write_predictions(prefix, dev[lang], cv_s[lang])
    off = official_scores(prefix)
    loc = heads[best]["cv"]
    print("\n## Official script on the chosen config's OOF predictions (dev)\n\n| | EN-DE | EN-FR | EN-IT |\n|---|---|---|---|")
    print("| official | " + " | ".join(f"{off[l]:.1f}" for l in LANGS) + " |")
    print("| local | " + " | ".join(f"{100 * loc[l]:.1f}" for l in LANGS) + " |")
    log("done")


if __name__ == "__main__":
    main()
