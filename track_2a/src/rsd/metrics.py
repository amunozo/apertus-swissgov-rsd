"""Local version of the official metric: global token-level Spearman.

Matches scripts/evaluate_predictions_admin.py:
- per language pair, side A of every document, plus side B of every document
  whose gold is not all -1, are concatenated;
- predictions must have the gold length (the official script would pad or
  truncate; we raise instead, so a length bug cannot hide);
- positions where gold == -1 are dropped;
- one Spearman over everything (ties get average ranks);
- the result is averaged over the three pairs (macro average).

drop_ids: documents to leave out of both prediction and gold. The official
script applies list_to_drop.txt to LLM predictions only; for encoder-style
predictions leave it empty (the default).
"""
import numpy as np
from scipy.stats import spearmanr

from rsd.data import Doc
from rsd.predictions import Scores

LANGS = ("de", "fr", "it")


def _fit(pred, n: int) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    assert len(pred) == n, f"prediction has {len(pred)} scores, gold has {n}"
    return pred


def doc_arrays(doc: Doc, pred: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, gold) of one document with gold == -1 positions removed."""
    sides = [(pred[0], doc.labels_a)]
    if not all(x == -1 for x in doc.labels_b):
        sides.append((pred[1], doc.labels_b))
    p = np.concatenate([_fit(s, len(g)) for s, g in sides])
    g = np.concatenate([np.asarray(g, dtype=float) for _, g in sides])
    keep = g != -1
    return p[keep], g[keep]


def _spearman(p: np.ndarray, g: np.ndarray) -> float:
    if np.ptp(p) == 0:
        raise ValueError("constant predictions: Spearman is undefined")
    rho = float(spearmanr(p, g).statistic)
    if np.isnan(rho):
        raise ValueError("Spearman is NaN (NaN in the predictions?)")
    return rho


def pair_arrays(docs: list[Doc], scores: Scores, drop_ids=()) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    drop = set(drop_ids)
    out = {d.page_en: doc_arrays(d, scores[d.id]) for d in docs if d.id not in drop}
    assert len(out) == len([d for d in docs if d.id not in drop]), "page_en must be unique within a pair"
    return out


def spearman_pair(docs: list[Doc], scores: Scores, drop_ids=()) -> float:
    """Global Spearman (not x100) for one language pair."""
    arrays = list(pair_arrays(docs, scores, drop_ids).values())
    return _spearman(np.concatenate([a[0] for a in arrays]), np.concatenate([a[1] for a in arrays]))


def evaluate(docs_by_lang: dict[str, list[Doc]], scores_by_lang: dict[str, Scores], drop_ids=()) -> dict[str, float]:
    """Spearman (not x100) per pair plus 'macro' = mean over pairs."""
    out = {lang: spearman_pair(docs_by_lang[lang], scores_by_lang[lang], drop_ids) for lang in LANGS}
    out["macro"] = float(np.mean([out[lang] for lang in LANGS]))
    return out


def _draws(docs_by_lang: dict[str, list[Doc]], systems: list[dict[str, Scores]],
           n_resamples: int, seed: int, drop_ids) -> np.ndarray:
    """Spearman per resample: array (n_resamples, n_systems, 4) with the pairs
    de, fr, it and the macro average.

    Each resample draws pages (page_en) with replacement and uses the same draw
    for all pairs and all systems. Each page is one document per pair.
    """
    arrays = [{lang: pair_arrays(docs_by_lang[lang], scores[lang], drop_ids) for lang in LANGS}
              for scores in systems]
    pages = sorted(set.intersection(*(set(a[lang]) for a in arrays for lang in LANGS)))
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, len(systems), len(LANGS) + 1))
    for r in range(n_resamples):
        chosen = [pages[i] for i in rng.integers(0, len(pages), len(pages))]
        for k, arr in enumerate(arrays):
            for j, lang in enumerate(LANGS):
                p = np.concatenate([arr[lang][pg][0] for pg in chosen])
                g = np.concatenate([arr[lang][pg][1] for pg in chosen])
                out[r, k, j] = _spearman(p, g)
    out[:, :, -1] = out[:, :, :-1].mean(axis=2)
    return out


def bootstrap_ci(docs_by_lang: dict[str, list[Doc]], scores_by_lang: dict[str, Scores],
                 n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05, drop_ids=()) -> dict[str, tuple[float, float]]:
    """Document-level bootstrap percentile CI for each pair and the macro average."""
    draws = _draws(docs_by_lang, [scores_by_lang], n_resamples, seed, drop_ids)[:, 0]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return {name: (float(lo[j]), float(hi[j])) for j, name in enumerate([*LANGS, "macro"])}


def paired_bootstrap(docs_by_lang: dict[str, list[Doc]], scores_a: dict[str, Scores], scores_b: dict[str, Scores],
                     n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05,
                     drop_ids=()) -> dict[str, tuple[float, float, float]]:
    """Difference A - B between two systems, with a paired bootstrap CI.

    Both systems are scored on the same page draws in every resample. Returns
    {name: (difference on the full set, ci_low, ci_high)} for de, fr, it, macro.
    If the CI excludes 0, the difference is unlikely to be resampling noise.
    """
    draws = _draws(docs_by_lang, [scores_a, scores_b], n_resamples, seed, drop_ids)
    diff = draws[:, 0] - draws[:, 1]
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    ea, eb = evaluate(docs_by_lang, scores_a, drop_ids), evaluate(docs_by_lang, scores_b, drop_ids)
    names = [*LANGS, "macro"]
    return {n: (ea[n] - eb[n], float(lo[j]), float(hi[j])) for j, n in enumerate(names)}
