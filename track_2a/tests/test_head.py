"""Tests for the scalar head (rsd.features, rsd.head). Dev split only, CPU, fast."""
import inspect
from pathlib import Path

import numpy as np
import pytest

import rsd.features as features
from rsd.data import LANGS, group_folds, load_all
from rsd.features import FEATURE_SETS, Standardiser, build
from rsd.head import N_INNER, N_OUTER, fit_path, predict

FEATURES_NPZ = Path(__file__).resolve().parents[1] / "kaggle/outputs/01_extract_dev/features.npz"


def synthetic_feats(seed=0, sizes=((5, 7), (4, 3), (9, 6))):
    """Small stand-in for features.npz: three documents, sides a and b."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("lang", "doc_id", "side", "word")}
    for i, (na, nb) in enumerate(sizes):
        for side, n in (("a", na), ("b", nb)):
            cols["lang"] += ["de"] * n
            cols["doc_id"] += [f"admin_de_{i}"] * n
            cols["side"] += [side] * n
            cols["word"] += list(range(n))
    n = len(cols["word"])
    feats = {k: np.array(v) for k, v in cols.items()}
    feats["word"] = feats["word"].astype(np.int32)
    for k in ("lp_alone_sum", "lp_alone_first", "lp_cond_sum", "lp_cond_first"):
        feats[k] = -rng.exponential(2, n).astype(np.float32)
    for k in ("da_alone_raw", "da_alone_cent", "da_cond_raw", "da_cond_cent"):
        feats[k] = rng.random((n, 33)).astype(np.float32)
    feats["n_sub"] = np.ones(n, np.float32)
    return feats


def test_features_take_no_gold():
    # The builder only sees the features.npz arrays: no Doc, no labels.
    assert list(inspect.signature(build).parameters) == ["feats", "feature_set"]
    src = inspect.getsource(features)
    assert "label" not in src.replace("labels are never an input", "")
    assert "rsd.data" not in src and "load_docs" not in src and "load_all" not in src


@pytest.mark.skipif(not FEATURES_NPZ.exists(), reason="dev features not downloaded")
def test_features_file_has_no_gold():
    with np.load(FEATURES_NPZ) as f:
        assert not any("label" in k or "gold" in k for k in f.files)


@pytest.mark.parametrize("fs", FEATURE_SETS)
def test_features_are_per_document(fs):
    # Changing one document's signals changes only that document's rows.
    feats = synthetic_feats()
    X, names = build(feats, fs)
    assert X.shape == (len(feats["word"]), len(names)) and np.isfinite(X).all()
    other = {k: v.copy() for k, v in feats.items()}
    doc1 = feats["doc_id"] == "admin_de_1"
    for k in ("da_cond_raw", "da_alone_cent", "lp_cond_sum"):
        other[k][doc1] = other[k][doc1] * 0.5 + 0.1
    X2, _ = build(other, fs)
    assert np.array_equal(X[~doc1], X2[~doc1])
    assert not np.array_equal(X[doc1], X2[doc1])


def test_standardiser_uses_fit_rows_only():
    rng = np.random.default_rng(0)
    tr, te = rng.normal(0, 1, (100, 3)), rng.normal(50, 1, (10, 3))
    st = Standardiser().fit(tr)
    assert np.allclose(st.mean, tr.mean(0))
    assert (st.transform(te) == 5.0).all()  # far from training stats: clipped


@pytest.fixture(scope="module")
def dev_docs():
    dev = load_all("dev")
    return [d for lang in LANGS for d in dev[lang]]


def test_outer_and_inner_folds_share_no_page(dev_docs):
    pages = [d.page_en for d in dev_docs]
    outer = group_folds(dev_docs, N_OUTER)
    seen = []
    for tr, va in outer:
        assert not {pages[i] for i in tr} & {pages[i] for i in va}
        # all three pairs of a page sit on the same side of the split
        for p in {pages[i] for i in va}:
            assert sum(pages[i] == p for i in va) == 3
        sub = [dev_docs[i] for i in tr]
        for itr, iva in group_folds(sub, N_INNER):
            assert not {sub[i].page_en for i in itr} & {sub[i].page_en for i in iva}
        seen += va
    assert sorted(seen) == list(range(len(dev_docs)))


def test_fit_path_ignores_unscored_rows_and_learns_direction():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 2))
    y = 1 / (1 + np.exp(-(2 * x[:, 0] - 1)))  # soft labels in (0, 1)
    y_masked = y.copy()
    y_masked[:500] = -1  # ignored rows
    lang = np.array(["de"] * n)
    m1 = fit_path(x, y_masked, lang, per_pair=False, lams=[1e-4])[1e-4]
    m2 = fit_path(x[500:], y[500:], lang[500:], per_pair=False, lams=[1e-4])[1e-4]
    p1, p2 = predict(m1, x, lang, False), predict(m2, x, lang, False)
    assert np.allclose(p1, p2)
    theta = m1[None][1]
    assert theta[0] > 1.5 and abs(theta[1]) < 0.2


def test_smoothing_matches_unsupervised_and_handles_short_blocks():
    from rsd.features import smooth_seg
    from rsd.unsupervised import smooth
    rng = np.random.default_rng(1)
    x = rng.random(60)
    starts, ends = np.array([0, 20, 57]), np.array([20, 57, 60])  # last block shorter than the window
    for k in (3, 7, 15):
        out = smooth_seg(x, starts, ends, k)
        for lo, hi in zip(starts[:2], ends[:2]):
            assert np.allclose(out[lo:hi], smooth(x[lo:hi], k))
        if k >= 7:
            assert np.allclose(out[57:], x[57:].mean())  # window covers the whole short block


COMPACT_NPZ = Path(__file__).resolve().parents[1] / "data/features_dev_head.npz"


def test_fit_predict_test_needs_final(tmp_path):
    from rsd.head import fit_predict
    with pytest.raises(PermissionError):  # raised before anything is loaded
        fit_predict(COMPACT_NPZ, COMPACT_NPZ, tmp_path / "h_", predict_split="test")
    with pytest.raises(ValueError):
        fit_predict(COMPACT_NPZ, COMPACT_NPZ, tmp_path / "h_", train_split="test")
    with pytest.raises(ValueError):
        fit_predict(COMPACT_NPZ, COMPACT_NPZ, tmp_path / "my_llm_", predict_split="val")


def test_compact_export_roundtrip(tmp_path):
    from rsd.features import export_compact, load
    feats = synthetic_feats()
    export_compact(feats, tmp_path / "c.npz")
    back = load(tmp_path / "c.npz")
    for fs in FEATURE_SETS:
        assert np.array_equal(build(feats, fs)[0], build(back, fs)[0])
    assert np.isnan(back["da_cond_raw"][:, 20]).all()  # unstored layers are NaN, never silently zero


@pytest.mark.skipif(not COMPACT_NPZ.exists(), reason="compact dev features not exported")
def test_fit_predict_reproduces_dev_val(tmp_path):
    # Final-run path, dev only: shared (e), lambda 1e-3, fit on dev/train, predict dev/val.
    # docs/scalar_head_dev_results.md reports 38.6 / 31.8 / 43.7, mean 38.0.
    import json

    from official_eval.fetch import LANGS as L
    from rsd.head import fit_predict
    from rsd.metrics import evaluate
    from rsd.predictions import read_predictions
    scores = fit_predict(COMPACT_NPZ, COMPACT_NPZ, tmp_path / "h_", train_split="train", predict_split="val")
    val = load_all("val")
    res = evaluate(val, scores)
    assert 100 * res["macro"] == pytest.approx(38.0, abs=0.06)
    assert evaluate(val, {l: read_predictions(tmp_path / "h_", l) for l in L}) == res  # written files agree
    cfg = json.loads((tmp_path / "h_config.json").read_text())
    assert cfg["feature_set"] == "e" and cfg["per_pair"] is False and cfg["lambda"] == 1e-3
    with np.load(tmp_path / "h_model.npz") as m:
        assert m["shared_weights"].shape == (96,) and len(m["feature_names"]) == 96
