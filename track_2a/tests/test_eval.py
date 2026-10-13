"""Tests for the data and evaluation layer. Dev split only."""
import pytest

from official_eval.fetch import fetch_system
from official_eval.run import check_order, official_scores
from rsd.data import LANGS, group_folds, load_all, load_docs, official_split_ids
from rsd.metrics import bootstrap_ci, evaluate, paired_bootstrap
from rsd.predictions import prediction_path, read_predictions, unit_interval, write_predictions

# Paper's released DiffAlign Qwen3-Embedding-8B predictions, dev rows (x100).
EXPECTED = {"de": 19.1, "fr": 12.4, "it": 17.9}


@pytest.fixture(scope="module")
def docs():
    return load_all("dev")


@pytest.fixture(scope="module")
def qwen_prefix():
    return fetch_system("DiffAlign(model=Qwen_Qwen3-Embedding-8B, layer=-1_admin_")


def test_official_script_reproduces_reference(qwen_prefix):
    scores = official_scores(qwen_prefix)
    for lang in LANGS:
        assert scores[lang] == pytest.approx(EXPECTED[lang], abs=0.1)


def test_local_metric_reproduces_reference(docs, qwen_prefix):
    res = evaluate(docs, {lang: read_predictions(qwen_prefix, lang) for lang in LANGS})
    off = official_scores(qwen_prefix)
    for lang in LANGS:
        assert 100 * res[lang] == pytest.approx(EXPECTED[lang], abs=0.1)
        assert 100 * res[lang] == pytest.approx(off[lang], abs=0.1)
    assert res["macro"] == pytest.approx(sum(res[lang] for lang in LANGS) / 3)


def test_bootstrap_ci_contains_point_estimate(docs, qwen_prefix):
    scores = {lang: read_predictions(qwen_prefix, lang) for lang in LANGS}
    point = evaluate(docs, scores)
    ci = bootstrap_ci(docs, scores, n_resamples=30, seed=0)
    assert ci == bootstrap_ci(docs, scores, n_resamples=30, seed=0)  # fixed seed
    for name, (lo, hi) in ci.items():
        assert lo < point[name] < hi


def test_dev_sizes_and_official_split(docs):
    for lang in LANGS:
        assert len(docs[lang]) == 168
        train, val = official_split_ids(lang, "train"), official_split_ids(lang, "val")
        assert (len(train), len(val)) == (134, 34)
        pages = {d.id: d.page_en for d in docs[lang]}
        assert set(train) | set(val) == set(pages) and not set(train) & set(val)
        assert not {pages[i] for i in train} & {pages[i] for i in val}
        assert len(load_docs(lang, "train")) == 134 and len(load_docs(lang, "val")) == 34


def test_test_split_is_gated():
    with pytest.raises(PermissionError):
        load_docs("de", "test")


def test_prediction_roundtrip(tmp_path, docs):
    de = docs["de"][:3]
    scores = {d.id: ([i / 10 % 1 for i in range(len(d.tokens_a))], [0.5] * len(d.tokens_b)) for d in de}
    path = write_predictions(tmp_path / "run_", de, scores)
    assert path.name == "run_de.jsonl.jsonl"
    back = read_predictions(tmp_path / "run_", "de")
    assert list(back) == [d.id for d in de]
    for d in de:
        assert back[d.id][0] == scores[d.id][0] and back[d.id][1] == scores[d.id][1]


def test_prediction_writer_rejects_bad_input(tmp_path, docs):
    d = docs["de"][0]
    ok_b = [0.0] * len(d.tokens_b)
    with pytest.raises(ValueError):  # too short
        write_predictions(tmp_path / "x_", [d], {d.id: ([0.0] * (len(d.tokens_a) - 1), ok_b)})
    with pytest.raises(ValueError):  # out of range
        write_predictions(tmp_path / "x_", [d], {d.id: ([2.0] * len(d.tokens_a), ok_b)})
    with pytest.raises(KeyError):  # missing document
        write_predictions(tmp_path / "x_", [d], {})


def test_group_folds_share_no_page(docs):
    both = docs["de"] + docs["fr"] + docs["it"]  # same pages in every pair
    folds = group_folds(both, n_splits=5)
    assert len(folds) == 5
    seen_val = []
    for tr, va in folds:
        assert not {both[i].page_en for i in tr} & {both[i].page_en for i in va}
        seen_val += va
    assert sorted(seen_val) == list(range(len(both)))  # every doc validated once


def test_writer_sorts_into_gold_order(tmp_path, docs):
    de = docs["de"][:3]
    scores = {d.id: ([0.1] * len(d.tokens_a), [0.2] * len(d.tokens_b)) for d in de}
    write_predictions(tmp_path / "r_", de[::-1], scores)
    assert list(read_predictions(tmp_path / "r_", "de")) == [d.id for d in de]


def test_prefix_with_llm_is_rejected():
    with pytest.raises(ValueError):
        prediction_path("/somewhere/my_llm_run_", "de")


def test_order_assert_on_dev(tmp_path, docs, qwen_prefix):
    check_order(qwen_prefix)  # released file is in gold order
    for lang in LANGS:
        rows = prediction_path(qwen_prefix, lang).read_text().splitlines()
        rows[0], rows[1] = rows[1], rows[0]  # swap two documents
        prediction_path(tmp_path / "swap_", lang).write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        check_order(tmp_path / "swap_")


def test_unit_interval_is_monotone_without_ties_change():
    raw = {"a": ([-3.0, 0.0, 5.0], [5.0, 1.0]), "b": ([2.0], [-3.0, 0.5])}
    out = unit_interval(raw)
    flat_raw = [x for a, b in raw.values() for x in (*a, *b)]
    flat = [x for a, b in out.values() for x in (*a, *b)]
    assert min(flat) == 0.0 and max(flat) == 1.0
    assert all((flat_raw[i] < flat_raw[j]) == (flat[i] < flat[j]) and (flat_raw[i] == flat_raw[j]) == (flat[i] == flat[j])
               for i in range(len(flat)) for j in range(len(flat)))


def test_metric_rejects_bad_predictions(docs):
    d = docs["de"][:2]
    n = lambda doc: (len(doc.tokens_a), len(doc.tokens_b))
    const = {x.id: ([0.5] * n(x)[0], [0.5] * n(x)[1]) for x in d}
    with pytest.raises(ValueError):
        evaluate({"de": d, "fr": d, "it": d}, {"de": const, "fr": const, "it": const})
    short = {x.id: ([0.1] * (n(x)[0] - 1), [0.1] * n(x)[1]) for x in d}
    with pytest.raises(AssertionError):
        evaluate({"de": d, "fr": d, "it": d}, {"de": short, "fr": short, "it": short})


def test_paired_bootstrap(docs, qwen_prefix):
    a = {lang: read_predictions(qwen_prefix, lang) for lang in LANGS}
    same = paired_bootstrap(docs, a, a, n_resamples=10)
    assert all(d == 0 and lo == 0 and hi == 0 for d, lo, hi in same.values())  # system vs itself
    # reversed ranks of the same system: clearly worse, CI below 0
    b = {lang: {k: ([1 - x for x in sa], [1 - x for x in sb]) for k, (sa, sb) in a[lang].items()} for lang in LANGS}
    res = paired_bootstrap(docs, a, b, n_resamples=20)
    assert res["macro"][0] > 0 and res["macro"][1] > 0
