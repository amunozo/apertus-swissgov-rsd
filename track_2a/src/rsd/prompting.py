"""The paper's few-shot prompting baseline, run with Apertus on a small dev subset.

The prompt is the official one (prompt_templates/template_{lang}_admin.txt in
ZurichNLP/SwissGov-RSD, fetched at the pinned commit, not vendored), rendered as
scripts/create_test_data_admin.py does: Sentence 1 / Sentence 2 are
json.dumps() of the whitespace-token list. The model answers with JSON
{"sentence1": [[token, score], ...], "sentence2": [...]} on a 0-5 similarity
scale (5 = equivalent, -1 = punctuation).

Parsing uses the official code (evaluation/predictions.py LLMPrediction and
evaluation/utils.py parse_token_labels), run in a subprocess in the official
root, with one change: the official get_json() calls eval() on the model
output; here eval is replaced by ast.literal_eval (a Python-literal parser that
cannot run code). On JSON-like outputs both give the same result. Tokens
the parser cannot match get the official fallback label 5 (-> 0.0 difference);
the score is 1 - label / 5, and -1 stays -1. A failed parse gives empty label
lists, which the official script pads with 0.0; we do the same.

Dev only. Subset selection: select_docs().

    uv run python -m rsd.prompting kaggle/outputs/03_prompting_baseline/generations.jsonl
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from official_eval.fetch import LANGS, OFFICIAL, _gh, _save, fetch_official
from rsd.data import Doc, load_all

MAX_WORDS = 300  # pool: dev docs with at most this many words on both sides together
PER_PAIR = 10
SEED = 0


def template(lang: str) -> str:
    path = OFFICIAL / "prompt_templates" / f"template_{lang}_admin.txt"
    if not path.exists():
        _save(path, _gh(f"prompt_templates/template_{lang}_admin.txt"))
    return path.read_text(encoding="utf-8")


def render(doc: Doc) -> str:
    """The user message, as jinja2.Template(template).render(sentence1=json.dumps(tokens_a), ...).

    The template has only two plain {{ var }} slots, so a string replace gives the
    same text; jinja2 also drops one trailing newline, which we mimic.
    """
    t = template(doc.lang)
    assert t.count("{{ sentence1 }}") == 1 and t.count("{{ sentence2 }}") == 1
    t = t.replace("{{ sentence1 }}", json.dumps(doc.tokens_a)).replace("{{ sentence2 }}", json.dumps(doc.tokens_b))
    return t[:-1] if t.endswith("\n") else t


def list_to_drop() -> set[str]:
    fetch_official()
    return set((OFFICIAL / "list_to_drop.txt").read_text().split())


def select_docs(docs_by_lang: dict[str, list[Doc]], max_words: int = MAX_WORDS, per_pair: int = PER_PAIR,
                seed: int = SEED) -> list[Doc]:
    """Deterministic subset, in processing order.

    Per pair: dev docs not in the official list_to_drop.txt (the paper drops them
    when scoring LLM predictions) with len(tokens_a) + len(tokens_b) <= max_words,
    in dataset order; a seeded permutation (numpy default_rng(seed)) picks
    per_pair of them. The result interleaves the pairs (de, fr, it, de, ...) so
    that a run cut short by the time budget stays balanced.
    """
    drop = list_to_drop()
    picked = {}
    for lang in LANGS:
        pool = [d for d in docs_by_lang[lang] if d.id not in drop and len(d.tokens_a) + len(d.tokens_b) <= max_words]
        order = np.random.default_rng(seed).permutation(len(pool))
        picked[lang] = [pool[i] for i in order[:per_pair]]
    return [picked[lang][i] for i in range(per_pair) for lang in LANGS if i < len(picked[lang])]


_PARSER = r'''
import ast, json, sys
import evaluation.predictions as P
from evaluation.utils import parse_token_labels

def safe_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, RecursionError):
        raise NameError("not a Python literal")  # the official code then tries json.loads

P.eval = safe_eval  # the module's eval() lookup now finds this instead of the builtin
out = []
for r in json.load(sys.stdin):
    pred = P.LLMPrediction(item_id=r["id"], sentence1=r["sentence1"], sentence2=r["sentence2"], api_request={},
                           api_response={"response": {"body": {"choices": [{"message": {"content": r["output"]}}]}}},
                           provider="openai")
    data = pred.get_json()
    s = pred.get_difference_sample()
    info = {"parsed": bool(data), "has_s1": isinstance(data, dict) and "sentence1" in data,
            "has_s2": isinstance(data, dict) and "sentence2" in data}
    for k, toks in (("sentence1", r["sentence1"].split()), ("sentence2", r["sentence2"].split())):
        if isinstance(data, dict) and k in data:
            try:
                info[k + "_entries"] = len(list(data[k]))
            except TypeError:
                info[k + "_entries"] = None
            matched = parse_token_labels(toks, data[k], fallback_label=-99.0)
            info[k + "_unmatched"] = sum(x == -99.0 for x in matched)
    out.append({"id": r["id"], "labels_a": list(s.labels_a), "labels_b": list(s.labels_b), "info": info})
json.dump(out, sys.stdout)
'''


def parse_outputs(rows: list[dict], docs: dict[str, Doc]) -> dict[str, dict]:
    """Official parsing of generations. rows: {"id", "output"}. Returns id -> {labels_a, labels_b, info}."""
    fetch_official()
    inp = [{"id": r["id"], "output": r["output"], "sentence1": docs[r["id"]].text_a,
            "sentence2": docs[r["id"]].text_b} for r in rows]
    res = subprocess.run([sys.executable, "-c", _PARSER], input=json.dumps(inp), cwd=OFFICIAL,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-2000:])
    return {r["id"]: r for r in json.loads(res.stdout)}


def pad(labels: list[float], n: int) -> list[float]:
    """The official script's length handling: pad with 0.0, or truncate."""
    return list(labels[:n]) + [0.0] * max(0, n - len(labels))


def main():
    import argparse

    from rsd.metrics import evaluate
    from rsd.predictions import read_predictions
    from official_eval.fetch import fetch_system
    from rsd.unsupervised import QWEN8B, load_features, smooth_scores, to_scores, variants

    p = argparse.ArgumentParser()
    p.add_argument("generations")
    p.add_argument("--features", default="kaggle/outputs/01_extract_dev/features.npz")
    a = p.parse_args()

    dev = load_all("dev")
    by_id = {d.id: d for docs in dev.values() for d in docs}
    rows = [json.loads(line) for line in open(a.generations) if line.strip()]
    parsed = parse_outputs(rows, by_id)
    ids = [r["id"] for r in rows]
    sub = {lang: [by_id[i] for i in ids if by_id[i].lang == lang] for lang in LANGS}

    prompt_scores = {lang: {} for lang in LANGS}
    print("| doc | words a+b | prompt tok | new tok | sec | stop | parsed | entries a/b vs tokens | unmatched a/b | distinct scores |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    stats = {"docs": 0, "parse_fail": 0, "missing_side": 0, "len_mismatch_sides": 0, "unmatched": 0, "tokens": 0,
             "truncated": 0}
    for r in rows:
        d, pr = by_id[r["id"]], parsed[r["id"]]
        la, lb = pad(pr["labels_a"], len(d.tokens_a)), pad(pr["labels_b"], len(d.tokens_b))
        prompt_scores[d.lang][d.id] = (la, lb)
        info = pr["info"]
        stats["docs"] += 1
        stats["parse_fail"] += not info["parsed"]
        stats["missing_side"] += info["parsed"] and not (info["has_s1"] and info["has_s2"])
        stats["truncated"] += r["stop"] != "eos"
        for k, toks in (("sentence1", d.tokens_a), ("sentence2", d.tokens_b)):
            e = info.get(k + "_entries")
            stats["len_mismatch_sides"] += e is not None and e != len(toks)
            stats["unmatched"] += info.get(k + "_unmatched", len(toks))
            stats["tokens"] += len(toks)
        print(f"| {d.id} | {len(d.tokens_a) + len(d.tokens_b)} | {r['prompt_tokens']} | {r['new_tokens']} | "
              f"{r['seconds']:.0f} | {r['stop']} | {info['parsed']} | "
              f"{info.get('sentence1_entries')}/{info.get('sentence2_entries')} vs {len(d.tokens_a)}/{len(d.tokens_b)} | "
              f"{info.get('sentence1_unmatched', '-')}/{info.get('sentence2_unmatched', '-')} | {len(set(la + lb))} |")
    print("\n" + json.dumps(stats))

    # Ties: share of scored (gold != -1) tokens that carry the most common predicted value.
    vals = np.concatenate([np.r_[s[0], s[1]] for lang in LANGS for s in prompt_scores[lang].values()])
    _, counts = np.unique(vals, return_counts=True)
    print(f"distinct predicted values: {len(counts)}; share of tokens with the most common value: "
          f"{counts.max() / counts.sum():.3f}; value counts: "
          f"{dict(zip(*[x.tolist() for x in np.unique(vals, return_counts=True)]))}")

    feats = load_features(a.features)
    v = variants(feats)
    apertus = {k: to_scores(feats, v[name], sm) for k, (name, sm) in {
        "Apertus DiffAlign cond raw L14 +smooth7": ("DiffAlign cond raw L14", True),
        "Apertus DiffAlign alone raw L14": ("DiffAlign alone raw L14", False),
        "Apertus DiffAlign alone raw L32 (last layer)": ("DiffAlign alone raw L32", False)}.items()}
    qwen = {lang: read_predictions(fetch_system(QWEN8B), lang) for lang in LANGS}
    systems = {"Apertus prompting (paper prompt, greedy)": prompt_scores, **apertus,
               "Qwen3-Embedding-8B DiffAlign (released)": qwen,
               "Qwen3-Embedding-8B DiffAlign (released) +smooth7": smooth_scores(qwen)}
    print(f"\nSubset: {', '.join(f'{lang} {len(sub[lang])}' for lang in LANGS)} docs\n")
    print("| System | EN-DE | EN-FR | EN-IT | mean |\n|---|---|---|---|---|")
    for name, s in systems.items():
        res = evaluate(sub, {lang: {d.id: s[lang][d.id] for d in sub[lang]} for lang in LANGS})
        print(f"| {name} | " + " | ".join(f"{100 * res[k]:.1f}" for k in (*LANGS, "macro")) + " |")


if __name__ == "__main__":
    main()
