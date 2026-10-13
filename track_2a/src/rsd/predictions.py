"""Write and read predictions in the official format.

The official script reads one file per language pair:

    <prefix><lang>.jsonl.jsonl      (lang = de, fr, it)

with one JSON object per document, in the same order as the gold file:

    {"text_a": ..., "text_b": ..., "labels_a": [...], "labels_b": [...], "id": ...}

labels_a / labels_b hold one score per whitespace token of text_a / text_b, in
[0, 1], and must be full length (the official script's padding path crashes
for list-typed predictions). Punctuation tokens are scored too; the metric
ignores them.

Scores for a set of documents: {doc_id: (scores_a, scores_b)}.

Raw model scores (cosine distances, logits, ...) must be mapped into [0, 1]
with unit_interval(), which is monotone. Never clip: clipping creates ties and
ties change Spearman.

The official script drops ids listed in list_to_drop.txt whenever the string
"llm" appears in the prediction path, and then crashes, so such paths are
refused. It also pairs predictions with gold by line position, so documents
are written in gold order.
"""
import json
import math
from pathlib import Path

from rsd.data import Doc, load_docs

Scores = dict[str, tuple[list[float], list[float]]]


def prediction_path(prefix: str | Path, lang: str) -> Path:
    if "llm" in str(Path(prefix).resolve()).lower():
        raise ValueError(f"prefix {str(prefix)!r}: a path containing 'llm' breaks the official script")
    return Path(f"{prefix}{lang}.jsonl.jsonl")


def unit_interval(scores: Scores) -> Scores:
    """Min-max map of all scores of one language pair into [0, 1].

    Monotone, so ranks and ties are unchanged. Raises if all scores are equal.
    """
    values = [x for a, b in scores.values() for x in (*a, *b)]
    lo, hi = min(values), max(values)
    if not hi > lo:
        raise ValueError("all scores are equal")
    f = lambda xs: [(x - lo) / (hi - lo) for x in xs]
    return {k: (f(a), f(b)) for k, (a, b) in scores.items()}


def write_predictions(prefix: str | Path, docs: list[Doc], scores: Scores, final: bool = False) -> Path:
    """Write the predictions for the documents of ONE language pair, in gold order.

    final=True uses the test order (final run only); the default is dev.
    """
    langs = {d.lang for d in docs}
    assert len(langs) == 1, f"one language pair per file, got {langs}"
    lang = langs.pop()
    path = prediction_path(prefix, lang)
    rank = {d.id: i for i, d in enumerate(load_docs(lang, "test" if final else "dev", final=final))}
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for doc in sorted(docs, key=lambda d: rank[d.id]):
        if doc.id not in scores:
            raise KeyError(f"no prediction for {doc.id}")
        sa, sb = scores[doc.id]
        for name, s, tokens in (("a", sa, doc.tokens_a), ("b", sb, doc.tokens_b)):
            if len(s) != len(tokens):
                raise ValueError(f"{doc.id} side {name}: {len(s)} scores for {len(tokens)} tokens")
            if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in s):
                raise ValueError(f"{doc.id} side {name}: scores must be finite and in [0, 1]")
        row = {"text_a": doc.text_a, "text_b": doc.text_b,
               "labels_a": [float(x) for x in sa], "labels_b": [float(x) for x in sb], "id": doc.id}
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def read_predictions(prefix: str | Path, lang: str) -> Scores:
    scores: Scores = {}
    with open(prediction_path(prefix, lang)) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                scores[row["id"]] = (row["labels_a"], row["labels_b"])
    return scores
