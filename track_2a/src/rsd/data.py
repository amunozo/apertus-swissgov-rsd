"""Load SwissGov-RSD documents.

Dev (168 documents per language pair) loads freely. The test split loads only
when final=True is passed explicitly, for the single final run.

Data come from the Hugging Face dataset ZurichNLP/SwissGov-RSD (pinned
revision, validation split = official dev). The official train/val split of
dev (134 / 34 pages) comes from the official GitHub repo (ids only).
Downloads are cached under cache/ (gitignored); see official_eval/fetch.py.

    uv run python -m rsd.data           # dev summary
    uv run python -m rsd.data --final   # test summary (final run only!)
"""
import json
from dataclasses import dataclass

from sklearn.model_selection import GroupKFold

from official_eval.fetch import GOLD_DIR, LANGS, OFFICIAL, fetch_official, hf_file


@dataclass
class Doc:
    id: str  # e.g. "admin_de_0"
    lang: str  # "de", "fr" or "it"; the other side of the pair is always English
    page_en: str
    tokens_a: list[str]  # English, whitespace tokens
    tokens_b: list[str]  # de / fr / it
    labels_a: list[float]  # gold in [0, 1]; -1 marks tokens the metric ignores
    labels_b: list[float]
    text_a: str  # whitespace-tokenized text (tokens joined by a space)
    text_b: str
    raw_a: str  # untokenized text
    raw_b: str


def _to_doc(row: dict, lang: str) -> Doc:
    tokens_a, tokens_b = row["text_a"].split(), row["text_b"].split()
    assert len(tokens_a) == len(row["labels_a"]) and len(tokens_b) == len(row["labels_b"])
    return Doc(
        id=row["id"], lang=lang, page_en=row["page_en"],
        tokens_a=tokens_a, tokens_b=tokens_b,
        labels_a=[float(x) for x in row["labels_a"]],
        labels_b=[float(x) for x in row["labels_b"]],
        text_a=row["text_a"], text_b=row["text_b"],
        raw_a=row["text_a_untokenized"], raw_b=row["text_b_untokenized"],
    )


def official_split_ids(lang: str, part: str) -> list[str]:
    """Ids of the official dev/train (134 pages) or dev/val (34 pages) split."""
    assert part in ("train", "val")
    fetch_official()
    with open(OFFICIAL / GOLD_DIR / part / f"gold_admin_{lang}.jsonl") as f:
        return [json.loads(line)["id"] for line in f]


def load_docs(lang: str, split: str = "dev", final: bool = False) -> list[Doc]:
    """Documents of one language pair.

    split: "dev" (168 docs), "train" (134, official dev/train), "val" (34,
    official dev/val), or "test" (needs final=True).
    """
    assert lang in LANGS, lang
    if split == "test":
        if not final:
            raise PermissionError("The test split is only for the final run: pass final=True.")
        path = hf_file(lang, "test")
    elif split in ("dev", "train", "val"):
        fetch_official()
        path = hf_file(lang, "dev")
    else:
        raise ValueError(split)
    with open(path) as f:
        docs = [_to_doc(json.loads(line), lang) for line in f if line.strip()]
    if split in ("train", "val"):
        keep = set(official_split_ids(lang, split))
        docs = [d for d in docs if d.id in keep]
    return docs


def load_all(split: str = "dev", final: bool = False) -> dict[str, list[Doc]]:
    return {lang: load_docs(lang, split, final) for lang in LANGS}


def group_folds(docs: list[Doc], n_splits: int = 5) -> list[tuple[list[int], list[int]]]:
    """Cross-validation folds over `docs` (list indices), grouped by page_en.

    No page_en appears in both the train and the validation part of a fold.
    Pass the documents of all three pairs in one list to keep a page together
    across pairs. Deterministic: no shuffling.
    """
    groups = [d.page_en for d in docs]
    gkf = GroupKFold(n_splits=n_splits)
    return [(tr.tolist(), va.tolist()) for tr, va in gkf.split(docs, groups=groups)]


if __name__ == "__main__":
    import sys

    final = "--final" in sys.argv
    split = "test" if final else "dev"
    for lang, docs in load_all(split, final=final).items():
        n_tok = sum(len(d.tokens_a) + len(d.tokens_b) for d in docs)
        print(f"{split} {lang}: {len(docs)} docs, {n_tok} tokens, {len({d.page_en for d in docs})} pages")
