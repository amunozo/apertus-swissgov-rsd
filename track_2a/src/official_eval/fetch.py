"""Download the official evaluation code, dev data and released dev predictions.

`python -m official_eval.fetch` downloads everything the dev evaluation needs
(official code, dev gold, HF dev data). Add `--predictions` for the released
reference systems. Everything goes to cache/ (gitignored), laid out like the
official repo so the official script runs unchanged. We fetch instead of
vendoring because the official repo has no LICENSE file for its evaluation/
and scripts/ directories (see src/official_eval/README.md).

Test-split hygiene: test data is only downloaded by fetch_test_gold(), which
the final run calls. The released prediction files in the official repo hold
dev and test rows together. We download them into memory and parse every row
with json.loads to read its id; only dev rows are kept and written to disk.
The test rows are dropped from memory without being used. The only
test-related file on disk is the official list_to_drop.txt (ids only).
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

COMMIT = "1807a42100e742ed03d337c54c4b9ea86995f565"  # ZurichNLP/SwissGov-RSD, master
GH_RAW = f"https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/{COMMIT}/"
HF_COMMIT = "f16aa0536811b8c0cf975fede21eec5996e4b3c1"  # ZurichNLP/SwissGov-RSD dataset
HF_RAW = f"https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/{HF_COMMIT}/"

ROOT = Path(__file__).resolve().parents[2]  # track_2a/
CACHE = ROOT / "cache"
OFFICIAL = CACHE / "official"  # mirrors the official repo layout
LANGS = ("de", "fr", "it")

CODE_FILES = [
    "scripts/evaluate_predictions_admin.py",
    "evaluation/__init__.py",
    "evaluation/utils.py",
    "evaluation/predictions.py",
    "list_to_drop.txt",
]
GOLD_DIR = "data/evaluation/gold_labels/dev"
TEST_GOLD_DIR = "data/evaluation/gold_labels/test"

# Released DiffAlign systems (directory "full" in the official repo), with the
# file name before the language code.
REFERENCE_SYSTEMS = [
    ("DiffAlign Qwen3-Embedding-8B", "DiffAlign(model=Qwen_Qwen3-Embedding-8B, layer=-1_admin_"),
    ("DiffAlign Qwen3-Embedding-4B", "DiffAlign(model=Qwen_Qwen3-Embedding-4B, layer=-1_admin_"),
    ("DiffAlign EuroBERT 610m", "DiffAlign(model=EuroBERT_EuroBERT-610m, layer=-1_admin_"),
    ("DiffAlign EuroBERT 2.1B", "DiffAlign(model=EuroBERT_EuroBERT-2.1B, layer=-1_admin_"),
    ("DiffAlign bge-m3", "DiffAlign(model=BAAI_bge-m3, layer=-1_admin_"),
    ("DiffAlign mmBERT-base", "DiffAlign(model=jhu-clsp_mmBERT-base, layer=-1_admin_"),
    ("DiffAlign embeddinggemma-300m", "DiffAlign(model=google_embeddinggemma-300m, layer=-1_admin_"),
]

# The official code imports rsd.recognizers.utils.DifferenceSample, but the
# upstream rsd package imports torch. We only need this dataclass.
SHIM = '''"""Minimal stand-in for upstream rsd.recognizers.utils (avoids importing torch)."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DifferenceSample:
    tokens_a: Tuple[str, ...]
    tokens_b: Tuple[str, ...]
    labels_a: Tuple[float, ...]
    labels_b: Optional[Tuple[float, ...]]
    annotator_tag: Optional[int] = None
'''


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _gh(path: str) -> bytes:
    return _get(GH_RAW + urllib.parse.quote(path))


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def hf_file(lang: str, split: str) -> Path:
    """HF data file (validation split = official dev; has untokenized texts).
    The only place that downloads from Hugging Face."""
    path = CACHE / "hf" / f"{split}_{lang}.jsonl"
    if not path.exists():
        _save(path, _get(HF_RAW + f"data/{lang}/{split}_{lang}.jsonl"))
    return path


def fetch_official() -> Path:
    """Official code, shim, dev gold (+ official train/val split) and HF dev data."""
    if not (OFFICIAL / ".complete").exists():
        for p in CODE_FILES:
            _save(OFFICIAL / p, _gh(p))
        _save(OFFICIAL / "rsd/__init__.py", b"")
        _save(OFFICIAL / "rsd/recognizers/__init__.py", b"")
        _save(OFFICIAL / "rsd/recognizers/utils.py", SHIM.encode())
        for lang in LANGS:
            for sub in ("", "train/", "val/"):
                p = f"{GOLD_DIR}/{sub}gold_admin_{lang}.jsonl"
                _save(OFFICIAL / p, _gh(p))
        (OFFICIAL / ".complete").write_text(COMMIT + "\n")
    for lang in LANGS:
        hf_file(lang, "dev")
    return OFFICIAL


def fetch_test_gold() -> None:
    """Official test gold. Only for the final run (official_eval.run, final=True)."""
    for lang in LANGS:
        p = f"{TEST_GOLD_DIR}/gold_admin_{lang}.jsonl"
        if not (OFFICIAL / p).exists():
            _save(OFFICIAL / p, _gh(p))


def dev_ids(lang: str) -> list[str]:
    fetch_official()
    with open(OFFICIAL / GOLD_DIR / f"gold_admin_{lang}.jsonl") as f:
        return [json.loads(line)["id"] for line in f]


def fetch_system(name: str) -> str:
    """Download one released DiffAlign system (3 languages), keep dev rows only.

    name: file name without the language suffix, e.g.
          'DiffAlign(model=Qwen_Qwen3-Embedding-8B, layer=-1_admin_'
    Returns the local prefix (absolute path) to pass to the official script.
    """
    out_prefix = OFFICIAL / "predictions" / "encoder_predictions" / name
    for lang in LANGS:
        out = Path(f"{out_prefix}{lang}.jsonl.jsonl")
        if out.exists():
            continue
        keep = set(dev_ids(lang))
        remote = f"data/evaluation/encoder_predictions/full/{name}{lang}.jsonl.jsonl"
        lines = [line for line in _gh(remote).decode().splitlines()
                 if line.strip() and json.loads(line)["id"] in keep]
        if len(lines) != len(keep):
            raise RuntimeError(f"{remote}: found {len(lines)} dev rows, expected {len(keep)}")
        _save(out, ("\n".join(lines) + "\n").encode())
    return str(out_prefix)


if __name__ == "__main__":
    fetch_official()
    print("official code and dev data in", CACHE)
    if "--predictions" in sys.argv:
        for label, name in REFERENCE_SYSTEMS:
            print(label, "->", fetch_system(name))
