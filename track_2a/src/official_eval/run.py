"""Run the official evaluation script.

    uv run python -m official_eval.run <prefix>

<prefix> is the path prefix of the prediction files: the official script
appends the language and ".jsonl.jsonl" (see rsd.predictions.write_predictions).
The "+-" values the script can print come from a bootstrap with n_resamples=1
and mean nothing, so this wrapper never asks for them.

Dev is the default. The test split is scored only with final=True, which
downloads the test gold file. That is for the single final run; never use it
during development.
"""
import json
import subprocess
import sys
from pathlib import Path

from official_eval.fetch import (
    GOLD_DIR, LANGS, OFFICIAL, TEST_GOLD_DIR, fetch_official, fetch_test_gold,
)
from rsd.predictions import prediction_path


def _ids(path: Path) -> list[str]:
    with open(path) as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


def _resolve_split(split: str | None) -> str:
    """Name of the gold directory (under gold_labels/) the official script reads.

    EN-DE dev is double-annotated. The main dev gold (= the HF data, and the file
    behind the paper's released dev scores) uses annotator 2's labels for 90 docs
    (annotator_tag == 2); the official dev/train and dev/val files hold annotator
    1's labels for 89 of them (17 in dev/val). FR and IT match exactly. To stay
    consistent with every other number we report, "dev/val" is scored against
    dev_val_from_dev/: the main dev gold rows of the dev/val ids."""
    if split != "dev/val":
        return split or "dev"
    fetch_official()
    base = OFFICIAL / "data/evaluation/gold_labels"
    for lang in LANGS:
        keep = set(_ids(base / "dev/val" / f"gold_admin_{lang}.jsonl"))
        with open(base / "dev" / f"gold_admin_{lang}.jsonl") as f:
            rows = [line for line in f if line.strip() and json.loads(line)["id"] in keep]
        out = base / "dev_val_from_dev" / f"gold_admin_{lang}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(rows))
    return "dev_val_from_dev"


def check_order(prefix: str | Path, final: bool = False, split: str | None = None) -> None:
    """The official script pairs predictions with gold by line position, so the
    prediction ids must equal the gold ids in the same order.

    split: "dev" (default), "dev/val" or "dev/train" (the official sub-splits of
    dev; "dev/val" is read from the dev gold, see _resolve_split), or "test"
    (final only)."""
    gold_dir = OFFICIAL / (TEST_GOLD_DIR if final else f"data/evaluation/gold_labels/{split or 'dev'}")
    for lang in LANGS:
        pred, gold = _ids(prediction_path(prefix, lang)), _ids(gold_dir / f"gold_admin_{lang}.jsonl")
        if pred != gold:
            raise ValueError(f"{lang}: prediction ids differ from gold ids (or are in another order)")


def run_official(prefix: str | Path, final: bool = False, split: str | None = None) -> subprocess.CompletedProcess:
    """Call the official script's main() on dev (or test if final=True).
    split="dev/val" scores predictions that hold only the 34 dev/val pages
    (gold: dev gold restricted to those ids).

    cwd must be the official root because the script reads list_to_drop.txt
    from the working directory."""
    fetch_official()
    if final:
        fetch_test_gold()
    prefix = Path(prefix).resolve()
    prediction_path(prefix, "de")  # rejects paths that make the script drop ids
    if final and split not in (None, "test"):
        raise ValueError("final=True scores the test split")
    split = _resolve_split(split)
    check_order(prefix, final, split)
    code = ("import sys; from pathlib import Path; "
            "from scripts.evaluate_predictions_admin import main; "
            "main(Path(sys.argv[1]), False, sys.argv[2], False)")
    return subprocess.run(
        [sys.executable, "-c", code, str(prefix), "test" if final else (split or "dev")],
        cwd=OFFICIAL, capture_output=True, text=True,
    )


def official_scores(prefix: str | Path, final: bool = False, split: str | None = None) -> dict[str, float]:
    """Spearman x100 per pair ({'de': ..., 'fr': ..., 'it': ...}) from the official script."""
    res = run_official(prefix, final, split)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-2000:])
    lines = res.stdout.splitlines()
    i = lines.index("Spearman:")
    values = lines[i + 2].split("\t")  # lines[i+1] is the header "de fr it"
    return {lang: float(v.replace(",", ".")) * 100 for lang, v in zip(LANGS, values)}


if __name__ == "__main__":
    res = run_official(sys.argv[1], final="--final" in sys.argv)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    sys.exit(res.returncode)
