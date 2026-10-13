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


def check_order(prefix: str | Path, final: bool = False) -> None:
    """The official script pairs predictions with gold by line position, so the
    prediction ids must equal the gold ids in the same order."""
    gold_dir = OFFICIAL / (TEST_GOLD_DIR if final else GOLD_DIR)
    for lang in LANGS:
        pred, gold = _ids(prediction_path(prefix, lang)), _ids(gold_dir / f"gold_admin_{lang}.jsonl")
        if pred != gold:
            raise ValueError(f"{lang}: prediction ids differ from gold ids (or are in another order)")


def run_official(prefix: str | Path, final: bool = False) -> subprocess.CompletedProcess:
    """Call the official script's main() on dev (or test if final=True).

    cwd must be the official root because the script reads list_to_drop.txt
    from the working directory."""
    fetch_official()
    if final:
        fetch_test_gold()
    prefix = Path(prefix).resolve()
    prediction_path(prefix, "de")  # rejects paths that make the script drop ids
    check_order(prefix, final)
    code = ("import sys; from pathlib import Path; "
            "from scripts.evaluate_predictions_admin import main; "
            "main(Path(sys.argv[1]), False, sys.argv[2], False)")
    return subprocess.run(
        [sys.executable, "-c", code, str(prefix), "test" if final else "dev"],
        cwd=OFFICIAL, capture_output=True, text=True,
    )


def official_scores(prefix: str | Path, final: bool = False) -> dict[str, float]:
    """Spearman x100 per pair ({'de': ..., 'fr': ..., 'it': ...}) from the official script."""
    res = run_official(prefix, final)
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
