"""What `make run` executes inside Docker (CPU only, no model, no network for dev).

    python -m rsd.pipeline [--out outputs]

1. Fits the scalar head (shared model, feature set e, lambda 1e-3) with
   `python -m rsd.head fit-predict`.
2. If data/features_test_head.npz exists: trains on all of dev, predicts the
   test split (--final) and scores it with the official script (final=True).
   The Docker build has already fetched the test texts and gold (no network).
3. Otherwise: dev check. Trains on dev/train, predicts dev/val, scores it with
   the official script (split dev/val), and says that test features are missing.

Outputs: <out>/test/ or <out>/dev_val/ (predictions in the official format,
model.npz, config.json) and <out>/scores.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from official_eval.run import official_scores

ROOT = Path(__file__).resolve().parents[2]
DEV_FEATURES = ROOT / "data/features_dev_head.npz"
TEST_FEATURES = ROOT / "data/features_test_head.npz"


def _fit_predict(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "rsd.head", "fit-predict", *args], check=True, cwd=ROOT)


def _report(scores: dict[str, float], title: str) -> dict:
    mean = sum(scores.values()) / len(scores)
    print(f"\n{title} (official script, Spearman x100)")
    print("  " + "  ".join(f"EN-{k.upper()} {v:.1f}" for k, v in scores.items()) + f"  mean {mean:.1f}")
    return {**{k: round(v, 1) for k, v in scores.items()}, "mean": round(mean, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs", help="output directory (gitignored)")
    out = (ROOT / p.parse_args().out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if TEST_FEATURES.exists():
        prefix = out / "test" / "head_"
        _fit_predict("--train-features", str(DEV_FEATURES), "--predict-features", str(TEST_FEATURES),
                     "--train-split", "dev", "--predict-split", "test", "--final", "--out", str(prefix))
        result = {"split": "test", "scores": _report(official_scores(prefix, final=True), "TEST")}
    else:
        prefix = out / "dev_val" / "head_"
        _fit_predict("--train-features", str(DEV_FEATURES), "--predict-features", str(DEV_FEATURES),
                     "--train-split", "train", "--predict-split", "val", "--out", str(prefix))
        result = {"split": "dev/val", "scores": _report(official_scores(prefix, split="dev/val"), "DEV/VAL check")}
        print("\ndata/features_test_head.npz is not present, so the test split was NOT run.")
        print("This is the dev check only: fit on dev/train, predict dev/val.")

    (out / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\npredictions and scores written to {out}")


if __name__ == "__main__":
    main()
