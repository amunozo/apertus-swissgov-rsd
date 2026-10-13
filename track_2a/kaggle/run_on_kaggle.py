"""Run a local script on Kaggle (2x T4) and download its log and outputs.

Usage:
    python kaggle/run_on_kaggle.py kaggle/00_smoke_test.py [--timeout SECONDS] [--no-wait]

The script is wrapped into a one-cell notebook and pushed as a new version of
KERNEL_ID with the private dataset amunozo/hf-token-private attached (it holds
hf_token.txt). Pushing starts a run. The log and files written to /kaggle/working land in kaggle/outputs/<script name>/.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

KERNEL_ID = "amunozo/notebookfcbf39c514"
HERE = Path(__file__).resolve().parent


def kaggle(*args):
    out = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    # Drop the CLI's "outdated version" warning so it doesn't clutter status lines.
    out.stdout = "\n".join(line for line in out.stdout.splitlines() if "outdated" not in line)
    return out


def push(script, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nb = {
            "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": script.read_text()}],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (tmp / "run.ipynb").write_text(json.dumps(nb))
        meta = {
            "id": KERNEL_ID,
            "title": KERNEL_ID.split("/")[1],
            "code_file": "run.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            # Private dataset holding only hf_token.txt. Kaggle secrets are not
            # available to versions pushed from the CLI.
            "dataset_sources": ["amunozo/hf-token-private"],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        args = ["kernels", "push", "-p", str(tmp)]
        if timeout:
            args += ["-t", str(timeout)]
        out = kaggle(*args)
        print(out.stdout.strip(), out.stderr.strip(), sep="\n")
        if out.returncode != 0 or "error" in out.stdout.lower():
            sys.exit("push failed")


def wait():
    while True:
        status = kaggle("kernels", "status", KERNEL_ID).stdout.strip().lower()
        print(time.strftime("%H:%M:%S"), status, flush=True)
        if any(s in status for s in ("complete", "error", "cancel")):
            return status
        time.sleep(30)


def download(name):
    dest = HERE / "outputs" / name
    dest.mkdir(parents=True, exist_ok=True)
    out = kaggle("kernels", "output", KERNEL_ID, "-p", str(dest), "--force")
    print(out.stdout.strip()[-500:], out.stderr.strip()[-500:], sep="\n")
    for log in dest.glob("*.log"):
        # The log is a JSON list of {"stream_name", "data"} records.
        try:
            text = "".join(r.get("data", "") for r in json.loads(log.read_text()))
        except json.JSONDecodeError:
            text = log.read_text()
        (dest / "log.txt").write_text(text)
        print(f"\n--- last 60 lines of {dest / 'log.txt'} ---")
        print("\n".join(text.splitlines()[-60:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", type=Path)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--no-wait", action="store_true")
    a = p.parse_args()
    push(a.script, a.timeout)
    if a.no_wait:
        return
    time.sleep(20)
    status = wait()
    download(a.script.stem)
    if "complete" not in status:
        sys.exit(f"run ended with: {status}")


if __name__ == "__main__":
    main()
