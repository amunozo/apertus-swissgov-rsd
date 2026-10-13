"""Run a local script on Kaggle (2x T4) and download its log and outputs.

Usage (from track_2a/):
    python kaggle/run_on_kaggle.py kaggle/00_smoke_test.py [--timeout SECONDS] [--no-wait]
    python kaggle/run_on_kaggle.py kaggle/01_extract_dev.py --kernel amunozo/rsd-extract-dev \\
        --files '^(features\\.npz|meta\\.json)$'
    python kaggle/run_on_kaggle.py kaggle/01_extract_dev.py --kernel amunozo/rsd-extract-dev \\
        --download-only --files '^meta\\.json$'

The notebook has two cells. The first writes the repo's src/*.py files under
/tmp/src (not /kaggle/working, so they are not part of the output) and puts
/tmp/src on sys.path, so the script can `import rsd`. The second cell is the
script. It is pushed as a new version of --kernel (default KERNEL_ID; a new
id is created by the push, with the slug as its title). The private dataset
amunozo/hf-token-private (hf_token.txt) is always attached. --kernel-source
attaches a previous kernel's output under /kaggle/input/<slug>/.

Pushing starts a run. The log and the files written to /kaggle/working land in
kaggle/outputs/<script name>/. --files is a regex matched (re.search) against
each output file name, so large files can be skipped; the log is always
downloaded.
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
SRC = HERE.parent / "src"


def kaggle(*args):
    out = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    # Drop the CLI's "outdated version" warning so it doesn't clutter status lines.
    out.stdout = "\n".join(line for line in out.stdout.splitlines() if "outdated" not in line)
    return out


def src_cell():
    files = {str(p.relative_to(SRC)): p.read_text() for p in sorted(SRC.rglob("*.py"))
             if "__pycache__" not in p.parts}
    return ("# Repo src/ bundled by run_on_kaggle.py\n"
            "import sys\nfrom pathlib import Path\n"
            f"FILES = {files!r}\n"
            "for name, text in FILES.items():\n"
            "    path = Path('/tmp/src') / name\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text(text)\n"
            "sys.path.insert(0, '/tmp/src')\n")


def push(script, kernel, sources, timeout):
    cell = lambda text: {"cell_type": "code", "metadata": {}, "execution_count": None,
                         "outputs": [], "source": text}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nb = {
            "cells": [cell(src_cell()), cell(script.read_text())],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (tmp / "run.ipynb").write_text(json.dumps(nb))
        meta = {
            "id": kernel,
            "title": kernel.split("/")[1],
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
            "kernel_sources": sources,
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        args = ["kernels", "push", "-p", str(tmp)]
        if timeout:
            args += ["-t", str(timeout)]
        out = kaggle(*args)
        print(out.stdout.strip(), out.stderr.strip(), sep="\n")
        if out.returncode != 0 or "error" in out.stdout.lower():
            sys.exit("push failed")


def wait(kernel):
    while True:
        status = kaggle("kernels", "status", kernel).stdout.strip().lower()
        print(time.strftime("%H:%M:%S"), status, flush=True)
        if any(s in status for s in ("complete", "error", "cancel")):
            return status
        time.sleep(30)


def download(kernel, name, files=None):
    dest = HERE / "outputs" / name
    dest.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", kernel, "-p", str(dest), "--force"]
    if files:
        args += ["--file-pattern", files]
    out = kaggle(*args)
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
    p.add_argument("--kernel", default=KERNEL_ID, help="owner/slug to push to (created if new)")
    p.add_argument("--kernel-source", action="append", default=[], help="attach this kernel's output")
    p.add_argument("--files", help="regex: download only output files whose name matches")
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--download-only", action="store_true", help="skip the push; fetch the latest outputs")
    a = p.parse_args()
    if a.download_only:
        download(a.kernel, a.script.stem, a.files)
        return
    push(a.script, a.kernel, a.kernel_source, a.timeout)
    if a.no_wait:
        return
    time.sleep(20)
    status = wait(a.kernel)
    download(a.kernel, a.script.stem, a.files)
    if "complete" not in status:
        sys.exit(f"run ended with: {status}")


if __name__ == "__main__":
    main()
