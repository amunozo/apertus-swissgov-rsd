# Dev-split feature extraction with Apertus 1.5 8B on Kaggle (2x T4, fp16).
# Run from track_2a/ with:
#   python kaggle/run_on_kaggle.py kaggle/01_extract_dev.py --kernel amunozo/rsd-extract-dev
# Uses the dev split only. All the work is in rsd.extract.main (shared with
# kaggle/04_extract_test.py, the final test run); see its docstring for the
# passes and the output files. This run also dumps word-pooled cond-pass
# states at STATE_LAYERS for the states head.
#
# CPU dry run (tiny random model, 2 dev docs per pair), from track_2a/:
#   RSD_DRY_RUN=1 PYTHONPATH=src python kaggle/01_extract_dev.py
# (needs torch and transformers==5.19.0 in that Python).

import os
import subprocess
import sys

if os.environ.get("RSD_DRY_RUN") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0", "accelerate"], check=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from rsd.extract import main

STATE_LAYERS = [8, 16, 24, 32]
main(split="dev", final=False, state_layers=STATE_LAYERS)
