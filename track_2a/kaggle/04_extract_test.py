# FINAL RUN ONLY. Test-split feature extraction with Apertus 1.5 8B on Kaggle
# (2x T4, fp16). Run once, from the frozen, tagged commit (docs/final_run.md):
#   python kaggle/run_on_kaggle.py kaggle/04_extract_test.py --kernel amunozo/rsd-extract-test \
#       --files '^(features\.npz|index\.tsv|meta\.json)$'
# Same code as the dev run (rsd.extract.main, called exactly as in
# kaggle/01_extract_dev.py): same model id and commit, frames, fp16 patches,
# pooling and output format. Only differences: split="test", final=True, and
# no hidden-state dumps (only features.npz is needed by the scalar head).
#
# CPU dry run (tiny random model; the split is forced to dev, so this never
# touches test), from track_2a/:
#   RSD_DRY_RUN=1 PYTHONPATH=src python kaggle/04_extract_test.py
# (needs torch and transformers==5.19.0 in that Python).

import os
import subprocess
import sys

if os.environ.get("RSD_DRY_RUN") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0", "accelerate"], check=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from rsd.extract import main

main(split="test", final=True, state_layers=[])
