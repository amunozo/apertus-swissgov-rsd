# The paper's few-shot prompting baseline with Apertus 1.5 8B on a small dev subset, on Kaggle (2x T4, fp16).
# Run from track_2a/ with:
#   python kaggle/run_on_kaggle.py kaggle/03_prompting_baseline.py --kernel amunozo/rsd-prompting-baseline
# Uses the dev split only (load_all("dev")); the subset comes from rsd.prompting.select_docs.
#
# Prompt: the official template (rsd.prompting.render), sent as a single user message through
# the model's own chat template (add_generation_prompt=True, thinking off, the template's
# default system prompt kept), as scripts/predict_llama.py does. Greedy decoding.
# max_new_tokens: 1.5 x the token length of a well-formed answer for the document (the paper
# used 4096; a cap is needed to keep runaway outputs inside the GPU budget), at most 4096.
# Documents are processed in select_docs order; no new document starts once GEN_BUDGET
# seconds of generation have been spent.
# Outputs in /kaggle/working:
#   generations.jsonl  one row per document: id, lang, prompt/new token counts, cap, seconds,
#                      stop ("eos" or "length"), raw output text
#   meta.json          model id and commit, chat-template example, timing
# Parsing and scoring run locally: uv run python -m rsd.prompting <generations.jsonl>

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0", "accelerate"], check=True)

import json
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi, login

from rsd import apertus
from rsd.apertus import MODEL
from rsd.data import load_all
from rsd.prompting import render, select_docs

SMOKE = False  # True: the first two documents only
GEN_BUDGET = 80 * 60
OUT = Path("/kaggle/working")

login(next(Path("/kaggle/input").rglob("hf_token.txt")).read_text().strip())
sha = HfApi().model_info(MODEL).sha
t0 = time.time()
tok, model = apertus.load(revision=sha)
# The patched decoder keeps the residual stream (and so the final norm output) in fp32;
# lm_head has fp16 weights.
model.lm_head.register_forward_pre_hook(lambda m, args: (args[0].to(m.weight.dtype),))
print(f"{MODEL}@{sha} loaded in {time.time() - t0:.0f}s; torch {torch.__version__}, "
      f"transformers {transformers.__version__}", flush=True)
eos = model.generation_config.eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])
print("generation_config:", model.generation_config.to_dict(), flush=True)

docs = select_docs(load_all("dev"))
if SMOKE:
    docs = docs[:2]
example = tok.apply_chat_template([{"role": "user", "content": "<USER MESSAGE>"}], tokenize=False,
                                  add_generation_prompt=True)
print("chat template around a user message:\n" + example, flush=True)

timing = {"docs": 0, "prompt_tokens": 0, "new_tokens": 0, "seconds": 0.0}
skipped = []
with open(OUT / "generations.jsonl", "w") as f:
    for doc in docs:
        if timing["seconds"] > GEN_BUDGET:
            skipped.append(doc.id)
            continue
        text = tok.apply_chat_template([{"role": "user", "content": render(doc)}], tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        ideal = json.dumps({"sentence1": [[w, 5] for w in doc.tokens_a], "sentence2": [[w, 5] for w in doc.tokens_b]},
                           ensure_ascii=False)
        cap = min(4096, int(1.5 * len(tok(ideal, add_special_tokens=False)["input_ids"])))
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=cap, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        sec = time.time() - t
        new = out[0, ids.shape[1]:].tolist()
        stop = "eos" if new and new[-1] in eos else "length"
        output = tok.decode(new, skip_special_tokens=True)
        row = {"id": doc.id, "lang": doc.lang, "prompt_tokens": ids.shape[1], "new_tokens": len(new),
               "max_new_tokens": cap, "seconds": sec, "stop": stop, "output": output}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        timing["docs"] += 1
        timing["prompt_tokens"] += ids.shape[1]
        timing["new_tokens"] += len(new)
        timing["seconds"] += sec
        print(f"{doc.id}: {ids.shape[1]} prompt tok, {len(new)}/{cap} new tok, {sec:.0f}s "
              f"({len(new) / sec:.1f} tok/s), stop={stop}; total {timing['seconds'] / 60:.1f} min; "
              f"output starts {output[:120]!r}", flush=True)

meta = {
    "model": MODEL,
    "model_sha": sha,
    "split": "dev",
    "smoke": SMOKE,
    "selected": [d.id for d in docs],
    "skipped_for_budget": skipped,
    "chat_template_example": example,
    "generation": "greedy (do_sample=False), max_new_tokens = min(4096, 1.5 x tokens of a well-formed answer)",
    "eos_token_ids": sorted(eos),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "timing": timing,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(json.dumps({k: meta[k] for k in ("timing", "skipped_for_budget")}, indent=2))
