# Kaggle smoke test: Apertus 1.5 8B on 2x T4 (fp16).
# Run from track_2a/ with: python kaggle/run_on_kaggle.py kaggle/00_smoke_test.py
# Checks: model loads across both GPUs, the longest DEV pair fits in one
# forward pass, hidden states stay finite in fp16, and how long a pass takes.
# Uses the dev split only.

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"], check=True)

import json
import time
from pathlib import Path
import urllib.request

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# The token comes from the private dataset attached by run_on_kaggle.py.
login(next(Path("/kaggle/input").rglob("hf_token.txt")).read_text().strip())

MODEL = "andreasmartin/apertus-v1.5-8b-text"
LANGS = ["de", "fr", "it"]
NAMES = {"de": "German", "fr": "French", "it": "Italian"}
RAW = "https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/master/data/{lang}/dev_{lang}.jsonl"


def load_dev(lang):
    with urllib.request.urlopen(RAW.format(lang=lang)) as f:
        return [json.loads(line) for line in f.read().decode().splitlines() if line.strip()]


dev = {lang: load_dev(lang) for lang in LANGS}
print({lang: len(rows) for lang, rows in dev.items()}, "dev docs per pair")

tok = AutoTokenizer.from_pretrained(MODEL)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="auto",
    max_memory={0: "14GiB", 1: "14GiB"},
)
model.eval()
print(f"loaded in {time.time() - t0:.0f}s; device map:", sorted(set(model.hf_device_map.values())))


def pair_text(row, lang, b_first):
    a, b = (t if isinstance(t, str) else " ".join(t) for t in (row["text_a"], row["text_b"]))
    if b_first:
        return f"{NAMES[lang]} document:\n{b}\n\nThe same document in English:\n{a}"
    return f"English document:\n{a}\n\nThe same document in {NAMES[lang]}:\n{b}"


# Longest dev pair by subword count of the concatenated text.
lengths = []
for lang in LANGS:
    for row in dev[lang]:
        n = len(tok(pair_text(row, lang, b_first=False))["input_ids"])
        lengths.append((n, lang, row["id"], row))
lengths.sort(key=lambda x: x[0])
total = sum(n for n, *_ in lengths)
print(f"concatenated pair length (subwords): median {lengths[len(lengths) // 2][0]}, "
      f"max {lengths[-1][0]}, total over dev {total:,}")

# Record max |activation| and finiteness after every decoder layer.
stats = {}


def hook(i):
    def fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        stats[i] = (h.abs().max().item(), torch.isfinite(h).all().item())
    return fn


handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(model.model.layers)]


@torch.no_grad()
def run(text):
    ids = tok(text, return_tensors="pt")["input_ids"].to(model.device)
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    torch.cuda.synchronize()
    t = time.time()
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    # Log-probs of the actual next tokens, in chunks (full logits would be several GB).
    lm_head = model.lm_head
    hidden = hidden.to(lm_head.weight.device)
    targets = ids[0, 1:].to(lm_head.weight.device)
    logps = []
    for s in range(0, hidden.shape[1] - 1, 1024):
        logits = lm_head(hidden[0, s:s + 1024]).float()
        logps.append(torch.log_softmax(logits, -1).gather(-1, targets[s:s + 1024, None])[:, 0])
    logps = torch.cat(logps)
    torch.cuda.synchronize()
    secs = time.time() - t
    peaks = [torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())]
    return ids.shape[1], secs, peaks, logps


for label, (n, lang, doc_id, row) in [("median", lengths[len(lengths) // 2]), ("longest", lengths[-1])]:
    stats.clear()
    n_tok, secs, peaks, logps = run(pair_text(row, lang, b_first=False))
    bad = [i for i, (_, ok) in stats.items() if not ok]
    worst = max(stats.items(), key=lambda kv: kv[1][0])
    print(f"\n[{label}] {doc_id} ({lang}): {n_tok} subwords, {secs:.1f}s, "
          f"{n_tok / secs:.0f} tok/s, peak GiB per GPU {[round(p, 1) for p in peaks]}")
    print(f"  non-finite layers: {bad or 'none'}; max |h| = {worst[1][0]:.0f} at layer {worst[0]} "
          f"(fp16 max is 65504)")
    print(f"  log-probs finite: {torch.isfinite(logps).all().item()}, mean {logps.mean().item():.2f}")

for h in handles:
    h.remove()

secs_per_tok = secs / n_tok
# Four passes per pair: A alone, B alone, B->A, A->B. Roughly 2x the concatenated length in total.
print(f"\nrough dev extraction estimate (4 passes/pair): {2 * total * secs_per_tok / 3600:.1f} GPU-hours")
