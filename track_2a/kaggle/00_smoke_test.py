# Kaggle smoke test: Apertus 1.5 8B on 2x T4 (fp16).
# Run from track_2a/ with: python kaggle/run_on_kaggle.py kaggle/00_smoke_test.py
# Checks: model loads across both GPUs, the longest DEV pair fits in one
# forward pass, hidden states stay finite in fp16, and how long a pass takes.
# Uses the dev split only.

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"], check=True)

import json
import os
import time
import urllib.request
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers
import transformers.integrations.sdpa_attention as sdpa_attention
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# The first run ran out of memory on the longest pair: SDPA fell back to the
# "math" kernel, which builds the full attention matrix. Asking SDPA for GQA
# seems to force that fallback, so expand the KV heads before SDPA instead,
# and disable the math kernel so a fallback fails loudly.
print("torch", torch.__version__, "transformers", transformers.__version__,
      "has use_gqa_in_sdpa:", hasattr(sdpa_attention, "use_gqa_in_sdpa"))
sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# fp16 overflows from layer 2 on (T4s have no bf16). xIELU is quadratic for
# positive inputs and the residual stream carries very large values. Keep the
# weights in fp16 but run the residual stream and the activation in fp32. The
# activation output is scaled down per token before the fp16 down_proj and
# scaled back up after it; down_proj is linear, so this is exact up to rounding.
from transformers.models.apertus import modeling_apertus


def mlp_forward(self, x):
    h = self.act_fn(self.up_proj(x).float())
    scale = torch.exp2(torch.ceil(torch.log2(h.abs().amax(-1, keepdim=True).clamp(min=1.0) / 2**12)).clamp(min=0))
    return self.down_proj((h / scale).half()).float() * scale


def layer_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                  use_cache=False, position_embeddings=None, **kwargs):
    residual = hidden_states.float()
    h, _ = self.self_attn(
        hidden_states=self.attention_layernorm(residual).half(),
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    residual = residual + h.float()
    return residual + self.mlp(self.feedforward_layernorm(residual).half())


ORIGINAL_FORWARDS = (modeling_apertus.ApertusMLP.forward, modeling_apertus.ApertusDecoderLayer.forward)
modeling_apertus.ApertusMLP.forward = mlp_forward
modeling_apertus.ApertusDecoderLayer.forward = layer_forward

# The token comes from the private dataset attached by run_on_kaggle.py.
login(next(Path("/kaggle/input").rglob("hf_token.txt")).read_text().strip())

MODEL = "andreasmartin/apertus-v1.5-8b-text"
LANGS = ["de", "fr", "it"]
NAMES = {"de": "German", "fr": "French", "it": "Italian"}
RAW = "https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/f16aa0536811b8c0cf975fede21eec5996e4b3c1/data/{lang}/dev_{lang}.jsonl"  # same pin as src/official_eval/fetch.py


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
    hidden = hidden.to(lm_head.weight.device).half()
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


run(pair_text(lengths[0][3], lengths[0][1], b_first=False))  # warm-up, not timed

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


# Validate the patch against an unpatched bf16 copy on the CPU, on a short text.
import gc

text = pair_text(lengths[len(lengths) // 2][3], lengths[len(lengths) // 2][1], b_first=False)
ids = tok(text, return_tensors="pt")["input_ids"][:, :128]
with torch.no_grad():
    gpu_states = model.model(input_ids=ids.to(model.device), use_cache=False, output_hidden_states=True).hidden_states
    gpu_states = [h[0].float().cpu() for h in gpu_states]
del model
gc.collect()
torch.cuda.empty_cache()

modeling_apertus.ApertusMLP.forward, modeling_apertus.ApertusDecoderLayer.forward = ORIGINAL_FORWARDS
ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
ref.eval()
with torch.no_grad():
    t = time.time()
    cpu_states = ref.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
    print(f"\nCPU bf16 reference on {ids.shape[1]} tokens took {time.time() - t:.0f}s")
cpu_states = [h[0].float() for h in cpu_states]
for i in [1, 2, 4, 8, 16, 24, 31, 32]:
    g, c = gpu_states[i], cpu_states[i]
    cos = torch.nn.functional.cosine_similarity(g[1:], c[1:], dim=-1)  # skip BOS
    print(f"layer {i:2d}: cos(gpu fp16-patched, cpu bf16) min {cos.min():.4f} mean {cos.mean():.4f}; "
          f"max|h| gpu {g.abs().max():.0f} cpu {c.abs().max():.0f}")
