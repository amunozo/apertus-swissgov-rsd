# Qwen control: DiffAlign with Qwen/Qwen3-Embedding-8B at every hidden layer, on Kaggle (2x T4, fp16).
# Run from track_2a/ with:
#   python kaggle/run_on_kaggle.py kaggle/02_qwen_control.py --kernel amunozo/rsd-qwen-control
# Uses the dev split only (load_docs(lang, "dev")).
#
# Question: is the gain of Apertus L14 DiffAlign over the released Qwen3-Embedding-8B
# DiffAlign (last layer) due to the layer, or due to Apertus?
#
# Follows the official recipe (ZurichNLP/SwissGov-RSD, rsd/recognizers/diff_align.py and
# feature_based.py, run through scripts/predict_encoder.py --type diffalign):
# - each document is encoded alone: tok(text, padding=True, truncation=True), i.e. the
#   whitespace-tokenized text followed by <|endoftext|> (the tokenizer's post-processor);
#   no prompt, no instruction;
# - subwords are grouped into words the official way for this tokenizer: a token that
#   starts with "Ġ" (or the first token) starts a new word, other tokens are appended
#   to the current word, special tokens are skipped;
# - per subword: 1 - max cosine to ALL positions of the other document's sequence
#   (including its <|endoftext|>), then the mean over the word's subwords.
# Deviations: fp16 weights with the residual stream and the MLP activation in fp32 (the
# official run used fp32 weights; T4s cannot hold 8B in fp32), and every hidden state is
# scored instead of only the last one.
# "cent" additionally subtracts, per layer, the mean state over the document subwords of
# both sides (special tokens excluded) before the cosine, like the Apertus features.
#
# Hidden-state indices follow transformers' hidden_states: 0 = embeddings,
# l = output of decoder layer l (1..35), 36 = final norm output (= the official layer=-1).
# Outputs in /kaggle/working:
#   features.npz  per-word float32 da_alone_raw, da_alone_cent [N, 37], n_sub, and the
#                 lang / doc_id / side / word index (same layout as 01_extract_dev)
#   meta.json     model id and commit, versions, timing, max |state|, grouping checks

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0", "accelerate"], check=True)

import json
import os
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import transformers
import transformers.integrations.sdpa_attention as sdpa_attention
from huggingface_hub import HfApi, login
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

from rsd.apertus import diffalign, pool, word_index
from rsd.data import LANGS, load_docs

MODEL = "Qwen/Qwen3-Embedding-8B"
N_STATES = 37  # embeddings + 36 layers (the last one after the final norm)
SMOKE = False  # True: two docs per pair (the first and the longest), to check memory and the pipeline
OUT = Path("/kaggle/working")


# Same fp16 safety measures as rsd/apertus.py: fp32 residual stream, fp32 activation
# scaled per token before the fp16 down_proj (linear, so exact up to rounding).
def _mlp_forward(self, x):
    h = self.act_fn(self.gate_proj(x).float()) * self.up_proj(x).float()
    scale = torch.exp2(torch.ceil(torch.log2(h.abs().amax(-1, keepdim=True).clamp(min=1.0) / 2**12)).clamp(min=0))
    return self.down_proj((h / scale).half()).float() * scale


def _layer_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                   use_cache=False, position_embeddings=None, **kwargs):
    residual = hidden_states.float()
    h, _ = self.self_attn(
        hidden_states=self.input_layernorm(residual).half(),
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    residual = residual + h.float()
    return residual + self.mlp(self.post_attention_layernorm(residual).half())


sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False  # expand KV heads; keeps the mem-efficient kernel
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
modeling_qwen3.Qwen3MLP.forward = _mlp_forward
modeling_qwen3.Qwen3DecoderLayer.forward = _layer_forward

login(next(Path("/kaggle/input").rglob("hf_token.txt")).read_text().strip())
sha = HfApi().model_info(MODEL).sha
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, revision=sha)
model, info = AutoModel.from_pretrained(
    MODEL, revision=sha, dtype=torch.float16, device_map="auto",
    max_memory={0: "14GiB", 1: "14GiB"}, output_loading_info=True,
)
model.eval()
assert not info["missing_keys"], f"missing keys: {info['missing_keys'][:10]}"
assert len(model.layers) == N_STATES - 1
print(f"{MODEL}@{sha} ({type(model).__name__}) loaded in {time.time() - t0:.0f}s; torch {torch.__version__}, "
      f"transformers {transformers.__version__}; unexpected keys: {len(info['unexpected_keys'])}", flush=True)
SPECIAL = set(tok.all_special_tokens)


def encode(text: str):
    """Official tokenization and word grouping. Returns (ids, positions of word subwords, word of each position)."""
    enc = tok(text, padding=True, truncation=True)
    e = enc.encodings[0]
    groups = []
    for i, (wid, t) in enumerate(zip(e.word_ids, e.tokens)):
        if wid is None:  # special token
            continue
        is_tail = not t.startswith("Ġ") and t not in SPECIAL
        if is_tail and groups:
            groups[-1].append(i)
        else:
            groups.append([i])
    pos = [i for g in groups for i in g]
    word = [w for w, g in enumerate(groups) for _ in g]
    return enc["input_ids"], pos, word, len(groups)


def offset_words(text: str):
    """Our own offset-based grouping (rsd.apertus.word_index), for comparison only."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return word_index(text, enc["offset_mapping"])


timing = {"passes": 0, "tokens": 0, "seconds": 0.0}
max_abs_doc = [0.0] * N_STATES
max_abs_special = [0.0] * N_STATES


@torch.no_grad()
def states_of(ids, pos):
    """fp32 hidden states [len(ids), d] at every index 0..36, all positions kept."""
    states = [None] * N_STATES

    def keep(i):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            states[i] = h[0].to(torch.float32, copy=True)
        return hook

    modules = [model.embed_tokens, *model.layers[:N_STATES - 2], model.norm]
    handles = [m.register_forward_hook(keep(i)) for i, m in enumerate(modules)]
    torch.cuda.synchronize()
    t = time.time()
    try:
        model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    torch.cuda.synchronize()
    timing["passes"] += 1
    timing["tokens"] += len(ids)
    timing["seconds"] += time.time() - t
    special = sorted(set(range(len(ids))) - set(pos))
    for i, h in enumerate(states):
        if h is None or not torch.isfinite(h).all():
            raise FloatingPointError(f"hidden state {i} is missing or not finite")
        max_abs_doc[i] = max(max_abs_doc[i], h[pos].abs().max().item())
        if special:
            max_abs_special[i] = max(max_abs_special[i], h[special].abs().max().item())
    return states


cols = {k: [] for k in ("n_sub", "da_alone_raw", "da_alone_cent")}
index = []  # (lang, doc id, side, word)
checks = {"docs": 0, "grouping_differs_from_offsets": [], "offset_grouping_failed": []}
start = time.time()
n_done = 0
for lang in LANGS:
    docs = load_docs(lang, "dev")
    if SMOKE:
        longest = max(docs, key=lambda d: len(d.tokens_a) + len(d.tokens_b))
        docs = [docs[0], longest]
    for doc in docs:
        text = {"a": doc.text_a, "b": doc.text_b}
        n_gold = {"a": len(doc.labels_a), "b": len(doc.labels_b)}
        enc, st = {}, {}
        for s in "ab":
            ids, pos, word, n_words = encode(text[s])
            assert n_words == n_gold[s], (doc.id, s, n_words, n_gold[s])
            try:
                if offset_words(text[s]) != word:
                    checks["grouping_differs_from_offsets"].append(f"{doc.id}:{s}")
            except ValueError:
                checks["offset_grouping_failed"].append(f"{doc.id}:{s}")
            enc[s] = (pos, torch.tensor(word), n_words)
            st[s] = states_of(ids, pos)
        for s, o in (("a", "b"), ("b", "a")):
            pos, word, n_words = enc[s]
            raw, cent = [], []
            for x, y in zip(st[s], st[o]):
                mean = torch.cat([x[pos], y[enc[o][0]]]).mean(0)
                w = word.to(x.device)
                raw.append(pool(diffalign(x, y)[pos], w, n_words).cpu())
                cent.append(pool(diffalign(x, y, mean)[pos], w, n_words).cpu())
            res = {"da_alone_raw": torch.stack(raw, 1).numpy(), "da_alone_cent": torch.stack(cent, 1).numpy(),
                   "n_sub": torch.bincount(word, minlength=n_words).numpy()}
            for k, v in res.items():
                v = np.asarray(v, dtype=np.float32)
                if not np.isfinite(v).all():
                    raise FloatingPointError(f"{doc.id} side {s}: {k} not finite")
                cols[k].append(v)
            index += [(lang, doc.id, s, i) for i in range(n_words)]
        del st
        checks["docs"] += 1
        n_done += 1
        if n_done % 25 == 0 or SMOKE:
            el = time.time() - start
            print(f"{n_done} pairs, {el / 60:.1f} min, {timing['tokens'] / el:.0f} tok/s, "
                  f"peak GiB {[round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in range(torch.cuda.device_count())]}",
                  flush=True)
    print(f"{lang} done: {len(docs)} docs", flush=True)

feats = {k: np.concatenate(v) for k, v in cols.items()}
np.savez(OUT / "features.npz", **feats,
         lang=np.array([r[0] for r in index]), doc_id=np.array([r[1] for r in index]),
         side=np.array([r[2] for r in index]), word=np.array([r[3] for r in index], dtype=np.int32))
meta = {
    "model": MODEL,
    "model_sha": sha,
    "split": "dev",
    "smoke": SMOKE,
    "input": "tok(text, padding=True, truncation=True): whitespace-tokenized text + <|endoftext|>, each document alone",
    "grouping": "official: a token starting with 'Ġ' starts a new word; special tokens skipped",
    "state_indices": "0 = embeddings, l = output of decoder layer l (1..35), 36 = final norm output (official layer=-1)",
    "diffalign": "1 - max cos to all positions of the other doc's sequence (incl. <|endoftext|>), mean over the "
                 "word's subwords; cent subtracts, per layer, the mean over the doc subwords of both sides",
    "dtype": "fp16 weights; residual stream and MLP activation in fp32; states and cosines in fp32",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "timing": timing,
    "wall_seconds": time.time() - start,
    "max_abs_state_doc": max_abs_doc,
    "max_abs_state_special": max_abs_special,
    "n_words": len(index),
    "checks": checks,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps({k: meta[k] for k in ("timing", "wall_seconds", "n_words")}, indent=2))
print("grouping checks:", {k: (v if isinstance(v, int) else len(v)) for k, v in checks.items()},
      checks["grouping_differs_from_offsets"][:10], checks["offset_grouping_failed"][:10])
print("max |state| at doc positions per layer:", [round(x) for x in max_abs_doc])
print("max |state| at special positions per layer:", [round(x) for x in max_abs_special])
