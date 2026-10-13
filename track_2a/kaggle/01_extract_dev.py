# Dev-split feature extraction with Apertus 1.5 8B on Kaggle (2x T4, fp16).
# Run from track_2a/ with:
#   python kaggle/run_on_kaggle.py kaggle/01_extract_dev.py --kernel amunozo/rsd-extract-dev
# Uses the dev split only (load_docs(lang, "dev")).
#
# Four passes per pair, for X = a (English) and X = b (de/fr/it), Y = the other side:
#   alone_X = [BOS] {NameX}:\n X
#   cond_X  = [BOS] {NameY}:\n Y \n\n{NameX}:\n X
# Only X's positions are read. Outputs in /kaggle/working:
#   features.npz   per-word float32 features (one row per word, see `index`):
#                  n_sub, lp_{alone,cond}_{sum,first}, da_{alone,cond}_{raw,cent} [N, 33]
#   index.tsv      row -> lang, doc id, side, word index, row in the state files
#   states/cond_L{l}_{lang}.pt  word-pooled cond-pass states (bf16), one per layer and pair
#   meta.json      model id and commit, frames, layers, versions, timing, max |state|

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
from huggingface_hub import HfApi, login

from rsd import apertus
from rsd.apertus import MODEL, N_STATES, build, diffalign, encode_doc, forward, pool
from rsd.data import LANGS, load_docs

STATE_LAYERS = [8, 16, 24, 32]
OUT = Path("/kaggle/working")
(OUT / "states").mkdir(parents=True, exist_ok=True)

login(next(Path("/kaggle/input").rglob("hf_token.txt")).read_text().strip())
sha = HfApi().model_info(MODEL).sha
t0 = time.time()
tok, model = apertus.load(revision=sha)
print(f"{MODEL}@{sha} loaded in {time.time() - t0:.0f}s; torch {torch.__version__}, "
      f"transformers {transformers.__version__}", flush=True)

timing = {k: {"passes": 0, "tokens": 0, "seconds": 0.0} for k in ("alone", "cond")}
max_abs = [0.0] * N_STATES


def run(kind, seq, x_start):
    torch.cuda.synchronize()
    t = time.time()
    states, logp = forward(model, seq, x_start)
    torch.cuda.synchronize()
    timing[kind]["passes"] += 1
    timing[kind]["tokens"] += len(seq)
    timing[kind]["seconds"] += time.time() - t
    for i, h in enumerate(states):
        max_abs[i] = max(max_abs[i], h.abs().max().item())
    return states, logp


def word_logp(logp, word, n_words):
    """Per word: sum of its subwords' log-probs and the first subword's log-prob."""
    total = torch.zeros(n_words, device=logp.device).index_add_(0, word, logp)
    is_first = torch.ones_like(word, dtype=torch.bool)
    is_first[1:] = word[1:] != word[:-1]
    first = logp[is_first]  # words are in order and each has >= 1 subword
    assert len(first) == n_words
    return total.cpu().numpy(), first.cpu().numpy()


def diffalign_layers(xs, ys, word, n_words):
    """[n_words, 33] raw and mean-centered DiffAlign scores of X against Y."""
    raw, cent = [], []
    for x, y in zip(xs, ys):
        mean = torch.cat([x, y]).mean(0)
        w = word.to(x.device)
        raw.append(pool(diffalign(x, y), w, n_words).cpu())
        cent.append(pool(diffalign(x, y, mean), w, n_words).cpu())
    return torch.stack(raw, 1).numpy(), torch.stack(cent, 1).numpy()


cols = {k: [] for k in ("n_sub", "lp_alone_sum", "lp_alone_first", "lp_cond_sum", "lp_cond_first",
                        "da_alone_raw", "da_alone_cent", "da_cond_raw", "da_cond_cent")}
index = []  # (lang, doc id, side, word, state row)
start = time.time()
n_done = 0
for lang in LANGS:
    docs = load_docs(lang, "dev")
    n_words_lang = sum(len(d.labels_a) + len(d.labels_b) for d in docs)
    states_out = {l: torch.empty((n_words_lang, model.config.hidden_size), dtype=torch.bfloat16)
                  for l in STATE_LAYERS}
    row = 0
    for doc in docs:
        side_lang = {"a": "en", "b": lang}
        text = {"a": doc.text_a, "b": doc.text_b}
        labels = {"a": doc.labels_a, "b": doc.labels_b}
        enc = {}
        for s in "ab":
            sub_ids, word = encode_doc(tok, text[s])
            assert word[-1] + 1 == len(labels[s]), (doc.id, s, word[-1] + 1, len(labels[s]))
            enc[s] = (sub_ids, torch.tensor(word, device=model.device))
        n_words = {s: len(labels[s]) for s in "ab"}
        other = {"a": "b", "b": "a"}

        alone = {s: run("alone", *build(tok, enc[s][0], side_lang[s])) for s in "ab"}
        res = {s: {} for s in "ab"}
        for s in "ab":
            o = other[s]
            res[s]["da_alone_raw"], res[s]["da_alone_cent"] = diffalign_layers(
                alone[s][0], alone[o][0], enc[s][1], n_words[s])
            w = enc[s][1].to(alone[s][1].device)
            res[s]["lp_alone_sum"], res[s]["lp_alone_first"] = word_logp(alone[s][1], w, n_words[s])
        del alone

        cond = {s: run("cond", *build(tok, enc[s][0], side_lang[s], enc[other[s]][0], side_lang[other[s]]))
                for s in "ab"}
        for s in "ab":
            o = other[s]
            res[s]["da_cond_raw"], res[s]["da_cond_cent"] = diffalign_layers(
                cond[s][0], cond[o][0], enc[s][1], n_words[s])
            w = enc[s][1].to(cond[s][1].device)
            res[s]["lp_cond_sum"], res[s]["lp_cond_first"] = word_logp(cond[s][1], w, n_words[s])
            res[s]["n_sub"] = torch.bincount(enc[s][1], minlength=n_words[s]).cpu().numpy()
            for l in STATE_LAYERS:
                h = cond[s][0][l]
                states_out[l][row:row + n_words[s]] = pool(h, enc[s][1].to(h.device), n_words[s]).to(torch.bfloat16).cpu()
            for k, v in res[s].items():
                v = np.asarray(v, dtype=np.float32)
                if not np.isfinite(v).all():
                    raise FloatingPointError(f"{doc.id} side {s}: {k} not finite")
                cols[k].append(v)
            index += [(lang, doc.id, s, i, row + i) for i in range(n_words[s])]
            row += n_words[s]
        del cond
        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - start
            print(f"{n_done}/{3 * 168} pairs, {el / 60:.1f} min, "
                  f"{(timing['alone']['tokens'] + timing['cond']['tokens']) / el:.0f} tok/s, "
                  f"peak GiB {[round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in range(torch.cuda.device_count())]}",
                  flush=True)
    assert row == n_words_lang
    for l in STATE_LAYERS:
        torch.save(states_out[l], OUT / "states" / f"cond_L{l}_{lang}.pt")
    del states_out
    print(f"{lang} done: {len(docs)} docs, {n_words_lang} words", flush=True)

feats = {k: np.concatenate(v) for k, v in cols.items()}
np.savez(OUT / "features.npz", **feats,
         lang=np.array([r[0] for r in index]), doc_id=np.array([r[1] for r in index]),
         side=np.array([r[2] for r in index]), word=np.array([r[3] for r in index], dtype=np.int32),
         state_row=np.array([r[4] for r in index], dtype=np.int32))
with open(OUT / "index.tsv", "w") as f:
    f.write("lang\tdoc_id\tside\tword\tstate_row\n")
    f.writelines("\t".join(map(str, r)) + "\n" for r in index)

sizes = {str(p.relative_to(OUT)): p.stat().st_size for p in sorted(OUT.rglob("*")) if p.is_file()}
meta = {
    "model": MODEL,
    "model_sha": sha,
    "split": "dev",
    "frames": {"alone": "[BOS] {NameX}:\\n X", "cond": "[BOS] {NameY}:\\n Y \\n\\n{NameX}:\\n X",
               "names": apertus.NAMES},
    "state_indices": "0 = embeddings, l = output of decoder layer l (1..31), 32 = final norm output",
    "diffalign": "1 - max cos to the other doc's subwords (doc positions only), mean over the word's subwords; "
                 "cent subtracts, per layer, the mean over all doc subwords of X and Y of the pair; "
                 "alone: X alone vs Y alone; cond: X from cond_X vs Y from cond_Y",
    "state_layers": STATE_LAYERS,
    "state_pooling": "mean over the word's subwords, cond pass of X, bfloat16",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "timing": timing,
    "wall_seconds": time.time() - start,
    "max_abs_state": max_abs,
    "n_words": len(index),
    "file_bytes": sizes,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps({k: meta[k] for k in ("timing", "wall_seconds", "n_words")}, indent=2))
print(f"output total {sum(sizes.values()) / 1e9:.2f} GB; features.npz {sizes['features.npz'] / 1e6:.0f} MB")
