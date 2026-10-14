"""Per-word Apertus features for one split. Shared by the dev and test runs.

kaggle/01_extract_dev.py (dev, with hidden-state dumps) and
kaggle/04_extract_test.py (test, final run, no state dumps) both call main()
below, so the test features come from exactly the code that made the dev
features. Needs torch and transformers (Kaggle, or a CPU dry run).

Four passes per pair, for X = a (English) and X = b (de/fr/it), Y = the other side:
  alone_X = [BOS] {NameX}:\\n X
  cond_X  = [BOS] {NameY}:\\n Y \\n\\n{NameX}:\\n X
Only X's positions are read. Outputs in `out`:
  features.npz   per-word float32 features (one row per word, see `index`):
                 n_sub, lp_{alone,cond}_{sum,first}, da_{alone,cond}_{raw,cent} [N, 33]
  index.tsv      row -> lang, doc id, side, word index, row in the state files
  states/cond_L{l}_{lang}.pt  word-pooled cond-pass states (bf16), one per layer
                 in `state_layers` and pair (none if state_layers is empty)
  meta.json      model id and commit, frames, layers, versions, timing, max |state|

Dry run (CPU, no GPU, no HF login, never the test split): set RSD_DRY_RUN=1.
The model is a tiny random Apertus (same architecture and patches, 32 layers,
hidden size 64), the tokenizer is the public swiss-ai/Apertus-8B-2509 one (or
RSD_DRY_RUN_TOKENIZER), the split is forced to dev, and only the first
RSD_DRY_RUN_DOCS (default 2) documents per pair are used. Output goes to
RSD_OUT (default cache/dry_run/<split>/). This exercises the whole code path
of a run except the model weights and, for the test script, the split.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from rsd import apertus
from rsd.apertus import MODEL, N_STATES, build, diffalign, encode_doc, forward, pool
from rsd.data import LANGS, load_docs

FEATURE_KEYS = ("n_sub", "lp_alone_sum", "lp_alone_first", "lp_cond_sum", "lp_cond_first",
                "da_alone_raw", "da_alone_cent", "da_cond_raw", "da_cond_cent")
# Commit of MODEL used by the dev run (kaggle/outputs/01_extract_dev/meta.json).
# Pinned so the test run loads exactly the same weights and tokenizer.
MODEL_SHA = "c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01"
DRY_RUN_TOKENIZER = "swiss-ai/Apertus-8B-2509"  # public; same tokenizer family, used only for dry runs


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Runner:
    """Forward passes with timing and the running max |state| per layer."""

    def __init__(self, model):
        self.model = model
        self.timing = {k: {"passes": 0, "tokens": 0, "seconds": 0.0} for k in ("alone", "cond")}
        self.max_abs = [0.0] * N_STATES

    def __call__(self, kind, seq, x_start):
        _sync()
        t = time.time()
        states, logp = forward(self.model, seq, x_start)
        _sync()
        self.timing[kind]["passes"] += 1
        self.timing[kind]["tokens"] += len(seq)
        self.timing[kind]["seconds"] += time.time() - t
        for i, h in enumerate(states):
            self.max_abs[i] = max(self.max_abs[i], h.abs().max().item())
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


def process_pair(tok, model, run, doc, state_layers):
    """Features of one document pair.

    Returns {side: (features, states)} for side "a" then "b": features maps
    each FEATURE_KEYS name to a float32 array [n_words(, 33)], states maps each
    layer in state_layers to the word-pooled cond-pass state (bf16, CPU).
    Raises if a feature is not finite.
    """
    lang = doc.lang
    side_lang = {"a": "en", "b": lang}
    text = {"a": doc.text_a, "b": doc.text_b}
    labels = {"a": doc.labels_a, "b": doc.labels_b}  # only their lengths are used
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
    out = {}
    for s in "ab":
        o = other[s]
        res[s]["da_cond_raw"], res[s]["da_cond_cent"] = diffalign_layers(
            cond[s][0], cond[o][0], enc[s][1], n_words[s])
        w = enc[s][1].to(cond[s][1].device)
        res[s]["lp_cond_sum"], res[s]["lp_cond_first"] = word_logp(cond[s][1], w, n_words[s])
        res[s]["n_sub"] = torch.bincount(enc[s][1], minlength=n_words[s]).cpu().numpy()
        states = {}
        for l in state_layers:
            h = cond[s][0][l]
            states[l] = pool(h, enc[s][1].to(h.device), n_words[s]).to(torch.bfloat16).cpu()
        feats = {}
        for k, v in res[s].items():
            v = np.asarray(v, dtype=np.float32)
            if not np.isfinite(v).all():
                raise FloatingPointError(f"{doc.id} side {s}: {k} not finite")
            feats[k] = v
        out[s] = (feats, states)
    del cond
    return out


def extract(tok, model, docs_by_lang, out: Path, state_layers):
    """Run process_pair over all documents and write features.npz, index.tsv
    and the state files. Returns (n_words, runner, wall seconds)."""
    if state_layers:
        (out / "states").mkdir(parents=True, exist_ok=True)
    run = Runner(model)
    cols = {k: [] for k in FEATURE_KEYS}
    index = []  # (lang, doc id, side, word, state row)
    n_total = sum(len(d) for d in docs_by_lang.values())
    start = time.time()
    n_done = 0
    for lang, docs in docs_by_lang.items():
        n_words_lang = sum(len(d.labels_a) + len(d.labels_b) for d in docs)
        states_out = {l: torch.empty((n_words_lang, model.config.hidden_size), dtype=torch.bfloat16)
                      for l in state_layers}
        row = 0
        for doc in docs:
            assert doc.lang == lang
            for s, (feats, states) in process_pair(tok, model, run, doc, state_layers).items():
                n = len(feats["n_sub"])
                for l in state_layers:
                    states_out[l][row:row + n] = states[l]
                for k, v in feats.items():
                    cols[k].append(v)
                index += [(lang, doc.id, s, i, row + i) for i in range(n)]
                row += n
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - start
                peak = ([round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in range(torch.cuda.device_count())]
                        if torch.cuda.is_available() else [])
                print(f"{n_done}/{n_total} pairs, {el / 60:.1f} min, "
                      f"{(run.timing['alone']['tokens'] + run.timing['cond']['tokens']) / el:.0f} tok/s, "
                      f"peak GiB {peak}", flush=True)
        assert row == n_words_lang
        for l in state_layers:
            torch.save(states_out[l], out / "states" / f"cond_L{l}_{lang}.pt")
        del states_out
        print(f"{lang} done: {len(docs)} docs, {n_words_lang} words", flush=True)

    feats = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(out / "features.npz", **feats,
             lang=np.array([r[0] for r in index]), doc_id=np.array([r[1] for r in index]),
             side=np.array([r[2] for r in index]), word=np.array([r[3] for r in index], dtype=np.int32),
             state_row=np.array([r[4] for r in index], dtype=np.int32))
    with open(out / "index.tsv", "w") as f:
        f.write("lang\tdoc_id\tside\tword\tstate_row\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in index)
    return len(index), run, time.time() - start


def tiny_random_model(tokenizer: str = DRY_RUN_TOKENIZER, seed: int = 0):
    """(tokenizer, model) for dry runs: a random 32-layer Apertus with hidden
    size 64 in fp16, patched like the real one. Deterministic given the seed."""
    from transformers import ApertusConfig, AutoModelForCausalLM, AutoTokenizer

    apertus.patch()
    tok = AutoTokenizer.from_pretrained(tokenizer)
    cfg = ApertusConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=N_STATES - 1,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=65536,
                        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg).half().eval()
    assert tok.bos_token_id is not None
    return tok, model


def _code_hashes():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ("apertus.py", "extract.py", "data.py")}


def main(split: str, final: bool, state_layers: list[int], out: Path = Path("/kaggle/working")):
    """One full extraction run (Kaggle), or a dry run if RSD_DRY_RUN=1."""
    import transformers

    dry = os.environ.get("RSD_DRY_RUN") == "1"
    if dry:
        n_docs = int(os.environ.get("RSD_DRY_RUN_DOCS", "2"))
        if split == "test":
            print("DRY RUN: test split replaced by dev", flush=True)
        split, final = "dev", False
        out = Path(os.environ.get("RSD_OUT", f"cache/dry_run/{split}"))
    if split == "test" and not final:
        raise PermissionError("The test split is only for the final run: pass final=True.")
    out.mkdir(parents=True, exist_ok=True)

    # Data first, so a data problem fails before the model download.
    docs_by_lang = {lang: load_docs(lang, split, final=final) for lang in LANGS}
    if dry:
        docs_by_lang = {lang: docs[:n_docs] for lang, docs in docs_by_lang.items()}
    print({lang: len(d) for lang, d in docs_by_lang.items()}, f"{split} documents", flush=True)

    t0 = time.time()
    if dry:
        model_id = f"tiny random Apertus (dry run), tokenizer {os.environ.get('RSD_DRY_RUN_TOKENIZER', DRY_RUN_TOKENIZER)}"
        sha = None
        tok, model = tiny_random_model(os.environ.get("RSD_DRY_RUN_TOKENIZER", DRY_RUN_TOKENIZER))
    else:
        from huggingface_hub import HfApi, login

        login(next(Path("/kaggle/input").rglob("hf_token.txt")).read_text().strip())
        model_id, sha = MODEL, HfApi().model_info(MODEL, revision=MODEL_SHA).sha
        assert sha == MODEL_SHA, (sha, MODEL_SHA)
        tok, model = apertus.load(revision=sha)
    print(f"{model_id}@{sha} loaded in {time.time() - t0:.0f}s; torch {torch.__version__}, "
          f"transformers {transformers.__version__}", flush=True)

    n_words, run, wall = extract(tok, model, docs_by_lang, out, state_layers)

    sizes = {str(p.relative_to(out)): p.stat().st_size for p in sorted(out.rglob("*")) if p.is_file()}
    meta = {
        "model": model_id,
        "model_sha": sha,
        "split": split,
        "final": final,
        "dry_run": dry,
        "n_docs": {lang: len(d) for lang, d in docs_by_lang.items()},
        "code_sha256": _code_hashes(),
        "frames": {"alone": "[BOS] {NameX}:\\n X", "cond": "[BOS] {NameY}:\\n Y \\n\\n{NameX}:\\n X",
                   "names": apertus.NAMES},
        "state_indices": "0 = embeddings, l = output of decoder layer l (1..31), 32 = final norm output",
        "diffalign": "1 - max cos to the other doc's subwords (doc positions only), mean over the word's subwords; "
                     "cent subtracts, per layer, the mean over all doc subwords of X and Y of the pair; "
                     "alone: X alone vs Y alone; cond: X from cond_X vs Y from cond_Y",
        "state_layers": state_layers,
        "state_pooling": "mean over the word's subwords, cond pass of X, bfloat16",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "timing": run.timing,
        "wall_seconds": wall,
        "max_abs_state": run.max_abs,
        "n_words": n_words,
        "file_bytes": sizes,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("timing", "wall_seconds", "n_words")}, indent=2))
    print(f"output total {sum(sizes.values()) / 1e9:.2f} GB; features.npz {sizes['features.npz'] / 1e6:.0f} MB")
    return meta
