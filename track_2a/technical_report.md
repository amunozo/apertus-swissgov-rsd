# Technical Report — Reading Cross-Lingual Semantic Differences from Apertus v1.5 8B

- **Track:** Track 2A — UZH: Detecting Cross-Lingual Semantic Differences in Swiss Government Websites (`SwissGov-RSD`)
- **Event:** Online Stage (1–16 October 2026)
- **Team:** Alberto Muñoz Ortiz (solo)
- **Demo / Artefacts:** Interactive token-level HTML heatmaps in `docs/heatmaps/index.html`; compact precomputed Apertus features in `data/features_dev_head.npz` and `data/features_test_head.npz`; frozen test predictions and fitted weights in `data/predictions_test/`; `make run` refits the head and reproduces the official test evaluation in Docker on CPU in ~1 minute.

## 1. Summary

**Task and metric.** SwissGov-RSD (Wastl et al., 2026) evaluates token-level recognition of cross-lingual semantic differences between parallel English (`en`) and German (`de`), French (`fr`), or Italian (`it`) web pages from `admin.ch`. Given a document pair pre-tokenised on whitespace, a system predicts a score in $[0, 1]$ for every token on both sides indicating how much its meaning differs from the other text. The official metric is global token-level Spearman rank correlation ($\rho \times 100$) computed over all non-punctuation tokens (`gold != -1`) of all documents in a language pair, macro-averaged across EN–DE, EN–FR, and EN–IT.

**Approach.** We use **Apertus v1.5 8B** (`andreasmartin/apertus-v1.5-8b-text`, pinned at commit `c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01`) frozen as a **reader rather than a text generator**. For each document pair, four causal forward passes (`alone_A`, `alone_B`, `cond_A` with $B$ before $A$, and `cond_B` with $A$ before $B$) extract two families of word-level signals:
1. **Middle-layer DiffAlign** ($1 - \max_j \cos(x_i, y_j)$ against subwords of the other document, pooled by subword mean, both raw and mean-centred) at intermediate decoder layers (12, 14, 16, 18); and
2. **Contrastive surprisal (`-gain`)**, the negative log-probability gain $-\bigl(\log p(\text{word} \mid \text{other doc, prefix}) - \log p(\text{word} \mid \text{prefix})\bigr)$, which measures how much conditioning on the other document fails to make each word more predictable.

An L2-regularised logistic regression head with 96 scalar features—combining DiffAlign at layers 12–18, contrastive surprisal and raw log-probabilities, within-side moving averages of widths 1, 3, 7, and 15 words, document-level aggregates (`side_mean`, `other_mean`, `pair_mean`), and layout features—is trained with soft-label binary cross-entropy via L-BFGS-B on the 504 labelled development documents (168 English pages across 3 pairs). No text is generated in the submitted system.

**Headline results.** The held-out test split (56 document pairs per language pair, 168 total) was touched once on 14 October 2026, from the frozen git tag `final-run` (commit `4c804d4`). On test, our shared 96-feature head reaches **37.4** macro Spearman (95% page-bootstrap CI [31.9, 42.4]), compared to **17.1** ([13.7, 20.5]) for the best of the seven rescored baselines on test (`DiffAlign Qwen3-Embedding-8B`) evaluated on the same split with the official evaluation script (`docs/test_results.md`, `docs/ablation_dev.md`; the paper's segmented `LaBSE` and `XLM-R + SimCSE` runs were not rescored):

| System | Split | EN–DE | EN–FR | EN–IT | Macro Mean [95% CI] |
|---|---|---:|---:|---:|---:|
| **Apertus v1.5 8B 96-feature scalar head (ours, shared)** | **Test (56/pair)** | **42.8** | **23.4** | **46.0** | **37.4** [31.9, 42.4] |
| DiffAlign `Qwen3-Embedding-8B` (best of 7 rescored baselines on test) | Test (56/pair) | 19.7 | 11.2 | 20.4 | 17.1 [13.7, 20.5] |
| Apertus v1.5 8B scalar head (5-fold `page_en`-grouped nested CV) | Dev (168/pair) | 43.2 | 29.1 | 47.5 | 39.9 |
| Apertus v1.5 8B scalar head (`dev/train` $\to$ `dev/val`) | Dev/val (34/pair) | 38.6 | 31.8 | 43.7 | 38.0 |
| Apertus v1.5 8B unsupervised (`cond raw L14 +smooth7`) | Dev (168/pair) | 39.8 | 26.3 | 43.2 | 36.4 |
| Controlled `Qwen3-Embedding-8B` (`alone raw L21`, best layer) | Dev (168/pair) | 30.6 | 19.7 | 33.0 | 27.8 |
| Apertus v1.5 8B DiffAlign (`alone raw L14`, best layer) | Dev (168/pair) | 33.0 | 21.4 | 35.7 | 30.0 |
| DiffAlign `Qwen3-Embedding-4B` (best of 7 rescored baselines on dev) | Dev (168/pair) | 20.6 | 11.7 | 20.2 | 17.5 |
| DiffAlign `Qwen3-Embedding-8B` (released baseline) | Dev (168/pair) | 19.1 | 12.4 | 17.9 | 16.4 |
| Apertus v1.5 8B DiffAlign last layer (`alone raw L32`, paper recipe) | Dev (168/pair) | 5.4 | 5.4 | 9.8 | 6.9 |
| Apertus v1.5 8B few-shot prompting (paper recipe, 30 short docs) | Dev subset | 16.3 | -10.2 | 3.6 | 3.2 |

**Main findings on development data:**
- **Layer depth is the main factor.** The paper's DiffAlign recipe reads the final hidden state (`L32`), which in a generative causal LLM specialises in next-token prediction and scores only **6.9** on dev. Reading middle layers raises unconditioned Apertus DiffAlign to **30.0** at layer 14. Running a matched all-layer control with `Qwen/Qwen3-Embedding-8B` shows that about 11 points of the gap over the paper's released `Qwen3-Embedding-8B` baseline (16.4 at `L36`) also come from layer depth (`L21` reaches **27.8**, paired difference +11.3 [+10.1, +12.7]). At each model's best layer under the same unconditioned recipe, Apertus leads `Qwen3-Embedding-8B` by **+2.2** (95% paired bootstrap CI [+1.8, +2.6], or **+1.0** [+0.5, +1.4] when both are smoothed).
- **Stepwise gains from conditioning, smoothing, surprisal, and supervision.** Conditioning each side's forward pass on the other document adds **+2.4** points at layer 14 (30.0 $\to$ 32.4). Averaging scores over a 7-word within-side window adds **+4.0** points (32.4 $\to$ 36.4); this uses the contiguous span structure of the gold labels and gives a similar boost to Qwen. Contrastive surprisal (`-gain sum`) scores **19.6** raw and **27.2** smoothed on its own (vs. **0.0** for an unconditioned $-\log p_{\text{alone}}$ rarity control). Combining all signals in the 96-feature logistic head adds **+3.5** points in 5-fold page-grouped nested CV (36.4 $\to$ 39.9, gaining +2.2 to +4.5 across the five folds) and **+3.6** points on the official `dev/train` $\to$ `dev/val` split (34.4 $\to$ 38.0).
- **Reading internal states vs. generating token scores.** On a 30-document short-page dev subset ($\le 300$ words per pair, median 218 words), reproducing the paper's 3-shot JSON prompting recipe with greedy decoding scores **3.2** (10/30 outputs fail JSON parsing, 29/40 parsed sides have the wrong token count, and 64.5% of tokens tie at 0.0) at ~2.2 minutes per document, whereas reading internal states from the same model on the same 30 documents scores **36.0** (`alone raw L14`) to **42.5** (`cond raw L14 +smooth7`) at ~1 second per forward pass.

**Scope of claims and evaluation hygiene.** We do not claim that Apertus is a universally stronger encoder than Qwen (the controlled margin is +2.2 raw / +1.0 smoothed on one dataset at layers chosen on dev), nor do we claim to beat human agreement. Because Table 3 of Wastl et al. (2026) evaluates on the whole dataset (`dev` + `test` pooled), we compare only against released baseline predictions rescored on the exact same split. On dev, the layer range (12–18), windows, and head configuration were chosen using dev data (with only $\lambda$ nested inside CV, and a disclosed post-hoc switch from the per-pair head to the shared head on a virtual tie: 40.0 vs. 39.9 CV, 37.6 vs. 38.0 `dev/val`), so dev CV is slightly optimistic and the held-out test run (37.4) is the primary check. Finally, bootstrap CIs resample pages with the fitted head held fixed, and on the test split only our final head and the paper's released baselines were scored.

## 2. Architecture

### 2.1 Two-Stage GPU/CPU Split and Data Flow

Because judges execute `make run` inside a Docker container without a GPU, the architecture separates offline **GPU feature extraction** on Kaggle (2x NVIDIA T4) from lightweight **CPU head training, prediction, and official evaluation** in Docker:

```text
                      GPU part (Kaggle, 2x T4, run by us; not part of make run)
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ HF dataset ZurichNLP/SwissGov-RSD @ f16aa05 (dev or test texts)              │
 │        │                                                                      │
 │        ▼                                                                      │
 │ rsd.extract.main  (kaggle/01_extract_dev.py, kaggle/04_extract_test.py)       │
 │   Apertus v1.5 8B (andreasmartin/apertus-v1.5-8b-text @ c0e7eeb), fp16,      │
 │   overflow patches (rsd.apertus), 4 passes per pair:                          │
 │     alone_A, alone_B, cond_A (B then A), cond_B (A then B)                    │
 │        │                                                                      │
 │        ▼                                                                      │
 │ features.npz: one row per whitespace word                                     │
 │   DiffAlign at 33 hidden states x {alone, cond} x {raw, centred},             │
 │   log-probs (alone / cond, sum / first subword), subword count, ids           │
 └────────┬──────────────────────────────────────────────────────────────────────┘
          │ download (kaggle/run_on_kaggle.py), then CPU export
          ▼
 rsd.features export_compact  ──►  data/features_dev_head.npz  (27.5 MB, committed)
                                   data/features_test_head.npz (10.2 MB, committed)
          │
          ▼             CPU part (Docker, `make run`, run by judges)
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ rsd.pipeline                                                                  │
 │   └► rsd.head fit-predict: build 96 features (rsd.features.build)             │
 │        → z-score with training stats, clip ±5                                 │
 │        → logistic regression (soft-label BCE + L2, L-BFGS), fit on 504 dev    │
 │        → predict test words → unit_interval per pair → official format        │
 │   └► official_eval.run: official script (ZurichNLP/SwissGov-RSD @ 1807a42)    │
 │        → Spearman per pair → outputs/scores.json                              │
 └───────────────────────────────────────────────────────────────────────────────┘

 Analysis-only modules (dev only, CPU, not in make run): rsd.unsupervised,
 rsd.qwen_control, rsd.prompting, rsd.annotators, rsd.heatmap.
 Extra GPU scripts (dev only): kaggle/02_qwen_control.py, kaggle/03_prompting_baseline.py.
```

### 2.2 Stage 1: Offline GPU Feature Extraction (`rsd.apertus`, `rsd.extract`)

For every document pair $(A, B)$ in `dev` (504 pairs = 168 per language pair, 405,654 total words) and `test` (168 pairs = 56 per language pair, 150,434 total words), `rsd.extract.main` runs four causal forward passes through frozen Apertus v1.5 8B (`alone_A`, `alone_B`, `cond_A`, `cond_B`). From each pass, hooks collect all 33 hidden states (embedding output `0`, decoder layers `1..31`, and final RMSNorm output `32`) and next-token log-probabilities at the target document's subword positions (`src/rsd/apertus.py`).

Subword quantities are mapped monotonically to whitespace tokens (`src/rsd/extract.py`):
- **DiffAlign scalars:** For each hidden state $\ell \in \{0, \dots, 32\}$, pass type (`alone`, `cond`), and centring mode (`raw`, `cent`), subword-level DiffAlign ($1 - \max_j \cos(x_i, y_j)$ over all document subwords of the other side) is averaged over the subwords belonging to each whitespace word.
- **Word log-probabilities:** For `alone` and `cond` passes, both the **sum** of subword log-probabilities (`lp_alone_sum`, `lp_cond_sum`) and the **first** subword's log-probability (`lp_alone_first`, `lp_cond_first`) are recorded per word.

To keep the repository under the 100 MB `data/` limit, `python -m rsd.features <in.npz> <out.npz>` (`export_compact`) writes compressed `float32` archives (`data/features_dev_head.npz` at 27.5 MB and `data/features_test_head.npz` at 10.2 MB) containing only the document/word identifiers, the four log-probability arrays, and the DiffAlign columns at layers 12, 14, 16, and 18 needed by the head—with no gold labels and no hidden-state vectors (the dev run also dumped 13.55 GB of word-pooled hidden states at layers 8, 16, 24, and 32 on Kaggle for a planned hidden-state head that was not pursued). Feature matrices and head predictions computed from the compact archives are byte-identical to those from the full 250 MB `features.npz` (`docs/scalar_head_dev_results.md`, `tests/test_head.py::test_compact_export_roundtrip`).

### 2.3 Stage 2: CPU Head Training and Official Scoring (`rsd.features`, `rsd.head`, `rsd.pipeline`)

Inside Docker (`make run`), `python -m rsd.pipeline` executes entirely on CPU in ~1 minute without torch, GPU drivers, model weights, or API keys:
1. **Feature construction (`src/rsd/features.py`):** Builds the 96 scalar features per word from `data/features_dev_head.npz` and `data/features_test_head.npz`—combining DiffAlign at layers 12, 14, 16, and 18 (`alone`/`cond`, `raw`/`cent`) and log-probability features across 4 within-side smoothing windows ($k \in \{1, 3, 7, 15\}$), 12 document-level signal aggregates (`side_mean`, `other_mean`, `pair_mean`), and 4 layout features (`log len side`, `log len ratio`, `side is B`, `rel position`).
2. **Standardisation and head refit (`src/rsd/head.py`):** Rather than merely shipping static prediction files, `make run` refits the shared logistic regression head from scratch on all 504 dev documents (z-scoring features on `gold != -1` training rows, clipping features to $[-5, 5]$, and minimising soft-label binary cross-entropy plus L2 penalty $\lambda = 10^{-3}$ via `scipy` L-BFGS-B in ~15 seconds on CPU).
3. **Prediction and official evaluation (`src/rsd/predictions.py`, `src/official_eval/run.py`):** The fitted head predicts logits on the 168 test documents, maps each language pair's scores into $[0, 1]$ via a rank-preserving pair-wide min-max (`unit_interval`), writes official-format JSONL files to `outputs/test/head_{de,fr,it}.jsonl.jsonl` (without overwriting the frozen files in `data/predictions_test/`), and scores them with the pinned official evaluation script fetched at Docker build time.

### 2.4 Execution Summary Across Stages

| Component | Module / Script | Where Run | Hardware | Runtime |
|---|---|---|---|---|
| Smoke test & CPU bf16 check | `kaggle/00_smoke_test.py` | Kaggle | 2x T4 GPU + CPU | ~a few min |
| Dev feature extraction (504 pairs, 2,016 passes) | `kaggle/01_extract_dev.py` (`rsd.extract`) | Kaggle | 2x T4 GPU | ~45 min session (2,117 s passes) |
| Qwen3-Embedding-8B all-layer control (dev) | `kaggle/02_qwen_control.py` (`rsd.qwen_control`) | Kaggle | 2x T4 GPU | ~16 min full run (630 s passes; +~4 min smoke) |
| Few-shot prompting baseline (30 short dev docs) | `kaggle/03_prompting_baseline.py` (`rsd.prompting`) | Kaggle | 2x T4 GPU | ~72 min full run (4,036 s generation; +~9 min smoke) |
| Test feature extraction (168 pairs, 672 passes) | `kaggle/04_extract_test.py` (`rsd.extract`) | Kaggle | 2x T4 GPU | ~15 min (794 s passes, 899.5 s wall) |
| End-to-end head refit + official test scoring | `rsd.pipeline` (`make run`) | Local / Judge Docker | CPU only | ~1 min (~15 s head fit-predict) |

## 3. Use of Apertus

### 3.1 Model Identity, Architecture, and Pinning

- **Model ID and revision:** `andreasmartin/apertus-v1.5-8b-text`, pinned and asserted at Hugging Face commit `c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01` (`src/rsd/apertus.py:23`, `src/rsd/extract.py:44`, `docs/final_run_meta.json`). The test run loads the weights by this pinned revision (`src/rsd/extract.py:253-255`), so an upstream change could not alter them.
- **Checkpoint provenance:** The official `swiss-ai/Apertus-v1.5-8B` release is multimodal (Apertus 1.5 Omni). Following the hackathon resources guide, we use the community text-only conversion `andreasmartin/apertus-v1.5-8b-text` so that it loads cleanly with standard `transformers`. This conversion is derived from the instruction-tuned release (its tokenizer chat template begins with `"You are Apertus 1.5 Omni, a multimodal assistant..."`). Note that bitwise identity between the community conversion's text weights and the official multimodal checkpoint was not independently verified.
- **Architecture:** ~8.05B parameters, 32 decoder layers yielding 33 hidden-state tensors per forward pass (`0` = token embeddings, `1..31` = decoder block outputs, `32` = final RMSNorm output immediately before `lm_head`), hidden dimension $d = 4096$, grouped-query attention (GQA) with 32 query heads sharing 8 key/value heads, and a non-gated MLP `down_proj(xIELU(up_proj(x)))` using the quadratic `xIELU` activation.

### 3.2 How Apertus Is Used, and Role of Other Models

- **Frozen reader, not generator:** Apertus is used strictly for inference without fine-tuning, adapters, or quantisation. In the submitted system, we never generate text; instead, we read all 33 intermediate hidden states (using layers 12, 14, 16, and 18 in the final head) and the exact next-token log-probabilities of the input words. Text generation is used only in our dev-only reproduction of the paper's few-shot prompting baseline (`kaggle/03_prompting_baseline.py`).
- **Execution environment:** On Kaggle's 2x NVIDIA T4 GPUs (16 GB each, `torch 2.11.0+cu128`, `transformers 5.19.0`), model weights are loaded in `fp16` and partitioned sequentially across the two cards via `device_map="auto"` with `max_memory={0: "14GiB", 1: "14GiB"}` (`src/rsd/apertus.py:68-80`).
- **Role of other models:**
  1. `Qwen/Qwen3-Embedding-8B` (pinned at commit `1d8ad4c`) was run on the development split only (`kaggle/02_qwen_control.py`, `src/rsd/qwen_control.py`) as a controlled scientific baseline to separate the effect of layer depth from the choice of backbone.
  2. Released predictions of seven DiffAlign models from the benchmark paper's repository (`Qwen3-Embedding-8B`, `Qwen3-Embedding-4B`, `EuroBERT-610m`, `EuroBERT-2.1B`, `mmBERT-base`, `embeddinggemma-300m`, `bge-m3`) are rescored on dev and test as external baselines.
  No model other than Apertus v1.5 8B contributes to the submitted system.

### 3.3 Input Framing and Subword-to-Word Pooling for a Causal Decoder

Because a causal language model attends only to left context, reading document $X$ with document $Y$ in context requires placing $Y$ before $X$. For each document pair $(A, B)$ and each target side $X \in \{A, B\}$ (with $Y$ denoting the other side), we run two forward passes (`src/rsd/apertus.py:113-126`):

```text
alone_X = [BOS] "{NameX}:\n" X
cond_X  = [BOS] "{NameY}:\n" Y "\n\n{NameX}:\n" X
```

where `NameX`, `NameY` $\in \{\text{"English"}, \text{"German"}, \text{"French"}, \text{"Italian"}\}$, giving four passes per pair (`alone_A`, `alone_B`, `cond_A`, `cond_B`). Three implementation details make this contrast clean:
1. **Identical subword IDs across passes:** Each document is tokenised once in isolation (`encode_doc`, `add_special_tokens=False`), and the four input frames are constructed by concatenating token-ID lists rather than re-tokenising concatenated strings. Thus $X$'s subword IDs are identical in `alone_X` and `cond_X`; only the prefix before $X$ (`[BOS] "{NameX}:\n"` vs. `[BOS] "{NameY}:\n" Y "\n\n{NameX}:\n"`) differs.
2. **Reading only document positions:** We slice hidden states and log-probabilities strictly from `x_start` onward (the subwords of $X$ itself). BOS and header tokens—where massive activations (~119k) concentrate—are never included in cosine comparisons or mean-centring.
3. **Subword-to-word mapping (`src/rsd/apertus.py:83-106`, `src/rsd/extract.py:74-93`):** Whitespace tokens are joined by single spaces (`Doc.text_a`, `Doc.text_b`). Using character offsets from the tokenizer, each subword is assigned to the whitespace word containing its **last character** (so a leading space attaches to the following word). Monotonicity and non-empty coverage of every whitespace word (matching the length of the gold label list) are asserted on every document.
   - **DiffAlign pooling:** At each layer $\ell \in \{0, \dots, 32\}$, subword-level DiffAlign $1 - \max_j \cos(x_i, y_j)$ (comparing each subword of $X$ against all document subwords of $Y$, both raw and mean-centred after subtracting the pair's mean document state $\mu_\ell$) is **averaged** over the subwords of each whitespace word.
   - **Log-probability pooling:** For each subword $t$ of $X$, $\log p(\text{seq}[t] \mid \text{seq}[:t])$ is read from the final hidden state at $t-1$. Per whitespace word, we store both the **sum** over its subwords (`lp_*_sum`) and the **first** subword's log-probability (`lp_*_first`).

### 3.4 Solving fp16 Out-of-Memory and Numerical Overflow on 2x T4 GPUs

Apertus v1.5 8B weights are stored in `bf16` (8-bit exponent, same dynamic range as `fp32`), whereas NVIDIA T4 GPUs lack fast hardware `bf16` support and must execute in `fp16` (5-bit exponent, max finite value `65,504`). Running long document pairs on 2x T4 in `fp16` triggered two critical failures, both fixed in `src/rsd/apertus.py`:

1. **Problem 1 — Attention OOM on long pairs (up to 5,183 subwords):**
   On our longest concatenated dev pair (5,183 subwords), PyTorch's `scaled_dot_product_attention` (SDPA) attempted a 3.2 GB allocation and ran out of memory. Because `transformers` delegated grouped-query attention (32 query heads, 8 KV heads) directly to SDPA via `enable_gqa`, PyTorch on T4 (where FlashAttention is unavailable) fell back to the $O(n^2)$ `math` kernel ($32 \times 5183^2$ elements) instead of the linear-memory `memory-efficient` kernel.
   - **Fix (`src/rsd/apertus.py:58-62`):** Set `use_gqa_in_sdpa = lambda ...: False` so `transformers` expands the 8 KV heads to 32 before calling SDPA (enabling the `memory-efficient` kernel), and call `enable_math_sdp(False)` so any fallback raises an immediate error. With this fix, the 5,183-subword smoke test peaked at **10.0 / 9.9 GiB** per GPU, full dev extraction peaked at **11.3 / 11.2 GiB**, and test extraction peaked at **13.0 / 12.9 GiB** (within ~1 GiB of the 14 GiB cap).

2. **Problem 2 — `fp16` overflow (`NaN` from layer 2 onward):**
   In standard `fp16`, every forward pass produced `NaN`s starting at decoder layer 2 for two reasons: (a) the MLP activation `xIELU` is quadratic for positive inputs ($\approx \alpha x^2 + \beta x$), so moderate activations exceed `65,504` when squared; and (b) from layer 4 onward, Apertus exhibits **massive activations** (Sun et al., 2024) in a few residual-stream dimensions reaching **119,311** at layer 22 on BOS/header tokens (and up to **5,676** on dev / **5,430** on test at document positions).
   - **Fix (`src/rsd/apertus.py:28-65`):**
     - Keep model weights in `fp16`, but run the **residual stream** (`_layer_forward`) and **`xIELU` activation** (`_mlp_forward`) in `fp32`: only the RMSNorm outputs (`norm1(r)`, `norm2(r)`), whose magnitudes are small, are cast to `fp16` before attention and `up_proj`.
     - Around the linear `down_proj` (which has no bias and whose weights are `fp16`), apply **per-token power-of-two scaling**:
       $$s = 2^{\max\bigl(0,\; \lceil \log_2(\max(|h|, 1) / 2^{12}) \rceil\bigr)}, \qquad \text{out} = \text{down\_proj}\bigl((h / s)\text{.half()}\bigr)\text{.float()} \times s.$$
       Dividing by $s$ bounds every `fp16` input to $\le 4096$, preventing overflow inside the matrix multiplication. Because `down_proj` is linear ($W(h/s)s = Wh$) and dividing/multiplying by a power of two only shifts the floating-point exponent without rounding the mantissa, this scaling adds no rounding error beyond casting $h/s$ to `fp16`.
     - Disable reduced-precision `fp16` reductions (`allow_fp16_reduced_precision_reduction = False`), evaluate `lm_head` in 1,024-token chunks with `fp32` `log_softmax`, and raise `FloatingPointError` on any non-finite hidden state or log-probability.
   - **Validation against an independent CPU `bf16` reference (`kaggle/00_smoke_test.py`):** We ran the unpatched model in `bf16` on CPU over the first 128 tokens of the median dev pair and compared against our patched `fp16` GPU states (excluding BOS) at layers 1, 2, 4, 8, 16, 24, 31, and 32. Per-layer cosine similarity averaged **0.9997–0.9999** with a minimum of **0.9968** (at layer 24), and the layer-4 massive activation matched closely (119,113 on GPU vs. 119,808 on CPU). All 2,016 dev passes and 672 test passes completed with zero `NaN` or `Inf` values.

## 4. Data

### 4.1 Source, Pinned Revisions, and Licensing

- **Benchmark and pinned commits:** We use **SwissGov-RSD** (Wastl, Vamvas, and Sennrich, ACL 2026) from Hugging Face (`ZurichNLP/SwissGov-RSD`, pinned at revision `f16aa0536811b8c0cf975fede21eec5996e4b3c1`) together with the official splits, evaluation script, and released baseline predictions from GitHub (`ZurichNLP/SwissGov-RSD`, pinned at commit `1807a42100e742ed03d337c54c4b9ea86995f565`, fetched by `src/official_eval/fetch.py`).
- **Licensing and why the official evaluator is fetched rather than vendored:** The human annotations are released under **CC BY 4.0**, whereas copyright of the underlying multilingual web pages remains with the **Swiss federal authorities (`admin.ch`)**. In the upstream GitHub repository, only the `rsd/` subdirectory carries an MIT `LICENSE` file; `evaluation/` and `scripts/` have no license header. To avoid redistributing code with unclear licensing, our repository fetches the official evaluation script at Docker build time (`src/official_eval/fetch.py`) rather than copying it into the tree.
- **Flagged third-party text in committed artefacts:**
  1. `data/predictions_test/apertus_head_{de,fr,it}.jsonl.jsonl` contain `text_a` and `text_b` (the whitespace-tokenised texts of the 168 test documents) because the official evaluation script requires them in every prediction row (`src/rsd/predictions.py:76-78`, matching the format of the predictions published in the paper's own repository).
  2. `docs/heatmaps/*.html` display the full tokenised texts of four development documents for error analysis.
  All `admin.ch` texts in these files remain under the copyright of the Swiss federal authorities. The texts are public government communications (occasionally naming public officials in their official capacity); no personal data was collected or created by this project.

### 4.2 Label Semantics, Sparsity, and Span Structure

In SwissGov-RSD, two annotators per language pair marked divergent spans between parallel English and German/French/Italian `admin.ch` pages on a 1–5 dissimilarity scale (the inverse of the iSTS scale: 1 = minor unimportant detail differs, 5 = completely dissimilar; unannotated tokens are 0 = equivalent). For **asymmetric differences** (additions or omissions present on one side only), only the side containing the extra content is labelled; for **symmetric differences**, spans on both sides are labelled.

In the released dataset, token labels take six discrete values $\{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ (consistent with dividing the 1–5 span rating by 5), plus `-1` for punctuation and symbol tokens (`.`, `,`, `(`, `)`, `-`, `'`, `:`, `/`, `"`, `?`, `%`, etc.), which account for **12.9–13.8%** of all tokens and are ignored by the metric (`src/rsd/metrics.py:31-39`) and excluded from head training (`src/rsd/head.py:146-148`). Across the 504 dev documents (405,654 total words; 351,942 scored tokens with `gold != -1`), the label distribution is heavily zero-skewed:

| Language Pair | Total Tokens | Scored (`!= -1`) | `0.0` | `0.2` | `0.4` | `0.6` | `0.8` | `1.0` | Share `> 0` | Mean Gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **EN–DE** | 126,421 | 108,945 | 95,120 | 1,286 | 2,258 | 2,208 | 1,205 | 6,868 | 12.7% | 0.095 |
| **EN–FR** | 142,905 | 124,262 | 111,649 | 1,280 | 3,380 | 2,746 | 1,334 | 3,873 | 10.2% | 0.066 |
| **EN–IT** | 136,328 | 118,735 | 97,114 | 660 | 2,976 | 5,241 | 4,111 | 8,633 | 18.2% | 0.138 |

Because annotators mark phrases, clauses, or omitted paragraphs rather than isolated words, **21.3–36.0% of scored positive (`> 0`) tokens sit inside runs of $\ge 20$ consecutive positive scored tokens (and 67.2–74.9% in runs of $\ge 5$)**—a structural property of the labels that explains why spatial window smoothing helps every system tested (Section 5.3).

### 4.3 Splits, Lack of a Train Split, and `page_en` Grouping

- **Splits (`dev` and `test` only, no `train`):** SwissGov-RSD has 224 pages per language pair split into **dev** (`validation` on HF: 168 documents per pair, 504 total) and **test** (56 documents per pair, 168 total; 150,434 words). There is no separate training split (the paper's fine-tuned baselines trained on synthetic iSTS-RSD data). The official GitHub repository further divides the 168 dev pages per pair into **`dev/train` (134 pages/pair, 402 docs)** and **`dev/val` (34 pages/pair, 102 docs)**, which we use as a held-out development check alongside 5-fold cross-validation on all 504 dev documents.
- **Cross-pair page sharing (`page_en`):** Each document `admin_{lang}_{n}` pairs an English page `page_en` with its German, French, or Italian counterpart. **The exact same 168 English pages appear in all three dev language pairs** (and the 56 test document IDs likewise share page numbers across all three pairs, with zero overlap with dev). If page 12 were placed in a training fold for EN–DE and a validation fold for EN–FR, the model would have already seen the English document during training. Therefore, **every cross-validation fold (`GroupKFold(n_splits=5)`) and every page-level bootstrap resample is grouped by `page_en` across all three language pairs simultaneously** (`src/rsd/data.py:86-95`, `src/rsd/metrics.py:71-92`).

### 4.4 The EN–DE Double-Annotation Discrepancy

While wiring the `dev/val` evaluation (`src/official_eval/run.py:30-51`), we discovered a discrepancy between the main Hugging Face `dev` gold (which is also the gold used by the paper's released dev predictions) and the official GitHub `dev/train` and `dev/val` split files:
- In the main HF dev gold, each EN–DE document carries an `annotator_tag`: 90 documents from annotator 2 and 78 from annotator 1 (EN–FR and EN–IT carry no tag and are identical across all files).
- In the official `dev/train` and `dev/val` files, **89 of the 90 annotator-2 EN–DE documents hold a different label set** (inferred to be annotator 1's labels, since the 78 annotator-1 documents and 1 annotator-2 document match the main gold exactly).
- **Convention used in this report:** Every development and `dev/val` score we report is evaluated against the **main HF dev gold** (restricted to the `dev/val` IDs for `dev/val`), so that all numbers are mutually consistent and directly comparable to the paper's released predictions (`_resolve_split` in `src/official_eval/run.py`, `docs/ablation_dev.md`). In Section 5.4, we analyse the 89 double-annotated EN–DE documents directly to quantify inter-annotator noise.

### 4.5 Official Evaluation Script Quirks and Safeguards

Auditing the fetched official evaluator (`evaluate_predictions_admin.py`) before generating any predictions revealed four pitfalls that our wrapper (`src/official_eval/run.py`, `src/rsd/predictions.py`) guards against:
1. **Positional line matching:** The official script pairs predictions with gold **by line number**, not by `id`. `write_predictions` sorts outputs into exact gold ID order, and `official_eval.run.check_order` asserts 1-to-1 ID alignment before scoring.
2. **The `"llm"` filename substring trap:** The official repo's `list_to_drop.txt` lists 33 document IDs (15 in dev: pages 18, 100, 106, 153, 196 across all 3 pairs) dropped when scoring LLM prompting baselines. The official script triggers this filter **whenever the substring `"llm"` appears anywhere in the prediction file path**, dropping those rows from the gold list and crashing with a length mismatch on full-split predictions. `rsd.predictions.prediction_path` explicitly rejects any path containing `"llm"` (`tests/test_eval.py::test_prefix_with_llm_is_rejected`), while our prompting baseline subset excludes `list_to_drop.txt` documents as intended.
3. **Silent zero-padding and truncation:** The official script pads short prediction lists with `0.0` and truncates long ones; our local metric (`src/rsd/metrics.py:25-28`) and extractor assert exact token-length equality on every document.
4. **Dummy bootstrap output:** The official script's printed `±` figures come from `n_resamples=1` and are not confidence intervals; our wrapper suppresses them and computes 1,000-resample `page_en`-grouped bootstrap CIs (`src/rsd/metrics.py:71-117`).

## 5. Evaluation

### 5.1 Metric, Validation Protocol, and Single-Run Test Hygiene

- **Official metric (`src/rsd/metrics.py:16-69`, `src/official_eval/run.py`):** For each language pair $L \in \{\text{de}, \text{fr}, \text{it}\}$, all non-punctuation tokens (`gold != -1`) from side $A$ (English) and side $B$ ($L$) across all documents in the split are concatenated into a single flat vector of length $N_L$, and scored by global Spearman rank correlation $\rho_L \times 100$ (`scipy.stats.spearmanr`). The headline metric is the macro average $\bar{\rho} = (\rho_{\text{de}} + \rho_{\text{fr}} + \rho_{\text{it}})/3$.
- **Why global Spearman makes cross-document calibration matter:** Unlike sentence-level benchmarks that average correlations within each item, SwissGov-RSD ranks all ~110,000–124,000 tokens of a language pair together in one pool. Consequently, a model must not only order tokens within a page, but also assign higher baseline scores to heavily divergent pages than to nearly identical translations. Our head outputs raw logits $z = Xw + b$ mapped per language pair to $[0, 1]$ via a rank-preserving linear min-max (`unit_interval`), which leaves $\rho_L$ unchanged while satisfying the $[0, 1]$ range constraint.
- **Why Table 3 of Wastl et al. (2026) is not directly comparable:** The benchmark paper's Table 3 reports numbers over all **224 pairs per language (`dev` + `test` pooled)**. Because our head is trained on `dev`, we never pool `dev` and `test`. Instead, we rescore the paper's released full-document baseline prediction files (`data/evaluation/encoder_predictions/full/` in `ZurichNLP/SwissGov-RSD`) on `dev` (168/pair) and `test` (56/pair) separately using the exact same official evaluation script.
- **Development validation and uncertainty estimation:** Design choices were evaluated on `dev` using (1) **5-fold `page_en`-grouped cross-validation** (`GroupKFold(n_splits=5)` over the 168 shared English pages, with inner 4-fold grouped CV for $\lambda$ selection) and (2) the official **`dev/train` (134/pair) $\to$ `dev/val` (34/pair)** split. Only $\lambda$ is nested inside the outer CV folds; the layer range (12–18), smoothing windows (1, 3, 7, 15), and head configuration were chosen on dev (`docs/scalar_head_dev_results.md`), so the outer CV score (39.9) is slightly optimistic and `dev/val` (38.0) is the cleaner development estimate. Confidence intervals (`bootstrap_ci` and `paired_bootstrap` in `src/rsd/metrics.py:71-117`) use 1,000 resamples of the shared `page_en` units (seed 0) so that all three language pairs are resampled together. Because these bootstrap CIs hold the fitted model fixed and do not include training-set or configuration-selection variance, they are narrower than the true uncertainty (the per-fold gain range of +2.2 to +4.5 in Section 5.4 gives a fuller picture of training variance).
- **Single-run test hygiene (`docs/final_run.md`, `docs/test_results.md`):** All extraction and head code, features, and hyperparameters ($\lambda = 10^{-3}$, feature set `(e)`, shared head) were frozen at git tag `final-run` (`4c804d40e8f528613e2237204f37a277141a3342`) on **2026-10-14** (15:37 CEST) before any test access (`docs/final_run_meta.json`). Test extraction (`kaggle/04_extract_test.py`) was executed once on 2026-10-14 (899.5 s wall time, zero `NaN`s). The head was fitted once on all 504 dev documents and wrote test predictions to `data/predictions_test/apertus_head_*` (16:05:02) before the official test gold was first downloaded (16:05:07); the predictions were committed immediately afterwards (`57ab49b`, 16:05:45) and have not been modified since. Post-tag code changes were limited to infrastructure and docstrings: commit `895f88d` added a download-only `--test` flag to `src/official_eval/fetch.py` and called `python -m official_eval.fetch --test` in `Dockerfile` so `make run` requires no network at run time, and commit `ac3ed55` updated documentation and a docstring in `src/rsd/head.py`; none of these affect predictions. Code gates (`tests/test_eval.py::test_test_split_is_gated`, `tests/test_head.py::test_fit_predict_test_needs_final`, `tests/test_extract.py::test_test_split_needs_final`) raise `PermissionError` if any test path is called without `final=True` / `--final`, and `tests/test_head.py::test_features_take_no_gold` verifies that `rsd.features.build` cannot access gold labels.

### 5.2 Baselines: Released Models and Reproduced Apertus Recipes

We compare against two baseline groups: (1) the seven full-document DiffAlign systems from `data/evaluation/encoder_predictions/full/` in the paper's repository, rescored on `dev` (`docs/reference_dev_scores.md`) and `test` (`docs/test_results.md`; the paper's segmented `LaBSE` and `XLM-R + SimCSE` runs were not rescored), and (2) our own reproductions of the paper's two recipes—last-layer DiffAlign and 3-shot JSON prompting—applied to **Apertus v1.5 8B** on `dev` (`docs/unsupervised_dev_results.md`, `docs/prompting_baseline_dev_results.md`):

| Baseline System | Type | Dev DE | Dev FR | Dev IT | Dev Mean | Test DE | Test FR | Test IT | Test Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-Embedding-4B` | Released DiffAlign (best of 7 on dev) | 20.6 | 11.7 | 20.2 | **17.5** | 18.6 | 9.8 | 20.6 | **16.3** |
| `Qwen3-Embedding-8B` | Released DiffAlign (best of 7 on test) | 19.1 | 12.4 | 17.9 | **16.4** | 19.7 | 11.2 | 20.4 | **17.1** |
| `Qwen3-Embedding-8B +smooth7` | Released preds + 7-word window | 23.7 | 15.4 | 22.5 | **20.5** | — | — | — | — |
| `EuroBERT-610m` | Released DiffAlign | 18.6 | 11.8 | 18.0 | **16.1** | 18.3 | 10.3 | 20.5 | **16.4** |
| `EuroBERT-2.1B` | Released DiffAlign | 13.6 | 8.8 | 16.8 | **13.1** | 13.8 | 9.1 | 18.8 | **13.9** |
| `mmBERT-base` | Released DiffAlign | 14.7 | 8.8 | 15.3 | **12.9** | 13.2 | 8.0 | 16.5 | **12.6** |
| `embeddinggemma-300m` | Released DiffAlign | 14.6 | 9.7 | 13.4 | **12.6** | 15.1 | 10.0 | 16.4 | **13.9** |
| `bge-m3` | Released DiffAlign | 13.8 | 9.8 | 10.4 | **11.4** | 17.2 | 9.8 | 21.0 | **16.0** |
| **Apertus v1.5 8B `alone raw L32`** | Paper DiffAlign recipe (last layer) | 5.4 | 5.4 | 9.8 | **6.9** | — | — | — | — |
| **Apertus v1.5 8B `alone cent L32`** | Paper DiffAlign recipe (centred) | 4.4 | 3.5 | 8.2 | **5.4** | — | — | — | — |
| **Apertus v1.5 8B 3-shot prompting** | Paper LLM recipe (30 short docs) | 16.3 | -10.2 | 3.6 | **3.2** | — | — | — | — |

Among the seven rescored baselines, the strongest on `dev` is `Qwen3-Embedding-4B` (**17.5**), followed by `Qwen3-Embedding-8B` (**16.4**) and `EuroBERT-610m` (**16.1**); on `test`, `Qwen3-Embedding-8B` is first (**17.1**), followed by `EuroBERT-610m` (**16.4**) and `Qwen3-Embedding-4B` (**16.3**).

**Why the paper's recipes fail on a generative causal LLM:**
1. **Last-layer DiffAlign (`L32` = 6.9):** Applying the paper's last-layer cosine alignment recipe directly to Apertus (`alone raw L32`) yields **6.9** macro Spearman (5.4 `cent`). A likely reason is that at the final RMSNorm output immediately before `lm_head`, a causal decoder's representation of token $t$ is specialised for predicting the *next* subword $t+1$ in the current language rather than aligning the semantic identity of token $t$ across languages.
2. **Few-shot JSON prompting (`3.2` on 30 short documents, `kaggle/03_prompting_baseline.py`, `src/rsd/prompting.py`, `docs/prompting_baseline_dev_results.md`):** We reproduced the paper's official prompt template (`prompt_templates/template_{lang}_admin.txt`: 3 sentence-level iSTS-RSD examples, the 0–5 similarity scale, and `json.dumps` token lists passed through Apertus's chat template with deliberation disabled) using **greedy decoding** (`max_new_tokens` set to $1.5\times$ the length of a well-formed answer, capped at 4,096; parsed with the official parser using `ast.literal_eval` in place of `eval`). The subset (`rsd.prompting.select_docs`) takes dev documents not in `list_to_drop.txt` with at most **300 words** on both sides combined (26 DE, 18 FR, 24 IT eligible) and draws a seeded random sample (`default_rng(0)`) of **10 documents per pair** (30 total; median length **218 words** vs. 589–675 on full dev). All 30 generations stopped at the end-of-turn token without hitting the token cap, taking **4,036 s (~67 min of generation, ~2.2 min/doc)** and scoring **3.2** (no bootstrap CI was computed because the three 10-document subsets share few English pages and some resamples contain only all-zero unparsed documents, making Spearman undefined):
   - **10 of 30 outputs failed JSON parsing** (3 DE, 5 FR, 2 IT; malformed JSON lists such as dropped opening brackets near `\u`-escaped tokens, unlabelled punctuation, or unbalanced brackets), falling back to all-zero predictions;
   - **29 of the 40 parsed sides had the wrong entry count**, leaving 851 of 4,414 tokens (19%) unaligned (assigned fallback `0.0`); and
   - **64.5% of all 6,700 tokens shared the single value `0.0`** (across only 7 distinct output levels).
   On the exact same 30 documents, `Apertus alone raw L32` scores **6.3** (`12.3 / 11.7 / -5.2`), released `Qwen3-Embedding-8B` scores **17.9** (**22.0** with `+smooth7`), and reading Apertus's layer-14 states takes ~1 second per pass and scores **36.0** (`alone raw L14`: `43.9 / 32.2 / 32.0`) and **42.5** (`cond raw L14 +smooth7`: `50.5 / 39.5 / 37.6`).

### 5.3 Unsupervised Apertus Signals and Controlled Layer-Depth Analysis

Before fitting any parameters, we evaluated 276 training-free signal variants extracted from Apertus across all 504 dev documents (`src/rsd/unsupervised.py`, `docs/unsupervised_dev_results.md`):

| Unsupervised Signal (all 504 dev docs) | EN–DE | EN–FR | EN–IT | Dev Mean |
|---|---:|---:|---:|---:|
| DiffAlign `alone raw L32` (last layer, paper recipe) | 5.4 | 5.4 | 9.8 | **6.9** |
| DiffAlign `alone cent L32` (last layer, centred) | 4.4 | 3.5 | 8.2 | **5.4** |
| DiffAlign `alone raw L14` (best unconditioned raw layer) | 33.0 | 21.4 | 35.7 | **30.0** |
| DiffAlign `alone cent L15` (best unconditioned centred layer) | 33.0 | 21.1 | 36.5 | **30.2** |
| DiffAlign `alone raw L14 +smooth7` | 37.5 | 24.8 | 41.0 | **34.4** |
| DiffAlign `alone cent L14 +smooth7` | 37.0 | 24.1 | 41.2 | **34.1** |
| DiffAlign `cond raw L14` (cross-lingually conditioned) | 35.6 | 23.2 | 38.3 | **32.4** |
| DiffAlign `cond cent L17` (best conditioned centred layer) | 34.6 | 22.0 | 37.9 | **31.5** |
| DiffAlign `cond cent L15 +smooth7` | 38.4 | 24.6 | 42.2 | **35.1** |
| **DiffAlign `cond raw L14 +smooth7` (best single unsupervised)** | **39.8** | **26.3** | **43.2** | **36.4** |
| Contrastive surprisal `-gain sum` ($\text{lp}_{\text{alone}} - \text{lp}_{\text{cond}}$) | 22.2 | 13.4 | 23.0 | **19.6** |
| Contrastive surprisal `-gain sum +smooth7` | 31.4 | 18.1 | 32.0 | **27.2** |
| Contrastive surprisal `-gain first` | 21.3 | 12.7 | 21.8 | **18.6** |
| Contrastive surprisal `-gain first +smooth7` | 31.1 | 17.7 | 31.1 | **26.7** |
| Conditioned $-\text{lp}_{\text{cond}}\text{ sum}$ | 20.7 | 11.6 | 20.2 | **17.5** |
| Conditioned $-\text{lp}_{\text{cond}}\text{ sum} + \text{smooth7}$ | 29.7 | 18.1 | 29.9 | **25.9** |
| Unconditioned $-\text{lp}_{\text{alone}}\text{ sum}$ (lexical rarity control, no access to $Y$) | 0.8 | -0.2 | -0.5 | **0.0** |
| Control: subwords per word | 4.6 | 2.0 | 1.3 | **2.6** |
| Control: word position | 4.0 | -3.9 | 2.2 | **0.8** |

**DiffAlign layer curve across all 33 hidden states (dev macro mean, unsmoothed, `docs/unsupervised_dev_results.md`):**

| Layer | `0` | `2` | `4` | `6` | `8` | `10` | `12` | `14` | `16` | `18` | `20` | `22` | `24` | `26` | `28` | `30` | `31` | `32` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `alone raw` | 9.1 | 11.4 | 17.5 | 20.6 | 24.8 | 26.6 | 28.9 | **30.0** | 29.2 | 28.0 | 26.2 | 23.8 | 20.9 | 18.2 | 14.4 | 12.1 | 10.2 | 6.9 |
| `alone cent` | 9.2 | 13.9 | 18.8 | 21.4 | 24.8 | 26.9 | 28.6 | **30.2** | 29.9 | 29.1 | 27.8 | 26.0 | 23.8 | 21.8 | 18.7 | 15.9 | 14.1 | 5.4 |
| `cond raw` | 9.1 | 11.3 | 17.0 | 20.1 | 24.3 | 27.4 | 30.2 | **32.4** | 31.6 | 30.7 | 28.8 | 25.7 | 22.8 | 18.0 | 12.0 | 10.1 | 8.3 | 4.4 |
| `cond cent` | 9.2 | 13.2 | 18.0 | 20.6 | 23.9 | 26.3 | 28.7 | **31.4** | 31.1 | 30.6 | 29.5 | 27.9 | 25.6 | 23.3 | 19.1 | 15.6 | 13.7 | 5.4 |

Four findings emerge from these unsupervised experiments:
1. **Layer curve and selection check (`docs/unsupervised_dev_results.md`):** Performance starts at **9.1** on subword embeddings (`L0`), rises through layers 2–10, forms a flat peak across **middle layers 12–18** (`L14` = **30.0** `alone raw` and **32.4** `cond raw`; official spot-checks on `cond raw +smooth7` give `39.8 / 26.3 / 43.2 = 36.4` at `L14`, `39.5 / 26.2 / 42.9 = 36.2` at `L15`, and `39.2 / 25.7 / 42.5 = 35.8` at `L16`), declines from layer 22 onward (`L24 cond raw` = **22.8**), and collapses at `L32` (`alone raw` = **6.9**, `cond raw` = **4.4**). Mean-centring (`cent`) changes little at the peak layers (30.2 vs. 30.0 `alone`; 31.4 vs. 32.4 `cond`), **but helps substantially from layer 22 onward** (e.g. `L24 cond`: 25.6 `cent` vs. 22.8 `raw`; `L28 cond`: 19.1 `cent` vs. 12.0 `raw`). Because picking the best layer and variant on all of dev is slightly optimistic, we repeated the selection on `dev/train` (134 pages/pair) and scored on `dev/val` (34 pages/pair): `dev/train` selects the exact same top variant (`cond raw L14 +smooth7`, scoring 36.8 on `dev/train` and **34.4** on `dev/val`), and picks `L14` or `L15` in every family.
2. **Cross-lingual conditioning (+2.4 points) and within- vs. across-document signal:** Placing the other document in the causal prefix (`cond_X`) improves `raw L14` from **30.0 to 32.4** and `+smooth7` from **34.4 to 36.4** (paired bootstrap over released `Qwen3-Embedding-8B`: **+20.0** [+18.3, +21.7], or **+15.9** [+14.4, +17.4] when Qwen is also smoothed). Roughly half of the global Spearman score is document-level: replacing each word's prediction with its document side's mean score retains **18.7** of the 36.4, whereas shuffling predictions within each document side retains **9.0**; at the same time, within-document ranking alone (mean per-document Spearman `31.4 / 23.6 / 35.3`, mean **30.1**) is more than double released `Qwen3-Embedding-8B` (`14.8 / 9.7 / 13.3`, mean **12.6**).
3. **Contrastive surprisal vs. lexical rarity:** Unconditioned negative log-probability ($-\text{lp}_{\text{alone}}\text{ sum}$) scores **0.0** (`0.8 / -0.2 / -0.5`), and trivial subword-count (2.6) and word-position (0.8) controls also score near zero. Conditioning on the other document raises $-\text{lp}_{\text{cond}}\text{ sum}$ to **17.5** raw (**25.9** smoothed), and subtracting the unconditioned baseline to form contrastive surprisal (`-gain sum` $= \text{lp}_{\text{alone}} - \text{lp}_{\text{cond}}$) reaches **19.6** raw and **27.2** smoothed (`-gain first +smooth7` = **26.7**).
4. **Controlled all-layer comparison with `Qwen/Qwen3-Embedding-8B` (`docs/qwen_control_dev_results.md`):** Because the paper's `Qwen3-Embedding-8B` baseline (16.4) used the final layer (`L36`) without smoothing, comparing it directly to middle-layer Apertus confounds layer depth with backbone choice. We therefore ran `Qwen/Qwen3-Embedding-8B` (commit `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`) across all 37 hidden states (`0..36`) on all 504 dev documents following the official unconditioned (`alone`) DiffAlign recipe (`kaggle/02_qwen_control.py`, `src/rsd/qwen_control.py`):

| Variant on Dev (504 docs) | EN–DE | EN–FR | EN–IT | Dev Mean |
|---|---:|---:|---:|---:|
| `Qwen3-Embedding-8B` `alone raw L36` (our run, official last-layer recipe) | 19.1 | 12.4 | 17.9 | **16.4** |
| `Qwen3-Embedding-8B` `alone cent L36` | — | — | — | **16.8** |
| `Qwen3-Embedding-8B` `alone raw L21` (best Qwen layer, 58% depth) | 30.6 | 19.7 | 33.0 | **27.8** |
| `Qwen3-Embedding-8B` `alone cent L21` (best Qwen centred layer) | — | — | — | **25.6** |
| `Qwen3-Embedding-8B` `alone raw L21 +smooth7` (best Qwen smoothed) | 36.4 | 23.9 | 40.1 | **33.5** |
| `Qwen3-Embedding-8B` `alone cent L24 +smooth7` (best Qwen centred smoothed) | — | — | — | **31.2** |
| **Apertus v1.5 8B** `alone raw L32` (last layer) | 5.4 | 5.4 | 9.8 | **6.9** |
| **Apertus v1.5 8B** `alone raw L14` (best Apertus layer, 44% depth) | 33.0 | 21.4 | 35.7 | **30.0** |
| **Apertus v1.5 8B** `alone raw L14 +smooth7` | 37.5 | 24.8 | 41.0 | **34.4** |
| **Apertus v1.5 8B** `cond raw L14 +smooth7` (our best unsupervised) | 39.8 | 26.3 | 43.2 | **36.4** |

| Paired Bootstrap Comparison on Dev (1,000 page resamples, seed 0) | EN–DE | EN–FR | EN–IT | Mean $\Delta$ [95% CI] |
|---|---:|---:|---:|---:|
| `Qwen alone raw L21` $-$ `Qwen L36` (layer effect inside Qwen) | +11.6 [+9.7, +13.7] | +7.3 [+5.7, +9.1] | +15.1 [+13.4, +17.2] | **+11.3** [+10.1, +12.7] |
| `Qwen alone raw L21 +smooth7` $-$ `Qwen L36 +smooth7` | +12.8 [+10.6, +14.8] | +8.5 [+6.6, +10.4] | +17.6 [+15.6, +19.7] | **+12.9** [+11.6, +14.3] |
| `Apertus alone raw L14` $-$ `Qwen alone raw L21` (same recipe, raw) | +2.3 [+1.7, +2.9] | +1.7 [+1.1, +2.3] | +2.6 [+2.0, +3.2] | **+2.2** [+1.8, +2.6] |
| `Apertus alone raw L14 +smooth7` $-$ `Qwen alone raw L21 +smooth7` | +1.1 [+0.3, +1.8] | +0.9 [+0.1, +1.7] | +0.9 [+0.2, +1.6] | **+1.0** [+0.5, +1.4] |
| `Apertus cond raw L14 +smooth7` $-$ `Qwen alone raw L21 +smooth7` | +3.4 [+2.7, +4.2] | +2.4 [+1.8, +3.1] | +3.1 [+2.4, +3.8] | **+3.0** [+2.5, +3.5] |

   - **Pipeline verification:** Our `L36 raw` run reproduces the paper's released `Qwen3-Embedding-8B` dev predictions (`19.1 / 12.4 / 17.9`, mean **16.4**), with per-token Pearson and Spearman correlation of **1.0000** against the released files on all three pairs (mean absolute difference 0.00024–0.00025 from `fp16` vs. `fp32`).
   - **Layer depth inside Qwen (+11.3 points):** Even in a dedicated embedding model trained with contrastive supervision at its final layer, per-token cross-lingual alignment peaks in the middle (`L21` at **27.8** vs. **16.4** at `L36`, paired difference **+11.3** [+10.1, +12.7]; choosing on `dev/train` also picks `L21`, scoring 25.3 on `dev/val` vs. 27.4 for Apertus `L14`). Thus **about 11 of the 13.6 points separating Apertus `L14` (30.0) from the paper's Qwen baseline (16.4) come from reading a middle layer rather than the last layer**.
   - **Controlled backbone margin (+2.2 raw, +1.0 smoothed):** Under the same unconditioned recipe at each model's best layer, Apertus v1.5 8B is ahead of `Qwen3-Embedding-8B` by **+2.2** [+1.8, +2.6] raw and **+1.0** [+0.5, +1.4] with 7-word smoothing. Adding causal cross-lingual conditioning (`cond raw L14 +smooth7`) widens the margin over Qwen's best smoothed layer to **+3.0** [+2.5, +3.5].

### 5.4 Supervised Scalar Head, Ablations, and Diagnostics

To combine DiffAlign across layers 12–18 with contrastive surprisal and document-level context, we fit an L2-regularised logistic regression head (`src/rsd/head.py`, `src/rsd/features.py`, `docs/scalar_head_dev_results.md`) on the 96 scalar features of feature set `(e)` (all oriented so that higher values indicate stronger semantic difference):
- **80 multi-window local token features ($k \in \{1, 3, 7, 15\}$ within-side centred moving averages):** `da cond raw L14` (4), `-gain sum` (4), `-gain first` (4), `-logp_cond sum` (4), `-logp_alone sum` (4), `da alone raw L{12,14,16,18}` (16), `da alone cent L{12,14,16,18}` (16), `da cond raw L{12,16,18}` (12), and `da cond cent L{12,14,16,18}` (16).
- **12 document-level signal aggregates:** `side_mean` (mean on the current document side), `other_mean` (mean on the paired document side), and `pair_mean` (length-weighted mean over both sides) for four core signals: `da cond raw L14`, `-gain sum`, `da cond cent L14`, and `da alone raw L14`.
- **4 layout features:** `log len side` ($\log(n_{\text{side}})$), `log len ratio` ($\log(n_{\text{side}}/n_{\text{other}})$), `side is B`, and `rel position` ($i / \max(n_{\text{side}}-1, 1)$). (`log len other` is omitted as an exact linear combination of the first two.)

Features are z-scored using training-split statistics (`gold != -1` words only) and clipped to $[-5, 5]$ to tame extreme log-probabilities. Weights $(w, b)$ are fitted by minimising soft-label binary cross-entropy with $y_i \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ plus L2 penalty $\frac{\lambda}{2}\|w\|_2^2$ on the weights ($\lambda \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ selected by inner 4-fold `page_en`-grouped CV) via `scipy` L-BFGS-B. Raw logits $z = Xw + b$ are mapped per language pair into $[0, 1]$ with `unit_interval`.

**Feature-set ablation across both validation protocols (`docs/scalar_head_dev_results.md`):**
- `(b)` (11 features): `da cond raw L14` at 4 windows + its 3 doc aggregates + 4 layout features.
- `(c)` (30 features): `(b)` + `-gain sum`, `-gain first`, `-logp_cond sum`, `-logp_alone sum` at 4 windows + 3 doc aggregates of `-gain sum`.
- `(d)` (77 features): `(b)` + DiffAlign `{alone, cond} x {raw, cent} x L{12,14,16,18}` at 4 windows + doc aggregates of `cond cent L14` and `alone raw L14`.
- `(e)` (96 features): all features in `(c)` and `(d)` combined.

| Configuration | Features | Nested 5-Fold CV (DE / FR / IT) | CV Mean | Paired $\Delta$ vs. `(a)` (CV) | `dev/train` $\to$ `dev/val` (DE / FR / IT) | `dev/val` Mean | Paired $\Delta$ vs. `(a)` (`dev/val`) |
|---|---:|---|---:|---|---:|---:|---|
| `(a)` Unsupervised `cond raw L14 +smooth7` | 1 | 39.8 / 26.3 / 43.2 | **36.4** | — | 33.9 / 29.3 / 40.1 | **34.4** | — |
| `(b)` L14 `cond` + doc aggregates, shared | 11 | 40.9 / 26.9 / 44.7 | **37.5** | +1.1 [+0.9, +1.3] | 35.0 / 29.9 / 41.5 | **35.5** | +1.0 [+0.6, +1.6] |
| `(b)` L14 `cond` + doc aggregates, per-pair | 11 | 40.3 / 26.8 / 44.6 | **37.2** | +0.8 [+0.5, +1.2] | 34.6 / 29.9 / 40.5 | **35.0** | +0.5 [-0.4, +1.8] |
| `(c)` = `(b)` + log-prob / `-gain`, shared | 30 | 42.7 / 28.1 / 47.1 | **39.3** | +2.8 [+2.5, +3.3] | 36.7 / 31.2 / 43.4 | **37.1** | +2.7 [+1.9, +3.7] |
| `(c)` = `(b)` + log-prob / `-gain`, per-pair | 30 | 42.3 / 28.0 / 46.8 | **39.0** | +2.6 [+2.2, +3.1] | 36.7 / 30.7 / 42.5 | **36.6** | +2.2 [+0.9, +3.7] |
| `(d)` = `(b)` + DiffAlign L12–18, shared | 77 | 41.8 / 28.3 / 45.7 | **38.6** | +2.1 [+1.7, +2.6] | 37.9 / 30.9 / 42.6 | **37.1** | +2.7 [+1.4, +3.9] |
| `(d)` = `(b)` + DiffAlign L12–18, per-pair | 77 | 41.3 / 29.0 / 46.0 | **38.8** | +2.3 [+1.7, +3.0] | 37.5 / 30.6 / 41.8 | **36.6** | +2.2 [+0.6, +4.1] |
| **`(e)` All features, shared (submitted final)** | **96** | **43.2 / 29.1 / 47.5** | **39.9** | **+3.5 [+3.0, +3.9]** | **38.6 / 31.8 / 43.7** | **38.0** | **+3.6 [+2.4, +5.0]** |
| `(e)` All features, per-pair (rule's pick) | 96 | 42.9 / 29.5 / 47.5 | **40.0** | +3.5 [+3.0, +4.1] | 38.6 / 31.5 / 42.9 | **37.6** | +3.2 [+1.6, +5.2] |

**Where the supervised gain comes from (`docs/scalar_head_dev_results.md`):**
- **By feature group (shared CV means):** Multi-window calibration and document aggregates of the L14 signal (`(b)` vs. `(a)`) add **+1.1** points; adding the log-probability and `-gain` signals (`(c)` vs. `(b)`) adds **+1.8** points; adding the remaining DiffAlign layers (12, 16, 18) and `alone`/`cent` variants (`(d)` vs. `(b)`) adds **+1.1** points (+1.6 per-pair). Combining both groups in `(e)` adds **+2.4** over `(b)` (**+3.5** over `(a)`), showing that the two feature families overlap partially while each contributes signal the other lacks. Across the 5 outer CV folds, `(e)` beats the unsupervised baseline on every fold (**+3.9, +3.4, +4.5, +2.2, +4.3** macro Spearman).
- **Within-document ranking vs. across-document calibration:** Evaluated on the out-of-fold predictions of `(e)` per-pair, mean per-document Spearman rises from **30.1** (`31.4 / 23.6 / 35.3`) for the baseline to **34.0** (`34.1 / 27.4 / 40.4`), a gain of **+3.9** points, whereas replacing each word's prediction with its document side's mean score rises from **18.7** (`22.6 / 12.6 / 21.0`) to **19.9** (`24.3 / 13.1 / 22.3`), a gain of **+1.2** points. Contrary to our initial expectation that the head would help mainly via cross-document calibration, most of the improvement comes from sharper token ranking *within* documents.
- **Learned weights:** In the shared `(e)` model ($\lambda = 10^{-3}$) fitted on all of dev, the widest smoothing window ($k = 15$) receives the largest sum of absolute weights (**3.8**, vs. **2.6** for $k = 7$, **2.2** for $k = 3$, **1.8** for unsmoothed $k = 1$, and **2.4** across all document aggregates), and `da cond raw L14 s15` has the largest individual weight (**+0.730** in the shared model, and `+1.70 / +1.37 / +1.73` in the per-pair models).

**Final configuration choice and disclosed post-hoc switch (`docs/scalar_head_dev_results.md`):**
Our pre-stated selection rule (best inner 4-fold CV mean on `dev/train`) picked `(e)` per-pair over `(e)` shared by 0.1 points (**40.5** vs. **40.4**). After seeing the dev results, we switched to **`(e)` shared ($\lambda = 10^{-3}$)** for the final run. We disclose this as a post-hoc switch on a virtual tie, made for four reasons:
1. The two configurations essentially tie in nested outer CV (**39.9** shared vs. **40.0** per-pair).
2. Shared is stronger on `dev/val` (**38.0** vs. **37.6**), and its `dev/val` paired bootstrap CI on EN–FR (`[+0.6, +5.0]`) excludes zero whereas per-pair's (`[-0.5, +5.1]`) does not.
3. Shared selects $\lambda = 10^{-3}$ in all 5 outer folds and on `dev/train`, whereas for `(e)` per-pair $\lambda$ moves between $10^{-3}$ and $10^{-4}$ across folds ($10^{-3}$ once, $10^{-4}$ four times; for `(b)` and `(c)` per-pair it ranges from $10^{-2}$ to $10^{-5}$).
4. Shared uses one-third as many parameters (96 weights + 1 intercept vs. $3 \times 97$).
Because of this switch, `dev/val` is no longer strictly untouched for the shared-vs.-per-pair choice. Note also that we did not train a head on the 4,096-dimensional hidden states ("states head", dropped due to overfitting risk on 134 pages and time constraints) or test non-linear heads.

**Sanity controls against pipeline artefacts (`docs/scalar_head_dev_results.md`):**
An independent reviewer ran two label-corruption controls with its own standalone script using non-nested 5-fold `page_en`-grouped CV (`(e)` shared, $\lambda = 10^{-3}$; reviewer script not in the repo, re-verified on dev):
1. **Shuffled gold labels** (training on randomly permuted gold labels across scored words, scored against real gold): CV collapses to **2.3** (**3.2** on a later re-run with seed 0), confirming that the feature builder, fold splitter, and evaluator do not create spurious signal.
2. **Document-side mean training labels** (training on the mean gold score of each word's document side, removing within-document positional targets, then scoring against real gold): CV drops from **39.9** to **21.0** (for reference, a model-free oracle that assigns every word its true gold document-side mean scores `31.2 / 26.2 / 33.3`, mean **30.3**). Because a head trained only on document-level label means stops at 21.0, the 39.9 score requires learning from word-level annotations within documents.

**EN–DE double-annotation comparison on 89 dev documents (`src/rsd/annotators.py`, `docs/ablation_dev.md`):**
On the 89 `annotator_tag = 2` EN–DE dev documents where the main HF dev gold (annotator 2) and the official `dev/train` + `dev/val` split files (inferred annotator 1) hold different label sets, we computed global token-level Spearman over the **60,300 tokens** where neither annotator marks `-1` (with 1,000-resample document bootstrap CIs; head row uses the cached out-of-fold `(e)` per-pair predictions in `cache/scalar_head_preds/`):

| Comparison on 89 Double-Annotated EN–DE Dev Documents (60,300 tokens) | Spearman $\times 100$ [95% CI] |
|---|---:|
| Annotator 1 vs. Annotator 2 | **18.9** [10.6, 28.3] |
| Unsupervised `cond raw L14 +smooth7` vs. Annotator 1 | **20.5** [15.5, 25.6] |
| Unsupervised `cond raw L14 +smooth7` vs. Annotator 2 (main dev gold) | **38.9** [33.5, 44.2] |
| Scalar head `(e)` per-pair (out-of-fold) vs. Annotator 1 | **21.5** [16.2, 26.8] |
| Scalar head `(e)` per-pair (out-of-fold) vs. Annotator 2 (main dev gold) | **41.5** [35.9, 47.1] |

- **What this shows:** Both the training-free baseline and the supervised head score roughly twice as high against Annotator 2 as against Annotator 1 (**38.9 vs. 20.5** for the baseline, **41.5 vs. 21.5** for the head) on the exact same 89 documents, and the head improves over the baseline under either label set (+1.0 vs. Annotator 1, +2.6 vs. Annotator 2). A plausible reason for the large annotator gap is noted in Appendix A.4 of Wastl et al. (2026): on EN–DE, Annotator 2 labels whole phrases whereas Annotator 1 labels minimal spans; if the annotator numbering matches `annotator_tag`, our 7–15 word smoothing naturally aligns better with phrase-level spans.
- **Why 18.9 is not a human ceiling:** In the benchmark paper (Table 2), inter-annotator Spearman on the 25 overlapping main-phase documents per pair is **56.14** for EN–DE (**43.61** EN–FR, **65.09** EN–IT), and footnote 5 notes that an initial EN–DE annotation run had low agreement (9.4) and was repeated with a new annotator. We therefore do not quote 18.9 as the human agreement of SwissGov-RSD, nor do we claim to exceed a human ceiling; rather, it measures the disagreement between the two EN–DE label sets shipped in the repository and illustrates that annotator variance on this task is large relative to small differences between models.

### 5.5 Single-Run Test Results

In our single final run (`docs/final_run.md`, `docs/test_results.md`), `rsd.head fit-predict` fitted the shared `(e)` head ($\lambda = 10^{-3}$) on all 504 dev documents, predicted the 168 test documents once (`data/predictions_test/apertus_head_*`), and scored them with `uv run python -m official_eval.run data/predictions_test/apertus_head_ --final` and 1,000-resample document-level bootstrap (`rsd.metrics.bootstrap_ci`). For comparison, the seven released DiffAlign baseline prediction files from the official repository (`data/evaluation/encoder_predictions/full/` at commit `1807a42`) were rescored on the test rows with the same official script (`final=True`) and `bootstrap_ci` by a one-off script:

| System (Test Split, 56 docs / pair, 168 total) | EN–DE | EN–FR | EN–IT | Test Macro Mean [95% CI] |
|---|---:|---:|---:|---:|
| **Apertus v1.5 8B 96-feature scalar head (ours, shared)** | **42.8** [33.3, 51.7] | **23.4** [18.2, 28.2] | **46.0** [37.1, 55.2] | **37.4** [31.9, 42.4] |
| DiffAlign `Qwen3-Embedding-8B` (best of 7 rescored baselines on test) | 19.7 | 11.2 | 20.4 | **17.1** [13.7, 20.5] |
| DiffAlign `EuroBERT-610m` | 18.3 | 10.3 | 20.5 | **16.4** [13.2, 19.3] |
| DiffAlign `Qwen3-Embedding-4B` | 18.6 | 9.8 | 20.6 | **16.3** [12.8, 20.1] |
| DiffAlign `bge-m3` | 17.2 | 9.8 | 21.0 | **16.0** [9.2, 21.4] |
| DiffAlign `EuroBERT-2.1B` | 13.8 | 9.1 | 18.8 | **13.9** [10.4, 17.1] |
| DiffAlign `embeddinggemma-300m` | 15.1 | 10.0 | 16.4 | **13.9** [10.2, 17.3] |
| DiffAlign `mmBERT-base` | 13.2 | 8.0 | 16.5 | **12.6** [9.8, 15.7] |

For our head, the official script and local metric match on every pair (unrounded local Spearman: `0.4282 / 0.2343 / 0.4600`; official Kendall $\tau_b$: `0.349 / 0.190 / 0.374`; 56 documents read per pair, zero dropped, no warnings). Note that these bootstrap CIs resample test documents with the fitted head held fixed; they do not include the uncertainty from fitting the head on dev or selecting the configuration, and are therefore narrower than the full uncertainty (`docs/test_results.md`).

**How test compares to dev, and the EN–FR drop (`docs/test_results.md`):**
Our test macro mean (**37.4**) is 0.6 points below `dev/val` (**38.0**) and 2.5 points below nested 5-fold CV (**39.9**), and the test CI `[31.9, 42.4]` contains both dev estimates. On EN–DE (**42.8** vs. 43.2 CV) and EN–IT (**46.0** vs. 47.5 CV), test performance is close to nested CV. On **EN–FR**, however, the test score (**23.4**, 95% CI `[18.2, 28.2]`) is 5.7 points below nested CV (**29.1**) and 8.4 points below `dev/val` (**31.8**), and its test CI does not reach the CV value. The released baselines also drop slightly on EN–FR from dev to test (`Qwen3-8B` 12.4 $\to$ 11.2, `Qwen3-4B` 11.7 $\to$ 9.8, `EuroBERT-610m` 11.8 $\to$ 10.3) while rising on EN–IT, indicating that the 56 test pages differ somewhat by pair; however, our EN–FR drop (~6 points from CV) is larger than the baselines' (~1–2 points), likely combining sample variance on 56 documents with mild dev selection optimism. To honour the test-once rule, we did not inspect test documents to diagnose the EN–FR drop.

### 5.6 Qualitative Error Analysis on Rule-Selected Dev Documents

To inspect token-level behaviour without cherry-picking, `src/rsd/heatmap.py` deterministically selects four dev documents (`docs/heatmaps/index.html`, limited to $\le 800$ words per side, preferring unused language pairs) and visualises the **gold label** alongside our two core unsupervised signals—**`DiffAlign cond raw L14 +smooth7`** and **`-gain sum +smooth7`**—expressed as **pair-wide percentiles** across all scored dev tokens of that language pair (note: the heatmaps display these two unsupervised input signals, not the final head's combined output):

| Document | Pair | Words (EN / Other) | Scored Tokens | Gold Mean (Overall; EN / Other) | Selection Rule (`src/rsd/heatmap.py`) | Within-Doc $\rho$: DiffAlign / `-gain` | Mean Pair Percentile: DiffAlign / `-gain` |
|---|---|---:|---:|---|---|---:|---:|
| `admin_de_127` | EN–DE | 201 / 656 | 771 | 0.615 (0.130 / 0.766) | `high gold mean` | 0.812 / 0.787 | 0.84 / 0.67 |
| `admin_fr_159` | EN–FR | 384 / 403 | 697 | 0.002 (0.001 / 0.003) | `gold mean near zero` | 0.099 / -0.017 | 0.37 / 0.35 |
| `admin_it_9` | EN–IT | 501 / 718 | 1,093 | 0.248 (0.066 / 0.373) | `long contiguous gold span` ($\ge 30$) | 0.665 / 0.534 | 0.63 / 0.55 |
| `admin_it_209` | EN–IT | 170 / 198 | 319 | 0.167 (0.167 / 0.167) | `DiffAlign does badly despite non-trivial gold` | 0.112 / 0.257 | 0.66 / 0.56 |

1. **`admin_de_127` (EN–DE, SFOE ProKilowatt tenders page, within-doc $\rho = 0.812$ DiffAlign / $0.787$ `-gain`):**
   - **Long asymmetric additions caught cleanly:** The German side is more than three times as long as the English side (656 vs. 201 words) because starting at `DE[149]` (`"Medien"`) it lists dozens of historical press release headlines omitted in English (`DE[150..488]`, 339 tokens of which 302 are scored with `gold = 1.0` and 37 are `-1` punctuation, and `DE[497..646]`, 150 tokens of which 138 are scored with `gold = 1.0` and 12 are `-1` punctuation, e.g. *"Wettbewerbliche Ausschreibungen zum Stromsparen : Ausschreibungen 2024 mit positiver Bilanz..."*). Both `DiffAlign cond raw L14 +smooth7` (mean percentile **0.99**) and `-gain sum +smooth7` (mean percentile **0.91**) rank these untranslated blocks at the very top of the EN–DE pair.
   - **Smoothing bleed and annotation gaps:** Between those two `gold = 1.0` blocks sits a single headline `DE[489..496]` (*"Achte Runde der Wettbewerblichen Ausschreibungen zum Stromsparen gestartet"*, 8 tokens) left at `gold = 0` in the reference annotation even though it belongs to the same untranslated media list; both signals rank it high (DiffAlign pct `0.99–1.00`, `-gain` pct `0.77–0.87`). On the English side, `EN[51..54]` (`"much electricity as possible"`, `gold = 0`) sits between `EN[46..47]` (`"specified requirements"`, `gold = 0.4`) and `EN[55..58]` (`"per provided Swiss franc"`, `gold = 0.4`), and the 7-word smoothing window pulls `EN[51..54]` up to the `0.91–0.94` DiffAlign percentile.

2. **`admin_fr_159` (EN–FR, SFOE battery research page, gold mean $0.002$, within-doc $\rho = 0.099$ / $-0.017$):**
   - **Near-identical page and cross-document calibration:** Out of 697 scored tokens, only 4 tokens are annotated as divergent: `EN[275]` (`"salt"`, 1 scored token, `gold = 0.4`) vs. `FR[283..285]` (`"batteries au sel"`, 3 scored tokens, `gold = 0.4`, where French repeats *"batteries"* inside parentheses). In addition, the opening announcement (`EN[0..96]` and `FR[0..96]`, *"Batteries The M-ERA.NET Call 2024 was launched..."*) appears in English on both pages; DiffAlign assigns near-zero percentiles (`0.00–0.03`) across `EN[0..91]` and `FR[0..91]` before rising at `#92..#96` due to the 7-word smoothing window crossing into the English/French body at `#97`.
   - **False positives on non-literal syntactic restructuring:** With almost no positive gold tokens, within-document Spearman is near zero (`0.099` / `-0.017`), and the errors that matter for global Spearman are false-positive spikes on `gold = 0` clauses that rephrase the same meaning—such as `EN[324..329]` (`"conjunction with its promotion options in"`, `gold = 0`, DiffAlign pct `0.87–0.93`, `-gain` pct `0.96–0.98`) vs. `FR[342..344]` (`"ainsi qu’au travers"`, `gold = 0`, DiffAlign pct `0.83–0.85`, `-gain` pct `0.99–1.00`). Overall, the page's mean percentile remains below the pair midpoint (`0.37` DiffAlign, `0.35` `-gain`).

3. **`admin_it_9` (EN–IT, Federal Gas Supply Act page, within-doc $\rho = 0.665$ DiffAlign / $0.534$ `-gain`):**
   - **Long untranslated link/press-release blocks vs. subtle body differences:** Like `admin_de_127`, the Italian page appends a long block of press-release and legal links (`IT[470..706]`, 212 positive tokens with `gold = 1.0` across 5 spans totaling 233 tokens including punctuation) absent in English, which both DiffAlign (mean pct `0.95–0.98`) and `-gain` (`0.73–0.91`) separate cleanly from the body.
   - **Complementary behaviour of `-gain` and DiffAlign:** In the body, `EN[178..182]` (`"the Associations Agreement on Gas"`, `gold = 0.6`) corresponds to `IT[177..183]` (`"una convenzione sull’accesso alla rete del gas"`, *"an agreement on access to the gas network"*, `gold = 0.6`). DiffAlign assigns moderate percentiles (mean `0.61` EN / `0.43` IT), whereas `-gain` fires strongly on both sides (mean `0.97` EN / `0.78` IT) because *"Associations"* and *"sull'accesso alla rete"* do not predict each other. Conversely, on `EN[90..95]` (`"Swiss Competition Commission ( COMCO )"`, `gold = 0.4`) vs. `IT[94..99]` (`"Commissione della concorrenza ( COMCO )"`, `gold = 0.4`, annotated because Italian omits *"Swiss"*), both signals under-score the span (DiffAlign mean pct `0.17` EN / `0.29` IT) because the shared acronym `COMCO` and standard institutional name align strongly.

4. **`admin_it_209` (EN–IT, OFCOM / UFCOM profile page, within-doc $\rho = 0.112$ DiffAlign / $0.257$ `-gain`):**
   - **Why DiffAlign fails (`0.112`) while `-gain` is more than twice as high (`0.257`):** This short page contains free paraphrases and clause reorderings rather than long paragraph omissions. In the second sentence, `EN[29..41]` (*"A stable communication infrastructure is what makes using all of these services possible"*, `gold = 0`) and `IT[35..50]` (*"Affinché tutti possano usufruire di questi servizi , è necessaria un ' infrastruttura di comunicazione solida"*, `gold = 0`) invert clause order while preserving meaning; DiffAlign assigns high distance (`0.71–0.89` percentile on Italian, `0.52–0.87` on English) to these `gold = 0` tokens.
   - **Surprisal catching paraphrase divergence and acronym expansion:** Immediately before that sentence, `EN[26..27]` (`"watch TV"`, `gold = 0.6`) corresponds to the looser Italian paraphrase `IT[24..33]` (*"passiamo un po ' del nostro tempo davanti alla televisione"*, `gold = 0.6`). DiffAlign misses the short English span (`pct = 0.33–0.43`, below the English side mean of `0.58`), whereas `-gain` ranks `EN[26..27]` at the **97th percentile** and `IT[24..33]` at an **87th mean percentile** (up to `1.00`). Likewise, where English spells out `EN[43..47]` (`"The Federal Office of Communications"`, `gold = 1.0`) while Italian uses only the acronym `"L ' UFCOM"` (`IT[52..54]`), `-gain` ranks the extra English words at the `0.80–0.92` percentile.

**Summary of recurring error patterns:**
1. **Paragraph-level additions vs. clause-level paraphrases:** Both signals excel at multi-sentence additions/omissions (`admin_de_127`, `admin_it_9`), whereas clause reorderings of equivalent sentences (`admin_it_209`, `admin_fr_159`) inflate DiffAlign on `gold = 0` words.
2. **Smoothing window trade-off:** The 7-word (and in the head, 15-word) moving averages substantially boost correlation on long spans, but bleed 3–7 words into adjacent `gold = 0` context (`admin_de_127`) and dilute 1–2 word divergences (`admin_fr_159`).
3. **Document-start tokens in `alone` vs. `cond`:** At the very first token of a document (e.g. `admin_de_127` `EN[0] "Competitive"`, `gold = 0`, DiffAlign pct `0.94` vs. `-gain` pct `0.01`; `admin_it_9` `EN[0] "The"`, `gold = 0`, DiffAlign pct `0.80` vs. `-gain` pct `0.01`), DiffAlign is often elevated while `-gain` stays near zero—which may partly explain why combining both signals and relative position in the supervised head improves within-document ranking (+3.9 per-document Spearman).

## 6. Limitations

1. **In-domain supervision, dev selection optimism, and the post-hoc switch:**
   SwissGov-RSD has only `dev` (168 pages/pair) and `test` (56 pages/pair), with no separate `train` split; the paper's baselines are either training-free (DiffAlign, prompting) or trained on synthetic iSTS-RSD data. Our submitted head is trained on `dev`. On `dev`, our training-free `cond raw L14 +smooth7` signal already scores **36.4** (vs. **17.5** for the best of the seven rescored baselines on dev, `Qwen3-Embedding-4B`, and **16.4** / **20.5** unsmoothed / smoothed for `Qwen3-Embedding-8B`; the paper's segmented `LaBSE` and `XLM-R + SimCSE` runs were not rescored), while the supervised head adds **+3.5** points in nested CV (**39.9**) and **+3.6** on `dev/val` (**38.0** vs. **34.4**). In addition, only $\lambda$ is nested inside CV: the layer range (12–18), smoothing windows (1, 3, 7, 15), and head configuration were chosen on dev, and we switched post hoc from `(e)` per-pair to `(e)` shared on a virtual tie (**40.5** vs. **40.4** on `dev/train` inner CV; **40.0** vs. **39.9** outer CV; **37.6** vs. **38.0** `dev/val`). Consequently, dev CV numbers are slightly optimistic, and `dev/val` (38.0) and `test` (37.4) are the more realistic estimates.

2. **Narrow bootstrap CIs and single-run test scope:**
   Our document-level bootstrap CIs (`bootstrap_ci`, `paired_bootstrap`) resample pages with the fitted head held fixed; they do not include variance from refitting the head on different training folds or choosing the configuration, so they understate total uncertainty (the outer CV per-fold gain range of **+2.2 to +4.5** is a fairer indicator of training variance). On the test split, only our final shared head and the seven released baseline prediction files were scored (with separate bootstrap CIs only). Our reproduced Apertus baselines (`alone raw L32` = **6.9**, 3-shot prompting on the 30-document subset = **3.2**) and unsupervised `cond raw L14 +smooth7` variant (**36.4**) were evaluated on `dev` only, and the 30-document prompting baseline has no CI because 10/30 outputs failed to parse.

3. **Scope of the `Qwen3-Embedding-8B` control, smoothing, and document-level ranking:**
   Our all-layer control with `Qwen/Qwen3-Embedding-8B` (Section 5.3) compares one embedding model on one dataset at layers chosen on dev, under unconditioned (`alone`) passes only (+11.3 within Qwen from `L36` to `L21`, and +2.2 raw / +1.0 smoothed for Apertus `L14` over Qwen `L21`). We did not run conditioned passes or train a multi-layer head on Qwen. Moreover, ~4 points of the unsupervised gain come from 7-word smoothing (which exploits contiguous label spans and helps Qwen equally), and roughly half of global Spearman comes from document-level calibration (giving every word its document side's mean score keeps **18.7** of 36.4 for the baseline and **19.9** of 40.0 for the head, while mean within-document Spearman is **30.1** and **34.0**).

4. **The `EN–FR` test drop and annotator variance:**
   Across every system, `EN–FR` is the hardest pair (lowest divergent-token rate on dev at **10.2%** and mean gold **0.066**). On `test`, our head's `EN–FR` score drops to **23.4** (`[18.2, 28.2]`) vs. **29.1** in nested CV and **31.8** on `dev/val`—a larger drop than the 1–2 point decrease seen in the released baselines (`docs/test_results.md`). Because of the single-run test rule, we did not inspect test documents to diagnose this drop. Furthermore, on the 89 double-annotated `EN–DE` dev documents (Section 5.4), the two human label sets agree at only **18.9** Spearman, and our systems score nearly twice as high against Annotator 2 (**38.9 / 41.5**) as against Annotator 1 (**20.5 / 21.5**), showing that small score differences lie well within annotator variance.

5. **Model conversion, `fp16` validation scope, T4 memory margin, and untried directions:**
   Bitwise identity between `andreasmartin/apertus-v1.5-8b-text` and the official multimodal `swiss-ai/Apertus-v1.5-8B` release was not checked. Our CPU `bf16` reference check for the `fp16` overflow patches covered 128 tokens of one document pair (plus zero `NaN`/`Inf` across all 2,688 passes). On 2x 16 GB T4 GPUs, test extraction peaked at **13.0 / 12.9 GiB** (within ~1 GiB of the 14 GiB cap), so longer document pairs would require segmentation. Finally, we did not test a hidden-state probe, non-linear heads, document segmentation, echo embeddings (to give early tokens in `alone` passes more context), or Apertus 70B, and our heatmaps (`docs/heatmaps/`) display the two unsupervised signals rather than the final head's combined score.

## 7. Reproducibility

### 7.1 Judge CPU Path (`make run`, ~1 Minute, No GPU or Keys)

From a clean checkout of `swissgov_rsd/track_2a/`, `make run` builds the CPU Docker image and reproduces the head training, test prediction, and official test evaluation with no GPU, no model weights, and no API keys (`Makefile`, `Dockerfile`, `src/rsd/pipeline.py`):

```bash
make run     # 1. builds Docker image `swissgov-rsd` (FROM python:3.12-slim + uv):
             #    installs locked deps from pyproject.toml / uv.lock, copies src/ and data/,
             #    and runs `python -m official_eval.fetch --test` at build time to cache the
             #    pinned official evaluation script, dev data, and test data.
             # 2. runs `python -m rsd.pipeline --out outputs` inside the container:
             #    - fits the shared 96-feature head (lambda = 1e-3) on data/features_dev_head.npz (27.5 MB)
             #    - predicts all 168 test docs from data/features_test_head.npz (10.2 MB)
             #    - writes outputs/test/head_{de,fr,it}.jsonl.jsonl, head_model.npz, and head_config.json
             #    - scores our test predictions with the official script (final=True), writes
             #      outputs/scores.json, and prints:
             #        TEST (official script, Spearman x100)
             #          EN-DE 42.8  EN-FR 23.4  EN-IT 46.0  mean 37.4
```

(Note: `make run` scores our submitted head on test; as noted in `docs/test_results.md`, the seven released baselines on test in Section 5.5 were rescored once by a standalone one-off script and are not part of `make run`.)

Other `Makefile` targets run locally on CPU via `uv`:
```bash
make test            # runs `uv run pytest` locally
make eval-reference  # runs `uv run pytest tests/test_eval.py` and `uv run python -m official_eval.reference`
```

Dependencies are locked in `pyproject.toml` and `uv.lock` (`numpy`, `scipy`, `scikit-learn`, `nlpstats`, `jsonlines`, plus `pytest` in the `dev` group; no `torch` is needed on CPU). Committed artefacts under `data/` total **41.8 MB** (limit 100 MB):
- `data/features_dev_head.npz` (27.5 MB): compact precomputed Apertus dev features (504 docs, 405,654 words; no gold labels).
- `data/features_test_head.npz` (10.2 MB): compact precomputed Apertus test features (168 docs, 150,434 words; no gold labels).
- `data/predictions_test/` (4.0 MB): frozen single-run test predictions (`apertus_head_{de,fr,it}.jsonl.jsonl`), fitted head weights (`apertus_head_model.npz`, 14 KB), and metadata (`apertus_head_config.json`).

### 7.2 Reproducing Development Tables and Analyses on CPU

From `track_2a/`, the development tables in `docs/reference_dev_scores.md` and `docs/scalar_head_dev_results.md` can be regenerated directly from the committed files:

```bash
# Download official code, dev gold, and released baseline predictions; rescore all 7 baselines on dev
# (writes docs/reference_dev_scores.md)
uv run python -m official_eval.fetch --predictions
uv run python -m official_eval.reference

# Run full scalar head evaluation: nested 5-fold CV, dev/val, ablations, and 1,000-resample bootstraps
# (~54 min on a 12-core CPU; use `--sets bc` for a quick check)
uv run python -m rsd.head data/features_dev_head.npz --official cache/scalar_head_preds --n-boot 1000

# Fast fixed-config dev/train -> dev/val check (~15 s; reproduces 38.6 / 31.8 / 43.7, mean 38.0)
uv run python -m rsd.head fit-predict --train-split train --predict-split val --out cache/head_check/devval_
```

The remaining development analysis modules read the full gitignored Kaggle outputs under `kaggle/outputs/` (documented at the end of each `docs/*.md` file):

```bash
# Unsupervised 33-layer curve and tables (needs full 250 MB dev features.npz; ~6 min)
uv run python -m rsd.unsupervised kaggle/outputs/01_extract_dev/features.npz --official cache/unsupervised_preds

# Qwen3-Embedding-8B 37-layer control tables (needs 148 MB Qwen features.npz; ~5 min)
uv run python -m rsd.qwen_control kaggle/outputs/02_qwen_control/features.npz

# Few-shot prompting baseline parsing and tables on the 30-doc subset (needs generations.jsonl)
uv run python -m rsd.prompting kaggle/outputs/03_prompting_baseline/generations.jsonl

# EN-DE double-annotation comparison on 89 dev docs (reads kaggle/outputs/01_extract_dev/features.npz
# and cache/scalar_head_preds/ written by `python -m rsd.head ... --official cache/scalar_head_preds`)
uv run python -m rsd.annotators

# Regenerate the 4 rule-selected interactive HTML heatmaps in docs/heatmaps/ (needs full dev features.npz)
uv run python -m rsd.heatmap kaggle/outputs/01_extract_dev/features.npz
```

### 7.3 Offline GPU Extraction Path on Kaggle (2x NVIDIA T4)

To regenerate the `.npz` archives from scratch on Kaggle (2x T4, `torch 2.11.0+cu128`, `transformers 5.19.0`), `kaggle/run_on_kaggle.py` bundles `src/` and executes each script in `kaggle/` (`docs/final_run.md`, `docs/unsupervised_dev_results.md`; note that because Kaggle CLI pushes do not support Kaggle secrets, the Hugging Face token is read from an attached private Kaggle dataset containing `hf_token.txt`):

```bash
# 0. Smoke test: fp16 overflow patches on shortest/median/longest dev pairs + CPU bf16 reference check
uv run python kaggle/run_on_kaggle.py kaggle/00_smoke_test.py

# 1. Dev extraction (504 pairs, ~45 min session, 2,117 s passes) + compact export to data/features_dev_head.npz
uv run python kaggle/run_on_kaggle.py kaggle/01_extract_dev.py --kernel amunozo/rsd-extract-dev \
    --files '^(features\.npz|meta\.json)$'
uv run python -m rsd.features kaggle/outputs/01_extract_dev/features.npz data/features_dev_head.npz

# 2. Qwen3-Embedding-8B 37-layer control on dev (~16 min full run, 630 s passes)
uv run python kaggle/run_on_kaggle.py kaggle/02_qwen_control.py --kernel amunozo/rsd-qwen-control

# 3. Few-shot prompting baseline on 30 short dev docs (~72 min full run, 4,036 s generation)
uv run python kaggle/run_on_kaggle.py kaggle/03_prompting_baseline.py --kernel amunozo/rsd-prompting-baseline

# 4. Single final test run (168 pairs, ~15 min extraction, 899.5 s wall) + compact export + head fit-predict + eval
uv run python kaggle/run_on_kaggle.py kaggle/04_extract_test.py --kernel amunozo/rsd-extract-test \
    --files '^(features\.npz|index\.tsv|meta\.json)$'
uv run python -m rsd.features kaggle/outputs/04_extract_test/features.npz data/features_test_head.npz
uv run python -m rsd.head fit-predict \
    --train-features data/features_dev_head.npz --train-split dev \
    --predict-features data/features_test_head.npz --predict-split test --final \
    --out data/predictions_test/apertus_head_
uv run python -m official_eval.run data/predictions_test/apertus_head_ --final
```

### 7.4 Pinned Revisions and Automated Test Suite

| Artefact | Identifier | Pinned Revision / Commit |
|---|---|---|
| Primary model | `andreasmartin/apertus-v1.5-8b-text` | `c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01` |
| Control model (dev only) | `Qwen/Qwen3-Embedding-8B` | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| Dataset (Hugging Face) | `ZurichNLP/SwissGov-RSD` | `f16aa0536811b8c0cf975fede21eec5996e4b3c1` |
| Official evaluation & baselines (GitHub) | `ZurichNLP/SwissGov-RSD` | `1807a42100e742ed03d337c54c4b9ea86995f565` |
| Pre-test code freeze tag | `final-run` (`amunozo/apertus-swissgov-rsd`) | `4c804d40e8f528613e2237204f37a277141a3342` |

The automated test suite (`make test` / `uv run pytest`, **26 test functions / 29 test items: 28 passed, 1 skipped locally; 27 passed, 2 skipped on a clean clone** where one check skips when the gitignored Kaggle `features.npz` is absent, across `tests/test_eval.py`, `tests/test_head.py`, and `tests/test_extract.py`) verifies:
- exact agreement between our local Spearman metric (`src/rsd/metrics.py`) and the official evaluation script on the released `Qwen3-Embedding-8B` dev predictions, plus deterministic `bootstrap_ci` and `paired_bootstrap`;
- `dev/train` (134/pair) and `dev/val` (34/pair) split sizes, ID disjointness, and zero `page_en` overlap across all outer and inner `GroupKFold` splits;
- `PermissionError` gates blocking accidental test-split access in `rsd.data.load_docs`, `rsd.head.fit_predict`, and `rsd.extract.main` (the last skipped when `torch` is not installed), plus identical `rsd.extract.main` wiring in `01_extract_dev.py` and `04_extract_test.py`;
- rejection of `"llm"` substrings in prediction file paths, strict gold ID ordering, and rank-preserving `unit_interval` scaling; and
- absence of gold label access in `rsd.features.build` and `features.npz`, per-document feature isolation, training-only standardiser statistics, byte-identical features from compact `.npz` archives, and exact reproduction of the `dev/val` score (**38.0**) by `rsd.head fit-predict`.

## 8. Next Steps

1. **Conditioned (`cond`) and supervised control for `Qwen3-Embedding-8B`:**
   Our controlled experiment in Section 5.3 isolated the unconditioned (`alone`) DiffAlign recipe across all layers (+11.3 within Qwen from `L36` to `L21`, and +2.2 raw / +1.0 smoothed for Apertus `L14` over Qwen `L21`). A natural follow-up is to run cross-lingually conditioned passes (`cond_A`, `cond_B`) and fit the same multi-layer scalar head on `Qwen3-Embedding-8B` to separate the effect of causal conditioning and multi-scale supervision from the choice of backbone.

2. **Boundary-aware span smoothing and entity/number matching features:**
   Fixed-width moving averages ($k \in \{3, 7, 15\}$) capture contiguous label spans effectively (`s15` receives the largest weight sum in the head), but they blur span boundaries (`admin_de_127`) and dilute isolated 1–3 word differences (`admin_fr_159`). Replacing uniform boxcar windows with similarity-weighted smoothing or a lightweight linear-chain CRF over the scalar logits—together with explicit features for shared acronyms/numbers (`COMCO` in `admin_it_9`) and document-initial positions—would sharpen span edges.

3. **Document segmentation, echo embeddings, and `Apertus-70B` evaluation:**
   On 2x 16 GB T4 GPUs, full-document `cond_X` concatenation reaches 13.0 GiB on the longest pairs, and in `alone_X` passes the earliest words of a document have minimal left context. Segmenting long documents into paragraph windows (which also helped encoder DiffAlign in Wastl et al., 2026) or using echo embeddings for `alone` passes would bound memory regardless of page length and make it feasible to test middle layers of the 70B Apertus model, particularly on `EN–FR`.

## License

- **Source code:** All source code in this repository (`src/`, `kaggle/`, `tests/`, `Makefile`, `Dockerfile`) is released under the **Apache License 2.0** (see root `LICENSE`).
- **Technical report and documentation:** This technical report (`technical_report.md`), `README.md`, and the accompanying documentation in `docs/` are licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Dataset annotations and `admin.ch` source text copyright notice:**
  - The human semantic-difference annotations of `ZurichNLP/SwissGov-RSD` (Wastl et al., 2026) are licensed under **CC BY 4.0**.
  - Copyright of the underlying multilingual web page texts belongs to the **Swiss federal authorities (`admin.ch`)**. Note that `data/predictions_test/apertus_head_{de,fr,it}.jsonl.jsonl` (which must include `text_a` and `text_b` to be accepted by the official evaluation script) and the four development heatmaps in `docs/heatmaps/*.html` contain whitespace-tokenised `admin.ch` texts that remain under Swiss federal copyright.
  - Because only the `rsd/` subdirectory of the upstream `ZurichNLP/SwissGov-RSD` GitHub repository carries an MIT license file while `evaluation/` and `scripts/` have no license header, our repository does not vendor those files; instead, `src/official_eval/fetch.py` downloads them at Docker build time from pinned commit `1807a42100e742ed03d337c54c4b9ea86995f565`.
- **Models and third-party libraries:**
  - `andreasmartin/apertus-v1.5-8b-text` (community text-only conversion of `swiss-ai/Apertus-v1.5-8B`) and `Qwen/Qwen3-Embedding-8B` are released under the **Apache License 2.0**.
  - Python dependencies (`numpy`, `scipy`, `scikit-learn`, `nlpstats`, `jsonlines`, `pytest`, `torch`, `transformers`) are used under their respective permissive open-source licenses (Apache-2.0, BSD-3-Clause, and MIT).

## References

1. **Wastl, M., Vamvas, J., & Sennrich, R. (2026).** *SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of Semantic Differences Between Related Documents.* In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026), pp. 31134–31163. Dataset: `https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD` (commit `f16aa05`); Code & baseline predictions: `https://github.com/ZurichNLP/SwissGov-RSD` (commit `1807a42`).
2. **Swiss AI Initiative — EPFL, ETH Zurich, and CSCS (2025–2026).** *Apertus: Open and Transparent Large Language Models for Switzerland (`swiss-ai/Apertus-v1.5-8B`, `arXiv:2509.14233`).* Community text-only conversion used in this work: `https://huggingface.co/andreasmartin/apertus-v1.5-8b-text` (commit `c0e7eeb`).
3. **Vamvas, J., & Sennrich, R. (2023).** *Towards Unsupervised Recognition of Token-Level Semantic Differences in Related Documents.* In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), pp. 13543–13552 (introduces the DiffAlign cosine alignment score and synthetic iSTS-RSD formulation).
4. **Vamvas, J., & Sennrich, R. (2022).** *As Little as Possible, as Much as Necessary: Detecting Over- and Undertranslations with Contrastive Conditioning.* In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022) (motivates contrastive conditioning log-probability gain).
5. **Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. (2024).** *Massive Activations in Large Language Models.* In Proceedings of the First Conference on Language Modeling (COLM 2024), `arXiv:2402.17762` (characterises massive residual-stream activations at BOS and delimiter tokens in transformer decoders).
6. **Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., & Zhou, J. (2025).** *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models* (`Qwen/Qwen3-Embedding-8B`, commit `1d8ad4c`). `arXiv:2506.05176`.
