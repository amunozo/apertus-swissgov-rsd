# Prompting baseline with Apertus (dev subset)

Spearman x 100, global per pair, local metric (`rsd.metrics`), on a **30-document
dev subset** (10 per pair). **mean** is the macro average. The test split was
not used.

## Why

The paper's second family of baselines is few-shot prompting: the LLM reads
both tokenized documents and writes JSON with one score per token. GPT-4o and
Llama-3.1 405B score about 0-7 on SwissGov-RSD. We reproduce this baseline with
Apertus, as the brief asks, on a small subset because generation is slow on a
T4.

## Setup

- **Model:** `andreasmartin/apertus-v1.5-8b-text` at commit
  `c0e7eeb69c55c7810c91de3a1f97dde4d4b14b01`. This is the text-only conversion
  of `swiss-ai/Apertus-v1.5-8B`, which is the instruction-tuned release, with a
  chat template.
  - It ran in fp16 on 2x T4, with the same overflow patches as the feature
    extraction (`src/rsd/apertus.py`).
  - The final norm output is cast to fp16 before `lm_head`.
- **Prompt:** the official template `prompt_templates/template_{lang}_admin.txt`
  (ZurichNLP/SwissGov-RSD, pinned commit; fetched, not vendored).
  - It is rendered as `scripts/create_test_data_admin.py` does: three
    sentence-level few-shot examples, the 0-5 scale, and then
    `Sentence 1: json.dumps(tokens_a)` / `Sentence 2: json.dumps(tokens_b)`.
  - `json.dumps` escapes non-ASCII characters (`"Änderung"`), exactly as in
    the released GPT-4o prompts.
  - The prompt is one user message passed through the model's own chat
    template, as `scripts/predict_llama.py` does. The template adds its
    default system prompt ("You are Apertus 1.5 Omni ...") and
    "Deliberation: disabled" (thinking off).
- **Decoding:** greedy. `max_new_tokens` = 1.5 x the token length of a
  well-formed answer for that document (at most 4096). The paper used a fixed
  4096; the cap keeps a runaway output inside the GPU budget. No output hit
  the cap: all 30 stopped at the end-of-turn token.
- **Subset rule** (`rsd.prompting.select_docs`, fixed in advance):
  - For each pair, the pool is the dev documents that are not in the
    official `list_to_drop.txt` (the paper drops these when it scores LLM
    predictions) and that have at most 300 words on both sides together.
    That leaves 26 / 18 / 24 documents for DE / FR / IT.
  - A seeded permutation (`numpy.random.default_rng(0)`) picks 10 per pair.
  - The 300-word cap was set from the budget: at about 6.7 output tokens per
    word and about 11 tokens/s, 30 documents take about 70 minutes.
  - **This favours short pages.** Their median is 218 words, against about
    590-680 for all dev documents. Scores on this subset are not comparable to
    the full-dev scores elsewhere in `docs/`.
- **Parsing:** the official code (`evaluation/predictions.py` `LLMPrediction`,
  `evaluation/utils.py` `parse_token_labels`), run in the official root by
  `rsd.prompting.parse_outputs`. There is one change: the official `get_json()` calls
  `eval()` on the model output, which we replace with `ast.literal_eval`, so
  model text is never executed. On these outputs the two give the same
  result.
  - The official handling of failures is kept:
    - Tokens the parser cannot align get the fallback label 5 (= 0.0
      difference).
    - A score s becomes 1 - s/5; the model's -1 stays -1.
    - An output that does not parse gives empty label lists, which the
      official script pads with 0.0. So that document scores 0.0
      everywhere.

## Output quality

| | count |
|---|---|
| Documents | 30 (10 per pair), all finished with the end-of-turn token |
| Outputs that do not parse (all-zero predictions) | **10 / 30** (DE 3, FR 5, IT 2) |
| Parsed sides whose entry count differs from the token count | 29 of 40 |
| Tokens of parsed documents the parser could not align (fallback 0.0) | 851 of 4,414 (19%) |
| Distinct predicted values over all 6,700 tokens | 7 (-1, 0, 0.2, ..., 1) |
| Tokens with the most common value (0.0) | 64.5% |

Value counts over all tokens: 0.0: 4,323; 1.0: 1,778; -1: 237; the four
values in between: 75-108 each.

Why outputs fail to parse. Most failures are malformed JSON in a long list:
- an item loses its opening bracket, e.g. `"étant", 0]`. This happens
  most often on tokens that the prompt showed as `\u` escapes;
- a punctuation token is written without a score, e.g. `".", ["Specialist", 0]`;
- the brackets are unbalanced at the end.

Even when the JSON parses, the model often merges or skips tokens, e.g.
`["d", "'", 0]` for the two tokens `d` and `'`. The official parser only
looks one token ahead, so after a skip it can lose alignment for the rest of
the side (up to 105 unaligned tokens on one side). Those tokens get the
fallback 0.0.

The model also uses the scale in a way that hurts the ranking. Label 0 ("no
relation", 1.0 difference) appears on many plainly translated function
words. Label 5 appears on content words. The few-shot examples use the scale
very differently.

## Scores on the same 30 documents

| System | EN-DE | EN-FR | EN-IT | mean |
|---|---|---|---|---|
| **Apertus prompting (paper prompt, greedy)** | 16.3 | -10.2 | 3.6 | **3.2** |
| Apertus DiffAlign alone raw L32 (last layer, paper recipe) | 12.3 | 11.7 | -5.2 | 6.3 |
| Qwen3-Embedding-8B DiffAlign (released) | 28.5 | 13.2 | 12.0 | 17.9 |
| Qwen3-Embedding-8B DiffAlign (released) +smooth7 | 34.7 | 18.1 | 13.3 | 22.0 |
| Apertus DiffAlign alone raw L14 | 43.9 | 32.2 | 32.0 | 36.0 |
| Apertus DiffAlign cond raw L14 +smooth7 (our best) | 50.5 | 39.5 | 37.6 | 42.5 |

Each DiffAlign row uses the same predictions as on full dev, restricted to
these documents.

## Conclusion

- Prompting Apertus with the paper's prompt gives **3.2** on this subset. That
  is near zero, in the same range as the paper's prompted GPT-4o (about 1-6)
  and Llama-3.1 405B (about -2 to 7). The per-pair values swing from -10.2
  to 16.3. With 10 short documents per pair and a third of the outputs
  unparseable, these per-pair numbers are mostly noise.
- We did not compute a bootstrap CI. Our bootstrap resamples pages jointly
  across the three pairs, and the three 10-document subsets share few pages.
  Some resamples also contain only all-zero (unparsed) documents, which makes
  Spearman undefined.
- The failure modes match the paper's diagnosis: the output has to repeat
  every token in order, so the model miscounts, merges or breaks tokens, and
  writes malformed JSON. The scores are coarse (six levels) and heavily tied.
  It is also slow: 45.6k generated tokens took 67 minutes for 30 short
  documents, about 2.2 minutes per document. One forward pass to extract
  features takes about 1 second per document.
- On the same documents, reading hidden states instead of generating text
  (DiffAlign at layer 14) scores 36.0, or 42.5 with conditioning and
  smoothing. The gap is the reason our method reads scores from the model
  instead of asking for them.

## Commands and cost

From `track_2a/`:

```
# GPU (Kaggle 2x T4), kernel amunozo/rsd-prompting-baseline
python kaggle/run_on_kaggle.py kaggle/03_prompting_baseline.py --kernel amunozo/rsd-prompting-baseline
# CPU: parsing and all tables above
uv run python -m rsd.prompting kaggle/outputs/03_prompting_baseline/generations.jsonl
```

GPU time:
- one smoke run (2 documents) of about 9 minutes;
- one full run of about 72 minutes in session: model load 149 s, then 4,036 s
  of generation (54.0k prompt tokens, 45.6k new tokens, 10.7-12.0
  tokens/s).

The raw generations are in `kaggle/outputs/03_prompting_baseline/generations.jsonl`
(gitignored).
