"""Apertus 1.5 8B on Kaggle's 2x T4 in fp16.

Model loading with the memory and fp16-overflow patches (verified in
kaggle/00_smoke_test.py: cos >= 0.997 against CPU bf16 at every layer), input
construction, a forward pass that returns the hidden states and next-token
log-probs of one document's subwords, and the subword-to-word mapping.

Needs torch and transformers, so it runs on Kaggle only; nothing else in rsd
imports it.

Hidden-state indices follow transformers' `hidden_states` convention:
0 = embeddings, l = output of decoder layer l (1..31), 32 = final norm output
(the input of lm_head). The states are fp32 and hold massive activations
(~119k) from layer 4 on: never store them as raw fp16.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

MODEL = "andreasmartin/apertus-v1.5-8b-text"
NAMES = {"en": "English", "de": "German", "fr": "French", "it": "Italian"}
N_STATES = 33  # embeddings + 32 layers


def _mlp_forward(self, x):
    # xIELU is quadratic for positive inputs: run it in fp32 and scale each
    # token down before the fp16 down_proj (linear, so exact up to rounding).
    h = self.act_fn(self.up_proj(x).float())
    scale = torch.exp2(torch.ceil(torch.log2(h.abs().amax(-1, keepdim=True).clamp(min=1.0) / 2**12)).clamp(min=0))
    return self.down_proj((h / scale).half()).float() * scale


def _layer_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                   use_cache=False, position_embeddings=None, **kwargs):
    # fp16 weights, fp32 residual stream.
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


def patch() -> None:
    """Apply the memory and overflow patches. Call before the first forward."""
    import transformers.integrations.sdpa_attention as sdpa_attention
    from transformers.models.apertus import modeling_apertus

    # Asking SDPA for GQA makes it fall back to the "math" kernel, which builds
    # the full attention matrix (OOM on the longest pair). Expand KV heads
    # first, and disable the math kernel so any fallback fails loudly.
    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    modeling_apertus.ApertusMLP.forward = _mlp_forward
    modeling_apertus.ApertusDecoderLayer.forward = _layer_forward


def load(revision: str | None = None):
    """(tokenizer, model) in fp16 across both GPUs, patched."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    patch()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, dtype=torch.float16, device_map="auto",
        max_memory={0: "14GiB", 1: "14GiB"},
    )
    model.eval()
    assert tok.bos_token_id is not None
    return tok, model


def word_index(text: str, offsets) -> list[int]:
    """Word (index into text.split(" ")) of each subword, from character offsets.

    A subword belongs to the word that holds its last character; a leading
    space belongs to the following word. Raises if a word gets no subword.
    """
    words = text.split(" ")
    char2word = []
    for i, w in enumerate(words):
        char2word += [i] * len(w) + [i + 1]  # the space after word i belongs to word i+1
    out = []
    for s, e in offsets:
        if e <= s:
            raise ValueError(f"empty offset ({s}, {e})")
        out.append(char2word[e - 1])
    if out != sorted(out) or sorted(set(out)) != list(range(len(words))):
        raise ValueError("subwords out of order, or some word has no subword")
    return out


def encode_doc(tok, text: str) -> tuple[list[int], list[int]]:
    """(subword ids, word index per subword) of a whitespace-tokenized text."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], word_index(text, enc["offset_mapping"])


def ids(tok, s: str) -> list[int]:
    return tok(s, add_special_tokens=False)["input_ids"]


def build(tok, x_ids: list[int], x_lang: str, y_ids: list[int] | None = None, y_lang: str | None = None):
    """Input ids and the start of X's subwords.

    alone: [BOS] {NameX}:\\n X
    cond:  [BOS] {NameY}:\\n Y \\n\\n{NameX}:\\n X
    X's ids are the same in both frames.
    """
    seq = [tok.bos_token_id]
    if y_ids is not None:
        seq += ids(tok, f"{NAMES[y_lang]}:\n") + y_ids + ids(tok, f"\n\n{NAMES[x_lang]}:\n")
    else:
        seq += ids(tok, f"{NAMES[x_lang]}:\n")
    x_start = len(seq)
    return seq + x_ids, x_start


@torch.no_grad()
def forward(model, seq: list[int], x_start: int):
    """Hidden states and log-probs of the subwords seq[x_start:].

    Returns (states, logp): states is a list of N_STATES fp32 tensors
    [n_x, d], each on the device of its layer; logp is fp32 [n_x] with
    log p(seq[t] | seq[:t]) for every X position t. Raises on NaN/inf.
    """
    inner = model.model
    states = [None] * N_STATES

    def keep(i):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            states[i] = h[0, x_start:].to(torch.float32, copy=True)
        return hook

    modules = [inner.embed_tokens, *inner.layers[:N_STATES - 2], inner.norm]
    handles = [m.register_forward_hook(keep(i)) for i, m in enumerate(modules)]
    try:
        input_ids = torch.tensor([seq], device=model.device)
        last = inner(input_ids=input_ids, use_cache=False).last_hidden_state
    finally:
        for h in handles:
            h.remove()

    head = model.lm_head
    hidden = last[0, x_start - 1:-1].to(head.weight.device).half()  # predicts seq[x_start:]
    targets = input_ids[0, x_start:].to(head.weight.device)
    logp = []
    for s in range(0, len(targets), 1024):
        logits = head(hidden[s:s + 1024]).float()
        logp.append(torch.log_softmax(logits, -1).gather(-1, targets[s:s + 1024, None])[:, 0])
    logp = torch.cat(logp)
    for i, h in enumerate(states):
        if h is None or not torch.isfinite(h).all():
            raise FloatingPointError(f"hidden state {i} is missing or not finite")
    if not torch.isfinite(logp).all():
        raise FloatingPointError("log-probs not finite")
    return states, logp


def pool(values: torch.Tensor, word: torch.Tensor, n_words: int) -> torch.Tensor:
    """Mean of subword rows per word. values [n_sub, ...], word [n_sub] (long)."""
    out = torch.zeros((n_words, *values.shape[1:]), dtype=values.dtype, device=values.device)
    out.index_add_(0, word, values)
    count = torch.bincount(word, minlength=n_words).to(values.dtype)
    return out / count.view(-1, *[1] * (values.dim() - 1))


def diffalign(x: torch.Tensor, y: torch.Tensor, mean: torch.Tensor | None = None) -> torch.Tensor:
    """1 - max cosine of each row of x to the rows of y (subword level).

    mean: optional vector subtracted from both sides first (mean-centering).
    """
    if mean is not None:
        x, y = x - mean, y - mean
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return 1 - (x @ y.T).max(-1).values
