"""Per-word feature matrix for the scalar head (rsd.head).

Input: the dict of arrays from features.npz (kaggle/01_extract_dev.py). Gold
labels are never an input here: every feature is a function of the model
signals, the word position and the document lengths only.

Rows follow features.npz: grouped by (doc_id, side), words in order.
Signals are oriented so that higher = more different.

Feature sets used by the ablation (all are column subsets of "e"):
- "b":  DiffAlign cond raw L14 at windows 1/3/7/15 + document aggregates
        (side mean, other-side mean, pair mean of that signal; log length of
        the side, log length ratio of the two sides, side indicator, relative position).
- "c":  "b" + log-prob signals (-gain sum/first, -log p_cond sum,
        -log p_alone sum) at windows 1/3/7/15, and doc aggregates of -gain sum.
- "d":  "b" + DiffAlign {alone, cond} x {raw, cent} at L12/14/16/18 at
        windows 1/3/7/15, and doc aggregates of DiffAlign cond cent L14 and
        alone raw L14.
- "e":  everything ("c" and "d" together).
"""
import numpy as np

WINDOWS = (1, 3, 7, 15)
LAYERS = (12, 14, 16, 18)
FEATURE_SETS = ("b", "c", "d", "e")


def segments(feats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Start and end rows of each (doc_id, side) block."""
    key = np.char.add(np.char.add(feats["doc_id"], "|"), feats["side"])
    bounds = np.flatnonzero(key[1:] != key[:-1]) + 1
    return np.r_[0, bounds], np.r_[bounds, len(key)]


def smooth_seg(x: np.ndarray, starts, ends, k: int) -> np.ndarray:
    """Centered k-word moving average inside each block (same as
    rsd.unsupervised.smooth, applied per block). k=1 returns x."""
    x = x.astype(np.float64)
    if k == 1:
        return x
    half = k // 2
    out = np.empty(len(x), dtype=np.float64)
    for lo, hi in zip(starts, ends):
        c = np.r_[0.0, np.cumsum(x[lo:hi])]
        i = np.arange(hi - lo)
        left, right = np.maximum(i - half, 0), np.minimum(i + half + 1, hi - lo)
        out[lo:hi] = (c[right] - c[left]) / (right - left)
    return out


class _Blocks:
    """Block bookkeeping: per-row block id, block lengths, partner block."""

    def __init__(self, feats: dict):
        self.starts, self.ends = segments(feats)
        self.n = self.ends - self.starts
        self.row_block = np.repeat(np.arange(len(self.starts)), self.n)
        doc = feats["doc_id"][self.starts]
        side = feats["side"][self.starts]
        index = {(d, s): i for i, (d, s) in enumerate(zip(doc, side))}
        self.partner = np.array([index.get((d, "b" if s == "a" else "a"), i)
                                 for i, (d, s) in enumerate(zip(doc, side))])
        self.is_b = (side == "b").astype(np.float64)

    def block_mean(self, x: np.ndarray) -> np.ndarray:
        return np.add.reduceat(x.astype(np.float64), self.starts) / self.n

    def aggregates(self, x: np.ndarray, name: str) -> dict[str, np.ndarray]:
        m = self.block_mean(x)
        other = m[self.partner]
        pair = (m * self.n + other * self.n[self.partner]) / (self.n + self.n[self.partner])
        rb = self.row_block
        return {f"{name} side_mean": m[rb], f"{name} other_mean": other[rb], f"{name} pair_mean": pair[rb]}

    def layout(self, feats: dict) -> dict[str, np.ndarray]:
        rb = self.row_block
        n_self, n_other = self.n[rb].astype(np.float64), self.n[self.partner][rb].astype(np.float64)
        pos = feats["word"].astype(np.float64) / np.maximum(n_self - 1, 1)
        # log len other = log len side - log len ratio, so it is left out
        # (an exactly collinear column makes the weights unidentifiable).
        return {"log len side": np.log(n_self),
                "log len ratio": np.log(n_self / n_other), "side is B": self.is_b[rb],
                "rel position": pos}


def _signals(feats: dict) -> dict[str, np.ndarray]:
    s = {}
    for kind in ("alone", "cond"):
        for c in ("raw", "cent"):
            for layer in LAYERS:
                s[f"da {kind} {c} L{layer}"] = feats[f"da_{kind}_{c}"][:, layer]
    s["-gain sum"] = -(feats["lp_cond_sum"] - feats["lp_alone_sum"])
    s["-gain first"] = -(feats["lp_cond_first"] - feats["lp_alone_first"])
    s["-logp_cond sum"] = -feats["lp_cond_sum"]
    s["-logp_alone sum"] = -feats["lp_alone_sum"]
    return s


MAIN = "da cond raw L14"
GAIN = ("-gain sum", "-gain first", "-logp_cond sum", "-logp_alone sum")


def _plan(feature_set: str) -> tuple[list[str], list[str]]:
    """(signals used at every window, signals with doc aggregates)."""
    assert feature_set in FEATURE_SETS, feature_set
    windowed, agg = [MAIN], [MAIN]
    if feature_set in ("c", "e"):
        windowed += list(GAIN)
        agg += ["-gain sum"]
    if feature_set in ("d", "e"):
        windowed += [f"da {k} {c} L{l}" for k in ("alone", "cond") for c in ("raw", "cent")
                     for l in LAYERS if f"da {k} {c} L{l}" != MAIN]
        agg += ["da cond cent L14", "da alone raw L14"]
    return windowed, agg


def build(feats: dict, feature_set: str = "e") -> tuple[np.ndarray, list[str]]:
    """Feature matrix (n_words x n_features, float64, not standardised) and names."""
    windowed, agg = _plan(feature_set)
    blocks = _Blocks(feats)
    sig = _signals(feats)
    cols = {}
    for name in windowed:
        for k in WINDOWS:
            cols[name if k == 1 else f"{name} s{k}"] = smooth_seg(sig[name], blocks.starts, blocks.ends, k)
    for name in agg:
        cols.update(blocks.aggregates(sig[name], name))
    cols.update(blocks.layout(feats))
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


class Standardiser:
    """z-score with statistics from the rows passed to fit (training rows only),
    then clip to [-clip, clip] so a few extreme log-prob values cannot dominate."""

    def __init__(self, clip: float = 5.0):
        self.clip = clip

    def fit(self, X: np.ndarray) -> "Standardiser":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean) / self.std, -self.clip, self.clip)


# ---------------------------------------------------------------- compact files

DA_KEYS = ("da_alone_raw", "da_alone_cent", "da_cond_raw", "da_cond_cent")
LP_KEYS = ("lp_alone_sum", "lp_alone_first", "lp_cond_sum", "lp_cond_first")
ID_KEYS = ("lang", "doc_id", "side", "word")
N_STATES = 33


def export_compact(feats: dict, path) -> None:
    """Write only what build() needs: the id arrays, the four log-prob arrays and
    DiffAlign at LAYERS, all float32 as in features.npz (so values are
    bit-identical), compressed. No gold, no hidden states."""
    layers = np.array(LAYERS)
    out = {k: feats[k] for k in ID_KEYS}
    out.update({k: feats[k].astype(np.float32) for k in LP_KEYS})
    out.update({k: feats[k][:, layers].astype(np.float32) for k in DA_KEYS})
    out["layers"] = layers
    np.savez_compressed(path, **out)


def load(path) -> dict[str, np.ndarray]:
    """Read features.npz (full) or a compact file from export_compact. Compact
    DiffAlign arrays are expanded back to 33 columns; layers not stored are
    NaN, so using one by mistake shows up as NaN features."""
    with np.load(path) as f:
        feats = {k: f[k] for k in f.files}
    if "layers" in feats:
        layers = feats.pop("layers")
        for k in DA_KEYS:
            full = np.full((len(feats[k]), N_STATES), np.nan, dtype=np.float32)
            full[:, layers] = feats[k]
            feats[k] = full
    return feats


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Export the compact feature file used by rsd.head.")
    p.add_argument("src", help="features.npz from the Kaggle extraction")
    p.add_argument("dst", help="output .npz, e.g. data/features_dev_head.npz")
    a = p.parse_args()
    export_compact(load(a.src), a.dst)
    print(f"{a.dst}: {os.path.getsize(a.dst) / 1e6:.1f} MB")
