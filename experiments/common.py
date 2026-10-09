"""Shared helpers for the four experiments.

Everything the website shows is produced by these scripts and written to
web/public/data/*.json. Nothing in the frontend is typed in by hand.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "web" / "public" / "data"
RESULTS_DIR = ROOT / "results"

torch.set_num_threads(1)               # bit-reproducible CPU runs
torch.use_deterministic_algorithms(True)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sig(a, digits: int = 7):
    """Round to `digits` significant figures and return a flat python list.

    The website recomputes model outputs from these exported numbers, so every
    metric we publish is computed from the *rounded* weights too (see
    `rounded_array`). That keeps Python and the browser in exact agreement.
    """
    flat = np.asarray(a, dtype=np.float64).ravel()
    return [float(f"{v:.{digits}g}") for v in flat]


def rounded_array(a, digits: int = 7) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.array(sig(a, digits), dtype=np.float64).reshape(a.shape)


def checkpoint_steps(total: int, n: int) -> list[int]:
    """0 plus ~n log-spaced steps up to `total` (dense early, where things move)."""
    steps = np.unique(np.round(np.geomspace(1, total, n)).astype(int))
    return [0] + [int(s) for s in steps]


# ---------------------------------------------------------------- ring data
def make_rings(n: int, seed: int, r_in=1.0, r_out=2.1, noise=0.22):
    """n noisy 2-D points: inner ring = class 0, outer ring = class 1 (n/2 each)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    radius = np.concatenate([np.full(half, r_in), np.full(n - half, r_out)])
    theta = rng.uniform(0, 2 * math.pi, n)
    radius = radius + rng.normal(0, noise, n)
    xy = np.stack([radius * np.cos(theta), radius * np.sin(theta)], 1)
    xy += rng.normal(0, noise * 0.35, xy.shape)       # a little isotropic jitter too
    y = np.concatenate([np.zeros(half), np.ones(n - half)])
    order = rng.permutation(n)
    return xy[order], y[order]


# ------------------------------------------------------- numpy forward passes
def mlp_logits(x: np.ndarray, layers: list[dict], relu: bool) -> np.ndarray:
    """float64 forward pass identical to the one the browser runs."""
    h = x
    for i, L in enumerate(layers):
        h = h @ np.asarray(L["W"]).T + np.asarray(L["b"])
        if relu and i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h[:, 0]


def bce_and_acc(logits: np.ndarray, y: np.ndarray):
    # numerically stable binary cross-entropy: log(1 + exp(-z)) for y=1, log(1 + exp(z)) for y=0
    loss = float(np.mean(np.logaddexp(0, -logits * (2 * y - 1))))
    acc = float(np.mean((logits > 0) == (y > 0.5)))
    return loss, acc


def export_linear_stack(model: torch.nn.Sequential, digits=7) -> list[dict]:
    """Every nn.Linear in order, weights rounded for export."""
    out = []
    for m in model:
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach().double().numpy()
            b = m.bias.detach().double().numpy()
            out.append({"W": rounded_array(W, digits).tolist(), "b": rounded_array(b, digits).tolist()})
    return out


def write_json(name: str, obj) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"  wrote {p.relative_to(ROOT)}  ({os.path.getsize(p)/1024:.1f} KB)")
    return p
