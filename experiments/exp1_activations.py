"""Experiment 1 - activations exist for a reason.

Two noisy rings (inner = class 0, outer = class 1, 300 points). Train
  (a) a single linear layer + sigmoid          2 -> 1
  (b) one hidden layer with ReLU + sigmoid     2 -> 32 -> ReLU -> 1
with the same data, optimiser, learning rate and number of steps.
Repeated for 5 seeds so nobody can say we picked a lucky one.
"""
import numpy as np
import torch
from torch import nn

from common import (bce_and_acc, checkpoint_steps, export_linear_stack, make_rings,
                    mlp_logits, rounded_array, seed_all, sig, write_json)

SEEDS = [0, 1, 2, 3, 4]
N_POINTS = 300
HIDDEN = 32
STEPS = 1500
LR = 0.02
N_CKPT = 40


def best_possible_line(x: np.ndarray, y: np.ndarray, n_dirs: int = 1440) -> float:
    """Brute force: the best accuracy ANY straight line can get on this data.

    For every direction, project the points onto it and try every threshold
    (both orientations). This is an exact upper bound up to direction resolution.
    """
    best = 0.0
    n = len(y)
    for a in np.linspace(0, np.pi, n_dirs, endpoint=False):
        proj = x @ np.array([np.cos(a), np.sin(a)])
        ys = y[np.argsort(proj)]
        ones_left = np.concatenate([[0], np.cumsum(ys)])          # class-1 count left of cut
        k = np.arange(n + 1)
        # predict 0 left / 1 right, or the opposite orientation
        acc1 = (k - ones_left) + (ys.sum() - ones_left)
        acc2 = n - acc1
        best = max(best, acc1.max() / n, acc2.max() / n)
    return float(best)


def train(model: nn.Module, x, y, relu: bool):
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    ckpts, want = [], set(checkpoint_steps(STEPS, N_CKPT))
    for step in range(STEPS + 1):
        if step in want:
            ckpts.append((step, export_linear_stack(model)))
        if step == STEPS:
            break
        opt.zero_grad()
        lossf(model(xt).squeeze(1), yt).backward()
        opt.step()
    return ckpts


def run_seed(seed: int):
    x, y = make_rings(N_POINTS, seed)
    x = rounded_array(x, 6)                             # train + score on exactly what we export
    xt, yt = make_rings(N_POINTS, seed + 1000)          # fresh points, never trained on
    seed_all(seed)
    linear = nn.Sequential(nn.Linear(2, 1))
    seed_all(seed)
    relu = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    out = {"seed": seed, "points": sig(x, 6), "labels": y.astype(int).tolist(),
           "bestLineAcc": best_possible_line(x, y), "models": {}}
    for name, model, is_relu in [("linear", linear, False), ("relu", relu, True)]:
        frames = []
        for step, layers in train(model, x, y, is_relu):
            loss, acc = bce_and_acc(mlp_logits(x, layers, is_relu), y)
            tloss, tacc = bce_and_acc(mlp_logits(xt, layers, is_relu), yt)
            frames.append({"step": step, "loss": loss, "acc": acc, "testAcc": tacc, "testLoss": tloss,
                           "layers": layers})
        out["models"][name] = {"frames": frames}
    return out


def main():
    print("exp1: activations")
    runs = [run_seed(s) for s in SEEDS]
    for r in runs:
        lf, rf = r["models"]["linear"]["frames"][-1], r["models"]["relu"]["frames"][-1]
        print(f"  seed {r['seed']}: linear {lf['acc']:.3f} (test {lf['testAcc']:.3f})  "
              f"relu {rf['acc']:.3f} (test {rf['testAcc']:.3f})  best line {r['bestLineAcc']:.3f}")
    write_json("exp1.json", {
        "config": {"points": N_POINTS, "hidden": HIDDEN, "steps": STEPS, "lr": LR,
                   "optimizer": "Adam (full batch)", "loss": "binary cross-entropy",
                   "ring": {"innerRadius": 1.0, "outerRadius": 2.1, "noise": 0.22},
                   "testPoints": N_POINTS},
        "runs": runs,
    })
    return runs


if __name__ == "__main__":
    main()
