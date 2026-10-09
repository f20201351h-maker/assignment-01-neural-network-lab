"""Experiment 4 - memorisation vs generalisation, and more data closes the gap.

Task: noisy "two moons" (two interlocking half-circles + Gaussian noise), so the
classes genuinely overlap and the true pattern is a smooth curve.

  * one fixed training pool of 2000 points, and a SEPARATE held-out test set
    of 2000 points drawn with a different random stream (never trained on)
  * training sets are nested prefixes of the pool: 20 ⊂ 50 ⊂ ... ⊂ 2000
  * every size uses the SAME network (2 -> 128 -> 128 -> 1, ReLU, 17,025
    parameters), the SAME initial weights, the SAME optimiser, learning rate and
    number of full-batch steps. The only thing that changes is how much data.
  * repeated for 5 seeds (new pool, new test set, new init) for error bars.
"""
import base64
import math

import numpy as np
import torch
from torch import nn

from common import bce_and_acc, checkpoint_steps, seed_all, sig, write_json

SIZES = [20, 50, 100, 200, 500, 1000, 2000]
SEEDS = [0, 1, 2, 3, 4]
N_TEST = 2000
NOISE = 0.28
HIDDEN = 128
STEPS = 2000
LR = 0.005
N_CKPT = 40
GRID = 72
DOMAIN = {"x": [-1.7, 2.7], "y": [-1.25, 1.75]}


def moons(n: int, rng: np.random.Generator):
    """Balanced two moons: alternate classes so every prefix is (near) 50/50."""
    y = np.arange(n) % 2
    t = rng.uniform(0, math.pi, n)
    x = np.where(y[:, None] == 0,
                 np.stack([np.cos(t), np.sin(t)], 1),
                 np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1))
    return x + rng.normal(0, NOISE, (n, 2)), y.astype(float)


def bayes_posterior(pts: np.ndarray) -> np.ndarray:
    """P(class 1 | x) under the TRUE generating process (numerical integral over
    the arc). This is 'the real pattern' - the best any model could do."""
    t = np.linspace(0, math.pi, 400)
    a0 = np.stack([np.cos(t), np.sin(t)], 1)
    a1 = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    def loglik(arc):          # log-space so far-away points don't underflow to 0/0
        d2 = ((pts[:, None, :] - arc[None]) ** 2).sum(-1)
        z = -d2 / (2 * NOISE ** 2)
        m = z.max(1, keepdims=True)
        return (m + np.log(np.exp(z - m).mean(1, keepdims=True)))[:, 0]
    return 1 / (1 + np.exp(loglik(a0) - loglik(a1)))


def grid_points():
    xs = np.linspace(*DOMAIN["x"], GRID)
    ys = np.linspace(*DOMAIN["y"], GRID)
    gx, gy = np.meshgrid(xs, ys)          # row = y, col = x
    return np.stack([gx.ravel(), gy.ravel()], 1)


def net(seed: int) -> nn.Sequential:
    seed_all(seed)                        # same init for every training size
    return nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, 1))


def to_u8(p: np.ndarray) -> str:
    """Probabilities -> bytes 0..255 -> base64 (keeps the artifact small)."""
    return base64.b64encode(np.clip(np.round(p * 255), 0, 255).astype(np.uint8).tobytes()).decode()


def run(seed: int, keep_visuals: bool):
    pool_x, pool_y = moons(max(SIZES), np.random.default_rng(10_000 + seed))
    test_x, test_y = moons(N_TEST, np.random.default_rng(20_000 + seed))
    G = torch.tensor(grid_points(), dtype=torch.float32)
    Xt = torch.tensor(test_x, dtype=torch.float32)
    want = checkpoint_steps(STEPS, N_CKPT)
    grid_steps = set(want[::2] + [want[-1]])
    init_sig = None
    results = []
    for n in SIZES:
        x, y = pool_x[:n], pool_y[:n]                           # nested prefix
        model = net(seed)
        s = float(sum(p.detach().double().sum() for p in model.parameters()))
        init_sig = s if init_sig is None else init_sig
        assert s == init_sig, "every size must start from identical weights"
        X, Y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss()
        curve, grids = [], []
        for step in range(STEPS + 1):
            if step in want:
                with torch.no_grad():
                    tr = model(X).squeeze(1).double().numpy()
                    te = model(Xt).squeeze(1).double().numpy()
                trl, tra = bce_and_acc(tr, y)
                tel, tea = bce_and_acc(te, test_y)
                curve.append({"step": step, "trainLoss": trl, "trainAcc": tra, "testLoss": tel, "testAcc": tea})
                if keep_visuals and step in grid_steps:
                    with torch.no_grad():
                        grids.append({"step": step, "p": to_u8(torch.sigmoid(model(G)).squeeze(1).numpy())})
            if step == STEPS:
                break
            opt.zero_grad()
            lossf(model(X).squeeze(1), Y).backward()
            opt.step()
        f = curve[-1]
        entry = {"n": n, "curve": curve, "final": f,
                 "gapAcc": f["trainAcc"] - f["testAcc"], "gapLoss": f["testLoss"] - f["trainLoss"]}
        if keep_visuals:
            with torch.no_grad():
                entry["testPred"] = to_u8(torch.sigmoid(model(Xt[:400])).squeeze(1).numpy())
            entry["grids"] = grids
        results.append(entry)
        print(f"  seed {seed} n={n:5d}: train acc {f['trainAcc']:.3f} loss {f['trainLoss']:.4f} | "
              f"test acc {f['testAcc']:.3f} loss {f['testLoss']:.3f}")
    out = {"seed": seed, "sizes": results}
    if keep_visuals:
        out["pool"] = sig(pool_x, 4)
        out["poolLabels"] = pool_y.astype(int).tolist()
        out["test"] = sig(test_x[:400], 4)
        out["testLabels"] = test_y[:400].astype(int).tolist()
        bt = bayes_posterior(test_x)
        out["bayesTestAcc"] = float(np.mean((bt > 0.5) == (test_y > 0.5)))
        out["bayesGrid"] = to_u8(bayes_posterior(grid_points()))
        # leakage guard: no test point coincides with any pool point
        d = np.abs(test_x[:, None, :] - pool_x[None]).sum(-1).min()
        out["minTestPoolDistance"] = float(d)
    return out


def main():
    print("exp4: generalisation")
    runs = [run(s, keep_visuals=(s == 0)) for s in SEEDS]
    summary = []
    for i, n in enumerate(SIZES):
        ga = np.array([r["sizes"][i]["gapAcc"] for r in runs])
        gl = np.array([r["sizes"][i]["gapLoss"] for r in runs])
        tr = np.array([r["sizes"][i]["final"]["trainAcc"] for r in runs])
        te = np.array([r["sizes"][i]["final"]["testAcc"] for r in runs])
        summary.append({"n": n, "gapAccMean": ga.mean(), "gapAccMin": ga.min(), "gapAccMax": ga.max(),
                        "gapLossMean": gl.mean(), "gapLossMin": gl.min(), "gapLossMax": gl.max(),
                        "trainAccMean": tr.mean(), "testAccMean": te.mean()})
        print(f"  n={n:5d}: gap(acc) mean {ga.mean():.3f} [{ga.min():.3f},{ga.max():.3f}]  "
              f"gap(loss) mean {gl.mean():.3f}")
    write_json("exp4.json", {
        "config": {"sizes": SIZES, "seeds": SEEDS, "nTest": N_TEST, "noise": NOISE, "hidden": HIDDEN,
                   "params": 2 * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN + 1,
                   "steps": STEPS, "lr": LR, "optimizer": "Adam (full batch)", "grid": GRID, "domain": DOMAIN},
        "showcase": runs[0],
        "seedRuns": [{"seed": r["seed"], "sizes": [{k: v for k, v in s.items() if k in ("n", "final", "gapAcc", "gapLoss")}
                                                   for s in r["sizes"]]} for r in runs],
        "summary": summary,
    })


if __name__ == "__main__":
    main()
