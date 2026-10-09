"""Experiment 2 - depth without nonlinearity is a lie.

Same ring data as experiment 1 (seed 0). Three models, same optimiser/steps:
  L1  : Linear(2,1)
  L5  : Linear(2,16) Linear(16,16) Linear(16,16) Linear(16,16) Linear(16,1)   (no activations)
  R5  : the same five layers with ReLU between them
L5 and R5 start from the *identical* initial weights, so the only difference
between them is the four ReLUs.

Then the proof: multiply the five trained L5 layers out into ONE affine map
    W* = W5 W4 W3 W2 W1
    b* = W5(W4(W3(W2 b1 + b2) + b3) + b4) + b5
and check the single layer reproduces the five-layer stack on many inputs.
"""
import numpy as np
import torch
from torch import nn

from common import (bce_and_acc, checkpoint_steps, export_linear_stack, make_rings,
                    mlp_logits, rounded_array, seed_all, sig, write_json)

SEED = 0
WIDTH = 16
STEPS = 1500
LR = 0.01
N_CKPT = 30


def five(relu: bool) -> nn.Sequential:
    seed_all(SEED)                                   # identical init for L5 and R5
    dims = [2, WIDTH, WIDTH, WIDTH, WIDTH, 1]
    mods = []
    for i in range(5):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if relu and i < 4:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


def collapse(layers: list[dict], use_bias: bool = True):
    """Fold a list of affine layers into one (float64). Biases optional for the 'forgot b' demo."""
    W = np.eye(2)
    b = np.zeros(2)
    for L in layers:
        Wi, bi = np.asarray(L["W"]), np.asarray(L["b"])
        W = Wi @ W
        b = Wi @ b + (bi if use_bias else 0)
    return W, b


def main():
    print("exp2: depth")
    x, y = make_rings(300, SEED)
    x = rounded_array(x, 6)                  # train + score on exactly what we export
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    seed_all(SEED)
    models = {"L1": (nn.Sequential(nn.Linear(2, 1)), False), "L5": (five(False), False), "R5": (five(True), True)}

    # Hard guard: the "linear" stacks must contain nothing but nn.Linear.
    assert all(isinstance(m, nn.Linear) for m in models["L1"][0])
    assert all(isinstance(m, nn.Linear) for m in models["L5"][0])
    assert sum(isinstance(m, nn.ReLU) for m in models["R5"][0]) == 4
    init_L5 = export_linear_stack(models["L5"][0])
    init_R5 = export_linear_stack(models["R5"][0])
    assert init_L5 == init_R5, "L5 and R5 must start from identical weights"

    want = set(checkpoint_steps(STEPS, N_CKPT))
    out = {"models": {}}
    for name, (model, relu) in models.items():
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss()
        frames = []
        for step in range(STEPS + 1):
            if step in want:
                layers = export_linear_stack(model, 6)
                loss, acc = bce_and_acc(mlp_logits(x, layers, relu), y)
                frames.append({"step": step, "loss": loss, "acc": acc, "layers": layers})
            if step == STEPS:
                break
            opt.zero_grad()
            lossf(model(xt).squeeze(1), yt).backward()
            opt.step()
        out["models"][name] = {"relu": relu, "frames": frames,
                               "params": sum(p.numel() for p in model.parameters())}
        print(f"  {name}: acc {frames[-1]['acc']:.3f}  loss {frames[-1]['loss']:.4f}")

    # ---- the collapse proof on the final, exported L5 weights (float64 maths)
    L5 = out["models"]["L5"]["frames"][-1]["layers"]
    W, b = collapse(L5)
    Wnb, bnb = collapse(L5, use_bias=False)
    rng = np.random.default_rng(123)
    probes = rng.uniform(-3.5, 3.5, (10000, 2))
    stack = mlp_logits(probes, L5, relu=False)
    single = probes @ W[0] + b[0]
    nobias = probes @ Wnb[0] + bnb[0]
    err = np.abs(stack - single)
    # same check on the R5 network: if we pretend its ReLUs are not there, does
    # the collapsed map still match the real network? (It must NOT.)
    R5 = out["models"]["R5"]["frames"][-1]["layers"]
    Wr, br = collapse(R5)
    r5_err = np.abs(mlp_logits(probes, R5, relu=True) - (probes @ Wr[0] + br[0]))
    single_layers = [{"W": W.tolist(), "b": b.tolist()}]
    acc_single = bce_and_acc(mlp_logits(x, single_layers, False), y)[1]
    acc_stack = out["models"]["L5"]["frames"][-1]["acc"]
    show = probes[:6]
    out["collapse"] = {
        "W": W[0].tolist(), "b": float(b[0]),
        "Wnobias": Wnb[0].tolist(), "bnobias": float(bnb[0]),
        "maxAbsErr": float(err.max()), "meanAbsErr": float(err.mean()),
        "maxAbsErrNoBias": float(np.abs(stack - nobias).max()),
        "maxAbsErrR5": float(r5_err.max()),
        "nProbes": len(probes), "probeRange": [-3.5, 3.5],
        "accStack": acc_stack, "accSingle": acc_single,
        "examples": [{"x": sig(p, 6), "stack": float(s), "single": float(c)}
                     for p, s, c in zip(show, stack[:6], single[:6])],
    }
    c = out["collapse"]
    print(f"  collapse: max|stack-single| = {c['maxAbsErr']:.2e}; without biases = {c['maxAbsErrNoBias']:.3f}; "
          f"R5 pretend-linear = {c['maxAbsErrR5']:.3f}; acc stack {acc_stack:.3f} vs single {acc_single:.3f}")
    out["config"] = {"seed": SEED, "width": WIDTH, "steps": STEPS, "lr": LR,
                     "optimizer": "Adam (full batch)", "sameInitL5R5": True}
    out["points"] = sig(x, 6)
    out["labels"] = y.astype(int).tolist()
    write_json("exp2.json", out)
    return out


if __name__ == "__main__":
    main()
