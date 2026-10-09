"""Adversarial checks: try to break every claim the website makes.

Reads the generated artifacts and re-derives the important numbers with
independent code. Every check can fail; the result (pass AND fail) is written to
results/validation.json and shown on the website as-is.
"""
import ast
import inspect
import json
import textwrap
from functools import reduce

import numpy as np

import exp3_embeddings
import exp4_generalization as e4
from common import DATA_DIR, RESULTS_DIR, bce_and_acc, make_rings, mlp_logits

checks = []


def check(exp: int, name: str, ok: bool, detail: str):
    checks.append({"exp": exp, "name": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] E{exp} {name}: {detail}")


def load(n):
    return json.loads((DATA_DIR / n).read_text())


def best_line_random(x, y, n_dirs=20000, seed=7):
    """Independent re-implementation of 'best possible straight line' (random directions)."""
    rng = np.random.default_rng(seed)
    best = 0
    for a in rng.uniform(0, np.pi, n_dirs // 10):
        p = x @ np.array([np.cos(a), np.sin(a)])
        for thr in np.quantile(p, np.linspace(0, 1, 101)):
            acc = np.mean((p > thr) == (y > 0.5))
            best = max(best, acc, 1 - acc)
    return best


def e1():
    d = load("exp1.json")
    worst_gap, max_metric_err = 1.0, 0.0
    for r in d["runs"]:
        x = np.array(r["points"]).reshape(-1, 2)
        y = np.array(r["labels"], float)
        x_regen, y_regen = make_rings(300, r["seed"])
        same = np.allclose(x, x_regen, rtol=1e-4, atol=1e-4) and np.array_equal(y, y_regen)
        check(1, f"seed {r['seed']}: data regenerates from seed", same, "300 points, 150 per class" if same else "mismatch")
        for name, relu in (("linear", False), ("relu", True)):
            for f in r["models"][name]["frames"]:
                loss, acc = bce_and_acc(mlp_logits(x, f["layers"], relu), y)
                max_metric_err = max(max_metric_err, abs(loss - f["loss"]), abs(acc - f["acc"]))
        bl = best_line_random(x, y)
        lin = r["models"]["linear"]["frames"][-1]["acc"]
        rel = r["models"]["relu"]["frames"][-1]["acc"]
        worst_gap = min(worst_gap, rel - max(bl, r["bestLineAcc"]))
        check(1, f"seed {r['seed']}: no straight line can separate the rings",
              max(bl, r["bestLineAcc"]) < 0.8,
              f"best line found: {max(bl, r['bestLineAcc']):.3f} (brute force {r['bestLineAcc']:.3f}, independent random search {bl:.3f})")
        check(1, f"seed {r['seed']}: ReLU beats every possible line", rel > max(bl, r["bestLineAcc"]) + 0.15,
              f"linear {lin:.3f}, ReLU {rel:.3f}")
        xt, _ = make_rings(300, r["seed"] + 1000)
        check(1, f"seed {r['seed']}: test rings are fresh points", not np.any(np.all(np.isclose(x[:, None], xt[None], atol=1e-6), -1)),
              "no test point equals a training point")
        assert len(r["models"]["linear"]["frames"][-1]["layers"]) == 1
    check(1, "displayed metrics recompute from exported weights", max_metric_err < 1e-9, f"max difference {max_metric_err:.1e}")


def e2():
    d = load("exp2.json")
    m = d["models"]
    check(2, "L1 / L5 contain no activation; R5 has ReLU", (not m["L1"]["relu"]) and (not m["L5"]["relu"]) and m["R5"]["relu"],
          f"L5 = {len(m['L5']['frames'][-1]['layers'])} affine layers, R5 = same + 4 ReLU")
    L5 = m["L5"]["frames"][-1]["layers"]
    R5 = m["R5"]["frames"][-1]["layers"]
    # affine maps preserve weighted averages: f(a*u + (1-a)*v) = a f(u) + (1-a) f(v)
    rng = np.random.default_rng(99)
    u, v = rng.uniform(-3, 3, (2, 5000, 2))
    a = rng.uniform(-1, 2, (5000, 1))
    def affine_violation(layers, relu):
        lhs = mlp_logits(a * u + (1 - a) * v, layers, relu)
        rhs = a[:, 0] * mlp_logits(u, layers, relu) + (1 - a[:, 0]) * mlp_logits(v, layers, relu)
        return float(np.abs(lhs - rhs).max())
    vl, vr = affine_violation(L5, False), affine_violation(R5, True)
    check(2, "5 linear layers behave as ONE affine map (functional test)", vl < 1e-9, f"max violation {vl:.1e}")
    check(2, "5 layers + ReLU are NOT affine (test can fail)", vr > 1e-2, f"max violation {vr:.2f}")
    # independent collapse in homogeneous coordinates (3x3-style augmented matrices)
    def aug(L):
        W, b = np.asarray(L["W"]), np.asarray(L["b"])
        M = np.zeros((W.shape[0] + 1, W.shape[1] + 1))
        M[:-1, :-1], M[:-1, -1], M[-1, -1] = W, b, 1
        return M
    M = reduce(lambda acc, L: aug(L) @ acc, L5, np.eye(3))
    c = d["collapse"]
    werr = max(abs(M[0, 0] - c["W"][0]), abs(M[0, 1] - c["W"][1]), abs(M[0, 2] - c["b"]))
    check(2, "collapsed W* and b* (incl. bias) re-derived independently", werr < 1e-12, f"difference {werr:.1e}; W* = [{c['W'][0]:.4g}, {c['W'][1]:.4g}], b* = {c['b']:.4g}")
    probes = rng.uniform(-3.5, 3.5, (20000, 2))
    err = np.abs(mlp_logits(probes, L5, False) - (probes @ M[0, :2] + M[0, 2])).max()
    check(2, "one collapsed layer reproduces the 5-layer stack", err < 1e-10, f"max |difference| over 20,000 inputs = {err:.1e}")
    check(2, "dropping the biases breaks the equivalence (bias handled)", c["maxAbsErrNoBias"] > 1e3 * max(c["maxAbsErr"], 1e-16),
          f"with biases {c['maxAbsErr']:.1e}, without {c['maxAbsErrNoBias']:.1e}")
    l1, l5, r5 = (m[k]["frames"][-1]["acc"] for k in ("L1", "L5", "R5"))
    check(2, "1 layer and 5 linear layers end at the same accuracy", abs(l1 - l5) < 0.02, f"L1 {l1:.3f}, L5 {l5:.3f}, R5 {r5:.3f}")
    check(2, "ReLU stack solves the rings", r5 > 0.95, f"R5 {r5:.3f}")
    i0 = [f["layers"] for f in (m["L5"]["frames"][0], m["R5"]["frames"][0])]
    check(2, "L5 and R5 start from identical weights", i0[0] == i0[1], "same seed, same shapes; only the ReLUs differ")


def e3():
    d = load("exp3.json")
    # every identifier the training path uses (code only - docstrings/comments ignored)
    used = set()
    for fn in (exp3_embeddings.train, exp3_embeddings.Bigram, exp3_embeddings.pairs_from_stream):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    leak = used & {"GROUPS", "group_of", "gof", "groups"}
    check(3, "training code never touches the word groups", not leak,
          "train(), Bigram and pairs_from_stream reference only token ids" if not leak else f"references {sorted(leak)}")
    V, G = d["vocab"], d["groups"]
    gof = {w: g for g, ws in G.items() for w in ws}
    def purity(E):
        E = np.asarray(E).reshape(len(V), -1)
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        S = En @ En.T
        np.fill_diagonal(S, -np.inf)
        return sum(gof[V[int(np.argmax(S[i]))]] == gof[w] for i, w in enumerate(V) if gof[w] != "glue")
    g, s = d["runs"]["grammar"], d["runs"]["shuffled"]
    pg, ps, p0 = purity(g["frames"][-1]["E"]), purity(s["frames"][-1]["E"]), purity(g["frames"][0]["E"])
    check(3, "nearest-neighbour score recomputed from learned vectors", pg == g["frames"][-1]["purity"]["hits"],
          f"{pg}/14 trained vs {p0}/14 at random start")
    check(3, "shuffled-word control does NOT form groups (test can fail)", ps <= pg - 6, f"grammar {pg}/14 vs shuffled {ps}/14")
    # the page explains each "miss" by saying both words are followed by the same things - check that claim
    E = np.asarray(g["frames"][-1]["E"]).reshape(len(V), -1)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = En @ En.T
    np.fill_diagonal(S, -np.inf)
    C = np.asarray(d["nextCounts"], float)
    P = C / C.sum(1, keepdims=True)
    worst_tv = 0.0
    pairs = []
    for i, w in enumerate(V):
        j = int(np.argmax(S[i]))
        if gof[w] != "glue" and gof[V[j]] != gof[w]:
            tv = 0.5 * np.abs(P[i] - P[j]).sum()
            worst_tv = max(worst_tv, tv)
            pairs.append(f"{w}~{V[j]} (TV {tv:.2f})")
    check(3, "every cross-group neighbour really has a similar next-word distribution", worst_tv < 0.25,
          ", ".join(pairs) or "no cross-group neighbours")
    a = {w: V.index(w) for w in ("cat", "dog", "horse", "cow")}
    S2 = En @ En.T
    tight = min(S2[a["cat"], a["dog"]], S2[a["cat"], a["horse"]], S2[a["dog"], a["horse"]])
    cow = max(S2[a["cow"], a[w]] for w in ("cat", "dog", "horse"))
    check(3, "page claim: cow sits apart from cat/dog/horse (cows never chase)", cow < tight,
          f"cow's best animal cosine {cow:.2f} < weakest cat/dog/horse pair {tight:.2f}")
    floor = d["lossFloor"]
    check(3, "model reaches the best possible loss for this language", g["frames"][-1]["loss"] - floor < 0.01,
          f"final {g['frames'][-1]['loss']:.4f} vs floor {floor:.4f}")


def e4c():
    d = load("exp4.json")
    cfg = d["config"]
    for seed in cfg["seeds"]:
        px, py = e4.moons(max(cfg["sizes"]), np.random.default_rng(10_000 + seed))
        tx, ty = e4.moons(cfg["nTest"], np.random.default_rng(20_000 + seed))
        dup = np.any(np.all(np.isclose(px[:, None], tx[None], atol=1e-9), -1))
        if seed == 0:
            sc = d["showcase"]
            match = np.allclose(np.array(sc["pool"]).reshape(-1, 2), px, atol=1e-3) and np.allclose(np.array(sc["test"]).reshape(-1, 2), tx[:400], atol=1e-3)
            check(4, "exported train pool / test set regenerate from their seeds", match, "pool seed 10000+s, test seed 20000+s (separate streams)")
        check(4, f"seed {seed}: held-out set shares no point with the training pool", not dup, "2000 test vs 2000 pool points")
    a, b = e4.net(0), e4.net(0)
    same = all(np.array_equal(p.detach().numpy(), q.detach().numpy()) for p, q in zip(a.parameters(), b.parameters()))
    check(4, "every training size starts from identical weights", same, f"{cfg['params']:,} parameters, same seed per size (also asserted during training)")
    tr20 = [r["sizes"][0]["final"]["trainAcc"] for r in d["seedRuns"]]
    check(4, "tiny data is memorised (n=20 train accuracy = 100%)", min(tr20) == 1.0, f"all 5 seeds: {', '.join(f'{t:.0%}' for t in tr20)}")
    ok = all(r["sizes"][-1]["gapAcc"] < r["sizes"][0]["gapAcc"] and r["sizes"][-1]["gapLoss"] < r["sizes"][0]["gapLoss"] for r in d["seedRuns"])
    sm = d["summary"]
    check(4, "more data closes the gap in every seed", ok, f"mean accuracy gap {sm[0]['gapAccMean']:.3f} -> {sm[-1]['gapAccMean']:.3f}; loss gap {sm[0]['gapLossMean']:.2f} -> {sm[-1]['gapLossMean']:.2f}")
    sizes = [s["n"] for s in d["showcase"]["sizes"]]
    check(4, "only the dataset size changes between runs", sizes == cfg["sizes"], f"same network, optimiser, lr {cfg['lr']}, {cfg['steps']} steps for n = {sizes}")


def main():
    print("validate")
    checks.clear()
    e1(); e2(); e3(); e4c()
    out = {"passed": sum(c["pass"] for c in checks), "total": len(checks), "checks": checks}
    RESULTS_DIR.mkdir(exist_ok=True)
    for p in (RESULTS_DIR / "validation.json", DATA_DIR / "validation.json"):
        p.write_text(json.dumps(out, indent=1))
    print(f"  {out['passed']}/{out['total']} checks passed")
    return out


if __name__ == "__main__":
    main()
