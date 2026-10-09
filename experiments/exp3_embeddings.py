"""Experiment 3 - embeddings learn similarity from nothing but next-token prediction.

A tiny made-up language. Sentences come from a handful of templates:
    <det> <animal> eats <some|the> <fruit> .
    <det> <animal> chases <the|a> <animal> .
    <det> <animal> sees <the|a|some> <animal|fruit> .
    the <fruit> is <sweet|ripe|fresh> .
(cows never chase; "a" only goes before animals; "some" only before fruit)

The model is the smallest possible language model:
    next-token logits = W @ E[current token] + b      (embedding -> softmax)
It is trained ONLY on (current token id, next token id) pairs. The word groups
below are used to *write* the sentences and later to *colour/score* the plot,
but they are never given to the model or its loss.

Control: the same words shuffled into random order (so no grammar). If the
groups still appeared there, they would not be coming from context.
"""
import numpy as np
import torch
from torch import nn

from common import checkpoint_steps, seed_all, sig, write_json

SEED = 0
DIM = 4
STEPS = 1200
LR = 0.05
N_SENT = 4000
N_CKPT = 45

GROUPS = {  # used to generate text + to colour / score the plot. NOT a training input.
    "animals": ["cat", "dog", "cow", "horse"],
    "fruits": ["apple", "mango", "banana", "grape"],
    "verbs": ["eats", "chases", "sees"],
    "describing": ["sweet", "ripe", "fresh"],
    "glue": ["the", "a", "some", "is", "."],
}
VOCAB = [w for g in GROUPS.values() for w in g]
IDX = {w: i for i, w in enumerate(VOCAB)}


def make_corpus(n: int, rng: np.random.Generator) -> list[list[str]]:
    A, F = GROUPS["animals"], GROUPS["fruits"]
    pick = lambda xs: xs[rng.integers(len(xs))]
    sents = []
    for _ in range(n):
        t = rng.integers(4)
        if t == 3:
            sents.append(["the", pick(F), "is", pick(GROUPS["describing"]), "."])
            continue
        subj = pick(A)
        det = pick(["the", "a"])
        if t == 1 and subj == "cow":            # cows don't chase things
            t = 0
        if t == 0:
            sents.append([det, subj, "eats", pick(["some", "the"]), pick(F), "."])
        elif t == 1:
            sents.append([det, subj, "chases", pick(["the", "a"]), pick(A), "."])
        else:
            d2 = pick(["the", "a", "some"])
            obj = pick(A) if d2 == "a" else pick(F) if d2 == "some" else pick(A + F)
            sents.append([det, subj, "sees", d2, obj, "."])
    return sents


def pairs_from_stream(tokens: list[int]):
    """(current, next) id pairs from one long token stream - this is ALL the model sees."""
    t = np.array(tokens)
    return t[:-1], t[1:]


class Bigram(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.E = nn.Embedding(V, d)
        self.out = nn.Linear(d, V)
        nn.init.normal_(self.E.weight, 0, 0.3)   # start as a small random cloud

    def forward(self, ids):
        return self.out(self.E(ids))


def train(cur: np.ndarray, nxt: np.ndarray):
    """Takes only integer id arrays. No group information exists in here."""
    seed_all(SEED)
    model = Bigram(len(VOCAB), DIM)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    c, n = torch.tensor(cur), torch.tensor(nxt)
    want = set(checkpoint_steps(STEPS, N_CKPT))
    frames = []
    for step in range(STEPS + 1):
        if step in want:
            with torch.no_grad():
                loss = float(lossf(model(c), n))
            frames.append({"step": step, "loss": loss,
                           "E": model.E.weight.detach().double().numpy().copy(),
                           "W": model.out.weight.detach().double().numpy().copy(),
                           "b": model.out.bias.detach().double().numpy().copy()})
        if step == STEPS:
            break
        opt.zero_grad()
        lossf(model(c), n).backward()
        opt.step()
    return frames


def cosine_matrix(E: np.ndarray) -> np.ndarray:
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    return En @ En.T


def nn_purity(E: np.ndarray) -> dict:
    """Of the 14 words in the four content groups, how many have a nearest
    neighbour (cosine, full 4-number vectors, all 21 words as candidates) from their own group?"""
    S = cosine_matrix(E)
    np.fill_diagonal(S, -np.inf)
    group_of = {w: g for g, ws in GROUPS.items() for w in ws}
    hits, scored = 0, 0
    for w in VOCAB:
        if group_of[w] == "glue":
            continue
        scored += 1
        hits += group_of[VOCAB[int(np.argmax(S[IDX[w]]))]] == group_of[w]
    return {"hits": int(hits), "of": scored}


def leaf_order(E: np.ndarray) -> list[int]:
    """Average-linkage clustering on cosine distance (labels never used) -> a
    row order that puts similar vectors next to each other."""
    D = 1 - cosine_matrix(E)
    clusters = {i: [i] for i in range(len(E))}
    while len(clusters) > 1:
        keys = list(clusters)
        best = None
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = clusters[keys[i]], clusters[keys[j]]
                d = D[np.ix_(a, b)].mean()
                if best is None or d < best[0]:
                    best = (d, keys[i], keys[j])
        _, i, j = best
        clusters[i] = clusters[i] + clusters.pop(j)
    return next(iter(clusters.values()))


def pca_basis(E: np.ndarray):
    mu = E.mean(0)
    U, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
    var = S ** 2
    return mu, Vt[:2], float(var[:2].sum() / var.sum())


def package(frames, label):
    final = frames[-1]["E"]
    mu, basis, kept = pca_basis(final)
    # fix sign so the picture is stable between runs
    for k in range(2):
        if basis[k][np.argmax(np.abs(basis[k]))] < 0:
            basis[k] = -basis[k]
    proj = lambda E: (E - mu) @ basis.T
    return {
        "label": label,
        "pcaVarianceKept": kept,
        "frames": [{"step": f["step"], "loss": f["loss"],
                    "xy": sig(proj(f["E"]), 5), "E": sig(f["E"], 6),
                    "W": sig(f["W"], 6), "b": sig(f["b"], 6),
                    "purity": nn_purity(f["E"])} for f in frames],
        "order": leaf_order(final),
    }


def main():
    print("exp3: embeddings")
    rng = np.random.default_rng(SEED)
    sents = make_corpus(N_SENT, rng)
    stream = [IDX[w] for s in sents for w in s]
    cur, nxt = pairs_from_stream(stream)
    real = package(train(cur, nxt), "grammar")

    shuffled = np.array(stream)[np.random.default_rng(SEED + 1).permutation(len(stream))]
    scur, snxt = pairs_from_stream(shuffled.tolist())
    ctrl = package(train(scur, snxt), "shuffled")

    # empirical next-token table (what a perfect model would predict) for the UI
    counts = np.zeros((len(VOCAB), len(VOCAB)))
    np.add.at(counts, (cur, nxt), 1)
    scounts = np.zeros_like(counts)
    np.add.at(scounts, (scur, snxt), 1)
    probs = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    floor = float(-(counts * np.log(np.where(probs > 0, probs, 1))).sum() / counts.sum())
    print(f"  best possible loss for this language (entropy floor): {floor:.4f}")
    for r in (real, ctrl):
        f0, f1 = r["frames"][0], r["frames"][-1]
        print(f"  {r['label']}: loss {f0['loss']:.3f} -> {f1['loss']:.3f}; "
              f"same-group nearest neighbour {f0['purity']['hits']}/{f0['purity']['of']} -> "
              f"{f1['purity']['hits']}/{f1['purity']['of']}; PCA keeps {r['pcaVarianceKept']:.0%}")
    write_json("exp3.json", {
        "vocab": VOCAB,
        "groups": GROUPS,
        "sentences": [" ".join(s) for s in sents[:60]],
        "nSentences": N_SENT, "nPairs": int(len(cur)),
        "nextCounts": counts.astype(int).tolist(), "nextCountsShuffled": scounts.astype(int).tolist(), "lossFloor": floor,
        "runs": {"grammar": real, "shuffled": ctrl},
        "config": {"dim": DIM, "steps": STEPS, "lr": LR, "optimizer": "Adam (full batch)",
                   "model": "next logits = W·E[token] + b", "seed": SEED},
    })


if __name__ == "__main__":
    main()
