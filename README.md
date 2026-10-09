# Neural Network Lab

An interactive web page that tests four basic claims about neural networks with small, real experiments. Every boundary, dot, curve and number on the page is read from JSON files written by the PyTorch experiment scripts. Nothing is typed in by hand: the build fails if a percentage appears directly in the HTML.

Built as ERA V5 Session 1, Assignment 1.

## The four claims

1. **Activations exist for a reason.** A linear model can only draw a straight boundary, so it fails on two concentric rings. One ReLU hidden layer wraps them. Across 5 seeds the linear model ends at 50–56% training accuracy, the ReLU model at 99.7–100%. A brute-force search shows that the best possible straight line reaches only 68–72%.
2. **Depth without nonlinearity collapses.** Five stacked linear layers are exactly one affine layer, biases included. I multiply the five layers out and the collapsed layer reproduces the stack to within 3.3e-16 over 20,000 inputs. One linear layer and five linear layers both end at 50.3% on the rings, while the same five layers with ReLU reach 100%.
3. **Embeddings learn similarity from next-token prediction.** A tiny embedding-to-softmax model trained only on (word, next word) pairs from a toy grammar groups animals, fruits, verbs and describing words. Same-group nearest neighbours go from 3/14 at initialization to 13/14 after training. A shuffled-word control trained the same way ends at 1/14.
4. **Memorization vs generalization.** The same 17,025-parameter network reaches 100% training accuracy on 20 points in all 5 seeds, with 79.7–86.6% held-out accuracy. As the training set grows to 2,000 points the mean accuracy gap shrinks from 0.160 to 0.018.

## Layout

```
experiments/                PyTorch experiments (CPU, fixed seeds)
  common.py                 shared data and helpers
  exp1_activations.py       rings: linear vs one ReLU hidden layer, 5 datasets
  exp2_depth.py             1 linear / 5 linear / 5 + ReLU, plus the affine-collapse proof
  exp3_embeddings.py        toy grammar, next-token embedding model, shuffled control
  exp4_generalization.py    two moons, n = 20 ... 2000, 5 seeds, held-out set
  validate.py               checks that can fail -> results/validation.json
  run_all.py                regenerate everything, then validate
results/                    validation.json (46/46 checks) and the run log
web/                        Vite + TypeScript site (Canvas + SVG, no framework)
  public/data/              experiment outputs the page reads
  scripts/verify-data.mjs   independent JS re-check of the data, runs on every build
netlify.toml                build settings (base = web, publish = web/dist)
```

## Run it

```bash
pip install -r requirements.txt
python experiments/run_all.py          # about 3-4 minutes on a laptop CPU, rewrites web/public/data
cd web && npm install && npm run build # type-check, bundle, then verify-data.mjs
npm run preview                        # serves web/dist
```

The committed JSON in `web/public/data/` is enough to build and view the site without rerunning the experiments.

## Notes and limitations

- Training uses Adam with full batches.
- Exp 1: the linear model sits near chance (50–56%). The 68–72% best-line figure is the ceiling for any straight line on that data, found by brute-force search.
- Exp 3: one content word ("chases") has "." as its nearest neighbour. In this grammar both are followed by *the* / *a*, so a next-token model is right to treat them as similar. The 2-D map is a PCA projection. All similarity numbers use the full 4-dimensional vectors.
- Exp 4: the "best possible" accuracy of 92.6% comes from the true data-generating distribution.
- Playback cross-fades the colour field between saved snapshots for smoothness. Boundaries, points and numbers always show a real saved snapshot.
