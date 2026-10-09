// Re-checks the website's data with JavaScript maths (independent of Python):
// the numbers the page displays must equal what the exported weights produce.
// Also fails if index.html contains hand-typed measurement numbers.
import { readFileSync, existsSync } from "node:fs";

const dir = new URL("../public/data/", import.meta.url);
const read = (n) => JSON.parse(readFileSync(new URL(n, dir), "utf8"));
let fails = 0;
const check = (name, ok, detail) => {
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? " — " + detail : ""}`);
  if (!ok) fails++;
};

const forward = (layers, relu, x, y) => {
  let h = [x, y];
  layers.forEach((L, li) => {
    h = L.W.map((row, o) => row.reduce((a, w, i) => a + w * h[i], L.b[o]));
    if (relu && li < layers.length - 1) h = h.map((v) => Math.max(0, v));
  });
  return h[0];
};
const scoreAcc = (layers, relu, pts, labels) =>
  labels.reduce((a, c, i) => a + ((forward(layers, relu, pts[2 * i], pts[2 * i + 1]) > 0) === (c === 1)), 0) / labels.length;

// ---- exp1: every displayed accuracy recomputes from weights
const e1 = read("exp1.json");
let worst = 0;
for (const r of e1.runs) for (const k of ["linear", "relu"]) for (const f of r.models[k].frames)
  worst = Math.max(worst, Math.abs(scoreAcc(f.layers, k === "relu", r.points, r.labels) - f.acc));
check("exp1 accuracies recompute in JS from exported weights", worst === 0, `max diff ${worst}`);
check("exp1 linear model has exactly one layer", e1.runs.every((r) => r.models.linear.frames.every((f) => f.layers.length === 1)));

// ---- exp2: accuracies + collapse identity in JS
const e2 = read("exp2.json");
let w2 = 0;
for (const k of ["L1", "L5", "R5"]) for (const f of e2.models[k].frames)
  w2 = Math.max(w2, Math.abs(scoreAcc(f.layers, e2.models[k].relu, e2.points, e2.labels) - f.acc));
check("exp2 accuracies recompute in JS", w2 === 0, `max diff ${w2}`);
let maxErr = 0;
for (const f of e2.models.L5.frames) {
  let W = [[1, 0], [0, 1]], b = [0, 0];
  for (const L of f.layers) {
    W = L.W.map((row) => [0, 1].map((j) => row.reduce((a, w, k) => a + w * W[k][j], 0)));
    b = L.W.map((row, o) => row.reduce((a, w, k) => a + w * b[k], 0) + L.b[o]);
  }
  for (let i = 0; i < 400; i++) {
    const x = -3.5 + 7 * ((i * 0.618) % 1), y = -3.5 + 7 * ((i * 0.377) % 1);
    maxErr = Math.max(maxErr, Math.abs(forward(f.layers, false, x, y) - (W[0][0] * x + W[0][1] * y + b[0])));
  }
}
check("exp2 collapse holds at EVERY training snapshot (JS, with biases)", maxErr < 1e-9, `max err ${maxErr.toExponential(2)}`);
check("exp2 L5 and R5 have identical starting weights", JSON.stringify(e2.models.L5.frames[0].layers) === JSON.stringify(e2.models.R5.frames[0].layers));

// ---- exp3: purity recomputes from embeddings
const e3 = read("exp3.json");
const gof = {};
for (const [g, ws] of Object.entries(e3.groups)) ws.forEach((w) => (gof[w] = g));
const V = e3.vocab.length, d = e3.config.dim;
const purity = (E) => {
  const v = (i) => E.slice(i * d, i * d + d), n = (a) => Math.hypot(...a);
  let hits = 0;
  e3.vocab.forEach((w, i) => {
    if (gof[w] === "glue") return;
    let best = -2, bj = -1;
    for (let j = 0; j < V; j++) if (j !== i) {
      const c = v(i).reduce((a, x, k) => a + x * v(j)[k], 0) / (n(v(i)) * n(v(j)));
      if (c > best) { best = c; bj = j; }
    }
    hits += gof[e3.vocab[bj]] === gof[w];
  });
  return hits;
};
let pm = 0;
for (const run of Object.values(e3.runs)) for (const f of run.frames) pm += purity(f.E) !== f.purity.hits;
check("exp3 neighbour scores recompute in JS for every snapshot", pm === 0, `${pm} mismatches`);

// ---- exp4: grids decode, gaps consistent, summary = mean of seeds
const e4 = read("exp4.json");
const N = e4.config.grid;
const gridsOk = e4.showcase.sizes.every((s) => s.grids.every((g) => Buffer.from(g.p, "base64").length === N * N));
check("exp4 every decision grid decodes to the right size", gridsOk);
const gapOk = e4.showcase.sizes.every((s) => Math.abs(s.gapAcc - (s.final.trainAcc - s.final.testAcc)) < 1e-12);
check("exp4 displayed gap = train acc - held-out acc", gapOk);
const sumOk = e4.summary.every((row, i) => {
  const m = e4.seedRuns.reduce((a, r) => a + r.sizes[i].gapAcc, 0) / e4.seedRuns.length;
  return Math.abs(m - row.gapAccMean) < 1e-12;
});
check("exp4 summary means = mean over the 5 seed runs", sumOk);
const predOk = e4.showcase.sizes.every((s) => {
  const p = Buffer.from(s.testPred, "base64");
  const acc = e4.showcase.testLabels.reduce((a, c, i) => a + ((p[i] > 127) === (c === 1)), 0) / p.length;
  return Math.abs(acc - s.final.testAcc) < 0.04;   // 400-point display subset vs full 2000-point test set
});
check("exp4 displayed held-out predictions agree with reported test accuracy", predOk);

// ---- no hand-typed results in the page
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8").replace(/<[^>]+>/g, " ");
const typed = html.match(/\d+(\.\d+)?\s?%/g) || [];
check("index.html has no hard-coded percentages (all bound from data)", typed.length === 0, typed.join(", "));

const val = read("validation.json");
check("python validation file present and all passing", val.passed === val.total, `${val.passed}/${val.total}`);
if (existsSync(new URL("../dist/index.html", import.meta.url))) check("production build contains data", existsSync(new URL("../dist/data/exp4.json", import.meta.url)));

if (fails) { console.error(`\n${fails} check(s) failed`); process.exit(1); }
console.log("\nall web data checks passed");
