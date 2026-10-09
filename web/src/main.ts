import "./style.css";
import { $, $$, C, bind, contour, fieldCanvas, h, loadJSON, makeNet, pct, probGrid, reducedMotion, rgba, sci, strokeSegs } from "./lib";
import { initExp1 } from "./exp1";
import { initExp2 } from "./exp2";
import { GROUP_COLORS, initExp3, nearestInfo } from "./exp3";
import { decodeGrid, initExp4 } from "./exp4";

const range = (xs: number[], f = (v: number) => pct(v)) => {
  const lo = Math.min(...xs), hi = Math.max(...xs);
  return lo === hi ? f(lo) : `${f(lo)} – ${f(hi)}`;
};

async function main() {
  const [e1, e2, e3, e4, val] = await Promise.all(["exp1.json", "exp2.json", "exp3.json", "exp4.json", "validation.json"].map((n) => loadJSON<any>(n)));

  // ---------------- numbers used in the prose (all read from the artifacts)
  const r0 = e1.runs[0];
  const L1 = e1.runs.map((r: any) => r.models.linear.frames.at(-1));
  const R1 = e1.runs.map((r: any) => r.models.relu.frames.at(-1));
  const m2 = (k: string) => e2.models[k].frames.at(-1);
  const g3 = e3.runs.grammar.frames, s3 = e3.runs.shuffled.frames;
  const sz = e4.showcase.sizes;
  const sm = e4.summary;
  const V3 = e3.vocab.length;
  const groupOf: Record<string, string> = {};
  for (const [g, ws] of Object.entries(e3.groups) as [string, string[]][]) ws.forEach((w) => (groupOf[w] = g));
  const { S } = nearestInfo(g3.at(-1).E, V3, e3.config.dim);
  const misses = e3.vocab.map((w: string, i: number) => {
    if (groupOf[w] === "glue") return null;
    let j = -1, best = -Infinity;
    S[i].forEach((v, k) => { if (k !== i && v > best) { best = v; j = k; } });
    return groupOf[e3.vocab[j]] === groupOf[w] ? null : [w, e3.vocab[j]];
  }).filter(Boolean) as [string, string][];
  const nModels = e1.runs.length * 2 + 3 + 2 + e4.seedRuns.length * e4.config.sizes.length;

  bind({
    nModels: String(nModels),
    "e1.hidden": String(e1.config.hidden),
    "e1.linAcc": pct(r0.models.linear.frames.at(-1).acc),
    "e1.reluAcc": pct(r0.models.relu.frames.at(-1).acc),
    "e1.reluTest": pct(r0.models.relu.frames.at(-1).testAcc),
    "e1.bestLine": pct(r0.bestLineAcc),
    "e2.l5params": String(e2.models.L5.params),
    "e2.l1acc": pct(m2("L1").acc), "e2.l1loss": m2("L1").loss.toFixed(4),
    "e2.l5acc": pct(m2("L5").acc), "e2.l5loss": m2("L5").loss.toFixed(4),
    "e2.r5acc": pct(m2("R5").acc),
    "e2.err": sci(e2.collapse.maxAbsErr),
    "e3.V": String(V3), "e3.nSent": e3.nSentences.toLocaleString(), "e3.dim": String(e3.config.dim),
    "e3.purity": `${g3.at(-1).purity.hits}/14`, "e3.purity0": `${g3[0].purity.hits}/14`, "e3.purityShuf": `${s3.at(-1).purity.hits}/14`,
    "e3.missText": misses.length
      ? `${misses.length === 1 ? "The one word that misses" : `The ${misses.length} words that miss`}: ${misses.map(([a, b]) => `“${a}” ends up closest to “${b}”`).join("; ")}.`
      : "Every content word's nearest neighbour is from its own group.",
    "e4.params": e4.config.params.toLocaleString(), "e4.nTest": e4.config.nTest.toLocaleString(),
    "e4.tr20": pct(sz[0].final.trainAcc), "e4.trl20": sz[0].final.trainLoss.toFixed(4),
    "e4.te20": pct(sz[0].final.testAcc), "e4.tel20": sz[0].final.testLoss.toFixed(2),
    "e4.tr2000": pct(sz.at(-1).final.trainAcc), "e4.te2000": pct(sz.at(-1).final.testAcc),
    "e4.bayes": pct(e4.showcase.bayesTestAcc),
    "e4.gap20": `${(sm[0].gapAccMean * 100).toFixed(1)} pts`, "e4.gap2000": `${(sm.at(-1).gapAccMean * 100).toFixed(1)} pts`,
    "e4.lgap20": sm[0].gapLossMean.toFixed(2), "e4.lgap2000": sm.at(-1).gapLossMean.toFixed(2),
    checksSummary: `${val.passed} / ${val.total} passed`,
  });

  // ---------------- experiments (each isolated so one failure can't blank the page)
  for (const [name, fn] of [["exp1", () => initExp1(e1)], ["exp2", () => initExp2(e2)], ["exp3", () => initExp3(e3)], ["exp4", () => initExp4(e4)]] as const) {
    try { fn(); } catch (err) { console.error(name, err); }
  }

  // ---------------- evidence table
  const rows: [string, string, string, string][] = [
    ["01 Activations", "Linear model, 5 datasets", `train ${range(L1.map((f: any) => f.acc))}`, `best possible line ${range(e1.runs.map((r: any) => r.bestLineAcc))}`],
    ["", "1 hidden layer + ReLU", `train ${range(R1.map((f: any) => f.acc))}`, `fresh test points ${range(R1.map((f: any) => f.testAcc))}`],
    ["02 Depth", "1 linear · 5 linear · 5 + ReLU", `${pct(m2("L1").acc)} · ${pct(m2("L5").acc)} · ${pct(m2("R5").acc)}`, `loss ${m2("L1").loss.toFixed(4)} · ${m2("L5").loss.toFixed(4)} · ${m2("R5").loss.toFixed(4)}`],
    ["", "5 layers collapsed to 1", `max error ${sci(e2.collapse.maxAbsErr)} over ${e2.collapse.nProbes.toLocaleString()} inputs`, `without biases: ${sci(e2.collapse.maxAbsErrNoBias)}`],
    ["03 Embeddings", "Nearest neighbour in own group", `${g3[0].purity.hits}/14 → ${g3.at(-1).purity.hits}/14`, `shuffled control: ${s3.at(-1).purity.hits}/14`],
    ["", "Next-token loss", `${g3[0].loss.toFixed(3)} → ${g3.at(-1).loss.toFixed(4)}`, `best possible ${e3.lossFloor.toFixed(4)}`],
    ["04 Generalization", "n = 20 (seed 1 of 5)", `train ${pct(sz[0].final.trainAcc)} / held-out ${pct(sz[0].final.testAcc)}`, `loss ${sz[0].final.trainLoss.toFixed(4)} / ${sz[0].final.testLoss.toFixed(2)}`],
    ["", "n = 2000 (seed 1 of 5)", `train ${pct(sz.at(-1).final.trainAcc)} / held-out ${pct(sz.at(-1).final.testAcc)}`, `best possible ≈ ${pct(e4.showcase.bayesTestAcc)}`],
    ["", "Mean gap, 5 seeds", `${(sm[0].gapAccMean * 100).toFixed(1)} → ${(sm.at(-1).gapAccMean * 100).toFixed(1)} accuracy pts`, `loss gap ${sm[0].gapLossMean.toFixed(2)} → ${sm.at(-1).gapLossMean.toFixed(2)}`],
  ];
  const table = $("#results");
  table.innerHTML = `<thead><tr><th>Experiment</th><th>What</th><th>Measured</th><th>Context</th></tr></thead>`;
  const tb = h("tbody");
  rows.forEach(([a, b, c, d]) => tb.append(h("tr", {}, h("td", {}, a), h("td", {}, b), h("td", { class: "n" }, c), h("td", { class: "n", style: "color:var(--ink2)" }, d))));
  table.append(tb);
  const checks = $("#checks");
  val.checks.forEach((c: any) => checks.append(h("div", { class: "check" },
    h("span", { class: `st ${c.pass ? "pass" : "fail"}` }, c.pass ? "PASS" : "FAIL"),
    h("div", {}, h("b", {}, `E${c.exp} · ${c.name}`), c.detail))));
  $("#methods").innerHTML = [
    `<b>Everything</b>: PyTorch on CPU, fixed seeds, full-batch training with Adam (a standard variant of gradient descent), binary/softmax cross-entropy loss. Each experiment script writes the JSON this page reads; <span class="mono">experiments/validate.py</span> re-derives the claims with independent code.`,
    `<b>01</b>: ${e1.config.points} points (inner radius ${e1.config.ring.innerRadius}, outer ${e1.config.ring.outerRadius}, noise ${e1.config.ring.noise}); linear 2→1 vs 2→${e1.config.hidden}→1 ReLU; ${e1.config.steps} steps, lr ${e1.config.lr}; 5 datasets (seeds 0–4); 300 extra fresh points per dataset for testing. “Best possible line” = brute force over 1,440 directions × every threshold.`,
    `<b>02</b>: same ring data (seed 0); 2→1, 2→16→16→16→16→1 without activations (${e2.models.L5.params} params), and the identical stack with 4 ReLUs, starting from identical weights; ${e2.config.steps} steps, lr ${e2.config.lr}. Collapse: W* = W5W4W3W2W1, b* = W5(W4(W3(W2b1+b2)+b3)+b4)+b5 in float64.`,
    `<b>03</b>: ${V3}-word vocabulary, ${e3.nSentences.toLocaleString()} template sentences → ${e3.nPairs.toLocaleString()} (word, next word) pairs. Model: next-word scores = W·E[word] + b, embedding size ${e3.config.dim}, ${e3.config.steps} steps, lr ${e3.config.lr}. Word groups are used only to write sentences and to colour/score the plot, never in training. 2-D map = PCA of the final embeddings (same projection for every snapshot). Control: same tokens in shuffled order.`,
    `<b>04</b>: two moons, noise ${e4.config.noise}; training sets are nested prefixes (${e4.config.sizes.join(", ")}) of one 2,000-point pool; held-out set = ${e4.config.nTest.toLocaleString()} points from a separate random stream. Network 2→128→128→1 ReLU (${e4.config.params.toLocaleString()} params), identical starting weights for every size, ${e4.config.steps} full-batch steps, lr ${e4.config.lr}, no regularization. 5 seeds. “Best possible” = accuracy of the true generating distribution's own posterior.`,
    `<b>Animation honesty</b>: scrubbers snap to saved training snapshots. During playback the colour field cross-fades between neighbouring snapshots for smoothness; boundaries, dots and numbers always show a real snapshot.`,
  ].map((t) => `<p>${t}</p>`).join("");

  // ---------------- thumbnails on the hero cards
  thumbs(e1, e2, e3, e4);
  hero(e1);
}

function thumbs(e1: any, e2: any, e3: any, e4: any) {
  const D = { x: [-3.4, 3.4] as [number, number], y: [-3.4, 3.4] as [number, number] };
  const prep = (cv: HTMLCanvasElement) => {
    const w = cv.clientWidth || 280, H = Math.round(w / 1.6);
    const dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = w * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, H);
    return { ctx, w, H };
  };
  const ring = (cv: HTMLCanvasElement, layers: any, relu: boolean, color: string, pts: number[], labels: number[]) => {
    const { ctx, w, H } = prep(cv);
    const n = 70, f = makeNet(layers, relu), g = probGrid(f, D, n);
    const side = H, x0 = (w - side) / 2;
    ctx.drawImage(fieldCanvas(g, n, n), x0, 0, side, side);
    ctx.save(); ctx.translate(x0, 0);
    strokeSegs(ctx, contour(g, n, n), side / (n - 1), side / (n - 1), color, 2);
    labels.forEach((c, i) => { ctx.fillStyle = c ? C.c1 : C.c0; ctx.beginPath(); ctx.arc(((pts[2 * i] + 3.4) / 6.8) * side, (1 - (pts[2 * i + 1] + 3.4) / 6.8) * side, 1.6, 0, 7); ctx.fill(); });
    ctx.restore();
  };
  const [c1, c2, c3, c4] = [1, 2, 3, 4].map((i) => $<HTMLCanvasElement>(`canvas[data-thumb="${i}"]`));
  const r0 = e1.runs[0];
  ring(c1, r0.models.relu.frames.at(-1).layers, true, C.relu, r0.points, r0.labels);
  ring(c2, e2.models.L5.frames.at(-1).layers, false, C.lin, e2.points, e2.labels);
  { // embeddings
    const { ctx, w, H } = prep(c3);
    const cvw = w;
    const fr = e3.runs.grammar.frames.at(-1), V = e3.vocab.length;
    const xs = Array.from({ length: V }, (_, i) => fr.xy[2 * i]), ys = Array.from({ length: V }, (_, i) => fr.xy[2 * i + 1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const P = xs.map((x, i) => [16 + ((x - x0) / (x1 - x0)) * (w - 32), H - 14 - ((ys[i] - y0) / (y1 - y0)) * (H - 28)]);
    const gOf: Record<string, string> = {};
    for (const [g, ws] of Object.entries(e3.groups) as [string, string[]][]) ws.forEach((x) => (gOf[x] = g));
    const { S } = nearestInfo(fr.E, V, e3.config.dim);
    ctx.lineWidth = 1;
    for (let i = 0; i < V; i++) for (let j = i + 1; j < V; j++) if (S[i][j] > 0.8) { ctx.strokeStyle = rgba(GROUP_COLORS[gOf[e3.vocab[i]]], 0.5); ctx.beginPath(); ctx.moveTo(P[i][0], P[i][1]); ctx.lineTo(P[j][0], P[j][1]); ctx.stroke(); }
    P.forEach(([x, y], i) => { ctx.fillStyle = GROUP_COLORS[gOf[e3.vocab[i]]]; ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 7); ctx.fill(); });
    ctx.font = "600 10px Inter, sans-serif";
    for (const w of ["cat", "mango", "sweet", "eats", "the"]) {   // a few labels so the thumbnail reads as words
      const i = e3.vocab.indexOf(w);
      ctx.fillStyle = rgba(GROUP_COLORS[gOf[w]], 0.95);
      ctx.fillText(w, Math.min(P[i][0] + 6, cvw - ctx.measureText(w).width - 4), P[i][1] - 5);
    }
  }
  { // generalization
    const { ctx, w, H } = prep(c4);
    const n = e4.config.grid, z = e4.showcase.sizes[0], g = decodeGrid(z.grids.at(-1).p, n);
    ctx.drawImage(fieldCanvas(g, n, n), 0, 0, w, H);
    strokeSegs(ctx, contour(g, n, n), w / (n - 1), H / (n - 1), "#fff", 1.6);
    const d = e4.config.domain;
    for (let i = 0; i < z.n; i++) {
      const x = ((e4.showcase.pool[2 * i] - d.x[0]) / (d.x[1] - d.x[0])) * w, y = (1 - (e4.showcase.pool[2 * i + 1] - d.y[0]) / (d.y[1] - d.y[0])) * H;
      ctx.fillStyle = e4.showcase.poolLabels[i] ? C.c1 : C.c0; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
    }
  }
}

/** Hero background: the experiment-1 ring points, slowly orbiting with mouse parallax. */
function hero(e1: any) {
  const cv = $<HTMLCanvasElement>(".hero canvas.bgfx");
  const ctx = cv.getContext("2d")!;
  const r0 = e1.runs[0];
  const pts = r0.labels.map((c: number, i: number) => ({ x: r0.points[2 * i], y: r0.points[2 * i + 1], c, z: 0.4 + Math.random() * 0.8 }));
  let mx = 0, my = 0, visible = true, w = 0, H = 0;
  const size = () => {
    const dpr = Math.min(2, devicePixelRatio || 1);
    w = cv.clientWidth; H = cv.clientHeight;
    cv.width = w * dpr; cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  size();
  addEventListener("resize", size);
  addEventListener("pointermove", (e) => { mx = e.clientX / innerWidth - 0.5; my = e.clientY / innerHeight - 0.5; });
  new IntersectionObserver((es) => (visible = es[0].isIntersecting)).observe(cv);
  const frame = (t: number) => {
    if (visible) {
      ctx.clearRect(0, 0, w, H);
      const R = Math.min(w * 0.42, H) / 6.2, cx = w > 900 ? w * 0.74 : w * 0.5, cy = H * 0.42;
      const a = reducedMotion ? 0.3 : t / 22000;
      for (const p of pts) {
        const ang = a * (p.c ? 1 : -1.6);
        const x = p.x * Math.cos(ang) - p.y * Math.sin(ang), y = p.x * Math.sin(ang) + p.y * Math.cos(ang);
        ctx.fillStyle = rgba(p.c ? C.c1 : C.c0, 0.22 + 0.4 * (p.z - 0.4));
        ctx.beginPath(); ctx.arc(cx + x * R - mx * 30 * p.z, cy + y * R - my * 30 * p.z, 1.2 + 1.6 * p.z, 0, 7); ctx.fill();
      }
    }
    if (!reducedMotion) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// ---------------- page chrome: nav, progress, reveal-on-scroll
function chrome() {
  const nav = $(".topnav"), bar = $(".progress", nav);
  const links = $$<HTMLAnchorElement>(".chapters a", nav);
  const onScroll = () => {
    nav.classList.toggle("on", scrollY > innerHeight * 0.55);
    bar.style.width = `${(scrollY / (document.documentElement.scrollHeight - innerHeight)) * 100}%`;
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) links.forEach((l) => l.classList.toggle("cur", l.getAttribute("href") === `#${e.target.id}`));
  }), { rootMargin: "-45% 0px -50% 0px" });
  $$("section.chapter").forEach((s) => io.observe(s));
  const rv = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); rv.unobserve(e.target); } }), { threshold: 0.08 });
  $$(".reveal").forEach((el) => (reducedMotion ? el.classList.add("in") : rv.observe(el)));
}

chrome();
main().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("afterbegin", `<p style="padding:20px;color:#ff5c8a">Could not load experiment data: ${String(err)}</p>`);
});
