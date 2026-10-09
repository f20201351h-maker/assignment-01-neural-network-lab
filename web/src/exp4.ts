// Experiment 4 — memorization vs generalization, and data closes the gap
import { $, C, Domain, LineChart, Player, Stage, contour, drawDot, ease, fieldCanvas, h, hideTip, mapper, onceVisible, pct, reducedMotion, rgba, s, showTip, strokeSegs, tweenText } from "./lib";

type Pt = { step: number; trainLoss: number; trainAcc: number; testLoss: number; testAcc: number };
type Size = { n: number; curve: Pt[]; final: Pt; gapAcc: number; gapLoss: number; testPred: string; grids: { step: number; p: string }[] };
type Data = {
  config: { sizes: number[]; nTest: number; params: number; steps: number; grid: number; domain: Domain; seeds: number[] };
  showcase: { pool: number[]; poolLabels: number[]; test: number[]; testLabels: number[]; sizes: Size[]; bayesGrid: string; bayesTestAcc: number };
  summary: { n: number; gapAccMean: number; gapAccMin: number; gapAccMax: number; gapLossMean: number; gapLossMin: number; gapLossMax: number }[];
  seedRuns: { seed: number; sizes: { n: number; gapAcc: number; gapLoss: number }[] }[];
};

const TRUE_COLOR = "#fff27a";

/** base64 bytes -> probabilities, flipped so row 0 is the TOP of the plot. */
export function decodeGrid(b64: string, n: number) {
  const bin = atob(b64);
  const g = new Float32Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) g[(n - 1 - r) * n + c] = bin.charCodeAt(r * n + c) / 255;
  return g;
}

export function initExp4(data: Data) {
  const root = $("#ch4");
  const cfg = data.config, sc = data.showcase, N = cfg.grid, D = cfg.domain;
  const aspect = (D.x[1] - D.x[0]) / (D.y[1] - D.y[0]);
  const pool = sc.poolLabels.map((c, i) => ({ x: sc.pool[2 * i], y: sc.pool[2 * i + 1], c }));
  const test = sc.testLabels.map((c, i) => ({ x: sc.test[2 * i], y: sc.test[2 * i + 1], c }));
  let si = 0;                       // selected size index
  let gi = sc.sizes[0].grids.length - 1;
  let frac = 0;
  let showTest = true, showTrue = false;
  let dropT = 1, dropFrom = 0;      // new-points animation
  let fadeFrom: HTMLCanvasElement | null = null, fadeT = 1;

  const cache = new Map<string, { cv: HTMLCanvasElement; segs: number[] }>();
  const field = (s_: number, g: number) => {
    const k = `${s_}|${g}`;
    let v = cache.get(k);
    if (!v) {
      const grid = decodeGrid(sc.sizes[s_].grids[g].p, N);
      v = { cv: fieldCanvas(grid, N, N), segs: contour(grid, N, N) };
      cache.set(k, v);
    }
    return v;
  };
  const trueSegs = contour(decodeGrid(sc.bayesGrid, N), N, N);
  const testPred = sc.sizes.map((z) => Uint8Array.from(atob(z.testPred), (ch) => ch.charCodeAt(0)));

  // ---------------- arena
  const arena = $("#e4-arena");
  const stage = new Stage($<HTMLCanvasElement>("canvas", arena), () => draw(), aspect);
  function draw() {
    const { ctx, w, h: H } = stage;
    if (!w) return;
    const M = mapper(D, w, H);
    const sz = sc.sizes[si];
    const cur = field(si, gi);
    ctx.drawImage(cur.cv, 0, 0, w, H);
    const nxt = Math.min(gi + 1, sz.grids.length - 1);
    if (frac > 0 && nxt !== gi) { ctx.globalAlpha = frac; ctx.drawImage(field(si, nxt).cv, 0, 0, w, H); ctx.globalAlpha = 1; }
    if (fadeFrom && fadeT < 1) { ctx.globalAlpha = 1 - fadeT; ctx.drawImage(fadeFrom, 0, 0, w, H); ctx.globalAlpha = 1; }
    const kx = w / (N - 1), ky = H / (N - 1);
    strokeSegs(ctx, cur.segs, kx, ky, "#ffffff", 2.4);
    if (showTrue) strokeSegs(ctx, trueSegs, kx, ky, TRUE_COLOR, 2.2);
    const final = gi === sz.grids.length - 1;
    const r = w < 420 ? 2.6 : 3.4;
    if (showTest) test.forEach((p, i) => {
      const X = M.sx(p.x), Y = M.sy(p.y);
      drawDot(ctx, X, Y, r, rgba(p.c ? C.c1 : C.c0, 0.6), true);
      // exact model predictions exist for the final snapshot only, so mistakes are marked only there
      if (final && (testPred[si][i] > 127) !== (p.c === 1)) {
        ctx.strokeStyle = rgba(C.test, 0.9); ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(X - 4, Y - 4); ctx.lineTo(X + 4, Y + 4); ctx.moveTo(X + 4, Y - 4); ctx.lineTo(X - 4, Y + 4); ctx.stroke();
      }
    });
    const n = sz.n;
    for (let i = 0; i < n; i++) {
      const p = pool[i];
      let Y = M.sy(p.y), a = 1;
      if (i >= dropFrom && dropT < 1) {
        const t = ease(Math.min(1, Math.max(0, dropT * 1.6 - ((i - dropFrom) / Math.max(1, n - dropFrom)) * 0.6)));
        Y = Y - (1 - t) * 60; a = t;
      }
      ctx.globalAlpha = a;
      const rr = n <= 50 ? r + 2.6 : n <= 200 ? r + 1 : r - 0.4;
      if (n <= 200) { ctx.beginPath(); ctx.arc(M.sx(p.x), Y, rr + 2.2, 0, 7); ctx.fillStyle = "#fff"; ctx.fill(); }
      drawDot(ctx, M.sx(p.x), Y, rr, p.c ? C.c1 : C.c0);
      ctx.globalAlpha = 1;
    }
    if (n <= 50) {
      ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.font = "600 12px JetBrains Mono, monospace";
      ctx.fillText(`only ${n} training points (white-ringed)`, 10, H - 12);
    }
  }

  // hover tooltip: which set, true class, model's call (final snapshot only)
  arena.addEventListener("pointermove", (e) => {
    const rect = arena.getBoundingClientRect();
    const M = mapper(D, stage.w, stage.h);
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best: { kind: string; i: number; d: number } | null = null;
    const consider = (kind: string, arr: { x: number; y: number }[], lim: number) => arr.slice(0, lim).forEach((p, i) => {
      const d = (M.sx(p.x) - mx) ** 2 + (M.sy(p.y) - my) ** 2;
      if (d < 100 && (!best || d < best.d)) best = { kind, i, d };
    });
    consider("train", pool, sc.sizes[si].n);
    if (showTest) consider("test", test, test.length);
    if (!best) return hideTip();
    const b = best as { kind: string; i: number };
    const p = b.kind === "train" ? pool[b.i] : test[b.i];
    const extra = b.kind === "test" ? `<div class="row"><span>model (final)</span><span>${pct(testPred[si][b.i] / 255, 0)} class 1 ${(testPred[si][b.i] > 127) === (p.c === 1) ? "✓" : "✗"}</span></div>` : `<div style="color:var(--dim)">the model trained on this one</div>`;
    showTip(e.clientX, e.clientY, `<div class="t" style="color:${b.kind === "train" ? C.train : C.test}">${b.kind === "train" ? "Training point" : "Held-out point"}</div><div class="row"><span>true class</span><span style="color:${p.c ? C.c1 : C.c0}">${p.c}</span></div>${extra}`);
  });
  arena.addEventListener("pointerleave", hideTip);

  // ---------------- gap meter + curves
  const el = (k: string) => $(`[data-e4="${k}"]`, root);
  const curveChart = new LineChart($("#e4-curve"), { H: 170, xLog: true, xLabel: "training step (log scale)" });
  curveChart.onSeek = (x) => {
    const g = sc.sizes[si].grids;
    let best = 0;
    g.forEach((q, i) => { if (Math.abs(Math.log(q.step + 1) - Math.log(x)) < Math.abs(Math.log(g[best].step + 1) - Math.log(x))) best = i; });
    player.pause(); player.set(best);
  };
  function drawCurve() {
    const c = sc.sizes[si].curve;
    const xs = c.map((p) => p.step + 1);
    const ymax = Math.max(3.6, ...c.map((p) => p.testLoss));
    curveChart.xDom = [1, xs.at(-1)!];
    curveChart.yDom = [0, ymax];
    curveChart.draw([
      { name: "train", color: C.train, xs, ys: c.map((p) => p.trainLoss) },
      { name: "held-out", color: C.test, xs, ys: c.map((p) => p.testLoss) },
    ], [1, 10, 100, 1000], [0, 1, 2, 3], (v) => String(v === 1 ? 0 : v), (v) => v.toFixed(0), (g) => {
      const top = xs.map((x, i) => `${curveChart.x(x)},${curveChart.y(c[i].testLoss)}`);
      const bot = xs.map((x, i) => `${curveChart.x(x)},${curveChart.y(c[i].trainLoss)}`).reverse();
      g.append(s("polygon", { points: [...top, ...bot].join(" "), fill: rgba(C.test, 0.13) }));
      const lab = (txt: string, color: string, y: number) => { const t = s("text", { x: curveChart.W - curveChart.pad.r, y, "text-anchor": "end", class: "tick", fill: color }); t.textContent = txt; g.append(t); };
      const yTe = curveChart.y(c.at(-1)!.testLoss), yTr = curveChart.y(c.at(-1)!.trainLoss);
      lab("held-out loss", C.test, yTe - 6);
      lab("train loss", C.train, yTr - yTe < 16 ? yTr + 13 : yTr - 6);
    });
  }
  function updateMeter() {
    const sz = sc.sizes[si];
    const step = sz.grids[gi].step;
    const ci = sz.curve.findIndex((p) => p.step === step);
    const p = sz.curve[ci];
    tweenText(el("tr"), p.trainAcc, (v) => pct(v), 400);
    tweenText(el("te"), p.testAcc, (v) => pct(v), 400);
    tweenText(el("gap"), (p.trainAcc - p.testAcc) * 100, (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)} pts`, 400);
    el("trBar").style.width = pct(p.trainAcc);
    el("teBar").style.width = pct(p.testAcc);
    el("best").style.left = pct(sc.bayesTestAcc);
    el("best").title = `best possible ≈ ${pct(sc.bayesTestAcc)}`;
    el("losses").textContent = `loss: train ${p.trainLoss.toFixed(4)} · held-out ${p.testLoss.toFixed(3)} · white tick = best possible (${pct(sc.bayesTestAcc)})`;
    el("stepnote").textContent = `n = ${sz.n} · step ${step}`;
    curveChart.mark(step + 1, ci);
  }

  // ---------------- gap vs size
  let gapKind: "acc" | "loss" = "acc";
  const gapChart = new LineChart($("#e4-gapchart"), { H: 210, xLog: true, xLabel: "training points (log scale)" });
  gapChart.W = 760;
  gapChart.svg.setAttribute("viewBox", `0 0 ${gapChart.W} ${gapChart.H}`);
  gapChart.onSeek = (x) => {
    let best = 0;
    cfg.sizes.forEach((n, i) => { if (Math.abs(Math.log(n) - Math.log(x)) < Math.abs(Math.log(cfg.sizes[best]) - Math.log(x))) best = i; });
    selectSize(best);
  };
  function drawGap() {
    const sm = data.summary;
    const xs = sm.map((r) => r.n);
    const mean = sm.map((r) => (gapKind === "acc" ? r.gapAccMean * 100 : r.gapLossMean));
    const lo = sm.map((r) => (gapKind === "acc" ? r.gapAccMin * 100 : r.gapLossMin));
    const hi = sm.map((r) => (gapKind === "acc" ? r.gapAccMax * 100 : r.gapLossMax));
    const ymax = Math.max(...hi) * 1.1;
    gapChart.xDom = [16, 2500];
    gapChart.yDom = [0, ymax];
    const yt = gapKind === "acc" ? [0, 5, 10, 15, 20] : [0, 1, 2, 3, 4];
    gapChart.draw([{ name: "mean gap", color: C.test, xs, ys: mean, width: 3 }], cfg.sizes, yt.filter((v) => v <= ymax),
      (v) => String(v), (v) => (gapKind === "acc" ? `${v} pts` : v.toFixed(0)), (g) => {
        const band = [...xs.map((x, i) => `${gapChart.x(x)},${gapChart.y(hi[i])}`), ...xs.map((x, i) => `${gapChart.x(x)},${gapChart.y(lo[i])}`).reverse()];
        g.append(s("polygon", { points: band.join(" "), fill: rgba(C.test, 0.16) }));
        data.seedRuns.forEach((r) => r.sizes.forEach((z) => g.append(s("circle", { cx: gapChart.x(z.n), cy: gapChart.y(gapKind === "acc" ? z.gapAcc * 100 : z.gapLoss), r: 2.4, fill: rgba(C.test, 0.5) }))));
      });
    gapChart.mark(cfg.sizes[si], si);
  }
  const kindSeg = $("#e4-gapkind");
  kindSeg.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest("button");
    if (!b) return;
    gapKind = b.dataset.k as any;
    kindSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    drawGap();
  });

  // ---------------- size buttons + feed
  const sizesRoot = $("#e4-sizes");
  const sizeBtns = cfg.sizes.map((n, i) => {
    const b = h("button", { "aria-pressed": String(i === 0), "aria-label": `${n} training points` }, h("b", {}, String(n)), h("small", {}, `gap ${(sc.sizes[i].gapAcc * 100).toFixed(1)}`));
    b.addEventListener("click", () => selectSize(i));
    sizesRoot.append(b);
    return b;
  });
  const feed = $<HTMLButtonElement>("#e4-feed");
  feed.addEventListener("click", () => selectSize(si < cfg.sizes.length - 1 ? si + 1 : 0));

  function selectSize(i: number) {
    if (i === si) return;
    const prevN = sc.sizes[si].n;
    fadeFrom = field(si, gi).cv;
    player.touched = true;
    player.pause();
    si = i;
    gi = sc.sizes[si].grids.length - 1;
    player.set(gi, false);
    sizeBtns.forEach((b, k) => b.setAttribute("aria-pressed", String(k === si)));
    feed.textContent = si < cfg.sizes.length - 1 ? `Feed it more data ➜ ${cfg.sizes[si + 1]}` : "Start over at 20";
    drawCurve(); updateMeter(); drawGap();
    dropFrom = sc.sizes[si].n > prevN ? prevN : sc.sizes[si].n;
    if (reducedMotion) { fadeT = 1; dropT = 1; draw(); return; }
    const t0 = performance.now();
    const anim = (t: number) => {
      const k = Math.min(1, (t - t0) / 800);
      fadeT = k; dropT = k;
      draw();
      if (k < 1) requestAnimationFrame(anim); else fadeFrom = null;
    };
    requestAnimationFrame(anim);
  }

  $<HTMLInputElement>("#e4-test").addEventListener("change", (e) => { showTest = (e.target as HTMLInputElement).checked; draw(); });
  $<HTMLInputElement>("#e4-true").addEventListener("change", (e) => { showTrue = (e.target as HTMLInputElement).checked; draw(); });

  const player = new Player($("#e4-transport"), sc.sizes[0].grids.length, (i, f) => {
    const changed = i !== gi;
    gi = i; frac = f;
    if (changed) updateMeter();
    draw();
  }, (i) => `n = ${sc.sizes[si].n} · step ${sc.sizes[si].grids[i].step} / ${cfg.steps}`);
  feed.textContent = `Feed it more data ➜ ${cfg.sizes[1]}`;
  player.set(gi, false);
  drawCurve(); updateMeter(); drawGap(); draw();
  onceVisible(arena, () => player.autoplay());
}
