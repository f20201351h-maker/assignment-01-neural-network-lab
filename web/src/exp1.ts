// Experiment 1 — activations exist for a reason (rings: linear vs one ReLU hidden layer)
import { $, C, Domain, LineChart, Player, Stage, clamp, contour, drawCross, drawDot, ease, fieldCanvas, h, hideTip, makeNet, mapper, onceVisible, pct, probGrid, reducedMotion, s, showTip, sigmoid, strokeSegs, tweenText, Layer } from "./lib";

type Frame = { step: number; loss: number; acc: number; testAcc: number; testLoss: number; layers: Layer[] };
type Run = { seed: number; points: number[]; labels: number[]; bestLineAcc: number; models: Record<"linear" | "relu", { frames: Frame[] }> };

const D: Domain = { x: [-3.4, 3.4], y: [-3.4, 3.4] };
const G = 110;
type Mode = "linear" | "split" | "relu";

export function initExp1(data: { runs: Run[]; config: any }) {
  const root = $("#ch1");
  const arena = $("#e1-arena");
  const canvas = $<HTMLCanvasElement>("canvas", arena);
  const handle = $(".handle", arena);
  let run = data.runs[0];
  let frame = 0, frac = 0;
  let mode: Mode = "split";
  let split = 0.5;
  let showMiss = true;
  let intro = reducedMotion ? 1 : 0;   // 0..1 point fly-in
  let probe: [number, number] | null = null;
  let hover = -1;

  // ---------------- caches: field image + contour + compiled net per (seed, model, frame)
  const cache = new Map<string, { cv: HTMLCanvasElement; segs: number[]; f: (x: number, y: number) => number }>();
  const get = (r: Run, m: "linear" | "relu", i: number) => {
    const k = `${r.seed}|${m}|${i}`;
    let v = cache.get(k);
    if (!v) {
      const f = makeNet(r.models[m].frames[i].layers, m === "relu");
      const g = probGrid(f, D, G);
      v = { cv: fieldCanvas(g, G, G), segs: contour(g, G, G), f };
      cache.set(k, v);
    }
    return v;
  };
  const pts = () => {
    const out: { x: number; y: number; c: number }[] = [];
    for (let i = 0; i < run.labels.length; i++) out.push({ x: run.points[2 * i], y: run.points[2 * i + 1], c: run.labels[i] });
    return out;
  };
  let P = pts();

  const stage = new Stage(canvas, () => draw());
  const nF = () => run.models.linear.frames.length;

  function draw() {
    const { ctx, w, h: H } = stage;
    if (!w) return;
    const M = mapper(D, w, H);
    const splitX = mode === "split" ? split * w : mode === "linear" ? w : 0;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, H);
    const nxt = Math.min(frame + 1, nF() - 1);
    const paint = (m: "linear" | "relu", x0: number, x1: number) => {
      if (x1 <= x0) return;
      ctx.save();
      ctx.beginPath(); ctx.rect(x0, 0, x1 - x0, H); ctx.clip();
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(get(run, m, frame).cv, 0, 0, w, H);
      if (frac > 0 && nxt !== frame) { ctx.globalAlpha = frac; ctx.drawImage(get(run, m, nxt).cv, 0, 0, w, H); ctx.globalAlpha = 1; }
      // faint grid
      ctx.strokeStyle = "rgba(255,255,255,0.045)"; ctx.lineWidth = 1;
      for (let v = -3; v <= 3; v++) {
        ctx.beginPath(); ctx.moveTo(M.sx(v), 0); ctx.lineTo(M.sx(v), H); ctx.moveTo(0, M.sy(v)); ctx.lineTo(w, M.sy(v)); ctx.stroke();
      }
      const kx = w / (G - 1), ky = H / (G - 1);
      strokeSegs(ctx, get(run, m, frame).segs, kx, ky, m === "linear" ? C.lin : C.relu, 3);
      ctx.restore();
    };
    paint("linear", 0, splitX);
    paint("relu", splitX, w);

    // points (fly in from the centre on first view)
    const t = ease(intro);
    const r = w < 420 ? 3.4 : 4.2;
    P.forEach((p, i) => {
      const sx = M.sx(p.x * t), sy = M.sy(p.y * t);
      drawDot(ctx, sx, sy, i === hover ? r + 2 : r, p.c ? C.c1 : C.c0);
      if (showMiss && intro >= 1) {
        const m = sx < splitX ? "linear" : "relu";
        const z = get(run, m, frame).f(p.x, p.y);
        if ((z > 0) !== (p.c === 1)) drawCross(ctx, sx, sy, r);
      }
    });

    if (probe) {
      const [px, py] = probe;
      const X = M.sx(px), Y = M.sy(py);
      ctx.save();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(X, Y, 9, 0, 7); ctx.moveTo(X - 15, Y); ctx.lineTo(X - 5, Y); ctx.moveTo(X + 5, Y); ctx.lineTo(X + 15, Y);
      ctx.moveTo(X, Y - 15); ctx.lineTo(X, Y - 5); ctx.moveTo(X, Y + 5); ctx.lineTo(X, Y + 15); ctx.stroke();
      const pl = sigmoid(get(run, "linear", frame).f(px, py)), pr = sigmoid(get(run, "relu", frame).f(px, py));
      const label = [`linear: ${pct(pl, 0)} outer`, `ReLU:   ${pct(pr, 0)} outer`];
      ctx.font = "600 12px JetBrains Mono, monospace";
      const tw = Math.max(...label.map((l) => ctx.measureText(l).width)) + 16;
      let bx = X + 16, by = Y - 44;
      if (bx + tw > w) bx = X - 16 - tw;
      if (by < 4) by = Y + 16;
      ctx.fillStyle = "rgba(10,13,22,.9)"; ctx.strokeStyle = "rgba(255,255,255,.18)";
      ctx.beginPath(); ctx.roundRect(bx, by, tw, 40, 8); ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.lin; ctx.fillText(label[0], bx + 8, by + 16);
      ctx.fillStyle = C.relu; ctx.fillText(label[1], bx + 8, by + 32);
      ctx.restore();
    }
    handle.style.left = `${split * 100}%`;
    handle.style.display = mode === "split" ? "" : "none";
    $(".tag.l", arena).style.opacity = mode === "relu" ? "0" : "1";
    $(".tag.r", arena).style.opacity = mode === "linear" ? "0" : "1";
  }

  // ---------------- side panels
  const lossChart = new LineChart($("#e1-loss"), { H: 150, xLog: true, xLabel: "training step (log scale)" });
  lossChart.onSeek = (x) => {
    const fr = run.models.linear.frames;
    let best = 0;
    fr.forEach((f, i) => { if (Math.abs(Math.log(f.step + 1) - Math.log(x)) < Math.abs(Math.log(fr[best].step + 1) - Math.log(x))) best = i; });
    player.pause(); player.set(best);
  };
  function drawLoss() {
    const L = run.models.linear.frames, R = run.models.relu.frames;
    const xs = L.map((f) => f.step + 1);
    const ymax = Math.max(...L.map((f) => f.loss), ...R.map((f) => f.loss)) * 1.05;
    lossChart.xDom = [1, xs[xs.length - 1]];
    lossChart.yDom = [0, ymax];
    lossChart.draw([
      { name: "linear", color: C.lin, xs, ys: L.map((f) => f.loss) },
      { name: "relu", color: C.relu, xs, ys: R.map((f) => f.loss) },
    ], [1, 10, 100, 1000], [0, 0.25, 0.5, 0.75].filter((v) => v <= ymax), (v) => String(v - 1 < 1 ? 0 : v), (v) => v.toFixed(2));
  }

  // network glyph: linear (2->1) and ReLU (2->H->1), edges from the current snapshot
  const netRoot = $("#e1-net");
  const NW = 400, NH = 190;
  const svg = s("svg", { viewBox: `0 0 ${NW} ${NH}`, class: "netglyph", role: "img", "aria-label": "Diagram of both networks; line thickness shows weight size." });
  netRoot.append(svg);
  function drawNet() {
    svg.innerHTML = "";
    const L = run.models.linear.frames[frame].layers, R = run.models.relu.frames[frame].layers;
    const lab = (x: number, y: number, t: string, c = C.dim, a = "middle") => {
      const e = s("text", { x, y, fill: c, "font-size": 10.5, "text-anchor": a, "font-family": "JetBrains Mono, monospace" }); e.textContent = t; svg.append(e);
    };
    const node = (x: number, y: number, r: number, c: string) => svg.append(s("circle", { cx: x, cy: y, r, fill: C.bg, stroke: c, "stroke-width": 1.6 }));
    const edge = (x1: number, y1: number, x2: number, y2: number, wgt: number, max: number, c: string) => {
      const a = Math.min(1, Math.abs(wgt) / max);
      svg.append(s("line", { x1, y1, x2, y2, stroke: wgt >= 0 ? c : "#7c86a0", "stroke-width": 0.4 + a * 3.2, opacity: 0.15 + 0.85 * a }));
    };
    // linear
    const inY = [NH / 2 - 26, NH / 2 + 26];
    const lw = L[0].W[0];
    const lmax = Math.max(1e-6, ...lw.map(Math.abs));
    inY.forEach((y, i) => edge(28, y, 112, NH / 2, lw[i], lmax, C.lin));
    inY.forEach((y, i) => { node(28, y, 8, C.ink); lab(28, y + 3.5, i ? "y" : "x", C.ink); });
    node(112, NH / 2, 11, C.lin); lab(112, NH / 2 + 3.5, "σ", C.lin);
    lab(70, 18, "LINEAR · 3 numbers", C.lin);
    lab(70, NH - 8, `b = ${L[0].b[0].toFixed(2)}`);
    // relu
    const W1 = R[0].W, W2 = R[1].W[0];
    const Hn = W1.length;
    const hx = 290, ix = 180, ox = 384;
    const hy = (j: number) => 30 + (j / (Hn - 1)) * (NH - 50);
    const m1 = Math.max(1e-6, ...W1.flat().map(Math.abs)), m2 = Math.max(1e-6, ...W2.map(Math.abs));
    for (let j = 0; j < Hn; j++) {
      inY.forEach((y, i) => edge(ix, y, hx, hy(j), W1[j][i], m1, C.relu));
      edge(hx, hy(j), ox, NH / 2, W2[j], m2, C.relu);
    }
    inY.forEach((y, i) => { node(ix, y, 8, C.ink); lab(ix, y + 3.5, i ? "y" : "x", C.ink); });
    for (let j = 0; j < Hn; j++) svg.append(s("circle", { cx: hx, cy: hy(j), r: 2.6, fill: C.relu }));
    node(ox, NH / 2, 11, C.relu); lab(ox, NH / 2 + 3.5, "σ", C.relu);
    lab(285, 18, `ReLU · ${Hn} hidden · ${Hn * 4 + 1} numbers`, C.relu);
    svg.append(s("line", { x1: 150, x2: 150, y1: 20, y2: NH - 10, stroke: "rgba(255,255,255,.08)" }));
  }

  const el = (k: string) => $(`[data-e1="${k}"]`, root);
  function updateMetrics() {
    const L = run.models.linear.frames[frame], R = run.models.relu.frames[frame];
    tweenText(el("linAcc"), L.acc, (v) => pct(v), 350);
    tweenText(el("reluAcc"), R.acc, (v) => pct(v), 350);
    const wrong = (a: number) => Math.round((1 - a) * run.labels.length);
    el("linSub").textContent = `${wrong(L.acc)} wrong · loss ${L.loss.toFixed(3)}`;
    el("reluSub").textContent = `${wrong(R.acc)} wrong · loss ${R.loss.toFixed(3)}`;
    el("linBar").style.width = pct(L.acc);
    el("reluBar").style.width = pct(R.acc);
    el("stepnote").textContent = `step ${L.step}`;
    lossChart.mark(L.step + 1, frame);
  }

  const player = new Player($("#e1-transport"), nF(), (i, f) => {
    if (player?.touched) intro = 1;   // visitor took over: skip the fly-in
    const changed = i !== frame;
    frame = i; frac = f;
    if (changed) { updateMetrics(); drawNet(); }
    draw();
  }, (i) => `step ${run.models.linear.frames[i].step} / ${data.config.steps} · snapshot ${i + 1}/${nF()}`);

  // ---------------- seeds
  const seedsRoot = $("#e1-seeds");
  data.runs.forEach((r, i) => {
    const L = r.models.linear.frames.at(-1)!, R = r.models.relu.frames.at(-1)!;
    const b = h("button", { class: "chip", "aria-pressed": i === 0 ? "true" : "false" },
      `dataset ${r.seed + 1}`, h("small", {}, `${pct(L.acc)} → ${pct(R.acc)}`));
    b.addEventListener("click", () => {
      seedsRoot.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      run = r; P = pts(); probe = null;
      drawLoss(); player.set(player.index);
      updateMetrics(); drawNet(); draw();
    });
    seedsRoot.append(b);
  });

  // ---------------- controls
  const modeRoot = $("#e1-mode");
  modeRoot.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest("button");
    if (!b) return;
    mode = b.dataset.mode as Mode;
    modeRoot.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    draw();
  });
  $<HTMLInputElement>("#e1-misses").addEventListener("change", (e) => { showMiss = (e.target as HTMLInputElement).checked; draw(); });

  // divider drag (mouse + touch) and keyboard
  const setSplit = (v: number) => { split = clamp(v, 0, 1); handle.setAttribute("aria-valuenow", String(Math.round(split * 100))); draw(); };
  handle.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    handle.setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => { const r = arena.getBoundingClientRect(); setSplit((ev.clientX - r.left) / r.width); };
    const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });
  handle.addEventListener("keydown", (e) => {
    const k = (e as KeyboardEvent).key;
    if (k === "ArrowLeft" || k === "ArrowRight") { e.preventDefault(); e.stopPropagation(); setSplit(split + (k === "ArrowLeft" ? -0.05 : 0.05)); }
  });

  // hover points / drop a probe
  const nearest = (cx: number, cy: number) => {
    const r = arena.getBoundingClientRect();
    const M = mapper(D, stage.w, stage.h);
    let best = -1, bd = 14 * 14;
    P.forEach((p, i) => {
      const dx = M.sx(p.x) - (cx - r.left), dy = M.sy(p.y) - (cy - r.top);
      if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; best = i; }
    });
    return best;
  };
  const pointTip = (i: number, cx: number, cy: number) => {
    const p = P[i];
    const zl = get(run, "linear", frame).f(p.x, p.y), zr = get(run, "relu", frame).f(p.x, p.y);
    const ok = (z: number) => ((z > 0) === (p.c === 1) ? "✓" : "✗");
    showTip(cx, cy, `<div class="t" style="color:${p.c ? C.c1 : C.c0}">${p.c ? "Outer ring" : "Inner ring"} point</div>
      <div class="row"><span style="color:${C.lin}">linear</span><span>${pct(sigmoid(zl), 0)} outer ${ok(zl)}</span></div>
      <div class="row"><span style="color:${C.relu}">ReLU</span><span>${pct(sigmoid(zr), 0)} outer ${ok(zr)}</span></div>`);
  };
  arena.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    const i = nearest(e.clientX, e.clientY);
    if (i !== hover) { hover = i; draw(); }
    if (i >= 0) pointTip(i, e.clientX, e.clientY); else hideTip();
  });
  arena.addEventListener("pointerleave", () => { hover = -1; hideTip(); draw(); });
  arena.addEventListener("pointerdown", (e) => {
    const i = nearest(e.clientX, e.clientY);
    if (i >= 0) { hover = i; pointTip(i, e.clientX, e.clientY); draw(); return; }
    const r = arena.getBoundingClientRect();
    const M = mapper(D, stage.w, stage.h);
    probe = [M.wx(e.clientX - r.left), M.wy(e.clientY - r.top)];
    hideTip(); draw();
  });
  arena.addEventListener("keydown", (e) => {
    const k = (e as KeyboardEvent).key;
    const d: Record<string, [number, number]> = { ArrowLeft: [-0.2, 0], ArrowRight: [0.2, 0], ArrowUp: [0, 0.2], ArrowDown: [0, -0.2] };
    if (k in d) {
      e.preventDefault();
      probe = probe ?? [0, 0];
      probe = [clamp(probe[0] + d[k][0], D.x[0], D.x[1]), clamp(probe[1] + d[k][1], D.y[0], D.y[1])];
      draw();
    } else if (k === "Escape") { probe = null; draw(); }
  });

  drawLoss(); updateMetrics(); drawNet(); draw();

  // start: points fly in, then training plays
  onceVisible(arena, () => {
    if (reducedMotion) { player.set(nF() - 1); return; }
    const t0 = performance.now();
    const fly = (t: number) => {
      intro = player.touched ? 1 : clamp((t - t0) / 1100, 0, 1);
      draw();
      if (intro < 1) requestAnimationFrame(fly); else setTimeout(() => player.autoplay(), 250);
    };
    requestAnimationFrame(fly);
  });
  if (reducedMotion) player.set(nF() - 1);

}
