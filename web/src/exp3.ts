// Experiment 3 — embeddings emerge from next-token prediction
import { $, C, LineChart, Player, h, hideTip, onceVisible, reducedMotion, rgba, s, showTip, tweenText } from "./lib";

type Frame = { step: number; loss: number; xy: number[]; E: number[]; W: number[]; b: number[]; purity: { hits: number; of: number } };
type Run = { label: string; pcaVarianceKept: number; frames: Frame[]; order: number[] };
type Data = {
  vocab: string[]; groups: Record<string, string[]>; sentences: string[]; nSentences: number; nPairs: number;
  nextCounts: number[][]; nextCountsShuffled: number[][]; lossFloor: number;
  runs: Record<"grammar" | "shuffled", Run>; config: { dim: number; steps: number };
};

export const GROUP_COLORS: Record<string, string> = {
  animals: "#f6c85f", fruits: "#6ee7a8", verbs: "#63a8ff", describing: "#f78fd0", glue: "#8a93a9",
};
const GROUP_NAMES: Record<string, string> = { animals: "animals", fruits: "fruits", verbs: "verbs", describing: "describing words", glue: "glue words" };
const NEUTRAL = "#dfe6f5";
let VW = 640, VH = 520;   // shrinks on phones so labels stay readable

export function nearestInfo(E: number[], V: number, dim: number) {
  const vec = (i: number) => E.slice(i * dim, i * dim + dim);
  const norm = (a: number[]) => Math.hypot(...a);
  const cos = (i: number, j: number) => {
    const a = vec(i), b = vec(j);
    return a.reduce((acc, v, k) => acc + v * b[k], 0) / (norm(a) * norm(b));
  };
  const S: number[][] = [];
  for (let i = 0; i < V; i++) { S.push([]); for (let j = 0; j < V; j++) S[i].push(cos(i, j)); }
  return { S, vec };
}

export function initExp3(data: Data) {
  const root = $("#ch3");
  const V = data.vocab.length, dim = data.config.dim;
  const groupOf: Record<string, string> = {};
  for (const [g, ws] of Object.entries(data.groups)) ws.forEach((w) => (groupOf[w] = g));
  let runKey: "grammar" | "shuffled" = "grammar";
  let run = data.runs.grammar;
  let frame = 0;
  let sel = data.vocab.indexOf("cat");
  let reveal = false;
  let order: "az" | "learned" = "az";
  const col = (i: number) => (reveal ? GROUP_COLORS[groupOf[data.vocab[i]]] : NEUTRAL);

  // ---------------- ticker of real training sentences
  const ticker = $("#e3-ticker");
  const line = h("span", { style: "display:inline-block;transition:transform .45s cubic-bezier(.2,.8,.2,1)" });
  const toks: HTMLElement[] = [];
  data.sentences.slice(0, 40).join(" ").split(" ").forEach((t) => {
    const e = h("span", { class: "tok" }, t);
    toks.push(e);
    line.append(e, " ");
  });
  ticker.append(line);
  let tp = 0;
  const tick = () => {
    toks.forEach((t, i) => { t.classList.toggle("cur", i === tp); t.classList.toggle("nxt", i === tp + 1); });
    line.style.transform = `translateX(${Math.min(0, ticker.clientWidth * 0.3 - toks[tp].offsetLeft)}px)`;
    tp = (tp + 1) % (toks.length - 2);
  };
  tick();
  let tickerVisible = false;
  new IntersectionObserver((es) => (tickerVisible = es[0].isIntersecting)).observe(ticker);
  if (!reducedMotion) setInterval(() => tickerVisible && tick(), 900);

  // ---------------- the embedding universe
  const svg = $<SVGSVGElement>("#e3-uni");
  const gGrid = s("g"), gHull = s("g"), gTrail = s("g"), gLinks = s("g"), gTok = s("g");
  svg.append(gGrid, gHull, gTrail, gLinks, gTok);
  const fitViewBox = () => {
    const w = svg.clientWidth || 640;
    [VW, VH] = w < 520 ? [400, 420] : [640, 520];
    svg.setAttribute("viewBox", `0 0 ${VW} ${VH}`);
    gGrid.innerHTML = "";
    for (let r = 60; r < 400; r += 60) gGrid.append(s("circle", { cx: VW / 2, cy: VH / 2, r, fill: "none", stroke: "rgba(255,255,255,.04)" }));
  };
  fitViewBox();
  let lastW = svg.clientWidth;
  new ResizeObserver(() => {
    if (Math.abs(svg.clientWidth - lastW) < 2) return;
    lastW = svg.clientWidth;
    fitViewBox(); computeBounds(); drawUniverse();
  }).observe(svg);
  let bounds = { x0: 0, x1: 1, y0: 0, y1: 1 };
  const computeBounds = () => {
    const f = run.frames.at(-1)!.xy;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < V; i++) { x0 = Math.min(x0, f[2 * i]); x1 = Math.max(x1, f[2 * i]); y0 = Math.min(y0, f[2 * i + 1]); y1 = Math.max(y1, f[2 * i + 1]); }
    const span = Math.max(x1 - x0, (y1 - y0) * (VW / VH)) * 1.18;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    bounds = { x0: cx - span / 2, x1: cx + span / 2, y0: cy - (span * VH) / VW / 2, y1: cy + (span * VH) / VW / 2 };
  };
  const px = (i: number, fr = run.frames[frame]) => [
    ((fr.xy[2 * i] - bounds.x0) / (bounds.x1 - bounds.x0)) * VW,
    (1 - (fr.xy[2 * i + 1] - bounds.y0) / (bounds.y1 - bounds.y0)) * VH,
  ];

  const tokEls = data.vocab.map((w, i) => {
    const g = s("g", { class: "tokdot", tabindex: 0, role: "button", "aria-label": `word ${w}` });
    const glow = s("circle", { r: 16, class: "glow" });
    const core = s("circle", { r: 7, class: "core" });
    const t = s("text", { x: 11, y: 4.5 });
    t.textContent = w;
    const lead = s("line", { stroke: "rgba(255,255,255,.35)", "stroke-width": 1 });
    g.append(glow, lead, core, t);
    g.style.transition = reducedMotion ? "" : "transform .16s linear";
    g.addEventListener("click", () => select(i));
    g.addEventListener("keydown", (e) => { if ((e as KeyboardEvent).key === "Enter" || (e as KeyboardEvent).key === " ") { e.preventDefault(); select(i); } });
    g.addEventListener("pointerenter", (e) => tokTip(i, e as PointerEvent));
    g.addEventListener("pointerleave", hideTip);
    gTok.append(g);
    return { g, glow, core, t, lead };
  });

  /** Points sit at their exact learned positions; only the LABELS are nudged apart (with leader lines). */
  function placeLabels(P: number[][]) {
    const box = data.vocab.map((w, i) => {
      const lw = w.length * 7.6 + 2;
      const x = P[i][0] + 11 + lw > VW - 4 ? P[i][0] - 11 - lw : P[i][0] + 11;   // flip left at the right edge
      return { x, y: P[i][1] + 4.5, ax: x, ay: P[i][1] + 4.5, w: lw };
    });
    for (let it = 0; it < 80; it++) {
      for (let a = 0; a < box.length; a++) for (let b = a + 1; b < box.length; b++) {
        const A = box[a], B = box[b];
        const ox = Math.min(A.x + A.w, B.x + B.w) - Math.max(A.x, B.x);
        const oy = Math.min(A.y + 4, B.y + 4) - Math.max(A.y - 12, B.y - 12);
        if (ox > 0 && oy > 0) {
          const push = (oy / 2 + 0.5) * (A.y <= B.y ? -1 : 1);
          A.y += push; B.y -= push;
        }
      }
      // labels must not sit on top of another word's dot
      for (const L of box) for (let j = 0; j < P.length; j++) {
        const [px, py] = P[j];
        if (px > L.x - 8 && px < L.x + L.w && py > L.y - 16 && py < L.y + 8) L.y += py > L.y - 4 ? -1.5 : 1.5;
      }
      for (const L of box) { L.y += (L.ay - L.y) * 0.04; L.y = Math.max(14, Math.min(VH - 6, L.y)); }
    }
    return box.map((L, i) => [L.x - P[i][0], L.y - P[i][1]]);
  }
  const tokTip = (i: number, e: PointerEvent) => {
    const { S } = nearestInfo(run.frames[frame].E, V, dim);
    const nb = S[i].map((v, j) => [v, j]).filter(([, j]) => j !== i).sort((a, b) => b[0] - a[0]).slice(0, 3);
    showTip(e.clientX, e.clientY, `<div class="t">${data.vocab[i]}</div>${nb.map(([v, j]) => `<div class="row"><span>${data.vocab[j]}</span><span>${v.toFixed(2)}</span></div>`).join("")}<div style="color:var(--dim);font-size:11.5px;margin-top:4px">click to pin</div>`);
  };

  function hull(points: number[][]) {
    const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (p.length < 3) return p;
    const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo: number[][] = [], up: number[][] = [];
    for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (const q of p.reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }

  function drawUniverse() {
    const fr = run.frames[frame];
    const { S } = nearestInfo(fr.E, V, dim);
    const P = data.vocab.map((_, i) => px(i));
    // trails: where each word has been, through the saved snapshots up to now
    gTrail.innerHTML = "";
    if (frame > 0) data.vocab.forEach((_, i) => {
      const pts = run.frames.slice(0, frame + 1).map((f) => px(i, f).map((v) => v.toFixed(1)).join(",")).join(" ");
      gTrail.append(s("polyline", { points: pts, fill: "none", stroke: rgba(col(i), i === sel ? 0.55 : 0.16), "stroke-width": i === sel ? 1.6 : 1, "stroke-linejoin": "round" }));
    });
    // links from the selected word: thickness = similarity
    gLinks.innerHTML = "";
    const ranked = S[sel].map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0]);
    ranked.forEach(([v, j], rank) => {
      if (v <= 0.05) return;
      const top = rank < 3;
      const l = s("line", { x1: P[sel][0], y1: P[sel][1], x2: P[j][0], y2: P[j][1], class: "link",
        stroke: top ? col(sel) : "rgba(255,255,255,.5)", "stroke-width": (0.5 + 6 * v * v).toFixed(2), opacity: top ? 0.85 : 0.18 * v });
      gLinks.append(l);
      if (top) {
        const t = s("text", { x: (P[sel][0] + P[j][0]) / 2, y: (P[sel][1] + P[j][1]) / 2 - 4, fill: "#fff", "font-size": 11, "text-anchor": "middle", "font-family": "JetBrains Mono, monospace", style: "paint-order:stroke;stroke:#0a0d16;stroke-width:3px" });
        t.textContent = v.toFixed(2);
        gLinks.append(t);
      }
    });
    // group hulls (only when revealed; the plot is never coloured by default)
    gHull.innerHTML = "";
    if (reveal) {
      for (const [g, ws] of Object.entries(data.groups)) {
        if (g === "glue") continue;
        const pts = hull(ws.map((w) => P[data.vocab.indexOf(w)]));
        const d = pts.map((q, k) => `${k ? "L" : "M"}${q[0].toFixed(1)},${q[1].toFixed(1)}`).join("") + "Z";
        gHull.append(s("path", { d, fill: rgba(GROUP_COLORS[g], 0.13), stroke: rgba(GROUP_COLORS[g], 0.13), "stroke-width": 44, "stroke-linejoin": "round", class: "hull" }));
      }
    }
    const top3 = new Set(ranked.slice(0, 3).map(([, j]) => j));
    const off = placeLabels(P);
    tokEls.forEach((t, i) => {
      t.g.setAttribute("transform", `translate(${P[i][0].toFixed(1)},${P[i][1].toFixed(1)})`);
      const [dx, dy] = off[i];
      t.t.setAttribute("x", dx.toFixed(1));
      t.t.setAttribute("y", dy.toFixed(1));
      const far = Math.abs(dy - 4.5) > 9;
      t.lead.setAttribute("x1", "0"); t.lead.setAttribute("y1", "0");
      t.lead.setAttribute("x2", far ? (dx > 0 ? dx - 2 : dx + data.vocab[i].length * 7.6).toFixed(1) : "0"); t.lead.setAttribute("y2", far ? (dy - 4).toFixed(1) : "0");
      const c = col(i);
      t.core.setAttribute("fill", c);
      t.core.setAttribute("r", i === sel ? "9.5" : "7");
      t.core.setAttribute("stroke", i === sel ? "#fff" : "rgba(10,13,22,.8)");
      t.core.setAttribute("stroke-width", i === sel ? "2.5" : "1.5");
      t.glow.setAttribute("fill", rgba(c, i === sel ? 0.28 : 0.12));
      t.g.classList.toggle("dim", !(i === sel || top3.has(i)) && frame > 0);
    });
  }

  // ---------------- side panels
  const pick = $("#e3-pick");
  const pickBtns = data.vocab.map((w, i) => {
    const b = h("button", { "aria-pressed": String(i === sel) }, w);
    b.addEventListener("click", () => select(i));
    pick.append(b);
    return b;
  });
  const el = (k: string) => $(`[data-e3="${k}"]`, root);
  function drawPanels() {
    const fr = run.frames[frame];
    const { S, vec } = nearestInfo(fr.E, V, dim);
    el("sel").textContent = `“${data.vocab[sel]}”`;
    el("sel2").textContent = `“${data.vocab[sel]}”`;
    pickBtns.forEach((b, i) => {
      b.setAttribute("aria-pressed", String(i === sel));
      b.style.borderColor = reveal ? rgba(GROUP_COLORS[groupOf[data.vocab[i]]], 0.6) : "";
    });
    // the 4 numbers themselves
    const v = vec(sel);
    const m = Math.max(...fr.E.map(Math.abs));
    $("#e3-vec").innerHTML = v.map((x) => `<span style="background:${rgba(x >= 0 ? "#63a8ff" : "#ff5c8a", 0.12 + 0.6 * Math.abs(x) / m)}">${x >= 0 ? "+" : ""}${x.toFixed(2)}</span>`).join("");
    // nearest neighbours
    const ranked = S[sel].map((val, j) => [val, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0]).slice(0, 6);
    $("#e3-nn").innerHTML = ranked.map(([val, j]) =>
      `<div class="barrow"><span class="nm">${data.vocab[j]}</span><div class="tr"><i style="width:${Math.max(0, val) * 100}%;background:${col(j)}"></i></div><span class="val">${val.toFixed(2)}</span></div>`).join("");
    // what it predicts next: softmax(W·e + b)
    const logits = data.vocab.map((_, o) => fr.b[o] + v.reduce((acc, x, k) => acc + x * fr.W[o * dim + k], 0));
    const mx = Math.max(...logits);
    const ex = logits.map((z) => Math.exp(z - mx));
    const Z = ex.reduce((a, b) => a + b, 0);
    const counts = (runKey === "grammar" ? data.nextCounts : data.nextCountsShuffled)[sel];
    const tot = counts.reduce((a, b) => a + b, 0) || 1;
    const nxt = ex.map((e, j) => [e / Z, j, counts[j] / tot]).sort((a, b) => b[0] - a[0]).slice(0, 6);
    $("#e3-next").innerHTML = nxt.map(([p, j, emp]) =>
      `<div class="barrow"><span class="nm">${data.vocab[j]}</span><div class="tr"><i class="ghost" style="width:${emp * 100}%"></i><i style="width:${p * 100}%;background:${rgba(C.c0, 0.85)};height:6px;top:2px"></i></div><span class="val">${(p * 100).toFixed(0)}%</span></div>`).join("");
    tweenText(el("p0"), run.frames[0].purity.hits, (x) => `${Math.round(x)}/14`, 300);
    tweenText(el("pnow"), fr.purity.hits, (x) => `${Math.round(x)}/14`, 300);
    lossChart.mark(fr.step + 1, frame);
  }

  // ---------------- similarity matrix
  const mat = $<SVGSVGElement>("#e3-matrix");
  const CELL = 22, LAB = 64;
  mat.setAttribute("viewBox", `0 0 ${LAB + CELL * V + 4} ${LAB + CELL * V + 4}`);
  const cells: SVGRectElement[][] = [];
  const rowLab: SVGTextElement[] = [], colLab: SVGTextElement[] = [];
  for (let i = 0; i < V; i++) {
    cells.push([]);
    const rl = s("text", { x: LAB - 6, y: 0, "text-anchor": "end" }) as SVGTextElement; rl.textContent = data.vocab[i];
    const cl = s("text", { x: 0, y: 0, "text-anchor": "start" }) as SVGTextElement; cl.textContent = data.vocab[i];
    mat.append(rl, cl); rowLab.push(rl); colLab.push(cl);
    for (let j = 0; j < V; j++) {
      const r = s("rect", { class: "cell", width: CELL - 2, height: CELL - 2, rx: 3 }) as SVGRectElement;
      r.addEventListener("pointerenter", (e) => {
        const S = nearestInfo(run.frames[frame].E, V, dim).S;
        showTip((e as PointerEvent).clientX, (e as PointerEvent).clientY, `<div class="t">${data.vocab[i]} ↔ ${data.vocab[j]}</div><div class="row"><span>cosine</span><span>${S[i][j].toFixed(3)}</span></div>`);
      });
      r.addEventListener("pointerleave", hideTip);
      r.addEventListener("click", () => select(i));
      mat.append(r);
      cells[i].push(r);
    }
  }
  function drawMatrix() {
    const S = nearestInfo(run.frames[frame].E, V, dim).S;
    const ord = order === "az" ? data.vocab.map((_, i) => i).sort((a, b) => data.vocab[a].localeCompare(data.vocab[b])) : run.order;
    const pos: number[] = []; ord.forEach((i, k) => (pos[i] = k));
    for (let i = 0; i < V; i++) {
      rowLab[i].style.transform = `translate(0px, ${LAB + pos[i] * CELL + CELL * 0.65}px)`;
      colLab[i].style.transform = `translate(${LAB + pos[i] * CELL + CELL * 0.65}px, ${LAB - 6}px) rotate(-60deg)`;
      rowLab[i].setAttribute("fill", reveal ? GROUP_COLORS[groupOf[data.vocab[i]]] : "");
      colLab[i].setAttribute("fill", reveal ? GROUP_COLORS[groupOf[data.vocab[i]]] : "");
      rowLab[i].style.fontWeight = i === sel ? "700" : "";
      for (let j = 0; j < V; j++) {
        const v = S[i][j];
        const a = Math.max(0, v);
        cells[i][j].style.transform = `translate(${LAB + pos[j] * CELL}px, ${LAB + pos[i] * CELL}px)`;
        cells[i][j].setAttribute("fill", i === j ? "rgba(255,255,255,.9)" : v >= 0 ? `rgba(99,168,255,${(0.06 + 0.94 * a * a).toFixed(3)})` : `rgba(255,92,138,${(0.05 + 0.3 * -v).toFixed(3)})`);
        cells[i][j].setAttribute("stroke", i === sel || j === sel ? "rgba(255,255,255,.35)" : "none");
      }
    }
  }
  const orderSeg = $("#e3-order");
  orderSeg.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest("button");
    if (!b) return;
    order = b.dataset.o as any;
    orderSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    drawMatrix();
  });

  // ---------------- loss chart
  const lossChart = new LineChart($("#e3-loss"), { H: 150, xLog: true, xLabel: "training step (log scale)" });
  lossChart.onSeek = (x) => {
    let best = 0;
    run.frames.forEach((f, i) => { if (Math.abs(Math.log(f.step + 1) - Math.log(x)) < Math.abs(Math.log(run.frames[best].step + 1) - Math.log(x))) best = i; });
    player.pause(); player.set(best);
  };
  function drawLoss() {
    const xs = run.frames.map((f) => f.step + 1);
    const ys = run.frames.map((f) => f.loss);
    lossChart.xDom = [1, xs.at(-1)!];
    lossChart.yDom = [0, Math.max(...ys) * 1.05];
    lossChart.draw([{ name: "loss", color: runKey === "grammar" ? "#63a8ff" : C.dim, xs, ys }], [1, 10, 100, 1000], [0, 1, 2, 3], (v) => String(v === 1 ? 0 : v), (v) => v.toFixed(0),
      (g) => {
        const y = lossChart.y(data.lossFloor);
        g.append(s("line", { x1: lossChart.pad.l, x2: lossChart.W - lossChart.pad.r, y1: y, y2: y, stroke: "#6ee7a8", "stroke-dasharray": "4 4", opacity: 0.7 }));
        const t = s("text", { x: lossChart.W - lossChart.pad.r, y: y - 5, class: "tick", "text-anchor": "end", fill: "#6ee7a8" });
        t.textContent = `best possible ${data.lossFloor.toFixed(3)}`;
        g.append(t);
      });
  }

  // ---------------- legend + notes
  function drawLegend() {
    $("#e3-legend").innerHTML = reveal
      ? Object.keys(data.groups).map((g) => `<span><i style="background:${GROUP_COLORS[g]}"></i>${GROUP_NAMES[g]}</span>`).join("")
      : `<span><i style="background:${NEUTRAL}"></i>every word looks the same until you reveal the groups — the model never saw them</span>`;
    el("pcanote").textContent = `The map squashes ${dim} numbers per word down to 2 (PCA keeps ${(run.pcaVarianceKept * 100).toFixed(0)}% of the spread), so distances on screen are approximate. Lines, rankings and the matrix all use the full ${dim}-number vectors.`;
  }

  function select(i: number) { sel = i; drawUniverse(); drawPanels(); drawMatrix(); }
  function redraw() { drawUniverse(); drawPanels(); drawMatrix(); }

  const runSeg = $("#e3-run");
  runSeg.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest("button");
    if (!b) return;
    runKey = b.dataset.run as any;
    run = data.runs[runKey];
    runSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    computeBounds(); drawLoss(); drawLegend(); redraw();
  });
  $<HTMLInputElement>("#e3-reveal").addEventListener("change", (e) => { reveal = (e.target as HTMLInputElement).checked; drawLegend(); redraw(); });

  const player = new Player($("#e3-transport"), run.frames.length, (i) => { if (i !== frame) { frame = i; redraw(); } },
    (i) => `step ${run.frames[i].step} / ${data.config.steps} · snapshot ${i + 1}/${run.frames.length}`, 9);
  computeBounds(); drawLoss(); drawLegend(); redraw();
  if (reducedMotion) player.set(run.frames.length - 1);
  onceVisible(svg, () => player.autoplay());
}
