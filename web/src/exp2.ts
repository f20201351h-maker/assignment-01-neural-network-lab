// Experiment 2 — depth without nonlinearity is a lie
import { $, C, Domain, Layer, Player, Stage, clamp, contour, drawCross, drawDot, fieldCanvas, h, makeNet, mapper, onceVisible, pct, probGrid, reducedMotion, rgb, sci, sigmoid, strokeSegs, tweenText } from "./lib";

type Frame = { step: number; loss: number; acc: number; layers: Layer[] };
type Model = { relu: boolean; frames: Frame[]; params: number };
type Data = { models: Record<"L1" | "L5" | "R5", Model>; collapse: any; config: any; points: number[]; labels: number[] };

const D: Domain = { x: [-3.4, 3.4], y: [-3.4, 3.4] };
const G = 90;
const KEYS = ["L1", "L5", "R5"] as const;
const TITLES = { L1: "1 linear layer", L5: "5 linear layers", R5: "5 layers + ReLU" };
const COLORS = { L1: C.lin, L5: C.lin, R5: C.relu };

/** Fold affine layers into one: W* = Wn…W1, b* = Wn(…(W2 b1 + b2)…) + bn. */
export function collapse(layers: Layer[], useBias = true) {
  let W = [[1, 0], [0, 1]];
  let b = [0, 0];
  for (const L of layers) {
    W = L.W.map((row) => [0, 1].map((j) => row.reduce((acc, w, k) => acc + w * W[k][j], 0)));
    b = L.W.map((row, o) => row.reduce((acc, w, k) => acc + w * b[k], 0) + (useBias ? L.b[o] : 0));
  }
  return { W: W[0], b: b[0] };
}

export function initExp2(data: Data) {
  const root = $("#ch2");
  const P = data.labels.map((c, i) => ({ x: data.points[2 * i], y: data.points[2 * i + 1], c }));
  const nF = data.models.L1.frames.length;
  let frame = 0;
  let probe: [number, number] | null = null;

  const cache = new Map<string, { cv: HTMLCanvasElement; segs: number[]; f: (x: number, y: number) => number }>();
  const get = (k: (typeof KEYS)[number], i: number) => {
    const key = `${k}|${i}`;
    let v = cache.get(key);
    if (!v) {
      const f = makeNet(data.models[k].frames[i].layers, data.models[k].relu);
      const g = probGrid(f, D, G);
      v = { cv: fieldCanvas(g, G, G), segs: contour(g, G, G), f };
      cache.set(key, v);
    }
    return v;
  };

  // ---------------- three synchronized maps
  const trio = $("#e2-trio");
  const cells = KEYS.map((k) => {
    const acc = h("span", { class: "acc", style: `color:${COLORS[k]}` }, "–");
    const sub = h("div", { class: "sub" }, "");
    const cv = h("canvas") as HTMLCanvasElement;
    const arena = h("div", { class: "arena", "aria-label": `${TITLES[k]} decision map` }, cv);
    const cell = h("div", { class: "cell" },
      h("h3", {}, h("span", { style: `color:${COLORS[k]}` }, TITLES[k]), h("small", {}, `${data.models[k].params} params`)),
      arena, h("div", { style: "display:flex;justify-content:space-between;align-items:baseline;margin-top:8px" }, acc, sub));
    trio.append(cell);
    const stage = new Stage(cv, () => drawCell(k));
    return { k, acc, sub, stage, arena };
  });

  function drawCell(k: (typeof KEYS)[number]) {
    const c = cells.find((x) => x.k === k);
    if (!c) return;
    const { ctx, w, h: H } = c.stage;
    if (!w) return;
    const M = mapper(D, w, H);
    const v = get(k, frame);
    ctx.drawImage(v.cv, 0, 0, w, H);
    strokeSegs(ctx, v.segs, w / (G - 1), H / (G - 1), COLORS[k], 2.6);
    if (k === "L5" && collapsedView) {
      // draw the collapsed single layer's boundary line on top: w·x + b = 0
      const cw = collapse(data.models.L5.frames[frame].layers);
      ctx.save(); ctx.setLineDash([6, 6]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.8;
      ctx.beginPath();
      if (Math.abs(cw.W[1]) > Math.abs(cw.W[0])) {
        const y = (x: number) => -(cw.W[0] * x + cw.b) / cw.W[1];
        ctx.moveTo(M.sx(D.x[0]), M.sy(y(D.x[0]))); ctx.lineTo(M.sx(D.x[1]), M.sy(y(D.x[1])));
      } else {
        const x = (y: number) => -(cw.W[1] * y + cw.b) / cw.W[0];
        ctx.moveTo(M.sx(x(D.y[0])), M.sy(D.y[0])); ctx.lineTo(M.sx(x(D.y[1])), M.sy(D.y[1]));
      }
      ctx.stroke(); ctx.restore();
    }
    const r = w < 300 ? 2.6 : 3.2;
    for (const p of P) {
      const sx = M.sx(p.x), sy = M.sy(p.y);
      drawDot(ctx, sx, sy, r, p.c ? C.c1 : C.c0);
      if ((v.f(p.x, p.y) > 0) !== (p.c === 1)) drawCross(ctx, sx, sy, r - 1);
    }
    if (k === "L5" && probe) {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(M.sx(probe[0]), M.sy(probe[1]), 8, 0, 7); ctx.stroke();
    }
  }
  const drawAll = () => KEYS.forEach(drawCell);

  // click on the 5-linear map = push that point through both versions
  const l5 = cells[1];
  l5.arena.style.cursor = "crosshair";
  l5.arena.addEventListener("pointerdown", (e) => {
    const r = l5.arena.getBoundingClientRect();
    const M = mapper(D, l5.stage.w, l5.stage.h);
    probe = [M.wx(e.clientX - r.left), M.wy(e.clientY - r.top)];
    updateProof(); drawCell("L5");
  });

  // ---------------- collapse machine
  const layersRoot = $("#e2-layers");
  let collapsedView = false;
  const heat = (W: number[][], b: number[], cell: number, max: number) => {
    const rows = W.length, cols = W[0].length + 1;
    const cv = h("canvas", { width: cols, height: rows }) as HTMLCanvasElement;
    cv.style.width = cols * cell + "px"; cv.style.height = rows * cell + "px";
    const ctx = cv.getContext("2d")!;
    const img = ctx.createImageData(cols, rows);
    const pos = rgb(C.lin), neg = rgb(C.c0), bg = rgb(C.bg);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = c < cols - 1 ? W[r][c] : b[r];
      const a = Math.min(1, Math.abs(v) / max);
      const col = v >= 0 ? pos : neg;
      const i = (r * cols + c) * 4;
      for (let ch = 0; ch < 3; ch++) img.data[i + ch] = bg[ch] + (col[ch] - bg[ch]) * (0.12 + 0.88 * a);
      img.data[i + 3] = c === cols - 1 ? 235 : 255;
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  };
  function drawLayers() {
    const layers = data.models.L5.frames[frame].layers;
    const max = Math.max(...layers.flatMap((L) => [...L.W.flat(), ...L.b]).map(Math.abs));
    layersRoot.innerHTML = "";
    layersRoot.classList.toggle("collapsed", collapsedView);
    if (collapsedView) {
      const cw = collapse(layers);
      const card = h("div", { class: "layer-card single" },
        heat([cw.W], [cw.b], 34, Math.max(...cw.W.map(Math.abs), Math.abs(cw.b))),
        h("div", { class: "lbl" }, `ONE layer · 2 → 1 · 3 numbers`));
      layersRoot.append(card);
      if (!reducedMotion) card.animate([{ transform: "scale(.4)", opacity: 0 }, { transform: "scale(1.08)", opacity: 1, offset: 0.7 }, { transform: "scale(1)" }], { duration: 650, easing: "cubic-bezier(.2,.8,.2,1)" });
      return;
    }
    // fit all five heatmaps in the panel: total columns = sum(in + 1 bias column)
    const cols = layers.reduce((a, L) => a + L.W[0].length + 1, 0);
    const cell = Math.max(2, Math.min(7, Math.floor((layersRoot.clientWidth - 5 * 24 - 4 * 34 - 8 * 6 - 8) / cols)));
    layers.forEach((L, i) => {
      if (i) layersRoot.append(h("span", { class: "relu-sock", title: "No activation between these layers" }, "no", h("br"), "ReLU"));
      layersRoot.append(h("div", { class: "layer-card" }, heat(L.W, L.b, cell, max), h("div", { class: "lbl" }, `W${i + 1} ${L.W[0].length}→${L.W.length}`)));
    });
  }
  const btn = $<HTMLButtonElement>("#e2-collapse");
  btn.addEventListener("click", () => {
    if (!collapsedView && !reducedMotion) {
      // fly the five cards together, then swap in the single card
      const cards = Array.from(layersRoot.children) as HTMLElement[];
      const mid = layersRoot.getBoundingClientRect();
      const cx = mid.left + mid.width / 2;
      cards.forEach((c) => {
        const r = c.getBoundingClientRect();
        c.animate([{ transform: "none" }, { transform: `translateX(${cx - (r.left + r.width / 2)}px) scale(.3)`, opacity: 0 }], { duration: 600, easing: "cubic-bezier(.6,0,.4,1)", fill: "forwards" });
      });
      setTimeout(() => { collapsedView = true; afterToggle(); }, 560);
    } else { collapsedView = !collapsedView; afterToggle(); }
  });
  const afterToggle = () => {
    btn.textContent = collapsedView ? "Show the five layers again" : "Multiply the five layers out";
    drawLayers(); updateProof(); drawCell("L5");
  };
  const noBias = $<HTMLInputElement>("#e2-nobias");
  noBias.addEventListener("change", () => updateProof());

  // deterministic probe inputs (same every load)
  const probes: [number, number][] = [];
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  for (let i = 0; i < 10000; i++) probes.push([-3.5 + 7 * rnd(), -3.5 + 7 * rnd()]);

  const e = (k: string) => $(`[data-e2="${k}"]`, root);
  function updateProof() {
    const layers = data.models.L5.frames[frame].layers;
    const useBias = !noBias.checked;
    const cw = collapse(layers, useBias);
    const stack = makeNet(layers, false);
    let maxErr = 0;
    for (const [x, y] of probes) maxErr = Math.max(maxErr, Math.abs(stack(x, y) - (cw.W[0] * x + cw.W[1] * y + cw.b)));
    e("n").textContent = probes.length.toLocaleString();
    e("err").textContent = sci(maxErr);
    e("err").style.color = maxErr < 1e-9 ? "#6ee7a8" : C.c1;
    e("errsub").textContent = useBias ? "5-layer output vs one layer (biases included)" : "biases dropped from the collapse";
    const ok = maxErr < 1e-9;
    e("verdict").innerHTML = ok ? `<span class="verdict ok">● identical</span>` : `<span class="verdict bad">● NOT equal</span>`;
    let a1 = 0, a2 = 0;
    for (const p of P) { a1 += +((stack(p.x, p.y) > 0) === (p.c === 1)); a2 += +((cw.W[0] * p.x + cw.W[1] * p.y + cw.b > 0) === (p.c === 1)); }
    e("accs").textContent = `accuracy: stack ${pct(a1 / P.length)} · one layer ${pct(a2 / P.length)}`;
    const f = (v: number) => (v >= 0 ? " " : "") + v.toPrecision(5);
    $("#e2-formula").innerHTML =
      `<span class="w">W*</span> = W5·W4·W3·W2·W1 = [${f(cw.W[0])}, ${f(cw.W[1])}]<br>` +
      (useBias ? `<span class="bb">b*</span> = W5(W4(W3(W2·b1 + b2) + b3) + b4) + b5 = ${f(cw.b)}`
               : `<span class="bb">b*</span> = 0 &nbsp;<span style="color:${C.c1}">← forgot the biases: the single layer no longer matches</span>`) +
      `<br><span style="color:var(--dim)">snapshot at step ${data.models.L5.frames[frame].step} — the identity holds at every point in training</span>`;
    if (probe) {
      const s5 = stack(probe[0], probe[1]), s1 = cw.W[0] * probe[0] + cw.W[1] * probe[1] + cw.b;
      e("probe").innerHTML = `point (${probe[0].toFixed(2)}, ${probe[1].toFixed(2)}) → 5 layers: <b style="color:var(--lin)">${s5.toPrecision(15)}</b> · one layer: <b style="color:var(--ink)">${s1.toPrecision(15)}</b>`;
    }
  }

  // ---------------- 3D landscape
  const landRoot = $("#e2-land");
  const lcv = $<HTMLCanvasElement>("canvas", landRoot);
  let landKey: (typeof KEYS)[number] = "L5";
  let yaw = -0.7, pitch = 0.62, spinning = !reducedMotion, lastInteract = 0;
  const NM = 30;
  const land = new Stage(lcv, () => drawLand(), 1.25);
  function drawLand() {
    const { ctx, w, h: H } = land;
    if (!w) return;
    ctx.clearRect(0, 0, w, H);
    const f = get(landKey, frame).f;
    const sc = w / 11.5;
    const proj = (x: number, y: number, z: number) => {
      const cx = Math.cos(yaw), sx = Math.sin(yaw);
      const X = x * cx - y * sx, Y = x * sx + y * cx;
      const Z = (z - 0.5) * 3;
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      return [w / 2 + X * sc, H * 0.52 - (Z * cp - Y * sp) * sc, Y * cp + Z * sp] as const;
    };
    const step = 6.8 / NM;
    const zs: number[][] = [];
    for (let i = 0; i <= NM; i++) { zs.push([]); for (let j = 0; j <= NM; j++) zs[i].push(sigmoid(f(-3.4 + i * step, -3.4 + j * step))); }
    const quads: { d: number; pts: (readonly [number, number, number])[]; z: number }[] = [];
    for (let i = 0; i < NM; i++) for (let j = 0; j < NM; j++) {
      const p = [proj(-3.4 + i * step, -3.4 + j * step, zs[i][j]), proj(-3.4 + (i + 1) * step, -3.4 + j * step, zs[i + 1][j]),
                 proj(-3.4 + (i + 1) * step, -3.4 + (j + 1) * step, zs[i + 1][j + 1]), proj(-3.4 + i * step, -3.4 + (j + 1) * step, zs[i][j + 1])];
      quads.push({ d: p.reduce((a, q) => a + q[2], 0), pts: p, z: (zs[i][j] + zs[i + 1][j + 1]) / 2 });
    }
    // floor outline
    ctx.strokeStyle = "rgba(255,255,255,.12)"; ctx.lineWidth = 1;
    ctx.beginPath();
    [[-3.4, -3.4], [3.4, -3.4], [3.4, 3.4], [-3.4, 3.4], [-3.4, -3.4]].forEach(([x, y], i) => { const q = proj(x, y, 0); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
    ctx.stroke();
    quads.sort((a, b) => b.d - a.d);
    const A = rgb(C.c0), B = rgb(C.c1);
    for (const q of quads) {
      const t = q.z;
      const col = A.map((a, i) => Math.round(a + (B[i] - a) * t));
      ctx.beginPath();
      q.pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.closePath();
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},0.55)`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},0.9)`;
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }
    // the data, floating at its label height: 0 = inner, 1 = outer
    for (const p of P) {
      const q = proj(p.x, p.y, p.c);
      ctx.fillStyle = p.c ? C.c1 : C.c0;
      ctx.beginPath(); ctx.arc(q[0], q[1], 2.1, 0, 7); ctx.fill();
    }
    ctx.fillStyle = C.dim; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText(`${TITLES[landKey]} · step ${data.models[landKey].frames[frame].step}`, 10, 18);
    ctx.fillText("dots float at height 0 (inner) and 1 (outer)", 10, H - 10);
  }
  landRoot.addEventListener("pointerdown", (ev) => {
    landRoot.setPointerCapture(ev.pointerId);
    let lx = ev.clientX, ly = ev.clientY;
    spinning = false;
    const move = (m: PointerEvent) => {
      yaw += (m.clientX - lx) * 0.01; pitch = clamp(pitch + (m.clientY - ly) * 0.006, 0.1, 1.35);
      lx = m.clientX; ly = m.clientY; lastInteract = performance.now(); drawLand();
    };
    const up = () => { landRoot.removeEventListener("pointermove", move); landRoot.removeEventListener("pointerup", up); };
    landRoot.addEventListener("pointermove", move); landRoot.addEventListener("pointerup", up);
  });
  landRoot.addEventListener("keydown", (ev) => {
    const k = (ev as KeyboardEvent).key;
    if (k === "ArrowLeft" || k === "ArrowRight") { yaw += k === "ArrowLeft" ? -0.15 : 0.15; }
    else if (k === "ArrowUp" || k === "ArrowDown") { pitch = clamp(pitch + (k === "ArrowUp" ? -0.08 : 0.08), 0.1, 1.35); }
    else return;
    ev.preventDefault(); spinning = false; drawLand();
  });
  const landSeg = $("#e2-land-mode");
  landSeg.addEventListener("click", (ev) => {
    const b = (ev.target as HTMLElement).closest("button");
    if (!b) return;
    landKey = b.dataset.k as any;
    landSeg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    drawLand();
  });
  let visible = false;
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(landRoot);
  const spin = () => {
    if (visible && !reducedMotion && (spinning || performance.now() - lastInteract > 6000)) { yaw += 0.004; drawLand(); }
    requestAnimationFrame(spin);
  };
  requestAnimationFrame(spin);

  // ---------------- playback
  function update() {
    for (const c of cells) {
      const fr = data.models[c.k].frames[frame];
      tweenText(c.acc, fr.acc, (v) => pct(v), 300);
      c.sub.textContent = `loss ${fr.loss.toFixed(4)}`;
    }
    drawAll(); drawLayers(); updateProof(); drawLand();
  }
  const player = new Player($("#e2-transport"), nF, (i) => { if (i !== frame) { frame = i; update(); } },
    (i) => `step ${data.models.L1.frames[i].step} / ${data.config.steps} · snapshot ${i + 1}/${nF}`);
  update();
  if (reducedMotion) player.set(nF - 1);
  onceVisible(trio, () => player.autoplay());
}
