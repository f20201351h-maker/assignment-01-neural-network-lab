// Shared plumbing for all four experiments: canvas, colour, the tiny network
// forward pass, contour tracing, playback, charts and tooltips.

export const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export const C = {
  bg: "#0a0d16",
  c0: "#2fe0d5",      // class 0 / inner ring
  c1: "#ff5c8a",      // class 1 / outer ring
  lin: "#ffb547",     // linear model
  relu: "#a98bff",    // ReLU model
  train: "#c6f45c",   // training data / training curve
  test: "#ff9a3d",    // held-out data / test curve
  ink: "#e9edf7",
  dim: "#8a93a9",
};

export const $ = <T extends Element = HTMLElement>(sel: string, root: ParentNode = document) =>
  root.querySelector(sel) as T;
export const $$ = <T extends Element = HTMLElement>(sel: string, root: ParentNode = document) =>
  Array.from(root.querySelectorAll(sel)) as T[];

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, any> = {}, ...kids: (Node | string)[]) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, String(v));
  }
  e.append(...kids);
  return e;
}

export const svgNS = "http://www.w3.org/2000/svg";
export function s<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, any> = {}) {
  const e = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
/** 2.2e-16 -> "2.2 × 10⁻¹⁶" */
export const sci = (v: number) =>
  v === 0 ? "0" : v.toExponential(1).replace(/e([+-])(\d+)/, (_, sg, d) => ` × 10${sg === "-" ? "⁻" : ""}${[...d].map((c) => SUP[+c]).join("")}`);
export const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function rgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
export const rgba = (hex: string, a: number) => {
  const [r, g, b] = rgb(hex);
  return `rgba(${r},${g},${b},${a})`;
};

// ------------------------------------------------------------- canvas stage
/** A canvas that tracks its CSS size and device pixel ratio. Draw in CSS px. */
export class Stage {
  ctx: CanvasRenderingContext2D;
  w = 0;
  h = 0;
  constructor(public canvas: HTMLCanvasElement, private onResize: () => void, private aspect = 1) {
    this.ctx = canvas.getContext("2d")!;
    this.resize(false);  // size now; first draw comes from the observer, after the caller finished setting up
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement!);
  }
  resize(emit = true) {
    const w = this.canvas.parentElement!.clientWidth;
    const h = Math.round(w / this.aspect);
    if (w === this.w && h === this.h) { if (emit) this.onResize(); return; }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = w; this.h = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = w + "px";
    this.canvas.style.height = h + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (emit) this.onResize();
  }
}

export type Domain = { x: [number, number]; y: [number, number] };
/** world <-> screen mapping for a square-ish plot. */
export function mapper(d: Domain, w: number, hgt: number) {
  const sx = (x: number) => ((x - d.x[0]) / (d.x[1] - d.x[0])) * w;
  const sy = (y: number) => (1 - (y - d.y[0]) / (d.y[1] - d.y[0])) * hgt;
  const wx = (px: number) => d.x[0] + (px / w) * (d.x[1] - d.x[0]);
  const wy = (py: number) => d.y[0] + (1 - py / hgt) * (d.y[1] - d.y[0]);
  return { sx, sy, wx, wy };
}

// ---------------------------------------------------- the network, in JS
export type Layer = { W: number[][]; b: number[] };

/** Compile exported layers into a fast float64 forward pass returning the logit. */
export function makeNet(layers: Layer[], relu: boolean) {
  const Ws = layers.map((L) => Float64Array.from(L.W.flat()));
  const bs = layers.map((L) => Float64Array.from(L.b));
  const dims = [layers[0].W[0].length, ...layers.map((L) => L.b.length)];
  const bufs = dims.map((d) => new Float64Array(d));
  return (x: number, y: number) => {
    bufs[0][0] = x; bufs[0][1] = y;
    for (let l = 0; l < Ws.length; l++) {
      const inp = bufs[l], out = bufs[l + 1], W = Ws[l], b = bs[l];
      const nin = dims[l], nout = dims[l + 1];
      const last = l === Ws.length - 1;
      for (let o = 0; o < nout; o++) {
        let acc = b[o];
        for (let i = 0; i < nin; i++) acc += W[o * nin + i] * inp[i];
        out[o] = relu && !last && acc < 0 ? 0 : acc;
      }
    }
    return bufs[Ws.length][0];
  };
}
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Probability grid, row 0 = TOP of the plot (largest y). */
export function probGrid(f: (x: number, y: number) => number, d: Domain, n: number) {
  const g = new Float32Array(n * n);
  for (let r = 0; r < n; r++) {
    const y = d.y[1] - (r / (n - 1)) * (d.y[1] - d.y[0]);
    for (let c = 0; c < n; c++) g[r * n + c] = sigmoid(f(d.x[0] + (c / (n - 1)) * (d.x[1] - d.x[0]), y));
  }
  return g;
}

/** Paint a probability grid into an offscreen canvas (bilinear-smoothed when drawn). */
export function fieldCanvas(g: ArrayLike<number>, gw: number, gh: number, scale = 1, c0 = C.c0, c1 = C.c1) {
  const cv = document.createElement("canvas");
  cv.width = gw; cv.height = gh;
  const ctx = cv.getContext("2d")!;
  const img = ctx.createImageData(gw, gh);
  const A = rgb(c0), B = rgb(c1), bg = rgb(C.bg);
  for (let i = 0; i < gw * gh; i++) {
    const p = g[i] / scale;
    const t = (p - 0.5) * 2;
    const col = t < 0 ? A : B;
    const a = Math.pow(Math.abs(t), 0.85) * 0.42 + 0.04;
    img.data[i * 4] = bg[0] + (col[0] - bg[0]) * a;
    img.data[i * 4 + 1] = bg[1] + (col[1] - bg[1]) * a;
    img.data[i * 4 + 2] = bg[2] + (col[2] - bg[2]) * a;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

/** Marching squares: line segments where the grid crosses `level` (grid coords). */
export function contour(g: ArrayLike<number>, gw: number, gh: number, level = 0.5): number[] {
  const segs: number[] = [];
  const v = (r: number, c: number) => g[r * gw + c] - level;
  const interp = (a: number, b: number) => a / (a - b);
  for (let r = 0; r < gh - 1; r++) {
    for (let c = 0; c < gw - 1; c++) {
      const tl = v(r, c), tr = v(r, c + 1), br = v(r + 1, c + 1), bl = v(r + 1, c);
      const pts: number[] = [];
      if ((tl > 0) !== (tr > 0)) pts.push(c + interp(tl, tr), r);
      if ((tr > 0) !== (br > 0)) pts.push(c + 1, r + interp(tr, br));
      if ((bl > 0) !== (br > 0)) pts.push(c + interp(bl, br), r + 1);
      if ((tl > 0) !== (bl > 0)) pts.push(c, r + interp(tl, bl));
      if (pts.length === 4) segs.push(...pts);
      else if (pts.length === 8) segs.push(pts[0], pts[1], pts[6], pts[7], pts[2], pts[3], pts[4], pts[5]);
    }
  }
  return segs;
}

export function strokeSegs(ctx: CanvasRenderingContext2D, segs: number[], kx: number, ky: number, color: string, width: number, glow = true, dash: number[] = []) {
  ctx.save();
  ctx.setLineDash(dash);
  ctx.lineCap = "round";
  ctx.strokeStyle = color;
  ctx.beginPath();
  for (let i = 0; i < segs.length; i += 4) {
    ctx.moveTo(segs[i] * kx, segs[i + 1] * ky);
    ctx.lineTo(segs[i + 2] * kx, segs[i + 3] * ky);
  }
  if (glow) {
    ctx.shadowColor = color; ctx.shadowBlur = 14; ctx.lineWidth = width + 2; ctx.globalAlpha = 0.45; ctx.stroke();
    ctx.shadowBlur = 0; ctx.globalAlpha = 1;
  }
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.restore();
}

export function drawDot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, hollow = false) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (hollow) { ctx.lineWidth = 1.6; ctx.strokeStyle = color; ctx.stroke(); }
  else { ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(5,8,15,.75)"; ctx.stroke(); }
}

export function drawCross(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.save();
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.arc(x, y, r + 3.5, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
}

// ------------------------------------------------------------ playback bar
/** Play / pause + scrubber over recorded snapshots. */
export class Player {
  index = 0;
  playing = false;
  touched = false;   // once the visitor drives the scrubber, never auto-play over them
  private raf = 0;
  private last = 0;
  private acc = 0;
  btn: HTMLButtonElement;
  range: HTMLInputElement;
  label: HTMLElement;
  constructor(public root: HTMLElement, public count: number, public onChange: (i: number, frac: number) => void,
              public describe: (i: number) => string, public fps = 11) {
    this.btn = h("button", { class: "play", "aria-label": "Play training" }) as HTMLButtonElement;
    this.btn.innerHTML = playIcon;
    this.range = h("input", { type: "range", min: 0, max: count - 1, step: 1, value: 0, "aria-label": "Training snapshot" }) as HTMLInputElement;
    this.label = h("span", { class: "tlabel mono" });
    const restart = h("button", { class: "ghost-btn", "aria-label": "Replay from the start", title: "Replay" });
    restart.innerHTML = replayIcon;
    restart.addEventListener("click", () => { this.set(0); this.play(); });
    root.append(this.btn, restart, h("div", { class: "range-wrap" }, this.range), this.label);
    this.btn.addEventListener("click", () => { this.touched = true; this.playing ? this.pause() : this.play(); });
    this.range.addEventListener("input", () => { this.touched = true; this.pause(); this.set(+this.range.value); });
    this.set(0, false);
  }
  setCount(n: number) {
    this.count = n;
    this.range.max = String(n - 1);
    if (this.index > n - 1) this.set(n - 1);
  }
  set(i: number, emit = true) {
    this.index = clamp(i, 0, this.count - 1);
    this.range.value = String(this.index);
    this.range.style.setProperty("--fill", `${(this.index / Math.max(1, this.count - 1)) * 100}%`);
    this.label.textContent = this.describe(this.index);
    if (emit) this.onChange(this.index, 0);
  }
  play() {
    if (this.index >= this.count - 1) this.set(0);
    this.playing = true;
    this.btn.innerHTML = pauseIcon;
    this.btn.setAttribute("aria-label", "Pause training");
    this.root.classList.add("is-playing");
    this.last = performance.now();
    this.acc = 0;
    const tick = (t: number) => {
      if (!this.playing) return;
      this.acc += (t - this.last) / 1000;
      this.last = t;
      const step = 1 / this.fps;
      if (this.acc >= step) {
        this.acc = 0;
        if (this.index >= this.count - 1) return this.pause();
        this.set(this.index + 1);
      } else if (!reducedMotion) {
        this.onChange(this.index, this.acc / step);   // presentation-only cross-fade toward the next snapshot
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }
  /** Play from the start when a section scrolls into view, unless the visitor already took control. */
  autoplay() {
    if (this.touched || reducedMotion) return;
    this.set(0);
    this.play();
  }
  pause() {
    this.playing = false;
    cancelAnimationFrame(this.raf);
    this.btn.innerHTML = playIcon;
    this.btn.setAttribute("aria-label", "Play training");
    this.root.classList.remove("is-playing");
  }
}
export const playIcon = `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M7 4.5v15l13-7.5z"/></svg>`;
export const pauseIcon = `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>`;
export const replayIcon = `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4.5h4.5"/></svg>`;

// ------------------------------------------------------------- tooltip
const tip = h("div", { class: "tip", role: "status", "aria-live": "polite" });
document.body.append(tip);
export function showTip(clientX: number, clientY: number, html: string) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const r = tip.getBoundingClientRect();
  let x = clientX + 16, y = clientY + 16;
  if (x + r.width > window.innerWidth - 8) x = clientX - r.width - 16;
  if (y + r.height > window.innerHeight - 8) y = clientY - r.height - 16;
  tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
}
export const hideTip = () => tip.classList.remove("on");

// ----------------------------------------------------------- number tween
export function tweenText(el: Element, to: number, fmt: (v: number) => string, ms = 650) {
  const from = (el as any)._v ?? to;
  (el as any)._v = to;
  if (reducedMotion || from === to) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const id = ((el as any)._id = ((el as any)._id || 0) + 1);
  const step = (t: number) => {
    if ((el as any)._id !== id) return;
    const k = clamp((t - t0) / ms, 0, 1);
    el.textContent = fmt(lerp(from, to, ease(k)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/** Run cb once the element scrolls into view. */
export function onceVisible(el: Element, cb: () => void, threshold = 0.35) {
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); cb(); }
  }, { threshold });
  io.observe(el);
}

// ------------------------------------------------------------ line chart
export type Series = { name: string; color: string; xs: number[]; ys: number[]; dash?: string; width?: number };
/** Minimal responsive SVG line chart with a scrub cursor. */
export class LineChart {
  svg: SVGSVGElement;
  private g: SVGGElement;
  private cursor: SVGLineElement;
  private dots: SVGGElement;
  W = 400; H = 170;
  pad = { l: 40, r: 12, t: 12, b: 36 };
  xDom: [number, number] = [0, 1];
  yDom: [number, number] = [0, 1];
  xLog = false;
  series: Series[] = [];
  onSeek?: (x: number) => void;
  constructor(public root: HTMLElement, opts: { H?: number; xLog?: boolean; xLabel?: string; yLabel?: string } = {}) {
    if (opts.H) this.H = opts.H;
    this.xLog = !!opts.xLog;
    this.svg = s("svg", { viewBox: `0 0 ${this.W} ${this.H}`, class: "chart", role: "img" }) as SVGSVGElement;
    this.g = s("g") as SVGGElement;
    this.cursor = s("line", { class: "cursor", y1: this.pad.t, y2: this.H - this.pad.b }) as SVGLineElement;
    this.dots = s("g") as SVGGElement;
    this.svg.append(this.g, this.cursor, this.dots);
    if (opts.xLabel) this.svg.append(Object.assign(s("text", { x: this.W - this.pad.r, y: this.H - 2, class: "axis-label", "text-anchor": "end" }), { textContent: opts.xLabel }));
    if (opts.yLabel) this.svg.append(Object.assign(s("text", { x: 4, y: 10, class: "axis-label" }), { textContent: opts.yLabel }));
    root.append(this.svg);
    const seek = (e: PointerEvent) => {
      if (!this.onSeek) return;
      const r = this.svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * this.W;
      this.onSeek(this.ix(px));
    };
    this.svg.addEventListener("pointerdown", (e) => { seek(e); this.svg.setPointerCapture(e.pointerId); });
    this.svg.addEventListener("pointermove", (e) => { if (e.buttons) seek(e); });
  }
  x(v: number) {
    const [a, b] = this.xDom;
    const t = this.xLog ? (Math.log(v) - Math.log(a)) / (Math.log(b) - Math.log(a)) : (v - a) / (b - a);
    return this.pad.l + t * (this.W - this.pad.l - this.pad.r);
  }
  ix(px: number) {
    const t = clamp((px - this.pad.l) / (this.W - this.pad.l - this.pad.r), 0, 1);
    const [a, b] = this.xDom;
    return this.xLog ? Math.exp(Math.log(a) + t * (Math.log(b) - Math.log(a))) : a + t * (b - a);
  }
  y(v: number) {
    const [a, b] = this.yDom;
    return this.H - this.pad.b - ((clamp(v, a, b) - a) / (b - a)) * (this.H - this.pad.t - this.pad.b);
  }
  path(xs: number[], ys: number[]) {
    return xs.map((x, i) => `${i ? "L" : "M"}${this.x(x).toFixed(1)},${this.y(ys[i]).toFixed(1)}`).join("");
  }
  /** Draw axes + series. `extra` lets callers add bands/markers in chart coordinates. */
  draw(series: Series[], xTicks: number[], yTicks: number[], fmtX = (v: number) => String(v), fmtY = (v: number) => String(v), extra?: (g: SVGGElement) => void) {
    this.series = series;
    this.g.innerHTML = "";
    for (const t of yTicks) {
      this.g.append(s("line", { x1: this.pad.l, x2: this.W - this.pad.r, y1: this.y(t), y2: this.y(t), class: "grid" }));
      this.g.append(Object.assign(s("text", { x: this.pad.l - 6, y: this.y(t) + 3.5, class: "tick", "text-anchor": "end" }), { textContent: fmtY(t) }));
    }
    for (const t of xTicks) {
      this.g.append(Object.assign(s("text", { x: this.x(t), y: this.H - this.pad.b + 15, class: "tick", "text-anchor": "middle" }), { textContent: fmtX(t) }));
    }
    extra?.(this.g);
    for (const se of series) {
      const p = s("path", { d: this.path(se.xs, se.ys), fill: "none", stroke: se.color, "stroke-width": se.width ?? 2.2, "stroke-linejoin": "round", "stroke-linecap": "round", class: "series" });
      if (se.dash) p.setAttribute("stroke-dasharray", se.dash);
      this.g.append(p);
    }
  }
  /** Move the cursor; mark each series at the given x-index. */
  mark(xv: number, idx: number | null) {
    const x = this.x(xv);
    this.cursor.setAttribute("x1", String(x));
    this.cursor.setAttribute("x2", String(x));
    this.dots.innerHTML = "";
    if (idx === null) return;
    for (const se of this.series) {
      if (idx >= se.ys.length) continue;
      this.dots.append(s("circle", { cx: this.x(se.xs[idx]), cy: this.y(se.ys[idx]), r: 4.2, fill: se.color, stroke: C.bg, "stroke-width": 2 }));
    }
  }
}

export async function loadJSON<T = any>(name: string): Promise<T> {
  const r = await fetch(`data/${name}`);
  if (!r.ok) throw new Error(`could not load ${name}`);
  return r.json();
}

/** Fill every [data-bind="key"] inside root with text from `values`. */
export function bind(values: Record<string, string>, root: ParentNode = document) {
  for (const el of $$("[data-bind]", root)) {
    const k = el.dataset.bind!;
    if (k in values) el.textContent = values[k];
  }
}
