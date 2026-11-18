/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Canvas animation runtime for the ovspatialquery guide. No third-party code.
//
// An animation module registers a factory:
//
//   OvsqAnim.register("name", (ctx) => ({ frame(t, dt), resize(), destroy() }))
//
// `ctx` gives the module: g (2D context), size() -> {w, h} in design units, root (the
// .ovsq-anim element) and its data attributes, the live palette P, and drawing helpers.
// Each animation draws in a fixed design space (data-w x data-h, default 760 x 400) that is
// scaled to the element, so text and geometry keep their proportions at every width.
// `frame(t, dt)` gets t in animation seconds; the clock runs only while the element is on
// screen and the reader has not asked for reduced motion. A module may return `holdAt`, the
// time of the still frame shown under reduced motion.
//
// The palette follows the page theme (pydata's html[data-theme]); modules must read colors
// through P every frame rather than caching strings.

(function () {
  const registry = new Map();

  const THEMES = {
    dark: {
      isDark: true, bg: "#10151b", panel: "#161d25", panel2: "#1d2630", line: "#2b3644", grid: "rgba(255,255,255,.05)",
      ink: "#e8edf2", ink2: "#aab6c3", ink3: "#6f7d8c", onAccent: "#0b0f14",
      green: "#76b900", green2: "#9be32a", blue: "#4fb3ff", amber: "#ffb547",
      red: "#ff6b6b", violet: "#b48cff", teal: "#3fd1c1", shadow: "rgba(0,0,0,.35)",
    },
    light: {
      isDark: false, bg: "#fbfcfd", panel: "#f2f5f8", panel2: "#e8edf2", line: "#cdd5de", grid: "rgba(16,24,32,.06)",
      ink: "#18212b", ink2: "#46525f", ink3: "#7b8794", onAccent: "#ffffff",
      green: "#5a8f00", green2: "#4b7a00", blue: "#1f7fd1", amber: "#c27c0e",
      red: "#d64545", violet: "#7a52cc", teal: "#14998b", shadow: "rgba(16,24,32,.18)",
    },
  };
  const P = Object.assign({
    mono: '"JetBrains Mono", "SF Mono", Menlo, Consolas, monospace',
    sans: '"NVIDIA Sans", -apple-system, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif',
  }, THEMES.dark);

  function themeName() {
    const t = document.documentElement.dataset.theme;
    if (t === "light" || t === "dark") return t;
    return window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  function applyTheme() { Object.assign(P, THEMES[themeName()]); }

  const ease = {
    linear: (x) => x,
    inOut: (x) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2),
    out: (x) => 1 - Math.pow(1 - x, 3),
    in: (x) => x * x * x,
    back: (x) => { const c = 1.70158, d = c + 1; return 1 + d * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); },
  };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const seg = (t, t0, t1, fn) => (fn || ease.inOut)(clamp((t - t0) / (t1 - t0), 0, 1));
  const pulse = (t, t0, t1) => { const u = clamp((t - t0) / (t1 - t0), 0, 1); return Math.sin(u * Math.PI); };

  // Color with alpha from "#rrggbb".
  function hex(c, a) {
    if (a === undefined) return c;
    const n = parseInt(c.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  // Lighten (k > 0) or darken (k < 0) "#rrggbb".
  function shade(c, k, alpha) {
    const n = parseInt(c.slice(1), 16);
    const f = (v) => clamp(Math.round(k >= 0 ? v + (255 - v) * k : v * (1 + k)), 0, 255);
    return `rgba(${f((n >> 16) & 255)},${f((n >> 8) & 255)},${f(n & 255)},${alpha === undefined ? 1 : alpha})`;
  }
  function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  function roundRect(g, x, y, w, h, r, fill, stroke, lw) {
    g.beginPath(); g.roundRect(x, y, w, h, r);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw || 1.5; g.stroke(); }
  }
  function text(g, s, x, y, o) {
    o = o || {};
    g.font = `${o.weight || 500} ${o.size || 14}px ${o.mono ? P.mono : P.sans}`;
    g.fillStyle = o.color || P.ink;
    g.textAlign = o.align || "left";
    g.textBaseline = o.base || "middle";
    g.fillText(s, x, y);
  }
  function line(g, x0, y0, x1, y1, o) {
    o = o || {};
    g.save();
    g.strokeStyle = o.color || P.line; g.lineWidth = o.width || 1.5;
    if (o.dash) g.setLineDash(o.dash);
    if (o.cap) g.lineCap = o.cap;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    g.restore();
  }
  function arrow(g, x0, y0, x1, y1, o) {
    o = o || {};
    const c = o.color || P.ink3, lw = o.width || 2, head = o.head || 8;
    const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
    g.save();
    g.strokeStyle = c; g.fillStyle = c; g.lineWidth = lw;
    if (o.dash) g.setLineDash(o.dash);
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1 - ux * head, y1 - uy * head); g.stroke();
    g.setLineDash([]);
    g.beginPath(); g.moveTo(x1, y1);
    g.lineTo(x1 - ux * head - uy * head * 0.55, y1 - uy * head + ux * head * 0.55);
    g.lineTo(x1 - ux * head + uy * head * 0.55, y1 - uy * head - ux * head * 0.55);
    g.closePath(); g.fill();
    g.restore();
  }
  function dot(g, x, y, r, fill, stroke, lw) {
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw || 1.5; g.stroke(); }
  }
  function glow(g, x, y, r, color, a) {
    const grd = g.createRadialGradient(x, y, 0, x, y, r);
    grd.addColorStop(0, hex(color, a === undefined ? 0.45 : a)); grd.addColorStop(1, hex(color, 0));
    g.fillStyle = grd; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
  }
  // Pill-shaped label; returns its width.
  function chip(g, s, x, y, o) {
    o = o || {};
    const size = o.size || 12;
    g.font = `${o.weight || 600} ${size}px ${o.mono === false ? P.sans : P.mono}`;
    const w = g.measureText(s).width + 16, h = size + 10;
    const ax = o.align === "center" ? x - w / 2 : o.align === "right" ? x - w : x;
    roundRect(g, ax, y - h / 2, w, h, h / 2, o.fill || P.panel2, o.stroke || P.line, 1.2);
    text(g, s, ax + w / 2, y + 0.5, { size, color: o.color || P.ink, align: "center", mono: o.mono !== false, weight: o.weight || 600 });
    return w;
  }

  // 2.5D projection: world (x, y, z) with y up, viewed from a fixed corner.
  function iso(scale, ox, oy, angle) {
    const a = angle === undefined ? Math.PI / 6 : angle;
    const ca = Math.cos(a), sa = Math.sin(a);
    const api = {
      p(x, y, z) { return [ox + (x - z) * ca * scale, oy + (x + z) * sa * scale - y * scale]; },
      poly(g, pts, fill, stroke, lw) {
        g.beginPath(); pts.forEach((q, i) => (i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]))); g.closePath();
        if (fill) { g.fillStyle = fill; g.fill(); }
        if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw || 1; g.stroke(); }
      },
      // Axis-aligned box with three shaded faces.
      box(g, x, y, z, w, h, d, color, o) {
        o = o || {};
        const p = api.p, al = o.alpha === undefined ? 1 : o.alpha, st = o.stroke === false ? null : (o.stroke || P.shadow);
        api.poly(g, [p(x + w, y, z), p(x + w, y, z + d), p(x + w, y + h, z + d), p(x + w, y + h, z)], shade(color, -0.3, al), st);
        api.poly(g, [p(x, y, z + d), p(x + w, y, z + d), p(x + w, y + h, z + d), p(x, y + h, z + d)], shade(color, -0.12, al), st);
        api.poly(g, [p(x, y + h, z), p(x + w, y + h, z), p(x + w, y + h, z + d), p(x, y + h, z + d)], shade(color, 0.14, al), st);
        if (o.outline) {
          g.save(); if (o.dash) g.setLineDash(o.dash);
          const edges = [[x, y, z], [x + w, y, z], [x + w, y, z + d], [x, y, z + d]];
          g.strokeStyle = o.outline; g.lineWidth = o.outlineWidth || 1.5;
          for (const yy of [y, y + h]) { g.beginPath(); edges.forEach((e, i) => { const q = p(e[0], yy, e[2]); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); }); g.closePath(); g.stroke(); }
          for (const e of edges) { const a0 = p(e[0], y, e[2]), a1 = p(e[0], y + h, e[2]); g.beginPath(); g.moveTo(a0[0], a0[1]); g.lineTo(a1[0], a1[1]); g.stroke(); }
          g.restore();
        }
      },
      floor(g, x0, z0, x1, z1, fill, stroke) {
        api.poly(g, [api.p(x0, 0, z0), api.p(x1, 0, z0), api.p(x1, 0, z1), api.p(x0, 0, z1)], fill, stroke, 1.2);
      },
      grid(g, x0, z0, x1, z1, step, color) {
        g.save(); g.strokeStyle = color || P.grid; g.lineWidth = 1;
        for (let x = x0; x <= x1 + 1e-6; x += step) { const a = api.p(x, 0, z0), b = api.p(x, 0, z1); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); }
        for (let z = z0; z <= z1 + 1e-6; z += step) { const a = api.p(x0, 0, z), b = api.p(x1, 0, z); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); }
        g.restore();
      },
    };
    return api;
  }

  const reduceMotion = () => window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  function mount(root) {
    const factory = registry.get(root.dataset.anim);
    if (!factory) return;
    const W = parseFloat(root.dataset.w || "760"), H = parseFloat(root.dataset.h || "400");
    const stage = root.querySelector(".ovsq-anim-stage");
    stage.style.aspectRatio = `${W} / ${H}`;
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-hidden", "true");
    stage.appendChild(canvas);
    root.classList.add("is-live");
    const g = canvas.getContext("2d");
    const ctx = { g, root, P, ease, seg, pulse, clamp, lerp, hex, shade, rng, iso,
      roundRect, text, arrow, line, dot, glow, chip, size: () => ({ w: W, h: H }) };
    const inst = factory(ctx) || {};
    // Under reduced motion, show one still frame: data-hold-at if set, else the module's own.
    const holdAt = parseFloat(root.dataset.holdAt ?? String(inst.holdAt ?? 0)) || 0;
    let t = reduceMotion() ? holdAt : 0;
    let playing = !reduceMotion(), visible = false, raf = 0, last = 0;
    const speed = parseFloat(root.dataset.speed || "1") || 1;

    const btn = document.createElement("button");
    btn.className = "ovsq-anim-toggle"; btn.type = "button";
    stage.appendChild(btn);
    const restart = document.createElement("button");
    restart.className = "ovsq-anim-restart"; restart.type = "button";
    restart.setAttribute("aria-label", "Restart animation"); restart.title = "Restart";
    restart.textContent = "↻";
    stage.appendChild(restart);
    function paintBtn() {
      btn.textContent = playing ? "❚❚" : "▶";
      btn.setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
      btn.title = playing ? "Pause" : "Play";
      root.classList.toggle("is-paused", !playing);
    }
    btn.addEventListener("click", () => { playing = !playing; paintBtn(); kick(); });
    restart.addEventListener("click", () => { t = 0; playing = true; paintBtn(); kick(); });

    function resize() {
      const r = stage.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
      g.setTransform((canvas.width / W), 0, 0, (canvas.height / H), 0, 0);
      inst.resize && inst.resize();
      draw(0);
    }
    function draw(dt) {
      g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, canvas.width, canvas.height); g.restore();
      inst.frame && inst.frame(t, dt);
    }
    function loop(now) {
      raf = 0;
      if (!visible) return;
      const dt = last ? Math.min(0.1, (now - last) / 1000) * speed : 0;
      last = now;
      if (playing) t += dt;
      draw(playing ? dt : 0);
      if (playing) raf = requestAnimationFrame(loop);
    }
    function kick() { last = 0; if (!raf) raf = requestAnimationFrame(loop); }

    new ResizeObserver(resize).observe(stage);
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) kick(); }, { rootMargin: "80px" }).observe(stage);
    new MutationObserver(() => { applyTheme(); draw(0); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    paintBtn();
    resize();
  }

  window.OvsqAnim = {
    // Modules may load after the runtime (deferred or dynamic scripts), so registering also
    // mounts any figure that was waiting for this module.
    P, register(name, factory) { registry.set(name, factory); if (document.readyState !== "loading") window.OvsqAnim.mountAll(); },
    // Mount every animation under `root` that is not already live (for content added later).
    mountAll(root) { (root || document).querySelectorAll(".ovsq-anim[data-anim]:not(.is-live)").forEach(mount); },
  };

  applyTheme();
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => window.OvsqAnim.mountAll());
  else window.OvsqAnim.mountAll();
})();
