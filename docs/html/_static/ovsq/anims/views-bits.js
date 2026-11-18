/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One ray, three targets. A view narrows which objects a query can see; participation bits
// narrow it per query with (query_bits & object_bits) != 0. The first accepted object along the
// ray is computed for each phase from the shapes drawn.

OvsqAnim.register("views-bits", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex } = ctx;
  const LOOP = 18, PHASE = 6;
  const GROUND = 0x1, OBSTACLE = 0x2, VEHICLE = 0x4;
  const bitsName = (b) => (b === 0 ? "0" : [[GROUND, "GROUND"], [OBSTACLE, "OBSTACLE"], [VEHICLE, "VEHICLE"]].filter(([m]) => b & m).map(([, n]) => n).join("|"));

  // Side view, y down in screen units. Rectangles and one ramp triangle.
  const FLOOR = 300;
  const objects = [
    { name: "/World/ground", bits: GROUND, kind: "rect", x: 24, y: FLOOR, w: 452, h: 16, color: "#5d6b78" },
    { name: "/World/ramp", bits: GROUND, kind: "ramp", x: 300, y: FLOOR, w: 150, h: 64, color: "#6f8091" },
    { name: "/World/robot", bits: VEHICLE, kind: "rect", x: 132, y: FLOOR - 46, w: 64, h: 46, color: "#76b900" },
    { name: "/World/crate", bits: OBSTACLE, kind: "rect", x: 222, y: FLOOR - 58, w: 52, h: 58, color: "#b08a5a" },
    { name: "/World/wall", bits: OBSTACLE, kind: "rect", x: 452, y: FLOOR - 150, w: 18, h: 150, color: "#8a96a3" },
  ];
  const VIEW = ["/World/ground", "/World/ramp"];
  const phases = [
    { target: "population scene", bits: 0, view: false },
    { target: "population scene", bits: OBSTACLE, view: false },
    { target: "view drivable_surfaces", bits: 0, view: true },
  ];

  const O = [40, 236], Dv = (() => { const d = [480, 84]; const l = Math.hypot(...d); return [d[0] / l, d[1] / l]; })();

  function hitRect(o) {
    let t0 = 0, t1 = Infinity;
    for (const [oo, d, lo, hi] of [[O[0], Dv[0], o.x, o.x + o.w], [O[1], Dv[1], o.y - (o.kind === "rect" ? 0 : 0), o.y + o.h]]) {
      if (Math.abs(d) < 1e-9) { if (oo < lo || oo > hi) return null; continue; }
      let a = (lo - oo) / d, b = (hi - oo) / d; if (a > b) [a, b] = [b, a];
      t0 = Math.max(t0, a); t1 = Math.min(t1, b); if (t0 > t1) return null;
    }
    return t0;
  }
  function hitRamp(o) {
    // Surface from (x, FLOOR) up to (x + w, FLOOR - h).
    const ax = o.x, ay = o.y, bx = o.x + o.w, by = o.y - o.h;
    const ex = bx - ax, ey = by - ay, den = Dv[0] * ey - Dv[1] * ex;
    if (Math.abs(den) < 1e-9) return null;
    const t = ((ax - O[0]) * ey - (ay - O[1]) * ex) / den, u = ((ax - O[0]) * Dv[1] - (ay - O[1]) * Dv[0]) / den;
    return t > 0 && u >= 0 && u <= 1 ? t : null;
  }
  const rectOf = (o) => (o.kind === "rect" ? o : { ...o, y: o.y - o.h });
  const distTo = (o) => (o.kind === "ramp" ? hitRamp(o) : hitRect(rectOf(o)));
  const accepted = (o, ph) => (!ph.view || VIEW.includes(o.name)) && (ph.bits === 0 || (ph.bits & o.bits) !== 0);
  const results = phases.map((ph) => {
    let best = null;
    for (const o of objects) { if (!accepted(o, ph)) continue; const d = distTo(o); if (d !== null && (!best || d < best.d)) best = { d, o }; }
    return best;
  });

  function drawObject(o, on) {
    const a = on ? 1 : 0.22;
    g.save(); g.globalAlpha = a;
    if (o.kind === "ramp") {
      g.beginPath(); g.moveTo(o.x, o.y); g.lineTo(o.x + o.w, o.y - o.h); g.lineTo(o.x + o.w, o.y); g.closePath();
      g.fillStyle = hex(o.color, 0.85); g.fill(); g.strokeStyle = P.shadow; g.lineWidth = 1; g.stroke();
    } else if (o.name === "/World/robot") {
      roundRect(g, o.x, o.y + 10, o.w, o.h - 18, 6, hex(o.color, 0.9), P.shadow, 1);
      roundRect(g, o.x + 18, o.y, 26, 12, 4, hex(o.color, 0.9));
      for (const wx of [o.x + 14, o.x + o.w - 14]) dot(g, wx, o.y + o.h - 7, 7, P.isDark ? "#222a33" : "#3a4654");
    } else roundRect(g, o.x, o.y, o.w, o.h, 3, hex(o.color, 0.9), P.shadow, 1);
    g.restore();
  }

  return {
    // The frame shown under reduced motion: the result is on screen.
    holdAt: 10,
    frame(tt) {
      const t = tt % LOOP, pi = Math.floor(t / PHASE), ph = phases[pi], lt = t - pi * PHASE;
      const res = results[pi];

      // Header.
      text(g, "target", 24, 30, { size: 11.5, mono: true, color: P.ink3 });
      chip(g, ph.target, 76, 30, { size: 11.5, color: ph.view ? P.violet : P.ink, stroke: ph.view ? P.violet : P.line, fill: P.panel });
      text(g, "query_bits", 24, 60, { size: 11.5, mono: true, color: P.ink3 });
      chip(g, ph.bits === 0 ? "0  (accept all)" : `${bitsName(ph.bits)}  0x${ph.bits.toString(16)}`, 106, 60, { size: 11.5, color: ph.bits ? P.amber : P.ink, stroke: ph.bits ? P.amber : P.line, fill: P.panel });

      // Scene.
      const fade = seg(lt, 0.2, 0.9);
      for (const o of objects) drawObject(o, accepted(o, ph) || fade < 1 && pi > 0 && accepted(o, phases[pi - 1]) && (1 - fade) > 0.5);

      // Sensor and ray.
      roundRect(g, O[0] - 12, O[1] - 9, 24, 18, 5, P.panel2, P.blue, 1.4);
      const travel = seg(lt, 1.0, 2.6, ctx.ease.inOut);
      const end = res ? res.d : 520;
      const cur = end * travel;
      // Rejected objects the ray passes through are marked where it crosses them.
      for (const o of objects) {
        if (accepted(o, ph)) continue;
        const d = distTo(o);
        if (d !== null && d < cur && (!res || d < res.d)) {
          const p = [O[0] + Dv[0] * d, O[1] + Dv[1] * d];
          line(g, p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5, { color: P.red, width: 2 });
          line(g, p[0] - 5, p[1] + 5, p[0] + 5, p[1] - 5, { color: P.red, width: 2 });
        }
      }
      line(g, O[0], O[1], O[0] + Dv[0] * cur, O[1] + Dv[1] * cur, { color: P.blue, width: 2.2, cap: "round" });
      if (travel >= 1 && res) {
        const p = [O[0] + Dv[0] * res.d, O[1] + Dv[1] * res.d];
        glow(g, p[0], p[1], 18, P.green, 0.6); dot(g, p[0], p[1], 4.5, P.green2);
        chip(g, `hit ${res.o.name}`, p[0] + 10, p[1] - 22, { size: 11, color: P.green2, fill: P.panel, stroke: P.green });
      }

      // Object table: bits and the filter verdict.
      const tx = 500, ty = 104;
      text(g, "object", tx, ty - 22, { size: 11, mono: true, color: P.ink3 });
      text(g, "bits", tx + 116, ty - 22, { size: 11, mono: true, color: P.ink3 });
      objects.forEach((o, i) => {
        const y = ty + i * 30, ok = accepted(o, ph), inView = !ph.view || VIEW.includes(o.name);
        const show = seg(lt, 0.4 + i * 0.12, 0.8 + i * 0.12);
        roundRect(g, tx - 8, y - 12, 250, 24, 6, ok ? hex(P.green, 0.08) : hex(P.ink, 0.03), ok ? hex(P.green, 0.4) : P.line, 1);
        text(g, o.name.replace("/World/", ""), tx, y, { size: 11.5, mono: true, color: ok ? P.ink : P.ink3 });
        if (inView) text(g, bitsName(o.bits), tx + 116, y, { size: 11, mono: true, color: ok ? P.ink2 : P.ink3 });
        else text(g, "outside the view", tx + 116, y, { size: 10.5, mono: true, color: P.violet });
        g.globalAlpha = show;
        text(g, ok ? "✓" : "✕", tx + 230, y, { size: 13, weight: 700, align: "center", color: ok ? P.green2 : P.red });
        g.globalAlpha = 1;
      });
      const rule = ph.view ? "a view sees only what its filter selected" : ph.bits ? "accepted: (query_bits & object_bits) != 0" : "query_bits = 0 accepts every object";
      text(g, rule, 24, 336, { size: 11, mono: true, color: ph.view ? P.violet : ph.bits ? P.amber : P.ink3 });

      // Phase progress.
      for (let k = 0; k < 3; k++) roundRect(g, 24 + k * 240, 356, 232, 3, 1.5, k === pi ? P.green : hex(P.ink, 0.08));
      roundRect(g, 24 + pi * 240, 356, 232 * Math.min(1, lt / PHASE), 3, 1.5, P.green2);
    },
  };
});
