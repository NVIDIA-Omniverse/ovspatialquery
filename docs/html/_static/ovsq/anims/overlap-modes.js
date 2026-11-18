/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// overlap_any and overlap_multiple, top view: the same query sphere over the same four objects.
// Which objects it touches is computed from the drawn shapes (sphere against box, sphere and
// capsule). overlap_any reports one of them; overlap_multiple stores them all.

OvsqAnim.register("overlap-modes", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex } = ctx;
  const LOOP = 8, S = 46, OY = 296;
  const query = { x: 3.3, y: 2.2, r: 1.25 };
  const objects = [
    { id: 40, kind: "box", x0: 0.3, x1: 1.6, y0: 1.6, y1: 2.9 },
    { id: 41, kind: "sphere", x: 4.6, y: 3.2, r: 0.5 },
    { id: 42, kind: "capsule", a: [1.4, 0.7], b: [3.0, 0.7], r: 0.35 },
    { id: 43, kind: "box", x0: 4.3, x1: 5.7, y0: 0.5, y1: 1.6 },
  ];
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  // Distance from the query center to each object's surface (negative inside).
  function gap(o) {
    if (o.kind === "box") return Math.hypot(query.x - clamp(query.x, o.x0, o.x1), query.y - clamp(query.y, o.y0, o.y1));
    if (o.kind === "sphere") return Math.hypot(query.x - o.x, query.y - o.y) - o.r;
    const d = [o.b[0] - o.a[0], o.b[1] - o.a[1]];
    const u = clamp(((query.x - o.a[0]) * d[0] + (query.y - o.a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]), 0, 1);
    return Math.hypot(query.x - (o.a[0] + u * d[0]), query.y - (o.a[1] + u * d[1])) - o.r;
  }
  objects.forEach((o) => { o.touches = gap(o) <= query.r; });
  const touched = objects.filter((o) => o.touches);
  const any = touched[touched.length - 1]; // any one touched object; the call promises no choice

  function panel(ox, mode, t, fade) {
    const p = (x, y) => [ox + x * S, OY - y * S];
    const a = seg(t, 0.4, 1.2), lit = seg(t, 1.8, 2.3) * fade;
    for (let x = 0; x <= 6; x++) { const q0 = p(x, 0), q1 = p(x, 4); line(g, q0[0], q0[1], q1[0], q1[1], { color: P.grid, width: 1 }); }
    for (let y = 0; y <= 4; y++) { const q0 = p(0, y), q1 = p(6, y); line(g, q0[0], q0[1], q1[0], q1[1], { color: P.grid, width: 1 }); }
    objects.forEach((o) => {
      const on = lit > 0.5 && (mode === "overlap_any" ? o === any : o.touches);
      const fill = on ? hex(P.green, 0.35) : P.isDark ? "#3a4654" : "#c4ced8";
      const stroke = on ? P.green : hex(P.ink, 0.3);
      let c;
      if (o.kind === "box") {
        const q = p(o.x0, o.y1);
        roundRect(g, q[0], q[1], (o.x1 - o.x0) * S, (o.y1 - o.y0) * S, 3, fill, stroke, 1.4);
        c = p((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2);
      } else if (o.kind === "sphere") {
        c = p(o.x, o.y); dot(g, c[0], c[1], o.r * S, fill, stroke, 1.4);
      } else {
        const q = p(o.a[0] - o.r, o.a[1] + o.r);
        roundRect(g, q[0], q[1], (o.b[0] - o.a[0] + 2 * o.r) * S, 2 * o.r * S, o.r * S, fill, stroke, 1.4);
        c = p((o.a[0] + o.b[0]) / 2, o.a[1]);
      }
      text(g, String(o.id), c[0], c[1], { size: 12, weight: 700, mono: true, align: "center" });
    });
    g.globalAlpha = a;
    const qc = p(query.x, query.y);
    dot(g, qc[0], qc[1], query.r * S, hex(P.amber, 0.14), P.amber, 2);
    text(g, "SPHERE", qc[0], qc[1], { size: 11, mono: true, weight: 700, color: P.amber, align: "center" });
    g.globalAlpha = 1;

    text(g, mode, ox - 16, 24, { size: 13, weight: 700, mono: true });
    text(g, mode === "overlap_any" ? "one touched object, any one" : "every touched object, up to capacity", ox - 16, 46, { size: 11.5, color: P.ink3 });
    if (lit > 0) {
      g.globalAlpha = lit;
      const s = mode === "overlap_any" ? `hit: object ${any.id}` : `stored_hit_count = ${touched.length}, truncated = false`;
      chip(g, s, ox - 16, 76, { size: 11, color: P.green2, fill: P.panel, stroke: P.green });
      g.globalAlpha = 1;
    }
  }

  return {
    // The frame shown under reduced motion: both results are on screen.
    holdAt: 5,
    frame(tt) {
      const t = tt % LOOP;
      const fade = 1 - seg(t, LOOP - 0.6, LOOP);
      panel(40, "overlap_any", t, fade);
      line(g, 380, 16, 380, 310, { color: P.line, width: 1 });
      panel(420, "overlap_multiple", t, fade);
    },
  };
});
