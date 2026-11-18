/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// sweep_box_closest, top view: a box turned 25 degrees moves along +x without rotating and stops
// at its first contact, where the corner of box 30 meets its lower edge. The contact distance
// and point come from a swept separating-axis test on the drawn boxes.

OvsqAnim.register("box-sweep", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex, arrow } = ctx;
  const LOOP = 9, S = 56, OX = 44, OY = 292, MAX = 9;
  const p = (x, y) => [OX + x * S, OY - y * S];
  const start = { x: 1.0, y: 2.0 }, HALF = [0.6, 0.4], ANGLE = (25 * Math.PI) / 180;
  const dir = [1, 0];
  const boxes = [
    { id: 30, x0: 5.0, x1: 6.2, y0: 0.4, y1: 1.65 },
    { id: 31, x0: 7.6, x1: 8.6, y0: 1.9, y1: 3.0 },
  ];

  const ca = Math.cos(ANGLE), sa = Math.sin(ANGLE);
  const local = [[HALF[0], HALF[1]], [-HALF[0], HALF[1]], [-HALF[0], -HALF[1]], [HALF[0], -HALF[1]]];
  const moving = (s) => local.map(([x, y]) => [start.x + x * ca - y * sa + dir[0] * s, start.y + x * sa + y * ca + dir[1] * s]);
  const corners = (b) => [[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]];
  const dotp = (a, b) => a[0] * b[0] + a[1] * b[1];
  const span = (pts, n) => { const v = pts.map((q) => dotp(q, n)); return [Math.min(...v), Math.max(...v)]; };

  // Swept separating axes: the first s at which the moving box touches b, and where.
  function contact(b) {
    const A = moving(0), B = corners(b);
    const axes = [{ n: [ca, sa], ofA: true }, { n: [-sa, ca], ofA: true }, { n: [1, 0], ofA: false }, { n: [0, 1], ofA: false }];
    let enter = -Infinity, exit = Infinity, hitAxis = null;
    for (const ax of axes) {
      const [a0, a1] = span(A, ax.n), [b0, b1] = span(B, ax.n), v = dotp(dir, ax.n);
      if (Math.abs(v) < 1e-12) { if (a1 < b0 || b1 < a0) return null; continue; }
      let t0 = (b0 - a1) / v, t1 = (b1 - a0) / v;
      if (t0 > t1) [t0, t1] = [t1, t0];
      if (t0 > enter) { enter = t0; hitAxis = { ...ax, v }; }
      exit = Math.min(exit, t1);
    }
    if (enter > exit || enter < 0) return null;
    // The touching feature: the moving box's leading corner on b's face, or b's corner on the
    // moving box's face, whichever axis separated last.
    const A1 = moving(enter), n = hitAxis.n, sgn = Math.sign(hitAxis.v);
    const point = hitAxis.ofA
      ? B.reduce((best, q) => (sgn * dotp(q, n) < sgn * dotp(best, n) ? q : best))
      : A1.reduce((best, q) => (sgn * dotp(q, n) > sgn * dotp(best, n) ? q : best));
    return { s: enter, point };
  }
  let first = null;
  boxes.forEach((b) => { const c = contact(b); if (c && c.s <= MAX && (!first || c.s < first.s)) first = { ...c, box: b }; });

  function drawMoving(s, fill, stroke, lw) {
    const q = moving(s).map((v) => p(...v));
    g.beginPath(); q.forEach((v, i) => (i ? g.lineTo(v[0], v[1]) : g.moveTo(v[0], v[1]))); g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    g.strokeStyle = stroke; g.lineWidth = lw; g.stroke();
    // A tick along the box's local +x shows that its orientation never changes.
    const c = p(start.x + s, start.y), e = p(start.x + s + 0.45 * ca, start.y + 0.45 * sa);
    line(g, c[0], c[1], e[0], e[1], { color: stroke, width: lw, cap: "round" });
  }

  return {
    // The frame shown under reduced motion: the box has stopped at its contact.
    holdAt: 6,
    frame(tt) {
      const t = tt % LOOP;
      const fade = 1 - seg(t, LOOP - 0.6, LOOP);
      const u = seg(t, 0.6, 3.2), s = first.s * u, done = u >= 1;

      text(g, "sweep_box_closest", 24, 24, { size: 13, weight: 700, mono: true });
      text(g, "the box moves along unit_direction; rotation_xyzw stays fixed", 24, 46, { size: 11.5, color: P.ink3 });

      for (let x = 0; x <= 10; x++) { const a = p(x, 0), b = p(x, 4); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }
      for (let y = 0; y <= 4; y++) { const a = p(0, y), b = p(10, y); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }

      const c0 = p(start.x, start.y), cEnd = p(start.x + MAX, start.y);
      line(g, c0[0], c0[1], cEnd[0], cEnd[1], { color: hex(P.amber, 0.45), width: 1.2, dash: [5, 5] });
      line(g, cEnd[0], cEnd[1] - 8, cEnd[0], cEnd[1] + 8, { color: P.ink3, width: 1.4 });
      text(g, "max_distance", cEnd[0], cEnd[1] - 16, { size: 10, mono: true, color: P.ink3, align: "center" });

      boxes.forEach((b) => {
        const lit = done && b === first.box && fade > 0.5;
        const a = p(b.x0, b.y1);
        roundRect(g, a[0], a[1], (b.x1 - b.x0) * S, (b.y1 - b.y0) * S, 3,
          lit ? hex(P.green, 0.35) : P.isDark ? "#3a4654" : "#c4ced8", lit ? P.green : hex(P.ink, 0.3), 1.4);
        const c = p((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2);
        text(g, String(b.id), c[0], c[1], { size: 12, weight: 700, mono: true, align: "center" });
      });

      for (let k = 0; k * 0.8 < s; k++) drawMoving(k * 0.8, null, hex(P.amber, 0.22), 1);
      drawMoving(s, hex(P.amber, 0.16), P.amber, 2);
      const c = p(start.x + s, start.y);
      dot(g, c[0], c[1], 3.5, P.amber);

      const a = seg(t, 3.3, 3.9) * fade;
      if (a > 0) {
        g.globalAlpha = a;
        const q = p(...first.point);
        glow(g, q[0], q[1], 14, P.green, 0.6); dot(g, q[0], q[1], 4, P.green2);
        text(g, "query_position", c[0], c[1] - 50, { size: 11, mono: true, color: P.amber, align: "center" });
        text(g, "object_position", q[0] - 8, q[1] + 34, { size: 11, mono: true, color: P.green2, align: "right" });
        const d0 = p(start.x, 0.7), d1 = p(start.x + first.s, 0.7);
        line(g, d0[0], d0[1] - 6, d0[0], d0[1] + 6, { color: P.ink3, width: 1.2 });
        arrow(g, d0[0], d0[1], d1[0], d1[1], { color: P.ink2, width: 1.6, head: 7 });
        text(g, `distance = ${first.s.toFixed(2)}`, (d0[0] + d1[0]) / 2, d0[1] + 16, { size: 11, mono: true, color: P.ink2, align: "center" });
        chip(g, `hit: object ${first.box.id}; box ${boxes.find((b) => b !== first.box).id} is not reported`, 744, 24, { align: "right", size: 11, color: P.green2, fill: P.panel, stroke: P.green });
        g.globalAlpha = 1;
      }
    },
  };
});
