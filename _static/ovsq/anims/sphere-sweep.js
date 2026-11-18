/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// sweep_sphere_closest, top view: a sphere moves along +x and stops at its first contact, the
// corner of box 20. Box 21 lies further along the path and is not reported. The contact distance,
// the sphere's center there (query_position) and the contact point on the box are computed from
// the drawn geometry.

OvsqAnim.register("sphere-sweep", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex, arrow } = ctx;
  const LOOP = 9, S = 56, OX = 44, OY = 292, MAX = 9;
  const p = (x, y) => [OX + x * S, OY - y * S];
  const start = { x: 1.0, y: 2.0 }, R = 0.7;
  const boxes = [
    { id: 20, x0: 5.0, x1: 6.2, y0: 0.4, y1: 1.6 },
    { id: 21, x0: 7.6, x1: 8.6, y0: 1.5, y1: 3.0 },
  ];

  // Distance a circle of radius R moving along +x from `start` travels before it touches box b,
  // and the contact point on b. Null when the path misses.
  function contact(b) {
    const cy = start.y;
    if (cy >= b.y0 && cy <= b.y1) return { s: b.x0 - R - start.x, point: [b.x0, cy] };
    const ey = cy > b.y1 ? b.y1 : b.y0, dy = cy - ey;
    if (Math.abs(dy) > R) return null;
    return { s: b.x0 - Math.sqrt(R * R - dy * dy) - start.x, point: [b.x0, ey] };
  }
  let first = null;
  boxes.forEach((b) => { const c = contact(b); if (c && c.s >= 0 && c.s <= MAX && (!first || c.s < first.s)) first = { ...c, box: b }; });
  const center = [start.x + first.s, start.y];

  return {
    // The frame shown under reduced motion: the sphere has stopped at its contact.
    holdAt: 6,
    frame(tt) {
      const t = tt % LOOP;
      const fade = 1 - seg(t, LOOP - 0.6, LOOP);
      const u = seg(t, 0.6, 3.2), s = first.s * u, done = u >= 1;

      text(g, "sweep_sphere_closest", 24, 24, { size: 13, weight: 700, mono: true });
      text(g, "the sphere moves along unit_direction and stops at its first contact", 24, 46, { size: 11.5, color: P.ink3 });

      for (let x = 0; x <= 10; x++) { const a = p(x, 0), b = p(x, 4); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }
      for (let y = 0; y <= 4; y++) { const a = p(0, y), b = p(10, y); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }

      // The swept band up to max_distance, and the path of the center.
      const band0 = p(start.x, start.y + R), band1 = p(start.x + MAX, start.y - R);
      g.fillStyle = hex(P.amber, 0.06); g.fillRect(band0[0], band0[1], band1[0] - band0[0], band1[1] - band0[1]);
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

      // Ghosts of the sphere along its path, then the sphere itself.
      for (let k = 0; k * 0.8 < s; k++) {
        const q = p(start.x + k * 0.8, start.y);
        dot(g, q[0], q[1], R * S, null, hex(P.amber, 0.22), 1);
      }
      const c = p(start.x + s, start.y);
      dot(g, c[0], c[1], R * S, hex(P.amber, 0.16), P.amber, 2);
      dot(g, c[0], c[1], 3.5, P.amber);

      const a = seg(t, 3.3, 3.9) * fade;
      if (a > 0) {
        g.globalAlpha = a;
        const q = p(...first.point);
        glow(g, q[0], q[1], 14, P.green, 0.6); dot(g, q[0], q[1], 4, P.green2);
        text(g, "query_position", c[0], c[1] - R * S - 12, { size: 11, mono: true, color: P.amber, align: "center" });
        text(g, "object_position", q[0] + 12, q[1] - 13, { size: 11, mono: true, color: P.green2 });
        // distance: how far the center travelled.
        const d0 = p(start.x, 0.7), d1 = p(center[0], 0.7);
        line(g, d0[0], d0[1] - 6, d0[0], d0[1] + 6, { color: P.ink3, width: 1.2 });
        arrow(g, d0[0], d0[1], d1[0], d1[1], { color: P.ink2, width: 1.6, head: 7 });
        text(g, `distance = ${first.s.toFixed(2)}`, (d0[0] + d1[0]) / 2, d0[1] + 16, { size: 11, mono: true, color: P.ink2, align: "center" });
        chip(g, `hit: object ${first.box.id}; box ${boxes.find((b) => b !== first.box).id} is not reported`, 744, 24, { align: "right", size: 11, color: P.green2, fill: P.panel, stroke: P.green });
        g.globalAlpha = 1;
      }
    },
  };
});
