/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The ovstage first-query program: the filter selects the two prims under /World, ordinal 2 seals
// their world transforms, publishing it makes version 2, and a ray along +x hits /World/Crate.
// The hit distance is computed from the scene here and matches the program's output.

OvsqAnim.register("ovstage-scene", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex } = ctx;
  const LOOP = 11, S = 52, OX = 418, OY = 300, PANEL = 364;
  const p = (x, y) => [OX + x * S, OY - y * S];
  const crate = { x: 4, y: 0, size: 1 };
  const ball = { x: 4, y: 3, r: 0.5 };
  const rayHit = crate.x - crate.size / 2; // a ray along +x from the origin meets the crate's face
  const prims = [
    { path: "/World", kind: "Xform", detail: "", selected: false, indent: 0 },
    { path: "/World/Crate", kind: "Cube", detail: "size 1 at (4, 0, 0)", selected: true, indent: 1 },
    { path: "/World/Ball", kind: "Sphere", detail: "radius 0.5 at (4, 3, 0)", selected: true, indent: 1 },
  ];
  const steps = [
    { at: 2.6, label: "ordinal 1", what: "open_usd_from_string: authored prims" },
    { at: 3.5, label: "ordinal 2", what: "compute_hierarchy: world transforms" },
    { at: 4.4, label: "enqueue_set(2)", what: "population version 2" },
  ];

  return {
    // The frame shown under reduced motion: the scene is published and the ray has hit.
    holdAt: 9,
    frame(tt) {
      const t = tt % LOOP;
      const ink = P.isDark ? "#3a4654" : "#c4ced8";

      // Left: the stage and the filter.
      text(g, "stage", 24, 28, { size: 12, weight: 700, color: P.ink3 });
      const filterOn = seg(t, 0.6, 1.4);
      prims.forEach((prim, i) => {
        const y = 62 + i * 40, x = 24 + prim.indent * 22;
        const lit = prim.selected && filterOn > 0;
        roundRect(g, x, y - 15, PANEL - x, 30, 6, lit ? hex(P.green, 0.14 * filterOn) : P.panel, lit ? hex(P.green, filterOn) : P.line, 1.2);
        text(g, prim.path, x + 10, y, { size: 12, mono: true, weight: 600, color: !prim.selected && filterOn > 0 ? P.ink3 : P.ink });
        text(g, prim.kind, 150, y, { size: 11, mono: true, color: P.ink2 });
        text(g, prim.detail || (filterOn > 0 ? "not selected" : ""), 206, y, { size: 10.5, color: P.ink3 });
      });
      if (filterOn > 0) {
        g.globalAlpha = filterOn;
        chip(g, 'filter: usd-path PREFIX "/World/"', 24, 186, { size: 11, color: P.green2, fill: P.panel, stroke: P.green });
        g.globalAlpha = 1;
      }

      // Left, below: the ordinals that lead to the published version.
      steps.forEach((step, i) => {
        const y = 232 + i * 36, on = t >= step.at;
        roundRect(g, 24, y - 14, PANEL - 24, 28, 6, on ? hex(P.blue, 0.14) : P.panel, on ? P.blue : P.line, 1.2);
        text(g, step.label, 34, y, { size: 11.5, mono: true, weight: 700, color: on ? P.blue : P.ink3 });
        text(g, step.what, 140, y, { size: 11, color: on ? P.ink : P.ink3 });
      });

      // Right: top view of what version 2 holds.
      for (let x = 0; x <= 6; x++) { const a = p(x, -0.5), b = p(x, 3.9); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }
      for (let y = -0.5; y <= 3.9; y += 0.5) { const a = p(-0.3, y), b = p(6.3, y); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }
      const o = p(0, 0), xa = p(6.4, 0), ya = p(0, 3.9);
      line(g, o[0], o[1], xa[0], xa[1], { color: P.line, width: 1.5 });
      line(g, o[0], o[1], ya[0], ya[1], { color: P.line, width: 1.5 });
      for (let x = 0; x <= 6; x += 2) text(g, String(x), p(x, 0)[0], p(x, 0)[1] + 16, { size: 10.5, mono: true, color: P.ink3, align: "center" });
      text(g, "x", xa[0] + 8, xa[1], { size: 11, mono: true, color: P.ink3 });
      text(g, "y", ya[0], ya[1] - 10, { size: 11, mono: true, color: P.ink3, align: "center" });

      const shown = seg(t, 4.4, 5.0);
      if (shown > 0) {
        g.globalAlpha = shown;
        const u = seg(t, 5.6, 7.0), hit = u >= 1;
        const a = p(crate.x - crate.size / 2, crate.y + crate.size / 2);
        roundRect(g, a[0], a[1], S, S, 4, hit ? hex(P.green, 0.35) : ink, hit ? P.green : hex(P.ink, 0.3), 1.4);
        text(g, "Crate", a[0] + S / 2, a[1] + S / 2, { size: 11.5, weight: 700, align: "center" });
        const bc = p(ball.x, ball.y);
        dot(g, bc[0], bc[1], ball.r * S, ink, hex(P.ink, 0.3), 1.4);
        text(g, "Ball", bc[0], bc[1], { size: 11.5, weight: 700, align: "center" });

        if (t >= 5.6) {
          const e = p(rayHit * u, 0);
          dot(g, o[0], o[1], 4, P.blue);
          line(g, o[0], o[1], e[0], e[1], { color: P.blue, width: 2.4, cap: "round" });
          if (hit) {
            glow(g, e[0], e[1], 16, P.green, 0.6); dot(g, e[0], e[1], 4, P.green2);
            chip(g, `ray hit /World/Crate at ${rayHit.toFixed(1)}, ordinal 2`, p(3, 0)[0], e[1] - 64, { align: "center", size: 11, color: P.green2, fill: P.panel, stroke: P.green });
          }
        }
        g.globalAlpha = 1;
      } else {
        text(g, "nothing published yet", p(3, 1.7)[0], p(3, 1.7)[1], { size: 12, color: P.ink3, align: "center" });
      }
      text(g, "population version 2 · top view, 1 grid square = 1 unit", 24, 356, { size: 10.5, mono: true, color: P.ink3 });
    },
  };
});
