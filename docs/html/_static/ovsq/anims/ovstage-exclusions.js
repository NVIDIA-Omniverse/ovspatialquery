/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The exclusions program: the filter selects four prims, publishing keeps the stretched Cube as a
// box and the sheared Cube as a convex hull, and lists the stretched Sphere and the Cylinder as
// excluded. Two rays along +x then stop on what was published. The shapes and both hit distances
// are computed here from the transforms the program authors and match its output.

OvsqAnim.register("ovstage-exclusions", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex } = ctx;
  const LOOP = 12, S = 36, OX = 400, OY = 292, PANEL = 364;
  const p = (x, y) => [OX + x * S, OY - y * S];

  // Crate: Cube of size 2 scaled (0.5, 1, 1) at (4, 0). Ramp: Cube of size 1 whose world matrix
  // moves x by half of y (a shear) and translates it to (7, 3).
  const crate = { x: 4, y: 0, hx: 0.5, hy: 1 };
  const ramp = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([u, v]) => [7 + u + 0.5 * v, 3 + v]);
  const egg = { x: 4, y: 3, rx: 0.5, ry: 1 };
  const post = { x: 8, y: 0, r: 1 };

  // Where a ray along +x at height y first meets a convex polygon.
  const entryX = (poly, y) => {
    let best = Infinity;
    poly.forEach((a, i) => {
      const b = poly[(i + 1) % poly.length];
      if ((a[1] - y) * (b[1] - y) > 0 || a[1] === b[1]) return;
      best = Math.min(best, a[0] + ((y - a[1]) / (b[1] - a[1])) * (b[0] - a[0]));
    });
    return best;
  };
  const rays = [
    { y: 0, hit: crate.x - crate.hx, name: "Crate", start: 5.2 },
    { y: 3.25, hit: entryX(ramp, 3.25), name: "Ramp", start: 6.8 },
  ];

  const prims = [
    { path: "/World", kind: "Xform", detail: "", outcome: null, indent: 0 },
    { path: "/World/Crate", kind: "Cube", detail: "scale (0.5, 1, 1)", outcome: ["box", true], indent: 1 },
    { path: "/World/Egg", kind: "Sphere", detail: "scale (1, 2, 1)", outcome: ["UNSUPPORTED_SHAPE", false], indent: 1 },
    { path: "/World/Post", kind: "Cylinder", detail: "", outcome: ["UNSUPPORTED_TYPE", false], indent: 1 },
    { path: "/World/Ramp", kind: "Cube", detail: "sheared", outcome: ["convex hull", true], indent: 1 },
  ];

  function poly(pts, fill, stroke, lw, dash) {
    g.save();
    if (dash) g.setLineDash(dash);
    g.beginPath();
    pts.forEach((q, i) => (i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])));
    g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw || 1.4; g.stroke(); }
    g.restore();
  }

  return {
    // The frame shown under reduced motion: both rays have hit.
    holdAt: 10,
    frame(tt) {
      const t = tt % LOOP;
      const ink = P.isDark ? "#3a4654" : "#c4ced8";

      // Left: the stage, the filter, and what publishing did with each selected prim.
      text(g, "stage", 24, 26, { size: 12, weight: 700, color: P.ink3 });
      const filterOn = seg(t, 0.6, 1.4), decided = seg(t, 3.0, 3.8);
      prims.forEach((prim, i) => {
        const y = 54 + i * 36, x = 24 + prim.indent * 18;
        const selected = prim.outcome !== null, lit = selected && filterOn > 0;
        const kept = selected && prim.outcome[1], color = kept ? P.green : P.red;
        const edge = decided > 0 && selected ? hex(color, decided) : lit ? hex(P.blue, filterOn) : P.line;
        const fill = decided > 0 && selected ? hex(color, 0.12 * decided) : lit ? hex(P.blue, 0.1 * filterOn) : P.panel;
        roundRect(g, x, y - 14, PANEL - x, 28, 6, fill, edge, 1.2);
        text(g, prim.path, x + 10, y, { size: 11.5, mono: true, weight: 600, color: !selected && filterOn > 0 ? P.ink3 : P.ink });
        text(g, prim.kind, 150, y, { size: 11, mono: true, color: P.ink2 });
        if (decided > 0 && selected) {
          g.globalAlpha = decided;
          text(g, prim.outcome[0], 220, y, { size: 10.5, mono: true, weight: 700, color: kept ? P.green2 : P.red });
          g.globalAlpha = 1;
        } else {
          text(g, prim.detail || (!selected && filterOn > 0 ? "not selected" : ""), 220, y, { size: 10.5, color: P.ink3 });
        }
      });
      if (filterOn > 0) {
        g.globalAlpha = filterOn;
        chip(g, 'filter: usd-path PREFIX "/World/"', 24, 246, { size: 11, color: P.blue, fill: P.panel, stroke: P.blue });
        g.globalAlpha = 1;
      }
      const published = t >= 2.4;
      roundRect(g, 24, 272, PANEL - 24, 64, 6, published ? hex(P.blue, 0.1) : P.panel, published ? P.blue : P.line, 1.2);
      text(g, "enqueue_set(2)", 34, 290, { size: 11.5, mono: true, weight: 700, color: published ? P.blue : P.ink3 });
      text(g, published ? "succeeds: population version 2" : "", 146, 290, { size: 11, color: P.ink });
      if (decided > 0) {
        g.globalAlpha = decided;
        text(g, "object_count 2", 34, 318, { size: 11, mono: true, color: P.green2 });
        text(g, "excluded_object_count 2", 152, 318, { size: 11, mono: true, color: P.red });
        g.globalAlpha = 1;
      }

      // Right: top view of what version 2 holds, with the excluded prims drawn as outlines.
      for (let x = 0; x <= 9; x++) { const a = p(x, -1.2), b = p(x, 4.2); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }
      for (let y = -1; y <= 4; y++) { const a = p(-0.2, y), b = p(9.2, y); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }
      const o = p(0, 0), xa = p(9.3, 0), ya = p(0, 4.3);
      line(g, o[0], o[1], xa[0], xa[1], { color: P.line, width: 1.5 });
      line(g, o[0], o[1], ya[0], ya[1], { color: P.line, width: 1.5 });
      for (let x = 0; x <= 8; x += 2) text(g, String(x), p(x, 0)[0], p(x, 0)[1] + 14, { size: 10.5, mono: true, color: P.ink3, align: "center" });
      text(g, "x", xa[0] + 8, xa[1], { size: 11, mono: true, color: P.ink3 });
      text(g, "y", ya[0], ya[1] - 10, { size: 11, mono: true, color: P.ink3, align: "center" });

      const shown = seg(t, 4.0, 4.6);
      if (shown > 0) {
        g.globalAlpha = shown;
        // Excluded: outlines only, so a ray passes through them.
        const ec = p(egg.x, egg.y);
        g.save(); g.setLineDash([4, 4]); g.strokeStyle = hex(P.red, 0.8); g.lineWidth = 1.3;
        g.beginPath(); g.ellipse(ec[0], ec[1], egg.rx * S, egg.ry * S, 0, 0, Math.PI * 2); g.stroke();
        const pc = p(post.x, post.y);
        g.beginPath(); g.arc(pc[0], pc[1], post.r * S, 0, Math.PI * 2); g.stroke(); g.restore();
        text(g, "Egg", ec[0] - egg.rx * S - 8, ec[1] + egg.ry * S * 0.6, { size: 11, weight: 700, color: P.red, align: "right" });
        text(g, "Post", pc[0], pc[1] - post.r * S - 10, { size: 11, weight: 700, color: P.red, align: "center" });

        const hitCrate = t >= rays[0].start + 1.4, hitRamp = t >= rays[1].start + 1.8;
        const a = p(crate.x - crate.hx, crate.y + crate.hy);
        roundRect(g, a[0], a[1], 2 * crate.hx * S, 2 * crate.hy * S, 3, hitCrate ? hex(P.green, 0.35) : ink, hitCrate ? P.green : hex(P.ink, 0.3), 1.4);
        text(g, "Crate", p(crate.x, -1.45)[0], p(crate.x, -1.45)[1], { size: 11, weight: 700, align: "center" });
        poly(ramp.map(([x, y]) => p(x, y)), hitRamp ? hex(P.green, 0.35) : ink, hitRamp ? P.green : hex(P.ink, 0.3));
        text(g, "Ramp", p(7.1, 2.15)[0], p(7.1, 2.15)[1], { size: 11, weight: 700, align: "center" });

        rays.forEach((ray, i) => {
          if (t < ray.start) return;
          const u = seg(t, ray.start, ray.start + (i ? 1.8 : 1.4)), s = p(0, ray.y), e = p(ray.hit * u, ray.y);
          dot(g, s[0], s[1], 3.5, P.blue);
          line(g, s[0], s[1], e[0], e[1], { color: P.blue, width: 2.2, cap: "round" });
          if (u >= 1) {
            glow(g, e[0], e[1], 14, P.green, 0.6); dot(g, e[0], e[1], 3.5, P.green2);
            const label = `${ray.name} at ${Number(ray.hit.toFixed(3))}`;
            chip(g, label, i ? e[0] - 20 : e[0] - 12, e[1] - (i ? 28 : 22), { align: i ? "left" : "right", size: 10.5, color: P.green2, fill: P.panel, stroke: P.green });
          }
        });
        g.globalAlpha = 1;
      } else {
        text(g, "nothing published yet", p(4.5, 1.5)[0], p(4.5, 1.5)[1], { size: 12, color: P.ink3, align: "center" });
      }
      text(g, "version 2 · top view, 1 grid square = 1 unit", 400, 384, { size: 10.5, mono: true, color: P.ink3 });
    },
  };
});
