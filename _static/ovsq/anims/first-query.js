/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The explicit first-query program, to scale: two unit boxes and a sphere, then the ray, the
// sphere sweep and the overlap it runs. Distances are computed here and match its output.
// data-only="ray" draws just box 10 and the ray, which is what the C program publishes and casts.

OvsqAnim.register("first-query", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex } = ctx;
  const rayOnly = ctx.root.dataset.only === "ray";
  const LOOP = rayOnly ? 5 : 13, S = 62, OX = 70, OY = 272;
  const p = (x, y) => [OX + x * S, OY - y * S];
  const boxes = rayOnly ? [{ id: 10, x: 4, y: 0 }] : [{ id: 10, x: 4, y: 0 }, { id: 11, x: 8, y: 0 }];
  const ball = rayOnly ? null : { id: 12, x: 4, y: 3, r: 0.5 };
  const rayHit = boxes[0].x - 0.5; // ray along +x from the origin meets the box face
  const sweepHit = ball ? ball.x - (ball.r + 0.5) : 0; // sphere of radius 0.5 on y = 3 touches the ball
  const probe = { x: 8, y: 1, r: 0.75 };
  const probeHits = probe.y - 0.5 <= probe.r; // distance to box 11's top face

  return {
    // The frame shown under reduced motion: the result is on screen.
    holdAt: 3,
    frame(tt) {
      const t = tt % LOOP;
      // Grid and axes.
      for (let x = 0; x <= 10; x++) { const a = p(x, -0.8), b = p(x, 3.8); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }
      for (let y = -0.5; y <= 3.5; y += 0.5) { const a = p(-0.3, y), b = p(10.3, y); line(g, a[0], a[1], b[0], b[1], { color: P.grid, width: 1 }); }
      const o = p(0, 0), xa = p(10.4, 0), ya = p(0, 3.8);
      line(g, o[0], o[1], xa[0], xa[1], { color: P.line, width: 1.5 });
      line(g, o[0], o[1], ya[0], ya[1], { color: P.line, width: 1.5 });
      for (let x = 0; x <= 10; x += 2) text(g, String(x), p(x, 0)[0], p(x, 0)[1] + 16, { size: 10.5, mono: true, color: P.ink3, align: "center" });
      text(g, "x", xa[0] + 8, xa[1], { size: 11, mono: true, color: P.ink3 });
      text(g, "y", ya[0], ya[1] - 10, { size: 11, mono: true, color: P.ink3, align: "center" });

      const phase = rayOnly || t < 4 ? 0 : t < 8 ? 1 : 2;
      // Objects.
      boxes.forEach((b) => {
        const hit = (phase === 0 && b.id === 10 && t > 2.4) || (phase === 2 && b.id === 11 && t > 9.6 && probeHits);
        const a = p(b.x - 0.5, b.y + 0.5);
        roundRect(g, a[0], a[1], S, S, 4, hit ? hex(P.green, 0.35) : P.isDark ? "#3a4654" : "#c4ced8", hit ? P.green : hex(P.ink, 0.3), 1.4);
        text(g, String(b.id), a[0] + S / 2, a[1] + S / 2, { size: 13, weight: 700, mono: true, align: "center" });
      });
      if (ball) {
        const bc = p(ball.x, ball.y), ballHit = phase === 1 && t > 6.4;
        dot(g, bc[0], bc[1], ball.r * S, ballHit ? hex(P.green, 0.35) : P.isDark ? "#3a4654" : "#c4ced8", ballHit ? P.green : hex(P.ink, 0.3), 1.4);
        text(g, "12", bc[0], bc[1], { size: 13, weight: 700, mono: true, align: "center" });
      }

      if (phase === 0) {
        const u = seg(t, 0.6, 2.4), e = p(rayHit * u, 0);
        dot(g, o[0], o[1], 4, P.blue);
        line(g, o[0], o[1], e[0], e[1], { color: P.blue, width: 2.4, cap: "round" });
        if (u >= 1) { glow(g, e[0], e[1], 16, P.green, 0.6); dot(g, e[0], e[1], 4, P.green2); chip(g, `raycast_closest → 10 at ${rayHit.toFixed(1)}`, e[0] - 10, e[1] - 54, { align: "center", size: 11, color: P.green2, fill: P.panel, stroke: P.green }); }
      }
      if (phase === 1) {
        const u = seg(t, 4.6, 6.4), cx = sweepHit * u, c = p(cx, 3), s0 = p(0, 3);
        g.fillStyle = hex(P.amber, 0.08); const top = p(0, 3.5), bot = p(cx, 2.5); g.fillRect(top[0], top[1], bot[0] - top[0], bot[1] - top[1]);
        line(g, s0[0], s0[1], c[0], c[1], { color: hex(P.amber, 0.6), width: 1.4, dash: [5, 5] });
        dot(g, c[0], c[1], 0.5 * S, hex(P.amber, 0.25), P.amber, 1.6);
        if (u >= 1) chip(g, `sweep_sphere_closest → 12 at ${sweepHit.toFixed(1)}`, c[0], c[1] + 52, { align: "center", size: 11, color: P.amber, fill: P.panel, stroke: P.amber });
      }
      if (phase === 2) {
        const u = seg(t, 8.4, 9.6, ctx.ease.back), c = p(probe.x, probe.y);
        dot(g, c[0], c[1], Math.max(0.01, probe.r * S * u), hex(P.violet, 0.16), P.violet, 1.6);
        if (u >= 1) chip(g, "overlap_any → 11", c[0], c[1] - probe.r * S - 18, { align: "center", size: 11, color: P.violet, fill: P.panel, stroke: P.violet });
      }
      text(g, "population ordinal 1 · top view, 1 grid square = 1 unit", 24, 336, { size: 10.5, mono: true, color: P.ink3 });
    },
  };
});
