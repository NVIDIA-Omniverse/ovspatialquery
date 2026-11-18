/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One ray, three hit modes, top view. The same three objects lie on the ray in every lane:
// raycast_closest stops at the nearest, raycast_any reports one hit that need not be the
// nearest, and raycast_multiple stores every hit up to its capacity. Hit distances are computed
// from the drawn boxes and sphere.

OvsqAnim.register("raycast-modes", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex } = ctx;
  const LOOP = 9, S = 54, OX = 52, MAX = 10, CAPACITY = 4;
  const LANES = [
    { mode: "raycast_closest", note: "the nearest hit, or none", y: 102 },
    { mode: "raycast_any", note: "one hit, not necessarily the nearest", y: 212 },
    { mode: "raycast_multiple", note: `every hit, up to capacity = ${CAPACITY}`, y: 322 },
  ];
  // Objects in ray coordinates: the ray runs along +x from (0, 0); y is the offset across it.
  const objects = [
    { id: 10, kind: "box", x0: 2.2, x1: 3.0, y0: -0.5, y1: 0.4 },
    { id: 11, kind: "sphere", cx: 4.9, cy: 0.28, r: 0.55 },
    { id: 12, kind: "box", x0: 7.0, x1: 7.9, y0: -0.35, y1: 0.6 },
  ];
  // Entry distance of the ray y = 0, x >= 0 into each object.
  objects.forEach((o) => {
    o.hit = o.kind === "box" ? (o.y0 <= 0 && o.y1 >= 0 ? o.x0 : null)
      : Math.abs(o.cy) <= o.r ? o.cx - Math.sqrt(o.r * o.r - o.cy * o.cy) : null;
  });
  const hits = objects.filter((o) => o.hit !== null && o.hit <= MAX).sort((a, b) => a.hit - b.hit);
  const closest = hits[0];
  const any = hits[1] || hits[0]; // any admitted hit; here deliberately not the nearest

  function drawObject(o, laneY, lit) {
    const p = (x, y) => [OX + x * S, laneY - y * S];
    const fill = lit ? hex(P.green, 0.35) : P.isDark ? "#3a4654" : "#c4ced8";
    const stroke = lit ? P.green : hex(P.ink, 0.3);
    if (o.kind === "box") {
      const a = p(o.x0, o.y1);
      roundRect(g, a[0], a[1], (o.x1 - o.x0) * S, (o.y1 - o.y0) * S, 3, P.bg); // hide the ray behind
      roundRect(g, a[0], a[1], (o.x1 - o.x0) * S, (o.y1 - o.y0) * S, 3, fill, stroke, 1.4);
      const c = p((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2);
      text(g, String(o.id), c[0], c[1], { size: 12, weight: 700, mono: true, align: "center" });
    } else {
      const c = p(o.cx, o.cy);
      dot(g, c[0], c[1], o.r * S, P.bg);
      dot(g, c[0], c[1], o.r * S, fill, stroke, 1.4);
      text(g, String(o.id), c[0], c[1], { size: 12, weight: 700, mono: true, align: "center" });
    }
  }

  return {
    // The frame shown under reduced motion: every lane shows its result.
    holdAt: 6,
    frame(tt) {
      const t = tt % LOOP;
      const fade = 1 - seg(t, LOOP - 0.6, LOOP);
      const reach = MAX * seg(t, 0.6, 3.4, ctx.ease.linear);

      LANES.forEach((lane, li) => {
        const y = lane.y, o = [OX, y], end = [OX + MAX * S, y];
        text(g, lane.mode, 24, y - 56, { size: 13, weight: 700, mono: true, color: P.ink });
        text(g, lane.note, 24 + 150, y - 56, { size: 11.5, color: P.ink3 });

        // The ray interval, with its max_distance end.
        line(g, o[0], o[1], end[0], end[1], { color: hex(P.blue, 0.18), width: 1.2, dash: [4, 5] });
        line(g, end[0], y - 8, end[0], y + 8, { color: P.ink3, width: 1.4 });
        text(g, "max_distance", end[0], y + 20, { size: 10, mono: true, color: P.ink3, align: "center" });

        const stop = lane.mode === "raycast_closest" ? Math.min(reach, closest.hit) : reach;
        const done = reach >= MAX;
        const shown = (h) => (lane.mode === "raycast_closest" ? h === closest && reach >= h.hit
          : lane.mode === "raycast_any" ? h === any && done : reach >= h.hit);
        // The ray first, so the objects it passes through cover it.
        dot(g, o[0], o[1], 4, P.blue);
        line(g, o[0], o[1], OX + stop * S, y, { color: P.blue, width: 2.4, cap: "round" });
        objects.forEach((ob) => drawObject(ob, y, shown(ob) && fade > 0.5));

        hits.forEach((h) => {
          const q = [OX + h.hit * S, y];
          if (shown(h)) { glow(g, q[0], q[1], 14, P.green, 0.55 * fade); dot(g, q[0], q[1], 4, P.green2); }
          else if (lane.mode === "raycast_any" && reach >= h.hit) dot(g, q[0], q[1], 3.5, null, hex(P.ink3, 0.8), 1.2);
        });

        // Result chip, top right of the lane.
        const a = seg(t, lane.mode === "raycast_closest" ? 1.8 : 3.6, lane.mode === "raycast_closest" ? 2.4 : 4.2) * fade;
        if (a > 0) {
          g.globalAlpha = a;
          let s;
          if (lane.mode === "raycast_closest") s = `hit: object ${closest.id}, distance ${closest.hit.toFixed(2)}`;
          else if (lane.mode === "raycast_any") s = `hit: object ${any.id}, distance ${any.hit.toFixed(2)}`;
          else s = `stored_hit_count = ${Math.min(hits.length, CAPACITY)}, truncated = ${hits.length > CAPACITY}`;
          chip(g, s, 744, y - 56, { align: "right", size: 11, color: P.green2, fill: P.panel, stroke: P.green });
          g.globalAlpha = 1;
        }
      });
      text(g, "top view · each lane casts the same ray from the same origin", 24, 20, { size: 10.5, mono: true, color: P.ink3 });
    },
  };
});
