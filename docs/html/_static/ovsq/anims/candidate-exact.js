/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The same ray against the same authored quad mesh at two precisions. BOUNDS_CANDIDATE stops
// where the ray enters the published AABB; EXACT stops on the stored surface and names the
// authored face and three of its corners. Both hits are computed here, not placed by hand.

OvsqAnim.register("candidate-exact", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, hex, iso } = ctx;
  const LOOP = 14;
  const NX = 4, NZ = 3;

  // Authored mesh: NX x NZ quads over a gently rolling ramp. Face f = row-major quad index;
  // its corners 0..3 run (x0,z0) (x1,z0) (x1,z1) (x0,z1).
  const hgt = (i, k) => 0.15 + 0.32 * i + 0.18 * Math.sin(k * 1.7 + i * 0.6);
  const V = (i, k) => [i, hgt(i, k), k];
  const faces = [];
  for (let k = 0; k < NZ; k++) for (let i = 0; i < NX; i++) faces.push([V(i, k), V(i + 1, k), V(i + 1, k + 1), V(i, k + 1)]);
  const lo = [0, Infinity, 0], hi = [NX, -Infinity, NZ];
  faces.flat().forEach((v) => { lo[1] = Math.min(lo[1], v[1]); hi[1] = Math.max(hi[1], v[1]); });
  lo[1] = 0;

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dotp = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => { const l = Math.hypot(...a); return a.map((x) => x / l); };
  const at = (o, d, s) => [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];

  const O = [-1.0, 2.35, 0.2];
  const D = norm(sub([2.55, hgt(2.55, 1.7), 1.7], O));

  // Möller–Trumbore against both triangles of every authored face.
  let exact = null;
  faces.forEach((f, fi) => {
    for (const tri of [[0, 1, 2], [0, 2, 3]]) {
      const [a, b, c] = tri.map((j) => f[j]);
      const e1 = sub(b, a), e2 = sub(c, a), pv = cross(D, e2), det = dotp(e1, pv);
      if (Math.abs(det) < 1e-9) continue;
      const tv = sub(O, a), u = dotp(tv, pv) / det; if (u < 0 || u > 1) continue;
      const qv = cross(tv, e1), v = dotp(D, qv) / det; if (v < 0 || u + v > 1) continue;
      const s = dotp(e2, qv) / det;
      if (s > 0 && (!exact || s < exact.s)) exact = { s, face: fi, corners: tri, bary: [1 - u - v, u, v] };
    }
  });
  // Slab entry into the published AABB.
  let entry = 0, enterAxis = 0;
  for (let ax = 0; ax < 3; ax++) {
    let a = (lo[ax] - O[ax]) / D[ax], b = (hi[ax] - O[ax]) / D[ax];
    if (a > b) [a, b] = [b, a];
    if (a > entry) { entry = a; enterAxis = ax; }
  }

  function panel(cx, mode, t) {
    const v = iso(38, cx - 22, 172, Math.PI / 6.5);
    const p = v.p;
    const travel = seg(t, 1.0, 3.6, ctx.ease.inOut);
    const stopAt = mode === "bounds" ? entry : exact.s;
    const s = stopAt * travel, done = travel >= 1;
    const accent = mode === "bounds" ? P.amber : P.green;

    v.floor(g, -0.5, -0.5, NX + 0.5, NZ + 0.5, null, null);
    v.grid(g, -0.5, -0.5, NX + 0.5, NZ + 0.5, 0.5);
    // AABB, back edges first.
    const box = (x, y, z) => p(x, y, z);
    const edges = [];
    for (const [a, b] of [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]], [[0, 0, 1], [0, 1, 1]], [[1, 0, 0], [1, 0, 1]], [[0, 0, 1], [1, 0, 1]], [[1, 0, 1], [1, 1, 1]], [[0, 1, 0], [1, 1, 0]], [[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]) {
      const w = (q) => [lo[0] + q[0] * (hi[0] - lo[0]), lo[1] + q[1] * (hi[1] - lo[1]), lo[2] + q[2] * (hi[2] - lo[2])];
      edges.push({ back: (a[1] === 0 && b[1] === 0) || (a[0] === 0 && a[2] === 0 && b[0] === 0 && b[2] === 0), seg: [box(...w(a)), box(...w(b))] });
    }
    const boxColor = mode === "bounds" ? hex(P.amber, done ? 0.95 : 0.6) : hex(P.ink3, 0.55);
    if (mode === "bounds" && done) {
      const top = [p(lo[0], hi[1], lo[2]), p(hi[0], hi[1], lo[2]), p(hi[0], hi[1], hi[2]), p(lo[0], hi[1], hi[2])];
      v.poly(g, top, hex(P.amber, 0.06));
    }
    const drawEdges = (back) => {
      for (const ed of edges) if (ed.back === back) line(g, ed.seg[0][0], ed.seg[0][1], ed.seg[1][0], ed.seg[1][1], { color: boxColor, width: mode === "bounds" ? 1.6 : 1, dash: mode === "bounds" ? null : [4, 4] });
    };
    drawEdges(true);
    // Mesh faces, far to near.
    const order = faces.map((f, i) => i).sort((a, b) => (faces[a][0][0] + faces[a][0][2]) - (faces[b][0][0] + faces[b][0][2]));
    for (const fi of order) {
      const f = faces[fi], q = f.map((c) => p(...c));
      const isHit = mode === "exact" && done && fi === exact.face;
      v.poly(g, q, isHit ? hex(P.green, 0.32) : P.isDark ? "#2a3747" : "#c9d5e1", P.isDark ? "#5d7086" : "#7f93a8", 1.2);
      // Internal triangulation: not authored, so dashed and faint.
      line(g, q[0][0], q[0][1], q[2][0], q[2][1], { color: hex(P.ink3, 0.35), width: 1, dash: [3, 4] });
    }
    drawEdges(false);
    // Below the AABB's lowest corner, clear of the ray.
    const lab = p(hi[0], lo[1], hi[2]);
    text(g, "published AABB", lab[0], lab[1] + 11, { size: 10.5, mono: true, color: boxColor, align: "center" });

    // Ray.
    const o = p(...O), e = p(...at(O, D, s));
    dot(g, o[0], o[1], 3.5, P.blue);
    text(g, "ray", o[0] - 10, o[1], { align: "right", size: 11, mono: true, color: P.blue });
    line(g, o[0], o[1], e[0], e[1], { color: P.blue, width: 2.2, cap: "round" });
    if (mode === "exact" && travel > 0) {
      const b = p(...at(O, D, entry));
      if (s > entry) dot(g, b[0], b[1], 3.2, null, hex(P.amber, 0.7), 1.4);
    }
    if (done) {
      glow(g, e[0], e[1], 18, accent, 0.55);
      dot(g, e[0], e[1], 4.5, accent);
      if (mode === "exact") {
        const f = faces[exact.face];
        exact.corners.forEach((c, j) => {
          const q = p(...f[c]);
          const a = seg(t, 4.0 + j * 0.25, 4.4 + j * 0.25);
          g.globalAlpha = a;
          dot(g, q[0], q[1], 4, P.green2);
          const cen = p(...[0, 1, 2].map((ax) => exact.corners.reduce((acc, cc) => acc + f[cc][ax], 0) / 3));
          const dx = q[0] - cen[0], dy = q[1] - cen[1], L = Math.hypot(dx, dy) || 1;
          chip(g, String(c), q[0] + (dx / L) * 16, q[1] + (dy / L) * 14, { align: "center", size: 10, color: P.onAccent, fill: P.green2, stroke: P.green2 });
          g.globalAlpha = 1;
        });
      }
    }
    return done;
  }

  function record(x, y, w, lines, accent, a) {
    g.save(); g.globalAlpha = a;
    roundRect(g, x, y, w, 22 + lines.length * 19, 10, hex(accent, 0.07), hex(accent, 0.55), 1.2);
    lines.forEach(([s, c], i) => text(g, s, x + 14, y + 20 + i * 19, { size: 11.5, mono: true, color: c || P.ink }));
    g.restore();
  }

  const f3 = (v) => `(${v.map((x) => x.toFixed(2)).join(", ")})`;

  return {
    // The frame shown under reduced motion: the result is on screen.
    holdAt: 7,
    frame(tt) {
      const t = tt % LOOP;
      text(g, "precision = BOUNDS_CANDIDATE", 24, 30, { size: 14, weight: 700, color: P.amber });
      text(g, "stop where the ray enters the AABB", 24, 50, { size: 11.5, mono: true, color: P.ink3 });
      text(g, "precision = EXACT", 400, 30, { size: 14, weight: 700, color: P.green2 });
      text(g, "stop on the stored surface", 400, 50, { size: 11.5, mono: true, color: P.ink3 });
      line(g, 380, 22, 380, 300, { color: P.line, width: 1 });

      panel(190, "bounds", t);
      panel(566, "exact", t);

      const a = seg(t, 4.0, 4.8) * (1 - seg(t, LOOP - 0.8, LOOP));
      const bp = at(O, D, entry), ep = at(O, D, exact.s);
      record(20, 318, 344, [
        ["truth = HIT_BOUNDS_CANDIDATE", P.amber],
        [`bounds_position = ${f3(bp)}`],
        [`distance = ${entry.toFixed(3)}`],
        ["no face, no corners, no barycentrics", P.ink3],
      ], P.amber, a);
      const b = seg(t, 4.6, 5.4) * (1 - seg(t, LOOP - 0.8, LOOP));
      record(396, 318, 344, [
        ["truth = HIT_EXACT_SURFACE", P.green2],
        [`position = ${f3(ep)}`],
        [`distance = ${exact.s.toFixed(3)}`],
        [`feature.data.triangle.source_face_index = ${exact.face}`],
        [`  source_face_corner_indices = { ${exact.corners.join(", ")} }`],
        [`  barycentrics = ${f3(exact.bary)}`],
      ], P.green, b);
      const c = seg(t, 6.0, 6.8) * (1 - seg(t, LOOP - 0.8, LOOP));
      g.globalAlpha = c;
      text(g, "the authored quad is the face; its dashed diagonal is internal and never reported", 24, 494, { size: 11, mono: true, color: P.ink3 });
      g.globalAlpha = 1;
    },
  };
});
