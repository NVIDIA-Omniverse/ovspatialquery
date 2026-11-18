/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A procedural warehouse aisle queried three ways against one published version:
// a raycast fan, a sphere sweep and an overlap box. Hits are computed, not drawn by hand.

OvsqAnim.register("query-families", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, glow, seg, clamp, lerp, iso, hex } = ctx;
  const LOOP = 19;
  const view = iso(31, 236, 140, Math.PI / 7);
  const p = view.p;

  // Objects in the population: [x, y, z, w, h, d, kind]. y is up; the aisle runs along x.
  const objects = [];
  for (const x0 of [0, 4, 8, 12]) objects.push({ x: x0, y: 0, z: 0, w: 3.6, h: 2.5, d: 1.1, kind: "rack" });
  for (const [x0, w, h] of [[0.2, 1.2, 1.15], [1.8, 1.2, 0.8], [4.6, 1.2, 1.3], [7.4, 2.4, 1.0], [12.6, 1.2, 1.25], [14.2, 1.2, 0.85]]) {
    objects.push({ x: x0, y: 0, z: 3.95, w, h, d: 1.0, kind: "stack" });
  }
  objects.push({ x: 10.3, y: 0, z: 2.8, w: 1.2, h: 0.95, d: 1.0, kind: "stack" });
  objects.forEach((o, i) => { o.id = i; });
  const cargo = [];
  {
    const r = ctx.rng(7);
    for (const o of objects.filter((q) => q.kind === "rack")) {
      for (const sy of [0.12, 0.95, 1.78]) {
        for (let cx = o.x + 0.15; cx < o.x + o.w - 0.6; cx += 0.85 + r() * 0.25) {
          if (r() < 0.22) continue;
          const h = 0.35 + r() * 0.35;
          cargo.push({ x: cx, y: sy + 0.08, z: o.z + 0.15, w: 0.6 + r() * 0.15, h, d: o.d - 0.3, tint: r() });
        }
      }
    }
  }

  // Ray against an AABB in the horizontal plane (all rays are level).
  function rayBox(ox, oz, dx, dz, b) {
    let t0 = 0, t1 = Infinity;
    for (const [o, d, lo, hi] of [[ox, dx, b.x, b.x + b.w], [oz, dz, b.z, b.z + b.d]]) {
      if (Math.abs(d) < 1e-9) { if (o < lo || o > hi) return null; continue; }
      let a = (lo - o) / d, c = (hi - o) / d;
      if (a > c) [a, c] = [c, a];
      t0 = Math.max(t0, a); t1 = Math.min(t1, c);
      if (t0 > t1) return null;
    }
    return t0;
  }
  function distToBox(x, z, b) {
    const dx = Math.max(b.x - x, 0, x - (b.x + b.w)), dz = Math.max(b.z - z, 0, z - (b.z + b.d));
    return Math.hypot(dx, dz);
  }

  const AISLE_Z = 2.45, SENSOR_Y = 0.75;
  const robotX = (t) => lerp(0.6, 5.2, seg(t, 0.2, 5.4));

  function drawRobot(x, z, glowOn) {
    view.box(g, x - 0.45, 0, z - 0.35, 0.9, 0.45, 0.7, P.isDark ? "#3a4654" : "#9aa7b4");
    view.box(g, x - 0.18, 0.45, z - 0.15, 0.36, 0.3, 0.3, "#76b900");
    const s = p(x, SENSOR_Y, z);
    if (glowOn) glow(g, s[0], s[1], 16, P.green, 0.5);
  }

  function drawScene(highlight, robot) {
    view.floor(g, -0.4, -0.4, 16, 5.4, hex(P.ink, 0.025), P.line);
    view.grid(g, 0, 0, 16, 5, 1);
    // Aisle floor marking.
    for (const zz of [1.35, 3.75]) {
      const a = p(0, 0, zz), b = p(16, 0, zz);
      line(g, a[0], a[1], b[0], b[1], { color: hex(P.amber, 0.55), width: 2, dash: [10, 8] });
    }
    // Painter's order: far (small x + z) first.
    const items = [];
    for (const o of objects) items.push({ k: o.x + o.w / 2 + o.z + o.d / 2, draw: () => drawObject(o, highlight[o.id]) });
    if (robot) items.push({ k: robot.x + AISLE_Z, draw: () => drawRobot(robot.x, AISLE_Z, robot.glow) });
    items.sort((a, b) => a.k - b.k).forEach((it) => it.draw());
  }
  function drawObject(o, hi) {
    if (o.kind === "rack") {
      const post = P.ink3;
      for (const [px, pz] of [[o.x, o.z + o.d], [o.x + o.w - 0.08, o.z + o.d]]) view.box(g, px, 0, pz - 0.08, 0.08, o.h, 0.08, post, { stroke: false });
      for (const sy of [0.08, 0.91, 1.74, 2.42]) view.box(g, o.x, sy, o.z, o.w, 0.06, o.d, P.isDark ? "#2f4a63" : "#9db9d3", { stroke: false });
      for (const c of cargo) {
        if (c.x < o.x || c.x > o.x + o.w || c.z < o.z || c.z > o.z + o.d) continue;
        view.box(g, c.x, c.y, c.z, c.w, c.h, c.d, c.tint < 0.5 ? "#b08a5a" : c.tint < 0.8 ? "#8c6c46" : "#5f7f99");
      }
      for (const [px, pz] of [[o.x, o.z], [o.x + o.w - 0.08, o.z]]) view.box(g, px, 0, pz, 0.08, o.h, 0.08, post, { stroke: false });
    } else {
      view.box(g, o.x, 0, o.z, o.w, 0.14, o.d, "#8c6c46");
      const n = Math.max(1, Math.round(o.w / 1.2)), bw = (o.w - 0.1) / n;
      for (let k = 0; k < n; k++) {
        const top = (o.h - 0.14) / 2;
        view.box(g, o.x + 0.05 + k * bw, 0.14, o.z + 0.05, bw - 0.04, top, o.d - 0.1, "#b08a5a");
        view.box(g, o.x + 0.05 + k * bw, 0.14 + top, o.z + 0.05, bw - 0.04, top, o.d - 0.1, k % 2 ? "#5f7f99" : "#a3825a");
      }
    }
    if (hi) view.box(g, o.x, o.y, o.z, o.w, o.h, o.d, hi, { alpha: 0.12, outline: hi, outlineWidth: 2, stroke: false });
  }

  function header(phase, t) {
    const names = [
      ["raycast", "raycast_multiple_sync"],
      ["sphere sweep", "sweep_sphere_closest_sync"],
      ["overlap", "overlap_multiple_sync"],
    ];
    let x = 24;
    names.forEach(([n], i) => {
      const on = i === phase;
      const w = chip(g, n, x, 26, { size: 12, mono: false, color: on ? P.onAccent : P.ink2, fill: on ? P.green : P.panel, stroke: on ? P.green : P.line });
      x += w + 8;
    });
    text(g, "ovspatialquery_" + names[phase][1] + "()", 24, 56, { size: 12.5, mono: true, color: P.ink2 });
    chip(g, "population scene · HEAD = 106", 736, 26, { align: "right", size: 11.5, color: P.ink2 });
  }

  return {
    // The frame shown under reduced motion: the result is on screen.
    holdAt: 5,
    frame(tt) {
      const t = tt % LOOP;
      const highlight = {};
      let phase = t < 7 ? 0 : t < 13 ? 1 : 2;
      const rx = phase === 0 ? robotX(t) : 5.2;

      // ---- Raycast fan from the sensor ----
      const rays = [];
      if (phase === 0) {
        const N = 44, reveal = seg(t, 0.8, 4.2, ctx.ease.linear) * N;
        let closest = null;
        for (let i = 0; i < Math.floor(reveal); i++) {
          const a = (i / N) * Math.PI * 2 - Math.PI / 2;
          const dx = Math.cos(a), dz = Math.sin(a);
          let best = 6, hitObj = null;
          for (const o of objects) { const d = rayBox(rx, AISLE_Z, dx, dz, o); if (d !== null && d < best) { best = d; hitObj = o; } }
          rays.push({ dx, dz, d: best, o: hitObj });
          if (hitObj && (!closest || best < closest.d)) closest = { d: best, dx, dz, o: hitObj };
        }
        drawScene(highlight, { x: rx, glow: true });
        const s = p(rx, SENSOR_Y, AISLE_Z);
        for (const r of rays) {
          const e = p(rx + r.dx * r.d, SENSOR_Y, AISLE_Z + r.dz * r.d);
          line(g, s[0], s[1], e[0], e[1], { color: hex(P.blue, r.o ? 0.55 : 0.18), width: 1.4 });
          if (r.o) { glow(g, e[0], e[1], 9, P.blue, 0.5); dot(g, e[0], e[1], 2.6, P.blue); }
        }
        if (closest && t > 4.4) {
          const e = p(rx + closest.dx * closest.d, SENSOR_Y, AISLE_Z + closest.dz * closest.d);
          const a = seg(t, 4.4, 5.0);
          g.globalAlpha = a;
          dot(g, e[0], e[1], 5.5, null, P.green2, 2);
          chip(g, `closest · ${closest.d.toFixed(2)} m`, e[0] + 12, e[1] - 18, { size: 11, color: P.green2, fill: P.panel, stroke: P.green });
          g.globalAlpha = 1;
        }
        const hits = rays.filter((r) => r.o).length;
        text(g, `${rays.length} rays · ${hits} hits · ${rays.length - hits} misses`, 24, 432, { size: 12, mono: true, color: P.ink3 });
      }

      // ---- Sphere sweep along the aisle ----
      if (phase === 1) {
        const R = 0.48, start = rx + 0.4, maxD = 9;
        let contact = maxD, hitObj = null;
        for (let d = 0; d <= maxD; d += 0.005) {
          for (const o of objects) { if (distToBox(start + d, AISLE_Z, o) <= R) { contact = d; hitObj = o; break; } }
          if (hitObj) break;
        }
        const u = seg(t, 7.5, 11, ctx.ease.inOut);
        const cur = contact * u;
        const done = u >= 1;
        if (done && hitObj) highlight[hitObj.id] = hex(P.amber, 1);
        drawScene(highlight, { x: rx, glow: false });
        // Swept volume outline.
        const a0 = p(start, SENSOR_Y, AISLE_Z - R), a1 = p(start + cur, SENSOR_Y, AISLE_Z - R);
        const b0 = p(start, SENSOR_Y, AISLE_Z + R), b1 = p(start + cur, SENSOR_Y, AISLE_Z + R);
        g.beginPath(); g.moveTo(a0[0], a0[1]); g.lineTo(a1[0], a1[1]); g.lineTo(b1[0], b1[1]); g.lineTo(b0[0], b0[1]); g.closePath();
        g.fillStyle = hex(P.amber, 0.08); g.fill();
        line(g, a0[0], a0[1], a1[0], a1[1], { color: hex(P.amber, 0.5), dash: [5, 5] });
        line(g, b0[0], b0[1], b1[0], b1[1], { color: hex(P.amber, 0.5), dash: [5, 5] });
        for (let k = 0; k <= 4; k++) {
          const c = p(start + (cur * k) / 4, SENSOR_Y, AISLE_Z);
          dot(g, c[0], c[1], R * 31, k === 4 ? null : hex(P.amber, 0.05), hex(P.amber, k === 4 ? 0.9 : 0.25), k === 4 ? 2 : 1);
        }
        const c = p(start + cur, SENSOR_Y, AISLE_Z);
        const grd = g.createRadialGradient(c[0] - 5, c[1] - 6, 2, c[0], c[1], R * 31);
        grd.addColorStop(0, hex(P.amber, 0.85)); grd.addColorStop(1, hex(P.amber, 0.25));
        dot(g, c[0], c[1], R * 31, grd);
        if (done && hitObj) {
          glow(g, c[0] + 14, c[1] + 2, 26, P.amber, 0.6);
          chip(g, `contact · distance ${contact.toFixed(2)} m`, c[0], c[1] - 30, { align: "center", size: 11, color: P.amber, fill: P.panel, stroke: P.amber });
        }
        text(g, `radius ${R} m · max_distance ${maxD} m · the first contact stops the sweep`, 24, 432, { size: 12, mono: true, color: P.ink3 });
      }

      // ---- Overlap box around the robot ----
      if (phase === 2) {
        const grow = seg(t, 13.3, 14.6, ctx.ease.back);
        const bx = { x: rx - 1.6, z: AISLE_Z - 1.7, w: 3.9 * grow, d: 3.4, y: 0, h: 1.6 };
        bx.x = rx - 1.95 * grow; bx.w = 3.9 * grow; bx.z = AISLE_Z - 1.7 * grow; bx.d = 3.4 * grow;
        const inside = objects.filter((o) => grow > 0.98 && o.x < bx.x + bx.w && o.x + o.w > bx.x && o.z < bx.z + bx.d && o.z + o.d > bx.z);
        const reveal = Math.floor(seg(t, 14.8, 16.2, ctx.ease.linear) * inside.length + 0.001);
        inside.slice(0, reveal).forEach((o) => { highlight[o.id] = hex(P.violet, 1); });
        drawScene(highlight, { x: rx, glow: false });
        if (grow > 0.01) {
          view.box(g, bx.x, 0, bx.z, bx.w, bx.h, bx.d, P.violet, { alpha: 0.07, outline: hex(P.violet, 0.9), outlineWidth: 1.6, dash: [6, 4], stroke: false });
        }
        if (reveal > 0) {
          const c = p(bx.x + bx.w, bx.h, bx.z);
          chip(g, `${reveal} object${reveal > 1 ? "s" : ""} overlap`, c[0] + 8, c[1] - 12, { size: 11, color: P.violet, fill: P.panel, stroke: P.violet });
        }
        text(g, "overlap reports every object whose shape meets the query box · no distance, no order", 24, 432, { size: 12, mono: true, color: P.ink3 });
      }

      header(phase, t);
      // Phase progress.
      const prog = phase === 0 ? t / 7 : phase === 1 ? (t - 7) / 6 : (t - 13) / 6;
      roundRect(g, 24, 446, 712, 3, 1.5, hex(P.ink, 0.06));
      roundRect(g, 24 + phase * 237.3, 446, 237.3 * clamp(prog, 0, 1), 3, 1.5, P.green);
    },
  };
});
