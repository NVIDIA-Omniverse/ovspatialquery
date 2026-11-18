/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Two populations on their own clocks: static `scene` (ordinal 20) and moving `robots`.
// A pair reporter advances only when asked and compares its accepted baseline with the AABB
// pairs at the new endpoints, so a contact that starts and ends between two advances is never
// reported. Every pair set here is computed from the boxes drawn.

OvsqAnim.register("pair-reporter", (ctx) => {
  const { g, P, text, chip, roundRect, line, dot, seg, hex, clamp, lerp } = ctx;
  const LOOP = 20;
  const scene = [
    { name: "ShelfA", x: 70, y: 70, w: 150, h: 40 },
    { name: "ShelfB", x: 250, y: 70, w: 150, h: 40 },
    { name: "Cone", x: 196, y: 214, w: 22, h: 22 },
    { name: "Dock", x: 340, y: 262, w: 70, h: 46 },
  ];
  // Robot centers along polylines, parameterised by simulation time.
  const paths = {
    RobotA: [[0, 120, 150], [6, 150, 112], [10, 300, 125], [15, 360, 222], [20, 360, 280]],
    RobotB: [[0, 60, 250], [9, 150, 250], [11.5, 207, 250], [13.5, 262, 250], [20, 280, 170]],
  };
  const R = { w: 36, h: 28 };
  const PUBS = [[100, 1.0], [101, 6.0], [104, 9.6], [110, 15.4], [112, 18.4]];

  function at(name, t) {
    const p = paths[name];
    for (let i = 1; i < p.length; i++) {
      if (t <= p[i][0]) { const u = (t - p[i - 1][0]) / (p[i][0] - p[i - 1][0]); return [lerp(p[i - 1][1], p[i][1], u), lerp(p[i - 1][2], p[i][2], u)]; }
    }
    const e = p[p.length - 1]; return [e[1], e[2]];
  }
  const boxOf = (c) => ({ x: c[0] - R.w / 2, y: c[1] - R.h / 2, w: R.w, h: R.h });
  const overlap = (a, b, tol) => a.x <= b.x + b.w + tol && b.x <= a.x + a.w + tol && a.y <= b.y + b.h + tol && b.y <= a.y + a.h + tol;
  const TOL = 4;
  function pairsAt(t) {
    const out = [];
    for (const r of Object.keys(paths)) for (const s of scene) if (overlap(boxOf(at(r, t)), s, TOL)) out.push(`${r} · ${s.name}`);
    return out;
  }
  // Reporter history: per advance, entered / exited relative to the previous baseline.
  const history = [];
  let baseline = [];
  for (const [ord, t] of PUBS) {
    const now = pairsAt(t);
    history.push({ ord, t, entered: now.filter((p) => !baseline.includes(p)), exited: baseline.filter((p) => !now.includes(p)), active: now.length });
    baseline = now;
  }
  // Contacts that live entirely between two advances.
  const transient = [];
  for (let s = 0; s < LOOP; s += 0.05) {
    for (const p of pairsAt(s)) {
      const prev = PUBS.filter(([, t]) => t <= s).pop(), next = PUBS.find(([, t]) => t > s);
      const seenBefore = prev && pairsAt(prev[1]).includes(p), seenAfter = next && pairsAt(next[1]).includes(p);
      if (!seenBefore && !seenAfter && !transient.find((q) => q.p === p)) transient.push({ p, t: s });
    }
  }

  return {
    // The frame shown under reduced motion: the result is on screen.
    holdAt: 11,
    frame(tt) {
      const t = tt % LOOP;
      const pub = PUBS.filter(([, at0]) => t >= at0).pop();

      // Floor.
      roundRect(g, 24, 40, 420, 300, 12, hex(P.ink, 0.025), P.line, 1);
      text(g, "scene · ordinal 20", 36, 56, { size: 10.5, mono: true, color: P.ink3 });
      text(g, `robots · ordinal ${pub ? pub[0] : "—"}`, 432, 56, { size: 10.5, mono: true, color: P.blue, align: "right" });
      for (const s of scene) {
        roundRect(g, s.x, s.y, s.w, s.h, 4, P.isDark ? "#3a4654" : "#b9c4cf", hex(P.ink, 0.25), 1);
        text(g, s.name, s.x + s.w / 2, s.y + s.h / 2, { size: 10.5, mono: true, align: "center", color: P.ink2 });
      }
      // Live simulation (faint) versus the published pose the reporter sees (solid).
      for (const name of Object.keys(paths)) {
        const live = boxOf(at(name, t));
        const liveTouch = scene.some((q) => overlap(live, q, TOL));
        roundRect(g, live.x, live.y, live.w, live.h, 6, null, liveTouch ? hex(P.violet, 0.8) : hex(P.blue, 0.35), 1.2);
        if (pub) {
          const pb = boxOf(at(name, pub[1]));
          const touching = scene.some((s) => overlap(pb, s, TOL));
          roundRect(g, pb.x, pb.y, pb.w, pb.h, 6, hex(touching ? P.amber : P.blue, 0.3), touching ? P.amber : P.blue, 1.6);
          text(g, name.replace("Robot", ""), pb.x + pb.w / 2, pb.y + pb.h / 2, { size: 11, weight: 700, mono: true, align: "center", color: P.ink });
        }
      }
      // Transient contact marker.
      for (const tr of transient) {
        if (t < tr.t || t > tr.t + 3.5) continue;
        const a = 1 - seg(t, tr.t + 2.5, tr.t + 3.5);
        g.globalAlpha = a;
        chip(g, "touched between advances: not reported", 36, 322, { size: 10, color: P.violet, fill: P.panel, stroke: P.violet });
        g.globalAlpha = 1;
      }
      text(g, "faint: the simulation now · solid: the published version", 24, 360, { size: 10.5, mono: true, color: P.ink3 });

      // Reporter panel.
      const px = 466;
      roundRect(g, px, 40, 270, 300, 12, hex(P.green, 0.04), hex(P.green, 0.4), 1.2);
      text(g, "pair reporter", px + 16, 60, { size: 13, weight: 700, color: P.green2 });
      text(g, "first = robots · second = scene", px + 16, 78, { size: 10.5, mono: true, color: P.ink3 });
      const shown = history.filter((h) => t >= h.t).slice(-3);
      shown.forEach((h, i) => {
        const y = 102 + i * 76;
        const fresh = i === shown.length - 1;
        const a = fresh ? seg(t, h.t, h.t + 0.5) : 0.55;
        g.save(); g.globalAlpha = a;
        text(g, `advance → (${h.ord}, 20)`, px + 16, y, { size: 11.5, mono: true, weight: 700, color: P.ink });
        text(g, `active ${h.active}`, px + 254, y, { size: 10.5, mono: true, color: P.ink3, align: "right" });
        let yy = y + 18;
        if (!h.entered.length && !h.exited.length) { text(g, "no change", px + 24, yy, { size: 11, mono: true, color: P.ink3 }); }
        for (const e of h.entered) { text(g, `+ ${e}`, px + 24, yy, { size: 11, mono: true, color: P.green2 }); yy += 16; }
        for (const e of h.exited) { text(g, `− ${e}`, px + 24, yy, { size: 11, mono: true, color: P.red }); yy += 16; }
        g.restore();
      });
      text(g, "forward only: each advance compares", px + 16, 312, { size: 10, mono: true, color: P.ink3 });
      text(g, "its baseline with the new endpoints", px + 16, 326, { size: 10, mono: true, color: P.ink3 });

      // Timeline of publications.
      const T0 = 24, T1 = 736, ty = 404;
      line(g, T0, ty, T1, ty, { color: P.line, width: 2 });
      for (const [ord, at0] of PUBS) {
        const x = T0 + (at0 / LOOP) * (T1 - T0), on = t >= at0;
        dot(g, x, ty, 5, on ? P.blue : P.panel2, P.blue, 1.4);
        text(g, String(ord), x, ty + 16, { size: 10.5, mono: true, align: "center", color: on ? P.ink : P.ink3 });
      }
      const nx = T0 + (t / LOOP) * (T1 - T0);
      line(g, nx, ty - 10, nx, ty + 6, { color: P.green, width: 2 });
      text(g, "robots publications; the reporter advances after each", T0, ty - 18, { size: 10.5, mono: true, color: P.ink3 });
    },
  };
});
