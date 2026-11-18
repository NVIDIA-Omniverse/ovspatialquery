/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The model on one page, built up in five steps. A population is the unit of update rate; a view
// groups objects that share that rate; participation bits are tags across views.
//   step 0  your process, one instance, and the sources: an ovstage stage and your own arrays
//   step 1  populations, each with its own publication clock and source
//   step 2  objects and a view: a second grouping level inside one population
//   step 3  participation bits: tags that cut across views
//   step 4  queries read one population or view at one ordinal; pair operations read two

OvsqAnim.register("model-map", (ctx) => {
  const { g, P, text, chip, roundRect, arrow, dot, line, seg, ease, hex } = ctx;
  const STEP = 1.8;
  const fade = (t, s) => seg(t, s * STEP, s * STEP + 0.5, ease.out);

  // A small clock whose hand turns with the population's publication period.
  function clock(x, y, r, t, period, col) {
    const ph = (t % period) / period, flash = Math.max(0, 1 - (t % period) / 0.3);
    dot(g, x, y, r + 3 + 3 * flash, hex(col, 0.12 + 0.2 * flash));
    dot(g, x, y, r, P.panel, col);
    const a = ph * Math.PI * 2 - Math.PI / 2;
    line(g, x, y, x + Math.cos(a) * (r - 3), y + Math.sin(a) * (r - 3), { color: col, width: 2 });
  }

  return {
    // The frame shown under reduced motion: every step is on screen.
    holdAt: 5 * STEP,
    frame(t) {
      const { w, h } = ctx.size();
      const m = 14;
      roundRect(g, m, m, w - 2 * m, h - 2 * m, 14, hex(P.ink, 0.02), P.line, 1);
      text(g, "your process", m + 16, m + 22, { size: 16, weight: 700 });
      text(g, "updates objects in space and publishes them", m + 128, m + 22, { size: 12.5, color: P.ink3, mono: true });

      const srcW = Math.round(w * 0.17), qW = Math.round(w * 0.19);
      const instX = m + 12 + srcW + 34, instW = w - instX - qW - 50;
      const instY = m + 46, instH = h - instY - m - 34;
      roundRect(g, instX, instY, instW, instH, 12, P.panel, P.green, 1.6);
      text(g, "instance", instX + 14, instY + 20, { size: 16, weight: 700, color: P.green2 });
      text(g, "owns everything · holds all budgets", instX + 98, instY + 20, { size: 12.5, color: P.ink3, mono: true });

      // Sources: an ovstage stage and the program's own arrays.
      const stA = { x: m + 12, y: instY + 38, w: srcW, h: Math.round(instH * 0.6) };
      const stB = { x: m + 12, y: stA.y + stA.h + 16, w: srcW, h: instY + instH - (stA.y + stA.h + 16) };
      [[stA, "ovstage stage", ["sealed ordinals", "", "20, 25:", "layout edits", "", "100 … 110:", "every frame"]],
       [stB, "your arrays", ["explicit", "population"]]].forEach(([b, name, lines]) => {
        roundRect(g, b.x, b.y, b.w, b.h, 10, P.panel2, P.line, 1.2);
        text(g, name, b.x + 12, b.y + 20, { size: 14.5, weight: 700 });
        lines.forEach((l, i) => text(g, l, b.x + 12, b.y + 44 + i * 17, { size: 12, color: i < 1 ? P.ink3 : P.ink2, mono: true }));
      });

      // Populations.
      const popX = instX + 14, popW = instW - 28, gapP = 10;
      const popY0 = instY + 36, popH = Math.round((instH - 36 - 14 - 2 * gapP) / 3);
      const pops = [
        { name: "scene", rate: "publishes on layout edits", period: 6, src: stA, sy: 0.4,
          filter: "filter: /World/Scene/", ords: "ordinals 20, 25",
          objs: [["ground", "GROUND"], ["ramp", "GROUND"], ["wall", ""]], view: true },
        { name: "actors", rate: "publishes every frame", period: 0.8, src: stA, sy: 0.8,
          filter: "filter: /World/Actors/", ords: "ordinals 100, 101, 104, 110",
          objs: [["robot_7", "VEHICLE"], ["robot_8", "VEHICLE"]] },
        { name: "parcels", rate: "publishes at its own rate", period: 2.2, src: stB, sy: 0.55,
          filter: "rows you supply", ords: "ordinals you choose",
          objs: [["parcel_1", "PROP"], ["parcel_2", "PROP"]] },
      ].map((p, i) => ({ ...p, y: popY0 + i * (popH + gapP) }));

      const a1 = fade(t, 1);
      if (a1 > 0) {
        g.save(); g.globalAlpha = a1;
        pops.forEach((p) => {
          roundRect(g, popX, p.y, popW, popH, 10, hex(P.blue, 0.08), P.blue, 1.3);
          text(g, "population", popX + 12, p.y + 20, { size: 14.5, weight: 700 });
          text(g, p.name, popX + 98, p.y + 20, { size: 14.5, weight: 700, color: P.blue, mono: true });
          clock(popX + 22, p.y + 48, 10, t, p.period, P.green);
          text(g, p.rate, popX + 40, p.y + 48, { size: 12.5, color: P.green2 });
          text(g, p.filter, popX + 12, p.y + 74, { size: 12, color: P.ink3, mono: true });
          text(g, p.ords, popX + 12, p.y + popH - 14, { size: 12, color: P.ink2, mono: true });
          arrow(g, p.src.x + p.src.w, p.src.y + p.src.h * p.sy, popX, p.y + popH / 2, { color: P.ink3, width: 1.6, head: 7 });
        });
        g.restore();
      }

      // Objects, the view and the participation bits.
      const a2 = fade(t, 2), a3 = fade(t, 3);
      if (a2 > 0) {
        g.save(); g.globalAlpha = a2;
        pops.forEach((p) => {
          const areaX = popX + Math.round(popW * 0.5), areaW = popW - Math.round(popW * 0.5) - 12;
          const gapO = 10, n = 3;
          const ow = Math.floor((areaW - gapO * (n - 1)) / n), oy = p.y + 22, oh = popH - 50;
          p.objs.forEach((o, i) => {
            const x = areaX + i * (ow + gapO);
            roundRect(g, x, oy, ow, oh, 7, P.panel, P.ink3, 1);
            text(g, o[0], x + ow / 2, oy + 17, { size: 11.5, align: "center" });
            if (a3 > 0) {
              g.save(); g.globalAlpha = a2 * a3;
              if (o[1]) chip(g, o[1], x + ow / 2, oy + oh - 16, { align: "center", size: 10, color: P.amber, stroke: P.amber, fill: hex(P.amber, 0.12) });
              else text(g, "no bits", x + ow / 2, oy + oh - 16, { size: 10.5, color: P.ink3, align: "center" });
              g.restore();
            }
          });
          if (p.view) {
            const vx = areaX - 6, vw = 2 * ow + gapO + 12;
            g.save(); g.setLineDash([6, 4]);
            roundRect(g, vx, oy - 6, vw, oh + 12, 9, null, P.teal, 1.6);
            g.restore();
            text(g, "view drivable_surfaces", vx, oy + oh + 19, { size: 11.5, weight: 600, color: P.teal, mono: true });
          }
        });
        g.restore();
      }

      // Queries and pairs.
      const a4 = fade(t, 4);
      if (a4 > 0) {
        g.save(); g.globalAlpha = a4;
        const qx = instX + instW + 20, qw = w - qx - m - 12;
        const q1 = { y: pops[0].y + 6, h: 96 }, q2 = { y: pops[2].y - 30, h: 116 };
        roundRect(g, qx, q1.y, qw, q1.h, 10, P.panel2, P.blue, 1.3);
        text(g, "raycast, sweep,", qx + 12, q1.y + 20, { size: 13.5, weight: 700 });
        text(g, "overlap", qx + 12, q1.y + 38, { size: 13.5, weight: 700 });
        text(g, "1 population or view", qx + 12, q1.y + 62, { size: 11.5, color: P.ink2, mono: true });
        text(g, "1 ordinal", qx + 12, q1.y + 80, { size: 11.5, color: P.ink2, mono: true });
        arrow(g, qx, q1.y + q1.h / 2, popX + popW, pops[0].y + popH / 2, { color: P.blue, width: 2, head: 8 });
        roundRect(g, qx, q2.y, qw, q2.h, 10, P.panel2, P.violet, 1.3);
        text(g, "pair snapshot,", qx + 12, q2.y + 20, { size: 13.5, weight: 700 });
        text(g, "pair reporter", qx + 12, q2.y + 38, { size: 13.5, weight: 700 });
        text(g, "2 targets, each", qx + 12, q2.y + 62, { size: 11.5, color: P.ink2, mono: true });
        text(g, "at its own ordinal", qx + 12, q2.y + 80, { size: 11.5, color: P.ink2, mono: true });
        arrow(g, qx, q2.y + 28, popX + popW, pops[1].y + popH / 2, { color: P.violet, width: 2, head: 8 });
        arrow(g, qx, q2.y + q2.h - 24, popX + popW, pops[2].y + popH / 2, { color: P.violet, width: 2, head: 8 });
        g.restore();
      }
    },
  };
});
