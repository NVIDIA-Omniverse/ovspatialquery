/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Why a batch: four synchronous calls each pay a per-call overhead before their work. The same
// four rows enqueued on a CPU plan pay it once, then its CPU job and an idle worker of the
// instance each take the next row in turn (a launch splits only from 128 rows, so the CPU lane is
// marked as illustrative); on a CUDA plan they pay it once and run in parallel on the device. The
// widths are illustrative, not measured: the work per row is the same in every lane.

OvsqAnim.register("batch-concept", (ctx) => {
  const { g, P, text, chip, roundRect, line, arrow, hex, clamp } = ctx;
  const N = 4, OVER = 0.55, WORK = 1; // in arbitrary time units
  const RATE = 1.25, START = 0.6; // units per second, delay before the clock starts
  const THREADS = 2; // the CPU job and one idle worker
  const X0 = 218, X1 = 724, END_A = N * (OVER + WORK), END_B = OVER + Math.ceil(N / THREADS) * WORK;
  const END_C = OVER + WORK;
  const tx = (u) => X0 + (u / END_A) * (X1 - X0);
  const LOOP = START + END_A / RATE + 3.2;
  const YA = 70, YB = 160, YC = 262, BH = 28, RH = 17, HB = BH * 1.5, HC = BH * 2.6;

  // Segments: [start, end, kind, label, y, height].
  const laneA = [], laneB = [[0, OVER, "over", "", YB, HB]], laneC = [[0, OVER, "over", "", YC, HC]];
  for (let i = 0; i < N; i++) {
    const s = i * (OVER + WORK);
    laneA.push([s, s + OVER, "over", "", YA, BH], [s + OVER, s + OVER + WORK, "work", `ray ${i}`, YA, BH]);
    // Row i goes to the thread that frees up first: the job and the worker alternate.
    const k = Math.floor(i / THREADS), sub = i % THREADS;
    laneB.push([OVER + k * WORK, OVER + (k + 1) * WORK, "work", `row ${i}`, YB + (sub - (THREADS - 1) / 2) * (RH + 3), RH]);
    laneC.push([OVER, OVER + WORK, "work", `row ${i}`, YC + (i - (N - 1) / 2) * (RH + 3), RH]);
  }

  function drawTrack(y, h) {
    roundRect(g, X0, y - h / 2, X1 - X0, h, 6, hex(P.ink, 0.025), P.line, 1);
  }
  function drawSegs(segs, u) {
    for (const [s, e, kind, label, y, h] of segs) {
      if (u <= s) continue;
      const a = tx(s), b = tx(Math.min(u, e)), over = kind === "over";
      roundRect(g, a + 1, y - h / 2 + 2, Math.max(1, b - a - 2), h - 4, 4,
        over ? hex(P.amber, 0.28) : hex(P.blue, 0.22), over ? P.amber : P.blue, 1.1);
      if (!over && u >= e) text(g, label, (a + b) / 2, y, { size: h < BH ? 10 : 11, mono: true, align: "center" });
    }
  }
  function finish(end, y, h, u) {
    if (u >= end) line(g, tx(end), y - h / 2 - 4, tx(end), y + h / 2 + 4, { color: P.ink, width: 2 });
  }
  function saved(end, y, h, label, u) {
    if (u < END_A) return;
    const a = tx(end), b = tx(END_A);
    line(g, b, YA + BH / 2 + 4, b, y + h / 2, { color: hex(P.ink, 0.3), dash: [3, 4] });
    roundRect(g, a + 4, y - 12, b - a - 8, 24, 4, null, hex(P.ink, 0.35), 1);
    text(g, label, (a + b) / 2, y, { size: 10.5, mono: true, color: P.ink2, align: "center" });
  }

  return {
    // The frame shown under reduced motion: every lane has finished.
    holdAt: LOOP - 0.5,
    frame(tt) {
      const t = tt % LOOP, u = clamp((t - START) * RATE, 0, END_A);
      arrow(g, X0, 20, X1, 20, { color: P.ink3, width: 1.2, head: 7 });
      text(g, "time", X0 - 10, 20, { size: 11, color: P.ink3, align: "right" });

      // Lane labels.
      text(g, "One call per query", 24, YA - 9, { size: 13, weight: 700 });
      text(g, "raycast_closest × 4", 24, YA + 10, { size: 11, mono: true, color: P.ink3 });
      text(g, "CPU plan launch", 24, YB - 9, { size: 13, weight: 700 });
      text(g, "one job, one idle worker*", 24, YB + 10, { size: 11, mono: true, color: P.ink3 });
      text(g, "* only from 128 rows", 24, YB + 26, { size: 10, color: P.ink3 });
      text(g, "CUDA plan launch", 24, YC - 9, { size: 13, weight: 700 });
      text(g, "rows in parallel", 24, YC + 10, { size: 11, mono: true, color: P.ink3 });

      drawTrack(YA, BH); drawTrack(YB, HB); drawTrack(YC, HC);
      drawSegs(laneA, u); drawSegs(laneB, u); drawSegs(laneC, u);

      // Overhead labels.
      if (u > 0) {
        for (let i = 0; i < N; i++) {
          const c = tx(i * (OVER + WORK) + OVER / 2);
          if (u > i * (OVER + WORK)) line(g, c, YA - BH / 2 - 2, c, YA - BH / 2 - 10, { color: P.amber, width: 1.2 });
        }
        line(g, tx(OVER / 2), YA - BH / 2 - 10, tx(Math.min(u, (N - 1) * (OVER + WORK)) + OVER / 2), YA - BH / 2 - 10, { color: P.amber, width: 1.2 });
        text(g, "per-call overhead, paid by every call", tx(OVER / 2) + 6, YA - BH / 2 - 19, { size: 11, color: P.amber });
        text(g, "overhead paid once per launch", tx(OVER) + 6, YB - HB / 2 - 10, { size: 11, color: P.amber });
      }

      finish(END_A, YA, BH, u);
      finish(END_B, YB, HB, u);
      finish(END_C, YC, HC, u);
      if (u >= END_B) chip(g, "1 operation · 1 result slot", tx(END_B) + 8, YB + HB / 2 + 14, { size: 10, color: P.green2, fill: P.panel, stroke: P.green });
      if (u >= END_C) chip(g, "1 operation · 1 result slot", tx(END_C) + 8, YC + HC / 2 + 14, { size: 10, color: P.green2, fill: P.panel, stroke: P.green });
      saved(END_B, YB, HB, "3 overheads and 2 rows of time saved", u);
      saved(END_C, YC, HC, "3 overheads and 3 rows of time saved", u);

      // Legend at the bottom left; the bottom right stays clear for the controls.
      roundRect(g, 24, 340, 16, 12, 3, hex(P.amber, 0.28), P.amber, 1.1);
      text(g, "overhead", 46, 346, { size: 11, color: P.ink2 });
      roundRect(g, 120, 340, 16, 12, 3, hex(P.blue, 0.22), P.blue, 1.1);
      text(g, "query work", 142, 346, { size: 11, color: P.ink2 });
      text(g, "widths are illustrative", 236, 346, { size: 11, color: P.ink3 });
    },
  };
});
