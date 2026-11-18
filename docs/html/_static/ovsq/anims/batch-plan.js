/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A reusable raycast plan with two result slots. Each launch takes a free slot for as long as its
// operation lives; a third concurrent launch is refused with RESULT_CAPACITY until destroy_op
// returns one. The fetched columns hold one range per query row and one row per stored hit.
// Operation 1 is the example program's launch: four rays along +x against its four unit boxes,
// so the rows below are computed from that scene and match its output.

OvsqAnim.register("batch-plan", (ctx) => {
  const { g, P, text, chip, roundRect, line, arrow, seg, hex, clamp } = ctx;
  const LOOP = 22;
  // Launch timeline: [enqueue, complete, destroy] in seconds; slot taken.
  const ops = [
    { id: 1, at: 1.2, done: 4.4, destroy: 12.0, slot: 0 },
    { id: 2, at: 2.6, done: 6.2, destroy: 17.0, slot: 1 },
    { id: 3, at: 13.2, done: 15.8, destroy: 20.4, slot: 0 },
  ];
  const REFUSED = { at: 8.0, until: 11.0 };
  const FETCH = { at: 7.0 };
  // The example's scene: unit boxes (object, x, y) and rays from (0, y) along +x up to 20.
  const boxes = [[20, 4, 0], [21, 6, 1], [22, 8, 2], [23, 10, 3]];
  const rays = [0, 1, 5, 3];
  // Op 1's rows: per query, the closest hit as (object, distance), if any.
  const rows = rays.map((y) => {
    const hits = boxes.filter((b) => Math.abs(b[2] - y) <= 0.5).map((b) => [b[0], b[1] - 0.5]);
    hits.sort((a, b) => a[1] - b[1]);
    return hits.length && hits[0][1] <= 20 ? [hits[0]] : [];
  });
  const flat = rows.flatMap((r) => r.filter(Boolean));

  const TX0 = 340, TX1 = 736, tx = (s) => TX0 + (s / LOOP) * (TX1 - TX0);

  function opState(o, t) {
    if (t < o.at) return "none";
    if (t >= o.destroy) return "destroyed";
    if (t < o.done) return "pending";
    return "complete";
  }

  return {
    // The frame shown under reduced motion: the result is on screen.
    holdAt: 9,
    frame(tt) {
      const t = tt % LOOP;

      // Plan card.
      roundRect(g, 24, 24, 248, 196, 12, hex(P.green, 0.05), hex(P.green, 0.45), 1.3);
      text(g, "raycast plan", 40, 46, { size: 14, weight: 700, color: P.green2 });
      text(g, "create_raycast_plan()", 40, 66, { size: 11, mono: true, color: P.ink3 });
      [["target", "population"], ["device", "kDLCPU"], ["maximum_query_count", "8"], ["result_slot_count", "2"]].forEach(([k, v], i) => {
        text(g, k, 40, 92 + i * 18, { size: 11, mono: true, color: P.ink3 });
        text(g, v, 256, 92 + i * 18, { size: 11, mono: true, color: P.ink, align: "right" });
      });
      // Slots.
      for (let s = 0; s < 2; s++) {
        const owner = ops.find((o) => o.slot === s && t >= o.at && t < o.destroy);
        const x = 40 + s * 112, y = 172;
        roundRect(g, x, y, 102, 34, 8, owner ? hex(P.blue, 0.16) : hex(P.ink, 0.03), owner ? P.blue : P.line, 1.3);
        text(g, `slot ${s}`, x + 10, y + 11, { size: 10, mono: true, color: P.ink3 });
        text(g, owner ? `op ${owner.id}` : "free", x + 10, y + 25, { size: 11.5, mono: true, weight: 700, color: owner ? P.blue : P.ink3 });
      }

      // Operation lanes.
      text(g, "operations", TX0, 30, { size: 12, weight: 700, color: P.ink2 });
      line(g, TX0, 196, TX1, 196, { color: P.line });
      const now = tx(t);
      line(g, now, 40, now, 200, { color: hex(P.ink, 0.25), dash: [3, 4] });
      ops.forEach((o, i) => {
        const y = 60 + i * 40;
        text(g, `op ${o.id}`, TX0 - 8, y, { size: 11, mono: true, color: P.ink3, align: "right" });
        if (t < o.at) return;
        const a = tx(o.at), d = tx(Math.min(t, o.done)), e = tx(Math.min(t, o.destroy));
        roundRect(g, a, y - 9, Math.max(2, d - a), 18, 5, hex(P.amber, 0.25), P.amber, 1);
        if (t > o.done) roundRect(g, tx(o.done), y - 9, Math.max(2, e - tx(o.done)), 18, 5, hex(P.green, 0.22), P.green, 1);
        if (t >= o.at && t < o.at + 1.6) chip(g, "enqueue → reservation", a + 4, y - 20, { size: 9.5, color: P.amber, fill: P.panel, stroke: P.amber });
        if (t >= o.destroy) chip(g, "destroy_op", e + 4, y, { size: 9.5, color: P.ink2, fill: P.panel, stroke: P.line });
      });
      // A refused third launch while both slots are owned.
      if (t >= REFUSED.at && t < REFUSED.until) {
        const a = seg(t, REFUSED.at, REFUSED.at + 0.3) * (1 - seg(t, REFUSED.until - 0.5, REFUSED.until));
        g.globalAlpha = a;
        chip(g, "enqueue → RESULT_CAPACITY (no free slot)", tx(REFUSED.at) - 40, 178, { size: 10, color: P.red, fill: P.panel, stroke: P.red });
        g.globalAlpha = 1;
      }

      // Fetched columns of op 1.
      const fa = clamp(seg(t, FETCH.at, FETCH.at + 0.6) * (1 - seg(t, 11.4, 12.0)), 0, 1);
      const y0 = 240;
      text(g, "fetch_raycast_batch(op 1)", 24, y0, { size: 12, weight: 700, color: fa > 0 ? P.ink : P.ink3 });
      text(g, fa > 0 ? `source_ordinal 1 · query_count 4 · stored_hit_count ${flat.length}` : t >= ops[0].destroy ? "op 1 destroyed: its columns are released, slot 0 is free again" : "results become readable when the operation completes", 220, y0, { size: 11, mono: true, color: P.ink3 });
      g.save(); g.globalAlpha = 0.25 + 0.75 * fa;
      // Per-query range columns.
      const cols = [["query_status", (r) => "SUCCESS"], ["hit_offsets", (r, i) => String(rows.slice(0, i).reduce((n, q) => n + q.filter(Boolean).length, 0))], ["hit_counts", (r) => String(r.filter(Boolean).length)]];
      text(g, "per query (Q = 4)", 24, y0 + 28, { size: 10.5, mono: true, color: P.blue });
      cols.forEach(([name, f], c) => {
        const x = 24 + c * 104;
        text(g, name, x, y0 + 48, { size: 10.5, mono: true, color: P.ink3 });
        rows.forEach((r, i) => {
          const y = y0 + 68 + i * 24, miss = r.filter(Boolean).length === 0;
          roundRect(g, x, y - 10, 96, 20, 4, miss && c === 2 ? hex(P.amber, 0.12) : hex(P.ink, 0.04), P.line, 1);
          text(g, f(r, i), x + 8, y, { size: 11, mono: true, color: miss && c === 2 ? P.amber : P.ink });
        });
      });
      text(g, "a miss is a count of 0", 24 + 2 * 104, y0 + 68 + 4 * 24 - 2, { size: 10, mono: true, color: P.amber });
      // Per-hit columns.
      const hx = 360;
      text(g, `per stored hit (H = ${flat.length})`, hx, y0 + 28, { size: 10.5, mono: true, color: P.green2 });
      [["object_ids", (h) => String(h[0])], ["distances", (h) => h[1].toFixed(2)]].forEach(([name, f], c) => {
        const x = hx + c * 104;
        text(g, name, x, y0 + 48, { size: 10.5, mono: true, color: P.ink3 });
        flat.forEach((h, i) => {
          const y = y0 + 68 + i * 24;
          roundRect(g, x, y - 10, 96, 20, 4, hex(P.green, 0.08), hex(P.green, 0.35), 1);
          text(g, f(h), x + 8, y, { size: 11, mono: true });
        });
      });
      // Range arrows from hit_offsets to hit rows.
      let k = 0;
      rows.forEach((r, i) => {
        const n = r.filter(Boolean).length;
        if (n) arrow(g, 24 + 2 * 104 + 98, y0 + 68 + i * 24, hx - 6, y0 + 68 + k * 24, { color: hex(P.blue, 0.55), width: 1.2, head: 6 });
        k += n;
      });
      g.restore();
      text(g, "same columns on CPU and CUDA", 24, 418, { size: 11, mono: true, color: P.ink3 });
    },
  };
});
