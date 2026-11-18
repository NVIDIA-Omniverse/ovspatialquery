/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Publications become immutable versions labelled with their ovstage ordinal. The population
// keeps the head plus `retained_ordinal_capacity` prior heads; work that pinned a version keeps
// it alive after it leaves that window, but nothing can select it any more.

OvsqAnim.register("version-window", (ctx) => {
  const { g, P, text, chip, roundRect, line, arrow, dot, glow, seg, clamp, hex } = ctx;
  const LOOP = 21, CAP = 2;
  const PUBS = [[100, 0.6], [101, 2.2], [104, 4.0], [106, 7.0], [109, 9.6], [111, 15.4]];
  const OP = { ordinal: 101, start: 5.0, end: 13.8 };
  const LATE = { ordinal: 101, at: 11.6 };
  const X0 = 196, DX = 46;
  const ox = (o) => X0 + (o - 100) * DX;
  const Y_STAGE = 78, Y_POP = 178, Y_READ = 292;

  function stateAt(t) {
    const published = PUBS.filter(([, at]) => t >= at).map(([o]) => o);
    const head = published[published.length - 1];
    const window = published.slice(-1 - CAP);
    const pinned = t >= OP.start && t < OP.end ? OP.ordinal : null;
    return { published, head, window, pinned };
  }

  return {
    // The frame shown under reduced motion: the result is on screen.
    holdAt: 12.5,
    frame(tt) {
      const t = tt % LOOP;
      const s = stateAt(t);

      // Lane labels.
      text(g, "ovstage", 24, Y_STAGE - 8, { size: 15, weight: 700 });
      text(g, "seals ordinals", 24, Y_STAGE + 12, { size: 11.5, mono: true, color: P.ink3 });
      text(g, "population", 24, Y_POP - 8, { size: 15, weight: 700, color: P.green2 });
      text(g, `retained_ordinal_`, 24, Y_POP + 12, { size: 11.5, mono: true, color: P.ink3 });
      text(g, `capacity = ${CAP}`, 24, Y_POP + 28, { size: 11.5, mono: true, color: P.ink3 });
      text(g, "readers", 24, Y_READ - 8, { size: 15, weight: 700, color: P.blue });
      text(g, "pin what they read", 24, Y_READ + 12, { size: 11.5, mono: true, color: P.ink3 });

      // Stage timeline with a playhead.
      const play = ox(100) - 20 + clamp(t / 16.2, 0, 1) * (ox(111) - ox(100) + 40);
      line(g, ox(100) - 20, Y_STAGE, ox(111) + 20, Y_STAGE, { color: P.line, width: 2 });
      for (let o = 100; o <= 111; o++) {
        const pub = PUBS.find(([q]) => q === o);
        const x = ox(o), passed = play >= x;
        line(g, x, Y_STAGE - 6, x, Y_STAGE + 6, { color: passed ? P.ink2 : P.line, width: 2 });
        text(g, String(o), x, Y_STAGE - 20, { size: 11.5, mono: true, align: "center", color: pub && passed ? P.ink : P.ink3, weight: pub ? 700 : 500 });
        if (pub && t >= pub[1]) {
          const a = seg(t, pub[1], pub[1] + 0.7);
          arrow(g, x, Y_STAGE + 10, x, Y_STAGE + 10 + (Y_POP - Y_STAGE - 42) * a, { color: hex(P.green, 0.35 + 0.65 * (1 - seg(t, pub[1] + 0.7, pub[1] + 2.2))), width: 1.6, head: 6 });
        }
      }
      glow(g, play, Y_STAGE, 14, P.green, 0.35);
      dot(g, play, Y_STAGE, 3.5, P.green);

      // Retained window bracket.
      if (s.window.length) {
        const xa = ox(s.window[0]) - 24, xb = ox(s.head) + 24;
        roundRect(g, xa, Y_POP - 34, xb - xa, 68, 12, hex(P.green, 0.06), hex(P.green, 0.35), 1.2);
        text(g, `selectable: HEAD + ${Math.min(CAP, s.window.length - 1)} retained`, xa + 4, Y_POP + 48, { size: 11, mono: true, color: P.green2 });
      }

      // Version cards.
      for (const [o, at] of PUBS) {
        if (t < at) continue;
        const x = ox(o);
        const inWindow = s.window.includes(o), isHead = o === s.head, pinned = s.pinned === o;
        const appear = seg(t, at + 0.35, at + 0.9, ctx.ease.back);
        let alpha = 1, state = "retained";
        if (!inWindow && !pinned) {
          // Leaving: when did it lose its last claim?
          const lostAt = o === OP.ordinal ? Math.max(OP.end, PUBS.find(([q]) => q === 109)[1]) : PUBS[PUBS.findIndex(([q]) => q === o) + CAP + 1][1];
          alpha = 1 - seg(t, lostAt + 0.8, lostAt + 1.8);
          state = "evicted";
          if (alpha <= 0) continue;
        } else if (!inWindow && pinned) state = "pinned";
        else if (isHead) state = "head";
        g.save(); g.globalAlpha = alpha;
        const sc = 0.6 + 0.4 * appear, w = 40 * sc, h = 40 * sc;
        const fill = state === "head" ? P.green : state === "retained" ? hex(P.green, 0.2) : state === "pinned" ? hex(P.blue, 0.12) : hex(P.ink, 0.04);
        const stroke = state === "head" || state === "retained" ? P.green : state === "pinned" ? P.blue : hex(P.red, 0.7);
        roundRect(g, x - w / 2, Y_POP - h / 2, w, h, 9, fill, stroke, 1.6);
        if (state === "pinned") {
          g.save(); g.beginPath(); g.roundRect(x - w / 2, Y_POP - h / 2, w, h, 9); g.clip();
          for (let k = -40; k < 40; k += 7) line(g, x + k, Y_POP + 20, x + k + 40, Y_POP - 20, { color: hex(P.blue, 0.25), width: 1 });
          g.restore();
        }
        text(g, String(o), x, Y_POP + 1, { size: 13 * sc, weight: 700, mono: true, align: "center", color: state === "head" ? P.onAccent : state === "evicted" ? P.ink3 : P.ink });
        if (state === "evicted") line(g, x - 15, Y_POP + 15, x + 15, Y_POP - 15, { color: P.red, width: 2 });
        if (state === "head") chip(g, "HEAD", x, Y_POP - 36, { align: "center", size: 10.5, color: P.onAccent, fill: P.green, stroke: P.green });
        if (state === "pinned") chip(g, "kept for its op", x, Y_POP - 36, { align: "center", size: 10.5, color: P.blue, fill: P.panel, stroke: P.blue });
        g.restore();
      }

      // Reader 1: a sensor reading HEAD with synchronous calls.
      const sensorX = 590;
      roundRect(g, sensorX - 74, Y_READ - 18, 148, 36, 9, hex(P.teal, 0.1), P.teal, 1.4);
      text(g, "sensor · ORDINAL_HEAD", sensorX, Y_READ, { size: 11, mono: true, align: "center", color: P.ink });
      if (s.head !== undefined) {
        const beat = (t * 1.6) % 1;
        const hx = ox(s.head);
        line(g, sensorX, Y_READ - 18, sensorX, Y_READ - 40, { color: hex(P.teal, 0.8), width: 1.4, dash: [4, 4] });
        line(g, sensorX, Y_READ - 40, hx, Y_READ - 40, { color: hex(P.teal, 0.8), width: 1.4, dash: [4, 4] });
        arrow(g, hx, Y_READ - 40, hx, Y_POP + 24, { color: P.teal, width: 1.6, head: 6 });
        dot(g, hx, Y_POP + 24 + (Y_READ - 40 - Y_POP - 24) * (1 - beat), 2.4, P.teal);
      }

      // Reader 2: a batch operation that selected RETAINED_EXACT 101 and holds it until destroy_op.
      if (t >= OP.start - 0.4) {
        const a = seg(t, OP.start - 0.4, OP.start + 0.2) * (1 - seg(t, OP.end, OP.end + 0.8));
        g.save(); g.globalAlpha = a;
        const bx = 300;
        roundRect(g, bx - 112, Y_READ - 18, 224, 36, 9, hex(P.blue, 0.1), P.blue, 1.4);
        text(g, "batch op · RETAINED_EXACT 101", bx, Y_READ, { size: 11, mono: true, align: "center", color: P.ink });
        const px = ox(OP.ordinal);
        line(g, bx - 40, Y_READ - 18, bx - 40, Y_READ - 30, { color: P.blue, width: 1.4 });
        line(g, bx - 40, Y_READ - 30, px, Y_READ - 30, { color: P.blue, width: 1.4 });
        arrow(g, px, Y_READ - 30, px, Y_POP + 24, { color: P.blue, width: 1.6, head: 6 });
        if (t > OP.end - 1.2) chip(g, "destroy_op", bx + 118, Y_READ, { size: 10.5, color: P.blue, fill: P.panel, stroke: P.blue });
        g.restore();
      }

      // Reader 3: a late selection of an ordinal that left the window.
      if (t >= LATE.at && t < LATE.at + 3.6) {
        const a = seg(t, LATE.at, LATE.at + 0.4) * (1 - seg(t, LATE.at + 3.0, LATE.at + 3.6));
        g.save(); g.globalAlpha = a;
        const lx = 300, ly = Y_READ + 54;
        roundRect(g, lx - 112, ly - 16, 224, 32, 9, hex(P.red, 0.08), hex(P.red, 0.8), 1.3);
        text(g, "new query · RETAINED_EXACT 101", lx, ly, { size: 11, mono: true, align: "center", color: P.ink });
        chip(g, "API_ORDINAL_NOT_RETAINED", lx + 120, ly, { size: 10.5, color: P.red, fill: P.panel, stroke: P.red });
        g.restore();
      }
    },
  };
});
