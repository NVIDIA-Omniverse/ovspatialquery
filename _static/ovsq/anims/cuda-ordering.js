/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One CUDA plan launch as a sequence diagram; each device lane waits on the lane below it. The host thread only queues work: enqueue returns
// the operation and its reservation at once, the consumer kernels are queued against the
// reservation's columns, a later status poll observes completion, and destroy_op_after_cuda
// hands the library the consumers' event. On the device, the launch runs on the caller's stream
// after the optional wait event and records result_ready; the consumer stream waits on
// result_ready and records consumer_done; the result storage is released after both.

OvsqAnim.register("cuda-ordering", (ctx) => {
  const { g, P, text, chip, roundRect, line, arrow, hex, clamp, seg } = ctx;
  const T = 10, X0 = 176, X1 = 736, tx = (s) => X0 + (s / T) * (X1 - X0);
  const RATE = 1.1, START = 0.4, LOOP = START + T / RATE + 3.5;
  const BH = 24;
  const lanes = [
    { y: 74, name: "host thread", sub: "" },
    { y: 156, name: "consumer stream", sub: "" },
    { y: 230, name: "cuda_stream", sub: "caller's stream" },
    { y: 304, name: "producer stream", sub: "optional" },
  ];
  const [HOST, CONS, CALL, PROD] = lanes.map((l) => l.y);

  // Times (in units of the axis) of each step.
  const S = {
    produce: [0.2, 3.2], // producer writes the input columns, then records cuda_wait_event
    enqueue: [0.4, 1.6], // host: enqueue returns operation + reservation
    consumer: [1.8, 3.0], // host: queue consumer kernels on the reservation columns
    query: [3.2, 5.6], // caller stream: the launch, after cuda_wait_event
    kernels: [5.6, 7.9], // consumer stream: after result_ready, then consumer_done
    status: [7.1, 8.0], // host: a later poll observes COMPLETED
    destroy: [8.2, 9.3], // host: destroy_op_after_cuda(consumer_done)
  };
  const RELEASE = 9.4; // after the destroy call and after consumer_done

  function bar(y, [s, e], u, fill, stroke, label) {
    if (u <= s) return;
    const a = tx(s), b = tx(Math.min(u, e));
    roundRect(g, a, y - BH / 2, Math.max(1, b - a), BH, 5, P.bg, null);
    roundRect(g, a, y - BH / 2, Math.max(1, b - a), BH, 5, fill, stroke, 1.2);
    if (label && u >= e) text(g, label, (a + tx(e)) / 2, y, { size: 10.5, mono: true, align: "center" });
  }
  function queued(y, s, e, u, label) {
    if (u <= s) return;
    const a = tx(s), b = tx(Math.min(u, e));
    roundRect(g, a, y - BH / 2, Math.max(1, b - a), BH, 5, P.bg, null);
    g.save(); g.setLineDash([4, 4]);
    roundRect(g, a, y - BH / 2, Math.max(1, b - a), BH, 5, null, P.ink3, 1);
    g.restore();
    if (label && u >= e) text(g, label, (a + tx(e)) / 2, y, { size: 10.5, mono: true, color: P.ink3, align: "center" });
  }
  function event(x, y, color) {
    g.save(); g.translate(x, y); g.rotate(Math.PI / 4);
    roundRect(g, -6, -6, 12, 12, 2, color, P.bg, 1.5);
    g.restore();
  }
  function link(x0, y0, x1, y1, color) { arrow(g, x0, y0, x1, y1, { color, width: 1.3, head: 7, dash: [4, 3] }); }

  return {
    // The frame shown under reduced motion: the whole sequence is on screen.
    holdAt: LOOP - 0.5,
    frame(tt) {
      const t = tt % LOOP, u = clamp((t - START) * RATE, 0, T);

      // Lanes.
      for (const l of lanes) {
        line(g, X0, l.y, X1, l.y, { color: P.line, width: 1, dash: [2, 5] });
        text(g, l.name, 24, l.sub ? l.y - 8 : l.y, { size: 12.5, weight: 700, mono: l.name === "cuda_stream" });
        if (l.sub) text(g, l.sub, 24, l.y + 10, { size: 11, color: P.ink3 });
      }
      arrow(g, X0, 22, X1, 22, { color: P.ink3, width: 1.2, head: 7 });
      text(g, "time", X0 - 10, 22, { size: 11, color: P.ink3, align: "right" });

      // Producer: writes the inputs, then records the optional wait event.
      bar(PROD, S.produce, u, hex(P.violet, 0.2), P.violet, "write inputs");
      if (u >= S.produce[1]) {
        event(tx(S.produce[1]), PROD, P.violet);
        text(g, "cuda_wait_event", tx(S.produce[1]) + 10, PROD - 20, { size: 11, mono: true, color: P.violet });
        link(tx(S.produce[1]), PROD - 8, tx(S.produce[1]), CALL + BH / 2 + 2, P.violet);
      }

      // Host: enqueue returns at once; the launch waits on the caller's stream.
      bar(HOST, S.enqueue, u, hex(P.blue, 0.2), P.blue, "enqueue");
      if (u >= S.enqueue[1]) {
        chip(g, "→ operation, reservation", tx(S.enqueue[0]), HOST - 30, { size: 10.5, color: P.blue, fill: P.panel, stroke: P.blue });
        link(tx(S.enqueue[1]) - 6, HOST + BH / 2, tx(S.enqueue[1]) - 6, CALL - BH / 2 - 2, P.blue);
      }
      queued(CALL, S.enqueue[1] - 6 / ((X1 - X0) / T), S.query[0], u, "queued");
      bar(CALL, S.query, u, hex(P.green, 0.22), P.green, "query launch");
      if (u >= S.query[1]) {
        event(tx(S.query[1]), CALL, P.green);
        text(g, "result_ready", tx(S.query[1]) + 10, CALL - 20, { size: 11, mono: true, color: P.green2 });
        link(tx(S.query[1]), CALL - 8, tx(S.query[1]), CONS + BH / 2 + 2, P.green);
      }

      // Host: consumer kernels are queued against the reservation's columns before the query runs.
      bar(HOST, S.consumer, u, hex(P.blue, 0.2), P.blue, "consumers");
      if (u >= S.consumer[1]) {
        text(g, "on reservation columns", tx(S.consumer[1]) + 4, HOST + 30, { size: 10.5, color: P.blue });
        link(tx(S.consumer[1]) - 6, HOST + BH / 2, tx(S.consumer[1]) - 6, CONS - BH / 2 - 2, P.blue);
      }
      queued(CONS, S.consumer[1] - 6 / ((X1 - X0) / T), S.kernels[0], u, "wait on result_ready");
      bar(CONS, S.kernels, u, hex(P.amber, 0.22), P.amber, "consumer kernels");
      if (u >= S.kernels[1]) {
        event(tx(S.kernels[1]), CONS, P.amber);
        text(g, "consumer_done", tx(S.kernels[1]) + 10, CONS - 20, { size: 11, mono: true, color: P.amber });
      }

      // Host: never blocks; a later poll observes completion and retires the operation.
      if (u > S.consumer[1]) {
        // A dashed line with a gap in the middle for its label.
        const a = tx(S.consumer[1]) + 4, end = tx(S.status[0]) - 4, m = (a + end) / 2;
        const b = Math.min(tx(u) - 4, end), gap = [m - 46, m + 46];
        if (b > a) line(g, a, HOST, Math.min(b, gap[0]), HOST, { color: P.ink3, width: 1.4, dash: [6, 4] });
        if (b > gap[1]) line(g, gap[1], HOST, b, HOST, { color: P.ink3, width: 1.4, dash: [6, 4] });
        if (u >= S.status[0]) text(g, "no host wait", m, HOST, { size: 11, color: P.ink2, align: "center" });
      }
      bar(HOST, S.status, u, hex(P.blue, 0.2), P.blue, "status");
      if (u >= S.status[1]) chip(g, "COMPLETED", tx(S.status[0]), HOST - 30, { size: 10.5, color: P.blue, fill: P.panel, stroke: P.blue });
      bar(HOST, S.destroy, u, hex(P.red, 0.16), P.red, "destroy");
      if (u >= S.destroy[1]) {
        text(g, "destroy_op_after_cuda(consumer_done)", X1, HOST + 30, { size: 10.5, mono: true, color: P.red, align: "right" });
      }

      // The library releases the result storage after consumer_done; nothing waits on the host.
      if (u >= RELEASE) {
        const a = seg(u, RELEASE, RELEASE + 0.3);
        g.globalAlpha = a;
        link(tx(S.kernels[1]) + 6, CONS + 6, tx(RELEASE) - 40, CALL - 12, P.amber);
        chip(g, "storage released", X1, CALL, { size: 10.5, align: "right", color: P.ink2, fill: P.panel, stroke: P.line });
        g.globalAlpha = 1;
      }

      // Legend at the bottom left; the bottom right stays clear for the controls.
      event(30, 352, P.ink3);
      text(g, "CUDA event", 44, 352, { size: 11, color: P.ink2 });
      g.save(); g.setLineDash([4, 4]); roundRect(g, 136, 345, 20, 14, 3, null, P.ink3, 1); g.restore();
      text(g, "queued, not yet running", 164, 352, { size: 11, color: P.ink2 });
    },
  };
});
