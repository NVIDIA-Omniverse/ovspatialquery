# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-grounded placement waves, analytical occupancy, through benchmarks/run.py."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np
from ovspatialquery import InstanceConfig, Library

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/warehouse"))
from examples.warehouse.placement import Placement, ledger_any, occupancy_oracle  # noqa: E402
from run import distribution, pin_cpu, process_memory, run  # noqa: E402

CASES = ((1000, 256), (10000, 1024))
HOST_BUDGET = 512 << 20
PHASES = ("query_ms", "publish_ms", "loop_ms", "ledger_query_ms")


def execute(opts, workload=None):
    cpu = pin_cpu(opts.cpu)
    owned = workload is None
    instance = None
    if owned:
        library = Library.open(
            opts.library, require_explicit=True, require_ovstage=opts.backend == "ovstage"
        )
        instance = library.instance(InstanceConfig(host_byte_budget=HOST_BUDGET))
    try:
        before = process_memory()
        if owned:
            workload = Placement(
                instance,
                source=opts.backend,
                count=opts.count,
                candidates=opts.candidates,
                capacity=opts.frames + opts.warmup,
                seed=47,
            )
        else:
            workload.rewind()
        ready = process_memory()
        # Record setup now: the parity reset below authors, loads and plans again.
        setup = dict(
            prepare_ms=workload.prepare_ms,
            stage_or_registration_ms=workload.stage_ms,
            usd_author_ms=workload.usd_author_ms,
            source_load_ms=workload.source_load_ms,
            initial_publication_ms=workload.initial_publish_ms,
            plan_setup_ms=workload.initial_plan_setup_ms,
            setup_ms=workload.setup_ms,
        )
        block_plan_setup_ms = workload.plan_setup_ms
        if opts.parity:
            # An all-blocked candidate wave has no possible accepted index.
            occupied = workload.centers[np.arange(opts.candidates) % opts.count]
            blocked, ordinal, _ = workload.query(occupied)
            np.testing.assert_array_equal(blocked, np.ones(opts.candidates, dtype=bool))
            if np.flatnonzero(~blocked).size or ordinal != workload.ordinal:
                raise AssertionError("All-blocked control produced a clear candidate")
        samples = []
        sequence = []
        for frame in range(opts.frames + opts.warmup):
            prior = workload.all_centers.copy()
            state = workload.step(validate=opts.parity)
            # This independent baseline returns the same any-hit Boolean vector.
            # It is not the client's full-list API, PhysX, or Newton mesh contacts.
            started = perf_counter()
            expected = ledger_any(workload.candidate_centers, prior, workload.all_halves)
            ledger_ms = (perf_counter() - started) * 1000
            np.testing.assert_array_equal(workload.candidate_hits, expected)
            if state["accepted_index"] is not None:
                accepted = workload.accepted_centers[-1]
                if occupancy_oracle(accepted[None], prior, workload.all_halves)[0]:
                    raise AssertionError("Accepted placement overlaps an occupied analytical box")
                # Query next snapshot after activation; new occupied position must hit.
                if opts.parity:
                    if not workload.query(accepted[None])[0][0]:
                        raise AssertionError(
                            "Accepted object is absent from its published snapshot"
                        )
                    if workload.query(accepted[None], ordinal=state["query_ordinal"])[0][0]:
                        raise AssertionError("Retained pre-acceptance snapshot changed")
            sequence.append(state["accepted_index"])
            if frame >= opts.warmup:
                samples.append(
                    {
                        **{key: state[key] for key in PHASES if key != "ledger_query_ms"},
                        "ledger_query_ms": ledger_ms,
                        "query_ordinal": state["query_ordinal"],
                        "published_ordinal": state["ordinal"],
                        "accepted": state["accepted"],
                        "blocked": state["blocked"],
                    }
                )
        if opts.parity:
            # Reset must work in the same instance without re-registering retired IDs.
            workload.reset()
            if workload.step(validate=True)["accepted_index"] != sequence[0]:
                raise AssertionError("Reset did not reproduce the first placement")
        return dict(
            host_byte_budget=HOST_BUDGET,
            backend=opts.backend,
            count=opts.count,
            candidates=opts.candidates,
            cpu=cpu,
            parity=opts.parity,
            samples=samples,
            block_plan_setup_ms=block_plan_setup_ms,
            accepted_sequence=sequence,
            setup=setup,
            memory=dict(before=before, ready=ready, after=process_memory()),
            semantics="Analytical boxes; fixed topology parked acceptance pool; one accepted pose per publication; CPU ANY batches",
            baseline="Independent linear any-hit AABB checker, not direct PhysX or live randomizer",
            query_timing="enqueue/wait/fetch and Boolean result observation; input packing is included in loop_ms",
        )
    finally:
        if owned:
            if workload is not None:
                workload.close()
            if instance is not None:
                instance.close()


def run_suite(opts, destination, library):
    repetitions = opts.repetitions if opts.measure or opts.dev else 0
    sys.path.insert(0, str(ROOT / "tools/bootstrap"))
    import contract

    identity = contract.source_identity(ROOT)
    if identity is None:
        raise RuntimeError("Placement metadata requires a Git checkout or stamped source archive")
    revision, dirty = identity
    metadata = dict(
        revision=revision,
        host_byte_budget=HOST_BUDGET,
        source_revision=revision,
        source_dirty=dirty,
        source="examples/warehouse/placement.py",
        library=str(library),
        ovstage=importlib.metadata.version("ovstage"),
        cpu=opts.cpu,
        qualification=opts.measure,
        repetitions=repetitions,
        limitations=[
            "Synthetic consumer-grounded analogue, not a live randomizer run",
            "Analytical occupancy only; no arbitrary mesh-vs-mesh replacement",
            "Preallocated parked objects activated by transform updates, not structural insertion",
            "Ledger baseline adapted to any-hit and inclusive touching, not client epsilon-policy parity",
            "Host process memory; library/device allocations are not independently attributed",
        ],
    )
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summaries = []
    for count, candidates in CASES:
        name = f"placement-{count}-candidates{candidates}"
        if opts.cases and not any(part in name for part in opts.cases):
            continue
        command = [
            sys.executable,
            Path(__file__),
            "--library",
            library,
            "--count",
            count,
            "--candidates",
            candidates,
            "--frames",
            opts.frames,
            "--warmup",
            opts.warmup,
            "--cpu",
            opts.cpu,
        ]
        parity = {}
        parity_memory = {}
        for backend in ("explicit", "ovstage"):
            value = json.loads(run([*command, "--backend", backend, "--parity"], capture=True))
            parity[backend] = value["accepted_sequence"]
            parity_memory[backend] = value["memory"]
            (destination / f"{name}-{backend}-parity.json").write_text(
                json.dumps(value, indent=2) + "\n"
            )
        if parity["explicit"] != parity["ovstage"]:
            raise AssertionError("Explicit and Ovstage placement sequences differ")
        if not repetitions:
            summaries.append(dict(case=name, parity="passed", measured=False))
        collected = {backend: [] for backend in parity}
        # Parse each source once per case. Rewind only acceptance-slot transforms
        # between warmed blocks; alternate adapters without repeating USD ingestion.
        if repetitions:
            loaded = Library.open(library, require_explicit=True, require_ovstage=True)
            instance = loaded.instance(InstanceConfig(host_byte_budget=HOST_BUDGET))
            workloads = {}
            try:
                for backend in parity:
                    workloads[backend] = Placement(
                        instance,
                        source=backend,
                        count=count,
                        candidates=candidates,
                        capacity=opts.frames + opts.warmup,
                    )
                for repetition in range(repetitions):
                    for backend in (
                        ("explicit", "ovstage") if repetition % 2 == 0 else ("ovstage", "explicit")
                    ):
                        worker = argparse.Namespace(
                            library=library,
                            backend=backend,
                            count=count,
                            candidates=candidates,
                            frames=opts.frames,
                            warmup=opts.warmup,
                            cpu=opts.cpu,
                            parity=False,
                        )
                        value = execute(worker, workloads[backend])
                        # Per-backend memory comes from isolated parity processes,
                        # not this process holding both sources concurrently.
                        value["memory"] = {}
                        (destination / f"{name}-{backend}-{repetition}.json").write_text(
                            json.dumps(value, indent=2) + "\n"
                        )
                        collected[backend].append(value)
            finally:
                for workload in workloads.values():
                    workload.close()
                instance.close()
        for backend, results in collected.items():
            if results:
                summaries.append(
                    dict(
                        case=name,
                        backend=backend,
                        phases={
                            field: distribution(
                                [
                                    statistics.median(row[field] for row in result["samples"])
                                    for result in results
                                ]
                            )
                            for field in PHASES
                        },
                        block_plan_setup_ms=distribution(
                            [result["block_plan_setup_ms"] for result in results]
                        ),
                        setup=results[0]["setup"],
                        setup_samples=1,
                        memory=parity_memory[backend],
                        memory_samples=1,
                    )
                )
        (destination / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if not summaries:
        raise ValueError("No placement cases matched")
    print(
        f"Placement: {len(summaries)} rows; {'development only' if opts.dev else 'measured' if repetitions else 'parity only'}; {destination}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("explicit", "ovstage"), required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--parity", action="store_true")
    print(json.dumps(execute(parser.parse_args()), indent=2))
