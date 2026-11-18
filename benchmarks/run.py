# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run the installed scene-query comparisons; generated evidence stays in _build."""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import errno
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
# The rack mesh export the warehouse-* workloads read; the benchmark CMake registers its parity tests on it.
WAREHOUSE_MESHES = ROOT / "_build/benchmarks/warehouse-racks.bin"


def run(args, *, capture=False):
    print("+ " + " ".join(map(str, args)), file=sys.stderr, flush=True)
    result = subprocess.run(
        list(map(str, args)),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if capture else sys.stderr,
        check=True,
        timeout=1800,
    )
    return result.stdout if capture else None


def distribution(values):
    values = sorted(values)
    return dict(
        median=statistics.median(values),
        p95=values[math.ceil(0.95 * len(values)) - 1],
        minimum=values[0],
        maximum=values[-1],
        samples=len(values),
    )


def field_distributions(rows, fields=None):
    """The distribution of each field, by default every field of the first row, across rows."""
    return {field: distribution([row[field] for row in rows]) for field in fields or rows[0]}


def memory_distributions(results):
    """Field distributions of every memory phase that all results recorded."""
    return {
        phase: field_distributions([result["memory"][phase] for result in results])
        for phase in results[0]["memory"]
        if all(phase in result["memory"] for result in results)
    }


def write_metadata(destination, opts, library, **fields):
    """Record the revision, host, installed library and protocol, plus a suite's own fields."""
    sys.path.insert(0, str(ROOT / "tools/bootstrap"))
    import contract

    identity = contract.source_identity(ROOT)
    metadata = dict(
        revision=identity[0] if identity else "unknown",
        working_tree=(
            run(["git", "status", "--porcelain"], capture=True) if (ROOT / ".git").exists() else ""
        ),
        host=platform.uname()._asdict(),
        python=sys.version,
        library=str(library),
        library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        physx_configuration=(
            "debug"
            if os.environ.get("OVSPATIALQUERY_NATIVE_CONFIG", "Release") == "Debug"
            else "release"
        ),
        physx_lock=json.loads(
            (Path(library).parent.parent / "licenses/physx-PROVENANCE.json").read_text()
            if os.environ.get("OVSPATIALQUERY_PHYSX_SDK_DIR")
            else (ROOT / "deps/physx-content-lock.json").read_text()
        ),
        repetitions=opts.repetitions,
        frames=opts.frames,
        warmup=opts.warmup,
        development_run=opts.dev,
        measurement_run=opts.measure,
        cases=opts.cases,
        backends=opts.backends,
        mesh_source_faces=getattr(opts, "mesh_source_faces", 0),
        **fields,
    )
    if sys.platform == "linux":
        (destination / "host.txt").write_text(run(["lscpu"], capture=True))
        (destination / "workloads.txt").write_text(
            run(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"], capture=True)
        )
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def windows_process():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    return kernel, kernel.GetCurrentProcess()


def process_memory():
    result = {}
    if os.name == "nt":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set",
                    "working_set",
                    "peak_paged_pool",
                    "paged_pool",
                    "peak_nonpaged_pool",
                    "nonpaged_pool",
                    "pagefile",
                    "peak_pagefile",
                    "private",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        _, process = windows_process()
        measure = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        measure.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not measure(process, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        result.update(
            rss_bytes=counters.working_set,
            private_commit_bytes=counters.private,
            lifetime_peak_rss_bytes=counters.peak_working_set,
        )
    else:
        values = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/self/smaps_rollup").read_text().splitlines()[1:]
        }
        result.update(
            rss_bytes=values["Rss"],
            pss_bytes=values["Pss"],
            private_resident_bytes=values["Private_Clean"] + values["Private_Dirty"],
        )
    return result


def pin_cpu(cpu=None):
    """Restrict this process, and the children it starts, to one logical CPU.

    Call it before importing USD or Ovstage: their worker pools size themselves from
    the affinity in force when they start, and on Linux threads that already exist
    keep theirs.
    """
    if os.name == "nt":
        kernel, process = windows_process()
        mask, system = ctypes.c_size_t(), ctypes.c_size_t()
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(mask), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        selected = (mask.value & -mask.value).bit_length() - 1 if cpu is None else cpu
        if selected < 0 or not mask.value & (1 << selected):
            raise ValueError("requested CPU is outside the process affinity")
        if not kernel.SetProcessAffinityMask(process, 1 << selected):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        selected = min(os.sched_getaffinity(0)) if cpu is None else cpu
        os.sched_setaffinity(0, {selected})
    return selected


MEMORY_FIELDS = ("rss_bytes", "private_bytes", "pss_bytes", "ovspatialquery_host_charged_bytes")


def memory_summary(results):
    """Aggregate process snapshots; these are measurements, not allocator ownership."""
    memories = [result.get("memory") for result in results]
    if not all(isinstance(memory, dict) for memory in memories):
        return None

    def summarize_snapshots(section, names):
        summarized = {}
        for name in names:
            snapshots = [memory.get(section, {}).get(name) for memory in memories]
            if not all(isinstance(snapshot, dict) for snapshot in snapshots):
                summarized[name] = None
                continue
            fields = {}
            for field in MEMORY_FIELDS:
                values = [
                    snapshot.get(field) for snapshot in snapshots if snapshot.get(field) is not None
                ]
                fields[field] = distribution(values) if len(values) == len(snapshots) else None
            summarized[name] = fields
        return summarized

    return dict(
        scope=memories[0].get("scope"),
        process_measurement=memories[0].get("process_measurement"),
        host_budget_bytes=memories[0].get("host_budget_bytes"),
        phases=summarize_snapshots(
            "phases",
            ("baseline", "ready", "plan_ready", "before_first_use", "first_use_ready", "steady"),
        ),
        deltas_from_baseline=summarize_snapshots(
            "deltas_from_baseline", ("ready", "plan_ready", "steady")
        ),
        deltas_from_before_first_use=summarize_snapshots(
            "deltas_from_before_first_use", ("first_use_ready",)
        ),
    )


def cases(suite):
    # Progressive, bounded sweeps; no Cartesian product of the largest axes.
    if suite == "smoke":
        return [
            ("occupancy", 100, 64, 0, "all"),
            ("zones", 100, 16, 10, "even"),
        ]
    if suite == "ovstage":
        # Stage prims; the update percent is the share of xforms rewritten and resealed per step.
        return [
            ("ovstage", objects, 256, update, "all")
            for objects in (1000, 10000)
            for update in (1, 10, 100)
        ]
    if suite == "deform":
        # Workcells with one deforming triangle sheet each; the update percent is the share of sheets
        # changed per step. Progressive: every mode at 8 cells, then vertex waves and the crumple that
        # degrades a refitted tree at 32 and 64 cells.
        return [
            ("deform-none", 8, 1024, 0, "all"),
            ("deform-rigid", 8, 1024, 100, "all"),
            ("deform-vertex", 8, 1024, 100, "all"),
            ("deform-fold", 8, 1024, 100, "all"),
            ("deform-crumple", 8, 1024, 100, "all"),
            ("deform-mixed", 8, 1024, 100, "all"),
            ("deform-vertex", 32, 1024, 25, "all"),
            ("deform-vertex", 32, 1024, 100, "all"),
            ("deform-vertex", 64, 1024, 100, "all"),
            ("deform-fold", 64, 1024, 100, "all"),
            ("deform-mixed", 64, 1024, 25, "all"),
        ]
    if suite == "pairs":
        # Objects per side; the update percent is the second side's moved share per step.
        return [
            ("pairs", objects, 1, update, "all")
            for objects in (100, 1000, 10000)
            for update in (1, 10)
        ]
    if suite == "pair-cells":
        # Engine-style cells (the query count is bodies per cell: 64 packed, 2 sparse). One
        # population against itself, two partially overlapping views of it, and the same
        # bodies split across two populations as the disjoint control; the self cases vary
        # motion and density, and --pair-tolerance-milli 100 selects the engine's margin.
        return [
            ("pairs-self", 1000, 64, 1, "all"),
            ("pairs-self", 10000, 64, 1, "all"),
            ("pairs-self", 10000, 64, 100, "all"),
            ("pairs-self", 10000, 2, 1, "all"),
            ("pairs-views", 10000, 64, 1, "all"),
            ("pairs-split", 10000, 64, 1, "all"),
        ]
    if suite == "sensor-point-cloud":
        # Synthetic beam counts; the largest is inspired by an output-scale comment.
        return [("sensor-point-cloud", 5, n, 0, "all") for n in (16384, 65536, 259200)]
    if suite == "warehouse":
        # Consumer-grounded rack meshes ingested from the warehouse demo (benchmarks/warehouse_meshes.py).
        return [
            ("warehouse-rays", 52, 1, 0, "all"),
            ("warehouse-rays", 52, 1024, 0, "all"),
            ("warehouse-rays", 52, 4096, 0, "all"),
            # The browser scans 16,384 exact rays on every animated frame.
            ("warehouse-rays", 52, 16384, 0, "all"),
            # One lattice row and the demo's full 1,024-sweep lattice; the latter is the case the
            # CPU sweep clipping round measured.
            ("warehouse-sweeps", 52, 64, 0, "all"),
            ("warehouse-sweeps", 52, 1024, 0, "all"),
            ("warehouse-box-sweeps", 52, 64, 0, "all"),
            ("warehouse-overlaps", 52, 12, 0, "all"),
            ("warehouse-overlaps", 52, 256, 0, "all"),
        ]
    if suite == "update-scaling":
        # First hold changed rows at 100, then vary them at 10,000 total objects.
        return [
            (workload, objects, 64, update, "all")
            for workload in ("convex-rays", "rays")
            for objects, update in ((100, 100), (1000, 10), (10000, 1), (10000, 10), (10000, 100))
        ]
    if suite == "convex":
        return [
            ("convex-rays", 1, 64, 0, "all"),
            ("convex-rays", 100, 1024, 0, "all"),
            ("convex-rays", 100, 1024, 10, "even"),
            ("convex-rays", 10000, 1024, 10, "even"),
            ("convex-sweeps", 100, 256, 0, "all"),
            ("convex-sweeps", 100, 256, 10, "all"),
            ("convex-sweeps", 10000, 1024, 10, "all"),
            ("convex-overlaps", 100, 256, 0, "all"),
            ("convex-overlaps", 100, 256, 10, "even"),
        ]
    if suite == "statics":
        # Static scenes whose queries reach far past their first contact: box sweeps over a sea of
        # 16,384 boxes at growing lengths with the first contact fixed at about 12 m, a sphere
        # sweep at the longest, sweeps and sphere overlaps of one dense mesh, long sweeps onto a
        # mesh level, and giant overlaps of levels of 16 and 1,024 tiles.
        return [
            *[
                ("sea-box-sweeps-" + str(length), 16384, 2, 0, "all")
                for length in (15, 30, 60, 120, 2000)
            ],
            ("sea-sweeps-2000", 16384, 2, 0, "all"),
            ("dense-mesh-box-sweeps", 1, 64, 0, "all"),
            ("dense-mesh-sweeps", 1, 64, 0, "all"),
            ("dense-mesh-sphere-any", 1, 64, 0, "all"),
            ("dense-mesh-sphere-overlaps", 1, 64, 0, "all"),
            ("level-box-sweeps", 16, 1024, 0, "all"),
            ("level-sweeps", 16, 1024, 0, "all"),
            *[
                (workload, tiles, 1, 0, "all")
                for workload in ("level-box-overlaps", "level-sphere-overlaps")
                for tiles in (16, 1024)
            ],
        ]
    if suite == "camera-rays":
        # One pinhole image of coherent rays: the empty scene (fixed per-ray cost), one box as a
        # bounds candidate (per-candidate cost) and a 69,136-triangle closed mesh (exact work).
        return [
            ("camera-mesh", 0, 65536, 0, "all"),
            ("camera-box", 1, 65536, 0, "all"),
            ("camera-mesh", 1, 16384, 0, "all"),
            ("camera-mesh", 1, 65536, 0, "all"),
            ("camera-mesh", 64, 65536, 0, "all"),
        ]
    if suite == "mesh-raycast":
        return [
            ("grounding-mesh", 1, 64, 0, "all"),
            ("grounding-mesh", 1, 1024, 0, "all"),
            ("grounding-mesh", 16, 1024, 10, "all"),
            ("grounding-mesh", 100, 1024, 0, "all"),
        ]
    if suite == "mesh-smoke":
        return [
            ("grounding-mesh", 1, 64, 0, "all"),
            ("mesh-sweeps", 1, 16, 0, "all"),
            ("mesh-box-sweeps", 1, 16, 0, "all"),
        ]
    if suite == "mesh":
        return [
            ("grounding-mesh", 1, 1024, 0, "all"),
            ("grounding-mesh", 16, 1024, 0, "all"),
            ("grounding-mesh", 100, 1024, 1, "all"),
            ("mesh-sweeps", 16, 64, 10, "all"),
            ("mesh-box-sweeps", 16, 64, 10, "all"),
        ]
    if suite == "actor-smoke":
        return [
            ("grounding", 100, 16, 0, "all"),
            ("support", 100, 1, 0, "all"),
            ("avoidance-mixed", 100, 8, 10, "all"),
            ("actor-frame", 13, 8, 100, "all"),
            ("actor-frame-bits", 13, 8, 100, "all"),
        ]
    if suite == "actor":
        return [
            ("grounding", 100, 64, 0, "all"),
            ("grounding", 10000, 1024, 0, "all"),
            ("grounding", 10000, 4096, 1, "all"),
            ("support", 10000, 1, 0, "all"),
            ("avoidance", 100, 16, 0, "all"),
            ("avoidance", 1000, 64, 10, "all"),
            ("avoidance-mixed", 1000, 64, 10, "all"),
            ("avoidance-mixed", 10000, 256, 1, "all"),
            ("actor-frame", 13, 2, 100, "all"),
            ("actor-frame-bits", 13, 2, 100, "all"),
            ("actor-frame", 13, 8, 100, "all"),
            ("actor-frame", 100, 64, 10, "all"),
            ("actor-frame-bits", 100, 64, 10, "all"),
            ("actor-frame", 1000, 256, 1, "all"),
        ]
    if suite == "iteration":
        return [
            ("rays", 10000, 1024, 0, "all"),
            ("occupancy", 100, 4096, 0, "all"),
            ("zones", 100, 16, 10, "even"),
            ("grounding", 100, 64, 0, "all"),
            ("avoidance-mixed", 100, 8, 10, "all"),
            ("grounding-mesh", 1, 64, 0, "all"),
            ("mesh-sweeps", 1, 16, 0, "all"),
        ]
    base = [
        ("rays", 1, 1, 0, "all"),
        ("rays", 2, 1024, 0, "all"),
        ("rays", 100, 1024, 0, "all"),
        ("occupancy", 100, 4096, 0, "all"),
        ("zones", 100, 16, 0, "all"),
        ("zones", 100, 16, 10, "even"),
    ]
    if suite == "scale":
        base += [("rays", 10000, n, 0, "all") for n in (64, 1024, 16384)]
        base += [("occupancy", 10000, n * n, 0, "all") for n in (256, 512)]
        base += [("zones", 10000, n, 0, "all") for n in (1, 256)]
        base += [("rays", 10000, 1024, n, "all") for n in (1, 10)]
    return base


ACTOR_WORKLOADS = {
    "grounding",
    "support",
    "avoidance",
    "avoidance-mixed",
    "actor-frame",
    "actor-frame-bits",
}
# Suites with their own runner module rather than the native harness's cases.
OWN_RUNNER_SUITES = ("handoff", "placement", "warehouse-publication", "warehouse-scans")
# Backends that need a CUDA driver and device; a host without them records each as unavailable.
CUDA_BACKENDS = (
    "cuda",
    "ovstage-cuda",
    "reporter-cuda",
    "reporter-gpu",
    "explicit-device",
    "ovstage-device",
)
# Plan workloads without a CUDA lane. The harness runs grounding-mesh as an actor cycle, which
# creates its own CPU plans and refuses CUDA.
CPU_ONLY_WORKLOADS = ACTOR_WORKLOADS | {"grounding-mesh"}


def case_backends(workload, queries):
    """The backends a native case measures; each CUDA one is compared with the CPU in its parity run."""
    if workload.startswith("pairs"):
        # The PhysX tree lane crosses two populations only; ABP is the cell workloads' baseline.
        if workload == "pairs":
            baseline = ("physx",)
        else:
            baseline = (*(("physx",) if workload == "pairs-split" else ()), "abp")
        return (*baseline, "snapshot", "reporter", "reporter-cuda", "reporter-gpu")
    if workload == "ovstage":
        return ("physx", "explicit", "ovstage", "explicit-device", "ovstage-device")
    if workload.startswith("deform-"):
        # Direct recook and refit baselines, the installed CPU plan and the CUDA plan.
        return ("physx", "physx-refit", "physx-clone", "ovstage", "ovstage-cuda")
    if workload == "sensor-point-cloud" and queries > 16384:
        return ("physx", "batch", "cuda")
    if workload in CPU_ONLY_WORKLOADS:
        return ("physx", "scalar", "batch")
    return ("physx", "scalar", "batch", "cuda")


def selected_cases(opts):
    """(name, case, backends) for the suite's cases that --cases and --backends select."""
    for case in cases(opts.suite):
        workload, objects, queries, update, subset = case
        name = f"{workload}-{objects}-{queries}-update{update}-{subset}"
        if opts.cases and not any(selected in name for selected in opts.cases):
            continue
        # The smoke suite runs parity only.
        backends = () if opts.suite == "smoke" else case_backends(workload, queries)
        if opts.backends is not None:
            backends = tuple(backend for backend in backends if backend in opts.backends)
            if not backends:
                continue
        yield name, case, backends


@contextlib.contextmanager
def heavy_lock():
    if sys.platform == "linux":
        import fcntl

        with open("/tmp/ovspatial-dgxc-heavy.lock", "a") as lock:
            print("Waiting for shared heavy-work lock", file=sys.stderr, flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
    else:
        import msvcrt

        lock_path = Path(tempfile.gettempdir()) / "ovspatial-heavy.lock"
        with open(lock_path, "a+b") as lock:
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            reported_wait = False
            while True:
                try:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno != errno.EACCES:
                        raise
                    if not reported_wait:
                        print("Waiting for shared heavy-work lock", file=sys.stderr, flush=True)
                        reported_wait = True
                    time.sleep(0.25)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    suite_option = parser.add_argument(
        "--suite",
        choices=(
            "smoke",
            "quick",
            "scale",
            "actor-smoke",
            "actor",
            "iteration",
            "mesh-smoke",
            "mesh",
            "mesh-raycast",
            "statics",
            "camera-rays",
            "convex",
            "sensor-point-cloud",
            "update-scaling",
            "pairs",
            "pair-cells",
            "ovstage",
            "deform",
            "warehouse",
            "placement",
            "warehouse-publication",
            "warehouse-scans",
            "handoff",
        ),
        default="quick",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Independent repetitions per backend: fresh processes, or measured blocks with "
        "fresh plans in one process where a suite parses a large scene (requires --measure or --dev)",
    )
    parser.add_argument(
        "--frames", type=int, help="Measured frames per process (default 7; 3 with --dev)"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--measure", action="store_true", help="Explicit repeated performance measurement"
    )
    modes.add_argument(
        "--dev",
        action="store_true",
        help="Iteration mode: one repetition, three frames, one warmup, no CTest lane, existing "
        "exports reused, results marked as not a qualification",
    )
    parser.add_argument(
        "--cases",
        help="Comma-separated substrings; only case names containing one of them run",
    )
    parser.add_argument(
        "--backends",
        help="Comma-separated backend names; each case measures only those of its own backends, "
        "and a case left with none is skipped",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected cases and their backends as JSON, without building or running",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail a case whose parity run did not compare a CUDA lane (GPU qualification)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--native-build", type=Path, default=ROOT / "_build/bootstrap/native-stable"
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="Logical CPU every suite pins the runner and its children to, on Linux and "
        "Windows; default first allowed logical CPU",
    )
    parser.add_argument("--python-observation", action="store_true")
    parser.add_argument(
        "--mesh-side", type=int, default=32, help="Terrain subdivisions per tile side (2..224)"
    )
    parser.add_argument(
        "--unique-meshes",
        type=int,
        choices=(0, 1),
        default=0,
        help="Register/cook one definition per tile instead of sharing",
    )
    parser.add_argument(
        "--mesh-source-faces",
        type=int,
        choices=(0, 1),
        default=0,
        help="Register every installed mesh with source-face columns repeating the submitted "
        "numbering, to measure what the columns cost (hits are unchanged)",
    )
    parser.add_argument(
        "--actor-hit-capacity",
        type=int,
        default=32,
        help="Actor batch overlap capacity per row; zero reserves population size",
    )
    parser.add_argument(
        "--reuse-plans",
        type=int,
        choices=(0, 1),
        default=1,
        help="Actor caller strategy: retain plans or recreate each cycle",
    )
    parser.add_argument(
        "--pair-tolerance-milli",
        type=int,
        default=0,
        help="Pairs contact tolerance in millimetres applied by both adapters",
    )
    parser.add_argument(
        "--radius-groups",
        type=int,
        default=0,
        help="Actor exact radius groups; zero selects workload default",
    )
    opts = parser.parse_args()
    if opts.repetitions is not None and not (opts.measure or opts.dev):
        parser.error("--repetitions requires --measure or --dev")
    if opts.repetitions is None:
        opts.repetitions = 5 if opts.measure else 1
    if opts.frames is None:
        opts.frames = 3 if opts.dev else 7
    opts.warmup = 1 if opts.dev else 3
    opts.cases = [c for c in (opts.cases or "").split(",") if c]
    if opts.backends is not None:
        opts.backends = [b for b in opts.backends.split(",") if b]
        known = {
            backend
            for suite in suite_option.choices
            for workload, _, queries, _, _ in cases(suite)
            for backend in case_backends(workload, queries)
        }
        unknown = sorted(set(opts.backends) - known)
        if not opts.backends or unknown:
            parser.error(f"--backends needs known backend names, not {','.join(unknown) or 'none'}")
    if (opts.list or opts.backends) and opts.suite in OWN_RUNNER_SUITES:
        parser.error(f"--list and --backends cover the native harness suites, not {opts.suite}")
    if opts.backends and opts.suite == "smoke":
        parser.error("--backends does not apply to the parity-only smoke suite")
    if opts.list:
        # The inventory the runner would measure; CUDA backends are listed even where this host
        # would record them as unavailable.
        listing = [dict(case=name, backends=list(b)) for name, _, b in selected_cases(opts)]
        print(json.dumps(listing, indent=2))
        return 0
    # Each selected case's parity run must compare CUDA; with none selected nothing would be.
    if opts.require_cuda and (opts.suite in OWN_RUNNER_SUITES or not any(selected_cases(opts))):
        parser.error("--require-cuda needs at least one selected native-harness case")
    if opts.repetitions < 1 or opts.frames < 1:
        parser.error("repetitions and frames must be positive")
    if opts.measure and (opts.repetitions < 5 or opts.frames < 5):
        parser.error(
            "use at least five independent repetitions and five samples per process, or --dev"
        )
    if opts.dev:
        print(
            "Development run: fewer processes and frames, no CTest lane; not a qualification",
            file=sys.stderr,
        )
    elif not opts.measure:
        print("Correctness qualification: parity checks, no repeated timing runs", file=sys.stderr)
    destination = (
        opts.output or ROOT / "_build/benchmarks" / time.strftime("%Y%m%d-%H%M%S")
    ).resolve()
    if not destination.is_relative_to(ROOT / "_build"):
        parser.error("output must be under this repository’s ignored _build directory")
    destination.mkdir(parents=True, exist_ok=False)
    native_build = opts.native_build.resolve()
    install = native_build / "install"
    deps = ROOT / "_build/target-deps"
    suffix = ".exe" if os.name == "nt" else ""
    cmake, ctest = (deps / "cmake/bin" / (name + suffix) for name in ("cmake", "ctest"))
    build = ROOT / "_build/benchmarks/native"
    executable = build / (
        "Release/scene_query_benchmark.exe" if os.name == "nt" else "scene_query_benchmark"
    )
    library = install / (
        "bin/ovspatialquery.dll" if os.name == "nt" else "lib/libovspatialquery.so"
    )
    os.environ["OVSPATIALQUERY_LIBRARY"] = str(library)
    with heavy_lock():
        # Reuse product verification for the exact SDK and installed bytes.
        sys.path.insert(0, str(ROOT / "tools/bootstrap"))
        import main as bootstrap

        target = bootstrap.contract.host_platform()
        cache = bootstrap._read_cmake_cache(native_build)
        selected_physx = bootstrap.contract._physx_sdk_root(deps).resolve()
        built_physx = Path(cache.get("OVSPATIALQUERY_PHYSX_SDK_DIR") or deps / "physx").resolve()
        if selected_physx != built_physx:
            raise RuntimeError(
                "Benchmark PhysX SDK differs from the installed build; use the same OVSPATIALQUERY_PHYSX_SDK_DIR"
            )
        for issue in (
            bootstrap.contract._staged_physx_issue(deps, target),
            bootstrap._install_issue(
                native_build, cache.get("OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION") == "ON", target
            ),
        ):
            if issue:
                raise RuntimeError(issue)
        # The ovstage workload builds a stage through the pinned SDK the companion was
        # verified against; its loader and runtime closure are found through the
        # process environment, which every timed child inherits.
        ovstage_sdk = cache.get("OVSPATIALQUERY_OVSTAGE_SDK_DIR", "")
        if not ovstage_sdk:
            raise RuntimeError("the verified install does not record its pinned ovstage package")
        runtime_variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        os.environ[runtime_variable] = os.pathsep.join(
            [str(Path(ovstage_sdk) / "bin" / "plugins"), str(Path(ovstage_sdk) / "bin")]
            + [entry for entry in (os.environ.get(runtime_variable, ""),) if entry]
        )
        if opts.suite == "handoff":
            # The co-simulation hand-off membership routes, timed by the experiment's harness in
            # fresh CPU-only processes on this one CPU.
            opts.cpu = pin_cpu(opts.cpu)
            sys.path.insert(0, str(ROOT / "benchmarks/experiments/handoff_membership"))
            importlib.import_module("handoff_membership").run_suite(opts, destination, library)
            return
        if opts.suite in ("placement", "warehouse-publication", "warehouse-scans"):
            # These suites import USD and Ovstage and may time work in this process, so
            # pin before the import; every child then starts on the same CPU.
            opts.cpu = pin_cpu(opts.cpu)
            suite = importlib.import_module(opts.suite.replace("-", "_"))
            suite.run_suite(opts, destination, library)
            return
        if opts.suite == "warehouse" and not (opts.dev and WAREHOUSE_MESHES.is_file()):
            # The export rebuilds the demo scene (about 20 s); iteration reuses the last one.
            run([sys.executable, ROOT / "benchmarks/warehouse_meshes.py", WAREHOUSE_MESHES])
        # Build before pinning: every timed child then inherits the same one-CPU budget.
        configure = [
            cmake,
            "-S",
            ROOT / "benchmarks",
            "-B",
            build,
            "-DOVSPATIALQUERY_INSTALL_DIR=" + str(install),
            "-DPHYSX_SDK_DIR=" + str(bootstrap.contract._physx_sdk_root(deps)),
            # A custom SDK may keep PVD in its release archives, so its link keeps Ws2_32.
            "-DOVSPATIALQUERY_CUSTOM_PHYSX="
            + ("ON" if os.environ.get("OVSPATIALQUERY_PHYSX_SDK_DIR") else "OFF"),
            "-DOVSPATIALQUERY_OVSTAGE_SDK_DIR=" + ovstage_sdk,
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        # The harness builds with the measured library's compiler, so timings reproduce. The
        # pinned MSVC builds through Ninja in its own environment.
        compilers = ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER")
        generator = None
        if os.name == "nt":
            if cache.get("CMAKE_GENERATOR") == "Ninja":
                bootstrap.contract.apply_compiler_environment(deps, target)
                generator = ["-G", "Ninja"]
                compilers += ("CMAKE_MAKE_PROGRAM",)
                executable = build / "scene_query_benchmark.exe"
            else:
                generator = ["-G", "Visual Studio 17 2022", "-A", "x64", "-T", "v143,host=x64"]
                compilers = ()
            configure += generator
        selected = {name: cache[name] for name in compilers if cache.get(name)}
        if os.name == "nt":
            for name, value in selected.items():
                # Reconfiguration may store just cl while CMake resolves it to an
                # absolute path in the harness. Compare the actual tools in the
                # selected compiler environment rather than their cache spellings.
                executable_path = shutil.which(value)
                if executable_path is None:
                    raise FileNotFoundError(f"Library's {name} executable was not found: {value}")
                selected[name] = Path(executable_path).resolve().as_posix()
        configure += [f"-D{name}={value}" for name, value in selected.items()]
        # A generated harness build cannot switch generators in place, and on a compiler change
        # CMake drops the command line's other cache values: start such a build from scratch.
        if (build / "CMakeCache.txt").is_file():
            previous = bootstrap._read_cmake_cache(build)
            if os.name == "nt":
                for name in selected:
                    if previous.get(name):
                        previous[name] = Path(previous[name]).resolve().as_posix()
            if (generator and previous.get("CMAKE_GENERATOR") != generator[1]) or any(
                previous.get(name) != value for name, value in selected.items()
            ):
                shutil.rmtree(build)
        run(configure)
        run([cmake, "--build", build, "--config", "Release", "--parallel", "8"])
        opts.cpu = pin_cpu(opts.cpu)
        write_metadata(
            destination,
            opts,
            library,
            affinity=[opts.cpu],
            ovstage_sdk=ovstage_sdk,
            suite=opts.suite,
        )
        (destination / "native-cache.txt").write_text((native_build / "CMakeCache.txt").read_text())
        (destination / "benchmark-cache.txt").write_text((build / "CMakeCache.txt").read_text())
        if not opts.dev:
            run(
                [
                    ctest,
                    "--test-dir",
                    build,
                    "-C",
                    "Release",
                    "--interactive-debug-mode",
                    "0",
                    "--output-on-failure",
                ]
            )
        summaries = []
        rows = []
        memory_rows = []
        for name, case, backends in selected_cases(opts):
            workload, objects, queries, update, subset = case
            common = [
                "--workload",
                workload,
                "--objects",
                objects,
                "--queries",
                queries,
                "--update-percent",
                update,
                "--subset",
                subset,
            ]
            if workload in ACTOR_WORKLOADS:
                common += [
                    "--actor-hit-capacity",
                    opts.actor_hit_capacity,
                    "--reuse-plans",
                    opts.reuse_plans,
                    "--radius-groups",
                    opts.radius_groups,
                ]
            # The actor-frame table is a terrain tile of the same side.
            if workload in {
                "grounding-mesh",
                "mesh-sweeps",
                "mesh-box-sweeps",
                "actor-frame",
                "actor-frame-bits",
            }:
                common += [
                    "--mesh-side",
                    opts.mesh_side,
                    "--unique-meshes",
                    opts.unique_meshes,
                ]
            # The level tiles share one terrain definition of this side.
            if workload.startswith("level-"):
                common += ["--mesh-side", opts.mesh_side]
            if workload.startswith("pairs"):
                common += ["--pair-tolerance-milli", opts.pair_tolerance_milli]
            if workload.startswith("warehouse-"):
                common += ["--warehouse-meshes", WAREHOUSE_MESHES]
            if opts.mesh_source_faces:
                common += ["--mesh-source-faces", opts.mesh_source_faces]
            parity = run([executable, "--backend", "parity", *common], capture=True)
            (destination / (name + "-parity.json")).write_text(parity)
            # Where the parity run compared a CUDA lane with the CPU, that lane must also run.
            parity_record = json.loads(parity)
            cuda_compared = (
                parity_record.get("cuda_parity") == "parity"
                or parity_record.get("device_parity") == "parity"
                or parity_record.get("cuda_reporter") is True
            )
            if opts.require_cuda and not cuda_compared:
                raise RuntimeError(
                    f"{name}: the parity run did not compare a CUDA lane: "
                    + str(parity_record.get("cuda_parity", "the workload has none"))
                )
            if opts.suite == "smoke" or not (opts.measure or opts.dev):
                continue
            collected = {backend: [] for backend in backends}
            iterations = {}
            unavailable = {}
            for backend in list(collected):
                try:
                    probe_text = run(
                        [
                            executable,
                            "--backend",
                            backend,
                            *common,
                            "--frames",
                            1,
                            "--warmup",
                            min(2, opts.warmup),
                            "--iterations",
                            1,
                        ],
                        capture=True,
                    )
                except subprocess.CalledProcessError as error:
                    if backend not in CUDA_BACKENDS or cuda_compared:
                        raise
                    # No CUDA driver, device or backend here: record the omission rather than hide it.
                    unavailable[backend] = f"probe exited with {error.returncode}"
                    del collected[backend]
                    continue
                probe = json.loads(probe_text)
                elapsed = probe["samples"][0]["query_ms"] + probe["samples"][0]["observation_ms"]
                iterations[backend] = (
                    1
                    if update or workload.startswith("deform-")
                    else max(1, min(10000, math.ceil(20 / max(elapsed, 0.0001))))
                )
            for repetition in range(opts.repetitions):
                order = list(collected)
                if repetition % 2:
                    order.reverse()
                for backend in order:
                    result = json.loads(
                        run(
                            [
                                executable,
                                "--backend",
                                backend,
                                *common,
                                "--frames",
                                opts.frames,
                                "--warmup",
                                opts.warmup,
                                "--iterations",
                                iterations[backend],
                            ],
                            capture=True,
                        )
                    )
                    (destination / f"{name}-{backend}-{repetition}.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
                    collected[backend].append(result)
                    memory = result.get("memory")
                    if isinstance(memory, dict):
                        for section, delta_reference in (
                            ("phases", None),
                            ("deltas_from_baseline", "baseline"),
                            ("deltas_from_before_first_use", "before_first_use"),
                        ):
                            for phase, snapshot in memory.get(section, {}).items():
                                if isinstance(snapshot, dict):
                                    memory_rows.append(
                                        dict(
                                            case=name,
                                            backend=backend,
                                            repetition=repetition,
                                            phase=phase,
                                            delta_reference=delta_reference,
                                            **snapshot,
                                        )
                                    )
                    for frame, sample in enumerate(result["samples"]):
                        rows.append(
                            dict(
                                case=name,
                                backend=backend,
                                repetition=repetition,
                                frame=frame,
                                iterations=iterations[backend],
                                **sample,
                            )
                        )
            item = dict(
                case=name,
                queries=queries,
                backends={},
                unavailable=unavailable,
            )
            for backend, results in collected.items():
                samples = [sample for result in results for sample in result["samples"]]
                phases = {}
                for field in sorted({field for sample in samples for field in sample}):
                    values = [sample.get(field) for sample in samples]
                    if all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        for value in values
                    ):
                        phases[field] = distribution(values)
                for field in (
                    "setup_ms",
                    "geometry_setup_ms",
                    "publication_ms",
                    "plan_ms",
                    "first_use_cycle_ms",
                    "first_use_query_ms",
                ):
                    values = [result.get(field) for result in results]
                    if all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        for value in values
                    ):
                        phases[field] = distribution(values)
                rss = [r["peak_rss_bytes"] for r in results]
                phases["peak_rss_bytes"] = (
                    distribution(rss) if all(v is not None for v in rss) else None
                )
                phases["memory"] = memory_summary(results)
                phases["repetition_query_medians_ms"] = [
                    statistics.median(s["query_ms"] for s in r["samples"]) for r in results
                ]
                throughput = queries * 1000 / phases["query_ms"]["median"]
                if workload.startswith("pairs"):
                    # One step per frame; the cell workloads' query count is bodies per cell.
                    phases["steps_per_second"] = 1000 / phases["query_ms"]["median"]
                    phases["throughput_unit"] = "steps_per_second"
                elif workload == "ovstage":
                    phases["steps_per_second"] = 1000 / phases["frame_ms"]["median"]
                    phases["throughput_unit"] = "steps_per_second"
                elif workload == "sensor-point-cloud":
                    phases["rays_per_second"] = throughput
                    phases["throughput_unit"] = "rays_per_second"
                elif workload.startswith(("avoidance", "actor-frame")):
                    phases["actor_cycles_per_second"] = throughput
                    phases["throughput_unit"] = "actor_cycles_per_second"
                else:
                    phases["queries_per_second"] = throughput
                    phases["throughput_unit"] = "queries_per_second"
                item["backends"][backend] = phases
            # Ratios need the baseline; --backends may leave it out, and then there are none.
            baseline = item["backends"].get("abp" if workload.startswith("pairs-") else "physx")
            item["end_to_end_ratios"] = (
                {
                    b: d["frame_ms"]["median"] / baseline["frame_ms"]["median"]
                    for b, d in item["backends"].items()
                }
                if baseline
                else None
            )
            summaries.append(item)
            (destination / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        if rows:
            with (destination / "samples.csv").open("w", newline="") as stream:
                fieldnames = sorted({field for row in rows for field in row})
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        if memory_rows:
            with (destination / "memory.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(memory_rows[0]))
                writer.writeheader()
                writer.writerows(memory_rows)
        if opts.python_observation:
            # The 1- and 30-row batch lanes track the fixed Python cost of one launch, which
            # 1,024 rows hide; a launch there takes well under a millisecond, so they run at
            # least 1,000 frames.
            lanes = [("typed", 1024), ("raw", 1024), ("batch", 1024), ("batch", 1), ("batch", 30)]
            for repetition in range(opts.repetitions):
                order = lanes[::-1] if repetition % 2 else lanes
                for mode, queries in order:
                    small = queries < 1024
                    payload = run(
                        [
                            sys.executable,
                            ROOT / "benchmarks/python_observation.py",
                            "--mode",
                            mode,
                            "--queries",
                            queries,
                            "--frames",
                            max(opts.frames, 1000) if small else opts.frames,
                            "--warmup",
                            max(opts.warmup, 100) if small else opts.warmup,
                        ],
                        capture=True,
                    )
                    name = f"python-{mode}-{repetition}.json"
                    if small:
                        name = f"python-{mode}-{queries}rows-{repetition}.json"
                    (destination / name).write_text(payload)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
