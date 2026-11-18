# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a cpp-gpu-producer solution: exact counts and an honest GPU loop.

Usage: python verify.py --sandbox <dir> --kit <kit>      (or --truth to print the truth)

Configures, builds and runs the sandbox's CMake project (target `solution`) and compares
its per-step, per-sensor overlap counts with an analytic AABB test of every moving box at
every step (files/world.json, next to this script). The run is observed by probe.c, an
LD_PRELOAD library this script compiles: it counts the host waits for the newest
publication or launch, the host-device copies and synchronizations inside the loop, and
how publications and launches are submitted. Exits 77 without a CUDA device or a
CUDA-enabled SDK. Uses only the
standard library and NumPy; never imports ovspatialquery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 300
# The waits on the newest publication or launch, and the copies and synchronizations while
# the loop runs, that a loop with several steps in flight may still show (warm-up).
SLACK = 2
# Exit code without a CUDA device or a CUDA-enabled SDK: nothing was verified.
SKIPPED = 77


def truth(world: dict) -> dict[str, Any]:
    """Counts per step and sensor; `ambiguous` marks pairs within 1e-4 m of touching."""
    extents = {box["id"]: box["half_extents"] for box in world["box_types"]}
    half = np.asarray([extents[box] for box in world["object_boxes"]], dtype=np.float64)
    start = np.asarray(world["object_positions"], dtype=np.float64)
    velocity = np.asarray(world["object_velocities"], dtype=np.float64)
    centers = np.asarray(world["query_centers"], dtype=np.float64)
    query_half = np.asarray(world["query_half_extents"], dtype=np.float64)
    counts, loose = [], []
    for step in range(1, int(world["steps"]) + 1):
        position = start + step * velocity
        gap = np.abs(centers[:, None, :] - position[None, :, :]) - (
            query_half[:, None, :] + half[None, :, :]
        )
        margin = gap.max(axis=2)  # <= 0: overlapping or touching
        counts.append((margin <= -1e-4).sum(axis=1))
        loose.append((margin <= 1e-4).sum(axis=1))
    return {"strict": np.asarray(counts), "loose": np.asarray(loose)}


def cuda_device_count() -> int:
    """CUDA devices visible through the driver; 0 when there is no driver."""
    import ctypes

    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return 0
    count = ctypes.c_int(0)
    if driver.cuInit(0) != 0 or driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return 0
    return count.value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", type=Path)
    parser.add_argument("--kit", type=Path, default=None)
    parser.add_argument("--prebuilt-executable", type=Path)
    parser.add_argument("--truth", action="store_true", help="print truth statistics and exit")
    args = parser.parse_args()
    world = json.loads((HERE / "files" / "world.json").read_text(encoding="utf-8"))
    expected = truth(world)
    if args.truth:
        strict = expected["strict"]
        print(
            json.dumps(
                {
                    "steps": int(strict.shape[0]),
                    "sensors": int(strict.shape[1]),
                    "hits_per_step": [int(row.sum()) for row in strict[:5]],
                    "max_per_sensor": int(strict.max()),
                    "ambiguous_pairs": int((expected["loose"] - strict).sum()),
                },
                indent=2,
            )
        )
        return 0
    if args.sandbox is None:
        parser.error("--sandbox is required")
    sandbox = args.sandbox.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> bool:
        entry: dict[str, Any] = {"name": name, "pass": bool(ok)}
        if detail is not None:
            entry["detail"] = detail
        checks.append(entry)
        return ok

    def finish() -> int:
        passed = bool(checks) and all(entry["pass"] for entry in checks)
        print(json.dumps({"pass": passed, "checks": checks}, indent=2))
        return 0 if passed else 1

    env = dict(os.environ)
    if args.kit is not None:
        kit = args.kit.resolve()
        env["OVSQ_SDK"] = str(kit / "sdk")
        env["OVSQ_PYTHON"] = str(kit / "venv" / "bin" / "python")
        env["OVSTAGE_DIR"] = str(kit / "venv" / "lib" / "python3.12" / "site-packages" / "ovstage")
    if not check(
        "environment",
        all(env.get(name) for name in ("OVSQ_SDK", "OVSTAGE_DIR")),
        {name: env.get(name) for name in ("OVSQ_SDK", "OVSTAGE_DIR")},
    ):
        return finish()
    if (
        cuda_device_count() == 0
        or not (Path(env["OVSQ_SDK"]) / "lib/ovspatialquery_cuda.ptx").is_file()
    ):
        print(json.dumps({"pass": False, "skipped": "needs a CUDA device and a CUDA-enabled SDK"}))
        return SKIPPED
    if not check("CMakeLists.txt exists", (sandbox / "CMakeLists.txt").is_file()):
        return finish()

    build = sandbox / "build"
    if not args.prebuilt_executable:
        shutil.rmtree(build, ignore_errors=True)
    steps = [
        [
            "cmake",
            "-S",
            ".",
            "-B",
            "build",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DOVSQ_SDK={env['OVSQ_SDK']}",
            f"-DOVSTAGE_DIR={env['OVSTAGE_DIR']}",
        ],
        ["cmake", "--build", "build"],
    ]
    for step, name in [] if args.prebuilt_executable else zip(steps, ("configure", "build")):
        try:
            done = subprocess.run(
                step, cwd=sandbox, env=env, capture_output=True, text=True, timeout=TIMEOUT_S
            )
        except subprocess.TimeoutExpired:
            check(name, False, f"timed out after {TIMEOUT_S} s")
            return finish()
        if not check(
            name,
            done.returncode == 0,
            None if done.returncode == 0 else (done.stdout + done.stderr)[-3000:],
        ):
            return finish()
    executable = (args.prebuilt_executable or build / "solution").resolve()
    if not check("solution executable", executable.is_file(), str(executable)):
        return finish()

    with tempfile.TemporaryDirectory() as scratch:
        # The probe, preloaded into the solution: compiled here so it matches this host, or,
        # for a cross-built oracle on a host without a compiler, the probe cross-built next
        # to it (<prebuilt root>/probes/<task>/probe.so, from tools/ci/arm64_cross_build.py).
        # Without one nothing about the GPU loop is verified, so the run fails.
        probe = Path(scratch) / "probe.so"
        report = Path(scratch) / "probe.json"
        compiler = env.get("CC") or shutil.which("cc") or shutil.which("gcc")
        if compiler:
            compiled = subprocess.run(
                [
                    compiler,
                    "-shared",
                    "-fPIC",
                    "-O2",
                    f"-I{env['OVSQ_SDK']}/include",
                    str(HERE / "probe.c"),
                    "-o",
                    str(probe),
                    "-ldl",
                    "-lpthread",
                ],
                capture_output=True,
                text=True,
            )
            if not check("probe builds", compiled.returncode == 0, compiled.stderr[-2000:] or None):
                return finish()
        else:
            prebuilt = (
                args.prebuilt_executable.parents[2] / "probes" / HERE.name / "probe.so"
                if args.prebuilt_executable
                else None
            )
            if not check(
                "probe available",
                prebuilt is not None and prebuilt.is_file(),
                f"no C compiler and no cross-built probe at {prebuilt}",
            ):
                return finish()
            probe = prebuilt
        run_env = dict(env, LD_PRELOAD=str(probe), OVSQ_GPU_PROBE=str(report))
        started = time.monotonic()
        try:
            run = subprocess.run(
                [str(executable)],
                cwd=sandbox,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            check("runs", False, f"timed out after {TIMEOUT_S} s")
            return finish()
        elapsed = time.monotonic() - started
        if not check(
            "runs",
            run.returncode == 0,
            {"returncode": run.returncode, "stderr": run.stderr[-2000:]},
        ):
            return finish()
        try:
            probed = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError) as failure:
            check("probe report", False, str(failure))
            return finish()
    try:
        output = json.loads(run.stdout)
    except ValueError as failure:
        check("stdout is JSON", False, {"error": str(failure), "stdout": run.stdout[-1000:]})
        return finish()
    if not check("stdout is a JSON object", isinstance(output, dict)):
        return finish()

    strict, loose = expected["strict"], expected["loose"]
    step_count, sensor_count = strict.shape
    check(
        "steps and sensors",
        output.get("steps") == step_count and output.get("sensors") == sensor_count,
        {
            "got": [output.get("steps"), output.get("sensors")],
            "expected": [step_count, sensor_count],
        },
    )
    counts = output.get("counts")
    shaped = (
        isinstance(counts, list)
        and len(counts) == step_count
        and all(
            isinstance(row, list)
            and len(row) == sensor_count
            and all(isinstance(v, int) and not isinstance(v, bool) for v in row)
            for row in counts
        )
    )
    detail: dict[str, Any] = {"shape_ok": shaped}
    if shaped:
        got = np.asarray(counts, dtype=np.int64)
        wrong = (got < strict) | (got > loose)
        detail["wrong_cells"] = int(wrong.sum())
        if wrong.any():
            step, sensor = (int(v) for v in np.argwhere(wrong)[0])
            detail["first_wrong"] = {
                "step": step + 1,
                "sensor": sensor,
                "got": int(got[step, sensor]),
                "expected": int(strict[step, sensor]),
            }
            detail["steps_wrong"] = int(wrong.any(axis=1).sum())
    check("counts", shaped and detail.get("wrong_cells") == 0, detail)

    gpu_ms, rate = output.get("gpu_ms_per_step"), output.get("steps_per_second")
    launch_seconds = probed["launch_seconds"]
    loop_span = (
        launch_seconds[-1] - launch_seconds[-step_count]
        if len(launch_seconds) >= step_count
        else 0.0
    )
    timing_ok = is_number(gpu_ms) and is_number(rate) and gpu_ms > 0 and rate > 0
    # CUDA-event timing: the reported device time per step must be one the program read from
    # CUDA events over the whole loop, divided by the steps (task.md). Setup and teardown
    # spans contain no launch and do not count.
    if timing_ok:
        measured = probed["cuda_event_elapsed_ms"]
        candidates = []
        # One span containing at least `steps` launches (a warm-up launch before it is
        # fine), or distinct, non-overlapping spans that together contain that many, summed.
        if measured["widest_launches"] >= step_count:
            candidates.append(measured["widest"] / step_count)
        if measured["covered_launches"] >= step_count:
            candidates.append(measured["covering_sum"] / step_count)
        timing_ok = any(abs(gpu_ms - value) <= 0.05 * value + 0.002 for value in candidates)
    check(
        "timing",
        timing_ok
        # The loop cannot be faster than the whole process, and the GPU's share of a step
        # cannot exceed the step's wall time by much.
        and step_count / rate <= elapsed
        # Nor faster than the host took to submit its last `steps` launches (a warm-up
        # launch before the loop is not part of it).
        and step_count / rate >= 0.9 * loop_span - 0.002
        and gpu_ms <= 1.5 * 1000.0 / rate + 0.05,
        {
            "gpu_ms_per_step": gpu_ms,
            "steps_per_second": rate,
            "process_seconds": round(elapsed, 3),
            "cuda_event_elapsed_time_calls": probed["cuda_event_elapsed_time_calls"],
            "loop_submission_seconds": round(loop_span, 6),
            "cuda_event_elapsed_ms": probed["cuda_event_elapsed_ms"],
        },
    )

    # How the loop drove the GPU, from the probe.
    enqueued = probed["enqueued"]
    publications, launches = enqueued["publication"], enqueued["launch"]
    check(
        "one publication and one launch per step",
        publications >= step_count and launches >= step_count,
        {"publications": publications, "launches": launches, "steps": step_count},
    )
    published = probed["publications"]
    check(
        "publications prepared on the device from device rows",
        published["cuda"] >= step_count
        and published["prepared_on_device"] >= step_count
        and published["rows_on_cuda"] >= step_count,
        published,
    )
    launched = probed["launches"]
    check(
        "query inputs resident on the device and reused",
        launched["rows_on_cuda"] >= step_count
        and launched["cuda_stream"] >= step_count
        and launched["distinct_row_pointers"] <= 4,
        launched,
    )
    newest = probed["blocking_waits_on_newest"]
    events = probed["event_synchronizations_in_loop"]
    check(
        "host never waits for the step it just submitted",
        newest["publication"] <= SLACK
        and newest["launch"] <= SLACK
        and events["newest"] <= SLACK
        and probed["cuda_in_loop"]["stream_or_context_synchronizations"] <= SLACK,
        {
            "waits_on_newest": newest,
            "back_pressure_waits": probed["blocking_waits_back_pressure"],
            "event_synchronizations_in_loop": events,
            "stream_or_context_synchronizations_in_loop": probed["cuda_in_loop"][
                "stream_or_context_synchronizations"
            ],
        },
    )
    in_loop = probed["cuda_in_loop"]
    check(
        "no per-step host-device copies",
        in_loop["device_to_host_copies"] <= SLACK
        and in_loop["host_to_device_copies"] <= SLACK
        # Per-step readbacks deferred past the last launch: one history copy, plus slack.
        and probed["cuda_total"]["device_to_host_copies"] <= 1 + 2 * SLACK,
        {"in_loop": in_loop, "total": probed["cuda_total"]},
    )
    return finish()


if __name__ == "__main__":
    sys.exit(main())
