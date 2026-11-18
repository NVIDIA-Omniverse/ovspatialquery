# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a cpp-warehouse-scan solution against analytic ray/box truth.

Usage: python verify.py --sandbox <dir> --kit <kit>      (or --truth to print the truth)

Configures, builds and runs the sandbox's CMake project (target `solution`), parses its
JSON, and compares the scan statistics with a slab test of all 16384 scanner rays
against the world-space rack boxes in scene.json (next to this script). The floor is
not part of the truth: it must be excluded. Uses only the standard library and NumPy;
never imports ovspatialquery or ovstage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 300
ANGLE_JITTER = 1e-6  # radians; a ray whose outcome changes under this jitter is grazing
EDGE_EPS = 1e-5  # metres; a distance this close to a histogram edge may land in either bin
MISMATCH_FRACTION = 0.001  # rays allowed to differ from the analytic truth (grazing rays)
MAX_QUERY_MS = 60_000.0


def ray_directions(scanner: dict, d_az: float = 0.0, d_el: float = 0.0) -> np.ndarray:
    """Unit directions, row-major: index = row * columns + column (row = elevation)."""
    columns, rows = int(scanner["columns"]), int(scanner["rows"])
    hfov = math.radians(scanner["horizontal_fov_deg"])
    vfov = math.radians(scanner["vertical_fov_deg"])
    yaw = math.radians(scanner["yaw_deg"])
    az = yaw + hfov * ((np.arange(columns) + 0.5) / columns - 0.5) + d_az
    el = vfov * ((np.arange(rows) + 0.5) / rows - 0.5) + d_el
    el_grid, az_grid = np.meshgrid(el, az, indexing="ij")
    return np.stack(
        (np.cos(el_grid) * np.cos(az_grid), np.cos(el_grid) * np.sin(az_grid), np.sin(el_grid)),
        axis=-1,
    ).reshape(-1, 3)


def closest(
    origin: np.ndarray, directions: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest entry distance and box index per ray (inf / -1 on a miss), slab test."""
    best = np.full(len(directions), np.inf)
    best_box = np.full(len(directions), -1, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / directions  # (R, 3)
    chunk = 64
    for start in range(0, len(lo), chunk):
        box_lo, box_hi = lo[start : start + chunk], hi[start : start + chunk]  # (B, 3)
        with np.errstate(invalid="ignore"):
            t0 = (box_lo[:, None, :] - origin) * inverse[None, :, :]  # (B, R, 3)
            t1 = (box_hi[:, None, :] - origin) * inverse[None, :, :]
        near = np.nan_to_num(np.minimum(t0, t1), nan=-np.inf).max(axis=2)
        far = np.nan_to_num(np.maximum(t0, t1), nan=np.inf).min(axis=2)
        near = np.maximum(near, 0.0)
        t = np.where(near <= far, near, np.inf)  # (B, R)
        index = t.argmin(axis=0)
        value = t[index, np.arange(t.shape[1])]
        better = value < best
        best[better] = value[better]
        best_box[better] = start + index[better]
    return best, best_box


def truth(scene: dict) -> dict:
    scanner = scene["scanner"]
    origin = np.asarray(scanner["position"], dtype=np.float64)
    max_range = float(scanner["max_range_m"])
    edges = np.asarray(scanner["histogram_edges_m"], dtype=np.float64)
    lo = np.asarray([box["min"] for box in scene["boxes"]], dtype=np.float64)
    hi = np.asarray([box["max"] for box in scene["boxes"]], dtype=np.float64)
    rack_of_box = np.asarray([box["rack"] for box in scene["boxes"]], dtype=np.int64)

    outcomes = []
    for d_az, d_el in (
        (0.0, 0.0),
        (ANGLE_JITTER, ANGLE_JITTER),
        (-ANGLE_JITTER, -ANGLE_JITTER),
        (ANGLE_JITTER, -ANGLE_JITTER),
        (-ANGLE_JITTER, ANGLE_JITTER),
    ):
        t, box = closest(origin, ray_directions(scanner, d_az, d_el), lo, hi)
        hit = t <= max_range
        rack = np.where(hit, rack_of_box[np.maximum(box, 0)], -1)
        outcomes.append((t, hit, rack))
    t, hit, rack = outcomes[0]
    bins = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
    grazing = np.zeros(len(t), dtype=bool)
    for _, other_hit, other_rack in outcomes[1:]:
        grazing |= (other_hit != hit) | (other_rack != rack)
    grazing |= hit & (np.abs(t[:, None] - edges[None, :]).min(axis=1) < EDGE_EPS)
    grazing |= np.abs(t - max_range) < EDGE_EPS

    counts = np.bincount(rack[hit], minlength=int(scene["rack_count"]))
    order = sorted(range(len(counts)), key=lambda index: (-int(counts[index]), index))
    histogram = np.bincount(bins[hit], minlength=len(edges) - 1)
    return {
        "ray_count": int(len(t)),
        "hit_count": int(hit.sum()),
        "histogram": [int(value) for value in histogram],
        "rack_hits": [int(value) for value in counts],
        "top_racks": [
            {"rack": int(index), "hits": int(counts[index])}
            for index in order[: scanner["top_racks"]]
        ],
        "grazing_rays": int(grazing.sum()),
    }


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", type=Path)
    parser.add_argument("--kit", type=Path, default=None)
    parser.add_argument("--prebuilt-executable", type=Path)
    parser.add_argument("--truth", action="store_true", help="print the analytic truth and exit")
    args = parser.parse_args()
    if args.prebuilt_executable:
        args.prebuilt_executable = args.prebuilt_executable.resolve()
    scene = json.loads((HERE / "scene.json").read_text(encoding="utf-8"))
    if args.truth:
        expected = truth(scene)
        expected.pop("rack_hits")
        print(json.dumps(expected, indent=2))
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

    if not check("CMakeLists.txt exists", (sandbox / "CMakeLists.txt").is_file()):
        return finish()
    sources = [
        path
        for path in sandbox.rglob("*")
        if path.suffix in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")
        and "build" not in path.relative_to(sandbox).parts
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in sources)
    check(
        "uses an ovspatialquery batch raycast plan",
        bool(re.search(r"ovspatialquery_create_raycast_plan", text))
        and bool(re.search(r"ovspatialquery_enqueue_raycast_plan", text)),
        "the C/C++ sources must create and launch an ovspatialquery raycast plan",
    )

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
    executable = args.prebuilt_executable or build / "solution"
    if not check("solution executable", executable.is_file(), str(executable)):
        return finish()

    try:
        run = subprocess.run(
            [str(executable)],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        check("runs", False, f"timed out after {TIMEOUT_S} s")
        return finish()
    if not check(
        "runs", run.returncode == 0, {"returncode": run.returncode, "stderr": run.stderr[-2000:]}
    ):
        return finish()
    try:
        output = json.loads(run.stdout)
    except ValueError as failure:
        check("stdout is JSON", False, {"error": str(failure), "stdout": run.stdout[-1000:]})
        return finish()
    if not check("stdout is a JSON object", isinstance(output, dict)):
        return finish()

    expected = truth(scene)
    scanner = scene["scanner"]
    tolerance = math.ceil(MISMATCH_FRACTION * expected["ray_count"]) + expected["grazing_rays"]

    check(
        "ray_count",
        output.get("ray_count") == expected["ray_count"],
        {"got": output.get("ray_count"), "expected": expected["ray_count"]},
    )

    hit_count = output.get("hit_count")
    check(
        "hit_count",
        is_int(hit_count) and abs(hit_count - expected["hit_count"]) <= tolerance,
        {"got": hit_count, "expected": expected["hit_count"], "tolerance": tolerance},
    )

    histogram = output.get("histogram")
    bins = len(scanner["histogram_edges_m"]) - 1
    shape_ok = (
        isinstance(histogram, list)
        and len(histogram) == bins
        and all(is_int(v) and v >= 0 for v in histogram)
    )
    l1 = sum(abs(a - b) for a, b in zip(histogram, expected["histogram"])) if shape_ok else None
    check(
        "histogram",
        shape_ok and l1 <= 2 * tolerance and (not is_int(hit_count) or sum(histogram) == hit_count),
        {
            "got": histogram,
            "expected": expected["histogram"],
            "l1_difference": l1,
            "allowed_l1": 2 * tolerance,
            "must_sum_to_hit_count": True,
        },
    )

    top = output.get("top_racks")
    want = expected["top_racks"]
    top_ok = (
        isinstance(top, list)
        and len(top) == len(want)
        and all(
            isinstance(row, dict) and is_int(row.get("rack")) and is_int(row.get("hits"))
            for row in top
        )
    )
    detail: dict[str, Any] = {"got": top, "expected": want, "tolerance": tolerance}
    if top_ok:
        racks = [row["rack"] for row in top]
        hits = [row["hits"] for row in top]
        top_ok = (
            all(0 <= rack < len(expected["rack_hits"]) for rack in racks)
            and len(set(racks)) == len(racks)
            and hits == sorted(hits, reverse=True)
            # Each reported count matches that rack's true count.
            and all(abs(h - expected["rack_hits"][r]) <= tolerance for r, h in zip(racks, hits))
            # And the racks are the most hit ones (ties within tolerance accepted).
            and all(
                expected["rack_hits"][r] + 2 * tolerance >= w["hits"] for r, w in zip(racks, want)
            )
        )
    check("top_racks", top_ok, detail)

    times = output.get("query_times_ms")
    mean = output.get("mean_query_ms")
    times_ok = (
        isinstance(times, list)
        and len(times) == int(scanner["repeats"])
        and all(is_number(v) and 0.0 < v < MAX_QUERY_MS for v in times)
        and is_number(mean)
        and abs(mean - sum(times) / len(times)) <= 1e-6 + 1e-3 * abs(mean)
    )
    check("query timing", times_ok, {"query_times_ms": times, "mean_query_ms": mean})
    return finish()


if __name__ == "__main__":
    sys.exit(main())
