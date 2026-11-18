# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a py-batch-lidar solution against analytic ray/AABB truth.

Usage: python verify.py --sandbox <dir> --kit <kit>

Runs `$OVSQ_PYTHON solution.py` in the sandbox, parses its JSON, and compares every ray
with a slab test against the world-space boxes in scene.json (next to this script).
Uses only the standard library and NumPy; never imports ovspatialquery or ovstage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 300
DISTANCE_TOLERANCE = 1e-3  # metres
ANGLE_JITTER = 1e-6  # radians; a ray whose outcome changes under this jitter is ambiguous
RANGE_EDGE = 1e-4  # metres; a hit this close to max range may be reported either way
MAX_EXAMPLES = 8
# Runs solution.py as `python solution.py` would, logging every native call the installed
# package makes on its behalf to argv[1], so a check sees what the solution really launched.
LAUNCHER = r"""
import json, os, runpy, sys
import ovspatialquery.raw as raw

log, bind = open(sys.argv[1], "a", encoding="utf-8", buffering=1), raw.bind_exports


def record(name, function):
    def call(*args):
        entry = {"call": name}
        if name == "ovspatialquery_enqueue_raycast_plan":
            batch = args[2].contents if hasattr(args[2], "contents") else args[2]._obj
            plan = int(getattr(args[1], "value", args[1]))
            entry.update(plan=plan, rays=batch.query_count, query_bits=bool(batch.query_bits))
        log.write(json.dumps(entry) + "\n")
        return function(*args)

    return call


raw.bind_exports = lambda library, specs: {n: record(n, f) for n, f in bind(library, specs).items()}
sys.argv = ["solution.py"]
runpy.run_path(os.path.abspath("solution.py"), run_name="__main__")
"""


def ray_box(origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Entry distance of a ray into a closed AABB, or inf when it misses (slab test)."""
    t_near, t_far = 0.0, math.inf
    for axis in range(3):
        o, d = origin[axis], direction[axis]
        if abs(d) < 1e-15:
            if o < lo[axis] or o > hi[axis]:
                return math.inf
            continue
        t0, t1 = (lo[axis] - o) / d, (hi[axis] - o) / d
        if t0 > t1:
            t0, t1 = t1, t0
        t_near, t_far = max(t_near, t0), min(t_far, t1)
        if t_near > t_far:
            return math.inf
    return t_near


def cast(scene: dict, origin: np.ndarray, angle: float) -> tuple[float, str | None]:
    direction = np.array([math.cos(angle), math.sin(angle), 0.0])
    best, best_path = math.inf, None
    for box in scene["boxes"]:
        t = ray_box(origin, direction, np.array(box["min"]), np.array(box["max"]))
        if t < best:
            best, best_path = t, box["path"]
    if best > scene["max_range_m"]:
        return math.inf, None
    return best, best_path


def truth(scene: dict) -> list[list[list[tuple[float | None, str | None]]]]:
    """Per pose, per ray: every acceptable (distance, path) outcome; (None, None) is a miss."""
    step = math.radians(scene["angular_step_deg"])
    max_range = scene["max_range_m"]
    poses = []
    for pose in scene["poses"]:
        origin = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
        yaw = math.radians(pose["yaw_deg"])
        rays = []
        for ray in range(scene["ray_count"]):
            angle = yaw + ray * step
            outcomes = []
            for jitter in (0.0, -ANGLE_JITTER, ANGLE_JITTER):
                t, path = cast(scene, origin, angle + jitter)
                outcome = (None, None) if path is None else (t, path)
                if all(outcome[1] != other[1] for other in outcomes):
                    outcomes.append(outcome)
            t, path = cast(scene, origin, angle)
            if path is not None and max_range - t < RANGE_EDGE:
                outcomes.append((None, None))
            if path is None:
                # A hit just beyond max range may be reported at the boundary.
                scene_far = dict(scene, max_range_m=max_range + RANGE_EDGE)
                t_far, path_far = cast(scene_far, origin, angle)
                if path_far is not None:
                    outcomes.append((t_far, path_far))
            rays.append(outcomes)
        poses.append(rays)
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--kit", type=Path, default=None)
    args = parser.parse_args()
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
    python = env.get("OVSQ_PYTHON")
    if not check("interpreter", bool(python) and Path(python).is_file(), python):
        return finish()

    solution = sandbox / "solution.py"
    if not check("solution.py exists", solution.is_file(), str(solution)):
        return finish()
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "calls.jsonl"
        log.touch()
        try:
            run = subprocess.run(
                [python, "-c", LAUNCHER, str(log)],
                cwd=sandbox,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            check("runs", False, f"timed out after {TIMEOUT_S} s")
            return finish()
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    if not check(
        "runs", run.returncode == 0, {"returncode": run.returncode, "stderr": run.stderr[-2000:]}
    ):
        return finish()
    try:
        output = json.loads(run.stdout)
    except ValueError as failure:
        check("stdout is JSON", False, {"error": str(failure), "stdout": run.stdout[-1000:]})
        return finish()
    check("stdout is JSON", isinstance(output, dict))
    if not isinstance(output, dict):
        return finish()

    scene = json.loads((HERE / "scene.json").read_text(encoding="utf-8"))
    launches = [call for call in calls if call["call"] == "ovspatialquery_enqueue_raycast_plan"]
    check(
        "launches one reused raycast plan per pose",
        len({call["plan"] for call in launches}) == 1
        and sum(call["rays"] == scene["ray_count"] for call in launches) >= len(scene["poses"]),
        {"launches (plan, rays)": [(call["plan"], call["rays"]) for call in launches][:10]},
    )
    expected_paths = sorted(box["path"] for box in scene["boxes"])
    paths = output.get("paths")
    if not check(
        "paths",
        paths == expected_paths,
        None if paths == expected_paths else {"got": paths, "expected": expected_paths},
    ):
        return finish()

    scans = output.get("scans")
    ray_count = scene["ray_count"]
    shape_ok = (
        isinstance(scans, list)
        and len(scans) == len(scene["poses"])
        and all(
            isinstance(scan, dict)
            and isinstance(scan.get("distances"), list)
            and isinstance(scan.get("hit_index"), list)
            and len(scan["distances"]) == ray_count
            and len(scan["hit_index"]) == ray_count
            for scan in scans
        )
    )
    if not check(
        "scans shape",
        shape_ok,
        f"{len(scene['poses'])} scans of {ray_count} distances and hit_index",
    ):
        return finish()

    expected = truth(scene)
    for pose_index, (scan, rays) in enumerate(zip(scans, expected)):
        errors: list[dict[str, Any]] = []
        ambiguous = sum(1 for outcomes in rays if len(outcomes) > 1)
        hits = sum(1 for outcomes in rays if outcomes[0][1] is not None)
        max_error = 0.0
        for ray, outcomes in enumerate(rays):
            distance, index = scan["distances"][ray], scan["hit_index"][ray]
            if (distance is None) != (index is None):
                errors.append(
                    {
                        "ray": ray,
                        "got": [distance, index],
                        "why": "distance and hit_index disagree on miss",
                    }
                )
                continue
            if index is not None and (
                isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(paths)
            ):
                errors.append(
                    {"ray": ray, "got": [distance, index], "why": "hit_index out of range"}
                )
                continue
            if distance is not None and (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(distance)
            ):
                errors.append(
                    {"ray": ray, "got": [distance, index], "why": "distance is not a finite number"}
                )
                continue
            got_path = None if index is None else paths[index]
            matched = False
            for t, path in outcomes:
                if path != got_path:
                    continue
                if path is None:
                    matched = True
                elif abs(float(distance) - t) <= DISTANCE_TOLERANCE:
                    max_error = max(max_error, abs(float(distance) - t))
                    matched = True
                if matched:
                    break
            if not matched:
                errors.append(
                    {
                        "ray": ray,
                        "got": [distance, got_path],
                        "expected": [[None if p is None else round(t, 6), p] for t, p in outcomes],
                    }
                )
        check(
            f"pose {pose_index} rays",
            not errors,
            {
                "rays": ray_count,
                "expected_hits": hits,
                "ambiguous_rays": ambiguous,
                "mismatches": len(errors),
                "max_distance_error": max_error,
                "examples": errors[:MAX_EXAMPLES],
            },
        )
    return finish()


if __name__ == "__main__":
    sys.exit(main())
