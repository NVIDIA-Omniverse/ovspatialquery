# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify py-time-samples: run <sandbox>/solution.py and compare with analytic truth.

Truth comes from scene.json alone: linear interpolation of the authored
translate samples (held outside the sampled range), the parent offset, and a
slab test of each population prim's world box against the fixed ray.
Never imports ovspatialquery or ovstage.
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

import numpy as np

HERE = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 300
TOLERANCE = 1.0e-4
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
QUERIES = (
    # output key, ordinal key, time code key
    ("head_after_time_10", "time_10", "second"),
    ("original_after_time_10", "time_0", "first"),
    ("head_after_time_5", "time_5", "third"),
)


def translate_at(samples: dict[str, list[float]], time_code: float) -> np.ndarray:
    times = sorted(float(key) for key in samples)
    values = {float(key): np.array(value, dtype=np.float64) for key, value in samples.items()}
    if time_code <= times[0]:
        return values[times[0]]
    if time_code >= times[-1]:
        return values[times[-1]]
    for low, high in zip(times, times[1:]):
        if low <= time_code <= high:
            weight = (time_code - low) / (high - low)
            return (1.0 - weight) * values[low] + weight * values[high]
    raise AssertionError("unreachable")


def world_box(description: dict, time_code: float) -> tuple[np.ndarray, np.ndarray]:
    if "translate_samples" in description:
        center = translate_at(description["translate_samples"], time_code)
    else:
        center = np.array(description["translate"], dtype=np.float64)
    center = center + np.array(
        description.get("parent_translate", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    half = description["size"] / 2.0 if description["kind"] == "cube" else description["radius"]
    return center - half, center + half


def slab(
    origin: np.ndarray,
    direction: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    max_distance: float,
):
    near, far = 0.0, max_distance
    for axis in range(3):
        if abs(direction[axis]) < 1.0e-12:
            if origin[axis] < low[axis] or origin[axis] > high[axis]:
                return None
            continue
        t0 = (low[axis] - origin[axis]) / direction[axis]
        t1 = (high[axis] - origin[axis]) / direction[axis]
        near, far = max(near, min(t0, t1)), min(far, max(t0, t1))
        if near > far:
            return None
    return near


def truth(scene: dict, time_code: float) -> dict:
    origin = np.array(scene["ray"]["origin"], dtype=np.float64)
    direction = np.array(scene["ray"]["direction"], dtype=np.float64)
    best = None
    for path in scene["population_paths"]:
        low, high = world_box(scene["objects"][path], time_code)
        distance = slab(origin, direction, low, high, scene["ray"]["max_distance"])
        if distance is not None and (best is None or distance < best[1]):
            best = (path, distance)
    if best is None:
        return {"prim_path": None, "distance": None, "position": None}
    return {
        "prim_path": best[0],
        "distance": float(best[1]),
        "position": [float(value) for value in origin + best[1] * direction],
    }


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_stdout(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--kit", type=Path, default=None)
    arguments = parser.parse_args()
    sandbox = arguments.sandbox.resolve()

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        return bool(ok)

    def finish() -> int:
        passed = bool(checks) and all(item["pass"] for item in checks)
        print(json.dumps({"pass": passed, "checks": checks}, indent=2))
        return 0 if passed else 1

    scene = json.loads((HERE / "truth" / "scene.json").read_text(encoding="utf-8"))
    environment = dict(os.environ)
    if arguments.kit is not None:
        kit = arguments.kit.resolve()
        environment["OVSQ_SDK"] = str(kit / "sdk")
        environment["OVSQ_PYTHON"] = str(kit / "venv" / "bin" / "python")
        environment["OVSTAGE_DIR"] = str(
            kit / "venv" / "lib" / "python3.12" / "site-packages" / "ovstage"
        )
    python = environment.get("OVSQ_PYTHON")
    if not check("OVSQ_PYTHON set", bool(python) and Path(python).exists(), str(python)):
        return finish()

    solution = sandbox / "solution.py"
    if not check("solution.py exists", solution.is_file(), str(solution)):
        return finish()

    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "calls.jsonl"
        log.touch()
        try:
            completed = subprocess.run(
                [python, "-c", LAUNCHER, str(log)],
                cwd=sandbox,
                env=environment,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            check("solution finishes", False, f"timed out after {TIMEOUT_SECONDS} s")
            return finish()
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    if not check(
        "solution exits 0",
        completed.returncode == 0,
        f"exit {completed.returncode}; stderr tail: {completed.stderr[-2000:]}",
    ):
        return finish()
    names = {call["call"] for call in calls}
    if not check(
        "raycasts an Ovstage population through ovspatialquery",
        "ovspatialquery_create_ovstage_population" in names
        and any("raycast" in name for name in names),
        f"library calls: {sorted(name for name in names if 'ovstage' in name or 'raycast' in name)}",
    ):
        return finish()
    output = parse_stdout(completed.stdout)
    if not check("stdout is one JSON object", isinstance(output, dict), completed.stdout[-2000:]):
        return finish()

    ordinals = output.get("ordinals")
    keys = ("time_0", "time_10", "time_5")
    if not check(
        "ordinals has integer time_0, time_10, time_5",
        isinstance(ordinals, dict) and all(is_int(ordinals.get(key)) for key in keys),
        json.dumps(ordinals),
    ):
        return finish()
    check(
        "ordinals strictly increase in publication order",
        0 <= ordinals["time_0"] < ordinals["time_10"] < ordinals["time_5"],
        json.dumps(ordinals),
    )

    for output_key, ordinal_key, time_key in QUERIES:
        time_code = float(scene["time_codes"][time_key])
        expected = truth(scene, time_code)
        actual = output.get(output_key)
        if not check(f"{output_key} present", isinstance(actual, dict), json.dumps(actual)):
            continue
        check(
            f"{output_key}.source_ordinal == ordinals.{ordinal_key}",
            is_int(actual.get("source_ordinal"))
            and actual["source_ordinal"] == ordinals[ordinal_key],
            f"got {actual.get('source_ordinal')!r}, expected {ordinals[ordinal_key]}",
        )
        check(
            f"{output_key}.prim_path",
            actual.get("prim_path") == expected["prim_path"],
            f"got {actual.get('prim_path')!r}, expected {expected['prim_path']!r} at time code {time_code:g}",
        )
        if expected["distance"] is None:
            check(
                f"{output_key} miss fields are null",
                actual.get("distance") is None and actual.get("position") is None,
                json.dumps(actual),
            )
            continue
        distance = actual.get("distance")
        check(
            f"{output_key}.distance",
            is_number(distance) and abs(distance - expected["distance"]) <= TOLERANCE,
            f"got {distance!r}, expected {expected['distance']:.6f}",
        )
        position = actual.get("position")
        position_ok = (
            isinstance(position, list)
            and len(position) == 3
            and all(is_number(value) for value in position)
            and max(abs(a - b) for a, b in zip(position, expected["position"])) <= TOLERANCE
        )
        check(
            f"{output_key}.position",
            position_ok,
            f"got {position!r}, expected {[round(v, 6) for v in expected['position']]}",
        )
    return finish()


if __name__ == "__main__":
    sys.exit(main())
