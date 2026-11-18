# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify py-pair-zones: run the sandbox's solution.py and compare with analytic truth.

Truth comes from truth/scene.json (written with scene.usda by gen_scene.py): world
AABBs of the robots (parent translate + linearly interpolated translate samples
+- half extent) and of the zones (AABB of the 8 corners of translate * rotateY *
scale applied to the local [-1, 1]^3 cube), overlapping when the closed intervals
intersect on every axis. Never imports ovspatialquery or ovstage.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 300
# Smallest allowed |separation| between any robot/zone AABB pair in the scene, so
# float32 world extents cannot flip a verdict. Checked, not assumed.
SCENE_MARGIN = 0.05


def interpolate(samples: dict[str, list[float]], time: float) -> np.ndarray:
    keys = sorted((float(k), np.asarray(v, dtype=float)) for k, v in samples.items())
    if time <= keys[0][0]:
        return keys[0][1]
    if time >= keys[-1][0]:
        return keys[-1][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= time <= t1:
            return v0 + (v1 - v0) * ((time - t0) / (t1 - t0))
    raise AssertionError("unreachable")


def zone_aabb(zone: dict, half: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(zone["rotate_y_deg"])
    c, s = math.cos(theta), math.sin(theta)
    rotate_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    linear = rotate_y @ np.diag(zone["scale"])
    corners = np.array(list(itertools.product((-half, half), repeat=3)))
    world = corners @ linear.T + np.asarray(zone["translate"], dtype=float)
    return world.min(axis=0), world.max(axis=0)


def robot_aabb(robot: dict, parent: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray]:
    center = parent + interpolate(robot["samples"], time)
    return center - robot["half"], center + robot["half"]


def separation(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """Positive: gap along the most separated axis; negative: overlap depth."""
    gaps = np.maximum(a[0] - b[1], b[0] - a[1])
    return float(gaps.max())


def truth(scene: dict) -> tuple[dict, float]:
    parent = np.asarray(scene["robots_parent_translate"], dtype=float)
    zones = [(z["path"], zone_aabb(z, scene["zone_local_half"])) for z in scene["zones"]]
    active_by_time: dict[int, set[tuple[str, str]]] = {}
    margin = math.inf
    for time in scene["times"]:
        active = set()
        for robot in scene["robots"]:
            box = robot_aabb(robot, parent, float(time))
            for zone_path, zone_box in zones:
                gap = separation(box, zone_box)
                margin = min(margin, abs(gap))
                if gap <= 0.0:
                    active.add((robot["path"], zone_path))
        active_by_time[time] = active
    steps = []
    previous: set[tuple[str, str]] = set()
    for time in scene["times"]:
        current = active_by_time[time]
        steps.append(
            {
                "time": time,
                "entered": sorted([list(p) for p in current - previous]),
                "exited": sorted([list(p) for p in previous - current]),
            }
        )
        previous = current
    first = scene["times"][0]
    return {"snapshot_t0": sorted([list(p) for p in active_by_time[first]]), "steps": steps}, margin


def parse_output(stdout: str):
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def pair_list_problem(value) -> str | None:
    if not isinstance(value, list):
        return f"expected a list of pairs, got {type(value).__name__}"
    for item in value:
        if not (
            isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item)
        ):
            return f"pair is not [robot_path, zone_path]: {item!r}"
    if len({tuple(item) for item in value}) != len(value):
        return "duplicate pairs"
    if value != sorted(value):
        return "pairs are not sorted"
    return None


def compare(output, expected) -> list[dict]:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    if not isinstance(output, dict):
        check("output is a JSON object", False, f"got {type(output).__name__}")
        return checks
    check("output is a JSON object", True)
    for key in ("snapshot_t0", "steps"):
        check(f"key '{key}' present", key in output)
    if "snapshot_t0" in output:
        problem = pair_list_problem(output["snapshot_t0"])
        check("snapshot_t0 well-formed", problem is None, problem or "")
        check(
            "snapshot_t0 matches",
            output["snapshot_t0"] == expected["snapshot_t0"],
            f"got {output['snapshot_t0']!r}, expected {expected['snapshot_t0']!r}",
        )
    steps = output.get("steps")
    if not isinstance(steps, list) or len(steps) != len(expected["steps"]):
        check("steps has one entry per time", False, f"got {steps!r:.300}")
        return checks
    check("steps has one entry per time", True)
    for got, want in zip(steps, expected["steps"]):
        t = want["time"]
        if not isinstance(got, dict):
            check(f"step {t} is an object", False, repr(got))
            continue
        check(
            f"step {t} time",
            isinstance(got.get("time"), (int, float)) and got.get("time") == t,
            f"got {got.get('time')!r}",
        )
        for kind in ("entered", "exited"):
            value = got.get(kind)
            problem = pair_list_problem(value)
            check(f"step {t} {kind} well-formed", problem is None, problem or "")
            check(
                f"step {t} {kind} matches",
                value == want[kind],
                f"got {value!r}, expected {want[kind]!r}",
            )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--kit", required=True, type=Path)
    args = parser.parse_args()
    sandbox = args.sandbox.resolve()
    kit = args.kit.resolve()

    scene = json.loads((TASK_DIR / "truth" / "scene.json").read_text(encoding="utf-8"))
    expected, margin = truth(scene)
    report: dict = {"pass": False, "checks": [], "expected": expected}
    report["checks"].append(
        {
            "name": "scene margin",
            "pass": margin >= SCENE_MARGIN,
            "detail": f"min |AABB separation| {margin:.4f}",
        }
    )

    solution = sandbox / "solution.py"
    python = kit / "venv" / "bin" / "python"
    if not solution.is_file():
        report["checks"].append(
            {"name": "solution.py exists", "pass": False, "detail": str(solution)}
        )
        print(json.dumps(report, indent=2))
        return 1
    env = dict(os.environ)
    env.update(
        {
            "OVSQ_SDK": str(kit / "sdk"),
            "OVSQ_PYTHON": str(python),
            "OVSTAGE_DIR": str(kit / "venv" / "lib" / "python3.12" / "site-packages" / "ovstage"),
        }
    )
    try:
        run = subprocess.run(
            [str(python), "solution.py"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        report["checks"].append(
            {"name": "solution runs", "pass": False, "detail": f"timeout {TIMEOUT_SECONDS}s"}
        )
        print(json.dumps(report, indent=2))
        return 1
    ran = run.returncode == 0
    report["checks"].append(
        {
            "name": "solution runs",
            "pass": ran,
            "detail": f"exit {run.returncode}; stderr tail: {run.stderr[-1500:]}",
        }
    )
    output = parse_output(run.stdout) if ran else None
    if ran and output is None:
        report["checks"].append(
            {"name": "stdout has JSON", "pass": False, "detail": run.stdout[-1500:]}
        )
    if output is not None:
        report["checks"].extend(compare(output, expected))
        report["output"] = output
    report["pass"] = ran and output is not None and all(c["pass"] for c in report["checks"])
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
