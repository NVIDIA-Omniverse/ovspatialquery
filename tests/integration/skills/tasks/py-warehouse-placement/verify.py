# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify py-warehouse-placement: run solution.py and compare with analytic truth.

Placements: separating-axis test of each yawed pallet box against every obstacle AABB.
Forklift path: the unrotated forklift box swept along each segment, as a ray against each
obstacle AABB grown by the forklift half extents (Minkowski sum).

Usage: python verify.py --sandbox <dir> --kit <kit>
Standard library + NumPy only; never imports ovspatialquery or ovstage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 300
DIST_TOL = 1.0e-3  # the task asks for millimetre accuracy


def obb_aabb_overlap(center, half, yaw_deg, lo, hi) -> bool:
    """Separating-axis test of a box yawed about +Z against an axis-aligned box (touching counts)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    obb_axes = [np.array([c, s, 0.0]), np.array([-s, c, 0.0]), np.array([0.0, 0.0, 1.0])]
    box_axes = list(np.eye(3))
    box_c = (np.asarray(lo) + np.asarray(hi)) / 2.0
    box_h = (np.asarray(hi) - np.asarray(lo)) / 2.0
    axes = obb_axes + box_axes
    for a in obb_axes:
        for b in box_axes:
            cross = np.cross(a, b)
            if np.linalg.norm(cross) > 1e-9:
                axes.append(cross / np.linalg.norm(cross))
    d = box_c - np.asarray(center, dtype=float)
    for axis in axes:
        r1 = sum(half[i] * abs(axis @ obb_axes[i]) for i in range(3))
        r2 = sum(box_h[i] * abs(axis[i]) for i in range(3))
        if abs(axis @ d) > r1 + r2:
            return False
    return True


def sweep_entry(start, unit, length, half, lo, hi) -> float | None:
    """First t in [0, length] at which the axis-aligned box centred at start + t*unit touches [lo, hi]."""
    t0, t1 = 0.0, length
    for axis in range(3):
        a, b = lo[axis] - half[axis], hi[axis] + half[axis]
        if abs(unit[axis]) < 1e-15:
            if start[axis] < a or start[axis] > b:
                return None
            continue
        u, v = (a - start[axis]) / unit[axis], (b - start[axis]) / unit[axis]
        t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
        if t0 > t1:
            return None
    return t0


def truth(scene: dict, requests: dict) -> dict:
    obstacles = scene["obstacles"]
    pallet_half = [v / 2.0 for v in requests["pallet"]["size"]]
    placements = []
    for placement in requests["placements"]:
        x, y = placement["position"]
        center = [x, y, pallet_half[2]]
        blocking = sorted(
            o["path"]
            for o in obstacles
            if obb_aabb_overlap(center, pallet_half, placement["yaw_deg"], o["min"], o["max"])
        )
        placements.append({"id": placement["id"], "free": not blocking, "blocking": blocking})
    forklift = requests["forklift"]
    half = [v / 2.0 for v in forklift["size"]]
    z = float(forklift["center_height"])
    path = []
    for index, segment in enumerate(forklift["path"]):
        start = np.array([*segment["from"], z], dtype=float)
        delta = np.array([*segment["to"], z], dtype=float) - start
        length = float(np.linalg.norm(delta))
        unit = delta / length
        best = None
        for o in obstacles:
            t = sweep_entry(start, unit, length, half, o["min"], o["max"])
            if t is not None and (best is None or t < best[0]):
                best = (t, o["path"])
        path.append(
            {
                "segment": index,
                "hit": None if best is None else best[1],
                "distance": None if best is None else best[0],
            }
        )
    return {"placements": placements, "path": path}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare(expected: dict, output: object, non_obstacles: set[str]) -> list[dict]:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    if not isinstance(output, dict):
        check("output is an object", False, f"got {type(output).__name__}")
        return checks

    placements = output.get("placements")
    if not isinstance(placements, list):
        check("placements is a list", False, f"got {type(placements).__name__}")
    else:
        want_all = expected["placements"]
        check(
            "placement count",
            len(placements) == len(want_all),
            f"{len(placements)} vs {len(want_all)}",
        )
        for index, want in enumerate(want_all):
            name = f"placement {want['id']}"
            got = placements[index] if index < len(placements) else None
            if not isinstance(got, dict):
                check(name, False, "missing or not an object")
                continue
            problems = []
            if got.get("id") != want["id"]:
                problems.append(
                    f"id {got.get('id')!r} at position {index}, expected {want['id']!r}"
                )
            if got.get("free") is not want["free"]:
                problems.append(f"free {got.get('free')!r} != {want['free']!r}")
            blocking = got.get("blocking")
            if blocking != want["blocking"]:
                detail = f"blocking {blocking!r} != {want['blocking']!r}"
                if isinstance(blocking, list) and non_obstacles & set(map(str, blocking)):
                    detail += " (floor/markings are not obstacles)"
                elif isinstance(blocking, list) and sorted(blocking) == want["blocking"]:
                    detail += " (not sorted)"
                problems.append(detail)
            check(name, not problems, "; ".join(problems))

    path = output.get("path")
    if not isinstance(path, list):
        check("path is a list", False, f"got {type(path).__name__}")
        return checks
    want_all = expected["path"]
    check("segment count", len(path) == len(want_all), f"{len(path)} vs {len(want_all)}")
    for index, want in enumerate(want_all):
        name = f"segment {index}"
        got = path[index] if index < len(path) else None
        if not isinstance(got, dict):
            check(name, False, "missing or not an object")
            continue
        problems = []
        if got.get("segment") != index or isinstance(got.get("segment"), bool):
            problems.append(f"segment {got.get('segment')!r} at position {index}")
        if "hit" not in got or "distance" not in got:
            problems.append("needs both 'hit' and 'distance' keys")
        elif want["hit"] is None:
            if got["hit"] is not None or got["distance"] is not None:
                problems.append(
                    f"expected hit=null, distance=null; got {got['hit']!r}, {got['distance']!r}"
                )
        else:
            if got["hit"] != want["hit"]:
                problems.append(f"hit {got['hit']!r} != {want['hit']!r}")
            distance = got["distance"]
            if not _is_number(distance) or abs(distance - want["distance"]) > DIST_TOL:
                problems.append(f"distance {distance!r} != {want['distance']:.6f}")
        check(name, not problems, "; ".join(problems))
    return checks


def run_solution(sandbox: Path, kit: Path) -> tuple[dict | None, str]:
    solution = sandbox / "solution.py"
    if not solution.is_file():
        return None, "solution.py is missing"
    python = kit / "venv" / "bin" / "python"
    env = dict(os.environ)
    env.update(
        {
            "OVSQ_SDK": str(kit / "sdk"),
            "OVSQ_PYTHON": str(python),
            "OVSTAGE_DIR": str(kit / "venv" / "lib" / "python3.12" / "site-packages" / "ovstage"),
        }
    )
    try:
        completed = subprocess.run(
            [str(python), "solution.py"],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, f"solution.py timed out after {TIMEOUT_S}s"
    if completed.returncode != 0:
        return None, f"solution.py exited {completed.returncode}: {completed.stderr[-2000:]}"
    try:
        return json.loads(completed.stdout), ""
    except ValueError as failure:
        return None, f"{failure}: {completed.stdout[-1000:]!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    args = parser.parse_args()
    # Truth comes from the task directory, never from the sandbox copy the agent could edit.
    scene = json.loads((HERE / "truth" / "scene.json").read_text(encoding="utf-8"))
    requests = json.loads((HERE / "files" / "requests.json").read_text(encoding="utf-8"))
    expected = truth(scene, requests)
    output, error = run_solution(args.sandbox.resolve(), args.kit.resolve())
    if output is None:
        checks = [{"name": "run solution", "pass": False, "detail": error}]
    else:
        checks = compare(expected, output, {o["path"] for o in scene["non_obstacles"]})
    passed = bool(checks) and all(c["pass"] for c in checks)
    print(json.dumps({"pass": passed, "checks": checks}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
