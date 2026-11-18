# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify py-raycast-closest: run solution.py and compare with analytic ray/shape truth.

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
DIST_TOL = 1.0e-4
POINT_TOL = 1.0e-4


def _local_to_world(chain: list[dict]) -> np.ndarray:
    """Chain is innermost first; each op is p -> T * S * p."""
    matrix = np.eye(4)
    for op in chain:
        m = np.diag([*op["scale"], 1.0])
        m[:3, 3] = op["translate"]
        matrix = m @ matrix
    return matrix


def _intersect(shape: dict, origin: np.ndarray, unit: np.ndarray, t_max: float) -> float | None:
    """Exact first entry parameter along a unit world ray, or None. Affine maps preserve the ray parameter."""
    inverse = np.linalg.inv(_local_to_world(shape["chain"]))
    o = inverse[:3, :3] @ origin + inverse[:3, 3]
    d = inverse[:3, :3] @ unit
    if shape["kind"] == "box":
        half = shape["size"] / 2.0
        t0, t1 = 0.0, t_max
        for axis in range(3):
            if abs(d[axis]) < 1e-15:
                if o[axis] < -half or o[axis] > half:
                    return None
                continue
            a = (-half - o[axis]) / d[axis]
            b = (half - o[axis]) / d[axis]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if t0 > t1:
                return None
        return t0
    radius = shape["radius"]
    a = float(d @ d)
    b = 2.0 * float(o @ d)
    c = float(o @ o) - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)):
        if 0.0 <= t <= t_max:
            return t
    return None


def truth(scene: dict) -> list[dict]:
    expected = []
    for ray in scene["rays"]:
        origin = np.asarray(ray["origin"], dtype=float)
        direction = np.asarray(ray["direction"], dtype=float)
        unit = direction / np.linalg.norm(direction)
        best = None
        for shape in scene["obstacles"]:
            t = _intersect(shape, origin, unit, float(ray["max_distance"]))
            if t is not None and (best is None or t < best[1]):
                best = (shape["path"], t)
        floor_t = _intersect(scene["floor"], origin, unit, float(ray["max_distance"]))
        expected.append(
            {
                "id": ray["id"],
                "hit": None
                if best is None
                else {
                    "prim": best[0],
                    "distance": best[1],
                    "point": (origin + unit * best[1]).tolist(),
                },
                "floor_distance": floor_t,
            }
        )
    return expected


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare(expected: list[dict], output: object) -> list[dict]:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list):
        check("results is a list", False, f"got {type(results).__name__}")
        return checks
    check("result count", len(results) == len(expected), f"{len(results)} vs {len(expected)}")
    for index, want in enumerate(expected):
        name = f"ray {want['id']}"
        if index >= len(results) or not isinstance(results[index], dict):
            check(name, False, "missing or not an object")
            continue
        got = results[index]
        if got.get("id") != want["id"]:
            check(name, False, f"id {got.get('id')!r} at position {index}, expected {want['id']!r}")
            continue
        if "hit" not in got:
            check(name, False, "no 'hit' key")
            continue
        hit = got["hit"]
        if want["hit"] is None:
            detail = "expected null"
            if want["floor_distance"] is not None:
                detail += " (the floor lies on this ray and must not be queried)"
            check(name, hit is None, f"{detail}; got {hit!r}")
            continue
        if not isinstance(hit, dict):
            check(name, False, f"expected hit on {want['hit']['prim']}; got {hit!r}")
            continue
        problems = []
        if hit.get("prim") != want["hit"]["prim"]:
            problems.append(f"prim {hit.get('prim')!r} != {want['hit']['prim']!r}")
        distance = hit.get("distance")
        if not _is_number(distance) or abs(distance - want["hit"]["distance"]) > DIST_TOL:
            problems.append(f"distance {distance!r} != {want['hit']['distance']:.6f}")
        point = hit.get("point")
        if (
            not isinstance(point, list)
            or len(point) != 3
            or not all(_is_number(v) for v in point)
            or max(abs(a - b) for a, b in zip(point, want["hit"]["point"])) > POINT_TOL
        ):
            problems.append(f"point {point!r} != {[round(v, 6) for v in want['hit']['point']]}")
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
    scene = json.loads((HERE / "truth" / "scene.json").read_text(encoding="utf-8"))
    expected = truth(scene)
    output, error = run_solution(args.sandbox.resolve(), args.kit.resolve())
    if output is None:
        checks = [{"name": "run solution", "pass": False, "detail": error}]
    else:
        checks = compare(expected, output)
    passed = bool(checks) and all(c["pass"] for c in checks)
    print(json.dumps({"pass": passed, "checks": checks}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
