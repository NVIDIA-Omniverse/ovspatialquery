# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify py-sweep-overlap: run the sandbox's solution.py and compare with analytic truth.

python verify.py --sandbox <dir> --kit <kit>

Truth comes from scene.json (task directory; the analytic boxes and spheres the USD
scene was generated from) and files/queries.json. Exact geometry: oriented boxes and
spheres; a sphere sweep's first contact is the least t in [0, L] at which the moving
sphere touches an object, found by bisection on the convex distance function.
Uses only the standard library and NumPy; never imports ovspatialquery or ovstage.
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
TIMEOUT_SECONDS = 300
DISTANCE_TOLERANCE = 1.0e-4  # metres; task.md promises 1e-4


# ---------------------------------------------------------------- analytic geometry


def _rotation_y(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    # USD xformOp:rotateY, column-vector form: world = R @ local.
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _box(obj: dict, use_bounds: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.array(obj["center"], dtype=np.float64)
    if obj["kind"] == "sphere":
        r = float(obj["radius"])
        return center, np.eye(3), np.array([r, r, r])
    rotation = _rotation_y(float(obj["yaw_degrees"]))
    half = np.array(obj["half_extents"], dtype=np.float64)
    if use_bounds:
        return center, np.eye(3), np.abs(rotation) @ half
    return center, rotation, half


def distance_to_object(obj: dict, point: np.ndarray, use_bounds: bool = False) -> float:
    """Euclidean distance from a point to the solid object (zero inside)."""
    if obj["kind"] == "sphere" and not use_bounds:
        return max(
            float(np.linalg.norm(point - np.array(obj["center"]))) - float(obj["radius"]), 0.0
        )
    center, rotation, half = _box(obj, use_bounds)
    local = rotation.T @ (point - center)
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


def _first_contact(
    obj: dict,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    use_bounds: bool,
) -> float | None:
    def gap(t: float) -> float:
        return distance_to_object(obj, start + t * direction, use_bounds) - radius

    if gap(0.0) <= 0.0:
        return 0.0
    # gap is convex in t: golden-section for its minimum, then bisection on [0, t_min].
    lo, hi = 0.0, length
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = gap(a), gap(b)
    for _ in range(200):
        if fa <= 0.0 or fb <= 0.0:
            break
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = gap(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = gap(b)
    candidates = [(fa, a), (fb, b), (gap(length), length)]
    value, t_min = min(candidates)
    if value > 0.0:
        return None
    lo, hi = 0.0, t_min
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gap(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def sweep_truth(objects: list[dict], sweep: dict, radius: float, use_bounds: bool = False) -> dict:
    start = np.array(sweep["start"], dtype=np.float64)
    end = np.array(sweep["end"], dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    direction = (end - start) / length
    best: tuple[float, str] | None = None
    for obj in objects:
        t = _first_contact(obj, start, direction, length, radius, use_bounds)
        if t is not None and (best is None or t < best[0]):
            best = (t, obj["path"])
    if best is None:
        return {"hit": None, "distance": None}
    return {"hit": best[1], "distance": best[0]}


def overlap_truth(objects: list[dict], sphere: dict, category: str | None = None) -> list[str]:
    center = np.array(sphere["center"], dtype=np.float64)
    return sorted(
        obj["path"]
        for obj in objects
        if (category is None or obj["category"] == category)
        and distance_to_object(obj, center) <= float(sphere["radius"])
    )


# ---------------------------------------------------------------- runner


def _kit_env(kit: Path) -> dict[str, str]:
    env = dict(os.environ)
    sites = sorted((kit / "venv" / "lib").glob("python3*/site-packages/ovstage"))
    env["OVSQ_SDK"] = str(kit / "sdk")
    env["OVSQ_PYTHON"] = str(kit / "venv" / "bin" / "python")
    if sites:
        env["OVSTAGE_DIR"] = str(sites[0])
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--kit", type=Path, default=None)
    args = parser.parse_args()

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        return ok

    def report() -> int:
        passed = bool(checks) and all(item["pass"] for item in checks)
        print(json.dumps({"pass": passed, "checks": checks}, indent=2))
        return 0 if passed else 1

    if args.kit is not None:
        env = _kit_env(args.kit.resolve())
    else:
        env = dict(os.environ)
    python = env.get("OVSQ_PYTHON")
    sandbox = args.sandbox.resolve()
    solution = sandbox / "solution.py"
    if not check("solution.py exists", solution.is_file(), str(solution)):
        return report()
    if not check("OVSQ_PYTHON set", bool(python), "pass --kit or set OVSQ_PYTHON"):
        return report()

    try:
        completed = subprocess.run(
            [python, str(solution)],
            cwd=sandbox,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        check("solution finishes", False, f"timed out after {TIMEOUT_SECONDS} s")
        return report()
    if not check(
        "solution exits 0",
        completed.returncode == 0,
        f"exit {completed.returncode}; stderr tail: {completed.stderr[-2000:]}",
    ):
        return report()
    try:
        output = json.loads(completed.stdout)
    except ValueError as failure:
        check("stdout is JSON", False, f"{failure}; stdout tail: {completed.stdout[-1000:]}")
        return report()
    if not check(
        "output is an object with sweeps and overlaps",
        isinstance(output, dict)
        and isinstance(output.get("sweeps"), list)
        and isinstance(output.get("overlaps"), list),
        f"got keys {sorted(output) if isinstance(output, dict) else type(output).__name__}",
    ):
        return report()

    objects = json.loads((HERE / "scene.json").read_text(encoding="utf-8"))["objects"]
    queries = json.loads((HERE / "files" / "queries.json").read_text(encoding="utf-8"))
    radius = float(queries["sweep_radius"])

    sweeps = output["sweeps"]
    if check(
        "sweep count",
        len(sweeps) == len(queries["sweeps"]),
        f"expected {len(queries['sweeps'])}, got {len(sweeps)}",
    ):
        for index, (sweep, got) in enumerate(zip(queries["sweeps"], sweeps)):
            want = sweep_truth(objects, sweep, radius)
            if not isinstance(got, dict) or "hit" not in got or "distance" not in got:
                check(f"sweep[{index}] shape", False, f"expected {{hit, distance}}, got {got!r}")
                continue
            check(
                f"sweep[{index}] hit",
                got["hit"] == want["hit"],
                f"expected {want['hit']!r}, got {got['hit']!r}",
            )
            if want["distance"] is None:
                check(
                    f"sweep[{index}] distance",
                    got["distance"] is None,
                    f"expected null, got {got['distance']!r}",
                )
            else:
                ok = (
                    isinstance(got["distance"], (int, float))
                    and not isinstance(got["distance"], bool)
                    and abs(float(got["distance"]) - want["distance"]) <= DISTANCE_TOLERANCE
                )
                check(
                    f"sweep[{index}] distance",
                    ok,
                    f"expected {want['distance']:.6f} +/- {DISTANCE_TOLERANCE}, got {got['distance']!r}",
                )

    overlaps = output["overlaps"]
    spheres = queries["overlap_spheres"]
    if check(
        "overlap count",
        len(overlaps) == len(spheres),
        f"expected {len(spheres)}, got {len(overlaps)}",
    ):
        for index, (sphere, got) in enumerate(zip(spheres, overlaps)):
            if not isinstance(got, dict) or "all" not in got or "shelves" not in got:
                check(f"overlap[{index}] shape", False, f"expected {{all, shelves}}, got {got!r}")
                continue
            for key, category in (("all", None), ("shelves", "shelf")):
                want = overlap_truth(objects, sphere, category)
                check(
                    f"overlap[{index}].{key}",
                    got[key] == want,
                    f"expected {want}, got {got[key]!r}",
                )
    return report()


if __name__ == "__main__":
    sys.exit(main())
