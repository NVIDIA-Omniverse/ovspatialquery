# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify py-explicit-mesh: run solution.py and compare with analytic closest-hit truth.

Usage: python verify.py --sandbox <dir> --kit <kit>
Standard library + NumPy only; never imports ovspatialquery or ovstage. Truth is computed
from this task's own files/ (not the sandbox copy) with Moller-Trumbore for triangles, a
quadratic for spheres and a slab test for yawed boxes.
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
BARY_TOL = 1.0e-4


def yaw_matrix(degrees: float) -> np.ndarray:
    """Counter-clockwise rotation about +Z (Z is up)."""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mesh_world_triangles(scene: dict, instance: dict) -> np.ndarray:
    vertices = np.asarray(scene["mesh"]["vertices"], dtype=float)
    triangles = np.asarray(scene["mesh"]["triangles"], dtype=int)
    world = (yaw_matrix(instance["yaw_degrees"]) @ (instance["scale"] * vertices).T).T + np.asarray(
        instance["translation"], dtype=float
    )
    return world[triangles]  # [T, 3 corners, 3]


def ray_triangles(
    tris: np.ndarray, origin: np.ndarray, unit: np.ndarray, t_max: float
) -> tuple[float, int, tuple[float, float, float]] | None:
    """Moller-Trumbore, front faces only (normal (v1-v0)x(v2-v0) facing the ray)."""
    best = None
    for face, (v0, v1, v2) in enumerate(tris):
        e1, e2 = v1 - v0, v2 - v0
        if float(np.dot(np.cross(e1, e2), unit)) >= 0.0:
            continue
        p = np.cross(unit, e2)
        det = float(np.dot(e1, p))
        if abs(det) < 1e-15:
            continue
        inv = 1.0 / det
        s = origin - v0
        u = float(np.dot(s, p)) * inv
        if u < 0.0 or u > 1.0:
            continue
        q = np.cross(s, e1)
        v = float(np.dot(unit, q)) * inv
        if v < 0.0 or u + v > 1.0:
            continue
        t = float(np.dot(e2, q)) * inv
        if 0.0 <= t <= t_max and (best is None or t < best[0]):
            best = (t, face, (1.0 - u - v, u, v))
    return best


def ray_sphere(
    center: np.ndarray, radius: float, origin: np.ndarray, unit: np.ndarray, t_max: float
):
    oc = origin - center
    b = float(np.dot(oc, unit))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    for t in (-b - root, -b + root):
        if 0.0 <= t <= t_max:
            return t
    return None


def ray_box(box: dict, origin: np.ndarray, unit: np.ndarray, t_max: float):
    rotation = yaw_matrix(box["yaw_degrees"])
    o = rotation.T @ (origin - np.asarray(box["center"], dtype=float))
    d = rotation.T @ unit
    half = np.asarray(box["half_extents"], dtype=float)
    t0, t1 = 0.0, t_max
    for axis in range(3):
        if abs(d[axis]) < 1e-15:
            if abs(o[axis]) > half[axis]:
                return None
            continue
        a = (-half[axis] - o[axis]) / d[axis]
        b = (half[axis] - o[axis]) / d[axis]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return None
    return float(t0)


def all_hits(scene: dict, origin, direction, t_max: float) -> list[dict]:
    """Every object the ray hits within t_max, sorted by distance."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    unit = direction / np.linalg.norm(direction)
    hits = []
    index = 0
    for instance in scene["mesh_instances"]:
        found = ray_triangles(mesh_world_triangles(scene, instance), origin, unit, t_max)
        if found is not None:
            hits.append(
                {
                    "object": index,
                    "kind": "mesh",
                    "face": found[1],
                    "barycentrics": list(found[2]),
                    "distance": found[0],
                }
            )
        index += 1
    for sphere in scene["spheres"]:
        t = ray_sphere(
            np.asarray(sphere["center"], dtype=float), float(sphere["radius"]), origin, unit, t_max
        )
        if t is not None:
            hits.append(
                {
                    "object": index,
                    "kind": "sphere",
                    "face": None,
                    "barycentrics": None,
                    "distance": t,
                }
            )
        index += 1
    for box in scene["boxes"]:
        t = ray_box(box, origin, unit, t_max)
        if t is not None:
            hits.append(
                {"object": index, "kind": "box", "face": None, "barycentrics": None, "distance": t}
            )
        index += 1
    return sorted(hits, key=lambda hit: hit["distance"])


def truth(scene: dict, rays: list[dict]) -> list[dict]:
    expected = []
    for ray in rays:
        hits = all_hits(scene, ray["origin"], ray["direction"], float(ray["max_distance"]))
        expected.append({"id": ray["id"], "hit": hits[0] if hits else None})
    return expected


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
        hit, wanted = got["hit"], want["hit"]
        if wanted is None:
            check(name, hit is None, f"expected null; got {hit!r}")
            continue
        if not isinstance(hit, dict):
            check(
                name,
                False,
                f"expected hit on object {wanted['object']} ({wanted['kind']}); got {hit!r}",
            )
            continue
        problems = []
        if not _is_int(hit.get("object")) or hit.get("object") != wanted["object"]:
            problems.append(f"object {hit.get('object')!r} != {wanted['object']}")
        if hit.get("kind") != wanted["kind"]:
            problems.append(f"kind {hit.get('kind')!r} != {wanted['kind']!r}")
        distance = hit.get("distance")
        if not _is_number(distance) or abs(distance - wanted["distance"]) > DIST_TOL:
            problems.append(f"distance {distance!r} != {wanted['distance']:.6f}")
        if wanted["kind"] == "mesh":
            if not _is_int(hit.get("face")) or hit.get("face") != wanted["face"]:
                problems.append(f"face {hit.get('face')!r} != {wanted['face']}")
            bary = hit.get("barycentrics")
            if (
                not isinstance(bary, list)
                or len(bary) != 3
                or not all(_is_number(v) for v in bary)
                or max(abs(a - b) for a, b in zip(bary, wanted["barycentrics"])) > BARY_TOL
            ):
                problems.append(
                    f"barycentrics {bary!r} != {[round(v, 6) for v in wanted['barycentrics']]}"
                )
        else:
            if hit.get("face", "absent") is not None:
                problems.append(
                    f"face must be null for a {wanted['kind']}; got {hit.get('face', 'absent')!r}"
                )
            if hit.get("barycentrics", "absent") is not None:
                problems.append(
                    f"barycentrics must be null for a {wanted['kind']}; got {hit.get('barycentrics', 'absent')!r}"
                )
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


def load_inputs() -> tuple[dict, list[dict]]:
    scene = json.loads((HERE / "files" / "mesh.json").read_text(encoding="utf-8"))
    rays = json.loads((HERE / "files" / "rays.json").read_text(encoding="utf-8"))["rays"]
    return scene, rays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    args = parser.parse_args()
    scene, rays = load_inputs()
    expected = truth(scene, rays)
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
