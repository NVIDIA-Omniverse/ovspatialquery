# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify py-filter-data-rays: run solution.py and compare with analytic filtered ray truth.

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
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 300
DIST_TOL = 1.0e-4
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


def intersect(shape: dict, origin: np.ndarray, unit: np.ndarray, t_max: float, grow: float = 0.0):
    """First entry parameter of a unit world ray into an axis-aligned box or a sphere, or None.

    `grow` widens (positive) or narrows (negative) the shape; gen_scene.py uses it to keep
    only rays whose answer does not depend on a grazing contact.
    """
    center = np.asarray(shape["translate"], dtype=float)
    if shape["kind"] == "box":
        half = np.asarray(shape["scale"], dtype=float) * shape["size"] / 2.0 + grow
        t0, t1 = 0.0, t_max
        for axis in range(3):
            o = origin[axis] - center[axis]
            if abs(unit[axis]) < 1e-15:
                if abs(o) > half[axis]:
                    return None
                continue
            a = (-half[axis] - o) / unit[axis]
            b = (half[axis] - o) / unit[axis]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if t0 > t1:
                return None
        return t0
    radius = shape["radius"] + grow
    o = origin - center
    b = float(o @ unit)
    c = float(o @ o) - radius * radius
    disc = b * b - c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    for t in (-b - root, -b + root):
        if 0.0 <= t <= t_max:
            return t
    return None


def accepts(query_words: list[int], shape_words: list[int]) -> bool:
    """PhysX's built-in scene-query filter: zero query data accepts all, else any shared bit."""
    if not any(query_words):
        return True
    return any(q & s for q, s in zip(query_words, shape_words))


def truth(scene: dict, rays: list[dict]) -> list[dict | None]:
    expected: list[dict | None] = []
    for ray in rays:
        origin = np.asarray(ray["origin"], dtype=float)
        direction = np.asarray(ray["direction"], dtype=float)
        unit = direction / np.linalg.norm(direction)
        best = None
        for shape in scene["shapes"]:
            if not accepts(ray["filter_data"], shape["filter_data"]):
                continue
            t = intersect(shape, origin, unit, float(ray["max_distance"]))
            if t is not None and (best is None or t < best["distance"]):
                best = {"prim": shape["path"], "distance": float(t)}
        expected.append(best)
    return expected


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare(expected: list[dict | None], output: object, rays: list[dict]) -> list[dict]:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        return ok

    hits = output.get("hits") if isinstance(output, dict) else None
    if not isinstance(hits, list):
        check("hits is a list", False, f"got {type(hits).__name__}")
        return checks
    if not check("hit count", len(hits) == len(expected), f"{len(hits)} vs {len(expected)}"):
        return checks
    wrong = []
    for index, (want, got) in enumerate(zip(expected, hits)):
        if want is None:
            ok = got is None
        else:
            ok = (
                isinstance(got, dict)
                and got.get("prim") == want["prim"]
                and _is_number(got.get("distance"))
                and abs(got["distance"] - want["distance"]) <= DIST_TOL
            )
        if not ok:
            wrong.append(
                {
                    "ray": index,
                    "filter_data": rays[index]["filter_data"],
                    "expected": want,
                    "got": got,
                }
            )
    check(
        "every ray's closest accepted hit",
        not wrong,
        f"{len(wrong)} of {len(expected)} wrong; first: {wrong[:5]}" if wrong else "",
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    args = parser.parse_args()
    sandbox = args.sandbox.resolve()
    kit = args.kit.resolve()
    scene = json.loads((HERE / "truth" / "scene.json").read_text(encoding="utf-8"))
    rays = json.loads((HERE / "files" / "rays.json").read_text(encoding="utf-8"))["rays"]
    checks: list[dict] = []

    def finish() -> int:
        passed = bool(checks) and all(entry["pass"] for entry in checks)
        print(json.dumps({"pass": passed, "checks": checks}, indent=2))
        return 0 if passed else 1

    solution = sandbox / "solution.py"
    if not solution.is_file():
        checks.append({"name": "solution.py exists", "pass": False, "detail": str(solution)})
        return finish()
    python = kit / "venv" / "bin" / "python"
    env = dict(os.environ)
    env.update(
        {
            "OVSQ_SDK": str(kit / "sdk"),
            "OVSQ_PYTHON": str(python),
            "OVSTAGE_DIR": str(kit / "venv" / "lib" / "python3.12" / "site-packages" / "ovstage"),
        }
    )
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "calls.jsonl"
        log.touch()
        try:
            completed = subprocess.run(
                [str(python), "-c", LAUNCHER, str(log)],
                cwd=sandbox,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            detail = f"timed out after {TIMEOUT_S} s"
            checks.append({"name": "runs", "pass": False, "detail": detail})
            return finish()
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    if completed.returncode != 0:
        detail = f"exited {completed.returncode}: {completed.stderr[-2000:]}"
        checks.append({"name": "runs", "pass": False, "detail": detail})
        return finish()
    try:
        output = json.loads(completed.stdout)
    except ValueError as failure:
        checks.append({"name": "output", "pass": False, "detail": f"{failure}"})
        return finish()
    # The filtering must happen inside the query: a closest hit filtered afterwards in
    # Python is another program.
    launches = [call for call in calls if call["call"] == "ovspatialquery_enqueue_raycast_plan"]
    checks.append(
        {
            "name": "filters inside a batched raycast plan",
            "pass": any(call["rays"] == len(rays) and call["query_bits"] for call in launches),
            "detail": "all rays must go in one raycast plan launch with a query_bits column; "
            f"launches seen: {[(call['rays'], call['query_bits']) for call in launches][:10]}",
        }
    )
    checks.extend(compare(truth(scene, rays), output, rays))
    return finish()


if __name__ == "__main__":
    sys.exit(main())
