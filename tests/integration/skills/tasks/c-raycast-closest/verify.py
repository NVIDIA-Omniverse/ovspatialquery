# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify c-raycast-closest: build the sandbox CMake project, run it, compare with analytic truth.

usage: python verify.py --sandbox <dir> [--kit <kit>]

Truth comes from scene_truth.json and files/rays.json beside this script (never from the
sandbox copies), intersected analytically. Standard library and NumPy only.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOTAL_TIMEOUT_S = 300.0
DISTANCE_TOL = 1.0e-3
POSITION_TOL = 1.0e-3


def ray_sphere(origin, direction, center, radius):
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc)) - radius * radius
    if c <= 0.0:
        return 0.0  # origin inside the solid sphere
    disc = b * b - c
    if disc < 0.0:
        return None
    t = -b - math.sqrt(disc)
    return t if t >= 0.0 else None


def ray_box(origin, direction, center, half, rot_z_deg):
    a = math.radians(rot_z_deg)
    ca, sa = math.cos(a), math.sin(a)
    # world -> local is the inverse rotation about +Z.
    inv = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    o = inv @ (origin - center)
    d = inv @ direction
    t_near, t_far = -math.inf, math.inf
    for axis in range(3):
        if abs(d[axis]) < 1.0e-15:
            if abs(o[axis]) > half[axis]:
                return None
            continue
        t1 = (-half[axis] - o[axis]) / d[axis]
        t2 = (half[axis] - o[axis]) / d[axis]
        t_near = max(t_near, min(t1, t2))
        t_far = min(t_far, max(t1, t2))
    if t_near > t_far or t_far < 0.0:
        return None
    return max(t_near, 0.0)


def truth_hits(truth, rays):
    expected = []
    for ray in rays:
        origin = np.array(ray["origin"], dtype=np.float64)
        direction = np.array(ray["direction"], dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        best = None
        for obstacle in truth["obstacles"]:
            center = np.array(obstacle["center"], dtype=np.float64)
            if obstacle["type"] == "sphere":
                t = ray_sphere(origin, direction, center, obstacle["radius"])
            else:
                t = ray_box(
                    origin, direction, center, np.array(obstacle["half"]), obstacle["rot_z_deg"]
                )
            if t is None or t > ray["max_distance"]:
                continue
            if best is None or t < best[1]:
                best = (obstacle["path"], t)
        if best is None:
            expected.append({"path": None, "distance": None, "position": None})
        else:
            position = origin + best[1] * direction
            expected.append({"path": best[0], "distance": best[1], "position": position.tolist()})
    return expected


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def run_step(name, command, cwd, env, deadline, checks):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        checks.append({"name": name, "ok": False, "detail": "timed out before start"})
        return None
    try:
        proc = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True, timeout=remaining
        )
    except subprocess.TimeoutExpired:
        checks.append(
            {"name": name, "ok": False, "detail": f"timed out after {TOTAL_TIMEOUT_S:.0f}s budget"}
        )
        return None
    except OSError as error:
        checks.append({"name": name, "ok": False, "detail": str(error)})
        return None
    ok = proc.returncode == 0
    detail = f"exit {proc.returncode}"
    if not ok:
        detail += ": " + (proc.stderr[-1500:] + proc.stdout[-500:]).strip()
    checks.append({"name": name, "ok": ok, "detail": detail})
    return proc if ok else None


def compare(expected, actual, checks):
    if not isinstance(actual, dict) or not isinstance(actual.get("hits"), list):
        checks.append(
            {"name": "output shape", "ok": False, "detail": "expected an object with a 'hits' list"}
        )
        return
    hits = actual["hits"]
    checks.append(
        {
            "name": "hit count",
            "ok": len(hits) == len(expected),
            "detail": f"expected {len(expected)}, got {len(hits)}",
        }
    )
    for index, want in enumerate(expected):
        name = f"ray {index}"
        if index >= len(hits):
            checks.append({"name": name, "ok": False, "detail": "missing entry"})
            continue
        got = hits[index]
        if not isinstance(got, dict) or not {"path", "distance", "position"} <= set(got):
            checks.append(
                {
                    "name": name,
                    "ok": False,
                    "detail": f"entry must have path/distance/position: {got!r}",
                }
            )
            continue
        if want["path"] is None:
            ok = got["path"] is None and got["distance"] is None and got["position"] is None
            checks.append(
                {"name": name, "ok": ok, "detail": f"expected miss (all null), got {got!r}"}
            )
            continue
        problems = []
        if got["path"] != want["path"]:
            problems.append(f"path {got['path']!r} != {want['path']!r}")
        if not is_number(got["distance"]) or abs(got["distance"] - want["distance"]) > DISTANCE_TOL:
            problems.append(f"distance {got['distance']!r} != {want['distance']:.6f}")
        position = got["position"]
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not all(is_number(v) for v in position)
            or max(abs(a - b) for a, b in zip(position, want["position"])) > POSITION_TOL
        ):
            problems.append(f"position {position!r} != {[round(v, 6) for v in want['position']]}")
        checks.append(
            {
                "name": name,
                "ok": not problems,
                "detail": "; ".join(problems)
                if problems
                else f"{want['path']} at {want['distance']:.6f}",
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--kit")
    parser.add_argument("--prebuilt-executable", type=Path)
    args = parser.parse_args()
    if args.prebuilt_executable:
        args.prebuilt_executable = args.prebuilt_executable.resolve()

    sandbox = Path(args.sandbox).resolve()
    env = dict(os.environ)
    if args.kit:
        kit = Path(args.kit).resolve()
        env["OVSQ_SDK"] = str(kit / "sdk")
        env["OVSQ_PYTHON"] = str(kit / "venv" / "bin" / "python")
        env["OVSTAGE_DIR"] = str(kit / "venv" / "lib" / "python3.12" / "site-packages" / "ovstage")
    for name in ("OVSQ_SDK", "OVSTAGE_DIR"):
        if not env.get(name):
            print(
                json.dumps(
                    {
                        "pass": False,
                        "checks": [{"name": "env", "ok": False, "detail": f"{name} unset"}],
                    }
                )
            )
            return 2

    truth = json.loads((HERE / "scene_truth.json").read_text())
    rays = json.loads((HERE / "files" / "rays.json").read_text())["rays"]
    expected = truth_hits(truth, rays)

    checks = []
    deadline = time.monotonic() + TOTAL_TIMEOUT_S
    if not (sandbox / "CMakeLists.txt").is_file():
        checks.append({"name": "CMakeLists.txt", "ok": False, "detail": "missing at sandbox root"})
    else:
        build = sandbox / "build"
        configure = [
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
        ]
        if args.prebuilt_executable or (
            run_step("configure", configure, sandbox, env, deadline, checks)
            and run_step("build", ["cmake", "--build", "build"], sandbox, env, deadline, checks)
        ):
            executable = args.prebuilt_executable or build / "solution"
            if not executable.is_file():
                checks.append(
                    {"name": "executable", "ok": False, "detail": "build/solution not produced"}
                )
            else:
                proc = run_step("run", [str(executable)], sandbox, env, deadline, checks)
                if proc is not None:
                    try:
                        actual = json.loads(proc.stdout)
                    except ValueError as error:
                        checks.append(
                            {
                                "name": "json",
                                "ok": False,
                                "detail": f"{error}: {proc.stdout[:500]!r}",
                            }
                        )
                    else:
                        compare(expected, actual, checks)

    passed = bool(checks) and all(check["ok"] for check in checks)
    print(json.dumps({"pass": passed, "checks": checks}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
