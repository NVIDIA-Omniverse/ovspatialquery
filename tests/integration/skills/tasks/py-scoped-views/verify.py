# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify py-scoped-views: run solution.py and compare with analytic per-consumer ray truth.

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
# Runs solution.py as `python solution.py` would, logging to argv[1] what the solution really
# did: the path lists each Ovstage population and view was created with, as the solution
# passed them to `create_population` / `create_view` (int path values are turned into strings
# through the stage's own path dictionary, after the library accepted them), and the target
# of every raycast it ran, sync or through a plan. A solution that bypasses those Python
# methods is not recorded and fails the scope check.
LAUNCHER = r"""
import json, os, runpy, sys
import ovspatialquery.ovstage as ovstage_adapter
import ovspatialquery.raw as raw

log, bind = open(sys.argv[1], "a", encoding="utf-8", buffering=1), raw.bind_exports
SYNC_RAYCASTS = (
    "ovspatialquery_raycast_closest_sync",
    "ovspatialquery_raycast_any_sync",
    "ovspatialquery_raycast_multiple_sync",
)
VIEW = 1


def _prim_path(value):
    # An object ID's value as the path dictionary takes it: an int on Ovstage 0.3, a
    # PrimPath on 0.4.
    import ovstage

    handle = getattr(ovstage, "PrimPath", None)
    value = int(getattr(value, "raw_id", value))
    return value if handle is None else handle(value)

def write(entry):
    log.write(json.dumps(entry) + "\n")


def target_of(ref):
    return {"kind": int(ref.kind), "handle": int(ref.handle.view)}


def deref(argument):
    return argument.contents if hasattr(argument, "contents") else argument._obj


def record(name, function):
    def call(*args):
        result = function(*args)
        if name in SYNC_RAYCASTS:
            write({"call": "raycast", "rays": 1, **target_of(deref(args[1]).common.target.target)})
        elif name == "ovspatialquery_create_raycast_plan":
            plan = int(deref(args[2]).value)
            write({"call": "plan", "plan": plan, **target_of(deref(args[1]).target)})
        elif name == "ovspatialquery_enqueue_raycast_plan":
            plan = int(getattr(args[1], "value", args[1]))
            write({"call": "launch", "plan": plan, "rays": int(deref(args[2]).query_count)})
        return result

    return call


raw.bind_exports = lambda library, specs: {n: record(n, f) for n, f in bind(library, specs).items()}


def paths(adapter, values):
    dictionary = None
    names = []
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            if dictionary is None:
                dictionary = adapter.ovstage_module.PathDictionary(adapter.stage)
            value = dictionary.path_to_string(_prim_path(value))
        names.append(str(value))
    return sorted(set(names))


def scoped(method, adapter_of, kind):
    def wrapper(self, *args, include_paths=(), exclude_paths=(), **kwargs):
        include, exclude = list(include_paths), list(exclude_paths)
        made = method(self, *args, include_paths=include, exclude_paths=exclude, **kwargs)
        adapter = adapter_of(self)
        write({"call": kind, "handle": int(made.handle),
               "include": paths(adapter, include), "exclude": paths(adapter, exclude)})
        return made

    return wrapper


ovstage_adapter.OvstageAdapter.create_population = scoped(
    ovstage_adapter.OvstageAdapter.create_population, lambda a: a, "population")
ovstage_adapter.OvstagePopulation.create_view = scoped(
    ovstage_adapter.OvstagePopulation.create_view, lambda p: p.adapter, "view")
sys.argv = ["solution.py"]
runpy.run_path(os.path.abspath("solution.py"), run_name="__main__")
"""


def scope_check(calls: list[dict], consumers: dict, ray_count: int) -> dict:
    """Each consumer queries, with every ray, a view created with exactly its own lists.

    A view matches a consumer when its includes equal the consumer's and its excludes are
    the consumer's, optionally plus `never` paths (harmless once the population drops
    them). Every raycast, sync or through a plan, must target such a view: a query on the
    population, on a view with another tool's lists, or on a fixed path-list view fails.
    """
    never = set(consumers["never"])
    views = {c["handle"]: c for c in calls if c["call"] == "view"}
    populations = [c for c in calls if c["call"] == "population"]
    plans = {c["plan"]: c for c in calls if c["call"] == "plan"}
    rays_by_view: dict[int, int] = {}
    stray: list[str] = []
    for c in calls:
        target = plans.get(c.get("plan")) if c["call"] == "launch" else c
        if c["call"] not in ("raycast", "launch") or target is None:
            continue
        if target["kind"] != 1:
            stray.append(f"{c['rays']} ray(s) on a population")
            continue
        rays_by_view[target["handle"]] = rays_by_view.get(target["handle"], 0) + c["rays"]

    def owner(view: dict) -> str | None:
        for name, lists in consumers["consumers"].items():
            exclude = set(view["exclude"])
            if set(view["include"]) == set(lists["include"]) and set(lists["exclude"]) <= exclude:
                if exclude - set(lists["exclude"]) <= never:
                    return name
        return None

    for handle, count in rays_by_view.items():
        view = views.get(handle)
        if view is None or owner(view) is None:
            lists = (view["include"], view["exclude"]) if view else "not created with scope lists"
            stray.append(f"{count} ray(s) on a view matching no tool: {lists}")
    missing = [
        name
        for name in consumers["consumers"]
        if not any(
            owner(view) == name and rays_by_view.get(handle, 0) >= ray_count
            for handle, view in views.items()
        )
    ]
    ok = len(populations) == 1 and not missing and not stray
    detail = (
        f"{len(populations)} population(s); tools without a view of their own lists that "
        f"every ray queried: {missing}; raycasts elsewhere: {stray[:5]}"
    )
    return {
        "name": "one population, each tool's rays on a view with its lists",
        "pass": ok,
        "detail": detail,
    }


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


def _elements(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split("/") if part)


def in_scope(path: str, include: list[str], exclude: list[str]) -> bool:
    """The task's rule: walking up from the prim itself, the first listed path decides.

    Paths compare by whole names, a path in both lists counts as excluded, and an empty
    include list stands for the root.
    """
    includes = {_elements(p) for p in include} if include else {()}
    excludes = {_elements(p) for p in exclude}
    elements = _elements(path)
    for depth in range(len(elements), -1, -1):
        prefix = elements[:depth]
        if prefix in excludes:
            return False
        if prefix in includes:
            return True
    return False


def selected(scene: dict, consumers: dict) -> dict[str, list[dict]]:
    """Shapes each consumer may hit: in its own scope, and outside every `never` subtree."""
    shapes = [shape for shape in scene["shapes"] if in_scope(shape["path"], [], consumers["never"])]
    return {
        name: [s for s in shapes if in_scope(s["path"], lists["include"], lists["exclude"])]
        for name, lists in consumers["consumers"].items()
    }


def closest(shapes: list[dict], ray: dict) -> dict | None:
    origin = np.asarray(ray["origin"], dtype=float)
    direction = np.asarray(ray["direction"], dtype=float)
    unit = direction / np.linalg.norm(direction)
    best = None
    for shape in shapes:
        t = intersect(shape, origin, unit, float(ray["max_distance"]))
        if t is not None and (best is None or t < best["distance"]):
            best = {"prim": shape["path"], "distance": float(t)}
    return best


def truth(scene: dict, consumers: dict, rays: list[dict]) -> dict[str, list[dict | None]]:
    return {
        name: [closest(shapes, ray) for ray in rays]
        for name, shapes in selected(scene, consumers).items()
    }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare(expected: dict[str, list[dict | None]], output: object) -> list[dict]:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        return ok

    if not isinstance(output, dict):
        check("output is an object", False, f"got {type(output).__name__}")
        return checks
    check(
        "one entry per consumer",
        set(output) == set(expected),
        f"{sorted(output)} vs {sorted(expected)}",
    )
    for name, want_hits in expected.items():
        hits = output.get(name)
        if not isinstance(hits, list) or len(hits) != len(want_hits):
            count = len(hits) if isinstance(hits, list) else type(hits).__name__
            check(f"{name}: one hit per ray", False, f"{count} vs {len(want_hits)}")
            continue
        wrong = []
        for index, (want, got) in enumerate(zip(want_hits, hits)):
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
                wrong.append({"ray": index, "expected": want, "got": got})
        check(
            f"{name}: every ray's closest hit",
            not wrong,
            f"{len(wrong)} of {len(want_hits)} wrong; first: {wrong[:5]}" if wrong else "",
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
    consumers = json.loads((HERE / "files" / "consumers.json").read_text(encoding="utf-8"))
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
    # The stage is published once, and each consumer's selection follows its subtrees
    # inside ovspatialquery: one population, and each tool's rays on a view created with
    # that tool's path lists, not a fixed list of prims or hits discarded afterwards.
    checks.append(scope_check(calls, consumers, len(rays)))
    checks.extend(compare(truth(scene, consumers, rays), output))
    return finish()


if __name__ == "__main__":
    sys.exit(main())
