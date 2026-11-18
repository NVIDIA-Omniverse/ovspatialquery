# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference solution: 2D lidar scans through one reusable ovspatialquery raycast plan."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import ovstage

from ovspatialquery import (
    BatchResultField,
    BatchRowStatus,
    HitMode,
    Library,
    QueryPrecision,
    RaycastPlanConfig,
)

HERE = Path(__file__).resolve().parent
SCENE = HERE / "warehouse.usda"
POSES = HERE / "poses.json"
if not SCENE.is_file():
    SCENE, POSES = HERE / "files" / SCENE.name, HERE / "files" / POSES.name


def _prim_path(value):
    """An object ID's value as the path dictionary takes it: an int on Ovstage 0.3, a
    PrimPath on 0.4."""
    import ovstage

    handle = getattr(ovstage, "PrimPath", None)
    value = int(getattr(value, "raw_id", value))
    return value if handle is None else handle(value)


class _ColumnView:
    """One attribute column of an Ovstage 0.4 read group, with the group's prim rows."""

    __slots__ = ("_group", "_column")

    def __init__(self, group, column):
        self._group = group
        self._column = column

    def __getattr__(self, name):
        try:
            return getattr(self._column, name)
        except AttributeError:
            return getattr(self._group, name)


def _read_columns(group):
    """A 0.3 read group holds one attribute; a 0.4 group one column per attribute."""
    columns = getattr(group, "columns", None)
    return (group,) if columns is None else tuple(_ColumnView(group, c) for c in columns)


def _handle_value(handle):
    """An Ovstage token or path as an int: 0.3 returns ints, 0.4 handles."""
    return int(getattr(handle, "raw_id", handle))


def physics_desc(ovstage):
    """What ovspatialquery reads from a stage on Ovstage 0.3 and 0.4: 0.3's ALL domain, or
    0.4's RENDERING, which has no PHYSICS domain, plus every prim's physics,
    physxConvexHullCollision and material:binding properties for convex hulls, their vertex
    limits and physics materials."""
    population = ovstage.population
    domain = ovstage.PopulationDomain
    physics = population.Selector(
        prim_predicate=population.PrimPredicate.all(),
        property_predicate=population.PropertyPredicate.in_namespace(
            "physics", "physxConvexHullCollision", "material:binding"
        ),
    )
    return population.Desc(domains=getattr(domain, "ALL", domain.RENDERING), selectors=[physics])


def cube_paths(stage: ovstage.Stage, ordinal: int) -> list[str]:
    """Every Cube prim path in the loaded stage, sorted (the output's path table)."""
    dictionary = ovstage.PathDictionary(stage)
    path_token = dictionary.intern_token("usd-path")
    cubes = ovstage.Filter([ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Cube"])])
    with stage.query(filter=cubes, attrs=[path_token]) as query:
        query.wait()
        if query.result().total_prim_count == 0:
            return []
        found = set()
        with stage.read_attributes(
            query, [path_token], ovstage.OrdinalRange.latest(ordinal)
        ) as read:
            read.wait()
            for read_group in read.groups():
                try:
                    for group in _read_columns(read_group):
                        if not group.is_delete:
                            found.update(
                                dictionary.path_to_string(_prim_path(token))
                                for token in np.asarray(group.array(0)).reshape(-1)
                            )
                finally:
                    stage.release_group(read_group)
        return sorted(found)


def main() -> int:
    config = json.loads(POSES.read_text(encoding="utf-8"))
    ray_count = int(config["ray_count"])
    step = math.radians(float(config["angular_step_deg"]))
    max_range = float(config["max_range_m"])
    stage = ovstage.Stage("lidar")
    scans: list[dict[str, Any]] = []
    try:
        # Positional time: Ovstage 0.2 names it `time_code`, 0.3 names it `time`.
        ovstage.population.open_usd_from_string_with_desc(
            stage, SCENE.read_text(encoding="utf-8"), 1, 0.0, physics_desc(ovstage)
        )
        stage.advance_write_floor(ordinal=1).wait()
        stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        stage.advance_write_floor(ordinal=2).wait()
        paths = cube_paths(stage, 2)
        index_of = {path: index for index, path in enumerate(paths)}

        # The instance closes its plan and population on exit, before the stage is destroyed.
        with Library.open(require_ovstage=True) as library, library.instance() as instance:
            # Only Cube/Sphere/Capsule/Mesh prims may be in the population: name them exactly.
            selected = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])
            population = instance.ovstage(ovstage, stage).create_population(selected)
            with population.enqueue_set(2) as publish:
                publish.wait()
                published = publish.fetch_population_update()
            if published.object_count != len(paths):
                raise RuntimeError(f"published {published.object_count} of {len(paths)} prims")

            plan = instance.create_raycast_plan(
                RaycastPlanConfig(
                    target=population.target.reference,
                    hit_mode=HitMode.CLOSEST,
                    precision=QueryPrecision.EXACT,  # plans default to BOUNDS_CANDIDATE
                    maximum_query_count=ray_count,
                    maximum_hits_per_query=1,
                    hit_capacity_per_slot=ray_count,
                    result_slot_count=1,
                    result_fields=BatchResultField.DISTANCE,
                )
            )
            dictionary = ovstage.PathDictionary(stage)
            angles = np.arange(ray_count, dtype=np.float64) * step
            max_distances = np.full(ray_count, max_range, dtype=np.float64)
            for pose in config["poses"]:
                yaw = math.radians(float(pose["yaw_deg"]))
                origins = np.tile(
                    np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64), (ray_count, 1)
                )
                directions = np.zeros((ray_count, 3), dtype=np.float32)
                directions[:, 0] = np.cos(yaw + angles)
                directions[:, 1] = np.sin(yaw + angles)
                operation, _ = plan.enqueue(
                    origins=origins,
                    unit_directions=directions,
                    max_distances=max_distances,
                    copy=True,
                )
                # Closing the operation frees the plan's single result slot for the next pose.
                with operation:
                    operation.wait()
                    result = operation.fetch_raycast_batch().to_numpy()
                ranges = result.hit_ranges
                if np.any(ranges.query_status != int(BatchRowStatus.SUCCESS)):
                    raise RuntimeError(f"row failures: {np.unique(ranges.query_status)}")
                if np.any(ranges.truncated != 0):
                    raise RuntimeError("hit capacity truncated the scan")
                distances: list[float | None] = []
                hits: list[int | None] = []
                for ray in range(ray_count):
                    if ranges.hit_counts[ray] == 0:
                        distances.append(None)
                        hits.append(None)
                        continue
                    slot = int(ranges.hit_offsets[ray])
                    path = str(dictionary.path_to_string(_prim_path(result.object_ids[slot])))
                    distances.append(round(float(result.exact.distances[slot]), 6))
                    hits.append(index_of[path])
                scans.append({"distances": distances, "hit_index": hits})
    finally:
        stage.destroy()

    json.dump({"paths": paths, "scans": scans}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
