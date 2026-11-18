# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference solution for py-warehouse-placement.

Obstacles are every supported gprim under /World/Racks, /World/Pallets and /World/Structure,
selected with a path-prefix filter ANDed with a prim-type filter. Each candidate pallet is an
oriented-box overlap-multiple query; each forklift segment is a box sweep. Single queries are
exact by default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import ovstage
from ovspatialquery import BoxSweepQuery, Library, OrientedBox, OverlapQuery

HERE = Path(__file__).resolve().parent
OBSTACLE_ROOTS = ("/World/Racks/", "/World/Pallets/", "/World/Structure/")
GPRIM_TYPES = ("Cube", "Sphere", "Capsule", "Mesh")
CAPACITY = 64


def _prim_path(value):
    """An object ID's value as the path dictionary takes it: an int on Ovstage 0.3, a
    PrimPath on 0.4."""
    import ovstage

    handle = getattr(ovstage, "PrimPath", None)
    value = int(getattr(value, "raw_id", value))
    return value if handle is None else handle(value)


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


def yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    """Unit (x, y, z, w) rotation about +Z, counter-clockwise seen from above."""
    half = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def main() -> None:
    usda = (HERE / "warehouse.usda").read_text(encoding="utf-8")
    requests = json.loads((HERE / "requests.json").read_text(encoding="utf-8"))
    # The type predicate keeps the AisleA/AisleB group Xforms out (they fail NOT_SUPPORTED);
    # the path prefixes keep the floor and lane markings out.
    obstacles = ovstage.Filter(
        [
            ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, OBSTACLE_ROOTS),
            ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, GPRIM_TYPES),
        ]
    )

    with ovstage.Stage("warehouse") as stage:
        # Time is positional: Ovstage 0.2 names it time_code, 0.3 names it time (seconds).
        ovstage.population.open_usd_from_string_with_desc(
            stage, usda, 1, 0.0, physics_desc(ovstage)
        )
        stage.advance_write_floor(ordinal=1).wait()
        stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        stage.advance_write_floor(ordinal=2).wait()
        names = ovstage.PathDictionary(stage)

        # The instance closes the population before the stage is destroyed.
        with Library.open(require_ovstage=True) as library, library.instance() as instance:
            population = instance.ovstage(ovstage, stage).create_population(obstacles)
            with population.enqueue_set(2) as operation:
                operation.wait()
                if operation.fetch_population_update().object_count == 0:
                    raise RuntimeError("the obstacle filter selected nothing")
            target = population.target

            pallet_half = tuple(s / 2.0 for s in requests["pallet"]["size"])
            placements = []
            for placement in requests["placements"]:
                x, y = placement["position"]
                result = target.overlap_multiple(
                    OverlapQuery(
                        OrientedBox(
                            center=(x, y, pallet_half[2]),
                            rotation_xyzw=yaw_quaternion(placement["yaw_deg"]),
                            half_extents=pallet_half,
                        )
                    ),
                    CAPACITY,
                )
                if result.truncated:
                    raise RuntimeError("overlap capacity too small")
                blocking = sorted(
                    str(names.path_to_string(_prim_path(hit.object_id))) for hit in result.hits
                )
                placements.append(
                    {"id": placement["id"], "free": not blocking, "blocking": blocking}
                )

            forklift = requests["forklift"]
            forklift_half = tuple(s / 2.0 for s in forklift["size"])
            z = float(forklift["center_height"])
            segments = []
            for index, segment in enumerate(forklift["path"]):
                delta = [b - a for a, b in zip(segment["from"], segment["to"])] + [0.0]
                length = math.sqrt(sum(d * d for d in delta))
                hit = target.sweep_box_closest(
                    BoxSweepQuery(
                        center=(*segment["from"], z),
                        half_extents=forklift_half,
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                        unit_direction=tuple(d / length for d in delta),
                        max_distance=length,
                    )
                ).hit
                segments.append(
                    {"segment": index, "hit": None, "distance": None}
                    if hit is None
                    else {
                        "segment": index,
                        "hit": str(names.path_to_string(_prim_path(hit.object_id))),
                        "distance": hit.distance,
                    }
                )

    print(json.dumps({"placements": placements, "path": segments}))


if __name__ == "__main__":
    main()
