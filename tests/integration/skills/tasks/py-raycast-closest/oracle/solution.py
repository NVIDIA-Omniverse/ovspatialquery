# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closest exact hit per ray against /World/Obstacles only, via ovspatialquery on Ovstage."""

from __future__ import annotations

import json
import math
from pathlib import Path

import ovstage
from ovspatialquery import Library, RaycastQuery

HERE = Path(__file__).resolve().parent
GPRIM_TYPES = ("Cube", "Sphere", "Capsule", "Mesh")


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


def main() -> None:
    usda = (HERE / "scene.usda").read_text(encoding="utf-8")
    rays = json.loads((HERE / "rays.json").read_text(encoding="utf-8"))["rays"]
    # Every matched prim must be a supported gprim: a bare PREFIX would also match the
    # /World/Obstacles/Rack Xform and fail with NOT_SUPPORTED, so AND it with a type predicate.
    obstacles = ovstage.Filter(
        [
            ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, ("/World/Obstacles/",)),
            ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, GPRIM_TYPES),
        ]
    )

    results = []
    with ovstage.Stage("py-raycast-closest") as stage:
        # Time is positional: Ovstage 0.2 names it time_code, 0.3 names it time (seconds).
        ovstage.population.open_usd_from_string_with_desc(
            stage, usda, 1, 0.0, physics_desc(ovstage)
        )
        stage.advance_write_floor(ordinal=1).wait()
        # World transforms (parent Xform scale/translate) exist only after the hierarchy pass.
        stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        stage.advance_write_floor(ordinal=2).wait()
        paths = ovstage.PathDictionary(stage)

        # The instance closes the population before the stage is destroyed.
        with Library.open(require_ovstage=True) as library, library.instance() as instance:
            population = instance.ovstage(ovstage, stage).create_population(obstacles)
            with population.enqueue_set(2) as operation:
                operation.wait()
                if operation.fetch_population_update().object_count == 0:
                    raise RuntimeError("the obstacle filter selected nothing")

            for ray in rays:
                direction = [float(v) for v in ray["direction"]]
                length = math.sqrt(sum(v * v for v in direction))
                # Single queries default to QueryPrecision.EXACT.
                hit = population.target.raycast_closest(
                    RaycastQuery(
                        origin=tuple(float(v) for v in ray["origin"]),
                        unit_direction=tuple(v / length for v in direction),
                        max_distance=float(ray["max_distance"]),
                    )
                ).hit
                results.append(
                    {
                        "id": ray["id"],
                        "hit": None
                        if hit is None
                        else {
                            "prim": str(paths.path_to_string(_prim_path(hit.object_id))),
                            "distance": float(hit.distance),
                            "point": [float(v) for v in hit.position],
                        },
                    }
                )

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
