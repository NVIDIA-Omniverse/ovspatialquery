# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference solution for py-sweep-overlap.

One Ovstage population holds every crate and shelf; a filtered view of the same
population restricts the second overlap to shelves. Single queries are exact by default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import ovstage
from ovspatialquery import Library, OverlapQuery, Sphere, SphereSweepQuery

HERE = Path(__file__).resolve().parent
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


def gprims_under(*prefixes: str) -> ovstage.Filter:
    """Supported gprims under `prefixes`; the type predicate keeps group Xforms out."""
    return ovstage.Filter(
        [
            ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, prefixes),
            ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, GPRIM_TYPES),
        ]
    )


def main() -> None:
    usda = (HERE / "corridor.usda").read_text(encoding="utf-8")
    queries = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))

    with ovstage.Stage("corridor") as stage:
        # Time is positional: Ovstage 0.2 names it time_code, 0.3 names it time (seconds).
        ovstage.population.open_usd_from_string_with_desc(
            stage, usda, 1, 0.0, physics_desc(ovstage)
        )
        stage.advance_write_floor(ordinal=1).wait()
        stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        stage.advance_write_floor(ordinal=2).wait()
        names = ovstage.PathDictionary(stage)

        # The instance closes the view and population before the stage is destroyed.
        with Library.open(require_ovstage=True) as library, library.instance() as instance:
            population = instance.ovstage(ovstage, stage).create_population(
                gprims_under("/World/Crates/", "/World/Shelves/")
            )
            # A view is ready only after a publication that follows its creation.
            shelf_view = population.create_view(gprims_under("/World/Shelves/"))
            with population.enqueue_set(2) as operation:
                operation.wait()
                if operation.fetch_population_update().object_count == 0:
                    raise RuntimeError("the filter selected nothing")

            radius = float(queries["sweep_radius"])
            sweeps = []
            for sweep in queries["sweeps"]:
                start, end = sweep["start"], sweep["end"]
                delta = [e - s for s, e in zip(start, end)]
                length = math.sqrt(sum(d * d for d in delta))
                hit = population.target.sweep_sphere_closest(
                    SphereSweepQuery(
                        center=tuple(start),
                        radius=radius,
                        unit_direction=tuple(d / length for d in delta),
                        max_distance=length,
                    )
                ).hit
                sweeps.append(
                    {"hit": None, "distance": None}
                    if hit is None
                    else {
                        "hit": str(names.path_to_string(_prim_path(hit.object_id))),
                        "distance": hit.distance,
                    }
                )

            def overlapping(target, query: OverlapQuery) -> list[str]:
                result = target.overlap_multiple(query, CAPACITY)
                if result.truncated:
                    raise RuntimeError("overlap capacity too small")
                return sorted(
                    str(names.path_to_string(_prim_path(hit.object_id))) for hit in result.hits
                )

            overlaps = []
            for sphere in queries["overlap_spheres"]:
                query = OverlapQuery(Sphere(tuple(sphere["center"]), float(sphere["radius"])))
                overlaps.append(
                    {
                        "all": overlapping(population.target, query),
                        "shelves": overlapping(shelf_view.target, query),
                    }
                )

    print(json.dumps({"sweeps": sweeps, "overlaps": overlaps}))


if __name__ == "__main__":
    main()
