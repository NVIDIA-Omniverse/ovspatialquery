# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference solution: one population scoped away from /World/Editor, a scoped view per consumer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import ovstage

from ovspatialquery import Library, RaycastQuery

HERE = Path(__file__).resolve().parent
FILES = HERE if (HERE / "scene.usda").is_file() else HERE / "files"


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


def main() -> int:
    consumers = json.loads((FILES / "consumers.json").read_text(encoding="utf-8"))
    rays = json.loads((FILES / "rays.json").read_text(encoding="utf-8"))["rays"]
    stage = ovstage.Stage("scoped-views")
    output: dict[str, list[dict | None]] = {}
    try:
        ovstage.population.open_usd_from_string_with_desc(
            stage, (FILES / "scene.usda").read_text(encoding="utf-8"), 1, 0.0, physics_desc(ovstage)
        )
        stage.advance_write_floor(ordinal=1).wait()
        stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        stage.advance_write_floor(ordinal=2).wait()
        dictionary = ovstage.PathDictionary(stage)
        shapes = ovstage.Filter(
            [ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Cube", "Sphere"])]
        )
        with Library.open(require_ovstage=True) as library, library.instance() as instance:
            # `never` leaves the population, so no view can bring it back: a view only
            # narrows its parent's objects. Merging it into each view's excludes would let
            # a deeper include (grasp's /World/Editor/Probe) win.
            population = instance.ovstage(ovstage, stage).create_population(
                shapes, exclude_paths=consumers["never"]
            )
            views = {
                name: population.create_view(
                    ovstage.Filter([]),
                    include_paths=lists["include"],
                    exclude_paths=lists["exclude"],
                )
                for name, lists in consumers["consumers"].items()
            }
            # Views exist before the publication, so they are ready at its ordinal.
            with population.enqueue_set(2) as publish:
                publish.wait()
                update = publish.fetch_population_update()
                if update.object_count != 14 or update.excluded_object_count != 0:
                    raise RuntimeError(f"unexpected publication: {update}")
            for name, view in views.items():
                hits: list[dict | None] = []
                for ray in rays:
                    direction = np.asarray(ray["direction"], dtype=np.float64)
                    unit = direction / np.linalg.norm(direction)
                    hit = view.target.raycast_closest(
                        RaycastQuery(
                            origin=tuple(ray["origin"]),
                            unit_direction=tuple(float(v) for v in unit),
                            max_distance=float(ray["max_distance"]),
                        )
                    ).hit
                    hits.append(
                        None
                        if hit is None
                        else {
                            "prim": str(dictionary.path_to_string(_prim_path(hit.object_id))),
                            "distance": float(hit.distance),
                        }
                    )
                output[name] = hits
    finally:
        stage.destroy()

    json.dump(output, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
