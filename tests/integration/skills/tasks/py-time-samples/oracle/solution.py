# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference solution: publish scene.usda at time codes 0, 10 and 5 and raycast each version."""

from __future__ import annotations

import json
from pathlib import Path

import ovstage

from ovspatialquery import Library, OrdinalSelection, PopulationConfig, RaycastQuery

HERE = Path(__file__).resolve().parent
SCENE = HERE / "scene.usda"
POPULATION_PATHS = ("/World/Rig/Mover", "/World/Wall")
TIME_CODES_PER_SECOND = 24.0  # authored in scene.usda; Ovstage time arguments are seconds


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


def seconds(time_code: float) -> float:
    return time_code / TIME_CODES_PER_SECOND


def main() -> None:
    # The stage block is outermost so every ovspatialquery object closes before it.
    with ovstage.Stage("py-time-samples") as stage:
        # Ordinal 1: USD population at time code 0; ordinal 2: world transforms and extents.
        # The time is positional: Ovstage 0.2 names it time_code, 0.3 names it time.
        ovstage.population.open_usd_with_desc(
            stage, str(SCENE), 1, seconds(0.0), physics_desc(ovstage)
        )
        stage.advance_write_floor(ordinal=1).wait()
        stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        stage.advance_write_floor(ordinal=2).wait()
        dictionary = ovstage.PathDictionary(stage)
        selected = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, POPULATION_PATHS)]
        )

        with (
            Library.open(require_ovstage=True) as library,
            library.instance() as instance,  # closes the population on exit
        ):
            population = instance.ovstage(ovstage, stage).create_population(
                selected, PopulationConfig(retained_ordinal_capacity=2)
            )

            def publish(base: int | None, target: int) -> int:
                operation = (
                    population.enqueue_set(target)
                    if base is None
                    else population.enqueue_update(base, target)
                )
                with operation:
                    operation.wait()
                    update = operation.fetch_population_update()
                if update.object_count != len(POPULATION_PATHS):
                    raise RuntimeError(f"published {update.object_count} objects")
                return update.published_ordinal

            def advance(ordinal: int, time_code: float) -> None:
                ovstage.population.update_from_usd_time(stage, ordinal, seconds(time_code))
                stage.advance_write_floor(ordinal=ordinal).wait()

            def raycast(selection: OrdinalSelection) -> dict:
                result = population.target.raycast_closest(
                    RaycastQuery(
                        origin=(-5.0, 0.0, 0.0),
                        unit_direction=(1.0, 0.0, 0.0),
                        max_distance=100.0,
                        ordinal=selection,
                    )
                )
                if result.hit is None:
                    return {
                        "source_ordinal": result.source_ordinal,
                        "prim_path": None,
                        "distance": None,
                        "position": None,
                    }
                return {
                    "source_ordinal": result.source_ordinal,
                    "prim_path": str(dictionary.path_to_string(_prim_path(result.hit.object_id))),
                    "distance": result.hit.distance,
                    "position": list(result.hit.position),
                }

            first = publish(None, 2)
            advance(3, 10.0)
            second = publish(first, 3)
            head_after_second = raycast(OrdinalSelection.head())
            original_after_second = raycast(OrdinalSelection.retained_exact(first))
            advance(4, 5.0)
            third = publish(second, 4)
            head_after_third = raycast(OrdinalSelection.head())

    print(
        json.dumps(
            {
                "ordinals": {"time_0": first, "time_10": second, "time_5": third},
                "head_after_time_10": head_after_second,
                "original_after_time_10": original_after_second,
                "head_after_time_5": head_after_third,
            }
        )
    )


if __name__ == "__main__":
    main()
