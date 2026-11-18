# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference solution: robot/zone enter and exit events with an Ovstage-backed pair reporter."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ovstage

from ovspatialquery import Library, PairReporterConfig, PairReporterOutcome

HERE = Path(__file__).resolve().parent
ROBOTS = ["/World/Robots/Rover", "/World/Robots/Drone", "/World/Robots/Crawler"]
ZONES = ["/World/Zones/Dock", "/World/Zones/Charge", "/World/Zones/Inspect"]
TIMES = [0, 1, 2, 3, 4]  # USD time codes
FIRST_ORDINAL = 2  # 1 = opened USD, 2 = computed hierarchy at time 0


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


def path_filter(paths: list[str]) -> ovstage.Filter:
    return ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])


def publish(population, ordinal: int, base: int | None, expected: int) -> None:
    if base is None:
        operation = population.enqueue_set(ordinal)
    else:
        operation = population.enqueue_update(base, ordinal)
    with operation:
        operation.wait()
        count = operation.fetch_population_update().object_count
    if count != expected:
        raise RuntimeError(f"published {count} objects, expected {expected}")


def main() -> None:
    usda = (HERE / "scene.usda").read_text(encoding="utf-8")
    # Ovstage takes time in seconds; USD's default is 24 time codes per second.
    match = re.search(r"timeCodesPerSecond\s*=\s*([0-9.eE+-]+)", usda)
    time_codes_per_second = float(match.group(1)) if match else 24.0

    # The stage block is outermost so every ovspatialquery object closes before it.
    with ovstage.Stage("py-pair-zones") as stage:
        # The time is positional: Ovstage 0.2 names it time_code, 0.3 names it time.
        ovstage.population.open_usd_from_string_with_desc(
            stage, usda, 1, TIMES[0] / time_codes_per_second, physics_desc(ovstage)
        )
        stage.advance_write_floor(ordinal=1).wait()
        stage.compute_hierarchy(1, FIRST_ORDINAL, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        stage.advance_write_floor(ordinal=FIRST_ORDINAL).wait()
        paths = ovstage.PathDictionary(stage)

        def name_pairs(rows) -> list[list[str]]:
            return sorted(
                [str(paths.path_to_string(_prim_path(a))), str(paths.path_to_string(_prim_path(b)))]
                for a, b in rows
            )

        with (
            Library.open(require_ovstage=True) as library,
            library.instance() as instance,  # closes reporter, then populations, on exit
        ):
            adapter = instance.ovstage(ovstage, stage)
            robots = adapter.create_population(path_filter(ROBOTS))
            zones = adapter.create_population(path_filter(ZONES))
            publish(robots, FIRST_ORDINAL, None, len(ROBOTS))
            publish(zones, FIRST_ORDINAL, None, len(ZONES))

            snapshot = instance.query_pairs(robots.target, zones.target, capacity=64)
            if snapshot.truncated:
                raise RuntimeError("pair snapshot truncated")
            snapshot_pairs = name_pairs(snapshot.rows)

            reporter = instance.create_pair_reporter(
                PairReporterConfig(
                    first=robots.target.reference,
                    second=zones.target.reference,
                    active_pair_capacity=len(ROBOTS) * len(ZONES),
                    delta_pair_capacity=2 * len(ROBOTS) * len(ZONES),
                )
            )
            steps = []
            ordinal = FIRST_ORDINAL
            for time in TIMES:
                if time != TIMES[0]:
                    next_ordinal = ordinal + 1
                    ovstage.population.update_from_usd_time(
                        stage, next_ordinal, time / time_codes_per_second
                    )
                    stage.advance_write_floor(ordinal=next_ordinal).wait()
                    # Republish every step: an advance with no newer publication is STALE_ORDINAL.
                    publish(robots, next_ordinal, ordinal, len(ROBOTS))
                    ordinal = next_ordinal
                operation, _reservation = reporter.advance()
                with operation:
                    operation.wait()
                    # Copy the deltas before the block closes the operation that owns them.
                    result = operation.fetch_pair_reporter().to_numpy()
                if result.outcome is not PairReporterOutcome.ADVANCED:
                    raise RuntimeError(f"pair reporter did not advance: {result.outcome!r}")
                steps.append(
                    {
                        "time": time,
                        "entered": name_pairs(result.entered_pairs.tolist()),
                        "exited": name_pairs(result.exited_pairs.tolist()),
                    }
                )

    print(json.dumps({"snapshot_t0": snapshot_pairs, "steps": steps}))


if __name__ == "__main__":
    main()
