# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference solution: closest hits filtered by PhysX filter data, as participation bits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def filter_bits(words: list[int]) -> int:
    """PxFilterData words 0-3 as one 128-bit participation value, word0 lowest."""
    return words[0] | words[1] << 32 | words[2] << 64 | words[3] << 96


def main() -> int:
    filters = json.loads((FILES / "filters.json").read_text(encoding="utf-8"))
    rays = json.loads((FILES / "rays.json").read_text(encoding="utf-8"))["rays"]
    count = len(rays)
    origins = np.array([ray["origin"] for ray in rays], dtype=np.float64)
    directions = np.array([ray["direction"] for ray in rays], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    max_distances = np.array([ray["max_distance"] for ray in rays], dtype=np.float64)
    # uint32[Q,4] filter words read as two little-endian uint64 words: the uint64[Q,2] column.
    query_bits = np.ascontiguousarray([ray["filter_data"] for ray in rays], dtype="<u4").view("<u8")

    stage = ovstage.Stage("filter-data-rays")
    hits: list[dict | None] = []
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
            population = instance.ovstage(ovstage, stage).create_population(shapes)
            # Prims without filter data have zero bits: no nonzero query accepts them.
            population.set_participation_map(
                0, {path: filter_bits(words) for path, words in filters.items()}
            )
            with population.enqueue_set(2) as publish:
                publish.wait()
                if publish.fetch_population_update().object_count != 10:
                    raise RuntimeError("expected the scene's 10 shapes")
            plan = instance.create_raycast_plan(
                RaycastPlanConfig(
                    target=population.target.reference,
                    hit_mode=HitMode.CLOSEST,
                    precision=QueryPrecision.EXACT,
                    maximum_query_count=count,
                    maximum_hits_per_query=1,
                    hit_capacity_per_slot=count,
                    result_slot_count=1,
                    result_fields=BatchResultField.DISTANCE,
                )
            )
            operation, _ = plan.enqueue(
                origins=origins,
                unit_directions=directions.astype(np.float32),
                max_distances=max_distances,
                query_bits=query_bits,
                copy=True,
            )
            with operation:
                operation.wait()
                result = operation.fetch_raycast_batch().to_numpy()
            ranges = result.hit_ranges
            if np.any(ranges.query_status != int(BatchRowStatus.SUCCESS)):
                raise RuntimeError(f"row failures: {np.unique(ranges.query_status)}")
            for ray in range(count):
                if ranges.hit_counts[ray] == 0:
                    hits.append(None)
                    continue
                row = int(ranges.hit_offsets[ray])
                hits.append(
                    {
                        "prim": str(dictionary.path_to_string(_prim_path(result.object_ids[row]))),
                        "distance": float(result.exact.distances[row]),
                    }
                )
    finally:
        stage.destroy()

    json.dump({"hits": hits}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
