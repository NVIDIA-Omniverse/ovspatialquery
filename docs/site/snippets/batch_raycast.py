# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run four rays through a reusable CPU raycast plan and read the result columns."""

import warp as wp

from ovspatialquery import (
    BatchResultField,
    GeometryType,
    HitMode,
    Library,
    QueryPrecision,
    RaycastPlanConfig,
)

wp.config.log_level = wp.LOG_WARNING  # skip Warp's start-up banner
CPU = "cpu"  # this program and its plan run on the CPU

with Library.open() as library, library.instance() as instance:
    # A row of four unit boxes at x = 4, 6, 8, 10 on y = 0, 1, 2, 3.
    adapter = instance.explicit()
    adapter.register_boxes(
        definition_values=wp.array([1], dtype=wp.uint64, device=CPU),
        half_extents=wp.full((1, 3), 0.5, dtype=wp.float64, device=CPU),
    ).finish()
    population = adapter.create_population()
    population.enqueue_set(
        1,
        object_values=wp.array([20, 21, 22, 23], dtype=wp.uint64, device=CPU),
        geometry_types=wp.full(4, int(GeometryType.BOX), dtype=wp.int32, device=CPU),
        geometry_values=wp.ones(4, dtype=wp.uint64, device=CPU),
        translations=wp.array(
            [[4.0, 0, 0], [6, 1, 0], [8, 2, 0], [10, 3, 0]], dtype=wp.float64, device=CPU
        ),
        rotations_xyzw=wp.array([[0.0, 0, 0, 1]] * 4, dtype=wp.float32, device=CPU),
    ).finish()

    # The plan fixes the family, target, device and capacities once.
    plan = instance.create_raycast_plan(
        RaycastPlanConfig(
            target=population.target.reference,
            hit_mode=HitMode.CLOSEST,
            precision=QueryPrecision.EXACT,
            maximum_query_count=8,
            maximum_hits_per_query=1,
            hit_capacity_per_slot=8,
            result_slot_count=2,
            result_fields=BatchResultField.DISTANCE,
        )
    )

    # Each launch passes its rows as columns and reads the head by default.
    # Ray 2 misses: nothing lies on y = 5.
    operation, reservation = plan.enqueue(
        origins=wp.array(
            [[0.0, 0, 0], [0, 1, 0], [0, 5, 0], [0, 3, 0]], dtype=wp.float64, device=CPU
        ),
        unit_directions=wp.array([[1.0, 0, 0]] * 4, dtype=wp.float32, device=CPU),
        max_distances=wp.full(4, 20.0, dtype=wp.float64, device=CPU),
        copy=True,
    )
    # Leaving the block closes the operation, which returns its result slot to
    # the plan. For a CPU result, to_numpy() copies the columns so they outlive it.
    with operation:
        operation.wait()
        result = operation.fetch_raycast_batch().to_numpy()

    counts, offsets = result.hit_ranges.hit_counts, result.hit_ranges.hit_offsets
    print("source ordinal", result.common.source_ordinal)
    for row, (count, offset) in enumerate(zip(counts, offsets)):
        if count == 0:
            print(f"ray {row}: miss")
        else:
            distance = result.exact.distances[offset]
            print(f"ray {row}: object {result.object_ids[offset]} at {distance:.2f}")
