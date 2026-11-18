# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run four rays through a CUDA raycast plan whose results the host reads in place."""

import warp as wp

from ovspatialquery import (
    BatchResultField,
    Device,
    DeviceBudget,
    GeometryType,
    HitMode,
    InstanceConfig,
    Library,
    RaycastPlanConfig,
)

wp.config.log_level = wp.LOG_WARNING  # skip Warp's start-up banner
CPU, GPU = "cpu", "cuda:0"

config = InstanceConfig(device_budgets=(DeviceBudget(Device.cuda(0), 256 << 20),))
with Library.open() as library, library.instance(config) as instance:
    # The same four unit boxes as the CUDA example.
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

    # HOST_MAPPED keeps the result columns in pinned host memory the GPU writes.
    plan = instance.create_raycast_plan(
        RaycastPlanConfig(
            target=population.target.reference,
            hit_mode=HitMode.CLOSEST,
            maximum_query_count=8,
            maximum_hits_per_query=1,
            hit_capacity_per_slot=8,
            result_fields=BatchResultField.DISTANCE | BatchResultField.HOST_MAPPED,
            device=Device.cuda(0),
        )
    )

    stream = wp.Stream(GPU)
    with wp.ScopedStream(stream):
        origins = wp.array(
            [[0.0, 0, 0], [0, 1, 0], [0, 5, 0], [0, 3, 0]], dtype=wp.float64, device=GPU
        )
        directions = wp.array([[1.0, 0, 0]] * 4, dtype=wp.float32, device=GPU)
        max_distances = wp.full(4, 20.0, dtype=wp.float64, device=GPU)
    operation, _ = plan.enqueue(
        origins=origins,
        unit_directions=directions,
        max_distances=max_distances,
        cuda_stream=stream,
    )

    # The host may read the columns once the wait has observed the end; to_numpy() copies
    # them so they outlive the operation. Nothing on the GPU reads them, so leaving the
    # block closes the operation without a consumer event.
    with operation:
        operation.wait()
        result = operation.fetch_raycast_batch().to_numpy()

    counts, offsets = result.hit_ranges.hit_counts, result.hit_ranges.hit_offsets
    for ray, count in enumerate(counts):
        distance = result.candidate.distances[offsets[ray]] if count else None
        print(f"ray {ray}: miss" if distance is None else f"ray {ray}: {distance:.2f}")
