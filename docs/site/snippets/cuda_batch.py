# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run four rays through a CUDA raycast plan and read the results in a Warp kernel."""

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


@wp.kernel
def closest_distance(
    counts: wp.array(dtype=wp.uint64),
    offsets: wp.array(dtype=wp.uint64),
    distances: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    ray = wp.tid()
    out[ray] = wp.float64(-1.0)  # a miss stores no hit
    if counts[ray] > wp.uint64(0):
        out[ray] = distances[wp.int32(offsets[ray])]


def consume(reservation, stream):
    """Reads the result columns on the GPU, ordered after the query, without a copy."""
    # from_dlpack exports each column on the current stream, which then waits on result_ready.
    with wp.ScopedStream(stream):
        out = wp.empty(4, dtype=wp.float64, device=GPU)
        wp.launch(
            closest_distance,
            dim=4,
            inputs=[
                wp.from_dlpack(reservation.hit_ranges.hit_counts),
                wp.from_dlpack(reservation.hit_ranges.hit_offsets),
                wp.from_dlpack(reservation.candidate.distances),
                out,
            ],
        )
    return out  # the column views end here, before the operation is retired


# CUDA work needs a byte budget for the device.
config = InstanceConfig(device_budgets=(DeviceBudget(Device.cuda(0), 256 << 20),))
with Library.open() as library, library.instance(config) as instance:
    # The same four unit boxes as the CPU example, published from CPU arrays.
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

    plan = instance.create_raycast_plan(
        RaycastPlanConfig(
            target=population.target.reference,
            hit_mode=HitMode.CLOSEST,
            maximum_query_count=8,
            maximum_hits_per_query=1,
            hit_capacity_per_slot=8,
            result_fields=BatchResultField.DISTANCE,
            device=Device.cuda(0),
        )
    )

    # The rays are written on a Warp stream, and the launch is queued on the same stream.
    stream = wp.Stream(GPU)
    with wp.ScopedStream(stream):
        origins = wp.array(
            [[0.0, 0, 0], [0, 1, 0], [0, 5, 0], [0, 3, 0]], dtype=wp.float64, device=GPU
        )
        directions = wp.array([[1.0, 0, 0]] * 4, dtype=wp.float32, device=GPU)
        max_distances = wp.full(4, 20.0, dtype=wp.float64, device=GPU)
    operation, reservation = plan.enqueue(
        origins=origins,
        unit_directions=directions,
        max_distances=max_distances,
        cuda_stream=stream,
    )

    # The reservation's columns exist at enqueue, so the consumer is queued before the query
    # has run. The host has not waited for the GPU yet.
    distances = consume(reservation, stream)
    consumer_done = stream.record_event()

    # Retire the operation after its consumer. The library takes the event and releases the
    # result slot once it completes; leaving the block waits for that release.
    operation.wait()
    operation.close_after_cuda(consumer_done.cuda_event)
    consumer_done.owner = False  # the library now owns the event

    for ray, distance in enumerate(distances.numpy()):  # copy to the host only to print
        print(f"ray {ray}: miss" if distance < 0 else f"ray {ray}: {distance:.2f}")
