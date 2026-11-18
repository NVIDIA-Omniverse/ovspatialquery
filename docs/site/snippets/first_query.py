# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish two boxes and a sphere, then query them three ways."""

import warp as wp

from ovspatialquery import (
    GeometryType,
    Library,
    OverlapQuery,
    RaycastQuery,
    Sphere,
    SphereSweepQuery,
)

wp.config.log_level = wp.LOG_WARNING  # skip Warp's start-up banner
CPU = "cpu"  # this program keeps its input arrays on the CPU

# The instance owns everything created through it and closes it when the block ends.
with Library.open() as library, library.instance() as instance:
    adapter = instance.explicit()

    # 1. Register shapes once: definition 1 is a 1 x 1 x 1 box,
    #    definition 2 a sphere of radius 0.5. Every column is a Warp array;
    #    finish() waits for the operation and closes it.
    adapter.register_boxes(
        definition_values=wp.array([1], dtype=wp.uint64, device=CPU),
        half_extents=wp.array([[0.5, 0.5, 0.5]], dtype=wp.float64, device=CPU),
    ).finish()
    adapter.register_spheres(
        definition_values=wp.array([2], dtype=wp.uint64, device=CPU),
        radii=wp.array([0.5], dtype=wp.float64, device=CPU),
    ).finish()

    # 2. Place three objects and publish them as ordinal 1.
    population = adapter.create_population()
    with population.enqueue_set(
        1,
        object_values=wp.array([10, 11, 12], dtype=wp.uint64, device=CPU),
        geometry_types=wp.array(
            [GeometryType.BOX, GeometryType.BOX, GeometryType.SPHERE],
            dtype=wp.int32,
            device=CPU,
        ),
        geometry_values=wp.array([1, 1, 2], dtype=wp.uint64, device=CPU),
        translations=wp.array(
            [[4.0, 0, 0], [8, 0, 0], [4, 3, 0]], dtype=wp.float64, device=CPU
        ),
        rotations_xyzw=wp.array([[0.0, 0, 0, 1]] * 3, dtype=wp.float32, device=CPU),
    ) as publication:
        publication.wait()
        count = publication.fetch_population_update().object_count
    print("published", count, "objects")

    target = population.target  # the head of this population

    # 3a. A ray along +x stops on the first box's surface.
    ray = target.raycast_closest(
        RaycastQuery(origin=(0, 0, 0), unit_direction=(1, 0, 0), max_distance=20)
    )
    hit = ray.hit
    print("ray hit", hit.object_id, "at distance", round(hit.distance, 3), end=" ")
    print("ordinal", ray.source_ordinal)

    # 3b. A sphere of radius 0.5 swept along y = 3 reaches the sphere object.
    sweep = target.sweep_sphere_closest(
        SphereSweepQuery(
            center=(0, 3, 0), radius=0.5, unit_direction=(1, 0, 0), max_distance=20
        )
    )
    hit = sweep.hit
    center = tuple(round(v, 3) for v in hit.query_position)
    print("sweep hit", hit.object_id, "at distance", round(hit.distance, 3), end=" ")
    print("with its center at", center)

    # 3c. Does anything overlap a sphere around the second box?
    overlap = target.overlap_any(OverlapQuery(Sphere(center=(8, 1, 0), radius=0.75)))
    print("overlap", overlap.hit.object_id if overlap.hit else None)
