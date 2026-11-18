# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find robot/scene pairs with a snapshot, then track changes with a reporter."""

import warp as wp

from ovspatialquery import (
    ExplicitUpdateKind,
    GeometryType,
    Library,
    PairReporterConfig,
)

wp.config.log_level = wp.LOG_WARNING  # skip Warp's start-up banner
CPU = "cpu"  # this program keeps its input arrays on the CPU
IDENTITY = [0.0, 0, 0, 1]


def advance(reporter):
    with reporter.advance()[0] as operation:  # both endpoints at their head
        operation.wait()
        result = operation.fetch_pair_reporter().to_numpy()
    print(
        f"advance ({result.first_source_ordinal}, {result.second_source_ordinal}):",
        f"entered {result.entered_pairs.tolist()}",
        f"exited {result.exited_pairs.tolist()}",
        f"active {result.accepted_active_pair_count}",
    )


with Library.open() as library, library.instance() as instance:
    adapter = instance.explicit()
    adapter.register_boxes(
        definition_values=wp.array([1], dtype=wp.uint64, device=CPU),
        half_extents=wp.full((1, 3), 0.5, dtype=wp.float64, device=CPU),
    ).finish()

    def publish(population, ordinal, ids, positions):
        population.enqueue_set(
            ordinal,
            object_values=wp.array(ids, dtype=wp.uint64, device=CPU),
            geometry_types=wp.full(
                len(ids), int(GeometryType.BOX), dtype=wp.int32, device=CPU
            ),
            geometry_values=wp.ones(len(ids), dtype=wp.uint64, device=CPU),
            translations=wp.array(positions, dtype=wp.float64, device=CPU),
            rotations_xyzw=wp.array([IDENTITY] * len(ids), dtype=wp.float32, device=CPU),
        ).finish()

    # Two populations on their own clocks:
    # a shelf (1) and a cone (2), and one robot (7).
    scene = adapter.create_population()
    publish(scene, 20, [1, 2], [[0, 0, 0], [5, 0, 0]])
    robots = adapter.create_population()
    publish(robots, 100, [7], [[0.8, 0, 0]])

    # A snapshot answers one instant and keeps nothing.
    snapshot = instance.query_pairs(robots.target, scene.target, capacity=8)
    ordinals = (snapshot.first_source_ordinal, snapshot.second_source_ordinal)
    print("snapshot", list(snapshot.rows), "at", ordinals)

    # A reporter keeps one baseline and reports what changed at each advance.
    reporter = instance.create_pair_reporter(
        PairReporterConfig(
            first=robots.target.reference,
            second=scene.target.reference,
            active_pair_capacity=8,
            delta_pair_capacity=8,
        )
    )
    advance(reporter)

    # The robot moves next to the cone: publish only its pose as robots ordinal 104.
    robots.enqueue_update(
        100,
        104,
        kind=ExplicitUpdateKind.POSES,
        object_values=wp.array([7], dtype=wp.uint64, device=CPU),
        translations=wp.array([[4.2, 0, 0]], dtype=wp.float64, device=CPU),
        rotations_xyzw=wp.array([IDENTITY], dtype=wp.float32, device=CPU),
    ).finish()
    advance(reporter)
