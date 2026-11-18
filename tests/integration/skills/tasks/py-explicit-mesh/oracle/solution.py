# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closest exact hit per ray against explicitly registered meshes, spheres and boxes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from ovspatialquery import GeometryType, Library, RaycastQuery

HERE = Path(__file__).resolve().parent
MESH_DEFINITION = 1
# Object values are the input index plus this offset: zero is a valid index but object
# value 0 is rejected, and an offset keeps "no object" distinct from object 0.
OBJECT_BASE = 1000


def u64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64)


def yaw_quaternion_xyzw(degrees: float) -> list[float]:
    half = math.radians(degrees) * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def main() -> None:
    scene = json.loads((HERE / "mesh.json").read_text(encoding="utf-8"))
    rays = json.loads((HERE / "rays.json").read_text(encoding="utf-8"))["rays"]

    instances = scene["mesh_instances"]
    spheres = scene["spheres"]
    boxes = scene["boxes"]
    kinds = ["mesh"] * len(instances) + ["sphere"] * len(spheres) + ["box"] * len(boxes)

    vertices = np.asarray(scene["mesh"]["vertices"], dtype=np.float32)  # float32 is required
    triangles = np.asarray(scene["mesh"]["triangles"], dtype=np.uint32)  # uint32 is required
    # One definition per sphere radius / box size; definition values are nonzero and only
    # need to be unique per geometry type.
    sphere_values = np.arange(1, len(spheres) + 1, dtype=np.uint64)
    box_values = np.arange(1, len(boxes) + 1, dtype=np.uint64)

    count = len(kinds)
    instance_columns = {
        "object_values": np.arange(OBJECT_BASE, OBJECT_BASE + count, dtype=np.uint64),
        "geometry_types": np.array(
            [int(GeometryType.TRIANGLE_MESH)] * len(instances)
            + [int(GeometryType.SPHERE)] * len(spheres)
            + [int(GeometryType.BOX)] * len(boxes),
            dtype=np.int32,
        ),
        "geometry_values": np.concatenate(
            [np.full(len(instances), MESH_DEFINITION, dtype=np.uint64), sphere_values, box_values]
        ),
        "translations": np.array(
            [i["translation"] for i in instances]
            + [s["center"] for s in spheres]
            + [b["center"] for b in boxes],
            dtype=np.float64,
        ),
        "rotations_xyzw": np.array(
            [yaw_quaternion_xyzw(i["yaw_degrees"]) for i in instances]
            + [[0.0, 0.0, 0.0, 1.0]] * len(spheres)
            + [yaw_quaternion_xyzw(b["yaw_degrees"]) for b in boxes],
            dtype=np.float64,
        ),
        "uniform_scales": np.array(
            [i["scale"] for i in instances] + [1.0] * (len(spheres) + len(boxes)),
            dtype=np.float64,
        ),
    }

    results = []
    # The instance destroys what it still owns on exit: the population, then the definitions.
    with Library.open(require_explicit=True) as library, library.instance() as instance:
        adapter = instance.explicit()
        # finish() waits, closes and raises ApiError if the operation failed.
        adapter.register_triangle_meshes(
            definition_values=u64([MESH_DEFINITION]),
            vertex_offsets=u64([0]),
            vertex_counts=u64([len(vertices)]),
            triangle_offsets=u64([0]),
            triangle_counts=u64([len(triangles)]),
            vertices=vertices,
            triangle_indices=triangles,
        ).finish()
        adapter.register_spheres(
            definition_values=sphere_values,
            radii=np.array([s["radius"] for s in spheres], dtype=np.float64),
        ).finish()
        adapter.register_boxes(
            definition_values=box_values,
            half_extents=np.array([b["half_extents"] for b in boxes], dtype=np.float64),
        ).finish()

        population = adapter.create_population()
        population.enqueue_set(1, **instance_columns).finish()

        for ray in rays:
            direction = np.asarray(ray["direction"], dtype=float)
            unit = direction / np.linalg.norm(direction)
            # Single queries default to QueryPrecision.EXACT.
            hit = population.target.raycast_closest(
                RaycastQuery(
                    origin=tuple(float(v) for v in ray["origin"]),
                    unit_direction=tuple(float(v) for v in unit),
                    max_distance=float(ray["max_distance"]),
                )
            ).hit
            if hit is None:
                results.append({"id": ray["id"], "hit": None})
                continue
            index = hit.object_id - OBJECT_BASE
            is_mesh = kinds[index] == "mesh"
            results.append(
                {
                    "id": ray["id"],
                    "hit": {
                        "object": index,
                        "kind": kinds[index],
                        "face": hit.source_face_index if is_mesh else None,
                        "barycentrics": list(hit.barycentrics) if is_mesh else None,
                        "distance": hit.distance,
                    },
                }
            )

    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
