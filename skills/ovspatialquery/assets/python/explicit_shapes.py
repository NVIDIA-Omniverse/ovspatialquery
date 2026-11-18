# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit population: register sphere, box, capsule, convex and triangle-mesh shapes,
place instances, query them exactly, move one with a POSES update, then clean up.

No USD and no Ovstage stage. Prints one JSON object; exits nonzero if an answer is wrong.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import ovsq  # noqa: E402  (unit, yaw_xyzw)
from ovspatialquery import (  # noqa: E402
    Aabb,
    ApiError,
    ExplicitUpdateKind,
    GeometryDefinitionRef,
    GeometryType,
    Library,
    OrdinalSelection,
    OverlapQuery,
    RaycastQuery,
    Sphere,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)  # quaternion order is (x, y, z, w)

# A 2 x 2 square in local z = 0, split along its diagonal. Both triangles are
# counter-clockwise seen from +Z, so only rays coming from above hit them.
SQUARE_VERTICES = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
SQUARE_TRIANGLES = [(0, 1, 2), (0, 2, 3)]
UNIT_CUBE = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]

# Object values are yours: nonzero and unique (0 is rejected). Hits report them.
BALL, CRATE, PIPE, BLOCK, RAMP = 101, 102, 103, 104, 105
ROWS = [
    # (object, geometry type, definition value, translation, rotation xyzw, uniform scale)
    (BALL, GeometryType.SPHERE, 1, (0, 0, 0), IDENTITY, 1.0),
    (CRATE, GeometryType.BOX, 1, (5, 0, 0), ovsq.yaw_xyzw(90), 1.0),
    (PIPE, GeometryType.CAPSULE, 1, (0, 5, 0), ovsq.yaw_xyzw(90), 1.0),  # local Y onto -X
    (BLOCK, GeometryType.CONVEX, 1, (0, -5, 0), IDENTITY, 2.0),  # unit cube scaled to 2
    (RAMP, GeometryType.TRIANGLE_MESH, 1, (10, 0, 1), IDENTITY, 1.0),
]


def u64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64)


def instance_columns(rows) -> dict:
    """Keyword columns for `enqueue_set` / a COMPLETE_ROWS `enqueue_update`."""
    return {
        "object_values": u64([r[0] for r in rows]),
        "geometry_types": np.array([int(r[1]) for r in rows], dtype=np.int32),
        "geometry_values": u64([r[2] for r in rows]),
        "translations": np.array([r[3] for r in rows], dtype=np.float64).reshape(-1, 3),
        "rotations_xyzw": np.array([r[4] for r in rows], dtype=np.float64).reshape(-1, 4),
        "uniform_scales": np.array([r[5] for r in rows], dtype=np.float64),
    }


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    return math.isclose(a, b, abs_tol=tol)


def register_shapes(adapter) -> None:
    """One definition of each family, all with definition value 1 (unique per type)."""
    adapter.register_spheres(definition_values=u64([1]), radii=np.array([0.5])).finish()
    adapter.register_boxes(
        definition_values=u64([1]), half_extents=np.array([(1.0, 0.5, 0.25)])
    ).finish()
    adapter.register_capsules(  # segment along local Y, from -half_segment to +half_segment
        definition_values=u64([1]), radii=np.array([0.25]), half_segments=np.array([1.0])
    ).finish()
    adapter.register_convexes(  # the library takes the convex hull of the points
        definition_values=u64([1]),
        vertex_offsets=u64([0]),
        vertex_counts=u64([len(UNIT_CUBE)]),
        vertices=np.array(UNIT_CUBE, dtype=np.float32),
    ).finish()
    adapter.register_triangle_meshes(  # indices are local to each mesh
        definition_values=u64([1]),
        vertex_offsets=u64([0]),
        vertex_counts=u64([len(SQUARE_VERTICES)]),
        triangle_offsets=u64([0]),
        triangle_counts=u64([len(SQUARE_TRIANGLES)]),
        vertices=np.array(SQUARE_VERTICES, dtype=np.float32),
        triangle_indices=np.array(SQUARE_TRIANGLES, dtype=np.uint32),
    ).finish()


def main() -> dict:
    out: dict = {}
    # The instance destroys what it still owns on exit: populations, then definitions.
    with Library.open(require_explicit=True) as library, library.instance() as instance:
        adapter = instance.explicit()
        register_shapes(adapter)
        with adapter.create_population() as population:
            bad = instance_columns([(0, GeometryType.SPHERE, 1, (0, 0, 0), IDENTITY, 1.0)])
            try:
                population.enqueue_set(1, **bad).finish()
                raise AssertionError("object value 0 was accepted")
            except ApiError as error:
                out["object_value_0"] = error.status.name
            check(out["object_value_0"] == "INVALID_ARGUMENT", "object 0 status")

            population.enqueue_set(1, **instance_columns(ROWS)).finish()
            check(population.info().object_count == len(ROWS), "every row published")
            target = population.target

            def ray(origin, direction, max_distance=100.0, **extra):
                # Single queries default to QueryPrecision.EXACT.
                query = RaycastQuery(origin, ovsq.unit(direction), max_distance, **extra)
                return target.raycast_closest(query).hit

            # Capsule rotated so its segment runs along X: a ray at x = 0.8 hits its top.
            hit = ray((0.8, 5, 3), (0, 0, -1))
            check(hit.object_id == PIPE and close(hit.distance, 2.75), "capsule pose")
            check(ray((0, 5.8, 3), (0, 0, -1)) is None, "unrotated capsule would be hit")
            # Convex scaled by 2: its top face is at z = 1.
            hit = ray((0, -5, 5), (0, 0, -1))
            check(hit.object_id == BLOCK and close(hit.distance, 4.0), "convex scale")
            # Box yawed 90 degrees: its local x half extent 1 now spans y in [-1, 1].
            hit = ray((5, -5, 0), (0, 1, 0))
            check(hit.object_id == CRATE and close(hit.distance, 4.0), "box yaw")

            # Mesh: face index is the submitted triangle; barycentrics weight its
            # listed corners in order. Point (1.5, 0.5) = 0.25*v0 + 0.5*v1 + 0.25*v2.
            hit = ray((11.5, 0.5, 5), (0, 0, -1))
            check(hit.object_id == RAMP and hit.source_face_index == 0, "mesh face 0")
            check(all(close(a, b) for a, b in zip(hit.barycentrics, (0.25, 0.5, 0.25))), "bary 0")
            out["mesh_hit"] = {
                "face": hit.source_face_index,
                "barycentrics": hit.barycentrics,
                "distance": hit.distance,
                "position": hit.position,
            }
            hit = ray((10.5, 1.5, 5), (0, 0, -1))  # triangle (0, 2, 3)
            check(hit.source_face_index == 1, "mesh face 1")
            check(all(close(a, b) for a, b in zip(hit.barycentrics, (0.25, 0.25, 0.5))), "bary 1")
            check(ray((11.5, 0.5, -5), (0, 0, 1)) is None, "back face is not hit")
            check(ray((11.5, 0.5, 5), (0, 0, -1), max_distance=3.9) is None, "max_distance")

            # Overlap: overlap_any gives one hit; overlap_multiple up to `capacity` hits.
            any_hit = target.overlap_any(OverlapQuery(Sphere((5, 0.9, 0), 0.2)))
            check(any_hit.hit is not None and any_hit.hit.object_id == CRATE, "overlap_any")
            region = target.overlap_multiple(OverlapQuery(Aabb((-1, -6, -1), (6, 6, 1))), 8)
            check(not region.truncated, "overlap capacity")
            out["overlap_multiple"] = sorted(h.object_id for h in region.hits)
            check(out["overlap_multiple"] == [BALL, CRATE, PIPE, BLOCK], "overlap_multiple")

            # POSES update: move BALL up by 3; base must be the current head (1).
            population.enqueue_update(
                1,
                2,
                kind=ExplicitUpdateKind.POSES,
                object_values=u64([BALL]),
                translations=np.array([(0.0, 0.0, 3.0)]),
                rotations_xyzw=np.array([IDENTITY]),
            ).finish()
            info = population.info()
            check(info.head_ordinal == 2 and info.object_count == len(ROWS), "poses")
            hit = ray((0, -3, 3), (0, 1, 0))
            check(hit.object_id == BALL and close(hit.distance, 2.5), "ball moved")
            old = target.raycast_closest(
                RaycastQuery(
                    (0, -3, 0), (0, 1, 0), 100.0, ordinal=OrdinalSelection.retained_exact(1)
                )
            )
            check(old.source_ordinal == 1 and old.hit.object_id == BALL, "ordinal 1 retained")
            try:
                population.enqueue_update(
                    1,
                    3,
                    kind=ExplicitUpdateKind.POSES,
                    object_values=u64([BALL]),
                    translations=np.zeros((1, 3)),
                    rotations_xyzw=np.array([IDENTITY]),
                ).finish()
                raise AssertionError("stale base accepted")
            except ApiError as error:
                out["stale_base"] = error.status.name
            check(out["stale_base"] == "ORDINAL_MISMATCH", "stale base status")

            # A definition used by a live population cannot be destroyed.
            try:
                adapter.destroy_definition(GeometryDefinitionRef(GeometryType.SPHERE, 1))
                raise AssertionError("destroyed a definition in use")
            except ApiError as error:
                out["destroy_in_use"] = error.status.name
            check(out["destroy_in_use"] == "BUSY", "destroy status")
            out["head_ordinal"] = population.info().head_ordinal
    return out


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
