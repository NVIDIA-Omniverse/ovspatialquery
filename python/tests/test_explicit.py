# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import base64
import contextlib
import ctypes
import gc
import itertools
import math
import os
from pathlib import Path
import sys
import time
import unittest
import weakref
from ovspatialquery import (
    ApiError,
    ApiStatus,
    BatchResultField,
    BoxSweepQuery,
    Capability,
    Device,
    DeviceBudget,
    ExplicitUpdateKind,
    GeometryDefinitionRef,
    GeometryType,
    HitMode,
    InstanceConfig,
    Library,
    OpState,
    OrientedBox,
    OverlapQuery,
    Preparation,
    QueryFlag,
    QueryPrecision,
    RaycastPlanConfig,
    RaycastQuery,
    Sphere,
    SphereSweepQuery,
    raw,
)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import warp as wp
except ImportError:
    wp = None


def _library_path() -> Path | None:
    install = Path(__file__).resolve().parents[2] / "_build/bootstrap/native-stable/install"
    artifact = install / (
        "bin/ovspatialquery.dll" if sys.platform == "win32" else "lib/libovspatialquery.so"
    )
    selected = os.environ.get("OVSPATIALQUERY_LIBRARY")
    return Path(selected) if selected else (artifact if artifact.is_file() else None)


def _u64(values):
    return np.array(values, dtype=np.uint64)


# One object per family, each centred on y = 0 at its own x: a unit sphere, a box of half
# extent 0.5, a capsule of radius 0.5 whose segment spans y = -1..1, a cube hull of half
# extent 0.25, and one downward-facing triangle that only a double-sided mesh shows from above.
_FAMILIES = (
    (1, GeometryType.SPHERE, 5.0, 9.0),
    (2, GeometryType.BOX, 10.0, 9.5),
    (3, GeometryType.CAPSULE, 15.0, 8.5),
    (4, GeometryType.CONVEX, 20.0, 9.75),
    (5, GeometryType.TRIANGLE_MESH, 25.0, 10.0),
)
_DEFINITION = 7

# A bevelled carton's collider as a content pipeline exports it: 152 float32 points (x, y, z,
# little-endian, base64) in metres, 0.51 by 0.51 by 0.26, with points along its rims and bevels.
_BEVELLED_CARTON = (
    "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"
)


def _instances(families=_FAMILIES):
    return {
        "object_values": _u64([family[0] for family in families]),
        "geometry_types": np.array([family[1] for family in families], dtype=np.int32),
        "geometry_values": _u64([_DEFINITION] * len(families)),
        "translations": np.array([(family[2], 0, 0) for family in families]).reshape(-1, 3),
        "rotations_xyzw": np.array([(0.0, 0.0, 0.0, 1.0)] * len(families)).reshape(-1, 4),
    }


# The collider of a larger bevelled carton (issue #364's): 279 float32 points in metres, about
# 0.70 by 0.51 by 0.52, whose hull has 137 vertices and 255 faces.
_CARTON_BEYOND_THE_FACE_LIMIT = (
    "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"
    "vpNlWb3unQM/zKiqvpRlWb3NJwU/LM+uvpNlWb2TzgM/2MeqvmbZWr24eR06bm2rvmbZWr1SSps6tnKrvj2JWb0UpZw7"
    "ON6uvrjYWb2Eo6o78syyvmjZWr0JL0s8EfKxvroxWr28yRY8clCwvrjYWb2pVM87iI+sPr2xXb2XQwU/yACtPr2xXb06"
    "AAU/HMesPtHSXL164gM/rNqyPrIzW70h6QE/D6qxPiMTXL3fsQI/YWKwPp3SXL0TnwM/s2erPgplYb04xbE6QzCrPgpl"
    "Yb1SeS06iHmrPnHbXr34CJg7zbuuPoWxXb2Do6o7H+SxProaW703PhM8bqqyPqV2Wr0JL0s88y2wPoWxXb2pVM87fESu"
    "vpRlWb3hLwU/d0ivvsNULj2t0QQ/A7muvsNULj3UKQU/LVmsvp+DWr0zagU/34eyvpNlWb075QI/+v6yvsRULj28wgI/"
    "+kSxvt1ULj0suQM/jbKwvpNlWb1c6AM/W7utvj2JWb13lLs6E0qwvrjYWb3U55s7kmGyvroxWr2bZQk8+p+yvvpULj2V"
    "1Q886Y6xvvtULj0hZbA7qTOwvvtULj07mNI6HVesvmbZWr2kRLk5NFWvvvtULj0Gwk45j9GuPjhSXb034gQ/5JWwPp3S"
    "XL0B9gM/BUOyPiMTXL3p/wI/eGKyPqV4MD1S7wI/LtKwPiWtMT0GAAQ/uDCvPo8YMz048QQ/wL+tPr2xXb1jTAU/7POt"
    "Pi6yMz1yZwU/CpavPuFULj0pXKcy88+wPvtULj0hxR078+2xPvtULj2Xqqw7Lg6zPvpULj0PHQ48CPCyPnzDWr2vKRQ8"
    "rIOwPoWxXb1KIaA7cTWvPoUmX72Wz0A7TdOsPricYb2qk2o5"
)


def _down(x: float) -> RaycastQuery:
    """A ray falling from y = 10 onto x."""
    return RaycastQuery((x, 10, 0), (0, -1, 0), 20, precision=QueryPrecision.EXACT)


def _hull_faces(points):
    """Each face of the points' hull as (outward unit normal, centroid of the points on it),
    from every plane through three of the points that has every point on one side to 1e-6 of
    their half extent. Triangles under 1e-4 of the squared half extent are skipped, since
    rounding tilts their planes."""
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    tolerance = 1e-6 * np.abs(points - points.mean(0)).max()
    first, second, third = np.array(list(itertools.combinations(range(len(points)), 3))).T
    normals = np.cross(points[second] - points[first], points[third] - points[first])
    length = np.linalg.norm(normals, axis=1)
    large = length > 1e4 * tolerance**2
    normals, first = normals[large] / length[large, None], first[large]
    outward = []
    for chunk, origin in zip(np.array_split(normals, 64), np.array_split(first, 64)):
        heights = points @ chunk.T - np.einsum("ij,ij->i", points[origin], chunk)
        outward += [chunk[heights.max(0) <= tolerance], -chunk[heights.min(0) >= -tolerance]]
    faces = {}
    for normal in np.vstack(outward):
        support = points @ normal
        on_face = support >= support.max() - tolerance
        faces.setdefault(tuple(np.round(normal, 4)), (normal, points[on_face].mean(0)))
    return list(faces.values())


@unittest.skipIf(np is None, "NumPy supplies the DLPack columns")
class ExplicitColumnTests(unittest.TestCase):
    def setUp(self) -> None:
        budgets = ()
        if wp is not None and wp.is_cuda_available():
            budgets = (DeviceBudget(Device.cuda(), 1 << 28),)
        self.library = Library.open(_library_path(), require_explicit=True)
        self.addCleanup(self.library.close)
        self.instance = self.library.instance(InstanceConfig(device_budgets=budgets))
        self.addCleanup(self.instance.close)
        self.adapter = self.instance.explicit()

    def _register_spheres(self) -> None:
        self.adapter.register_spheres(
            definition_values=_u64([_DEFINITION]), radii=np.array([1.0])
        ).finish()
        self.addCleanup(
            self.adapter.destroy_definition,
            GeometryDefinitionRef(GeometryType.SPHERE, _DEFINITION),
        )

    def _population(self):
        population = self.adapter.create_population()
        self.addCleanup(population.close)
        return population

    def test_every_family_and_publication_kind_reaches_queries(self) -> None:
        cube = [(x, y, z) for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)]
        registrations = {
            GeometryType.SPHERE: lambda: self.adapter.register_spheres(
                definition_values=_u64([_DEFINITION]), radii=np.array([1.0])
            ),
            GeometryType.BOX: lambda: self.adapter.register_boxes(
                definition_values=_u64([_DEFINITION]), half_extents=np.full((1, 3), 0.5)
            ),
            GeometryType.CAPSULE: lambda: self.adapter.register_capsules(
                definition_values=_u64([_DEFINITION]),
                radii=np.array([0.5]),
                half_segments=np.array([1.0]),
            ),
            GeometryType.CONVEX: lambda: self.adapter.register_convexes(
                definition_values=_u64([_DEFINITION]),
                vertex_offsets=_u64([0]),
                vertex_counts=_u64([8]),
                vertices=np.array(cube, dtype=np.float32),
            ),
            GeometryType.TRIANGLE_MESH: lambda: self.adapter.register_triangle_meshes(
                definition_values=_u64([_DEFINITION]),
                vertex_offsets=_u64([0]),
                vertex_counts=_u64([3]),
                triangle_offsets=_u64([0]),
                triangle_counts=_u64([1]),
                vertices=np.array([(-1, 0, -1), (1, 0, -1), (0, 0, 1)], dtype=np.float32),
                triangle_indices=np.array([(0, 1, 2)], dtype=np.uint32),
                double_sided=np.array([1], dtype=np.uint8),
            ),
        }
        for family, register in registrations.items():
            register().finish()
            self.addCleanup(
                self.adapter.destroy_definition, GeometryDefinitionRef(family, _DEFINITION)
            )
        population = self._population()
        population.enqueue_set(1, **_instances()).finish()
        for object_id, family, x, distance in _FAMILIES:
            with self.subTest(family=family.name):
                hit = population.target.raycast_closest(_down(x)).hit
                self.assertEqual((hit.object_id, hit.geometry_type), (object_id, family))
                self.assertAlmostEqual(hit.distance, distance, places=5)

        # The sphere drops by 3 in a POSES update, which carries no binding columns.
        population.enqueue_update(
            1,
            2,
            kind=ExplicitUpdateKind.POSES,
            object_values=_u64([1]),
            translations=np.array([(5.0, -3.0, 0.0)]),
            rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)]),
        ).finish()
        self.assertAlmostEqual(population.target.raycast_closest(_down(5)).hit.distance, 12.0, 5)
        # A complete-rows update with no changed rows removes the box.
        population.enqueue_update(2, 3, **_instances(()), removed_object_values=_u64([2])).finish()
        self.assertIsNone(population.target.raycast_closest(_down(10)).hit)
        self.assertEqual(population.info().object_count, 4)

        view = population.create_view(_u64([3]))
        self.addCleanup(view.close)
        self.assertIsNone(view.target.raycast_closest(_down(5)).hit)
        self.assertEqual(view.target.raycast_closest(_down(15)).hit.object_id, 3)

    def _last_error(self) -> str:
        detail = self.library.exports["ovspatialquery_get_last_error"]()
        return ctypes.string_at(detail.ptr, detail.length).decode() if detail.length else ""

    def test_a_triangle_without_area_is_dropped_and_its_mesh_kept(self) -> None:
        # Four triangles at y = 0 with materials 2000..2003. Triangle 1 repeats a corner and
        # triangle 3 has collinear corners along z = -1 out to x = 5, so both are dropped:
        # the mesh registers with triangles 0 (x < 0) and 2 (x > 0), which keep their own
        # ordinals and materials on the CPU and on CUDA, and nothing touches the segment
        # triangle 3 would have covered. The registration names what it dropped.
        vertices = np.array(
            [(-3, 0, -1), (-1, 0, -1), (-2, 0, 1), (1, 0, -1), (3, 0, -1), (2, 0, 1), (5, 0, -1)],
            dtype=np.float32,
        )
        registration = self.adapter.register_triangle_meshes(
            definition_values=_u64([_DEFINITION]),
            vertex_offsets=_u64([0]),
            vertex_counts=_u64([7]),
            triangle_offsets=_u64([0]),
            triangle_counts=_u64([4]),
            vertices=vertices,
            triangle_indices=np.array(
                [(0, 1, 2), (3, 3, 4), (3, 4, 5), (0, 1, 6)], dtype=np.uint32
            ),
            double_sided=np.array([1], dtype=np.uint8),
            source_face_material_ids=_u64([2000, 2001, 2002, 2003]),
        )
        report = self._last_error()
        registration.finish()
        self.addCleanup(
            self.adapter.destroy_definition,
            GeometryDefinitionRef(GeometryType.TRIANGLE_MESH, _DEFINITION),
        )
        self.assertRegex(report, rf"definition {_DEFINITION}\b.*\b2 of its 4 triangles")
        self.assertRegex(report, r"triangles 1, 3\)")
        self.assertEqual(self.adapter.dropped_triangle_count(_DEFINITION), 2)

        population = self._population()
        population.enqueue_set(
            1, **_instances(((2, GeometryType.TRIANGLE_MESH, 0.0, 0.0),)), material_ids=_u64([5000])
        ).finish()
        target = population.target
        for x, face, material in ((-2, 0, 2000), (2, 2, 2002)):
            hit = target.raycast_closest(_down(x)).hit
            self.assertEqual((hit.source_face_index, hit.material_id), (face, material))
        self.assertIsNone(target.overlap_any(OverlapQuery(Sphere((4, 0, -1), 0.25))).hit)
        sweep = SphereSweepQuery((4, 2, -1), 0.25, (0, -1, 0), 4.0)
        self.assertIsNone(target.sweep_sphere_closest(sweep).hit)

        devices = [Device()]
        if wp is not None and wp.is_cuda_available():
            devices.append(Device.cuda())
        for device in devices:
            with self.subTest(device=device):
                cpu = device == Device()
                columns = (
                    (lambda value: value)
                    if cpu
                    else (lambda value: wp.array(value, device="cuda:0"))
                )
                config = RaycastPlanConfig(
                    target.reference,
                    HitMode.CLOSEST,
                    2,
                    1,
                    2,
                    precision=QueryPrecision.EXACT,
                    device=device,
                    result_fields=BatchResultField.SOURCE_FEATURE | BatchResultField.MATERIAL,
                )
                stream = None if cpu else wp.Stream(wp.get_device("cuda:0"))
                with (
                    self.instance.create_raycast_plan(config) as plan,
                    contextlib.nullcontext() if cpu else wp.ScopedStream(stream),
                ):
                    operation, _ = plan.enqueue(
                        origins=columns(np.array([(-2.0, 10.0, 0.0), (2.0, 10.0, 0.0)])),
                        unit_directions=columns(np.array([(0, -1, 0)] * 2, dtype=np.float32)),
                        max_distances=columns(np.array([20.0, 20.0])),
                        **({} if cpu else {"cuda_stream": stream}),
                    )
                    self.assertIs(operation.wait(), OpState.COMPLETED)
                    result = operation.fetch_raycast_batch()
                    if cpu:
                        with operation:
                            result = result.to_numpy()
                            found = (
                                result.exact.features.source_face_indices.tolist(),
                                result.material_ids.tolist(),
                            )
                    else:
                        wp.wait_event(
                            wp.Event(wp.get_device("cuda:0"), cuda_event=result.common.result_ready)
                        )
                        found = tuple(
                            wp.from_dlpack(column).numpy().tolist()
                            for column in (
                                result.exact.features.source_face_indices,
                                result.material_ids,
                            )
                        )
                        done = wp.record_event()
                        wp.synchronize_event(done)
                        operation.close_after_cuda(done.cuda_event)
                        done.owner = False  # Retirement transfers the event to the library.
                self.assertEqual(found, ([0, 2], [2000, 2002]))

    def test_hits_report_the_source_faces_the_caller_registers(self) -> None:
        # A quad at y = 0 the caller split into corners (0, 1, 2) and (0, 2, 3) of source
        # face 4: rays down at x = 1 and x = -1 cross one triangle each and report face 4,
        # those corners and the face's material. The two columns come together, and a face
        # whose triangles carry two materials is refused and named.
        quad = {
            "definition_values": _u64([_DEFINITION]),
            "vertex_offsets": _u64([0]),
            "vertex_counts": _u64([4]),
            "triangle_offsets": _u64([0]),
            "triangle_counts": _u64([2]),
            "vertices": np.array(
                [(-2, 0, -1), (2, 0, -1), (2, 0, 1), (-2, 0, 1)], dtype=np.float32
            ),
            "triangle_indices": np.array([(0, 1, 2), (0, 2, 3)], dtype=np.uint32),
            "double_sided": np.array([1], dtype=np.uint8),
            "source_face_indices": _u64([4, 4]),
            "source_face_corner_indices": np.array([(0, 1, 2), (0, 2, 3)], dtype=np.uint32),
        }
        with self.assertRaisesRegex(ValueError, "come together"):
            self.adapter.register_triangle_meshes(**{**quad, "source_face_corner_indices": None})
        with self.assertRaises(ApiError) as refused:
            self.adapter.register_triangle_meshes(
                **quad, source_face_material_ids=_u64([2000, 2001])
            ).finish()
        self.assertEqual(refused.exception.status, ApiStatus.INVALID_ARGUMENT)
        self.assertRegex(refused.exception.detail, r"triangles 0 and 1 belong to source face 4")
        self.adapter.register_triangle_meshes(
            **quad, source_face_material_ids=_u64([2000, 2000])
        ).finish()
        self.addCleanup(
            self.adapter.destroy_definition,
            GeometryDefinitionRef(GeometryType.TRIANGLE_MESH, _DEFINITION),
        )
        population = self._population()
        population.enqueue_set(
            1, **_instances(((2, GeometryType.TRIANGLE_MESH, 0.0, 0.0),)), material_ids=_u64([5000])
        ).finish()
        for x, corners in ((1, (0, 1, 2)), (-1, (0, 2, 3))):
            hit = population.target.raycast_closest(_down(x)).hit
            self.assertEqual(
                (hit.source_face_index, hit.source_face_corner_indices, hit.material_id),
                (4, corners, 2000),
            )

    def test_a_mesh_left_without_triangles_or_naming_a_missing_vertex_is_refused(self) -> None:
        # Every triangle of definition 8 lacks area, and triangle 2 of definition 9 names a
        # vertex the definition does not have: each refuses the registration and names
        # the definition and why. The mesh with one triangle with area registers.
        vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0)], dtype=np.float32)

        def register(value, triangles):
            return self.adapter.register_triangle_meshes(
                definition_values=_u64([value]),
                vertex_offsets=_u64([0]),
                vertex_counts=_u64([4]),
                triangle_offsets=_u64([0]),
                triangle_counts=_u64([len(triangles)]),
                vertices=vertices,
                triangle_indices=np.array(triangles, dtype=np.uint32),
            ).finish()

        for value, triangles, detail in (
            (8, [(0, 1, 1), (0, 1, 3)], r"definition 8\b.*\bno(ne)?\b.*\barea"),
            (9, [(0, 1, 2), (1, 2, 3), (0, 1, 4)], r"definition 9\b.*\btriangle 2\b.*\bvertex"),
        ):
            with self.subTest(triangles=triangles):
                with self.assertRaises(ApiError) as refused:
                    register(value, triangles)
                self.assertEqual(refused.exception.status, ApiStatus.INVALID_ARGUMENT)
                self.assertRegex(refused.exception.detail, detail)
        register(10, [(0, 1, 1), (0, 1, 2)])
        self.assertEqual(self.adapter.dropped_triangle_count(10), 1)
        self.adapter.destroy_definition(GeometryDefinitionRef(GeometryType.TRIANGLE_MESH, 10))
        with self.assertRaises(ApiError) as unknown:
            self.adapter.dropped_triangle_count(10)
        self.assertEqual(unknown.exception.status, ApiStatus.INVALID_ARGUMENT)

    def test_hits_report_the_material_the_population_assigns(self) -> None:
        # Two triangles at y = 0: triangle 0 (x < 0) has its own material, triangle 1 (x > 0)
        # has zero and reports its instance's. The sphere reports its object's material.
        self._register_spheres()
        self.adapter.register_triangle_meshes(
            definition_values=_u64([_DEFINITION]),
            vertex_offsets=_u64([0]),
            vertex_counts=_u64([6]),
            triangle_offsets=_u64([0]),
            triangle_counts=_u64([2]),
            vertices=np.array(
                [(-3, 0, -1), (-1, 0, -1), (-2, 0, 1), (1, 0, -1), (3, 0, -1), (2, 0, 1)],
                dtype=np.float32,
            ),
            triangle_indices=np.array([(0, 1, 2), (3, 4, 5)], dtype=np.uint32),
            double_sided=np.array([1], dtype=np.uint8),
            source_face_material_ids=_u64([2000, 0]),
        ).finish()
        self.addCleanup(
            self.adapter.destroy_definition,
            GeometryDefinitionRef(GeometryType.TRIANGLE_MESH, _DEFINITION),
        )
        families = ((1, GeometryType.SPHERE, 10.0, 0.0), (2, GeometryType.TRIANGLE_MESH, 0.0, 0.0))
        population = self._population()
        population.enqueue_set(1, **_instances(families), material_ids=_u64([1000, 3000])).finish()
        target = population.target
        self.assertEqual(target.raycast_closest(_down(10)).hit.material_id, 1000)
        self.assertEqual(target.raycast_closest(_down(-2)).hit.material_id, 2000)
        self.assertEqual(target.raycast_closest(_down(2)).hit.material_id, 3000)
        candidate = RaycastQuery(
            (-2, 10, 0), (0, -1, 0), 20, precision=QueryPrecision.BOUNDS_CANDIDATE
        )
        self.assertEqual(target.raycast_closest(candidate).hit.material_id, 3000)

        plan = self.instance.create_raycast_plan(
            RaycastPlanConfig(
                target.reference,
                HitMode.CLOSEST,
                3,
                1,
                3,
                precision=QueryPrecision.EXACT,
                result_fields=BatchResultField.MATERIAL,
            )
        )
        self.addCleanup(plan.close)
        operation, _ = plan.enqueue(
            origins=np.array([(10, 10, 0), (-2, 10, 0), (2, 10, 0)], dtype=np.float64),
            unit_directions=np.array([(0, -1, 0)] * 3, dtype=np.float32),
            max_distances=np.full(3, 20.0),
        )
        self.assertIs(operation.wait(), OpState.COMPLETED)
        result = operation.fetch_raycast_batch().to_numpy()
        operation.close()
        self.assertEqual(
            result.material_ids[result.hit_ranges.hit_offsets].tolist(), [1000, 2000, 3000]
        )

        # A material belongs to the version: poses rows keep it and refuse a new one.
        with self.assertRaisesRegex(ValueError, "material_ids"):
            population.enqueue_update(
                1,
                2,
                kind=ExplicitUpdateKind.POSES,
                object_values=_u64([1]),
                translations=np.array([(10.0, 0.0, 0.0)]),
                rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)]),
                material_ids=_u64([7]),
            )
        population.enqueue_update(
            1, 2, **_instances(families[:1]), material_ids=_u64([1111]), removed_object_values=None
        ).finish()
        self.assertEqual(target.raycast_closest(_down(10)).hit.material_id, 1111)

    def test_float32_translations_are_widened_exactly(self) -> None:
        # A float32 translation column is an input only: 2**20 + 0.5 is exact in float32, and
        # the sphere is placed there in float64, so a ray grazing x + 0.75 still hits it.
        self._register_spheres()
        population = self._population()
        x = 2.0**20 + 0.5
        rows = {
            **_instances(_FAMILIES[:1]),
            "translations": np.array([(x, 0.0, 0.0)], dtype=np.float32),
        }
        population.enqueue_set(1, **rows).finish()
        self.assertAlmostEqual(population.target.raycast_closest(_down(x)).hit.distance, 9.0, 5)
        self.assertIsNotNone(population.target.raycast_closest(_down(x + 0.75)).hit)
        self.assertIsNone(population.target.raycast_closest(_down(x + 1.25)).hit)

    def test_read_only_columns_are_inputs(self) -> None:
        # The library only reads input columns, so read-only NumPy arrays are columns too: a
        # broadcast view, and arrays whose write flag was cleared.
        def frozen(value):
            value.setflags(write=False)
            return value

        self.adapter.register_boxes(
            definition_values=_u64([_DEFINITION]), half_extents=np.broadcast_to(0.5, (1, 3))
        ).finish()
        population = self._population()
        rows = {
            name: frozen(value)
            for name, value in _instances(((1, GeometryType.BOX, 5.0, 0.0),)).items()
        }
        population.enqueue_set(1, **rows).finish()
        self.assertAlmostEqual(population.target.raycast_closest(_down(5.25)).hit.distance, 9.5)
        self.assertIsNone(population.target.raycast_closest(_down(5.75)).hit)

    def _zero_length_capsule(self):
        """Publishes a capsule with half segment 0, a unit sphere at x = 5. Its quarter turn
        about z would lay a nonzero segment along x, under the ray at 6.2."""
        self.adapter.register_capsules(
            definition_values=_u64([_DEFINITION]), radii=np.array([1.0]), half_segments=np.zeros(1)
        ).finish()
        self.addCleanup(
            self.adapter.destroy_definition,
            GeometryDefinitionRef(GeometryType.CAPSULE, _DEFINITION),
        )
        turn = math.sqrt(0.5)
        rows = {
            **_instances(((1, GeometryType.CAPSULE, 5.0, 0.0),)),
            "rotations_xyzw": np.array([(0.0, 0.0, turn, turn)]),
        }
        population = self._population()
        population.enqueue_set(1, **rows).finish()
        return population, rows

    def _plan_hits(self, target, precision, stream=None):
        """Per-query hit counts, object IDs and distances of the rays at 5.6 and 6.2 through a
        CPU plan, or a CUDA plan on `stream`."""
        cpu = stream is None
        columns = (lambda value: value) if cpu else (lambda value: wp.array(value, device="cuda:0"))
        config = RaycastPlanConfig(
            target.reference,
            HitMode.CLOSEST,
            2,
            1,
            2,
            precision=precision,
            device=Device() if cpu else Device.cuda(),
            result_fields=BatchResultField.DISTANCE,
        )
        with self.instance.create_raycast_plan(config) as plan:
            operation, _ = plan.enqueue(
                origins=columns(np.array([(5.6, 10.0, 0.0), (6.2, 10.0, 0.0)])),
                unit_directions=columns(np.array([(0, -1, 0)] * 2, dtype=np.float32)),
                max_distances=columns(np.array([20.0, 20.0])),
                **({} if cpu else {"cuda_stream": stream}),
            )
            self.assertIs(operation.wait(), OpState.COMPLETED)
            result = operation.fetch_raycast_batch()
            if cpu:
                with operation:
                    result = result.to_numpy()
                    contact = result.exact or result.candidate
                    return (
                        result.hit_ranges.hit_counts.tolist(),
                        result.object_ids.tolist(),
                        contact.distances.tolist(),
                    )
            contact = result.exact or result.candidate
            wp.wait_event(wp.Event(wp.get_device("cuda:0"), cuda_event=result.common.result_ready))
            found = tuple(
                wp.from_dlpack(column).numpy().tolist()
                for column in (result.hit_ranges.hit_counts, result.object_ids, contact.distances)
            )
            done = wp.record_event()
            wp.synchronize_event(done)
            operation.close_after_cuda(done.cuda_event)
            done.owner = False  # Retirement transfers the event to the library.
            return found

    def _assert_sphere_hit(self, found, distance) -> None:
        hit_counts, object_ids, distances = found
        self.assertEqual((hit_counts, object_ids), ([1, 0], [1]))
        self.assertAlmostEqual(distances[0], distance, places=5)

    def test_supports_answers_for_the_devices_the_instance_can_use(self) -> None:
        cpu = Device.cpu()
        for capability in Capability:
            with self.subTest(capability=capability):
                self.assertEqual(
                    self.instance.supports(capability, cpu),
                    capability is not Capability.PREPARE_ON_DEVICE,
                )
        # setUp configures CUDA device 0 exactly when Warp sees one; publication on it is
        # proven by test_cuda_columns_publish_on_the_device.
        cuda = wp is not None and wp.is_cuda_available()
        self.assertEqual(self.instance.supports(Capability.PREPARE_ON_DEVICE, Device.cuda()), cuda)
        self.assertFalse(self.instance.supports(Capability.RAYCAST_PLAN, Device.cuda(1 << 20)))
        with self.assertRaises(ApiError) as malformed:
            self.instance.supports(Capability.RAYCAST_PLAN, Device(raw.K_DL_CPU, 1))
        self.assertEqual(malformed.exception.status, ApiStatus.INVALID_ARGUMENT)
        self.assertIs(Capability, raw.Capability)

    def test_a_zero_length_capsule_answers_as_a_sphere(self) -> None:
        target = self._zero_length_capsule()[0].target
        # The ray at 5.6 enters the sphere at y = 0.8.
        self.assertAlmostEqual(target.raycast_closest(_down(5.6)).hit.distance, 9.2, places=5)
        self.assertIsNone(target.raycast_closest(_down(6.2)).hit)
        box = BoxSweepQuery((5, 10, 0), (0.5, 0.5, 0.5), (0, 0, 0, 1), (0, -1, 0), 20)
        self.assertAlmostEqual(target.sweep_box_closest(box).hit.distance, 8.5, places=5)
        self._assert_sphere_hit(self._plan_hits(target, QueryPrecision.EXACT), 9.2)

    @unittest.skipIf(wp is None, "Warp supplies CUDA columns and streams")
    def test_a_zero_length_capsule_answers_as_a_sphere_on_cuda(self) -> None:
        if not wp.is_cuda_available():
            self.skipTest("no CUDA device")
        population, rows = self._zero_length_capsule()
        cuda = wp.get_device("cuda:0")
        stream = wp.Stream(cuda)
        on_device = self._population()
        with wp.ScopedStream(stream):
            found = self._plan_hits(population.target, QueryPrecision.EXACT, stream)
            self._assert_sphere_hit(found, 9.2)
            # A device-prepared version answers from the bounds it derives: the sphere's
            # box tops out at y = 1 and ends at x = 6.
            device_rows = {name: wp.array(value, device=cuda) for name, value in rows.items()}
            ready = wp.record_event()
            on_device.enqueue_set(
                1,
                **device_rows,
                preparation=Preparation.ON_DEVICE,
                cuda_stream=stream,
                cuda_wait_event=ready,
            ).finish()
            found = self._plan_hits(on_device.target, QueryPrecision.BOUNDS_CANDIDATE, stream)
            self._assert_sphere_hit(found, 9.0)

    def test_borrowed_columns_live_until_close_and_copied_columns_do_not(self) -> None:
        # A destroyed definition's value is not registered again, so each pass uses its own.
        for copy, definition in ((False, 7), (True, 8)):
            with self.subTest(copy=copy):
                radii = np.array([1.0])
                translations = np.array([(5.0, 0.0, 0.0)])
                watched = (weakref.ref(radii), weakref.ref(translations))
                registration = self.adapter.register_spheres(
                    definition_values=_u64([definition]), radii=radii, copy=copy
                )
                # A set may only name a definition whose registration has completed.
                self.assertIs(registration.wait(), OpState.COMPLETED)
                population = self.adapter.create_population()
                publication = population.enqueue_set(
                    1,
                    **{
                        **_instances(_FAMILIES[:1]),
                        "geometry_values": _u64([definition]),
                        "translations": translations,
                    },
                    copy=copy,
                )
                del radii, translations
                gc.collect()
                self.assertEqual([ref() is None for ref in watched], [copy, copy])
                registration.finish()
                publication.finish()
                gc.collect()
                self.assertEqual([ref() for ref in watched], [None, None])
                population.close()
                self.adapter.destroy_definition(
                    GeometryDefinitionRef(GeometryType.SPHERE, definition)
                )

    def test_columns_are_checked_before_the_library_sees_them(self) -> None:
        self._register_spheres()
        population = self._population()
        adapter = self.adapter
        good_sphere = {"definition_values": _u64([8]), "radii": np.array([1.0])}
        for label, call, error, text in (
            (
                "dtype",
                lambda: adapter.register_spheres(
                    **{**good_sphere, "radii": np.array([1.0], np.float32)}
                ),
                ValueError,
                "radii must be float64",
            ),
            (
                "width",
                lambda: adapter.register_boxes(
                    definition_values=_u64([8]), half_extents=np.ones((1, 4))
                ),
                ValueError,
                r"half_extents must have shape \(G, 3\)",
            ),
            (
                "rows",
                lambda: adapter.register_spheres(**{**good_sphere, "radii": np.ones(2)}),
                ValueError,
                "radii has 2 rows; definition_values has 1",
            ),
            (
                "protocol",
                lambda: adapter.register_spheres(**{**good_sphere, "radii": [1.0]}),
                TypeError,
                "radii must support DLPack",
            ),
            (
                "topology",
                lambda: adapter.register_convexes(
                    definition_values=_u64([8]),
                    vertex_offsets=_u64([0]),
                    vertex_counts=_u64([4]),
                    vertices=np.eye(4, 3, dtype=np.float32),
                    face_offsets=_u64([0]),
                ),
                ValueError,
                "topology",
            ),
            (
                "instance dtype",
                lambda: population.enqueue_set(
                    1, **{**_instances(), "geometry_types": np.ones(5, np.int64)}
                ),
                ValueError,
                "geometry_types must be int32",
            ),
            (
                "poses binding",
                lambda: population.enqueue_update(
                    1, 2, **_instances(), kind=ExplicitUpdateKind.POSES
                ),
                ValueError,
                "POSES update takes no geometry_types",
            ),
            (
                "poses bits",
                lambda: population.enqueue_update(
                    1,
                    2,
                    object_values=_u64([1]),
                    translations=np.zeros((1, 3)),
                    rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)]),
                    kind=ExplicitUpdateKind.POSES_AND_BITS,
                ),
                ValueError,
                "POSES_AND_BITS update requires participation_bits",
            ),
            (
                "CPU stream",
                lambda: population.enqueue_set(1, **_instances(), cuda_stream=1),
                ValueError,
                "CPU publication takes no cuda_stream",
            ),
            (
                "CPU preparation",
                lambda: population.enqueue_set(
                    1, **_instances(), preparation=Preparation.ON_DEVICE
                ),
                ValueError,
                "ON_DEVICE requires CUDA columns",
            ),
            (
                "view ids",
                lambda: population.create_view(np.array([1.0])),
                ValueError,
                "object_ids must be uint64",
            ),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(error, text):
                    call()
        # The same population still takes its first set, so no mistake reached the library.
        population.enqueue_set(1, **_instances(_FAMILIES[:1])).finish()
        self.assertEqual(population.info().head_ordinal, 1)

    def test_operation_blocks_wait_and_keep_the_error_raised_inside(self) -> None:
        self._register_spheres()
        population = self._population()
        # Leaving the block waits for the publication; closing it while pending is BUSY.
        with population.enqueue_set(1, **_instances(_FAMILIES[:1])) as publication:
            pass
        self.assertEqual(publication.op_id, 0)
        self.assertEqual(population.target.raycast_closest(_down(5)).hit.object_id, 1)
        with self.instance.create_raycast_plan(
            RaycastPlanConfig(population.target.reference, HitMode.CLOSEST, 1, 1, 1)
        ) as plan:
            # A live export makes the close fail, and the body's own error propagates.
            with self.assertRaisesRegex(LookupError, "inside the block"):
                with plan.enqueue(
                    origins=np.array([(5.0, 10.0, 0.0)]),
                    unit_directions=np.array([(0, -1, 0)], dtype=np.float32),
                    max_distances=np.array([20.0]),
                )[0] as operation:
                    operation.wait()
                    held = np.from_dlpack(operation.fetch_raycast_batch().object_ids)
                    raise LookupError("inside the block")
            self.assertEqual(held.tolist(), [1])
            del held
            operation.close()

    def test_wait_zero_polls_a_pending_operation(self) -> None:
        self._register_spheres()
        population = self._population()
        population.enqueue_set(1, **_instances(_FAMILIES[:1])).finish()
        rows = 100000
        with self.instance.create_raycast_plan(
            RaycastPlanConfig(population.target.reference, HitMode.CLOSEST, rows, 1, rows)
        ) as plan:
            # A launch this size runs for milliseconds, so its first poll almost always finds
            # it pending; a few launches make sure one does. No poll raises.
            first_polls = []
            while OpState.PENDING not in first_polls and len(first_polls) < 5:
                operation, _ = plan.enqueue(
                    origins=np.tile((5.0, 10.0, 0.0), (rows, 1)),
                    unit_directions=np.tile(np.array((0, -1, 0), np.float32), (rows, 1)),
                    max_distances=np.full(rows, 20.0),
                )
                with operation:
                    states = [operation.wait(0)]
                    while states[-1] is OpState.PENDING:
                        states.append(operation.wait(0))
                    self.assertIs(states[-1], OpState.COMPLETED)
                    ids = np.from_dlpack(operation.fetch_raycast_batch().object_ids)
                    self.assertEqual(ids[[0, -1]].tolist(), [1, 1])
                    del ids
                first_polls.append(states[0])
            self.assertIn(OpState.PENDING, first_polls)

    def test_a_instance_closes_what_it_still_owns(self) -> None:
        def scene(instance, origins=None):
            adapter = instance.explicit()
            # A set may only name a completed registration; this one stays open.
            adapter.register_spheres(
                definition_values=_u64([_DEFINITION]), radii=np.array([1.0])
            ).wait()
            # This one is first seen to complete while the instance closes.
            adapter.register_spheres(definition_values=_u64([8]), radii=np.array([1.0]))
            population = adapter.create_population()
            # Query admission needs a ready head; keep the completed operation open.
            population.enqueue_set(1, **_instances(_FAMILIES[:1])).wait()
            population.create_view(_u64([1]))
            plan = instance.create_raycast_plan(
                RaycastPlanConfig(population.target.reference, HitMode.CLOSEST, 1, 1, 1)
            )
            return plan.enqueue(
                origins=np.array([(5.0, 10.0, 0.0)]) if origins is None else origins,
                unit_directions=np.array([(0, -1, 0)], dtype=np.float32),
                max_distances=np.array([20.0]),
            )[0]

        # A normal exit closes everything left open and destroys the instance.
        with self.library.instance() as instance:
            operation = scene(instance)
        self.assertEqual((instance.handle, operation.op_id), (0, 0))

        # A raising block closes what it can, and its own error propagates unchained.
        with self.assertRaises(LookupError) as raised:
            with self.library.instance() as instance:
                scene(instance)
                raise LookupError("inside the block")
        self.assertIsNone(raised.exception.__context__)
        self.assertEqual(instance.handle, 0)

        # What cannot be closed becomes a note, and the instance is kept until it can be.
        with self.assertRaises(LookupError) as raised:
            with self.library.instance() as instance:
                operation = scene(instance)
                operation.wait()
                held = np.from_dlpack(operation.fetch_raycast_batch().object_ids)
                raise LookupError("inside the block")
        notes = "\n".join(raised.exception.__notes__)
        self.assertIn("could not close a Operation", notes)
        self.assertIn("kept the instance: ovspatialquery_destroy_instance: BUSY", notes)
        self.assertNotEqual(instance.handle, 0)
        del held
        instance.close()
        self.assertEqual(instance.handle, 0)

        # A normal exit that cannot close raises, and the dropped instance keeps what the
        # library borrowed until a later close succeeds.
        origins = np.array([(5.0, 10.0, 0.0)])
        watched = weakref.ref(origins)
        with self.assertRaises(BufferError):
            with self.library.instance() as instance:
                operation = scene(instance, origins)
                operation.wait()
                held = np.from_dlpack(operation.fetch_raycast_batch().object_ids)
        kept = weakref.ref(instance)
        del instance, operation, origins
        gc.collect()
        self.assertIsNotNone(watched())
        del held
        kept().close()
        gc.collect()
        self.assertIsNone(watched())

    def test_closing_a_instance_scales_linearly_with_what_it_owns(self) -> None:
        def close_seconds(count: int) -> float:
            best = float("inf")
            for _ in range(3):
                instance = self.library.instance()
                instance.explicit().register_spheres(
                    definition_values=np.arange(1, count + 1, dtype=np.uint64),
                    radii=np.ones(count),
                ).finish()
                start = time.perf_counter()
                instance.close()
                best = min(best, time.perf_counter() - start)
            return best

        # 16 times the definitions costs about 16 times as long; a close that rescans what it
        # owns per item costs about 256 times as long.
        self.assertLess(close_seconds(8000) / close_seconds(500), 64)

    @unittest.skipIf(wp is None, "Warp supplies CUDA columns and streams")
    def test_cuda_columns_publish_on_the_device(self) -> None:
        if not wp.is_cuda_available():
            self.skipTest("no CUDA device")
        self._register_spheres()
        population = self._population()
        device = wp.get_device("cuda:0")
        stream = wp.Stream(device)
        with wp.ScopedStream(stream):
            rows = {
                name: wp.array(value, device=device)
                for name, value in _instances(_FAMILIES[:1]).items()
            }
            dropped = wp.array(np.array([(5.0, -3.0, 0.0)]), device=device)
            ready = wp.record_event()
        with self.assertRaisesRegex(ValueError, "CUDA publication requires the cuda_stream"):
            population.enqueue_set(1, **rows, cuda_wait_event=ready)
        with self.assertRaisesRegex(ValueError, "translations is on .*; object_values is on"):
            population.enqueue_set(
                1,
                **{**rows, "translations": np.array([(5.0, 0, 0)])},
                cuda_stream=stream,
                cuda_wait_event=ready,
            )
        with self.assertRaisesRegex(ValueError, "radii is on .*; a registration reads CPU"):
            self.adapter.register_spheres(
                definition_values=_u64([8]),
                radii=wp.array(np.array([1.0]), device=device),
            )
        # The columns were written on `stream`, which alone orders the library's reads.
        population.enqueue_set(
            1, **rows, preparation=Preparation.ON_DEVICE, cuda_stream=stream
        ).finish()
        # A device-prepared version has no host objects for a CPU reader.
        with self.assertRaises(ApiError) as refused:
            population.target.raycast_closest(_down(5))
        self.assertIs(refused.exception.status, ApiStatus.NOT_SUPPORTED)

        # A host-prepared set that fails on the worker, here on a zero identity, is closed
        # by finish, which raises the failure.
        with wp.ScopedStream(stream):
            zero = wp.array(_u64([0]), device=device)
            zero_ready = wp.record_event()
        failing = self._population().enqueue_set(
            1, **{**rows, "object_values": zero}, cuda_stream=stream, cuda_wait_event=zero_ready
        )
        with self.assertRaises(ApiError) as failed:
            failing.finish()
        self.assertIs(failed.exception.status, ApiStatus.OPERATION_FAILED)
        # The detail is the operation's own terminal diagnostic, not the thread's last error.
        self.assertTrue(failed.exception.detail)
        self.assertEqual(failing.op_id, 0)

        # A POSES update that removes nothing still hands the library a removed column on
        # the device; the sphere drops by 3.
        population.enqueue_update(
            1,
            2,
            kind=ExplicitUpdateKind.POSES,
            object_values=rows["object_values"],
            translations=dropped,
            rotations_xyzw=rows["rotations_xyzw"],
            preparation=Preparation.ON_DEVICE,
            cuda_stream=stream,
            cuda_wait_event=ready,
        ).finish()
        plan = self.instance.create_raycast_plan(
            RaycastPlanConfig(
                population.target.reference,
                HitMode.CLOSEST,
                1,
                1,
                1,
                device=Device.cuda(),
                result_fields=BatchResultField.DISTANCE,
            )
        )
        with wp.ScopedStream(stream):
            operation, _ = plan.enqueue(
                origins=wp.array(np.array([(5.0, 10.0, 0.0)]), device=device),
                unit_directions=wp.array(np.array([(0, -1, 0)], dtype=np.float32), device=device),
                max_distances=wp.array(np.array([20.0]), device=device),
                cuda_stream=stream,
            )
        self.assertIs(operation.wait(), OpState.COMPLETED)
        result = operation.fetch_raycast_batch()
        wp.wait_event(wp.Event(device, cuda_event=result.common.result_ready))
        self.assertEqual(wp.from_dlpack(result.object_ids).numpy().tolist(), [1])
        with self.assertRaisesRegex(ValueError, "host results only"):
            result.to_numpy()
        # The bounds of the dropped sphere end at y = -2.
        distances = wp.from_dlpack(result.candidate.distances).numpy()
        self.assertAlmostEqual(float(distances[0]), 12.0, 5)
        # Only the caller can name the event that covers every reader of the columns.
        with self.assertRaisesRegex(RuntimeError, "close_after_cuda"):
            operation.finish()
        done = wp.record_event()
        wp.synchronize_event(done)
        operation.close_after_cuda(done.cuda_event)
        done.owner = False  # Retirement transfers the event to the library.
        plan.close()  # waits for the slot retired after the consumers

    def test_mesh_rays_do_not_depend_on_the_size_a_definition_is_authored_at(self) -> None:
        # The triangle (0,0,0) (2,0,0) (0,2,0) authored at size S, and the unit one placed with
        # uniform scale S. An oblique ray crosses their plane at S (1 + b, 1 + b), where the
        # barycentric of the corner opposite the hypotenuse is -b: it hits at 1.5 S exactly when
        # b < 0. A sphere of radius S / 2 and a cube of half extent 0.3 S centred at
        # S (0.5, 0.5, 0.25) overlap it; centred at S (1.5, 1.5, 0), beyond the hypotenuse, they
        # do not. A margin derived from the coordinates once let rays 10 % of an edge outside a
        # triangle authored at 1e6 hit it, and float products missed triangles authored at 1e-8
        # and 1e14.
        unit = np.array([(0, 0, 0), (2, 0, 0), (0, 2, 0)], dtype=np.float32)
        direction = np.array([0.4, 0.3, -0.866])
        direction /= np.linalg.norm(direction)
        for scale, authored in ((1e-8, 20), (1e4, 21), (1e6, 22), (1e14, 23)):
            self.adapter.register_triangle_meshes(
                definition_values=_u64([authored, authored + 10]),
                vertex_offsets=_u64([0, 3]),
                vertex_counts=_u64([3, 3]),
                triangle_offsets=_u64([0, 1]),
                triangle_counts=_u64([1, 1]),
                vertices=np.concatenate([unit * np.float32(scale), unit]),
                triangle_indices=np.array([(0, 1, 2)] * 2, dtype=np.uint32),
            ).finish()
            for value, placed_scale in ((authored, 1.0), (authored + 10, scale)):
                self.addCleanup(
                    self.adapter.destroy_definition,
                    GeometryDefinitionRef(GeometryType.TRIANGLE_MESH, value),
                )
                population = self._population()
                population.enqueue_set(
                    1,
                    object_values=_u64([1]),
                    geometry_types=np.array([GeometryType.TRIANGLE_MESH], dtype=np.int32),
                    geometry_values=_u64([value]),
                    translations=np.zeros((1, 3)),
                    rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)]),
                    uniform_scales=np.array([placed_scale]),
                ).finish()
                for margin in (-1e-2, -1e-4, 1e-4, 1e-2, 0.1):
                    with self.subTest(scale=scale, authored=placed_scale == 1.0, margin=margin):
                        crossing = np.array([1 + margin, 1 + margin, 0.0])
                        query = RaycastQuery(
                            tuple((crossing - 1.5 * direction) * scale), tuple(direction), 3 * scale
                        )
                        hit = population.target.raycast_closest(query).hit
                        crossed = population.target.raycast_multiple(query, 4).hits
                        self.assertEqual(
                            (hit is not None, len(crossed)), (margin < 0, int(margin < 0))
                        )
                        if hit is not None:
                            self.assertAlmostEqual(hit.distance / scale, 1.5, places=5)
                            self.assertGreaterEqual(min(hit.barycentrics), -1e-6)
                for inside in (True, False):
                    with self.subTest(
                        scale=scale, authored=placed_scale == 1.0, overlap_inside=inside
                    ):
                        center = tuple(
                            scale * np.array((0.5, 0.5, 0.25) if inside else (1.5, 1.5, 0))
                        )
                        for shape in (
                            Sphere(center, 0.5 * scale),
                            OrientedBox(center, (0.0, 0.0, 0.0, 1.0), (0.3 * scale,) * 3),
                        ):
                            found = population.target.overlap_any(OverlapQuery(shape)).hit
                            self.assertEqual(found is not None, inside, shape)

    def test_convex_queries_do_not_depend_on_the_size_a_definition_is_authored_at(self) -> None:
        # The hull of (0,0,0) (2,0,0) (0,2,0) (0,0,2) authored at size S, and the unit one
        # placed with uniform scale S. A ray falling from z = 3 S at S (0.5, 0.5) meets the face
        # x + y + z = 2 S after 2 S; at S (1.1, 1.1) it passes beside the hull. A sphere of
        # radius 0.2 S and a cube of half extent 0.2 S centred at S (0.5, 0.5, 0.5) overlap it;
        # centred at S (1.5, 1.5, 1.5), 1.44 S beyond that face, they do not. PhysX's absolute
        # hull tolerances once replaced hulls authored at 1e-8 or 1e10 by other shapes.
        unit = np.array([(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)], dtype=np.float32)
        for scale, authored in ((1e-8, 30), (1e4, 31), (1e10, 32), (1e14, 33)):
            self.adapter.register_convexes(
                definition_values=_u64([authored, authored + 10]),
                vertex_offsets=_u64([0, 4]),
                vertex_counts=_u64([4, 4]),
                vertices=np.concatenate([unit * np.float32(scale), unit]),
            ).finish()
            for value, placed_scale in ((authored, 1.0), (authored + 10, scale)):
                self.addCleanup(
                    self.adapter.destroy_definition,
                    GeometryDefinitionRef(GeometryType.CONVEX, value),
                )
                population = self._population()
                population.enqueue_set(
                    1,
                    object_values=_u64([1]),
                    geometry_types=np.array([GeometryType.CONVEX], dtype=np.int32),
                    geometry_values=_u64([value]),
                    translations=np.zeros((1, 3)),
                    rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)]),
                    uniform_scales=np.array([placed_scale]),
                ).finish()
                for inside in (True, False):
                    with self.subTest(scale=scale, authored=placed_scale == 1.0, inside=inside):
                        x = 0.5 if inside else 1.1
                        down = RaycastQuery(
                            (x * scale, x * scale, 3 * scale), (0, 0, -1), 4 * scale
                        )
                        hit = population.target.raycast_closest(down).hit
                        self.assertEqual(hit is not None, inside)
                        if hit is not None:
                            self.assertAlmostEqual(hit.distance / scale, 2.0, places=5)
                        center = tuple(scale * np.array((0.5,) * 3 if inside else (1.5,) * 3))
                        for shape in (
                            Sphere(center, 0.2 * scale),
                            OrientedBox(center, (0.0, 0.0, 0.0, 1.0), (0.2 * scale,) * 3),
                        ):
                            found = population.target.overlap_any(OverlapQuery(shape)).hit
                            self.assertEqual(found is not None, inside, shape)

    def test_a_hull_beyond_255_vertices_or_faces_is_refused(self) -> None:
        # A hull may have at most 255 vertices and 254 faces. A 128-gon prism has 256
        # vertices, 130 points spread over a sphere have a hull of 2 * 130 - 4 = 256
        # triangles, and a 254-gon pyramid has 255 faces; every reader once queried them as
        # the box of their points. A 127-gon prism with one more point just outside the side
        # face at x = -1 has 255 vertices and 132 faces, 129 points on the sphere have 254
        # faces, and a cube with 300 points inside it has 8 vertices: each registers and is
        # queried as its hull. The 128-gon prism 1e-3 wide in x and y keeps each corner only
        # 6e-7 of the height outside its neighbours' chord, so PhysX's hull leaves corners out
        # before reaching the vertex limit: it is refused for enclosure, with the distance of
        # its farthest corner, and registers when the call allows 2^-19 (ADR-0078).
        def prism(sides):
            angle = 2 * np.pi * np.arange(sides) / sides
            ring = np.stack([np.cos(angle), np.sin(angle)], 1)
            return np.vstack([np.column_stack([ring, np.full(sides, z)]) for z in (-1.0, 1.0)])

        def sphere(count):
            z = 1 - 2 * (np.arange(count) + 0.5) / count
            angle = 2.3999632297286533 * np.arange(count)
            radius = np.sqrt(1 - z * z)
            return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])

        def register(value, points, **options):
            self.adapter.register_convexes(
                definition_values=_u64([value]),
                vertex_offsets=_u64([0]),
                vertex_counts=_u64([len(points)]),
                vertices=np.asarray(points, dtype=np.float32),
                **options,
            ).finish()
            self.addCleanup(
                self.adapter.destroy_definition, GeometryDefinitionRef(GeometryType.CONVEX, value)
            )
            population = self._population()
            population.enqueue_set(
                1,
                object_values=_u64([1]),
                geometry_types=np.array([GeometryType.CONVEX], dtype=np.int32),
                geometry_values=_u64([value]),
                translations=np.zeros((1, 3)),
                rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)]),
            ).finish()
            return lambda origin, direction: (
                population.target.raycast_closest(
                    RaycastQuery(origin, direction, 20.0)
                ).hit.distance
            )

        pyramid = np.vstack([prism(254)[:254] * (1, 1, 0), [(0.0, 0.0, 1.0)]])
        for value, points, limit in (
            (2, prism(128), "its hull has more than 255 vertices"),
            (8, prism(128) * (1e-3, 1e-3, 1), "no hull with its points as vertices encloses"),
            (3, sphere(130), "its hull has more than 254 faces"),
            (7, pyramid, "its hull has more than 254 faces"),
        ):
            with self.subTest(limit=limit):
                with self.assertRaises(ApiError) as refused:
                    register(value, points)
                self.assertEqual(refused.exception.status, ApiStatus.INVALID_ARGUMENT)
                self.assertIn(f"convex definition {value}: {limit}", refused.exception.detail)
                if value == 8:
                    # Its farthest corner lies between 2^-20 (9.5e-7) and 2^-19 (1.9e-6) outside.
                    self.assertIn(
                        "its farthest point lies 1.2e-06 of the half extent outside, beyond the "
                        "tolerance 9.54e-07",
                        refused.exception.detail,
                    )
        register(9, prism(128) * (1e-3, 1e-3, 1), enclosure_tolerance=2**-19)

        # A ray toward the axis at 45 degrees meets the 127-gon's edge whose midpoint lies at
        # angle psi, apothem cos(pi / 127) from the axis; the ray along x meets the extra point.
        ray = register(4, np.vstack([prism(127), [(-1.0, 0.0, 0.0)]]))
        psi = np.pi * (2 * math.floor(math.pi / 4 / (2 * np.pi / 127)) + 1) / 127
        radius = math.cos(np.pi / 127) / math.cos(math.pi / 4 - psi)
        diagonal = math.sqrt(0.5)
        self.assertAlmostEqual(
            ray((10 * diagonal, 10 * diagonal, 0), (-diagonal, -diagonal, 0)), 10 - radius, places=5
        )
        self.assertAlmostEqual(ray((-10, 0, 0), (1, 0, 0)), 9.0, places=5)
        ray = register(5, sphere(129))
        for direction in np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]):
            unit = direction / np.linalg.norm(direction)
            self.assertGreaterEqual(ray(tuple(10 * unit), tuple(-unit)), 9 - 1e-6)
        inside = np.random.default_rng(28).uniform(-0.9, 0.9, (300, 3))
        cube = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        ray = register(6, np.vstack([cube, inside]))
        self.assertAlmostEqual(ray((0.3, 0.2, 10), (0, 0, -1)), 9.0, places=6)

    def test_a_hull_vertex_limit_registers_the_hull_physx_simulates(self) -> None:
        # The larger carton's own hull has 255 faces and is refused. Under a hull vertex limit
        # it registers as the hull PhysX cooks for that collider (ADR-0078), which may leave
        # points outside. A sphere of 1e-6 of the half extent on each point overlaps the hull
        # only where the hull reaches it; every point overlaps the points' own hull, so a
        # library that ignored the limit would report them all. At 64 PhysX keeps 34 of the
        # points and leaves 155 out; at 8 it keeps an oriented box that leaves 8 out, so the
        # two limits answer differently. A ray from beyond each face of the points' box, aimed
        # at the box centre, stops at or behind that face. The call refuses a limit or
        # tolerance outside its range, names the value and admits nothing, so the same
        # definition value then registers.
        carton = np.frombuffer(
            base64.b64decode("".join(_CARTON_BEYOND_THE_FACE_LIMIT)), "<f4"
        ).reshape(-1, 3)
        low, high = carton.min(0).astype(np.float64), carton.max(0).astype(np.float64)
        centre, half = 0.5 * (low + high), 0.5 * float((high - low).max())

        def register(value, **options):
            self.adapter.register_convexes(
                definition_values=_u64([value]),
                vertex_offsets=_u64([0]),
                vertex_counts=_u64([len(carton)]),
                vertices=carton,
                **options,
            ).finish()
            self.addCleanup(
                self.adapter.destroy_definition, GeometryDefinitionRef(GeometryType.CONVEX, value)
            )
            population = self._population()
            population.enqueue_set(
                1,
                object_values=_u64([1]),
                geometry_types=np.array([GeometryType.CONVEX], dtype=np.int32),
                geometry_values=_u64([value]),
                translations=np.zeros((1, 3)),
                rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)]),
            ).finish()
            return population.target

        def refusal(**options):
            with self.assertRaises(ApiError) as refused:
                register(60, **options)
            self.assertEqual(refused.exception.status, ApiStatus.INVALID_ARGUMENT)
            return refused.exception.detail

        self.assertIn("convex definition 60: its hull has more than 254 faces", refusal())
        for limit in (7, 65):
            self.assertIn(
                f"convex registration: hull vertex limit {limit} is not zero or between 8 and 64",
                refusal(hull_vertex_limit=limit),
            )
        # ctypes would wrap 2**32 + 8 to the valid 8; the descriptor refuses it instead.
        with self.assertRaises(ValueError):
            register(60, hull_vertex_limit=2**32 + 8)
        for tolerance in (2**-21, 2**-15, -(2**-20), math.nan):
            self.assertIn(
                "is not zero or between 2^-20 and 2^-16", refusal(enclosure_tolerance=tolerance)
            )

        def outside(target):
            return sum(
                target.overlap_any(OverlapQuery(Sphere(tuple(map(float, point)), 1e-6 * half))).hit
                is None
                for point in carton
            )

        left_out = {}
        for limit in (64, 8):
            target = register(60 + limit, hull_vertex_limit=limit)
            left_out[limit] = outside(target)
            for axis, side in itertools.product(range(3), (-1, 1)):
                start = centre.copy()
                start[axis] = (high if side > 0 else low)[axis] + side * half
                direction = tuple(-side * float(axis == k) for k in range(3))
                hit = target.raycast_closest(RaycastQuery(tuple(start), direction, 4 * half)).hit
                self.assertGreaterEqual(hit.distance, half - 1e-6 * half)
        self.assertGreater(left_out[64], 0)
        self.assertGreater(left_out[8], 0)
        self.assertNotEqual(left_out[8], left_out[64])
        # Registered with a valid limit, the refused value is free: nothing was admitted.
        register(60, hull_vertex_limit=64)

    def test_a_convex_whose_points_lie_in_its_faces_registers_as_their_hull(self) -> None:
        # PhysX's quickhull leaves out a point within its plane thickness of a face, a few float
        # steps of the hull's size. Float32 points in a face's plane, as on the carton or on a
        # chamfered box turned off the axes, lie off it by about that much, so a hull that left
        # them out was refused as if the points were flat. Each registers here at sizes 1e-3 to
        # 1e3 and answers as the hull of its points: for each face, rays and sphere sweeps
        # straight down onto the centre of its points stop at the face, a ray 2^-20 of the half
        # extent (eight float32 steps) above the face along it misses, and a sphere overlaps
        # when its centre is 2^-20 nearer the face than its radius and not when 2^-20 further;
        # PhysX's float32 sphere test reported up to 2e-4 there. A sphere centred on any of the
        # points overlaps, which PhysX's test missed near some corners. A box of half extent
        # 1e-3 of the half extent, one face toward an edge's outward bisector, overlaps when the
        # edge's midpoint lies 2^-20 inside that face and not when 2^-20 outside; PhysX's float32
        # box test missed 16 of the carton's 165 such boxes. Points 1e-7 or 3e-6 off a plane, the
        # header's nearly flat examples, are still refused.
        def chamfered_box(half, bevel, seed):
            # The box's 24 corners where its 12 chamfers meet its faces, and a 3 by 3 grid of
            # points on each face, turned by a random rotation.
            inner = np.subtract(half, bevel)
            corners = [
                np.multiply(sign, np.where(np.arange(3) == axis, half, inner))
                for axis in range(3)
                for sign in itertools.product((-1, 1), repeat=3)
            ]
            grid = np.array(list(itertools.product((-0.5, 0.0, 0.5), repeat=2)))
            faces = [
                np.insert(grid * np.delete(inner, axis), axis, side * half[axis], axis=1)
                for axis in range(3)
                for side in (-1, 1)
            ]
            rotation = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))[0]
            return np.vstack([corners, *faces]) @ rotation.T

        def register(value, points):
            self.adapter.register_convexes(
                definition_values=_u64([value]),
                vertex_offsets=_u64([0]),
                vertex_counts=_u64([len(points)]),
                vertices=points,
            ).finish()
            self.addCleanup(
                self.adapter.destroy_definition, GeometryDefinitionRef(GeometryType.CONVEX, value)
            )
            population = self._population()
            population.enqueue_set(
                1,
                object_values=_u64([1]),
                geometry_types=np.array([GeometryType.CONVEX], dtype=np.int32),
                geometry_values=_u64([value]),
                translations=np.zeros((1, 3)),
                rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)]),
            ).finish()
            return population.target

        carton = np.frombuffer(base64.b64decode("".join(_BEVELLED_CARTON)), "<f4").reshape(-1, 3)
        clouds = [("carton", carton)]
        for seed, size in enumerate((1e-3, 1e-1, 1e1, 1e3)):
            box = chamfered_box((0.5, 0.4, 0.3), 0.02, seed) + (0.2, -0.3, 0.1)
            clouds.append((f"chamfered box at {size}", (box * size).astype(np.float32)))
        for value, (name, points) in enumerate(clouds, start=40):
            with self.subTest(name):
                target = register(value, points)
                half = float(np.abs(points - points.mean(0)).max())
                margin, radius = 2**-20 * half, 0.1 * half
                for normal, centre in _hull_faces(points):
                    above, down = tuple(centre + half * normal), tuple(-normal)
                    hit = target.raycast_closest(RaycastQuery(above, down, 2 * half)).hit
                    self.assertAlmostEqual(hit.distance, half, delta=margin)
                    sweep = SphereSweepQuery(above, radius, down, 2 * half)
                    hit = target.sweep_sphere_closest(sweep).hit
                    self.assertAlmostEqual(hit.distance, half - radius, delta=margin)
                    along = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
                    along /= np.linalg.norm(along)
                    start = tuple(centre + margin * normal - 2 * half * along)
                    graze = RaycastQuery(start, tuple(along), 4 * half)
                    self.assertIsNone(target.raycast_closest(graze).hit)
                    for offset, overlaps in ((radius - margin, True), (radius + margin, False)):
                        sphere = Sphere(tuple(centre + offset * normal), radius)
                        found = target.overlap_any(OverlapQuery(sphere)).hit
                        self.assertEqual(found is not None, overlaps)
                for point in points:
                    sphere = Sphere(tuple(float(coordinate) for coordinate in point), 1e-3 * half)
                    self.assertIsNotNone(target.overlap_any(OverlapQuery(sphere)).hit, point)
                # An edge joins two faces that share at least two points.
                hull = points.astype(np.float64)
                faces = [
                    (n, hull @ n >= (hull @ n).max() - 1e-6 * half) for n, _ in _hull_faces(points)
                ]
                extent = 1e-3 * half
                for (first, on_first), (second, on_second) in itertools.combinations(faces, 2):
                    shared = hull[on_first & on_second]
                    if len(shared) < 2 or abs(first @ second) > 0.999:
                        continue
                    run = shared @ np.cross(first, second)
                    midpoint = 0.5 * (shared[run.argmin()] + shared[run.argmax()])
                    outward = (first + second) / np.linalg.norm(first + second)
                    # The quaternion turning +z to the bisector (or its opposite: the box is the same).
                    z = outward if outward[2] >= 0 else -outward
                    turn = np.array([-z[1], z[0], 0.0, 1.0 + z[2]])
                    turn = tuple(float(c) for c in turn / np.linalg.norm(turn))
                    for offset, overlaps in ((extent - margin, True), (extent + margin, False)):
                        box = OrientedBox(tuple(midpoint + offset * outward), turn, (extent,) * 3)
                        found = target.overlap_any(OverlapQuery(box)).hit
                        self.assertEqual(found is not None, overlaps, (midpoint, offset))

        angle = 2 * np.pi * np.arange(32) / 32
        ring = np.column_stack([np.cos(angle), np.sin(angle)])
        for value, points in (
            (50, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.3, 0.3, 1e-7)]),
            (51, np.vstack([np.column_stack([ring, np.full(32, z)]) for z in (0.0, 3e-6)])),
        ):
            with self.subTest(refused=value):
                with self.assertRaises(ApiError) as refused:
                    register(value, np.asarray(points, dtype=np.float32))
                self.assertEqual(refused.exception.status, ApiStatus.INVALID_ARGUMENT)
                self.assertIn(f"convex definition {value}: no hull", refused.exception.detail)

    def test_sweeps_ignore_single_sided_triangles_they_move_along(self) -> None:
        # ADR-0051 on a floor triangle (0,0,0) (4,0,0) (0,4,0), front +z, and a ridge of two
        # front-up triangles at x = 20. A sphere ignores a single-sided triangle with n . d > 0,
        # a box one with n . d >= 0; every other triangle stops a sweep at its first touch from
        # either side. Each case gives the single-sided and double-sided distance; None is no hit.
        floor = [(0, 0, 0), (4, 0, 0), (0, 4, 0)]
        ridge = [(0, -2, 0), (0, 2, 0), (-2, 0, -1), (2, 0, -1)]
        self.adapter.register_triangle_meshes(
            definition_values=_u64([11, 12, 13, 14]),
            vertex_offsets=_u64([0, 0, 3, 3]),
            vertex_counts=_u64([3, 3, 4, 4]),
            triangle_offsets=_u64([0, 0, 1, 1]),
            triangle_counts=_u64([1, 1, 2, 2]),
            vertices=np.array(floor + ridge, dtype=np.float32),
            triangle_indices=np.array([(0, 1, 2), (2, 0, 1), (3, 1, 0)], dtype=np.uint32),
            double_sided=np.array([0, 1, 0, 1], dtype=np.uint8),
        ).finish()
        targets = {}
        for double_sided, (floor_value, ridge_value) in ((False, (11, 13)), (True, (12, 14))):
            self.addCleanup(
                self.adapter.destroy_definition,
                GeometryDefinitionRef(GeometryType.TRIANGLE_MESH, floor_value),
            )
            self.addCleanup(
                self.adapter.destroy_definition,
                GeometryDefinitionRef(GeometryType.TRIANGLE_MESH, ridge_value),
            )
            population = self._population()
            population.enqueue_set(
                1,
                object_values=_u64([1, 2]),
                geometry_types=np.full(2, GeometryType.TRIANGLE_MESH, dtype=np.int32),
                geometry_values=_u64([floor_value, ridge_value]),
                translations=np.array([(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)]),
                rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)] * 2),
            ).finish()
            targets[double_sided] = population.target

        def sweeps(center, direction, both_sides=False):
            sphere = SphereSweepQuery(center, 1.0, direction, 5.0, both_sides=both_sides)
            box = BoxSweepQuery(
                center, (1, 1, 1), (0, 0, 0, 1), direction, 5.0, both_sides=both_sides
            )
            return (lambda t: t.sweep_sphere_closest(sphere).hit), (
                lambda t: t.sweep_box_closest(box).hit
            )

        diagonal = (math.sqrt(0.5), 0.0, -math.sqrt(0.5))
        # The unit sphere meets the edge x = 0 where (s - 1.5)^2 + (0.2 - s)^2 = 1, s = t / sqrt(2);
        # the box's face x = s - 0.5 reaches it at s = 0.5.
        rim_sphere = math.sqrt(2) * (3.4 - math.sqrt(1.24)) / 4
        cases = (
            ("falling past the rim", (-1.5, 1, 0.2), diagonal, (rim_sphere,) * 2, (0.5**0.5,) * 2),
            ("from above", (1, 1, 3), (0, 0, -1), (2.0, 2.0), (2.0, 2.0)),
            ("from below", (1, 1, -3), (0, 0, 1), (None, 2.0), (None, 2.0)),
            ("resting on it, leaving", (1, 1, 0.9), (0, 0, 1), (None, 0.0), (None, 0.0)),
            ("overlapping from behind, sliding", (1, 1, -0.3), (1, 0, 0), (0.0, 0.0), (None, 0.0)),
            ("overlapping from behind, sinking", (1, 1, -0.3), (0, 0, -1), (0.0, 0.0), (0.0, 0.0)),
        )
        for name, center, direction, sphere_expected, box_expected in cases:
            # ADR-0065: both_sides makes the single-sided floor answer as the double-sided one.
            for both_sides in (False, True):
                sphere, box = sweeps(center, direction, both_sides)
                for shape, run, expected in (
                    ("sphere", sphere, sphere_expected),
                    ("box", box, box_expected),
                ):
                    for double_sided, distance in zip((False, True), expected):
                        if both_sides:
                            distance = expected[1]
                        with self.subTest(
                            case=name, shape=shape, double_sided=double_sided, both_sides=both_sides
                        ):
                            hit = run(targets[double_sided])
                            if distance is None:
                                self.assertIsNone(hit)
                            else:
                                self.assertEqual(hit.object_id, 1)
                                self.assertAlmostEqual(hit.distance, distance, places=6)
        # A ray from below meets the single-sided floor only with MESH_BOTH_SIDES, and keeps
        # its front normal there, as PhysX does; the double-sided floor's faces the ray. A
        # plan launch takes the flag for every row; an unknown bit is refused.
        below = RaycastQuery((1, 1, -3), (0, 0, 1), 5.0)
        self.assertIsNone(targets[False].raycast_closest(below).hit)
        for query in (
            RaycastQuery((1, 1, -3), (0, 0, 1), 5.0, both_sides=True),
            RaycastQuery((1, 1, -3), (0, 0, 1), 5.0, query_flags=QueryFlag.MESH_BOTH_SIDES),
        ):
            hit = targets[False].raycast_closest(query).hit
            self.assertAlmostEqual(hit.distance, 3.0, places=6)
            self.assertEqual(tuple(hit.normal), (0.0, 0.0, 1.0))
            self.assertEqual(
                tuple(targets[True].raycast_closest(query).hit.normal), (0.0, 0.0, -1.0)
            )
        with self.assertRaises(TypeError):
            RaycastQuery((1, 1, -3), (0, 0, 1), 5.0, both_sides=1).to_raw(targets[False].reference)
        with self.assertRaises(ValueError):
            RaycastQuery((1, 1, -3), (0, 0, 1), 5.0, query_flags=QueryFlag(1 << 32)).to_raw(
                targets[False].reference
            )
        with self.assertRaises(ApiError) as refused:
            targets[False].raycast_closest(
                RaycastQuery((1, 1, -3), (0, 0, 1), 5.0, query_flags=QueryFlag(2))
            )
        self.assertEqual(refused.exception.status, ApiStatus.NOT_SUPPORTED)
        config = RaycastPlanConfig(
            targets[False].reference,
            HitMode.CLOSEST,
            2,
            1,
            2,
            precision=QueryPrecision.EXACT,
            result_fields=BatchResultField.DISTANCE,
        )
        with self.instance.create_raycast_plan(config) as plan:
            for both_sides, counts in ((False, [0, 0]), (True, [1, 1])):
                operation, _ = plan.enqueue(
                    origins=np.array([(1.0, 1.0, -3.0), (2.0, 1.0, -3.0)]),
                    unit_directions=np.array([(0, 0, 1)] * 2, dtype=np.float32),
                    max_distances=np.array([5.0, 5.0]),
                    both_sides=both_sides,
                )
                self.assertIs(operation.wait(), OpState.COMPLETED)
                with operation:
                    result = operation.fetch_raycast_batch().to_numpy()
                    self.assertEqual(result.hit_ranges.hit_counts.tolist(), counts)
                    if both_sides:
                        self.assertAlmostEqual(result.exact.distances[0], 3.0, places=6)
        # A contact from behind keeps the normal from the contact toward the sphere's center.
        sphere, _ = sweeps((1, 1, -0.3), (0, 0, -1))
        for axis, value in enumerate((0.0, 0.0, -1.0)):
            self.assertAlmostEqual(sphere(targets[False]).normal[axis], value, places=6)

        # The ridge's faces both point along a sphere rising through them from behind, so the
        # single-sided ridge is not hit; the double-sided one is touched one radius away.
        rise = SphereSweepQuery((18, -1, -3), 0.5, (1 / 5**0.5, 0, 2 / 5**0.5), 10.0)
        self.assertIsNone(targets[False].sweep_sphere_closest(rise).hit)
        hit = targets[True].sweep_sphere_closest(rise).hit
        self.assertEqual(hit.object_id, 2)
        self.assertAlmostEqual(math.dist(hit.position, hit.query_position), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
