# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real installed explicit-ingestion coverage for supported floating-point inputs."""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
from pathlib import Path
import unittest

import numpy as np

from ovspatialquery import raw
from ovspatialquery.explicit import GeometryDefinitionRef
from ovspatialquery.instance import ApiError
from ovspatialquery.library import Library
from ovspatialquery.types import Device, InstanceConfig, RaycastPlanConfig, RaycastQuery


def _u64(values, shape: tuple[int, ...]) -> np.ndarray:
    return np.array(values, dtype=np.uint64).reshape(shape)


def _u32(values, shape: tuple[int, ...]) -> np.ndarray:
    return np.array(values, dtype=np.uint32).reshape(shape)


def _i32(values, shape: tuple[int, ...]) -> np.ndarray:
    return np.array(values, dtype=np.int32).reshape(shape)


def _floating(values, bits: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.array(values, dtype=np.float32 if bits == 32 else np.float64).reshape(shape)


def _triangle_mesh(*columns) -> dict[str, np.ndarray]:
    names = (
        "definition_values",
        "vertex_offsets",
        "vertex_counts",
        "triangle_offsets",
        "triangle_counts",
        "vertices",
        "triangle_indices",
    )
    return dict(zip(names, columns, strict=True))


def _instances(*columns) -> dict[str, np.ndarray]:
    names = ("object_values", "geometry_types", "geometry_values", "translations", "rotations_xyzw")
    return dict(zip(names, columns, strict=True))


def _raycast_columns() -> dict[str, np.ndarray]:
    return {
        "origins": np.array([(-1.0, 0.0, 0.0)]),
        "unit_directions": np.array([(1.0, 0.0, 0.0)], dtype=np.float32),
        "min_distances": np.zeros(1),
        "max_distances": np.full(1, 2.0),
    }


def _cpu_values(view, element):
    count = math.prod(view.shape)
    pointer = ctypes.cast(view.data + view.byte_offset, ctypes.POINTER(element))
    return tuple(pointer[index] for index in range(count))


def _repeated_triangle_mesh(value: int, count: int = 1024) -> dict[str, np.ndarray]:
    vertices = (1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -1.0) * count
    return _triangle_mesh(
        _u64((value,), (1,)),
        _u64((0,), (1,)),
        _u64((3 * count,), (1,)),
        _u64((0,), (1,)),
        _u64((count,), (1,)),
        _floating(vertices, 32, (3 * count, 3)),
        _u32(tuple(range(3 * count)), (count, 3)),
    )


class ExplicitInstalledDtypeTests(unittest.TestCase):
    library_path: Path

    def _wait(self, operation) -> None:
        self.assertIs(operation.wait(), raw.OpState.COMPLETED)
        operation.close()

    def _populate(self, adapter, value: int):
        population = adapter.create_population()
        self._wait(
            population.enqueue_set(
                1,
                **_instances(
                    _u64((61,), (1,)),
                    _i32((raw.GeometryType.TRIANGLE_MESH,), (1,)),
                    _u64((value,), (1,)),
                    _floating((0.0, 0.0, 0.0), 64, (1, 3)),
                    _floating((0.0, 0.0, 0.0, 1.0), 64, (1, 4)),
                ),
            )
        )
        return population

    def _exact_raycast_plan(self, instance, population):
        return instance.create_raycast_plan(
            RaycastPlanConfig(
                population.target.reference,
                raw.HitMode.CLOSEST,
                1,
                1,
                1,
                precision=raw.QueryPrecision.EXACT,
                result_fields=raw.BatchResultField.SOURCE_FEATURE,
            )
        )

    def test_float32_mesh_reserves_its_structure_at_registration(self) -> None:
        # The registration charges the mesh's query structure and its build scratch, so a
        # budget that admits it, the publication and the plan admits every first exact
        # reader, scalar or batch, and one byte less refuses the registration (ADR-0075).
        query = RaycastQuery(
            origin=(-1.0, 0.0, 0.0),
            unit_direction=(1.0, 0.0, 0.0),
            max_distance=2.0,
            precision=raw.QueryPrecision.EXACT,
        )
        with Library.open(self.library_path, require_explicit=True) as library:
            with library.instance() as measured:
                before = measured.memory_usage(Device())
                adapter = measured.explicit()
                self._wait(adapter.register_triangle_meshes(**_repeated_triangle_mesh(4001)))
                registered = measured.memory_usage(Device())
                self.assertGreater(registered, before)
                population = self._populate(adapter, 4001)
                plan = self._exact_raycast_plan(measured, population)
                planned = measured.memory_usage(Device())
                operation, _ = plan.enqueue(**_raycast_columns())
                self.assertIs(operation.wait(), raw.OpState.COMPLETED)
                launched = measured.memory_usage(Device())
                operation.close()
                # The launch built the mesh and released the build scratch; what it
                # charged itself ended with the operation.
                closed = measured.memory_usage(Device())
                self.assertLess(closed, planned)
                budget = planned + launched - closed
                plan.close()
                self.assertIsNotNone(population.target.raycast_closest(query).hit)
                population.close()
                adapter.destroy_definition(
                    GeometryDefinitionRef(raw.GeometryType.TRIANGLE_MESH, 4001)
                )

            with library.instance(InstanceConfig(host_byte_budget=budget)) as tight:
                adapter = tight.explicit()
                self._wait(adapter.register_triangle_meshes(**_repeated_triangle_mesh(4002)))
                population = self._populate(adapter, 4002)
                plan = self._exact_raycast_plan(tight, population)
                operation, _ = plan.enqueue(**_raycast_columns())
                self.assertIs(operation.wait(), raw.OpState.COMPLETED)
                self.assertIsNotNone(operation.fetch_raycast_batch().exact)
                operation.close()
                plan.close()
                built = tight.memory_usage(Device())
                self.assertIsNotNone(population.target.raycast_closest(query).hit)
                self.assertEqual(tight.memory_usage(Device()), built)
                population.close()
                adapter.destroy_definition(
                    GeometryDefinitionRef(raw.GeometryType.TRIANGLE_MESH, 4002)
                )

            with library.instance(InstanceConfig(host_byte_budget=registered - 1)) as refused:
                with self.assertRaises(ApiError) as failure:
                    refused.explicit().register_triangle_meshes(**_repeated_triangle_mesh(4003))
                self.assertIs(failure.exception.status, raw.ApiStatus.HOST_BUDGET)
                self.assertEqual(refused.memory_usage(Device()), before)

    def test_float32_triangle_scalar_and_batch_keep_exact_features(self) -> None:
        triangle_value = 3001
        triangle = _triangle_mesh(
            _u64((triangle_value,), (1,)),
            _u64((0,), (1,)),
            _u64((3,), (1,)),
            _u64((0,), (1,)),
            _u64((1,), (1,)),
            _floating((1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -1.0), 32, (3, 3)),
            _u32((0, 1, 2), (1, 3)),
        )
        with (
            Library.open(self.library_path, require_explicit=True) as library,
            library.instance() as instance,
        ):
            adapter = instance.explicit()
            self._wait(adapter.register_triangle_meshes(**triangle))
            population = adapter.create_population()
            self._wait(
                population.enqueue_set(
                    1,
                    **_instances(
                        _u64((51,), (1,)),
                        _i32((raw.GeometryType.TRIANGLE_MESH,), (1,)),
                        _u64((triangle_value,), (1,)),
                        _floating((0.0, 0.0, 0.0), 64, (1, 3)),
                        _floating((0.0, 0.0, 0.0, 1.0), 64, (1, 4)),
                    ),
                )
            )
            query = RaycastQuery(
                origin=(-1.0, 0.0, 0.0),
                unit_direction=(1.0, 0.0, 0.0),
                max_distance=2.0,
                precision=raw.QueryPrecision.EXACT,
            )
            before = instance.memory_usage(Device())
            result = population.target.raycast_closest(query)
            after_first = instance.memory_usage(Device())
            repeated = population.target.raycast_closest(query)
            single = population.target.raycast_closest(
                RaycastQuery(
                    origin=query.origin,
                    unit_direction=query.unit_direction,
                    max_distance=query.max_distance,
                    precision=raw.QueryPrecision.EXACT_SINGLE,
                )
            )
            # The first query builds the mesh within the registration's reservation and
            # releases the build scratch; later queries charge nothing (ADR-0075).
            self.assertLess(after_first, before)
            self.assertEqual(instance.memory_usage(Device()), after_first)
            for candidate in (result, repeated, single):
                self.assertIsNotNone(candidate.hit)
                self.assertEqual(candidate.hit.object_id, 51)
                self.assertAlmostEqual(candidate.hit.distance, 2.0, places=12)
                self.assertEqual(candidate.hit.source_face_index, 0)
                self.assertEqual(candidate.hit.source_face_corner_indices, (0, 1, 2))
                for actual, expected in zip(candidate.hit.barycentrics, (0.25, 0.5, 0.25)):
                    self.assertAlmostEqual(actual, expected, places=6)

            for precision in (raw.QueryPrecision.EXACT, raw.QueryPrecision.EXACT_SINGLE):
                with self.subTest(precision=precision):
                    plan = instance.create_raycast_plan(
                        RaycastPlanConfig(
                            population.target.reference,
                            raw.HitMode.CLOSEST,
                            1,
                            1,
                            1,
                            precision=precision,
                            result_fields=(
                                raw.BatchResultField.POSITION
                                | raw.BatchResultField.NORMAL
                                | raw.BatchResultField.DISTANCE
                                | raw.BatchResultField.SOURCE_FEATURE
                            ),
                        )
                    )
                    operation, _ = plan.enqueue(**_raycast_columns())
                    self.assertIs(operation.wait(), raw.OpState.COMPLETED)
                    batch = operation.fetch_raycast_batch()
                    operation.close()
                    self.assertIsNotNone(batch.exact)
                    self.assertIsNotNone(batch.exact.features)
                    features = batch.exact.features
                    self.assertEqual(features.barycentrics.device, Device.cpu())
                    self.assertEqual(
                        _cpu_values(features.source_face_indices, ctypes.c_uint64), (0,)
                    )
                    self.assertEqual(
                        _cpu_values(features.source_corner_indices, ctypes.c_uint32),
                        (0, 1, 2),
                    )
                    for actual, expected in zip(
                        _cpu_values(features.barycentrics, ctypes.c_float),
                        (0.25, 0.5, 0.25),
                    ):
                        self.assertAlmostEqual(actual, expected, places=6)
                    plan.close()
            population.close()
            adapter.destroy_definition(
                GeometryDefinitionRef(raw.GeometryType.TRIANGLE_MESH, triangle_value)
            )

    def test_mesh_and_convex_float_inputs_and_float64_rotation_reach_exact_queries(self) -> None:
        triangle_vertices = (1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -1.0)
        convex_vertices = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        angle = 0.6
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = (0.0, 0.0, math.sin(angle * 0.5), math.cos(angle * 0.5))
        self.assertNotEqual(ctypes.c_float(rotation[2]).value, rotation[2])
        origin = (5.0 - 9.0 * cosine, -9.0 * sine, 0.0)
        direction = (ctypes.c_float(cosine).value, ctypes.c_float(sine).value, 0.0)
        expected_distance = 10.0 / (direction[0] * cosine + direction[1] * sine)
        expected_position = tuple(
            origin[axis] + expected_distance * direction[axis] for axis in range(3)
        )
        observed_positions = []
        observed_convex_positions = []
        with (
            Library.open(self.library_path, require_explicit=True) as library,
            library.instance() as instance,
        ):
            adapter = instance.explicit()
            for bits, triangle_value, convex_value in ((32, 1001, 2001),):
                with self.subTest(bits=bits):
                    triangle = _triangle_mesh(
                        _u64((triangle_value,), (1,)),
                        _u64((0,), (1,)),
                        _u64((3,), (1,)),
                        _u64((0,), (1,)),
                        _u64((1,), (1,)),
                        _floating(triangle_vertices, bits, (3, 3)),
                        _u32((0, 1, 2), (1, 3)),
                    )
                    self._wait(adapter.register_triangle_meshes(**triangle))

                    convex = dict(
                        definition_values=_u64((convex_value,), (1,)),
                        vertex_offsets=_u64((0,), (1,)),
                        vertex_counts=_u64((4,), (1,)),
                        vertices=_floating(convex_vertices, bits, (4, 3)),
                    )
                    self._wait(adapter.register_convexes(**convex))

                    population = adapter.create_population()
                    instances = dict(
                        object_values=_u64((41,), (1,)),
                        geometry_types=_i32((raw.GeometryType.TRIANGLE_MESH,), (1,)),
                        geometry_values=_u64((triangle_value,), (1,)),
                        translations=_floating((5.0, 0.0, 0.0), 64, (1, 3)),
                        rotations_xyzw=_floating(rotation, 64, (1, 4)),
                    )
                    publication = population.enqueue_set(1, **instances)
                    self._wait(publication)
                    result = population.target.raycast_closest(
                        RaycastQuery(
                            origin=origin,
                            unit_direction=direction,
                            max_distance=20.0,
                            precision=raw.QueryPrecision.EXACT,
                        )
                    )
                    self.assertIsNotNone(result.hit)
                    self.assertEqual(result.hit.object_id, 41)
                    self.assertAlmostEqual(result.hit.distance, expected_distance, places=12)
                    # Surface contacts use float local arithmetic; the expected world pose remains double.
                    for observed, expected in zip(result.hit.position, expected_position):
                        self.assertAlmostEqual(observed, expected, delta=2e-7)
                    observed_positions.append(result.hit.position)

                    population.close()
                    convex_population = adapter.create_population()
                    convex_instances = dict(
                        object_values=_u64((42,), (1,)),
                        geometry_types=_i32((raw.GeometryType.CONVEX,), (1,)),
                        geometry_values=_u64((convex_value,), (1,)),
                        translations=_floating((0.0, 0.0, 0.0), 64, (1, 3)),
                        rotations_xyzw=_floating((0.0, 0.0, 0.0, 1.0), 64, (1, 4)),
                    )
                    self._wait(convex_population.enqueue_set(1, **convex_instances))
                    convex_result = convex_population.target.raycast_closest(
                        RaycastQuery(
                            origin=(-1.0, 0.25, 0.25),
                            unit_direction=(1.0, 0.0, 0.0),
                            max_distance=2.0,
                            precision=raw.QueryPrecision.EXACT,
                        )
                    )
                    self.assertIsNotNone(convex_result.hit)
                    self.assertEqual(convex_result.hit.object_id, 42)
                    self.assertAlmostEqual(convex_result.hit.distance, 1.0, places=12)
                    self.assertAlmostEqual(convex_result.hit.position[0], 0.0, places=12)
                    observed_convex_positions.append(convex_result.hit.position)
                    convex_population.close()
                    adapter.destroy_definition(
                        GeometryDefinitionRef(raw.GeometryType.TRIANGLE_MESH, triangle_value)
                    )
                    adapter.destroy_definition(
                        GeometryDefinitionRef(raw.GeometryType.CONVEX, convex_value)
                    )
        self.assertEqual(len(observed_positions), 1)
        self.assertEqual(len(observed_convex_positions), 1)

        # These registrations and queries use CPU tensors exclusively. Releasing
        # their operations must not load a GPU driver just to clean up nothing.
        if sys.platform == "win32":
            get_module = ctypes.WinDLL("kernel32").GetModuleHandleW
            get_module.argtypes = [ctypes.c_wchar_p]
            get_module.restype = ctypes.c_void_p
            self.assertFalse(get_module("nvcuda.dll"), "CPU ingestion loaded the CUDA driver")
        elif sys.platform.startswith("linux"):
            self.assertFalse(
                "libcuda.so" in Path("/proc/self/maps").read_text(),
                "CPU ingestion loaded the CUDA driver",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, required=True)
    arguments = parser.parse_args()
    ExplicitInstalledDtypeTests.library_path = arguments.dll.resolve(strict=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ExplicitInstalledDtypeTests)
    )
    raise SystemExit(not result.wasSuccessful())
