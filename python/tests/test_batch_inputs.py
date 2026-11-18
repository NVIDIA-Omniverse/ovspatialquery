# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import ctypes
import gc
import os
from pathlib import Path
import sys
import threading
import time
import types
import unittest
import weakref
from ovspatialquery import (
    Aabb,
    ApiError,
    ApiStatus,
    BatchResultField,
    BoxSweepPlanConfig,
    BoxSweepQuery,
    Device,
    DeviceBudget,
    GeometryDefinitionRef,
    GeometryType,
    HitMode,
    HitTruth,
    InstanceConfig,
    Library,
    MeshHits,
    OpState,
    OverlapBatchShape,
    OverlapPlanConfig,
    OverlapQuery,
    OverlapShapeKind,
    PairReporterConfig,
    QueryPrecision,
    RaycastPlanConfig,
    RaycastQuery,
    SphereSweepPlanConfig,
    SphereSweepQuery,
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


def _cpu_column(view, element) -> list:
    pointer = ctypes.cast(view.data + view.byte_offset, ctypes.POINTER(element))
    return [pointer[index] for index in range(view.shape[0])]


@unittest.skipIf(np is None, "NumPy supplies the DLPack columns")
class BatchInputTests(unittest.TestCase):
    """Unit spheres 1 and 2 sit at x = 5 and x = 10."""

    def setUp(self) -> None:
        budgets = ()
        if wp is not None and wp.is_cuda_available():
            budgets = (DeviceBudget(Device.cuda(), 1 << 28),)
        self.library = Library.open(_library_path(), require_explicit=True)
        self.instance = self.library.instance(InstanceConfig(device_budgets=budgets))
        self.adapter = adapter = self.instance.explicit()
        self.robots = []
        definition = adapter.register_spheres(
            definition_values=np.array([7], dtype=np.uint64), radii=np.array([1.0])
        )
        definition.wait()
        definition.close()
        self.population = adapter.create_population()
        publication = self.population.enqueue_set(
            1,
            object_values=np.array([1, 2], dtype=np.uint64),
            geometry_types=np.array([GeometryType.SPHERE] * 2, dtype=np.int32),
            geometry_values=np.array([7, 7], dtype=np.uint64),
            translations=np.array([(5, 0, 0), (10, 0, 0)], dtype=np.float64),
            rotations_xyzw=np.array([(0, 0, 0, 1)] * 2, dtype=np.float64),
        )
        publication.wait()
        publication.close()

    def tearDown(self) -> None:
        for robot in self.robots:
            robot.close()
        self.population.close()
        self.adapter.destroy_definition(GeometryDefinitionRef(GeometryType.SPHERE, 7))
        self.instance.close()
        self.library.close()

    def _raycast_plan(self, device: Device = Device.cpu()):
        return self.instance.create_raycast_plan(
            RaycastPlanConfig(
                self.population.target.reference,
                HitMode.CLOSEST,
                4,
                1,
                4,
                precision=QueryPrecision.EXACT,
                device=device,
            )
        )

    def test_borrowed_columns_live_until_close_and_copied_columns_do_not(self) -> None:
        plan = self._raycast_plan()
        for copy in (False, True):
            with self.subTest(copy=copy):
                origins = np.zeros((2, 3), dtype=np.float64)
                directions = np.array([(1, 0, 0), (-1, 0, 0)], dtype=np.float32)
                watched = weakref.ref(origins)
                operation, _ = plan.enqueue(
                    origins=origins,
                    unit_directions=directions,
                    max_distances=np.full(2, 20.0),
                    copy=copy,
                )
                del origins
                gc.collect()
                self.assertEqual(watched() is None, copy)
                self.assertIs(operation.wait(), OpState.COMPLETED)
                result = operation.fetch_raycast_batch()
                self.assertEqual(_cpu_column(result.hit_ranges.hit_counts, ctypes.c_uint64), [1, 0])
                self.assertEqual(_cpu_column(result.object_ids, ctypes.c_uint64), [1])
                # A copied result holds no export, so the operation closes and the copy remains.
                copied = result.to_numpy()
                operation.close()
                self.assertEqual(copied.hit_ranges.hit_counts.tolist(), [1, 0])
                self.assertEqual(copied.object_ids.tolist(), [1])
                gc.collect()
                self.assertIsNone(watched())
        plan.close()

    def test_columns_are_checked_before_the_library_sees_them(self) -> None:
        plan = self._raycast_plan()
        good = {
            "origins": np.zeros((2, 3)),
            "unit_directions": np.array([(1, 0, 0)] * 2, dtype=np.float32),
            "max_distances": np.full(2, 20.0),
        }
        for label, change, error, text in (
            ("dtype", {"origins": np.zeros((2, 3), np.float32)}, ValueError, "origins.*float64"),
            ("width", {"origins": np.zeros((2, 4))}, ValueError, r"origins.*\(Q, 3\)"),
            ("rows", {"max_distances": np.full(3, 20.0)}, ValueError, "max_distances.*3 rows"),
            ("protocol", {"max_distances": [20.0, 20.0]}, TypeError, "max_distances.*DLPack"),
            ("stream", {"cuda_stream": 1}, ValueError, "CPU plan"),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(error, text):
                    plan.enqueue(**{**good, **change})
        operation, _ = plan.enqueue(**good)
        self.assertIs(operation.wait(), OpState.COMPLETED)
        operation.close()
        plan.close()

    def test_overlap_plan_shape_sets_the_default_row_size(self) -> None:
        for shape, center_x, hit in (
            (OverlapBatchShape.sphere(0.5), 3.4, False),
            (OverlapBatchShape.sphere(0.7), 3.4, True),
            (OverlapBatchShape.box((0.7, 0.1, 0.1)), 3.4, True),
            (OverlapBatchShape.aabb((0.5, 0.1, 0.1)), 3.4, False),
        ):
            with self.subTest(shape=shape):
                plan = self.instance.create_overlap_plan(
                    OverlapPlanConfig(
                        self.population.target.reference,
                        shape,
                        HitMode.ANY,
                        1,
                        1,
                        1,
                        precision=QueryPrecision.EXACT,
                    )
                )
                operation, _ = plan.enqueue(
                    translations=np.array([(center_x, 0, 0)], dtype=np.float64),
                    rotations_xyzw=(
                        np.array([(0, 0, 0, 1)], dtype=np.float32)
                        if shape.kind is OverlapShapeKind.BOX
                        else None
                    ),
                )
                self.assertIs(operation.wait(), OpState.COMPLETED)
                result = operation.fetch_overlap_batch()
                self.assertEqual(
                    _cpu_column(result.hit_ranges.hit_counts, ctypes.c_uint64), [int(hit)]
                )
                operation.close()
                plan.close()

    def test_multiple_hit_queries_store_hits_up_to_capacity(self) -> None:
        target = self.population.target
        around_both = OverlapQuery(Aabb((3, -2, -2), (12, 2, 2)))
        # A capacity is any integer, NumPy's included, as every other count is.
        for capacity, count, truncated in ((np.int64(2), 2, False), (1, 1, True), (0, 0, True)):
            with self.subTest(capacity=capacity):
                result = target.overlap_multiple(around_both, capacity)
                self.assertEqual(len(result.hits), count)
                self.assertEqual(result.truncated, truncated)
                self.assertEqual(result.source_ordinal, 1)
                self.assertLessEqual({hit.object_id for hit in result.hits}, {1, 2})
        rays = target.raycast_multiple(
            RaycastQuery((0, 0, 0), (1, 0, 0), 20, precision=QueryPrecision.EXACT), np.uint32(4)
        )
        self.assertFalse(rays.truncated)
        hits = sorted(rays.hits, key=lambda hit: hit.distance)
        self.assertEqual([hit.object_id for hit in hits], [1, 2])
        for hit, distance in zip(hits, (4.0, 9.0)):
            self.assertAlmostEqual(hit.distance, distance, places=5)
        # Spheres report one hit each whichever mesh_hits asks for.
        nearest = target.raycast_multiple(
            RaycastQuery((0, 0, 0), (1, 0, 0), 20, precision=QueryPrecision.EXACT),
            4,
            mesh_hits=MeshHits.NEAREST,
        )
        self.assertEqual(sorted(hit.object_id for hit in nearest.hits), [1, 2])
        # Every crossing needs a ray with no width.
        tube = RaycastQuery(
            (0, 0, 0), (1, 0, 0), 20, precision=QueryPrecision.EXACT, contact_tolerance=0.01
        )
        with self.assertRaisesRegex(ApiError, "NOT_SUPPORTED"):
            target.raycast_multiple(tube, 4)
        self.assertEqual(len(target.raycast_multiple(tube, 4, mesh_hits=MeshHits.NEAREST).hits), 2)
        with self.assertRaisesRegex(ValueError, "capacity"):
            target.overlap_multiple(around_both, -1)

    def test_sweep_hits_report_the_object_contact_and_the_shape_center(self) -> None:
        # A half-unit sphere moving along +x first touches unit sphere 1 at x = 4, its AABB
        # face and its surface alike, with its center at x = 3.5.
        # Queries default to exact precision.
        for precision, truth in (
            ({"precision": QueryPrecision.BOUNDS_CANDIDATE}, HitTruth.BOUNDS_CANDIDATE),
            ({}, HitTruth.EXACT_SURFACE),
        ):
            with self.subTest(precision=precision):
                hit = self.population.target.sweep_sphere_closest(
                    SphereSweepQuery((0, 0, 0), 0.5, (1, 0, 0), 20, **precision)
                ).hit
                self.assertEqual((hit.object_id, hit.truth), (1, truth))
                self.assertAlmostEqual(hit.distance, 3.5, places=5)
                for actual, expected in (
                    (hit.position, (4, 0, 0)),
                    (hit.query_position, (3.5, 0, 0)),
                ):
                    for value, want in zip(actual, expected, strict=True):
                        self.assertAlmostEqual(value, want, places=5)

    def test_sweeps_report_any_hit_or_one_hit_per_object(self) -> None:
        # A half-unit sphere or box moving along +x reaches sphere 1 at 3.5 and sphere 2 at 8.5.
        target = self.population.target
        expected = {1: 3.5, 2: 8.5}
        sphere = SphereSweepQuery((0, 0, 0), 0.5, (1, 0, 0), 20)
        box = BoxSweepQuery((0, 0, 0), (0.5, 0.5, 0.5), (0, 0, 0, 1), (1, 0, 0), 20)
        for label, any_hit, multiple in (
            ("sphere", lambda: target.sweep_sphere_any(sphere), target.sweep_sphere_multiple),
            ("box", lambda: target.sweep_box_any(box), target.sweep_box_multiple),
        ):
            query = sphere if label == "sphere" else box
            with self.subTest(label):
                hit = any_hit().hit
                self.assertAlmostEqual(hit.distance, expected[hit.object_id], places=5)
                for capacity, count, truncated in (
                    (np.int64(4), 2, False),
                    (1, 1, True),
                    (0, 0, True),
                ):
                    result = multiple(query, capacity)
                    self.assertEqual((len(result.hits), result.truncated), (count, truncated))
                    for row in result.hits:
                        self.assertAlmostEqual(row.distance, expected[row.object_id], places=5)
                # Truncation keeps the nearest object.
                self.assertEqual([row.object_id for row in multiple(query, 1).hits], [1])
                with self.assertRaisesRegex(ValueError, "capacity"):
                    multiple(query, -1)

    def test_multiple_sweep_plans_bound_hits_per_query(self) -> None:
        reference = self.population.target.reference
        for config in (
            SphereSweepPlanConfig(
                reference,
                0.5,
                2,
                precision=QueryPrecision.EXACT,
                result_fields=BatchResultField.DISTANCE,
                hit_mode=HitMode.MULTIPLE,
                maximum_hits_per_query=1,
                hit_capacity_per_slot=4,
            ),
            BoxSweepPlanConfig(
                reference,
                (0.5, 0.5, 0.5),
                2,
                precision=QueryPrecision.EXACT,
                result_fields=BatchResultField.DISTANCE,
                hit_mode=HitMode.MULTIPLE,
                maximum_hits_per_query=2,
                hit_capacity_per_slot=4,
            ),
        ):
            with self.subTest(type(config).__name__):
                box = isinstance(config, BoxSweepPlanConfig)
                plan = (
                    self.instance.create_box_sweep_plan(config)
                    if box
                    else self.instance.create_sphere_sweep_plan(config)
                )
                columns = {
                    "centers": np.zeros((2, 3), dtype=np.float64),
                    "unit_directions": np.array([(1, 0, 0), (-1, 0, 0)], dtype=np.float32),
                    "max_distances": np.full(2, 20.0),
                }
                if box:
                    columns["rotations_xyzw"] = np.array([(0, 0, 0, 1)] * 2, dtype=np.float32)
                operation, _ = plan.enqueue(**columns)
                self.assertIs(operation.wait(), OpState.COMPLETED)
                result = (
                    operation.fetch_box_sweep_batch()
                    if box
                    else operation.fetch_sphere_sweep_batch()
                )
                limit = config.maximum_hits_per_query
                self.assertEqual(
                    _cpu_column(result.hit_ranges.hit_counts, ctypes.c_uint64), [limit, 0]
                )
                self.assertEqual(
                    _cpu_column(result.hit_ranges.truncated, ctypes.c_uint8), [int(limit < 2), 0]
                )
                self.assertEqual(
                    sorted(_cpu_column(result.object_ids, ctypes.c_uint64)), [1, 2][:limit]
                )
                operation.close()
                plan.close()

    def _robot(self, x: float):
        """A second population holding one unit sphere, object 9, at (x, 0, 0)."""
        robot = self.adapter.create_population()
        self.robots.append(robot)
        publication = robot.enqueue_set(
            1,
            object_values=np.array([9], dtype=np.uint64),
            geometry_types=np.array([GeometryType.SPHERE], dtype=np.int32),
            geometry_values=np.array([7], dtype=np.uint64),
            translations=np.array([(x, 0, 0)], dtype=np.float64),
            rotations_xyzw=np.array([(0, 0, 0, 1)], dtype=np.float64),
        )
        publication.wait()
        publication.close()
        return robot

    def test_cpu_pair_reporter_reservation_has_no_columns(self) -> None:
        robot = self._robot(5.5)
        reporter = self.instance.create_pair_reporter(
            PairReporterConfig(robot.target.reference, self.population.target.reference, 4, 4)
        )
        operation, reservation = reporter.advance()
        self.assertEqual(reservation.result_device, Device.cpu())
        self.assertEqual(reservation.delta_pair_capacity, 4)
        self.assertEqual(reservation.result_ready, 0)
        self.assertIsNone(reservation.counts)
        operation.wait()
        self.assertEqual(
            np.from_dlpack(operation.fetch_pair_reporter().entered_pairs).tolist(), [[9, 1]]
        )
        operation.close()
        reporter.close()

    @unittest.skipIf(wp is None, "Warp reads CUDA reporter columns")
    def test_cuda_pair_reporter_reservation_columns_read_through_dlpack(self) -> None:
        if not wp.is_cuda_available():
            self.skipTest("no CUDA device")
        robot = self._robot(5.5)
        reporter = self.instance.create_pair_reporter(
            PairReporterConfig(
                robot.target.reference,
                self.population.target.reference,
                4,
                4,
                result_device=Device.cuda(),
            )
        )
        operation, reservation = reporter.advance()
        self.assertEqual(reservation.result_device, Device.cuda())
        self.assertNotEqual(reservation.result_ready, 0)
        operation.wait()
        wp.wait_event(wp.Event("cuda:0", cuda_event=reservation.result_ready))
        counts = wp.from_dlpack(reservation.counts).numpy()
        entered = wp.from_dlpack(reservation.entered_pairs).numpy()
        self.assertEqual(int(counts[1]), 1)
        self.assertEqual(entered[0].tolist(), [9, 1])
        done = wp.record_event()
        wp.synchronize_event(done)
        operation.close_after_cuda(done.cuda_event)
        done.owner = False
        reporter.close()  # waits for the slot retired after the consumers

    @unittest.skipIf(wp is None, "Warp supplies CUDA columns and streams")
    def test_cuda_plan_reads_columns_after_their_producer_stream(self) -> None:
        if not wp.is_cuda_available():
            self.skipTest("no CUDA device")
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        driver.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        driver.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
        driver.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        device = wp.get_device("cuda:0")
        self.assertEqual(driver.cuCtxSetCurrent(device.context), 0)
        # The library reads on its own non-blocking stream, so only the wait that the
        # producer queues when the wrapper forwards that stream to __dlpack__ orders it.
        library_stream = ctypes.c_void_p()
        self.assertEqual(driver.cuStreamCreate(ctypes.byref(library_stream), 1), 0)
        self.addCleanup(driver.cuStreamDestroy_v2, library_stream)
        plan = self._raycast_plan(Device.cuda())
        producer = wp.Stream(device)
        gate = threading.Event()
        self.addCleanup(gate.set)  # A failed assertion must not leave the producer blocked.
        hold = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: gate.wait(10))
        with wp.ScopedStream(producer):
            # Rays start above the spheres and miss until the producer writes their origins.
            origins = wp.array(np.array([(0, 9, 0), (12, 9, 0)], dtype=np.float64), device=device)
            written = wp.array(np.array([(0, 0, 0), (12, 0, 0)], dtype=np.float64), device=device)
            directions = wp.array(
                np.array([(1, 0, 0), (-1, 0, 0)], dtype=np.float32), device=device
            )
            maximum = wp.array(np.full(2, 20.0), device=device)
            wp.synchronize_stream(producer)
            self.assertEqual(driver.cuLaunchHostFunc(producer.cuda_stream, hold, None), 0)
            wp.copy(origins, written)
            with self.assertRaisesRegex(TypeError, "cuda_stream handle must be an int"):
                plan.enqueue(
                    origins=origins,
                    unit_directions=directions,
                    max_distances=maximum,
                    cuda_stream=types.SimpleNamespace(cuda_stream=None),
                )
            with self.assertRaisesRegex(ValueError, "cuda_stream"):
                plan.enqueue(origins=origins, unit_directions=directions, max_distances=maximum)
            with self.assertRaisesRegex(ValueError, "origins is on"):
                plan.enqueue(
                    origins=np.zeros((2, 3)),
                    unit_directions=directions,
                    max_distances=maximum,
                    cuda_stream=library_stream.value,
                )
            operation, _ = plan.enqueue(
                origins=origins,
                unit_directions=directions,
                max_distances=maximum,
                cuda_stream=library_stream.value,
            )
        # Without that ordering the library reads the unwritten origins while the gate holds
        # the producer, and both rays miss; give such a read time to happen.
        time.sleep(0.5)
        gate.set()
        self.assertIs(operation.wait(), OpState.COMPLETED)
        result = operation.fetch_raycast_batch()
        self.assertEqual(result.common.execution_device, Device.cuda())
        wp.wait_event(wp.Event(device, cuda_event=result.common.result_ready))
        self.assertEqual(wp.from_dlpack(result.hit_ranges.hit_counts).numpy().tolist(), [1, 1])
        self.assertEqual(wp.from_dlpack(result.object_ids).numpy().tolist(), [1, 2])
        done = wp.record_event()
        wp.synchronize_event(done)
        operation.close_after_cuda(done.cuda_event)
        done.owner = False  # Retirement transfers the event to the library.
        plan.close()  # waits for the slot retired after the consumers

    @unittest.skipIf(wp is None, "Warp supplies CUDA columns and streams")
    def test_cuda_stream_zero_is_the_legacy_default_stream(self) -> None:
        if not wp.is_cuda_available():
            self.skipTest("no CUDA device")
        device = wp.get_device("cuda:0")
        plan = self.instance.create_raycast_plan(
            RaycastPlanConfig(
                self.population.target.reference, HitMode.CLOSEST, 1, 1, 1, device=Device.cuda()
            )
        )
        origins = wp.array(np.array([(0.0, 0, 0)]), device=device)
        directions = wp.array(np.array([(1, 0, 0)], dtype=np.float32), device=device)
        maximum = wp.array(np.array([20.0]), device=device)
        # 0 supplies CUDA's legacy default stream; only None supplies no stream.
        operation, _ = plan.enqueue(
            origins=origins, unit_directions=directions, max_distances=maximum, cuda_stream=0
        )
        self.assertIs(operation.wait(), OpState.COMPLETED)
        result = operation.fetch_raycast_batch()
        wp.wait_event(wp.Event(device, cuda_event=result.common.result_ready))
        self.assertEqual(wp.from_dlpack(result.hit_ranges.hit_counts).numpy().tolist(), [1])
        self.assertEqual(wp.from_dlpack(result.object_ids).numpy().tolist(), [1])
        done = wp.record_event()
        wp.synchronize_event(done)
        operation.close_after_cuda(done.cuda_event)
        done.owner = False  # Retirement transfers the event to the library.
        plan.close()

    @unittest.skipIf(wp is None, "Warp supplies CUDA columns and streams")
    def test_closing_a_cuda_plan_waits_only_for_retired_slots(self) -> None:
        if not wp.is_cuda_available():
            self.skipTest("no CUDA device")
        device = wp.get_device("cuda:0")
        plan = self.instance.create_raycast_plan(
            RaycastPlanConfig(
                self.population.target.reference,
                HitMode.CLOSEST,
                4,
                1,
                4,
                device=Device.cuda(),
                result_slot_count=2,
            )
        )
        stream = wp.Stream(device)
        with wp.ScopedStream(stream):
            origins = wp.array(np.array([(0.0, 0, 0)]), device=device)
            directions = wp.array(np.array([(1, 0, 0)], dtype=np.float32), device=device)
            maximum = wp.array(np.array([20.0]), device=device)
        launches = [
            plan.enqueue(
                origins=origins,
                unit_directions=directions,
                max_distances=maximum,
                cuda_stream=stream.cuda_stream,
            )[0]
            for _ in range(2)
        ]
        for operation in launches:
            self.assertIs(operation.wait(), OpState.COMPLETED)
        done = stream.record_event()
        wp.synchronize_event(done)
        launches[0].close_after_cuda(done.cuda_event)
        done.owner = False  # Retirement transfers the event to the library.
        # The second operation still owns a slot: that BUSY is the caller's and is not waited for.
        started = time.monotonic()
        with self.assertRaises(ApiError) as busy:
            plan.close()
        self.assertIs(busy.exception.status, ApiStatus.BUSY)
        self.assertLess(time.monotonic() - started, 1.0)
        # Hold the last consumer event on the host for 0.2 s: close must wait for the library to
        # release the slot after that event, not fail with BUSY while it is pending.
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        driver.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        gate = threading.Event()
        self.addCleanup(gate.set)
        hold = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: gate.wait(10))
        self.assertEqual(driver.cuLaunchHostFunc(stream.cuda_stream, hold, None), 0)
        second = stream.record_event()
        launches[1].close_after_cuda(second.cuda_event)
        second.owner = False
        threading.Timer(0.2, gate.set).start()
        started = time.monotonic()
        plan.close()
        self.assertGreaterEqual(time.monotonic() - started, 0.15)
        wp.synchronize_stream(stream)


if __name__ == "__main__":
    unittest.main()
