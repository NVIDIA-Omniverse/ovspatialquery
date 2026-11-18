# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real installed result columns read through DLPack, on the CPU and, with --cuda, on a GPU."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np

from ovspatialquery import raw
from ovspatialquery.instance import ApiError
from ovspatialquery.explicit import GeometryDefinitionRef
from ovspatialquery.library import Library
from ovspatialquery.types import (
    Device,
    DeviceBudget,
    InstanceConfig,
    PairReporterConfig,
    RaycastPlanConfig,
)

CUDA = Device(raw.K_DL_CUDA, 0)


def _sphere_rows(object_ids, centers) -> dict[str, np.ndarray]:
    """Instance columns placing unit-sphere definition 1 at each center."""
    count = len(object_ids)
    return {
        "object_values": np.array(object_ids, dtype=np.uint64),
        "geometry_types": np.array([raw.GeometryType.SPHERE] * count, dtype=np.int32),
        "geometry_values": np.ones(count, dtype=np.uint64),
        "translations": np.array(centers, dtype=np.float64),
        "rotations_xyzw": np.array([[0, 0, 0, 1]] * count, dtype=np.float64),
    }


def _spheres(instance, object_ids, centers):
    """One population of unit spheres; returns the adapter and population."""
    adapter = instance.explicit()
    registration = adapter.register_spheres(
        definition_values=np.array([1], dtype=np.uint64), radii=np.array([1.0])
    )
    registration.wait()
    registration.close()
    population = adapter.create_population()
    publication = population.enqueue_set(1, **_sphere_rows(object_ids, centers))
    publication.wait()
    publication.close()
    return adapter, population


def _release(adapter, *populations) -> None:
    for population in populations:
        population.close()
    adapter.destroy_definition(GeometryDefinitionRef(raw.GeometryType.SPHERE, 1))


class _Cuda:
    """The driver calls a CUDA consumer of a DLPack capsule makes, on device 0's primary context."""

    def __init__(self) -> None:
        self.driver = (
            ctypes.CDLL("nvcuda.dll", winmode=0x800)
            if os.name == "nt"
            else ctypes.CDLL("libcuda.so.1")
        )
        pointer, u64 = ctypes.c_void_p, ctypes.c_uint64
        for name, arguments in {
            "cuInit": [ctypes.c_uint],
            "cuDevicePrimaryCtxRetain": [ctypes.POINTER(pointer), ctypes.c_int],
            "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
            "cuCtxSetCurrent": [pointer],
            "cuStreamCreate": [ctypes.POINTER(pointer), ctypes.c_uint],
            "cuStreamDestroy_v2": [pointer],
            "cuStreamSynchronize": [pointer],
            "cuMemAlloc_v2": [ctypes.POINTER(u64), ctypes.c_size_t],
            "cuMemFree_v2": [u64],
            "cuMemAllocHost_v2": [ctypes.POINTER(pointer), ctypes.c_size_t],
            "cuMemFreeHost": [pointer],
            "cuMemcpyHtoD_v2": [u64, pointer, ctypes.c_size_t],
            "cuMemcpyDtoHAsync_v2": [pointer, u64, ctypes.c_size_t, pointer],
            "cuEventCreate": [ctypes.POINTER(pointer), ctypes.c_uint],
            "cuEventRecord": [pointer, pointer],
            "cuLaunchHostFunc": [pointer, pointer, pointer],
        }.items():
            function = getattr(self.driver, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        self.call("cuInit", 0)
        self.instance = pointer()
        self.call("cuDevicePrimaryCtxRetain", ctypes.byref(self.instance), 0)
        self.call("cuCtxSetCurrent", self.instance)
        self.allocations: list[ctypes.c_uint64] = []
        self.callbacks: list = []
        self.pinned: list[ctypes.c_void_p] = []

    def call(self, name: str, *arguments) -> None:
        status = getattr(self.driver, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed with CUDA status {status}")

    def stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.call("cuStreamCreate", ctypes.byref(stream), 1)  # non-blocking: no implicit ordering
        return stream

    def upload(self, values, dtype) -> "_DeviceColumn":
        host = np.ascontiguousarray(values, dtype=dtype)
        allocation = ctypes.c_uint64()
        self.call("cuMemAlloc_v2", ctypes.byref(allocation), host.nbytes)
        self.allocations.append(allocation)
        self.call("cuMemcpyHtoD_v2", allocation, host.ctypes.data, host.nbytes)
        # A pageable copy may return before its DMA lands, and nothing orders the legacy stream
        # before the non-blocking one the library reads on: wait for it here.
        self.call("cuStreamSynchronize", None)
        return _DeviceColumn(host, allocation.value)

    def read(self, column, stream: ctypes.c_void_p, dtype):
        """Takes the capsule as a consumer on `stream` and queues its copy to the host. Returns the
        host array, valid once `stream` is synchronized and until `close`, and the call that
        releases the capsule.
        """
        capsule = column.__dlpack__(stream=stream.value)
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        managed = ctypes.cast(
            get_pointer(capsule, b"dltensor"), ctypes.POINTER(_DLManagedTensor)
        ).contents
        set_name = ctypes.pythonapi.PyCapsule_SetName
        set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
        set_name(capsule, b"used_dltensor")
        tensor = managed.dl_tensor
        shape = tuple(tensor.shape[index] for index in range(tensor.ndim))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        host = np.zeros(shape, dtype)
        if nbytes:
            # Pinned memory, so the copy is queued without the host waiting for the stream.
            pinned = ctypes.c_void_p()
            self.call("cuMemAllocHost_v2", ctypes.byref(pinned), nbytes)
            self.pinned.append(pinned)
            self.call(
                "cuMemcpyDtoHAsync_v2", pinned, tensor.data + tensor.byte_offset, nbytes, stream
            )
            host = np.frombuffer(
                (ctypes.c_byte * nbytes).from_address(pinned.value), dtype
            ).reshape(shape)

        def release() -> None:
            ctypes.CFUNCTYPE(None, ctypes.c_void_p)(managed.deleter)(ctypes.addressof(managed))

        return host, release

    def hold(self, stream: ctypes.c_void_p, gate: threading.Event) -> None:
        """Blocks `stream` until `gate` is set, or ten seconds pass."""
        callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: gate.wait(10))
        self.callbacks.append(callback)  # The driver calls it later; keep it alive until close.
        self.call("cuLaunchHostFunc", stream, callback, None)

    def done_event(self, stream: ctypes.c_void_p) -> int:
        event = ctypes.c_void_p()
        self.call("cuEventCreate", ctypes.byref(event), 2)
        self.call("cuEventRecord", event, stream)
        return event.value

    def close(self) -> None:
        for allocation in self.allocations:
            self.call("cuMemFree_v2", allocation)
        for pinned in self.pinned:
            self.call("cuMemFreeHost", pinned)
        self.call("cuDevicePrimaryCtxRelease_v2", 0)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", raw.DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class _DeviceColumn:
    """A DLPack producer for one device allocation shaped like `host`; NumPy cannot export one."""

    def __init__(self, host, address: int) -> None:
        self.shape = (ctypes.c_int64 * host.ndim)(*host.shape)
        self.managed = _DLManagedTensor(
            raw.DLTensor(
                data=address,
                device=CUDA.to_raw(),
                ndim=host.ndim,
                dtype=raw.DLDataType(
                    {"f": raw.K_DL_FLOAT, "u": raw.K_DL_UINT}[host.dtype.kind],
                    host.dtype.itemsize * 8,
                    1,
                ),
                shape=self.shape,
            )
        )

    def __dlpack_device__(self):
        return (raw.K_DL_CUDA, 0)

    def __dlpack__(self, stream=None):
        capsule = ctypes.pythonapi.PyCapsule_New
        capsule.restype = ctypes.py_object
        capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        return capsule(ctypes.addressof(self.managed), b"dltensor", None)


class ResultDLPackTests(unittest.TestCase):
    library_path: Path
    cuda: bool

    def setUp(self) -> None:
        self.library = Library.open(self.library_path)
        self.addCleanup(self.library.close)

    def test_cpu_batch_columns_read_through_numpy_while_the_operation_lives(self) -> None:
        with self.library.instance() as instance:
            adapter, population = _spheres(instance, [20, 21], [[4, 0, 0], [6, 3, 0]])
            plan = instance.create_raycast_plan(
                RaycastPlanConfig(
                    population.target.reference,
                    raw.HitMode.CLOSEST,
                    4,
                    1,
                    4,
                    precision=raw.QueryPrecision.EXACT,
                    result_fields=raw.BatchResultField.DISTANCE,
                )
            )
            operation, reservation = plan.enqueue(
                origins=np.array([[0, 0, 0], [0, 6, 0], [0, 3, 0]], dtype=np.float64),
                unit_directions=np.array([[1, 0, 0]] * 3, dtype=np.float32),
                max_distances=np.full(3, 20.0),
            )

            # CPU storage is unreadable until a wait, status or fetch observes the end.
            with self.assertRaisesRegex(BufferError, "terminal"):
                np.from_dlpack(reservation.object_ids)
            self.assertIs(operation.wait(), raw.OpState.COMPLETED)
            result = operation.fetch_raycast_batch()

            counts = np.from_dlpack(result.hit_ranges.hit_counts)
            offsets = np.from_dlpack(result.hit_ranges.hit_offsets)
            objects = np.from_dlpack(result.object_ids)
            distances = np.from_dlpack(result.exact.distances)
            self.assertEqual(counts.tolist(), [1, 0, 1])
            self.assertEqual(objects[offsets[[0, 2]]].tolist(), [20, 21])
            np.testing.assert_allclose(distances[offsets[[0, 2]]], [3.0, 5.0], atol=1e-6)
            # A reservation column is the same storage as its fetched column.
            self.assertEqual(np.from_dlpack(reservation.object_ids)[:2].tolist(), [20, 21])

            # The export borrows operation storage, so destroying the operation waits for it.
            with self.assertRaisesRegex(BufferError, "still alive"):
                operation.close()
            copied = objects.copy()
            del counts, offsets, objects, distances
            operation.close()
            self.assertEqual(copied.tolist(), [20, 21])
            with self.assertRaisesRegex(BufferError, "destroyed"):
                np.from_dlpack(result.object_ids)

            plan.close()
            _release(adapter, population)

    def test_cpu_pair_reporter_columns_read_through_numpy_including_empty_ones(self) -> None:
        with self.library.instance() as instance:
            adapter, scene = _spheres(instance, [1, 2], [[0, 0, 0], [5, 0, 0]])
            robots = adapter.create_population()
            publication = robots.enqueue_set(1, **_sphere_rows([7], [[0.5, 0, 0]]))
            publication.wait()
            publication.close()
            reporter = instance.create_pair_reporter(
                PairReporterConfig(robots.target.reference, scene.target.reference, 4, 4)
            )
            operation, _ = reporter.advance()
            operation.wait()
            result = operation.fetch_pair_reporter()

            entered = np.from_dlpack(result.entered_pairs)
            exited = np.from_dlpack(result.exited_pairs)
            self.assertEqual(entered.tolist(), [[7, 1]])
            self.assertEqual(exited.shape, (0, 2))
            del entered, exited
            operation.close()
            reporter.close()
            _release(adapter, robots, scene)

    def test_cuda_reservation_columns_are_ordered_on_the_consumer_stream(self) -> None:
        if not self.cuda:
            self.skipTest("run with --cuda on a CUDA install")
        cuda = _Cuda()
        self.addCleanup(cuda.close)
        config = InstanceConfig(device_budgets=(DeviceBudget(CUDA, 64 << 20),))
        with self.library.instance(config) as instance:
            adapter, population = _spheres(instance, [20, 21], [[4, 0, 0], [6, 3, 0]])
            plan = instance.create_raycast_plan(
                RaycastPlanConfig(
                    population.target.reference,
                    raw.HitMode.CLOSEST,
                    4,
                    1,
                    4,
                    precision=raw.QueryPrecision.EXACT,
                    device=CUDA,
                    result_fields=raw.BatchResultField.DISTANCE,
                )
            )
            inputs = (
                cuda.upload([[0, 0, 0], [0, 6, 0], [0, 3, 0]], "float64"),
                cuda.upload([[1, 0, 0]] * 3, "float32"),
                cuda.upload([20] * 3, "float64"),
            )
            # The launch waits on a host gate on its own stream. The consumer queues its copies of
            # the reservation on a separate non-blocking stream before the gate opens, so only the
            # export's queued wait on result_ready keeps them from reading unwritten storage.
            launch, consumer = cuda.stream(), cuda.stream()
            gate = threading.Event()
            self.addCleanup(gate.set)  # A failed assertion must not leave the launch blocked.
            cuda.hold(launch, gate)
            operation, reservation = plan.enqueue(
                origins=inputs[0],
                unit_directions=inputs[1],
                max_distances=inputs[2],
                cuda_stream=launch.value,
            )
            self.assertEqual(reservation.object_ids.__dlpack_device__(), (raw.K_DL_CUDA, 0))
            reads = [
                cuda.read(reservation.hit_ranges.hit_counts, consumer, np.uint64),
                cuda.read(reservation.hit_ranges.hit_offsets, consumer, np.uint64),
                cuda.read(reservation.object_ids, consumer, np.uint64),
                cuda.read(reservation.exact.distances, consumer, np.float64),
            ]
            gate.set()
            cuda.call("cuStreamSynchronize", consumer)
            for _, release in reads:
                release()
            counts, offsets, objects, distances = (host for host, _ in reads)
            self.assertEqual(counts.tolist(), [1, 0, 1])
            self.assertEqual(objects[offsets[[0, 2]]].tolist(), [20, 21])
            np.testing.assert_allclose(distances[offsets[[0, 2]]], [3.0, 5.0], atol=1e-6)
            with self.assertRaisesRegex(BufferError, "ambiguous"):
                reservation.object_ids.__dlpack__(stream=0)

            self.assertIs(operation.wait(), raw.OpState.COMPLETED)
            held = operation.fetch_raycast_batch().object_ids.__dlpack__(stream=consumer.value)
            done = cuda.done_event(consumer)
            with self.assertRaisesRegex(BufferError, "still alive"):
                operation.close_after_cuda(done)
            del held  # An unconsumed capsule releases its export when it is freed.
            operation.close_after_cuda(done)
            cuda.call("cuStreamSynchronize", consumer)
            cuda.call("cuStreamDestroy_v2", consumer)
            cuda.call("cuStreamDestroy_v2", launch)

            # The library releases the slot once the retired operation is device-safe.
            deadline = time.monotonic() + 10
            while True:
                try:
                    plan.close()
                    break
                except ApiError as error:
                    if error.status != raw.ApiStatus.BUSY or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.001)
            _release(adapter, population)

    def test_host_mapped_cuda_columns_read_through_numpy_once_terminal(self) -> None:
        if not self.cuda:
            self.skipTest("run with --cuda on a CUDA install")
        cuda = _Cuda()
        self.addCleanup(cuda.close)
        config = InstanceConfig(device_budgets=(DeviceBudget(CUDA, 64 << 20),))
        with self.library.instance(config) as instance:
            adapter, population = _spheres(instance, [20, 21], [[4, 0, 0], [6, 3, 0]])
            plan = instance.create_raycast_plan(
                RaycastPlanConfig(
                    population.target.reference,
                    raw.HitMode.CLOSEST,
                    4,
                    1,
                    4,
                    precision=raw.QueryPrecision.EXACT,
                    device=CUDA,
                    result_slot_count=1,
                    result_fields=raw.BatchResultField.DISTANCE | raw.BatchResultField.HOST_MAPPED,
                )
            )
            inputs = (
                cuda.upload([[0, 0, 0], [0, 6, 0], [0, 3, 0]], "float64"),
                cuda.upload([[1, 0, 0]] * 3, "float32"),
                cuda.upload([20] * 3, "float64"),
            )
            # A host gate holds the launch, so the columns are first seen while it is pending.
            launch = cuda.stream()
            gate = threading.Event()
            self.addCleanup(gate.set)
            cuda.hold(launch, gate)
            operation, reservation = plan.enqueue(
                origins=inputs[0],
                unit_directions=inputs[1],
                max_distances=inputs[2],
                cuda_stream=launch.value,
            )
            self.assertEqual(reservation.object_ids.__dlpack_device__(), (raw.K_DL_CUDA_HOST, 0))
            with self.assertRaisesRegex(BufferError, "terminal"):
                np.from_dlpack(reservation.object_ids)
            gate.set()
            self.assertIs(operation.wait(), raw.OpState.COMPLETED)
            result = operation.fetch_raycast_batch()
            self.assertEqual(result.common.execution_device, CUDA)
            # A CUDA consumer of pinned memory passes its stream, which has nothing to wait for.
            with self.assertRaisesRegex(BufferError, "ambiguous"):
                result.object_ids.__dlpack__(stream=0)
            capsule = result.object_ids.__dlpack__(stream=launch.value)
            del capsule  # An unconsumed capsule releases its export when it is freed.

            # NumPy reads the pinned rows in place; to_numpy() copies them.
            counts = np.from_dlpack(result.hit_ranges.hit_counts)
            offsets = np.from_dlpack(result.hit_ranges.hit_offsets)
            objects = np.from_dlpack(result.object_ids)
            self.assertEqual(counts.tolist(), [1, 0, 1])
            self.assertEqual(objects[offsets[[0, 2]]].tolist(), [20, 21])
            copied = result.to_numpy()
            np.testing.assert_allclose(
                copied.exact.distances[copied.hit_ranges.hit_offsets[[0, 2]]], [3.0, 5.0], atol=1e-6
            )
            with self.assertRaisesRegex(BufferError, "still alive"):
                operation.close()
            del counts, offsets, objects
            # close() retires a host reader's launch at once, so the plan's one slot is free.
            operation.close()
            again, _ = plan.enqueue(
                origins=inputs[0],
                unit_directions=inputs[1],
                max_distances=inputs[2],
                cuda_stream=launch.value,
            )
            with again:
                self.assertIs(again.wait(), raw.OpState.COMPLETED)
                self.assertEqual(
                    again.fetch_raycast_batch().to_numpy().hit_ranges.hit_counts.tolist(), [1, 0, 1]
                )
            cuda.call("cuStreamDestroy_v2", launch)
            plan.close()
            self.assertEqual(copied.object_ids[copied.hit_ranges.hit_offsets[2]], 21)
            _release(adapter, population)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true", help="also run the CUDA consumer case")
    arguments = parser.parse_args()
    ResultDLPackTests.library_path = arguments.dll.resolve(strict=True)
    ResultDLPackTests.cuda = arguments.cuda
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ResultDLPackTests)
    )
    if not result.wasSuccessful():
        # A failure can leave an instance with pending device work that process teardown would
        # wait on; report the failure instead of hanging until the CTest timeout.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
