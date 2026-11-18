# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real installed device-prepared publication consumer; requires a CUDA candidate artifact.

Tests explicit populations published with preparation ON_DEVICE or ON_DEVICE_HOST_READABLE:
CUDA rows of every exact geometry family define membership in a complete set; complete-row,
pose and bits updates are validated and prepared by device kernels, also while earlier
publications are pending (ADMITTED_EXACT); and CUDA overlap, ray and sweep plans and CUDA pair
reporters read the device versions as they read a host-prepared twin. Inputs are host-authored
and uploaded with the CUDA driver, so this verifies native behavior and device result columns,
not GPU producer residency or transfer costs.
Run with the installed Python package and --library pointing to its candidate.
"""

from __future__ import annotations

import argparse
import ctypes as C
from fractions import Fraction
import math
import os
from pathlib import Path
import threading
import time
import unittest

from ovspatialquery import raw
from ovspatialquery.library import Library


IDS = (2**53 + 1, 2**63 + 11, 2**64 - 1)
GEOMETRIES = (17, 18, 17)
BITS = (1, 2, 3)
CUDA = raw.DLDevice(raw.K_DL_CUDA, 0)
CPU = raw.DLDevice(raw.K_DL_CPU, 0)
WAIT_NS = 10_000_000_000
CUDA_HOST_FN = C.CFUNCTYPE(None, C.c_void_p)


def words(bits):
    """The uint64[N,2] participation column of `bits`, bits 0-63 first, row by row."""
    return tuple(word for value in bits for word in (value & (2**64 - 1), value >> 64))


def participation(value):
    return raw.ParticipationBits((value & (2**64 - 1), value >> 64))


def dl_code(element):
    """The DLPack type code of a ctypes element type."""
    if element in (C.c_double, C.c_float):
        return raw.K_DL_FLOAT
    return raw.K_DL_INT if element == C.c_int32 else raw.K_DL_UINT


def lattice(side, first_id):
    """A side**3 lattice of boxes 1.5 apart, every third one yawed: its ids and rows."""
    count = side**3
    ids = tuple(first_id + i for i in range(count))
    positions = tuple(
        v
        for i in range(count)
        for v in (1.5 * (i % side) - 3.0, 1.5 * (i // side % side) - 3.0, 1.5 * (i // side**2))
    )
    rotations = tuple(
        v
        for i in range(count)
        for v in ((0, 0, math.sin(0.3), math.cos(0.3)) if i % 3 else (0, 0, 0, 1))
    )
    return ids, dict(
        positions=positions, rotations=rotations, geometry=(17,) * count, bits=(1,) * count
    )


# Six boxes on the X axis for the pair-reporter tests: (0, 1) and (2, 3) overlap.
PAIR_IDS = tuple(50_000 + i for i in range(6))
PAIR_ROWS = dict(
    positions=(0, 0, 0, 1.2, 0, 0, 5, 0, 0, 6.3, 0, 0, 10, 0, 0, 20, 0, 0),
    geometry=(17,) * 6,
    bits=(1,) * 6,
)


def status(result, expected=raw.ApiStatus.SUCCESS):
    actual = raw.ApiStatus(result.status)
    if actual != expected:
        raise AssertionError(f"expected {expected.name}, got {actual.name}")
    return result


class Cuda:
    """The CUDA driver calls the probe makes: a primary context, one stream and its ready
    event, device and pinned host allocations, copies and host functions."""

    def __init__(self):
        self.driver = (
            C.CDLL("nvcuda.dll", winmode=0x800) if os.name == "nt" else C.CDLL("libcuda.so.1")
        )
        p, u64 = C.c_void_p, C.c_uint64
        signatures = {
            "cuInit": [C.c_uint],
            "cuDevicePrimaryCtxRetain": [C.POINTER(p), C.c_int],
            "cuDevicePrimaryCtxRelease_v2": [C.c_int],
            "cuCtxGetCurrent": [C.POINTER(p)],
            "cuCtxSetCurrent": [p],
            "cuStreamCreate": [C.POINTER(p), C.c_uint],
            "cuStreamDestroy_v2": [p],
            "cuMemAlloc_v2": [C.POINTER(u64), C.c_size_t],
            "cuMemFree_v2": [u64],
            "cuMemcpyHtoD_v2": [u64, p, C.c_size_t],
            "cuStreamSynchronize": [p],
            "cuMemcpyDtoH_v2": [p, u64, C.c_size_t],
            "cuMemcpyHtoDAsync_v2": [u64, p, C.c_size_t, p],
            "cuMemHostAlloc": [C.POINTER(p), C.c_size_t, C.c_uint],
            "cuMemFreeHost": [p],
            "cuLaunchHostFunc": [p, CUDA_HOST_FN, p],
            "cuEventQuery": [p],
            "cuEventSynchronize": [p],
            "cuEventCreate": [C.POINTER(p), C.c_uint],
            "cuEventRecord": [p, p],
            "cuEventDestroy_v2": [p],
        }
        for name, args in signatures.items():
            function = getattr(self.driver, name)
            function.argtypes, function.restype = args, C.c_int
        self.call("cuInit", 0)
        self.previous, self.primary, self.stream = p(), p(), p()
        self.call("cuCtxGetCurrent", C.byref(self.previous))
        self.call("cuDevicePrimaryCtxRetain", C.byref(self.primary), 0)
        self.call("cuCtxSetCurrent", self.primary)
        self.call("cuStreamCreate", C.byref(self.stream), 1)
        self.ready = p()
        self.call("cuEventCreate", C.byref(self.ready), 2)
        self.allocations = []
        self.host_allocations = []

    def call(self, name, *args):
        code = getattr(self.driver, name)(*args)
        if code != 0:
            raise RuntimeError(f"{name} returned CUDA status {code}")

    def upload(self, storage):
        allocation = C.c_uint64()
        self.call("cuMemAlloc_v2", C.byref(allocation), C.sizeof(storage))
        self.allocations.append(allocation)
        self.call("cuMemcpyHtoD_v2", allocation, C.cast(storage, C.c_void_p), C.sizeof(storage))
        # A pageable copy may return before its DMA lands, and nothing orders the legacy
        # stream before the non-blocking one the library reads on: wait for it here.
        self.call("cuStreamSynchronize", None)
        return allocation.value

    def download(self, tensor, element, count):
        if tensor.device.device_type != raw.K_DL_CUDA or tensor.device.device_id != 0:
            raise AssertionError("result column is not on the requested CUDA device")
        if (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes) != (
            dl_code(element),
            C.sizeof(element) * 8,
            1,
        ):
            raise AssertionError("result column has an unexpected dtype")
        values = (element * count)()
        if count:
            self.call(
                "cuMemcpyDtoH_v2",
                C.cast(values, C.c_void_p),
                tensor.data + tensor.byte_offset,
                C.sizeof(values),
            )
        return tuple(values)

    def close(self):
        for allocation in self.allocations:
            self.call("cuMemFree_v2", allocation)
        for allocation in self.host_allocations:
            self.call("cuMemFreeHost", allocation)
        self.call("cuEventDestroy_v2", self.ready)
        self.call("cuStreamDestroy_v2", self.stream)
        self.call("cuCtxSetCurrent", self.previous)
        self.call("cuDevicePrimaryCtxRelease_v2", 0)


class Fixture:
    def __init__(
        self,
        path,
        *,
        host_budget=64 << 20,
        device_budget=64 << 20,
        operation_capacity=32,
        half_extents=(1, 0.5, 0.25, 0.25, 0.5, 0.75),
    ):
        self.cuda = Cuda()
        self.library = Library.open(path, require_explicit=True)
        self.exports = self.library.exports | self.library.require_explicit()
        self.keepalive, self.plans, self.operations, self.populations = [], [], [], []
        self.reporters = []
        self.query_operations = set()
        self.population, self.instance = C.c_uint64(), C.c_uint64()
        self.device_budget = raw.DeviceBudget(CUDA, device_budget)
        desc = raw.InstanceDesc(
            operation_capacity, 1, host_budget, C.pointer(self.device_budget), 1
        )
        self.call("ovspatialquery_create_instance", C.byref(desc), C.byref(self.instance))
        definitions = self.array((17, 18), C.c_uint64, (2,))
        extents = self.array(half_extents, C.c_double, (2, 3))
        table = raw.BoxDefinitionTable(C.pointer(definitions), C.pointer(extents))
        self.finish(
            self.call("ovspatialquery_enqueue_box_registration", self.instance, C.byref(table))
        )
        self.shapes = []  # (type, value) of the other families' definitions, when registered

    # A sphere, a capsule, a convex cube and a triangle-mesh box: definitions 21 to 24.
    SPHERE, CAPSULE, CONVEX, MESH = 21, 22, 23, 24

    def register_shapes(self):
        """Register one definition of each other exact family."""
        T = raw.GeometryType
        sphere = raw.SphereDefinitionTable(
            C.pointer(self.array((self.SPHERE,), C.c_uint64, (1,))),
            C.pointer(self.array((0.3,), C.c_double, (1,))),
        )
        self.finish(
            self.call("ovspatialquery_enqueue_sphere_registration", self.instance, C.byref(sphere))
        )
        capsule = raw.CapsuleDefinitionTable(
            C.pointer(self.array((self.CAPSULE,), C.c_uint64, (1,))),
            C.pointer(self.array((0.2,), C.c_double, (1,))),
            C.pointer(self.array((0.5,), C.c_double, (1,))),
        )
        self.finish(
            self.call(
                "ovspatialquery_enqueue_capsule_registration", self.instance, C.byref(capsule)
            )
        )
        corners = tuple(
            v
            for i in range(8)
            for v in (
                0.4 * (1 if i & 1 else -1),
                0.4 * (1 if i & 2 else -1),
                0.4 * (1 if i & 4 else -1),
            )
        )
        faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
        convex = raw.ConvexDefinitionTable(
            C.pointer(self.array((self.CONVEX,), C.c_uint64, (1,))),
            C.pointer(self.array((0,), C.c_uint64, (1,))),
            C.pointer(self.array((8,), C.c_uint64, (1,))),
            C.pointer(self.array((0,), C.c_uint64, (1,))),
            C.pointer(self.array((6,), C.c_uint64, (1,))),
            C.pointer(self.array(corners, C.c_float, (8, 3))),
            C.pointer(self.array(tuple(4 * i for i in range(6)), C.c_uint64, (6,))),
            C.pointer(self.array((4,) * 6, C.c_uint32, (6,))),
            C.pointer(self.array(tuple(i for face in faces for i in face), C.c_uint32, (24,))),
        )
        self.finish(
            self.call("ovspatialquery_enqueue_convex_registration", self.instance, C.byref(convex))
        )
        mesh_corners = tuple(v * 0.35 / 0.4 for v in corners)
        triangles = tuple(i for a, b, c, d in faces for i in (a, b, c, a, c, d))
        mesh = raw.TriangleMeshDefinitionTable(
            C.pointer(self.array((self.MESH,), C.c_uint64, (1,))),
            C.pointer(self.array((0,), C.c_uint64, (1,))),
            C.pointer(self.array((8,), C.c_uint64, (1,))),
            C.pointer(self.array((0,), C.c_uint64, (1,))),
            C.pointer(self.array((12,), C.c_uint64, (1,))),
            C.pointer(self.array(mesh_corners, C.c_float, (8, 3))),
            C.pointer(self.array(triangles, C.c_uint32, (12, 3))),
            None,
        )
        self.finish(
            self.call(
                "ovspatialquery_enqueue_triangle_mesh_registration", self.instance, C.byref(mesh)
            )
        )
        self.shapes = [
            (T.SPHERE, self.SPHERE),
            (T.CAPSULE, self.CAPSULE),
            (T.CONVEX, self.CONVEX),
            (T.TRIANGLE_MESH, self.MESH),
        ]

    def call(self, name, *args, expected=raw.ApiStatus.SUCCESS):
        return status(self.exports[name](*args), expected)

    def array(self, values, element, shape, *, device=False):
        if len(values) != math.prod(shape):
            raise AssertionError("input shape does not cover values")
        storage, dimensions = (element * len(values))(*values), (C.c_int64 * len(shape))(*shape)
        strides = (C.c_int64 * len(shape))(
            *(math.prod(shape[index + 1 :]) for index in range(len(shape)))
        )
        data = (
            None
            if not values
            else self.cuda.upload(storage)
            if device
            else C.cast(storage, C.c_void_p).value
        )
        tensor = raw.DLTensor(
            data,
            CUDA if device else CPU,
            len(shape),
            raw.DLDataType(dl_code(element), C.sizeof(element) * 8, 1),
            dimensions,
            strides,
            0,
        )
        array = raw.InputArray(tensor, raw.InputOwnership.BORROWED_UNTIL_TERMINAL)
        self.keepalive.extend((storage, dimensions, strides, array))
        return array

    def create_ordinary(self, *, pending=1):
        self.population = C.c_uint64()
        desc = raw.PopulationDesc(3, pending)
        self.call(
            "ovspatialquery_create_explicit_population",
            self.instance,
            C.byref(desc),
            C.byref(self.population),
        )
        self.populations.append(C.c_uint64(self.population.value))

    def create(
        self,
        ids=IDS,
        *,
        expected=raw.ApiStatus.SUCCESS,
        pending=1,
        slot_capacity=0,
        cpu=False,
        preparation=raw.Preparation.ON_DEVICE,
        **columns,
    ):
        """Create an ordinary population and publish its first complete set: prepared on
        the device unless `preparation` says otherwise, or on the host with `cpu`."""
        self.create_ordinary(pending=pending)
        self.publish(
            0,
            self.rows(ids, device=not cpu, **columns),
            expected=expected,
            cpu=cpu,
            preparation=preparation,
            slot_capacity=slot_capacity,
        )

    def gate(self):
        """Hold the producer stream in a host function until the returned event is set, so
        publications enqueued meanwhile stay pending on the device."""
        gate = threading.Event()

        @CUDA_HOST_FN
        def delayed_producer(_):
            gate.wait(5)

        self.keepalive.append(delayed_producer)
        self.cuda.call("cuLaunchHostFunc", self.cuda.stream, delayed_producer, None)
        return gate

    def rows(
        self,
        ids=IDS,
        *,
        positions=None,
        geometry=None,
        bits=None,
        rotations=None,
        scales=None,
        float32=False,
        device=True,
        types=None,
        poses=False,
    ):
        count = len(ids)
        positions = (
            positions if positions is not None else (0, 0, 0, 10, 0, 0, 0, 0, 0)[: 3 * count]
        )
        geometry = (
            geometry
            if geometry is not None
            else tuple(GEOMETRIES[IDS.index(i)] if i in IDS else 17 for i in ids)
        )
        with_bits = bits is not None or not poses  # pose rows carry bits only when given
        bits = (
            bits if bits is not None else tuple(BITS[IDS.index(i)] if i in IDS else 1 for i in ids)
        )
        rotations = rotations if rotations is not None else (0, 0, 0, 1) * count
        element = C.c_float if float32 else C.c_double
        types = types if types is not None else (raw.GeometryType.BOX,) * count
        # A pose update names objects and moves them; the binding columns stay NULL
        # and the bits column is present only when given (POSES_AND_BITS).
        columns = [
            self.array(ids, C.c_uint64, (count,), device=device),
            None if poses else self.array(types, C.c_int32, (count,), device=device),
            None if poses else self.array(geometry, C.c_uint64, (count,), device=device),
            self.array(positions, element, (count, 3), device=device),
            self.array(rotations, element, (count, 4), device=device),
            self.array(scales, element, (count,), device=device) if scales is not None else None,
            self.array(words(bits), C.c_uint64, (count, 2), device=device) if with_bits else None,
        ]
        return raw.ExplicitInstanceTable(
            *(C.pointer(column) if column is not None else None for column in columns)
        )

    def move(self, ordinal, ids, positions, *, base=None, cpu=False, **publish):
        """Publish a POSES update moving `ids` to `positions`, built on `ordinal - 1`
        unless `base` names another ordinal."""
        return self.publish(
            ordinal,
            self.rows(ids, positions=positions, poses=True, device=not cpu),
            base=ordinal - 1 if base is None else base,
            kind=raw.ExplicitUpdateKind.POSES,
            cpu=cpu,
            **publish,
        )

    def publish(
        self,
        ordinal,
        rows=None,
        *,
        base=None,
        removed=None,
        expected=raw.ApiStatus.SUCCESS,
        terminal=raw.ApiStatus.SUCCESS,
        cpu=False,
        kind=raw.ExplicitUpdateKind.COMPLETE_ROWS,
        preparation=raw.Preparation.ON_DEVICE,
        wait=True,
        slot_capacity=0,
    ):
        rows = self.rows() if rows is None else rows
        self.cuda.call("cuEventRecord", self.cuda.ready, self.cuda.stream)
        execution = raw.DynamicExecution(
            CUDA, raw.CudaSync(1, self.cuda.stream.value, self.cuda.ready.value), preparation
        )
        execution_ptr = None if cpu else C.pointer(execution)
        if base is None:
            desc = raw.ExplicitPopulationSet(ordinal, rows, execution_ptr, slot_capacity)
            name = "ovspatialquery_enqueue_explicit_population_set"
        else:
            if removed is None:
                removed = self.array((), C.c_uint64, (0,), device=not cpu)
            desc = raw.ExplicitPopulationUpdate(
                base,
                ordinal,
                kind,
                rows,
                C.pointer(removed),
                execution_ptr,
            )
            name = "ovspatialquery_enqueue_explicit_population_update"
        if expected is None:
            # A transient refusal while pending work drains; the caller retries.
            result = self.exports[name](self.instance, self.population, C.byref(desc))
            if result.status in (raw.ApiStatus.BUSY, raw.ApiStatus.ORDINAL_MISMATCH):
                return None
            status(result)
        else:
            result = self.call(
                name, self.instance, self.population, C.byref(desc), expected=expected
            )
            if expected != raw.ApiStatus.SUCCESS:
                return None
        if not wait:
            operation = C.c_uint64(result.op_id)
            self.operations.append(operation)
            return operation
        return self.finish(result, terminal=terminal)

    def finish(self, result, *, terminal=raw.ApiStatus.SUCCESS):
        operation = C.c_uint64(result.op_id)
        self.operations.append(operation)
        return self.wait(operation, terminal=terminal)

    def wait(self, operation, *, terminal=raw.ApiStatus.SUCCESS, release=True):
        wait = raw.OpWaitResult()
        self.call("ovspatialquery_wait_op", self.instance, operation, WAIT_NS, C.byref(wait))
        observed = raw.OpStatus()
        self.call("ovspatialquery_query_op_status", self.instance, operation, C.byref(observed))
        status(raw.Result(observed.terminal_status), terminal)
        expected_state = (
            raw.OpState.COMPLETED if terminal == raw.ApiStatus.SUCCESS else raw.OpState.FAILURE
        )
        if wait.state != expected_state or observed.state != expected_state:
            raise AssertionError("wait/status disagree with expected terminal state")
        if release:
            self.release(operation)
        return operation

    def release(self, operation):
        if operation.value in self.query_operations:
            done = C.c_void_p()
            self.cuda.call("cuEventCreate", C.byref(done), 2)
            self.cuda.call("cuEventRecord", done, self.cuda.stream)
            self.cuda.call("cuEventSynchronize", done)
            event = raw.CudaEvent(done.value)
            self.call(
                "ovspatialquery_destroy_op_after_cuda",
                self.instance,
                C.byref(operation),
                C.byref(event),
            )
        else:
            self.call("ovspatialquery_destroy_op", self.instance, C.byref(operation))

    @staticmethod
    def retry(call, transient=(raw.ApiStatus.BUSY,)):
        """Repeat `call` while it returns a transient status, for up to ten seconds; the
        last result."""
        deadline = time.monotonic() + 10
        while True:
            result = call()
            if result.status not in transient or time.monotonic() >= deadline:
                return result
            time.sleep(0.001)

    def destroy(self, name, *args):
        """Destroy a handle, retrying while the library reports it BUSY."""
        status(self.retry(lambda: self.exports[name](*args)))

    def info(self):
        info = raw.PopulationInfo()
        self.call(
            "ovspatialquery_get_population_info", self.instance, self.population, C.byref(info)
        )
        return info

    def usage(self, device):
        value = C.c_uint64()
        self.call("ovspatialquery_get_instance_memory_usage", self.instance, device, C.byref(value))
        return value.value

    def plan(
        self,
        *,
        capacity=9,
        slots=1,
        half_extent=0.05,
        device=CUDA,
        precision=raw.QueryPrecision.BOUNDS_CANDIDATE,
        hit_mode=raw.HitMode.MULTIPLE,
        queries=3,
        hits_per_query=3,
    ):
        target = raw.TargetRef(kind=raw.TargetKind.POPULATION)
        target.handle.population = self.population.value
        # `half_extent` is a cube's, a per-axis triple, or ("sphere", radius) /
        # ("box", triple) for the other overlap shapes.
        if isinstance(half_extent, tuple) and half_extent[0] == "sphere":
            shape = raw.OverlapBatchShape(kind=raw.OverlapShapeKind.SPHERE)
            shape.data.sphere.radius = half_extent[1]
        elif isinstance(half_extent, tuple) and half_extent[0] == "box":
            shape = raw.OverlapBatchShape(kind=raw.OverlapShapeKind.BOX)
            shape.data.box.half_extents[:] = half_extent[1]
        else:
            shape = raw.OverlapBatchShape(kind=raw.OverlapShapeKind.AABB)
            shape.data.aabb.half_extents[:] = (
                tuple(half_extent) if isinstance(half_extent, tuple) else (half_extent,) * 3
            )
        desc = raw.OverlapPlanDesc(
            target,
            hit_mode,
            precision,
            shape,
            device,
            queries,
            hits_per_query if hit_mode == raw.HitMode.MULTIPLE else 1,
            capacity,
            slots,
            raw.BatchResultField.GEOMETRY_IDENTITY,
            0,
        )
        plan = C.c_uint64()
        self.call("ovspatialquery_create_overlap_plan", self.instance, C.byref(desc), C.byref(plan))
        self.plans.append(plan)
        return plan

    def query(
        self,
        plan,
        *,
        ordinal=None,
        admitted=False,
        centers=(0, 0, 0, 10, 0, 0, 0, 0, 0),
        bits=(1, 2, 2),
        release=True,
        expected=raw.ApiStatus.SUCCESS,
        device=True,
        wait=True,
        half_extents=None,
        rotations=None,
        rotation_element=C.c_float,
        exact=False,
    ):
        count = len(bits)
        positions = self.array(centers, C.c_double, (count, 3), device=device)
        query_bits = self.array(words(bits), C.c_uint64, (count, 2), device=device)
        # BOX plans take a float32[Q,4] or float64[Q,4] rotation per row (ADR-0026).
        quaternions = (
            self.array(rotations, rotation_element, (count, 4), device=device)
            if rotations is not None
            else None
        )
        # Per-row AABB half extents (float64[Q,3]) replace the plan's shape.
        extents = (
            self.array(half_extents, C.c_double, (count, 3), device=device)
            if half_extents is not None
            else None
        )
        kind = (
            raw.OrdinalSelectionKind.ADMITTED_EXACT
            if admitted
            else raw.OrdinalSelectionKind.HEAD
            if ordinal is None
            else raw.OrdinalSelectionKind.RETAINED_EXACT
        )
        selection = raw.OrdinalSelection(kind, 0 if ordinal is None else ordinal)
        batch = raw.OverlapBatch(
            selection,
            count,
            C.pointer(query_bits),
            C.pointer(positions),
            C.pointer(quaternions) if quaternions is not None else None,
            C.pointer(extents) if extents is not None else None,
            None,
            raw.CudaSync(int(device), self.cuda.stream.value if device else 0, 0),
        )
        reservation = raw.OverlapBatchReservation()
        result = self.retry(
            lambda: self.exports["ovspatialquery_enqueue_overlap_plan"](
                self.instance, plan, C.byref(batch), C.byref(reservation)
            ),
            (raw.ApiStatus.RESULT_CAPACITY,) if expected == raw.ApiStatus.SUCCESS else (),
        )
        status(result, expected)
        if expected != raw.ApiStatus.SUCCESS:
            return None
        operation = C.c_uint64(result.op_id)
        self.operations.append(operation)
        if device:
            self.query_operations.add(operation.value)  # retired after its consumer event
        if not wait:
            return operation, reservation, count
        self.wait(operation, release=False)
        return self.read_query(operation, count, release=release, exact=exact)

    def column(self, tensor, element, count):
        """One result column as a tuple, from the device or, for a CPU plan, from host memory."""
        if tensor.device.device_type == raw.K_DL_CPU:
            if not count:
                return ()
            return tuple((element * count).from_address(tensor.data + tensor.byte_offset))
        return self.cuda.download(tensor, element, count)

    def read_failed_query(self, operation, reservation, count, *, release=True):
        """A launch whose selected publication failed (ADR-0021): fetch reports the failure
        and the reservation's device columns hold the safe empty result."""
        fetched = raw.OverlapBatchResult()
        self.call(
            "ovspatialquery_fetch_overlap_batch",
            self.instance,
            operation,
            C.byref(fetched),
            expected=raw.ApiStatus.OPERATION_FAILED,
        )
        self.cuda.call("cuEventSynchronize", reservation.common.result_ready.value)
        columns = reservation.columns
        query_status = self.cuda.download(columns.hit_ranges.query_status, C.c_uint32, count)
        counts = self.cuda.download(columns.hit_ranges.hit_counts, C.c_uint64, count)
        stored = self.cuda.download(columns.hit_ranges.stored_hit_count, C.c_uint64, 1)[0]
        if release:
            self.release(operation)
        return query_status, counts, stored

    def read_query(self, operation, count, *, release=True, exact=False):
        fetched = raw.OverlapBatchResult()
        self.call("ovspatialquery_fetch_overlap_batch", self.instance, operation, C.byref(fetched))
        on_device = fetched.common.execution_device.device_type == raw.K_DL_CUDA
        truth = raw.HitTruth.EXACT_SURFACE if exact else raw.HitTruth.BOUNDS_CANDIDATE
        if fetched.common.truth != truth:
            raise AssertionError("unexpected candidate/exact tag")
        if on_device:
            self.cuda.call("cuEventSynchronize", fetched.common.result_ready.value)
        columns = fetched.columns
        query_status = self.column(columns.hit_ranges.query_status, C.c_uint32, count)
        if query_status != (raw.BatchRowStatus.SUCCESS,) * count:
            raise AssertionError(f"query rows failed: {query_status}")
        stored = self.column(columns.hit_ranges.stored_hit_count, C.c_uint64, 1)[0]
        offsets = self.column(columns.hit_ranges.hit_offsets, C.c_uint64, count)
        counts = self.column(columns.hit_ranges.hit_counts, C.c_uint64, count)
        truncated = self.column(columns.hit_ranges.truncated, C.c_uint8, count)
        ids = self.column(columns.object_ids, C.c_uint64, stored)
        types = self.column(columns.geometry.types, C.c_int32, stored)
        values = self.column(columns.geometry.values, C.c_uint64, stored)
        for object_id, kind, geometry in zip(ids, types, values):
            # The initial objects are boxes 17 and 18; objects `rows()` adds without a
            # geometry are boxes 17; other identities are checked by their tests.
            if object_id not in IDS and geometry != 17:
                continue
            expected_geometry = GEOMETRIES[IDS.index(object_id)] if object_id in IDS else 17
            if kind != raw.GeometryType.BOX or geometry != expected_geometry:
                raise AssertionError("result geometry identity changed")
        rows = tuple(
            frozenset(ids[offset : offset + size]) for offset, size in zip(offsets, counts)
        )
        if release:
            self.release(operation)
        return rows, truncated, fetched.common.source_ordinal, operation

    def path_plan(
        self,
        family,
        *,
        device=CUDA,
        hit_mode=raw.HitMode.MULTIPLE,
        radius=0.0,
        half_extents=(0.0, 0.0, 0.0),
        hits_per_query=64,
        precision=raw.QueryPrecision.BOUNDS_CANDIDATE,
        queries=4,
        fields=raw.BatchResultField.GEOMETRY_IDENTITY | raw.BatchResultField.DISTANCE,
    ):
        """A ray, sphere sweep or box sweep plan over the population: four rows by default."""
        target = raw.TargetRef(kind=raw.TargetKind.POPULATION)
        target.handle.population = self.population.value
        bounds = precision
        # A sweep reports one hit per row, so its slot holds exactly the rows.
        capacity = 256 if family == "ray" else queries
        if family == "ray":
            desc = raw.RaycastPlanDesc(
                target, hit_mode, bounds, device, queries, hits_per_query, capacity, 1, fields, 0.0
            )
            name = "ovspatialquery_create_raycast_plan"
        elif family == "sphere":
            desc = raw.SphereSweepPlanDesc(
                target, bounds, radius, device, queries, 1, fields, capacity, 0.0
            )
            name = "ovspatialquery_create_sphere_sweep_plan"
        else:
            desc = raw.BoxSweepPlanDesc(
                target,
                bounds,
                (C.c_double * 3)(*half_extents),
                device,
                queries,
                1,
                fields,
                capacity,
                0.0,
            )
            name = "ovspatialquery_create_box_sweep_plan"
        plan = C.c_uint64()
        self.call(name, self.instance, C.byref(desc), C.byref(plan))
        self.plans.append(plan)
        return plan

    def path_query(
        self,
        family,
        plan,
        *,
        origins,
        directions,
        max_distance,
        bits,
        device=True,
        rotations=None,
        exact=False,
        contacts=False,
        expected=raw.ApiStatus.SUCCESS,
    ):
        """Per row, the sorted (object id, rounded entry distance) pairs a ray, sphere sweep
        or box sweep plan reports; sweeps report their single closest candidate. With
        `contacts`, an exact plan's hits as reported instead: object id, distance, position,
        normal and source feature, unrounded and in their reported order."""
        count = len(bits)
        arrays = [
            self.array(words(bits), C.c_uint64, (count, 2), device=device),
            self.array(origins, C.c_double, (count, 3), device=device),
            self.array(directions, C.c_float, (count, 3), device=device),
            self.array((max_distance,) * count, C.c_double, (count,), device=device),
        ]
        sync = raw.CudaSync(int(device), self.cuda.stream.value if device else 0, 0)
        selection = raw.OrdinalSelection(raw.OrdinalSelectionKind.HEAD, 0)
        if family == "ray":
            batch = raw.RaycastBatch(
                selection,
                count,
                C.pointer(arrays[0]),
                C.pointer(arrays[1]),
                C.pointer(arrays[2]),
                None,
                C.pointer(arrays[3]),
                None,
                sync,
            )
            enqueue, fetch, result = (
                "ovspatialquery_enqueue_raycast_plan",
                "ovspatialquery_fetch_raycast_batch",
                raw.RaycastBatchResult(),
            )
            reservation = raw.RaycastBatchReservation()
        elif family == "sphere":
            batch = raw.SphereSweepBatch(
                selection,
                count,
                C.pointer(arrays[0]),
                C.pointer(arrays[1]),
                C.pointer(arrays[2]),
                None,
                C.pointer(arrays[3]),
                None,
                sync,
            )
            enqueue, fetch, result = (
                "ovspatialquery_enqueue_sphere_sweep_plan",
                "ovspatialquery_fetch_sphere_sweep_batch",
                raw.SphereSweepBatchResult(),
            )
            reservation = raw.SphereSweepBatchReservation()
        else:
            quaternions = self.array(rotations, C.c_float, (count, 4), device=device)
            arrays.append(quaternions)
            batch = raw.BoxSweepBatch(
                selection,
                count,
                C.pointer(arrays[0]),
                C.pointer(arrays[1]),
                C.pointer(quaternions),
                C.pointer(arrays[2]),
                None,
                C.pointer(arrays[3]),
                sync,
            )
            enqueue, fetch, result = (
                "ovspatialquery_enqueue_box_sweep_plan",
                "ovspatialquery_fetch_box_sweep_batch",
                raw.SphereSweepBatchResult(),
            )
            reservation = raw.SphereSweepBatchReservation()
        admitted = self.call(
            enqueue, self.instance, plan, C.byref(batch), C.byref(reservation), expected=expected
        )
        if expected != raw.ApiStatus.SUCCESS:
            return None
        operation = C.c_uint64(admitted.op_id)
        self.operations.append(operation)
        if device:
            self.query_operations.add(operation.value)
        self.wait(operation, release=False)
        self.call(fetch, self.instance, operation, C.byref(result))
        if result.common.execution_device.device_type == raw.K_DL_CUDA:
            self.cuda.call("cuEventSynchronize", result.common.result_ready.value)
        columns = result.columns
        statuses = self.column(columns.hit_ranges.query_status, C.c_uint32, count)
        if statuses != (raw.BatchRowStatus.SUCCESS,) * count:
            raise AssertionError(f"path rows failed: {statuses}")
        stored = self.column(columns.hit_ranges.stored_hit_count, C.c_uint64, 1)[0]
        offsets = self.column(columns.hit_ranges.hit_offsets, C.c_uint64, count)
        counts = self.column(columns.hit_ranges.hit_counts, C.c_uint64, count)
        ids = self.column(columns.object_ids, C.c_uint64, stored)
        contact = columns.contact.exact if exact else columns.contact.candidate
        distances = self.column(contact.distances, C.c_double, stored)
        if contacts:
            positions = self.column(contact.positions, C.c_double, 3 * stored)
            normals = self.column(contact.normals, C.c_float, 3 * stored)
            kinds = self.column(contact.features.kinds, C.c_uint32, stored)
            faces = self.column(contact.features.source_face_indices, C.c_uint64, stored)
            self.release(operation)
            return tuple(
                tuple(
                    (
                        ids[i],
                        distances[i],
                        positions[3 * i : 3 * i + 3],
                        normals[3 * i : 3 * i + 3],
                        kinds[i],
                        faces[i],
                    )
                    for i in range(offset, offset + size)
                )
                for offset, size in zip(offsets, counts)
            )
        self.release(operation)
        return tuple(
            tuple(sorted((ids[i], round(distances[i], 9)) for i in range(offset, offset + size)))
            for offset, size in zip(offsets, counts)
        )

    def pair_reporter(
        self,
        *,
        device=CUDA,
        capacity=256,
        delta_capacity=None,
        slots=1,
    ):
        """A self-pair reporter over the population with results on `device`."""
        target = raw.TargetRef(kind=raw.TargetKind.POPULATION)
        target.handle.population = self.population.value
        delta = capacity if delta_capacity is None else delta_capacity
        none = participation(0)
        desc = raw.PairReporterDesc(target, none, target, none, capacity, delta, slots, 0.0, device)
        reporter = C.c_uint64()
        self.call(
            "ovspatialquery_create_pair_reporter", self.instance, C.byref(desc), C.byref(reporter)
        )
        self.reporters.append(reporter)
        return reporter

    def advance(
        self,
        reporter,
        ordinal,
        *,
        expected=raw.ApiStatus.SUCCESS,
        admitted=False,
        stream=False,
        wait=True,
        details=False,
    ):
        """Advance the reporter to `ordinal` on both endpoints: (entered, exited) pair sets.

        `admitted` selects the ordinal with ADMITTED_EXACT; `stream` queues a CUDA
        reporter's advance on the fixture's stream after its ready event (ADR-0022).
        With `wait=False` the pending (operation, reservation) is returned for
        `finish_advance`; `details` adds the reservation and its device counts.
        """
        kind = (
            raw.OrdinalSelectionKind.ADMITTED_EXACT
            if admitted
            else raw.OrdinalSelectionKind.RETAINED_EXACT
        )
        selection = raw.OrdinalSelection(kind, ordinal)
        sync = raw.CudaSync()
        if stream:
            self.cuda.call("cuEventRecord", self.cuda.ready, self.cuda.stream)
            sync = raw.CudaSync(1, self.cuda.stream.value, self.cuda.ready.value)
        advance = raw.PairReporterAdvance(selection, selection, sync)
        reservation = raw.PairReporterReservation()
        # A retired advance frees its result slot only once its consumer event has
        # fired and the reaper has run, so a slot may still be held for a moment.
        admitted_result = self.retry(
            lambda: self.exports["ovspatialquery_enqueue_pair_reporter_advance"](
                self.instance, reporter, C.byref(advance), C.byref(reservation)
            ),
            (raw.ApiStatus.RESULT_CAPACITY,) if expected == raw.ApiStatus.SUCCESS else (),
        )
        status(admitted_result, expected)
        if expected != raw.ApiStatus.SUCCESS:
            return None
        operation = C.c_uint64(admitted_result.op_id)
        self.operations.append(operation)
        if not wait:
            return operation, reservation
        return self.finish_advance((operation, reservation), details=details)

    def finish_advance(
        self,
        pending,
        *,
        terminal=raw.ApiStatus.SUCCESS,
        details=False,
        outcome=raw.PairReporterOutcome.ADVANCED,
    ):
        """Wait for an advance: its (entered, exited) sets, or with `details` also the
        reservation and the counts its device tensor holds; a failed advance yields the
        counts alone. `outcome` is the typed outcome fetch must report."""
        operation, reservation = pending
        self.wait(operation, terminal=terminal, release=False)
        on_device = reservation.result_device.device_type == raw.K_DL_CUDA
        if on_device:
            self.query_operations.add(operation.value)
        result = raw.PairReporterResult()
        if terminal != raw.ApiStatus.SUCCESS:
            self.call(
                "ovspatialquery_fetch_pair_reporter",
                self.instance,
                operation,
                C.byref(result),
                expected=terminal,
            )
            counts = self.reservation_counts(reservation)
            self.release(operation)
            return counts
        self.call("ovspatialquery_fetch_pair_reporter", self.instance, operation, C.byref(result))
        if result.outcome != outcome:
            raise AssertionError(f"reporter outcome {result.outcome}, expected {outcome}")
        if on_device:
            self.cuda.call("cuEventSynchronize", result.result_ready.value)
        pairs = []
        for tensor in (result.entered_pairs, result.exited_pairs):
            count = tensor.shape[0] if tensor.shape else 0
            flat = self.column(tensor, C.c_uint64, 2 * count) if count else ()
            pairs.append(frozenset((flat[2 * i], flat[2 * i + 1]) for i in range(count)))
        counts = self.reservation_counts(reservation) if details else None
        if details and on_device:
            # The reservation's capacity-shaped rows are the fetched rows.
            for tensor, fetched in zip(
                (reservation.entered_pairs, reservation.exited_pairs), pairs
            ):
                count = len(fetched)
                flat = self.cuda.download(tensor, C.c_uint64, 2 * count) if count else ()
                rows = frozenset((flat[2 * i], flat[2 * i + 1]) for i in range(count))
                if rows != fetched:
                    raise AssertionError("reservation rows differ from the fetched rows")
        self.release(operation)
        return (tuple(pairs), reservation, counts) if details else tuple(pairs)

    def reservation_counts(self, reservation):
        """The (outcome, entered, exited, accepted) counts a CUDA reporter's reservation
        describes, after its ready event; None for a CPU reporter."""
        if reservation.result_device.device_type != raw.K_DL_CUDA or not reservation.counts.shape:
            return None
        self.cuda.call("cuEventSynchronize", reservation.result_ready.value)
        return self.cuda.download(reservation.counts, C.c_uint64, 4)

    def cpu_pairs(self, first_bits, second_bits, capacity=8):
        """The head's pair snapshot between objects carrying `first_bits` and `second_bits`."""
        target = raw.TargetRef(kind=raw.TargetKind.POPULATION)
        target.handle.population = self.population.value
        head = raw.QueryTarget(target, raw.OrdinalSelection(raw.OrdinalSelectionKind.HEAD, 0))
        desc = raw.PairSnapshotDesc(
            head, participation(first_bits), head, participation(second_bits), 0.0
        )
        pairs = (raw.Pair * capacity)()
        buffer, result = raw.PairBuffer(pairs, capacity), raw.PairSnapshotResult()
        self.call(
            "ovspatialquery_query_pairs_sync",
            self.instance,
            C.byref(desc),
            C.byref(buffer),
            C.byref(result),
        )
        return frozenset(
            (pairs[i].first.value, pairs[i].second.value) for i in range(result.stored_pair_count)
        )

    def cpu_query(
        self,
        centers,
        bits,
        *,
        half_extent,
        expected=raw.ApiStatus.SUCCESS,
        selection=raw.OrdinalSelection(raw.OrdinalSelectionKind.HEAD, 0),
        capacity=len(IDS),
        rotations=None,
    ):
        target = raw.TargetRef(kind=raw.TargetKind.POPULATION)
        target.handle.population = self.population.value
        selected = raw.QueryTarget(target, selection)
        rows = []
        for index, query_bits in enumerate(bits):
            center = centers[3 * index : 3 * index + 3]
            if isinstance(half_extent, tuple) and half_extent[0] == "sphere":
                shape = raw.OverlapShape(kind=raw.OverlapShapeKind.SPHERE)
                shape.data.sphere.center[:] = center
                shape.data.sphere.radius = half_extent[1]
            elif isinstance(half_extent, tuple) and half_extent[0] == "box":
                shape = raw.OverlapShape(kind=raw.OverlapShapeKind.BOX)
                shape.data.box.center[:] = center
                shape.data.box.half_extents[:] = half_extent[1]
                shape.data.box.rotation_xyzw[:] = rotations[4 * index : 4 * index + 4]
            else:
                shape = raw.OverlapShape(kind=raw.OverlapShapeKind.AABB)
                half = half_extent if isinstance(half_extent, tuple) else (half_extent,) * 3
                shape.data.aabb.min_corner[:] = tuple(value - h for value, h in zip(center, half))
                shape.data.aabb.max_corner[:] = tuple(value + h for value, h in zip(center, half))
            desc = raw.OverlapDesc(
                raw.QueryCommon(
                    selected, raw.QueryPrecision.BOUNDS_CANDIDATE, participation(query_bits), 0
                ),
                shape,
            )
            hits = (raw.OverlapHit * capacity)()
            buffer, result = raw.OverlapHitBuffer(hits, capacity), raw.MultipleResult()
            self.call(
                "ovspatialquery_overlap_multiple_sync",
                self.instance,
                C.byref(desc),
                C.byref(buffer),
                C.byref(result),
                expected=expected,
            )
            if expected != raw.ApiStatus.SUCCESS:
                return None
            if result.truncated:
                raise AssertionError("CPU reference truncated its candidates")
            rows.append(
                frozenset(
                    hits[row].identity.object_id.value for row in range(result.stored_hit_count)
                )
            )
        return tuple(rows)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            for operation in self.operations:
                if operation.value:
                    self.release(operation)
            for reporter in reversed(self.reporters):
                self.destroy("ovspatialquery_destroy_pair_reporter", self.instance, reporter)
            for plan in reversed(self.plans):
                self.destroy("ovspatialquery_destroy_batch_plan", self.instance, plan)
            for population in reversed(self.populations):
                self.destroy("ovspatialquery_destroy_population", self.instance, population)
            for kind, value in [
                (raw.GeometryType.BOX, 17),
                (raw.GeometryType.BOX, 18),
                *self.shapes,
            ]:
                self.destroy(
                    "ovspatialquery_destroy_geometry_definition",
                    self.instance,
                    raw.GeometryRef(kind, value),
                )
            self.destroy("ovspatialquery_destroy_instance", self.instance)
        finally:
            self.library.close()
            self.cuda.close()


class DevicePublicationInstalledTests(unittest.TestCase):
    library_path: Path

    def test_delayed_producer_event_orders_async_pose_replacement(self):
        with Fixture(self.library_path) as f:
            f.create()
            rows = f.rows((IDS[0],), positions=(0, 0, 0))
            updated = (C.c_double * 3)(6, 0, 0)
            pinned = C.c_void_p()
            f.cuda.call("cuMemHostAlloc", C.byref(pinned), C.sizeof(updated), 0)
            f.cuda.host_allocations.append(pinned)
            C.memmove(pinned, updated, C.sizeof(updated))
            gate = threading.Event()

            @CUDA_HOST_FN
            def delayed_producer(_):
                # CUDA forbids driver calls from a stream callback.
                gate.wait(2)

            f.keepalive.append(delayed_producer)
            f.cuda.call("cuLaunchHostFunc", f.cuda.stream, delayed_producer, None)
            f.cuda.call(
                "cuMemcpyHtoDAsync_v2",
                rows.translations.contents.tensor.data,
                pinned,
                C.sizeof(updated),
                f.cuda.stream,
            )
            f.cuda.call("cuEventRecord", f.cuda.ready, f.cuda.stream)
            release = threading.Timer(0.02, gate.set)
            release.start()
            try:
                f.publish(1, rows, base=0)
            finally:
                gate.set()
                release.join(2)
            self.assertEqual(
                f.query(f.plan(), centers=(6, 0, 0), bits=(1,))[0], (frozenset((IDS[0],)),)
            )

    def test_a_producer_thread_without_a_current_context_publishes(self):
        # The stream a CUDA publication names belongs to the device's primary context,
        # where the library runs its own work (ovspatialquery_cuda_sync_t), so a producer
        # thread with no current CUDA context publishes from CUDA rows as plans launch,
        # host- or device-prepared, and still has no current context afterwards.
        for preparation in (raw.Preparation.ON_HOST, raw.Preparation.ON_DEVICE):
            with self.subTest(preparation=preparation.name), Fixture(self.library_path) as f:
                f.create_ordinary()
                rows = f.rows()
                f.cuda.call("cuEventRecord", f.cuda.ready, f.cuda.stream)
                execution = raw.DynamicExecution(
                    CUDA, raw.CudaSync(1, f.cuda.stream.value, f.cuda.ready.value), preparation
                )
                desc = raw.ExplicitPopulationSet(0, rows, C.pointer(execution), 0)
                seen = {}

                def produce():
                    current = C.c_void_p()
                    f.cuda.call("cuCtxGetCurrent", C.byref(current))
                    seen["before"] = current.value
                    seen["result"] = f.exports["ovspatialquery_enqueue_explicit_population_set"](
                        f.instance, f.population, C.byref(desc)
                    )
                    f.cuda.call("cuCtxGetCurrent", C.byref(current))
                    seen["after"] = current.value

                producer = threading.Thread(target=produce)
                producer.start()
                producer.join(30)
                self.assertEqual((seen["before"], seen["after"]), (None, None))
                f.finish(status(seen["result"]))
                self.assertEqual(
                    f.query(f.plan(), centers=(10, 0, 0), bits=(2,))[0], (frozenset((IDS[1],)),)
                )

    def test_operation_capacity_preserves_head_then_allows_publication(self):
        with Fixture(self.library_path, operation_capacity=1) as f:
            f.create()
            held = f.query(f.plan(), release=False)
            rows = f.rows((IDS[0],), positions=(6, 0, 0))
            f.publish(1, rows, base=0, expected=raw.ApiStatus.OPERATION_CAPACITY)
            self.assertEqual(f.info().head_ordinal, 0)
            f.release(held[3])
            f.publish(1, rows, base=0)
            self.assertEqual(f.info().head_ordinal, 1)

    def test_large_and_tiny_local_boxes_scale_to_valid_world_bounds(self):
        # The input domain's extremes: local extents and scales of 2^60 and 2^-60.
        with Fixture(self.library_path, half_extents=(2.0**60,) * 3 + (2.0**-60,) * 3) as f:
            scales = (2.0**-60, 2.0**60, 2.0**-60)
            centers, bits = (0, 0, 0, 10, 0, 0, 0, 0, 0), (1, 2, 2)
            f.create(cpu=True, scales=scales)
            reference = f.cpu_query(centers, bits, half_extent=0.05)
            self.assertEqual(
                reference, (frozenset((IDS[0], IDS[2])), frozenset((IDS[1],)), frozenset((IDS[2],)))
            )
            # Row order differs from ID order; each row keeps its object's pose and scale.
            f.create(
                (IDS[2], IDS[0], IDS[1]),
                positions=(0, 0, 0, 0, 0, 0, 10, 0, 0),
                scales=(scales[2], scales[0], scales[1]),
            )
            self.assertEqual(f.query(f.plan(), centers=centers, bits=bits)[0], reference)

    def test_empty_population_and_failed_initial_set_recovery(self):
        with Fixture(self.library_path) as f:
            f.create(())
            self.assertEqual((f.info().ready, f.info().object_count), (1, 0))
            self.assertEqual(f.query(f.plan())[0], (frozenset(),) * 3)
        with Fixture(self.library_path) as f:
            f.create_ordinary()
            # Poses are validated only on the device, so a bad pose fails at completion.
            f.publish(
                0,
                f.rows(positions=(math.nan, 0, 0, 10, 0, 0, 0, 0, 0)),
                terminal=raw.ApiStatus.OPERATION_FAILED,
            )
            self.assertEqual(f.info().ready, 0)
            plan = f.plan()
            f.query(plan, expected=raw.ApiStatus.NOT_READY)
            f.publish(0)
            self.assertEqual((f.info().ready, f.info().head_ordinal), (1, 0))
            self.assertEqual(f.info().object_count, len(IDS))

    def test_set_defines_membership_and_rejects_bad_columns_at_admission(self):
        with Fixture(self.library_path) as f:
            f.create_ordinary()
            invalid = {
                "duplicate id": (f.rows((IDS[0], IDS[0])), raw.ApiStatus.INVALID_ARGUMENT),
                "zero id": (f.rows((0, IDS[0])), raw.ApiStatus.INVALID_ARGUMENT),
                "unregistered geometry": (
                    f.rows((IDS[0], IDS[1]), geometry=(17, 99)),
                    raw.ApiStatus.INVALID_ARGUMENT,
                ),
                "type naming another family's definition": (
                    f.rows((IDS[0], IDS[1]), types=(raw.GeometryType.BOX, raw.GeometryType.SPHERE)),
                    raw.ApiStatus.INVALID_ARGUMENT,
                ),
                "unknown geometry type": (
                    f.rows((IDS[0], IDS[1]), types=(raw.GeometryType.BOX, 9)),
                    raw.ApiStatus.INVALID_ARGUMENT,
                ),
            }
            for name, (rows, expected) in invalid.items():
                with self.subTest(invalid=name):
                    f.publish(0, rows, expected=expected)
                    self.assertEqual(f.info().ready, 0)
            # The set's row count, not the creation call, defines the membership.
            f.publish(0, f.rows(IDS[:2]))
            self.assertEqual((f.info().ready, f.info().object_count), (1, 2))

    def test_update_row_count_above_population_rejects_at_admission(self):
        with Fixture(self.library_path) as f:
            f.create()
            f.publish(
                1,
                f.rows((*IDS, IDS[0]), positions=(0, 0, 0) * 4),
                base=0,
                expected=raw.ApiStatus.INVALID_ARGUMENT,
            )
            self.assertEqual(f.info().head_ordinal, 0)

    def test_poses_update_moves_objects_and_rejects_binding_columns(self):
        with Fixture(self.library_path) as f:
            f.create()
            plan = f.plan()
            initial = (frozenset((IDS[0], IDS[2])), frozenset((IDS[1],)), frozenset((IDS[2],)))
            self.assertEqual(f.query(plan)[0], initial)
            poses = raw.ExplicitUpdateKind.POSES
            # Sparse and out of ID order; geometry, types and bits columns are NULL.
            moved = f.rows((IDS[1], IDS[0]), positions=(-10, 0, 0, 6, 0, 0), poses=True)
            f.publish(1, moved, base=0, kind=poses)
            self.assertEqual(f.info().head_ordinal, 1)
            centers, bits = (6, 0, 0, -10, 0, 0, 0, 0, 0), (1, 2, 2)
            self.assertEqual(
                f.query(plan, centers=centers, bits=bits)[0],
                (frozenset((IDS[0],)), frozenset((IDS[1],)), frozenset((IDS[2],))),
            )
            self.assertEqual(f.query(plan, ordinal=0)[0], initial)
            # A scale given by the update replaces the object's scale; NULL keeps it.
            scaled = f.rows((IDS[0],), positions=(6, 0, 0), scales=(2,), poses=True)
            f.publish(2, scaled, base=1, kind=poses)
            wide = (7.5, 0, 0)
            self.assertEqual(f.query(plan, centers=wide, bits=(1,))[0], (frozenset((IDS[0],)),))
            kept = f.rows((IDS[2],), positions=(0, 5, 0), poses=True)
            f.publish(3, kept, base=2, kind=poses)
            self.assertEqual(f.query(plan, centers=wide, bits=(1,))[0], (frozenset((IDS[0],)),))
            # Zero rows publish an unchanged copy.
            f.publish(4, f.rows((), positions=(), poses=True), base=3, kind=poses)
            self.assertEqual(f.info().head_ordinal, 4)
            self.assertEqual(f.query(plan, centers=wide, bits=(1,))[0], (frozenset((IDS[0],)),))
            # Binding columns are not allowed with POSES, nor are removals.
            f.publish(
                5, f.rows((IDS[0],)), base=4, kind=poses, expected=raw.ApiStatus.INVALID_ARGUMENT
            )
            removed = f.array((IDS[2],), C.c_uint64, (1,), device=True)
            f.publish(
                5,
                f.rows((IDS[0],), poses=True),
                base=4,
                removed=removed,
                kind=poses,
                expected=raw.ApiStatus.INVALID_ARGUMENT,
            )
            self.assertEqual(f.info().head_ordinal, 4)
            # A row naming an unknown object fails the publication on the device.
            f.publish(
                5,
                f.rows((123,), poses=True),
                base=4,
                kind=poses,
                terminal=raw.ApiStatus.OPERATION_FAILED,
            )
            self.assertEqual(f.info().head_ordinal, 4)
            f.query(plan, ordinal=5, expected=raw.ApiStatus.ORDINAL_NOT_RETAINED)

    def test_cpu_reference_parity_for_rotated_scaled_touching_bounds(self):
        for float32 in (False, True):
            for angle in (0.27, math.pi / 2, 0.7, None):
                with self.subTest(float32=float32, angle=angle), Fixture(self.library_path) as f:
                    rotations = (
                        (0.5, 0.5, 0.5, 0.5)
                        if angle is None
                        else (0, 0, math.sin(angle / 2), math.cos(angle / 2))
                    ) * 3
                    positions, scales = (0, 0, 0, 10, 0, 0, 30, 0, 0), (1.25, 0.125, 2.5)
                    half = 0.0625
                    touching = 0.25 * scales[0] + half
                    centers = (
                        0,
                        0,
                        touching,
                        0,
                        0,
                        math.nextafter(touching, math.inf),
                        0,
                        0,
                        math.nextafter(touching, -math.inf),
                    )
                    if angle is None:
                        # This 3D rotation maps the local Z extent onto world X.
                        centers = (
                            touching,
                            0,
                            0,
                            math.nextafter(touching, math.inf),
                            0,
                            0,
                            math.nextafter(touching, -math.inf),
                            0,
                            0,
                        )
                    bits = (1, 1, 1)
                    f.create(
                        positions=positions, rotations=rotations, scales=scales, float32=float32
                    )
                    actual = f.query(f.plan(half_extent=half), centers=centers, bits=bits)[0]
                    f.create(
                        cpu=True,
                        positions=positions,
                        rotations=rotations,
                        scales=scales,
                        float32=float32,
                    )
                    reference = f.cpu_query(centers, bits, half_extent=half)
                    self.assertEqual(actual, reference)
                    self.assertEqual(
                        reference, (frozenset((IDS[0],)), frozenset(), frozenset((IDS[0],)))
                    )

    def test_grid_queries_match_the_host_path_over_a_lattice(self):
        # The device version's grid must answer exactly what the scan
        # did: a lattice of boxes with a few scaled (oversize) and rotated ones, queried
        # at cell corners, between cells, far outside the lattice, with a box covering
        # more cells than the grid serves (the scan fallback), and with bits filters;
        # a host-prepared twin queried on the CPU is the reference. Removal and a pose
        # update then rebuild the grid of the next version.
        side = 7
        ids = tuple(10_000 + i for i in range(side**3))
        positions = tuple(
            v
            for i in range(side**3)
            for v in (1.5 * (i % side) - 4.0, 1.5 * (i // side % side) - 4.0, 1.5 * (i // side**2))
        )
        scales = tuple(3.0 if i % 37 == 0 else 0.5 if i % 11 == 0 else 1.0 for i in range(side**3))
        rotations = tuple(
            v
            for i in range(side**3)
            for v in ((0, 0, math.sin(0.4), math.cos(0.4)) if i % 5 == 0 else (0, 0, 0, 1))
        )
        bits = tuple(1 + (i % 3) for i in range(side**3))
        geometry = (17,) * side**3
        centers = (
            *(0, 0, 0),
            *(1.5, 1.5, 1.5),  # on a lattice point
            *(0.75, 0.75, 0.75),  # between lattice points
            *(-4.0, -4.0, 0.0),  # the lattice's corner
            *(-3.99, 2.3, 7.1),
            *(-40.0, 0.0, 0.0),  # far outside: empty
            *(0.0, 0.0, 9.5),  # just above the top layer
            *(2.0, -1.0, 4.0),
        )
        query_bits = (7, 7, 7, 7, 1, 7, 2, 4)
        rows = dict(
            positions=positions, scales=scales, rotations=rotations, bits=bits, geometry=geometry
        )
        # 12 covers more cells than the lattice has slots (the scan fallback); the long
        # thin box is a conveyor segment over small boxes: hundreds of cells, on the grid.
        for half in (0.3, 1.0, 12.0, (60.0, 0.3, 0.3)):
            with self.subTest(half_extent=half), Fixture(self.library_path) as f:
                f.create(ids, **rows)
                plan = f.plan(
                    capacity=4096, slots=1, half_extent=half, queries=8, hits_per_query=512
                )
                actual = f.query(plan, centers=centers, bits=query_bits)[0]
                f.create(ids, cpu=True, **rows)
                reference = f.cpu_query(centers, query_bits, half_extent=half, capacity=4096)
                self.assertEqual(actual, reference)
                self.assertEqual(actual[5] == frozenset(), half != (60.0, 0.3, 0.3))
                self.assertTrue(any(actual))
        # One batch whose rows carry their own half extents: a point-like box, the
        # plan's cube, a long thin segment and a box wider than the grid serves.
        with Fixture(self.library_path) as f:
            f.create(ids, **rows)
            plan = f.plan(capacity=4096, slots=1, half_extent=1.0, queries=8, hits_per_query=512)
            per_row = ((0.3, 0.3, 0.3), (1.0, 1.0, 1.0), (60.0, 0.3, 0.3), (12.0, 12.0, 12.0))
            mixed = f.query(
                plan,
                centers=centers[: 3 * len(per_row)],
                bits=query_bits[: len(per_row)],
                half_extents=tuple(v for half in per_row for v in half),
            )[0]
            f.create(ids, cpu=True, **rows)
            for row, half in enumerate(per_row):
                reference = f.cpu_query(
                    centers[3 * row : 3 * row + 3],
                    query_bits[row : row + 1],
                    half_extent=half,
                    capacity=4096,
                )
                self.assertEqual((mixed[row],), reference)
        with Fixture(self.library_path) as f:
            f.create(ids, slot_capacity=side**3 + 2, **rows)
            plan = f.plan(capacity=4096, slots=1, half_extent=1.0, queries=8, hits_per_query=512)
            before = f.query(plan, centers=centers, bits=query_bits)[0]
            # Remove the object at the origin and move a neighbour onto it; add one far away.
            removed = f.array((ids[0],), C.c_uint64, (1,), device=True)
            f.publish(
                1,
                f.rows(
                    (ids[1], 20_000),
                    positions=(0, 0, 0, 50, 50, 50),
                    geometry=(17, 17),
                    bits=(1, 1),
                ),
                base=0,
                removed=removed,
            )
            after = f.query(plan, centers=centers, bits=query_bits)[0]
            self.assertNotIn(ids[0], after[0])
            self.assertIn(ids[1], after[0])
            self.assertEqual(
                f.query(plan, centers=(50, 50, 50), bits=(1,))[0], (frozenset((20_000,)),)
            )
            self.assertEqual(f.query(plan, ordinal=0, centers=centers, bits=query_bits)[0], before)
            f.move(2, (ids[1],), (50, 50, 50))
            self.assertEqual(
                f.query(plan, centers=(50, 50, 50), bits=(1,))[0], (frozenset((ids[1], 20_000)),)
            )
            self.assertNotIn(ids[1], f.query(plan, centers=centers, bits=query_bits)[0][0])

    def test_held_reader_survives_retention_eviction_and_slot_reuse(self):
        with Fixture(self.library_path) as f:
            f.create()
            plan = f.plan(slots=2)
            held = f.query(plan, release=False)
            for ordinal in range(1, 5):
                f.publish(
                    ordinal, f.rows((IDS[0],), positions=(ordinal * 2, 0, 0)), base=ordinal - 1
                )
            f.query(plan, ordinal=0, expected=raw.ApiStatus.ORDINAL_NOT_RETAINED)
            self.assertEqual(f.read_query(held[3], 3, release=False)[:3], held[:3])
            f.release(held[3])
            self.assertEqual(f.query(plan)[2], 4)

    def test_version_device_budget_rejects_successor_without_changing_head(self):
        # The first set charges the fixed membership tables and one version; a
        # retained successor charges one more version.
        with Fixture(self.library_path) as measured:
            measured.create()
            initial = measured.usage(CUDA)
            measured.publish(1, measured.rows((IDS[0],)), base=0)
            version = measured.usage(CUDA) - initial
            self.assertGreater(version, 0)
        with Fixture(self.library_path, device_budget=initial + version - 1) as tight:
            tight.create()
            tight.publish(1, tight.rows((IDS[0],)), base=0, expected=raw.ApiStatus.DEVICE_BUDGET)
            self.assertEqual(tight.info().head_ordinal, 0)

    def test_device_budget_rejects_first_set_at_admission(self):
        with Fixture(self.library_path, device_budget=1) as f:
            f.create(expected=raw.ApiStatus.DEVICE_BUDGET)
            self.assertEqual(f.info().ready, 0)

    def test_full_set_sparse_reorder_noop_retention_scale_rotation_and_filtering(self):
        with Fixture(self.library_path) as f:
            f.create_ordinary()
            self.assertEqual((f.info().ready, f.info().object_count), (0, 0))
            f.publish(0)
            self.assertEqual((f.info().ready, f.info().object_count), (1, 3))
            plan = f.plan()
            initial = (frozenset((IDS[0], IDS[2])), frozenset((IDS[1],)), frozenset((IDS[2],)))
            self.assertEqual(f.query(plan)[:3], (initial, (0, 0, 0), 0))
            rotation = (0, 0, 0, 1, 0, 0, math.sqrt(0.5), math.sqrt(0.5))
            rows = f.rows(
                (IDS[2], IDS[0]),
                positions=(20, 0, 0, 2, 0, 0),
                rotations=rotation,
                scales=(1, 2),
                float32=True,
            )
            f.publish(1, rows, base=0)
            self.assertEqual(
                f.query(plan, centers=(2.9, 0, 0), bits=(1,))[0], (frozenset((IDS[0],)),)
            )
            self.assertEqual(f.query(plan, ordinal=0)[0], initial)
            f.publish(2, f.rows((), positions=()), base=1)
            self.assertEqual(f.info().head_ordinal, 2)
            self.assertEqual(
                f.query(plan, centers=(2.9, 0, 0), bits=(1,))[0], (frozenset((IDS[0],)),)
            )

    def test_bad_device_rows_preserve_head_and_allow_recovery(self):
        with Fixture(self.library_path) as f:
            f.create()
            plan = f.plan()
            initial = f.query(plan)[0]
            invalid = [
                f.rows((IDS[0],), geometry=(18,)),
                f.rows((IDS[0],), positions=(math.nan, 0, 0)),
                f.rows((IDS[0],), positions=(1e100, 0, 0)),
                # Finite values beyond the input domain: a position, scales, and
                # a world AABB that leaves it although the position does not.
                f.rows((IDS[0],), positions=(2.0**61, 0, 0)),
                f.rows((IDS[0],), scales=(2.0**61,)),
                f.rows((IDS[0],), scales=(2.0**-61,)),
                f.rows((IDS[0],), positions=(2.0**60, 0, 0), scales=(2.0**20,)),
                f.rows((IDS[0],), rotations=(0, 0, 0, 0)),
                f.rows((IDS[0],), rotations=(0, 0, math.inf, 1)),
                f.rows((IDS[0],), scales=(-1,)),
                f.rows((IDS[0],), scales=(0,)),
                f.rows((IDS[0],), scales=(math.inf,)),
            ]
            for index, rows in enumerate(invalid):
                with self.subTest(invalid_row=index):
                    f.publish(1, rows, base=0, terminal=raw.ApiStatus.OPERATION_FAILED)
                    self.assertEqual(f.info().head_ordinal, 0)
                    self.assertEqual(f.query(plan)[0], initial)
                    f.query(plan, ordinal=1, expected=raw.ApiStatus.ORDINAL_NOT_RETAINED)
            f.publish(1, f.rows((IDS[0],), positions=(5, 0, 0)), base=0)
            self.assertEqual(f.info().head_ordinal, 1)

    def test_unsupported_operations_and_bad_base_reject_at_admission(self):
        with Fixture(self.library_path) as f:
            f.create()
            f.call(
                "ovspatialquery_destroy_geometry_definition",
                f.instance,
                raw.GeometryRef(raw.GeometryType.BOX, 17),
                expected=raw.ApiStatus.BUSY,
            )
            # A set on a population with a head is INVALID_STATE; an update does not
            # move a version between host and device preparation.
            f.publish(
                1,
                f.rows((IDS[0],), device=False),
                base=0,
                cpu=True,
                expected=raw.ApiStatus.NOT_SUPPORTED,
            )
            host = raw.Preparation.ON_HOST
            f.publish(1, preparation=host, expected=raw.ApiStatus.INVALID_STATE)
            f.publish(1, base=0, preparation=host, expected=raw.ApiStatus.NOT_SUPPORTED)
            f.cpu_query((0, 0, 0), (1,), half_extent=0.05, expected=raw.ApiStatus.NOT_SUPPORTED)
            # Plans are created before any version exists; the launch is what the
            # device-prepared version refuses.
            f.query(f.plan(device=CPU), device=False, expected=raw.ApiStatus.NOT_SUPPORTED)
            f.query(
                f.plan(precision=raw.QueryPrecision.EXACT), expected=raw.ApiStatus.NOT_SUPPORTED
            )
            # An ANY-hit overlap is served: the lowest overlapping id per row.
            self.assertEqual(
                f.query(f.plan(hit_mode=raw.HitMode.ANY))[0],
                (frozenset((min(IDS[0], IDS[2]),)), frozenset((IDS[1],)), frozenset((IDS[2],))),
            )
            ids = f.array(IDS, C.c_uint64, (len(IDS),))
            ids.ownership = raw.InputOwnership.COPIED_ON_SUBMISSION
            view = C.c_uint64()
            desc = raw.ExplicitViewDesc(C.pointer(ids))
            try:
                f.call(
                    "ovspatialquery_create_explicit_view",
                    f.instance,
                    f.population,
                    C.byref(desc),
                    C.byref(view),
                    expected=raw.ApiStatus.NOT_SUPPORTED,
                )
            finally:
                if view.value:
                    f.call("ovspatialquery_destroy_view", f.instance, view)
            if f.library.has_ovstage:
                f.exports.update(f.library.require_ovstage())
                # A participation table belongs to Ovstage populations only.
                table = raw.OvstageParticipationTable(participation(3), None, None, 0)
                f.call(
                    "ovspatialquery_set_ovstage_participation_map",
                    f.instance,
                    f.population,
                    C.byref(table),
                    expected=raw.ApiStatus.INVALID_ARGUMENT,
                )
            f.publish(100, base=99, expected=raw.ApiStatus.ORDINAL_MISMATCH)
            # Removing an unknown object is refused at admission; removals themselves
            # are supported on the device path (ADR-0024, amended).
            removed = f.array((123,), C.c_uint64, (1,), device=True)
            f.publish(1, base=0, removed=removed, expected=raw.ApiStatus.INVALID_ARGUMENT)
            self.assertEqual((f.info().head_ordinal, f.info().object_count), (0, 3))
        with Fixture(self.library_path) as f:
            # The other direction: a host-prepared head rejects device preparation.
            f.create(cpu=True)
            f.publish(1, f.rows((IDS[0],)), base=0, expected=raw.ApiStatus.NOT_SUPPORTED)
            self.assertEqual(f.info().head_ordinal, 0)
            self.assertEqual(
                f.cpu_query((0, 0, 0), (1,), half_extent=0.05), (frozenset((IDS[0], IDS[2])),)
            )

    def test_result_slot_capacity_and_hit_truncation(self):
        with Fixture(self.library_path) as f:
            f.create()
            plan = f.plan(capacity=1, half_extent=100)
            result = f.query(plan, centers=(0, 0, 0), bits=(3,), release=False)
            self.assertEqual(len(result[0][0]), 1)
            self.assertEqual(result[1], (1,))
            f.query(plan, centers=(0, 0, 0), bits=(3,), expected=raw.ApiStatus.RESULT_CAPACITY)
            f.release(result[3])
            self.assertEqual(f.query(plan, centers=(0, 0, 0), bits=(3,))[1], (1,))

    def test_admitted_selection_pipelines_pending_publications_without_a_host_wait(self):
        # ADR-0021: two device-prepared updates are admitted while the producer is still
        # busy, the second built on the first; CUDA launches select each by ADMITTED_EXACT
        # before either completes, and nothing waits on the host until the results are read.
        with Fixture(self.library_path) as f:
            f.create(pending=2)
            plan = f.plan(capacity=16, slots=4)
            gate = f.gate()
            try:
                first = f.move(1, (IDS[0],), (6, 0, 0), wait=False)
                second = f.move(2, (IDS[1],), (6, 0, 0), wait=False)
                info = f.info()
                self.assertEqual((info.head_ordinal, info.update_pending), (0, 1))
                # Completed-publication selections do not see pending work.
                f.query(plan, ordinal=1, expected=raw.ApiStatus.ORDINAL_NOT_RETAINED)
                f.query(plan, ordinal=2, expected=raw.ApiStatus.ORDINAL_NOT_RETAINED)
                head = f.query(plan, centers=(6, 0, 0), bits=(3,), wait=False)
                at_first = f.query(
                    plan, ordinal=1, admitted=True, centers=(6, 0, 0), bits=(3,), wait=False
                )
                at_second = f.query(
                    plan, ordinal=2, admitted=True, centers=(6, 0, 0), bits=(3,), wait=False
                )
                # A third publication exceeds the pending capacity; a stale base is a mismatch.
                f.move(3, (IDS[2],), (6, 0, 0), expected=raw.ApiStatus.BUSY)
                f.move(3, (IDS[2],), (6, 0, 0), base=0, expected=raw.ApiStatus.BUSY)
                self.assertEqual(f.cuda.driver.cuEventQuery(f.cuda.ready), 600)
            finally:
                gate.set()
            f.wait(first)
            f.wait(second)
            self.assertEqual((f.info().head_ordinal, f.info().update_pending), (2, 0))
            for operation, reservation, count in (head, at_first, at_second):
                f.wait(operation, release=False)
            self.assertEqual(f.read_query(head[0], head[2])[0], (frozenset(),))
            self.assertEqual(f.read_query(at_first[0], at_first[2])[0], (frozenset((IDS[0],)),))
            self.assertEqual(
                f.read_query(at_second[0], at_second[2])[0], (frozenset((IDS[0], IDS[1])),)
            )
            # Once complete, the same ordinals are ordinary retained versions.
            self.assertEqual(
                f.query(plan, ordinal=1, centers=(6, 0, 0), bits=(3,))[0], (frozenset((IDS[0],)),)
            )
            self.assertEqual(
                f.query(plan, ordinal=2, admitted=True, centers=(6, 0, 0), bits=(3,))[0],
                (frozenset((IDS[0], IDS[1])),),
            )
            # With the window drained a new base must be the head again.
            f.move(3, (IDS[2],), (6, 0, 0), base=1, expected=raw.ApiStatus.ORDINAL_MISMATCH)

    def test_failed_admitted_publication_fails_its_chain_and_its_launches(self):
        # ADR-0021: a bad row in the first pending publication fails it, the update built on
        # it, and the launch that selected the later ordinal, whose rows carry the FAILED
        # status with zero hits; the head is unchanged and publishing from it recovers.
        with Fixture(self.library_path) as f:
            f.create(pending=2)
            plan = f.plan(capacity=16)
            gate = f.gate()
            try:
                first = f.move(1, (IDS[0],), (math.nan, 0, 0), wait=False)
                second = f.move(2, (IDS[1],), (6, 0, 0), wait=False)
                launch = f.query(
                    plan, ordinal=2, admitted=True, centers=(6, 0, 0), bits=(3,), wait=False
                )
            finally:
                gate.set()
            f.wait(first, terminal=raw.ApiStatus.OPERATION_FAILED)
            f.wait(second, terminal=raw.ApiStatus.OPERATION_FAILED)
            f.wait(launch[0], terminal=raw.ApiStatus.OPERATION_FAILED, release=False)
            statuses, counts, stored = f.read_failed_query(*launch)
            self.assertEqual(statuses, (raw.BatchRowStatus.FAILED,))
            self.assertEqual((counts, stored), ((0,), 0))
            self.assertEqual((f.info().head_ordinal, f.info().update_pending), (0, 0))
            f.query(plan, ordinal=2, admitted=True, expected=raw.ApiStatus.ORDINAL_NOT_RETAINED)
            f.move(3, (IDS[1],), (6, 0, 0), base=0)
            self.assertEqual(f.info().head_ordinal, 3)
            self.assertEqual(
                f.query(plan, centers=(6, 0, 0), bits=(3,))[0], (frozenset((IDS[1],)),)
            )

    def test_admitted_selection_is_for_cuda_launches_and_completed_host_versions(self):
        # A completed retained ordinal is selectable as ADMITTED_EXACT from a CUDA launch
        # whatever prepared it, and a pending ordinal of a host-prepared publication is not
        # retained.
        with Fixture(self.library_path) as f:
            f.create(pending=2, cpu=True)
            plan = f.plan()
            self.assertEqual(
                f.query(plan, ordinal=0, admitted=True)[0],
                (frozenset((IDS[0], IDS[2])), frozenset((IDS[1],)), frozenset((IDS[2],))),
            )
            f.query(plan, ordinal=7, admitted=True, expected=raw.ApiStatus.ORDINAL_NOT_RETAINED)

    def test_publications_complete_in_any_worker_order_without_moving_the_head_back(self):
        # The fixture's instance has one worker, which takes the newest queued operation
        # first. Two chained updates queued behind a gated publication of another population
        # therefore complete newest first: the later ordinal becomes the head and the earlier
        # one joins the retained window behind it (ADR-0021).
        with Fixture(self.library_path) as f:
            f.create(pending=2)
            bodies = C.c_uint64(f.population.value)
            plan = f.plan(capacity=16)
            f.create(ids=(5, 6, 7), pending=1)
            other = C.c_uint64(f.population.value)
            gate = f.gate()
            try:
                # The worker blocks on this publication until the gate opens.
                blocking = f.move(1, (5,), (6, 0, 0), wait=False)
                f.population = bodies
                first = f.move(1, (IDS[0],), (6, 0, 0), wait=False)
                second = f.move(2, (IDS[1],), (6, 0, 0), wait=False)
            finally:
                gate.set()
            f.wait(second)
            f.wait(first)
            f.population = other
            f.wait(blocking)
            f.population = bodies
            info = f.info()
            self.assertEqual(
                (info.head_ordinal, info.retained_ordinal_count, info.update_pending), (2, 2, 0)
            )
            self.assertEqual(
                f.query(plan, centers=(6, 0, 0), bits=(3,))[0], (frozenset((IDS[0], IDS[1])),)
            )
            self.assertEqual(
                f.query(plan, ordinal=1, centers=(6, 0, 0), bits=(3,))[0], (frozenset((IDS[0],)),)
            )
            self.assertEqual(
                f.query(plan, ordinal=0, centers=(6, 0, 0), bits=(3,))[0], (frozenset(),)
            )

    def test_device_updates_remove_and_add_objects_within_the_slot_capacity(self):
        # Membership changes on the device path: a complete-rows update removes objects
        # and adds unknown IDs into free slots up to the set's slot capacity; removed IDs
        # are retired for the population's life; retained versions keep the object.
        with Fixture(self.library_path) as f:
            f.create(slot_capacity=5)
            plan = f.plan(capacity=16)
            self.assertEqual(f.info().object_count, 3)
            removed = f.array((IDS[1],), C.c_uint64, (1,), device=True)
            f.publish(1, f.rows(()), base=0, removed=removed)
            self.assertEqual((f.info().head_ordinal, f.info().object_count), (1, 2))
            self.assertEqual(f.query(plan, centers=(10, 0, 0), bits=(2,))[0], (frozenset(),))
            self.assertEqual(
                f.query(plan, ordinal=0, centers=(10, 0, 0), bits=(2,))[0], (frozenset((IDS[1],)),)
            )
            # A pose row naming the removed object fails the publication; the head stays.
            f.move(2, (IDS[1],), (6, 0, 0), terminal=raw.ApiStatus.OPERATION_FAILED)
            self.assertEqual(f.info().head_ordinal, 1)
            # A retired ID is never reused, an unknown ID cannot be removed, twice neither.
            f.publish(
                2,
                f.rows((IDS[1],), positions=(6, 0, 0), geometry=(18,), bits=(2,)),
                base=1,
                expected=raw.ApiStatus.INVALID_ARGUMENT,
            )
            f.publish(
                2,
                f.rows(()),
                base=1,
                removed=f.array((123,), C.c_uint64, (1,), device=True),
                expected=raw.ApiStatus.INVALID_ARGUMENT,
            )
            f.publish(
                2,
                f.rows(()),
                base=1,
                removed=f.array((IDS[1],), C.c_uint64, (1,), device=True),
                expected=raw.ApiStatus.INVALID_ARGUMENT,
            )
            # Two additions fill the remaining slots.
            f.publish(
                2,
                f.rows(
                    (1001, 1002), positions=(20, 0, 0, 30, 0, 0), geometry=(17, 17), bits=(1, 1)
                ),
                base=1,
            )
            self.assertEqual((f.info().head_ordinal, f.info().object_count), (2, 4))
            self.assertEqual(
                f.query(plan, centers=(20, 0, 0, 30, 0, 0), bits=(1, 1))[0],
                (frozenset((1001,)), frozenset((1002,))),
            )
            # Slots are never reused: the capacity of five is spent although four objects live.
            f.publish(
                3,
                f.rows((1003,), positions=(50, 0, 0), geometry=(17,), bits=(1,)),
                base=2,
                expected=raw.ApiStatus.INVALID_ARGUMENT,
            )
            # Added objects move like any other.
            f.move(3, (1001,), (40, 0, 0))
            self.assertEqual(f.query(plan, centers=(40, 0, 0), bits=(1,))[0], (frozenset((1001,)),))
            self.assertEqual(f.query(plan, centers=(20, 0, 0), bits=(1,))[0], (frozenset(),))
            self.assertEqual(
                f.query(plan, ordinal=0, centers=(10, 0, 0), bits=(2,))[0], (frozenset((IDS[1],)),)
            )

    def test_a_removed_object_releases_its_definition_once_no_version_binds_it(self):
        # Destroying a definition is BUSY only while a published version or an admitted
        # operation references it (BH-005). A device update that removes the last object
        # bound to a convex releases it once the versions holding that object leave the
        # retained window, and the population keeps publishing and answering after. An
        # exact CUDA plan created then over a readable version leaves the removed object,
        # whose definition is gone, out and answers like a host-prepared twin (BH-065).
        T = raw.GeometryType
        readable = raw.Preparation.ON_DEVICE_HOST_READABLE
        exact = raw.QueryPrecision.EXACT
        fields = (
            raw.BatchResultField.GEOMETRY_IDENTITY
            | raw.BatchResultField.DISTANCE
            | raw.BatchResultField.POSITION
            | raw.BatchResultField.NORMAL
            | raw.BatchResultField.SOURCE_FEATURE
        )
        rays = dict(
            origins=(12.0, 5.0, 0.0, 12.2, 5.0, 0.1, 0.0, 5.0, 0.0, -5.0, 0.0, 0.0),
            directions=(0.0, -1.0, 0.0) * 3 + (1.0, 0.0, 0.0),
            max_distance=60.0,
            bits=(1,) * 4,
            exact=True,
            contacts=True,
        )
        convex = raw.GeometryRef(T.CONVEX, Fixture.CONVEX)
        for preparation in (raw.Preparation.ON_DEVICE, readable):
            with self.subTest(preparation=preparation.name), Fixture(self.library_path) as f:
                f.register_shapes()
                f.create(
                    (1, 2),
                    slot_capacity=4,
                    preparation=preparation,
                    positions=(0, 0, 0, 10, 0, 0),
                    geometry=(Fixture.CONVEX, 17),
                    types=(T.CONVEX, T.BOX),
                    bits=(1, 1),
                )
                plan = f.plan(queries=1)
                removed = f.array((1,), C.c_uint64, (1,), device=True)
                f.publish(1, f.rows(()), base=0, removed=removed, preparation=preparation)

                def destroy_convex():
                    return f.exports["ovspatialquery_destroy_geometry_definition"](
                        f.instance, convex
                    ).status

                # Ordinal 0 is retained and still holds object 1.
                self.assertEqual(destroy_convex(), raw.ApiStatus.BUSY)
                for ordinal in range(2, 5):
                    f.move(ordinal, (2,), (10, 0, 0), preparation=preparation)
                self.assertEqual(destroy_convex(), raw.ApiStatus.SUCCESS)
                f.shapes.remove((T.CONVEX, Fixture.CONVEX))
                f.move(5, (2,), (12, 0, 0), preparation=preparation)
                self.assertEqual(
                    f.query(plan, centers=(12, 0, 0), bits=(1,))[0], (frozenset((2,)),)
                )
                if preparation != readable:
                    continue
                actual = f.path_query(
                    "ray", f.path_plan("ray", precision=exact, fields=fields), **rays
                )
                f.create((2,), cpu=True, positions=(12, 0, 0), geometry=(17,), bits=(1,))
                twin = f.path_query(
                    "ray", f.path_plan("ray", precision=exact, fields=fields), **rays
                )
                self.assertEqual(actual, twin)
                self.assertEqual({hit[0] for row in actual for hit in row}, {2})
        # A removal whose publication fails leaves the head with the object, and every
        # version built on that head keeps it, so its definition stays referenced.
        with Fixture(self.library_path) as f:
            f.register_shapes()
            f.create(
                (1, 2),
                slot_capacity=4,
                positions=(0, 0, 0, 10, 0, 0),
                geometry=(Fixture.CONVEX, 17),
                types=(T.CONVEX, T.BOX),
                bits=(1, 1),
            )
            f.publish(
                1,
                f.rows((2,), positions=(math.nan, 0, 0), geometry=(17,), bits=(1,)),
                base=0,
                removed=f.array((1,), C.c_uint64, (1,), device=True),
                terminal=raw.ApiStatus.OPERATION_FAILED,
            )
            for ordinal in range(2, 6):
                f.move(ordinal, (2,), (10, 0, 0), base=ordinal - 1 if ordinal > 2 else 0)
            self.assertEqual(
                f.exports["ovspatialquery_destroy_geometry_definition"](f.instance, convex).status,
                raw.ApiStatus.BUSY,
            )

    def test_an_exact_plan_releases_a_removed_mesh_once_its_versions_leave(self):
        # BH-005 with an exact CUDA plan over a readable population (BH-065): the plan
        # holds a mesh definition only while a catalogue it keeps reads a version that
        # binds the mesh, so once the mesh's object is removed, the versions holding it
        # leave the retained window and the plan has read two newer versions, the mesh
        # can be destroyed, and the plan keeps answering.
        T = raw.GeometryType
        readable = raw.Preparation.ON_DEVICE_HOST_READABLE
        fields = (
            raw.BatchResultField.GEOMETRY_IDENTITY
            | raw.BatchResultField.DISTANCE
            | raw.BatchResultField.POSITION
            | raw.BatchResultField.NORMAL
            | raw.BatchResultField.SOURCE_FEATURE
        )
        rays = dict(
            origins=(0.0, 5.0, 0.0, 0.1, 5.0, 0.1, 10.0, 5.0, 0.0, -5.0, 0.0, 0.0),
            directions=(0.0, -1.0, 0.0) * 3 + (1.0, 0.0, 0.0),
            max_distance=60.0,
            bits=(1,) * 4,
            exact=True,
            contacts=True,
        )
        with Fixture(self.library_path) as f:
            f.register_shapes()
            f.create(
                (1, 2),
                slot_capacity=4,
                preparation=readable,
                positions=(0, 0, 0, 10, 0, 0),
                geometry=(Fixture.MESH, 17),
                types=(T.TRIANGLE_MESH, T.BOX),
                bits=(1, 1),
            )
            plan = f.path_plan("ray", precision=raw.QueryPrecision.EXACT, fields=fields)

            def hit_ids():
                return {hit[0] for row in f.path_query("ray", plan, **rays) for hit in row}

            self.assertEqual(hit_ids(), {1, 2})
            removed = f.array((1,), C.c_uint64, (1,), device=True)
            f.publish(1, f.rows(()), base=0, removed=removed, preparation=readable)
            for ordinal in range(2, 5):
                f.move(ordinal, (2,), (10, 0, 0), preparation=readable)
            self.assertEqual(hit_ids(), {2})
            f.move(5, (2,), (10, 0, 0), preparation=readable)
            self.assertEqual(hit_ids(), {2})
            mesh = raw.GeometryRef(T.TRIANGLE_MESH, Fixture.MESH)
            self.assertEqual(
                f.exports["ovspatialquery_destroy_geometry_definition"](f.instance, mesh).status,
                raw.ApiStatus.SUCCESS,
            )
            f.shapes.remove((T.TRIANGLE_MESH, Fixture.MESH))
            f.move(6, (2,), (10, 0, 0), preparation=readable)
            self.assertEqual(hit_ids(), {2})

    def test_slot_capacity_belongs_to_device_prepared_sets(self):
        # A set prepared ON_HOST has no device tables to size, so a capacity is
        # INVALID_ARGUMENT; a device set without one holds exactly its rows and admits no
        # addition.
        with Fixture(self.library_path) as f:
            f.create_ordinary()
            f.publish(
                0,
                f.rows(),
                preparation=raw.Preparation.ON_HOST,
                slot_capacity=2,
                expected=raw.ApiStatus.INVALID_ARGUMENT,
            )
            f.publish(0, f.rows())
            f.publish(
                1,
                f.rows((1001,), positions=(20, 0, 0), geometry=(17,), bits=(1,)),
                base=0,
                expected=raw.ApiStatus.INVALID_ARGUMENT,
            )
            self.assertEqual((f.info().head_ordinal, f.info().object_count), (0, 3))

    def test_bits_update_sets_participation_for_the_version_and_the_ones_built_on_it(self):
        # ADR-0003 decision 4 on the device path: a POSES_AND_BITS update writes the rows'
        # bits into its version, later versions inherit them, retained versions keep theirs,
        # and a complete row may change an object's bits. The column is required with
        # POSES_AND_BITS and refused with POSES.
        with_bits = raw.ExplicitUpdateKind.POSES_AND_BITS
        at_origin = (0, 0, 0)
        with Fixture(self.library_path) as f:
            f.create(pending=2)
            plan = f.plan(capacity=16, slots=4)
            self.assertEqual(
                f.query(plan, centers=at_origin, bits=(2,))[0], (frozenset((IDS[2],)),)
            )
            for rows, kind in (
                (f.rows((IDS[0],), poses=True), with_bits),
                (f.rows((IDS[0],), poses=True, bits=(2,)), raw.ExplicitUpdateKind.POSES),
            ):
                f.publish(1, rows, base=0, kind=kind, expected=raw.ApiStatus.INVALID_ARGUMENT)
            gate = f.gate()
            try:
                moved = f.publish(
                    1,
                    f.rows((IDS[0],), positions=at_origin, poses=True, bits=(2,)),
                    base=0,
                    kind=with_bits,
                    wait=False,
                )
                old = f.query(
                    plan, ordinal=0, admitted=True, centers=at_origin, bits=(2,), wait=False
                )
                new = f.query(
                    plan, ordinal=1, admitted=True, centers=at_origin, bits=(2,), wait=False
                )
            finally:
                gate.set()
            f.wait(moved)
            for launch in (old, new):
                f.wait(launch[0], release=False)
            self.assertEqual(f.read_query(old[0], old[2])[0], (frozenset((IDS[2],)),))
            self.assertEqual(f.read_query(new[0], new[2])[0], (frozenset((IDS[0], IDS[2])),))
            # Bit 1 left the object, and a later pose update keeps the new bits.
            self.assertEqual(
                f.query(plan, centers=at_origin, bits=(1,))[0], (frozenset((IDS[2],)),)
            )
            f.move(2, (IDS[0],), (6, 0, 0))
            self.assertEqual(
                f.query(plan, centers=(6, 0, 0), bits=(2,))[0], (frozenset((IDS[0],)),)
            )
            self.assertEqual(f.query(plan, centers=(6, 0, 0), bits=(1,))[0], (frozenset(),))
            # A complete row changes an object's bits as well (IDS[1] had bit 2 only).
            f.publish(3, f.rows((IDS[1],), positions=(10, 0, 0), bits=(1,)), base=2)
            self.assertEqual(
                f.query(plan, centers=(10, 0, 0), bits=(1,))[0], (frozenset((IDS[1],)),)
            )
            self.assertEqual(
                f.query(plan, ordinal=2, centers=(10, 0, 0), bits=(1,))[0], (frozenset(),)
            )
        # The kind describes the rows, so the host path takes it as well.
        with Fixture(self.library_path) as f:
            f.create(cpu=True)
            f.publish(
                1,
                f.rows((IDS[0],), poses=True, bits=(2,), device=False),
                base=0,
                kind=with_bits,
                cpu=True,
            )
            self.assertEqual(
                f.cpu_query(at_origin, (1,), half_extent=0.05), (frozenset((IDS[2],)),)
            )
            self.assertEqual(
                f.cpu_query(at_origin, (2,), half_extent=0.05), (frozenset((IDS[0], IDS[2])),)
            )

    def test_every_exact_family_prepares_on_the_device_like_the_host(self):
        # ADR-0024 decision 2: spheres, capsules, convex hulls and triangle meshes
        # prepare on the device next to boxes, with the host's world bounds (ADR-0023):
        # a mixed, rotated and scaled population answers exactly as its host-prepared twin,
        # including queries touching the shapes' bounds, after a pose update, through a
        # membership change that adds a mesh (the definition pool grows), and from a host
        # copy of a readable version. A type naming a definition of another family is
        # INVALID_ARGUMENT at admission.
        T = raw.GeometryType
        ids = tuple(5000 + i for i in range(8))
        types = (
            T.SPHERE,
            T.CAPSULE,
            T.CONVEX,
            T.TRIANGLE_MESH,
            T.BOX,
            T.SPHERE,
            T.CONVEX,
            T.CAPSULE,
        )
        geometry = (21, 22, 23, 24, 17, 21, 23, 22)
        positions = tuple(v for i in range(8) for v in (3.0 * i, 0.5 * (i % 2), 0.0))
        angle = 0.7
        rotations = tuple(
            v
            for i in range(8)
            for v in ((0, 0, math.sin(angle / 2), math.cos(angle / 2)) if i % 2 else (0, 0, 0, 1))
        )
        scales = (1.0, 2.0, 0.5, 1.5, 1.0, 1.25, 1.0, 0.75)
        bits = (1,) * 8
        rows = dict(
            positions=positions,
            rotations=rotations,
            scales=scales,
            bits=bits,
            geometry=geometry,
            types=types,
        )
        centers = tuple(
            v
            for i in range(8)
            for center in (
                (3.0 * i, 0.5 * (i % 2), 0.0),
                (3.0 * i + 0.9, 0.5 * (i % 2), 0.0),
                (3.0 * i, 1.4, 0.0),
            )
            for v in center
        )
        query_bits = (1,) * 24
        for half in (0.05, 0.6):
            with self.subTest(half_extent=half), Fixture(self.library_path) as f:
                f.register_shapes()
                f.create(ids, **rows)
                plan = f.plan(
                    capacity=256, slots=1, half_extent=half, queries=24, hits_per_query=16
                )
                actual = f.query(plan, centers=centers, bits=query_bits)[0]
                f.create(ids, cpu=True, **rows)
                reference = f.cpu_query(centers, query_bits, half_extent=half, capacity=256)
                self.assertEqual(actual, reference)
                self.assertTrue(any(actual))
        with Fixture(self.library_path) as f:
            f.register_shapes()
            # Queries touching the sphere's and the capsule's bounds, from both sides.
            for scale, kind, value, reach in (
                (1.25, T.SPHERE, 21, 0.3 * 1.25),
                (2.0, T.CAPSULE, 22, 0.2 * 2.0),
            ):
                f.create_ordinary()
                row = dict(
                    positions=(0, 0, 0),
                    geometry=(value,),
                    types=(kind,),
                    scales=(scale,),
                    bits=(1,),
                )
                f.publish(0, f.rows((7001,), **row))
                half = 0.0625
                touching = reach + half
                probes = (
                    touching,
                    0,
                    0,
                    math.nextafter(touching, math.inf),
                    0,
                    0,
                    math.nextafter(touching, -math.inf),
                    0,
                    0,
                )
                actual = f.query(f.plan(half_extent=half), centers=probes, bits=(1, 1, 1))[0]
                f.create((7001,), cpu=True, **row)
                self.assertEqual(actual, f.cpu_query(probes, (1, 1, 1), half_extent=half))
                self.assertEqual(actual[2], frozenset((7001,)))
        with Fixture(self.library_path) as f:
            f.register_shapes()
            f.create(ids, slot_capacity=10, **rows)
            plan = f.plan(capacity=256, slots=1, half_extent=0.6, queries=24, hits_per_query=16)
            before = f.query(plan, centers=centers, bits=query_bits)[0]
            # Move the capsule, add a second mesh far away, remove the box.
            removed = f.array((ids[4],), C.c_uint64, (1,), device=True)
            f.publish(
                1,
                f.rows(
                    (ids[1], 9000),
                    positions=(3.0, 1.4, 0.0, 50.0, 0.0, 0.0),
                    geometry=(22, 24),
                    types=(T.CAPSULE, T.TRIANGLE_MESH),
                    bits=(1, 1),
                    scales=(2.0, 1.0),
                ),
                base=0,
                removed=removed,
            )
            after = f.query(plan, centers=centers, bits=query_bits)[0]
            self.assertNotIn(ids[4], set().union(*after))
            self.assertIn(ids[1], after[5])
            self.assertEqual(
                f.query(plan, centers=(50.0, 0.0, 0.0), bits=(1,))[0], (frozenset((9000,)),)
            )
            self.assertEqual(f.query(plan, ordinal=0, centers=centers, bits=query_bits)[0], before)
        with Fixture(self.library_path) as f:
            f.register_shapes()
            f.create(ids, preparation=raw.Preparation.ON_DEVICE_HOST_READABLE, **rows)
            readable = f.cpu_query(centers, query_bits, half_extent=0.6, capacity=256)
            f.create(ids, cpu=True, **rows)
            self.assertEqual(
                readable, f.cpu_query(centers, query_bits, half_extent=0.6, capacity=256)
            )

    def test_sphere_and_box_overlaps_read_device_versions_through_the_grid(self):
        # ADR-0024 decision 4: bounds-candidate sphere and oriented-box overlap plans
        # read a device-prepared version through its grid like AABB plans, in MULTIPLE and
        # ANY hit modes, answering as the host-prepared twin does; ANY names the lowest
        # overlapping id whatever the walk order.
        ids, rows = lattice(5, 20_000)
        centers = (0.2, 0.1, 0.0, 1.6, -1.4, 3.1, -3.0, -3.0, 0.0, 0.7, 0.7, 6.0, 40.0, 0.0, 0.0)
        query_bits = (1,) * 5
        yaw = tuple(v for _ in range(5) for v in (0, 0, math.sin(0.45), math.cos(0.45)))
        for shape in (("sphere", 0.9), ("box", (1.2, 0.3, 0.5)), ("sphere", 3.0)):
            with self.subTest(shape=shape), Fixture(self.library_path) as f:
                f.create(ids, **rows)
                turned = yaw if shape[0] == "box" else None
                plan = f.plan(
                    capacity=512, slots=1, half_extent=shape, queries=5, hits_per_query=128
                )
                actual = f.query(plan, centers=centers, bits=query_bits, rotations=turned)[0]
                any_plan = f.plan(
                    capacity=8,
                    slots=1,
                    half_extent=shape,
                    queries=5,
                    hits_per_query=1,
                    hit_mode=raw.HitMode.ANY,
                )
                single = f.query(any_plan, centers=centers, bits=query_bits, rotations=turned)[0]
                f.create(ids, cpu=True, **rows)
                reference = f.cpu_query(
                    centers, query_bits, half_extent=shape, capacity=512, rotations=turned
                )
                self.assertEqual(actual, reference)
                self.assertTrue(any(actual) and actual[4] == frozenset())
                for found, one in zip(reference, single):
                    self.assertEqual(one, frozenset((min(found),)) if found else frozenset())

    def test_rays_and_sweeps_read_device_versions_through_the_grid(self):
        # ADR-0024 decision 4: bounds-candidate rays, sphere sweeps and box sweeps
        # read a device-prepared version through its grid, walking the path one cell at a
        # time and counting each object in the segment its entry falls in, so long rays
        # through many boxes, rays starting inside a box, misses, sweeps whose radius is
        # wider than a cell, and ANY-hit rays answer as the host-prepared twin's CPU plan.
        ids, rows = lattice(5, 30_000)
        origins = (-20.0, 0.1, 0.2, 0.0, 0.0, 0.0, -20.0, -20.0, -20.0, 0.3, -40.0, 50.0)
        s3 = 1 / math.sqrt(3)
        directions = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, s3, s3, s3, 0.0, 1.0, 0.0)
        bits = (1, 1, 1, 1)
        yaw = tuple(v for _ in range(4) for v in (0, 0, math.sin(0.25), math.cos(0.25)))
        cases = (
            ("ray", dict()),
            ("sphere", dict(radius=0.4)),
            ("sphere", dict(radius=3.0)),
            ("box", dict(half_extents=(0.3, 0.2, 0.5))),
        )
        for family, shape in cases:
            with self.subTest(family=family, shape=shape), Fixture(self.library_path) as f:
                f.create(ids, **rows)
                turned = yaw if family == "box" else None
                plan = f.path_plan(family, **shape)
                actual = f.path_query(
                    family,
                    plan,
                    origins=origins,
                    directions=directions,
                    max_distance=60.0,
                    bits=bits,
                    rotations=turned,
                )
                if family == "ray":
                    nearest = f.path_query(
                        "ray",
                        f.path_plan("ray", hit_mode=raw.HitMode.ANY, hits_per_query=1),
                        origins=origins,
                        directions=directions,
                        max_distance=60.0,
                        bits=bits,
                    )
                f.create(ids, cpu=True, **rows)
                reference = f.path_query(
                    family,
                    f.path_plan(family, device=CPU, **shape),
                    origins=origins,
                    directions=directions,
                    max_distance=60.0,
                    bits=bits,
                    device=False,
                    rotations=turned,
                )
                self.assertEqual(actual, reference)
                self.assertTrue(actual[0] and actual[1] and actual[2] and not actual[3])
                if family == "ray":
                    self.assertEqual(
                        nearest, tuple((min(row),) if row else () for row in reference)
                    )

    def test_exact_plans_derive_a_readable_device_version_without_a_host_copy(self):
        # ADR-0024 decision 4: an exact CUDA plan over a readable explicit device version
        # derives the version's exact objects on the device, in stream order after the
        # publication, so the launch builds no host copy (the host usage stays put until a
        # CPU reader asks for one) and its hits equal the host-prepared twin's bit for bit:
        # every family, rotated and scaled, after a POSES update and after a COMPLETE_ROWS
        # update that removes and adds objects. A plain ON_DEVICE version stays refused.
        T = raw.GeometryType
        readable = raw.Preparation.ON_DEVICE_HOST_READABLE
        exact = raw.QueryPrecision.EXACT
        fields = (
            raw.BatchResultField.GEOMETRY_IDENTITY
            | raw.BatchResultField.DISTANCE
            | raw.BatchResultField.POSITION
            | raw.BatchResultField.NORMAL
            | raw.BatchResultField.SOURCE_FEATURE
        )
        ids = tuple(50_000 + i for i in range(10))
        types = (
            T.SPHERE,
            T.CAPSULE,
            T.CONVEX,
            T.TRIANGLE_MESH,
            T.BOX,
            T.SPHERE,
            T.CONVEX,
            T.CAPSULE,
            T.TRIANGLE_MESH,
            T.BOX,
        )
        geometry = (21, 22, 23, 24, 17, 21, 23, 22, 24, 18)

        def unit(axis, angle):
            """A rotation about `axis` near `angle` whose squared length, summed with fused
            multiply-adds, rounds to 1: normalizing it leaves it unchanged, so the twin and
            the version, whose stored rotation a host copy normalizes again, place it alike."""
            length = math.sqrt(sum(v * v for v in axis))
            while True:
                half = math.sin(angle / 2) / length
                q = [v * half for v in axis] + [math.cos(angle / 2)]
                squared = 0.0
                for v in q:
                    squared = float(Fraction(v) * Fraction(v) + Fraction(squared))
                if squared == 1.0:
                    return q
                angle += 1.0e-3

        def poses(step):
            """Positions, rotations about varied axes, and scales of every object."""
            positions, rotations = [], []
            for i in range(10):
                positions += [3.0 * i + 0.2 * step, 0.1 * (i % 3) - 0.1 * step, -0.05 * i]
                rotations += unit((1.0 + i, 2.0 - 0.5 * step, 3.0 - i), 0.3 + 0.4 * i + 0.25 * step)
            scales = tuple(
                v * (1.0 + 0.125 * step)
                for v in (1.0, 2.0, 0.5, 1.5, 1.25, 0.75, 1.1, 0.9, 2.0, 0.6)
            )
            return dict(positions=tuple(positions), rotations=tuple(rotations), scales=scales)

        def rows_of(f, step, cpu, update=False):
            """Complete rows of every object, or with `update` their POSES rows."""
            binding = {} if update else dict(geometry=geometry, types=types, bits=(1,) * 10)
            return f.rows(ids, device=not cpu, poses=update, **binding, **poses(step))

        # Rays down onto each object, one along the row and one oblique; a sweep of each;
        # overlaps at each object's center and beside it.
        origins = tuple(v for i in range(10) for v in (3.0 * i + 0.07, 6.0, 0.03 - 0.05 * i))
        origins += (-5.0, 0.05, -0.1, 2.0, 8.0, 0.0)
        directions = (0.0, -1.0, 0.0) * 10 + (1.0, 0.0, 0.0, 0.6, -0.8, 0.0)
        bits = (1,) * 12
        centers = tuple(v for i in range(10) for v in (3.0 * i, 0.0, -0.05 * i))
        centers += tuple(v for i in range(10) for v in (3.0 * i + 0.8, 0.3, 0.0))
        added = (60_000, 60_001)

        def answers(f, plans, population):
            f.population = population
            ray, sweep, overlap = plans
            path = dict(origins=origins, directions=directions, max_distance=60.0, bits=bits)
            return (
                f.path_query("ray", ray, exact=True, contacts=True, **path),
                f.path_query("sphere", sweep, exact=True, contacts=True, **path),
                f.query(overlap, centers=centers, bits=(1,) * 20, exact=True)[0],
            )

        with Fixture(self.library_path) as f:
            f.register_shapes()
            twins = {}
            for cpu in (False, True):
                f.create_ordinary(pending=2)
                # Slot capacity belongs to device-prepared sets.
                capacity = 0 if cpu else 12
                f.publish(
                    0, rows_of(f, 0, cpu), cpu=cpu, preparation=readable, slot_capacity=capacity
                )
                plans = (
                    f.path_plan("ray", precision=exact, queries=12, fields=fields),
                    f.path_plan("sphere", radius=0.2, precision=exact, queries=12, fields=fields),
                    f.plan(
                        capacity=256,
                        half_extent=("sphere", 0.4),
                        precision=exact,
                        queries=20,
                        hits_per_query=8,
                    ),
                )
                twins[cpu] = (plans, C.c_uint64(f.population.value))
            device, host = twins[False], twins[True]
            used = f.usage(CPU)
            actual = answers(f, *device)
            self.assertEqual(f.usage(CPU), used)
            self.assertEqual(actual, answers(f, *host))
            self.assertTrue(all(actual[0][:10]) and all(actual[1][:10]) and any(actual[2]))
            # A CPU reader still gets the copy, and is charged for it.
            f.population = device[1]
            copied = f.cpu_query(centers, (1,) * 20, half_extent=("sphere", 0.4), capacity=16)
            self.assertGreater(f.usage(CPU), used)
            f.population = host[1]
            self.assertEqual(
                copied, f.cpu_query(centers, (1,) * 20, half_extent=("sphere", 0.4), capacity=16)
            )
            # A POSES update moves, turns and scales every object.
            for cpu, (_, population) in ((False, device), (True, host)):
                f.population = population
                f.publish(
                    1,
                    rows_of(f, 1, cpu, update=True),
                    base=0,
                    kind=raw.ExplicitUpdateKind.POSES,
                    cpu=cpu,
                    preparation=readable,
                )
            moved = answers(f, *device)
            self.assertEqual(moved, answers(f, *host))
            self.assertNotEqual(moved, actual)
            # Remove the first mesh and box, add a convex and a mesh under the first rays.
            for cpu, (_, population) in ((False, device), (True, host)):
                f.population = population
                f.publish(
                    2,
                    f.rows(
                        added,
                        positions=(9.1, 0.0, 0.0, 12.1, 0.2, 0.0),
                        rotations=(0.0, 0.0, math.sin(0.2), math.cos(0.2), 0.0, 0.0, 0.0, 1.0),
                        scales=(1.5, 2.5),
                        geometry=(23, 24),
                        types=(T.CONVEX, T.TRIANGLE_MESH),
                        bits=(1, 1),
                        device=not cpu,
                    ),
                    base=1,
                    removed=f.array((ids[3], ids[4]), C.c_uint64, (2,), device=not cpu),
                    cpu=cpu,
                    preparation=readable,
                )
            changed = answers(f, *device)
            self.assertEqual(changed, answers(f, *host))
            found = {hit[0] for row in changed[0] for hit in row}
            self.assertTrue(set(added) <= found and not {ids[3], ids[4]} & found)
        with Fixture(self.library_path) as f:
            f.create(ids[:2], geometry=(17, 18), bits=(1, 1))
            f.path_query(
                "ray",
                f.path_plan("ray", precision=exact),
                origins=origins[:12],
                directions=directions[:12],
                max_distance=60.0,
                bits=(1,) * 4,
                expected=raw.ApiStatus.NOT_SUPPORTED,
            )

    def test_box_rows_take_float64_rotations_and_their_own_half_extents(self):
        # ADR-0026 and ADR-0025 amended: a BOX plan's rows may carry float64 rotations and
        # their own half extents; a CUDA plan over a device-prepared version answers as the
        # same plan over the host-prepared twin, and as the float32 rotations of the same
        # rows away from touching boundaries.
        yaws = (0.3, 1.1, 2.5)
        rotations = tuple(
            v for yaw in yaws for v in (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        )
        centers = (0, 0, 0, 10, 0, 0, 0, 0, 0)
        extents = (0.2, 0.2, 0.2, 3.0, 0.5, 0.5, 11.0, 0.5, 0.5)
        bits = (1, 2, 3)

        def answers(f):
            plan = f.plan(
                capacity=9, queries=3, hits_per_query=3, half_extent=("box", (0.5, 0.5, 0.5))
            )
            return (
                f.query(
                    plan,
                    centers=centers,
                    bits=bits,
                    rotations=rotations,
                    rotation_element=C.c_double,
                    half_extents=extents,
                )[0],
                f.query(
                    plan, centers=centers, bits=bits, rotations=rotations, half_extents=extents
                )[0],
            )

        with Fixture(self.library_path) as f:
            f.create()
            device = answers(f)
        with Fixture(self.library_path) as f:
            f.create(cpu=True)
            host = answers(f)
        self.assertEqual(device, host)
        self.assertEqual(device[0], device[1])
        # Rows 0 and 2 sit on the two objects at the origin; row 1 reaches only the far one.
        self.assertEqual(
            device[0],
            (frozenset((IDS[0], IDS[2])), frozenset((IDS[1],)), frozenset((IDS[0], IDS[2]))),
        )

    def test_cuda_reporter_patches_from_the_device_version_change_list(self):
        # ADR-0019 decision 3 on the device path: a POSES update's kernel lists the slots it
        # rewrote, and a CUDA reporter advance over the device-prepared version patches its
        # pair set from that list when the rows are at most a quarter of the objects (one
        # of six here), recomputing otherwise (two of six); both report the host twin's
        # deltas across moves that make and break pairs, including a chain of two versions,
        # and across a removal. A CPU reporter over the device population is NOT_SUPPORTED.
        ids, rows = PAIR_IDS, PAIR_ROWS
        moves = (
            ((ids[4],), (20.5, 0, 0)),  # one row: patched; (4,5) enters
            ((ids[2], ids[0]), (40, 0, 0, 100, 0, 0)),  # two rows: recompute; (2,3) and (0,1) exit
            ((ids[1],), (99.3, 0, 0)),  # one row: patched; (0,1) enters again
            ((ids[3],), (41, 0, 0)),  # one row, read as a chain of two below: (2,3) enters
        )
        expected = (
            (frozenset(((ids[0], ids[1]), (ids[2], ids[3]))), frozenset()),
            (frozenset(((ids[4], ids[5]),)), frozenset()),
            (frozenset(), frozenset(((ids[0], ids[1]), (ids[2], ids[3])))),
            (frozenset(((ids[0], ids[1]), (ids[2], ids[3]))), frozenset()),
            (frozenset(), frozenset(((ids[0], ids[1]),))),  # removing ids[0]
        )

        def history(f, cpu):
            reporter = f.pair_reporter(capacity=16)
            results = [f.advance(reporter, 0)]
            for ordinal, (moved, moved_to) in enumerate(moves, start=1):
                f.move(ordinal, moved, moved_to, cpu=cpu)
                if ordinal != 3:
                    # Ordinal 3 is skipped, so the last advance patches through two records.
                    results.append(f.advance(reporter, ordinal))
            removed = f.array((ids[0],), C.c_uint64, (1,), device=not cpu)
            f.publish(5, f.rows((), positions=(), device=not cpu), base=4, removed=removed, cpu=cpu)
            results.append(f.advance(reporter, 5))
            return tuple(results)

        with Fixture(self.library_path) as f:
            f.create(ids, slot_capacity=8, **rows)
            actual = history(f, False)
            f.advance(f.pair_reporter(device=CPU), 5, expected=raw.ApiStatus.NOT_SUPPORTED)
        with Fixture(self.library_path) as f:
            f.create(ids, cpu=True, **rows)
            reference = history(f, True)
        self.assertEqual(actual, reference)
        self.assertEqual(actual, expected)

    def test_cuda_reporter_reservations_describe_the_device_result(self):
        # ADR-0022 decisions 1, 2 and 6: a CUDA reporter's advance runs on the
        # caller's stream after its wait event, and its reservation names capacity-shaped
        # device tensors for the entered and exited rows and a counts tensor that hold,
        # once result_ready fires, exactly what fetch later describes. A CPU reporter's
        # reservation leaves the descriptors absent and rejects a stream or ADMITTED_EXACT.
        ids, rows = PAIR_IDS, PAIR_ROWS
        expected_pairs = frozenset(((ids[0], ids[1]), (ids[2], ids[3])))
        with Fixture(self.library_path) as f:
            f.create(ids, slot_capacity=8, **rows)
            reporter = f.pair_reporter(capacity=16)
            pairs, reservation, counts = f.advance(reporter, 0, stream=True, details=True)
            self.assertEqual(pairs, (expected_pairs, frozenset()))
            self.assertEqual(counts, (raw.PairReporterOutcome.ADVANCED, 2, 0, 2))
            self.assertEqual(reservation.result_device.device_type, raw.K_DL_CUDA)
            for tensor in (reservation.entered_pairs, reservation.exited_pairs):
                self.assertEqual(
                    (tensor.ndim, tuple(tensor.shape[:2]), tuple(tensor.strides[:2])),
                    (2, (16, 2), (2, 1)),
                )
                self.assertEqual((tensor.dtype.code, tensor.dtype.bits), (raw.K_DL_UINT, 64))
            self.assertEqual(
                (
                    reservation.counts.ndim,
                    reservation.counts.shape[0],
                    reservation.counts.strides[0],
                ),
                (1, 4, 1),
            )
            # Without a stream the library's own stream serves, and the consumer waits on
            # result_ready as before.
            f.move(1, (ids[4],), (20.5, 0, 0))
            pairs, _, counts = f.advance(reporter, 1, details=True)
            self.assertEqual(pairs, (frozenset(((ids[4], ids[5]),)), frozenset()))
            self.assertEqual(counts, (raw.PairReporterOutcome.ADVANCED, 1, 0, 3))
        with Fixture(self.library_path) as f:
            f.create(ids, cpu=True, **rows)
            cpu = f.pair_reporter(device=CPU)
            f.advance(cpu, 0, stream=True, expected=raw.ApiStatus.INVALID_ARGUMENT)
            f.advance(cpu, 0, admitted=True, expected=raw.ApiStatus.INVALID_ARGUMENT)
            pairs, reservation, counts = f.advance(cpu, 0, details=True)
            self.assertEqual((pairs[0], counts), (expected_pairs, None))
            self.assertEqual(reservation.result_device.device_type, raw.K_DL_CPU)
            for tensor in (reservation.entered_pairs, reservation.exited_pairs, reservation.counts):
                self.assertFalse(tensor.shape)
                self.assertFalse(tensor.data)

    def test_cuda_reporter_advances_over_admitted_publications_and_fails_with_a_failed_one(self):
        # ADR-0022 decision 3 and ADR-0021 for reporters: a CUDA reporter
        # advance selects a pending device-prepared publication with ADMITTED_EXACT and is
        # ordered after it on the device; when the selected publication fails, the advance
        # completes with OPERATION_FAILED, the device counts carry the FAILED outcome with
        # zero deltas and the unchanged accepted count, and the baseline is preserved for
        # the next advance.
        ids, rows = PAIR_IDS, PAIR_ROWS
        with Fixture(self.library_path) as f:
            f.create(ids, slot_capacity=8, pending=2, **rows)
            reporter = f.pair_reporter(capacity=16)
            self.assertEqual(
                f.advance(reporter, 0, stream=True),
                (frozenset(((ids[0], ids[1]), (ids[2], ids[3]))), frozenset()),
            )
            gate = f.gate()
            try:
                publication = f.move(1, (ids[4], ids[2]), (20.5, 0, 0, 40, 0, 0), wait=False)
                pending = f.advance(reporter, 1, admitted=True, stream=True, wait=False)
            finally:
                gate.set()
            f.wait(publication)
            self.assertEqual(
                f.finish_advance(pending),
                (frozenset(((ids[4], ids[5]),)), frozenset(((ids[2], ids[3]),))),
            )
            gate = f.gate()
            try:
                bad = f.move(2, (ids[0],), (math.nan, 0, 0), wait=False)
                failed = f.advance(reporter, 2, admitted=True, stream=True, wait=False)
            finally:
                gate.set()
            f.wait(bad, terminal=raw.ApiStatus.OPERATION_FAILED)
            self.assertEqual(
                f.finish_advance(failed, terminal=raw.ApiStatus.OPERATION_FAILED),
                (raw.PairReporterOutcome.FAILED, 0, 0, 2),
            )
            self.assertEqual((f.info().head_ordinal, f.info().update_pending), (1, 0))
            # The baseline is still the one accepted at ordinal 1.
            f.move(3, (ids[4],), (60, 0, 0), base=1)
            pairs, _, counts = f.advance(reporter, 3, stream=True, details=True)
            self.assertEqual(pairs, (frozenset(), frozenset(((ids[4], ids[5]),))))
            self.assertEqual(counts, (raw.PairReporterOutcome.ADVANCED, 0, 1, 1))

    def test_cuda_reporter_pipelines_pending_advances_behind_pending_publications(self):
        # ADR-0022 decision 5: a CUDA reporter with three result slots admits
        # three advances at once, each selecting a pending chained publication with
        # ADMITTED_EXACT; a fourth is RESULT_CAPACITY, and stale and backward ordinals are
        # judged against the newest admitted endpoints. The advances compare on the
        # device in admission order: the second exceeds the delta capacity, so the third
        # compares with the baseline the first accepted, as the host twin does when it
        # advances one at a time; a fourth publication is reported once a slot is free.
        ids, rows = PAIR_IDS, PAIR_ROWS
        moves = (
            ((ids[4], ids[2]), (20.5, 0, 0, 40, 0, 0)),  # (4,5) enters, (2,3) exits
            (
                (ids[0], ids[3]),
                (100, 0, 0, 21, 0, 0),
            ),  # (0,1) exits, (3,4) and (3,5) enter: 3 deltas
            ((ids[3],), (6.3, 0, 0)),  # against the baseline after 1: (0,1) exits
            ((ids[5],), (200, 0, 0)),  # (4,5) exits
        )
        expected = (
            (frozenset(((ids[0], ids[1]), (ids[2], ids[3]))), frozenset()),
            (frozenset(((ids[4], ids[5]),)), frozenset(((ids[2], ids[3]),))),
            raw.PairReporterOutcome.DELTA_CAPACITY,
            (frozenset(), frozenset(((ids[0], ids[1]),))),
            (frozenset(), frozenset(((ids[4], ids[5]),))),
        )
        with Fixture(self.library_path) as f:
            f.create(ids, slot_capacity=8, pending=4, **rows)
            reporter = f.pair_reporter(capacity=16, delta_capacity=2, slots=3)
            self.assertEqual(f.advance(reporter, 0, stream=True), expected[0])
            gate = f.gate()
            try:
                publications = [
                    f.move(ordinal, moved, moved_to, wait=False)
                    for ordinal, (moved, moved_to) in enumerate(moves, start=1)
                ]
                pending = [
                    f.advance(reporter, ordinal, admitted=True, stream=True, wait=False)
                    for ordinal in (1, 2, 3)
                ]
                f.advance(reporter, 4, admitted=True, expected=raw.ApiStatus.RESULT_CAPACITY)
                f.advance(reporter, 3, admitted=True, expected=raw.ApiStatus.STALE_ORDINAL)
                f.advance(reporter, 2, admitted=True, expected=raw.ApiStatus.BACKWARD_ORDINAL)
            finally:
                gate.set()
            for publication in publications:
                f.wait(publication)
            self.assertEqual(f.finish_advance(pending[0]), expected[1])
            pairs, _, counts = f.finish_advance(pending[1], details=True, outcome=expected[2])
            self.assertEqual((pairs, counts), ((frozenset(), frozenset()), (expected[2], 0, 0, 2)))
            pairs, _, counts = f.finish_advance(pending[2], details=True)
            self.assertEqual(
                (pairs, counts), (expected[3], (raw.PairReporterOutcome.ADVANCED, 0, 1, 1))
            )
            # With a slot free again the fourth publication is reported, and with nothing
            # pending stale is judged against the accepted endpoints.
            pairs, _, counts = f.advance(reporter, 4, stream=True, details=True)
            self.assertEqual(
                (pairs, counts), (expected[4], (raw.PairReporterOutcome.ADVANCED, 0, 1, 0))
            )
            f.advance(reporter, 4, expected=raw.ApiStatus.STALE_ORDINAL)
        with Fixture(self.library_path) as f:
            f.create(ids, cpu=True, **rows)
            reporter = f.pair_reporter(device=CPU, capacity=16, delta_capacity=2)
            self.assertEqual(f.advance(reporter, 0), expected[0])
            for ordinal, (moved, moved_to) in enumerate(moves, start=1):
                f.move(ordinal, moved, moved_to, cpu=True)
            self.assertEqual(f.advance(reporter, 1), expected[1])
            self.assertEqual(
                f.finish_advance(f.advance(reporter, 2, wait=False), outcome=expected[2]),
                (frozenset(), frozenset()),
            )
            self.assertEqual(f.advance(reporter, 3), expected[3])
            self.assertEqual(f.advance(reporter, 4), expected[4])

    def test_host_readable_device_versions_serve_cpu_readers_once_copied(self):
        # ADR-0024 decision 4: a publication prepared ON_DEVICE_HOST_READABLE serves
        # synchronous queries and CPU plans from a host copy built on first read and
        # charged to the host budget; plain ON_DEVICE versions of the same population
        # stay refused, a retained readable version answers with its own objects, and
        # a host budget too small for the copy refuses the reader with HOST_BUDGET.
        readable = raw.Preparation.ON_DEVICE_HOST_READABLE
        with Fixture(self.library_path) as f:
            f.create(preparation=readable)
            plan = f.plan()
            expected = (frozenset((IDS[0], IDS[2])), frozenset((IDS[1],)), frozenset((IDS[2],)))
            self.assertEqual(f.query(plan)[0], expected)
            centers, bits = (0, 0, 0, 10, 0, 0, 0, 0, 0), (1, 2, 2)
            self.assertEqual(f.cpu_query(centers, bits, half_extent=0.05), expected)
            # A pair snapshot reads the copied objects' participation bits.
            self.assertEqual(f.cpu_pairs(2, 1), frozenset(((IDS[2], IDS[0]),)))
            cpu_plan = f.plan(device=CPU)
            self.assertEqual(f.query(cpu_plan, device=False)[0], expected)
            # A plain ON_DEVICE pose update is not readable; the readable base still is.
            f.move(1, (IDS[0],), (6, 0, 0))
            f.cpu_query(centers, bits, half_extent=0.05, expected=raw.ApiStatus.NOT_SUPPORTED)
            f.query(cpu_plan, device=False, expected=raw.ApiStatus.NOT_SUPPORTED)
            retained = raw.OrdinalSelection(raw.OrdinalSelectionKind.RETAINED_EXACT, 0)
            self.assertEqual(
                f.cpu_query(centers, bits, half_extent=0.05, selection=retained), expected
            )
            # A readable pose update is read with its moved object.
            f.move(2, (IDS[0],), (6, 0, 0), preparation=readable)
            moved = (frozenset((IDS[0],)), frozenset((IDS[1],)), frozenset((IDS[2],)))
            self.assertEqual(
                f.cpu_query((6, 0, 0, 10, 0, 0, 0, 0, 0), bits, half_extent=0.05), moved
            )
            self.assertEqual(
                f.query(plan, centers=(6, 0, 0, 10, 0, 0, 0, 0, 0), bits=bits)[0], moved
            )
        # The copy is charged at the first read: a budget that holds the instance and the
        # publication but not the copy of 2,000 boxes refuses the CPU reader, and the CUDA
        # plan keeps answering.
        many = tuple(10_000 + i for i in range(2000))
        rows = dict(
            positions=tuple(v for i in range(2000) for v in (2.0 * (i % 50), 2.0 * (i // 50), 0.0)),
            geometry=(17,) * 2000,
            bits=(1,) * 2000,
        )
        with Fixture(self.library_path) as f:
            f.create(many, preparation=readable, **rows)
            used = f.usage(CPU)
        with Fixture(self.library_path, host_budget=used + (64 << 10)) as f:
            f.create(many, preparation=readable, **rows)
            f.cpu_query((0, 0, 0), (1,), half_extent=0.05, expected=raw.ApiStatus.HOST_BUDGET)
            self.assertEqual(
                f.query(f.plan(capacity=16), centers=(0, 0, 0), bits=(1,))[0],
                (frozenset((many[0],)),),
            )

    def test_a_pending_device_publication_does_not_block_host_work(self):
        # The fixture's instance has one worker. A device publication whose producer has
        # not finished must not hold that worker in a wait: the worker requeues it and serves
        # other operations, so a host-prepared publication of another population completes
        # while the gate is still closed.
        with Fixture(self.library_path) as f:
            f.create(pending=2)
            bodies = C.c_uint64(f.population.value)
            f.create_ordinary()
            other = C.c_uint64(f.population.value)
            gate = f.gate()
            try:
                f.population = bodies
                pending = f.move(1, (IDS[0],), (6, 0, 0), wait=False)
                f.population = other
                started = time.monotonic()
                f.publish(0, f.rows(device=False), cpu=True)
                elapsed = time.monotonic() - started
                self.assertLess(elapsed, 2.0)
                self.assertEqual((f.info().ready, f.info().head_ordinal), (1, 0))
                f.population = bodies
                self.assertEqual((f.info().head_ordinal, f.info().update_pending), (0, 1))
            finally:
                gate.set()
            f.wait(pending)
            self.assertEqual(f.info().head_ordinal, 1)
            self.assertEqual(
                f.query(f.plan(), centers=(6, 0, 0), bits=(1,))[0], (frozenset((IDS[0],)),)
            )

    def test_destroy_calls_report_busy_while_a_gated_chain_and_its_consumer_are_pending(self):
        # Teardown around pending device work: the population and the instance stay BUSY
        # while a chained publication and the launch that selected it are pending; the plan
        # stays BUSY after the launch's retirement until its consumer event fires, and the
        # reaper then frees the slot without a further call on the operation.
        with Fixture(self.library_path) as f:
            f.create(pending=2)
            population = C.c_uint64(f.population.value)
            plan = f.plan(capacity=16)
            consumer_done = C.c_void_p()
            f.cuda.call("cuEventCreate", C.byref(consumer_done), 2)
            gate = f.gate()
            try:
                first = f.move(1, (IDS[0],), (6, 0, 0), wait=False)
                second = f.move(2, (IDS[1],), (6, 0, 0), wait=False)
                launch = f.query(
                    plan, ordinal=2, admitted=True, centers=(6, 0, 0), bits=(3,), wait=False
                )
                # The consumer of the launch's result is queued behind the gate too.
                f.cuda.call("cuEventRecord", consumer_done, f.cuda.stream)
                for name, handle in (
                    ("ovspatialquery_destroy_population", population),
                    ("ovspatialquery_destroy_batch_plan", plan),
                ):
                    f.call(name, f.instance, handle, expected=raw.ApiStatus.BUSY)
                f.call("ovspatialquery_destroy_instance", f.instance, expected=raw.ApiStatus.BUSY)
                self.assertEqual(f.cuda.driver.cuEventQuery(consumer_done), 600)
            finally:
                gate.set()
            f.wait(first)
            f.wait(second)
            f.wait(launch[0], release=False)
            self.assertEqual(
                f.read_query(launch[0], launch[2], release=False)[0], (frozenset((IDS[0], IDS[1])),)
            )
            # Retire the launch behind its consumer event; the plan's slot is released when the
            # event fires, which the destroy call reports as BUSY until then.
            f.cuda.call("cuEventRecord", consumer_done, f.cuda.stream)
            event = raw.CudaEvent(consumer_done.value)
            f.call(
                "ovspatialquery_destroy_op_after_cuda",
                f.instance,
                C.byref(launch[0]),
                C.byref(event),
            )
            self.assertEqual((launch[0].value, event.value), (0, 0))
            f.operations.remove(launch[0])
            f.query_operations.discard(launch[0].value)
            f.destroy("ovspatialquery_destroy_batch_plan", f.instance, plan)
            f.plans.remove(plan)
            f.call("ovspatialquery_destroy_population", f.instance, population)
            f.populations.remove(next(p for p in f.populations if p.value == population.value))

    def test_result_capacity_with_publications_pending_and_republishing_while_a_failed_chain_drains(
        self,
    ):
        # Back-pressure on the device path: with every result slot owned by a launch that
        # waits for a pending publication, the next launch is RESULT_CAPACITY until the first
        # is retired. When a chain fails, publishing from the head again is admitted as soon as
        # the failed ordinals have left the pending window, without a host wait on them.
        with Fixture(self.library_path) as f:
            f.create(pending=2)
            plan = f.plan(capacity=16, slots=1)
            gate = f.gate()
            try:
                first = f.move(1, (IDS[0],), (6, 0, 0), wait=False)
                launch = f.query(
                    plan, ordinal=1, admitted=True, centers=(6, 0, 0), bits=(3,), wait=False
                )
                f.query(
                    plan,
                    ordinal=1,
                    admitted=True,
                    centers=(6, 0, 0),
                    bits=(3,),
                    expected=raw.ApiStatus.RESULT_CAPACITY,
                )
                f.query(plan, centers=(6, 0, 0), bits=(3,), expected=raw.ApiStatus.RESULT_CAPACITY)
            finally:
                gate.set()
            f.wait(first)
            f.wait(launch[0], release=False)
            self.assertEqual(f.read_query(launch[0], launch[2])[0], (frozenset((IDS[0],)),))
            self.assertEqual(
                f.query(plan, centers=(6, 0, 0), bits=(3,))[0], (frozenset((IDS[0],)),)
            )
            gate = f.gate()
            try:
                bad = f.move(2, (IDS[1],), (math.nan, 0, 0), wait=False)
                chained = f.move(3, (IDS[2],), (6, 0, 0), wait=False)
                # Pending ordinals 2 and 3 are the newest admitted; the head is not a base yet.
                f.move(4, (IDS[2],), (20, 0, 0), base=1, expected=raw.ApiStatus.BUSY)
            finally:
                gate.set()
            deadline = time.monotonic() + 10
            while True:
                recovered = f.move(4, (IDS[2],), (20, 0, 0), base=1, expected=None, wait=False)
                if recovered is not None or time.monotonic() >= deadline:
                    break
                time.sleep(0.001)
            self.assertIsNotNone(recovered)
            f.wait(recovered)
            self.assertEqual((f.info().head_ordinal, f.info().update_pending), (4, 0))
            f.wait(bad, terminal=raw.ApiStatus.OPERATION_FAILED)
            f.wait(chained, terminal=raw.ApiStatus.OPERATION_FAILED)
            self.assertEqual(
                f.query(plan, centers=(20, 0, 0), bits=(3,))[0], (frozenset((IDS[2],)),)
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    options, unittest_args = parser.parse_known_args()
    DevicePublicationInstalledTests.library_path = options.library.resolve()
    unittest.main(argv=[__file__, *unittest_args])


if __name__ == "__main__":
    main()
