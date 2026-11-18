# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise installed-wheel CPU queries and optional real CUDA execution."""

from __future__ import annotations

import ctypes
import argparse
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import ovspatialquery
from ovspatialquery import (
    ApiError,
    Device,
    DeviceBudget,
    InstanceConfig,
    Library,
    OverlapQuery,
    RaycastPlanConfig,
    RaycastQuery,
    Sphere,
    SphereSweepQuery,
    raw,
)
from ovspatialquery.explicit import GeometryDefinitionRef


class _ManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", raw.DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class Column:
    """A minimal DLPack producer over ctypes storage; the isolated wheel venv has no NumPy."""

    def __init__(self, ctype, values, shape, code):
        self.storage = (ctype * len(values))(*values)
        self.dimensions = (ctypes.c_int64 * len(shape))(*shape)
        self.managed = _ManagedTensor(
            raw.DLTensor(
                data=ctypes.cast(self.storage, ctypes.c_void_p),
                device=Device.cpu().to_raw(),
                ndim=len(shape),
                dtype=raw.DLDataType(code, ctypes.sizeof(ctype) * 8, 1),
                shape=self.dimensions,
            )
        )

    def move_to(self, device, address):
        self.managed.dl_tensor.device = device.to_raw()
        self.managed.dl_tensor.data = address

    def __dlpack_device__(self):
        device = self.managed.dl_tensor.device
        return (device.device_type, device.device_id)

    def __dlpack__(self, stream=None):
        capsule = ctypes.pythonapi.PyCapsule_New
        capsule.restype = ctypes.py_object
        capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        return capsule(ctypes.addressof(self.managed), b"dltensor", None)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_matching_versions(package_version, native_version):
    """The native getter reports the release part of a ``major.minor.patch[.devN|rcN][+local]`` wheel."""
    public = package_version.split("+", 1)[0]
    release = tuple(int(part) for part in public.split("rc", 1)[0].split(".dev", 1)[0].split("."))
    require(
        tuple(native_version) == release,
        f"native and wheel versions disagree: native {native_version}, wheel {package_version}",
    )


def cuda_batch(instance, target):
    """Upload real inputs and compare installed CUDA results with the CPU smoke case."""
    driver = (
        ctypes.CDLL("nvcuda.dll", winmode=0x800) if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    )
    pointer = ctypes.c_void_p
    u64 = ctypes.c_uint64
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDevicePrimaryCtxRetain": [ctypes.POINTER(pointer), ctypes.c_int],
        "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
        "cuCtxGetCurrent": [ctypes.POINTER(pointer)],
        "cuCtxSetCurrent": [pointer],
        "cuStreamCreate": [ctypes.POINTER(pointer), ctypes.c_uint],
        "cuStreamDestroy_v2": [pointer],
        "cuStreamSynchronize": [pointer],
        "cuMemAlloc_v2": [ctypes.POINTER(u64), ctypes.c_size_t],
        "cuMemFree_v2": [u64],
        "cuMemcpyHtoD_v2": [u64, pointer, ctypes.c_size_t],
        "cuMemcpyDtoH_v2": [pointer, u64, ctypes.c_size_t],
        "cuEventCreate": [ctypes.POINTER(pointer), ctypes.c_uint],
        "cuEventRecord": [pointer, pointer],
        "cuEventSynchronize": [pointer],
    }
    for name, arguments in signatures.items():
        function = getattr(driver, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int

    def call(name, *arguments):
        status = getattr(driver, name)(*arguments)
        require(status == 0, f"{name} failed with CUDA status {status}")

    call("cuInit", 0)
    previous, primary, stream = pointer(), pointer(), pointer()
    call("cuCtxGetCurrent", ctypes.byref(previous))
    call("cuDevicePrimaryCtxRetain", ctypes.byref(primary), 0)
    allocations = []
    try:
        call("cuCtxSetCurrent", primary)
        call("cuStreamCreate", ctypes.byref(stream), 1)
        inputs = (
            Column(ctypes.c_double, [-3, 0, 0, -3, 4, 0], [2, 3], raw.K_DL_FLOAT),
            Column(ctypes.c_float, [1, 0, 0, 1, 0, 0], [2, 3], raw.K_DL_FLOAT),
            Column(ctypes.c_double, [10, 10], [2], raw.K_DL_FLOAT),
        )
        for column in inputs:
            storage = column.storage
            allocation = u64()
            call("cuMemAlloc_v2", ctypes.byref(allocation), ctypes.sizeof(storage))
            allocations.append(allocation)
            call(
                "cuMemcpyHtoD_v2", allocation, ctypes.cast(storage, pointer), ctypes.sizeof(storage)
            )
            # A pageable copy may return before its DMA lands, and nothing orders the legacy
            # stream before the non-blocking one the library reads on: wait for it here.
            call("cuStreamSynchronize", None)
            column.move_to(Device.cuda(), allocation.value)
        plan = instance.create_raycast_plan(
            RaycastPlanConfig(
                target.reference,
                raw.HitMode.CLOSEST,
                2,
                1,
                2,
                precision=raw.QueryPrecision.EXACT,
                device=Device(raw.K_DL_CUDA, 0),
                result_fields=raw.BatchResultField.DISTANCE,
            )
        )
        try:
            operation, _ = plan.enqueue(
                origins=inputs[0],
                unit_directions=inputs[1],
                max_distances=inputs[2],
                cuda_stream=stream.value,
            )
            try:
                require(operation.wait() == raw.OpState.COMPLETED, "CUDA batch failed")
                result = operation.fetch_raycast_batch()
                require(
                    result.common.execution_device == Device(raw.K_DL_CUDA, 0),
                    "CUDA batch fell back to CPU",
                )
                call("cuEventSynchronize", result.common.result_ready)

                def download(column, ctype, count):
                    require(
                        column.device == Device(raw.K_DL_CUDA, 0),
                        "CUDA result is not on the device",
                    )
                    values = (ctype * count)()
                    call(
                        "cuMemcpyDtoH_v2",
                        ctypes.cast(values, pointer),
                        column.data + column.byte_offset,
                        ctypes.sizeof(values),
                    )
                    return tuple(values)

                require(
                    download(result.hit_ranges.hit_counts, u64, 2) == (1, 0),
                    "CUDA/CPU count parity failed",
                )
                require(
                    download(result.object_ids, u64, 1) == (11,), "CUDA/CPU identity parity failed"
                )
                require(
                    math.isclose(
                        download(result.exact.distances, ctypes.c_double, 1)[0], 2, abs_tol=1e-6
                    ),
                    "CUDA/CPU distance parity failed",
                )
            finally:
                # The library owns this event after close_after_cuda accepts it.
                done = pointer()
                call("cuEventCreate", ctypes.byref(done), 2)
                call("cuEventRecord", done, stream)
                call("cuEventSynchronize", done)
                operation.close_after_cuda(done.value)
        finally:
            deadline = time.monotonic() + 10
            while True:
                try:
                    plan.close()
                    break
                except ApiError as error:
                    if error.status != raw.ApiStatus.BUSY or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.001)
    finally:
        for allocation in allocations:
            call("cuMemFree_v2", allocation)
        if stream.value:
            call("cuStreamDestroy_v2", stream)
        call("cuCtxSetCurrent", previous)
        call("cuDevicePrimaryCtxRelease_v2", 0)


def run(require_cuda=False, require_unavailable_cuda=False):
    config = InstanceConfig(
        device_budgets=(DeviceBudget(Device(raw.K_DL_CUDA, 0), 64 << 20),) if require_cuda else ()
    )
    with Library.open() as library, library.instance(config) as instance:
        package_version = importlib.metadata.version("ovspatialquery")
        require_matching_versions(package_version, library.version)
        adapter = instance.explicit()
        registration = adapter.register_spheres(
            definition_values=Column(ctypes.c_uint64, [7], [1], raw.K_DL_UINT),
            radii=Column(ctypes.c_double, [1], [1], raw.K_DL_FLOAT),
        )
        try:
            require(registration.wait() == raw.OpState.COMPLETED, "sphere registration failed")
        finally:
            registration.close()
        population = adapter.create_population()
        try:
            publication = population.enqueue_set(
                0,
                object_values=Column(ctypes.c_uint64, [11], [1], raw.K_DL_UINT),
                geometry_types=Column(ctypes.c_int32, [raw.GeometryType.SPHERE], [1], raw.K_DL_INT),
                geometry_values=Column(ctypes.c_uint64, [7], [1], raw.K_DL_UINT),
                translations=Column(ctypes.c_double, [0, 0, 0], [1, 3], raw.K_DL_FLOAT),
                rotations_xyzw=Column(ctypes.c_float, [0, 0, 0, 1], [1, 4], raw.K_DL_FLOAT),
            )
            try:
                publication.wait()
                require(
                    publication.fetch_population_update().object_count == 1, "publication failed"
                )
            finally:
                publication.close()
            target = population.target
            precision = raw.QueryPrecision.EXACT
            ray = target.raycast_closest(
                RaycastQuery((-3, 0, 0), (1, 0, 0), 10, precision=precision)
            )
            require(ray.hit is not None and ray.hit.object_id == 11, "raycast returned no sphere")
            require(math.isclose(ray.hit.distance, 2, abs_tol=1e-6), "wrong raycast distance")
            miss = target.raycast_closest(
                RaycastQuery((-3, 4, 0), (1, 0, 0), 10, precision=precision)
            )
            require(miss.hit is None, "raycast miss returned a hit")
            overlap = target.overlap_any(OverlapQuery(Sphere((0, 0, 0), 0.25), precision=precision))
            require(overlap.hit is not None and overlap.hit.object_id == 11, "overlap failed")
            sweep = target.sweep_sphere_closest(
                SphereSweepQuery((-3, 0, 0), 0.5, (1, 0, 0), 10, precision=precision)
            )
            require(sweep.hit is not None and sweep.hit.object_id == 11, "sphere sweep failed")
            require(math.isclose(sweep.hit.distance, 1.5, abs_tol=1e-6), "wrong sweep distance")
            config = RaycastPlanConfig(
                target.reference, raw.HitMode.CLOSEST, 2, 1, 2, precision=precision
            )
            plan = instance.create_raycast_plan(config)
            try:
                operation, _ = plan.enqueue(
                    origins=Column(ctypes.c_double, [-3, 0, 0, -3, 4, 0], [2, 3], raw.K_DL_FLOAT),
                    unit_directions=Column(
                        ctypes.c_float, [1, 0, 0, 1, 0, 0], [2, 3], raw.K_DL_FLOAT
                    ),
                    max_distances=Column(ctypes.c_double, [10, 10], [2], raw.K_DL_FLOAT),
                )
                try:
                    operation.wait()
                    result = operation.fetch_raycast_batch()
                    require(
                        result.common.execution_device == Device(), "batch did not execute on CPU"
                    )
                    counts = ctypes.cast(
                        result.hit_ranges.hit_counts.data, ctypes.POINTER(ctypes.c_uint64)
                    )
                    require((counts[0], counts[1]) == (1, 0), "wrong batch hit counts")
                    ids = ctypes.cast(result.object_ids.data, ctypes.POINTER(ctypes.c_uint64))
                    require(ids[0] == 11, "wrong batch object identity")
                finally:
                    operation.close()
            finally:
                plan.close()
            # A CPU artifact must report unavailable CUDA without disabling its CPU queries.
            if require_cuda:
                require(
                    (library.path.parent / "ovspatialquery_cuda.ptx").is_file(),
                    "CUDA wheel has no PTX",
                )
                cuda_batch(instance, target)
            if (
                require_unavailable_cuda
                or not (library.path.parent / "ovspatialquery_cuda.ptx").is_file()
            ):
                try:
                    cuda_plan = instance.create_raycast_plan(
                        RaycastPlanConfig(
                            target.reference,
                            raw.HitMode.CLOSEST,
                            1,
                            1,
                            1,
                            device=Device(raw.K_DL_CUDA, 0),
                        )
                    )
                except ApiError as error:
                    require(
                        error.status == raw.ApiStatus.NOT_SUPPORTED, "wrong unavailable CUDA status"
                    )
                else:
                    cuda_plan.close()
                    raise RuntimeError("artifact unexpectedly accepted an unavailable CUDA plan")
            return {
                "module_path": str(Path(ovspatialquery.__file__).resolve()),
                "library_path": str(library.path.resolve()),
                "version": list(library.version),
                "package_version": package_version,
                "has_ovstage": library.has_ovstage,
                "context_smoke": "passed",
                "cpu_queries": "passed",
                "cuda_queries": "passed"
                if require_cuda
                else "unavailable"
                if require_unavailable_cuda
                else "not-requested",
            }
        finally:
            population.close()
            adapter.destroy_definition(GeometryDefinitionRef(raw.GeometryType.SPHERE, 7))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--require-cuda", action="store_true")
    mode.add_argument("--require-unavailable-cuda", action="store_true")
    options = parser.parse_args()
    print(json.dumps(run(options.require_cuda, options.require_unavailable_cuda), sort_keys=True))
