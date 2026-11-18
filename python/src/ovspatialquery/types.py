# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable Python values for the current API projection."""

from __future__ import annotations
import array
import ctypes
from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
import os
from typing import Any, Iterable, Self, Sequence
from . import raw
from .raw import (
    BatchResultField,
    GeometryFeatureKind,
    GeometryType,
    HitMode,
    MeshHits,
    HitTruth,
    OverlapShapeKind,
    PairReporterOutcome,
    QueryFlag,
    QueryPrecision,
)


class _DLManagedTensor(ctypes.Structure):
    # ctypes caches pointer types, so this layout must not be recreated per export.
    _fields_ = [
        ("dl_tensor", raw.DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class _DLManagedTensorVersioned(ctypes.Structure):
    # DLPack 1.x; only `major` may be read before checking it is 1.
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", raw.DLTensor),
    ]


# Private function objects, so their argtypes do not clash with other pythonapi users.
_capsule_tensor = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)
_capsule_named = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)


def _finite_vector(values: Sequence[float], length: int, label: str) -> tuple[float, ...]:
    result = tuple((float(value) for value in values))
    if len(result) != length or not all((math.isfinite(value) for value in result)):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _tolerance(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("contact tolerance must be finite and nonnegative")
    return result


def _participation(value: int, label: str = "participation bits") -> raw.ParticipationBits:
    """An int below 2**128 as its two uint64 words, bits 0-63 first."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int; got {type(value).__name__}")
    if value < 0 or value >> 128:
        raise ValueError(f"{label} must fit 128 bits")
    return raw.ParticipationBits((value & 0xFFFFFFFFFFFFFFFF, value >> 64))


def _interval(minimum: float, maximum: float, label: str) -> None:
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 <= minimum <= maximum):
        raise ValueError(f"{label} interval must be finite, ordered, and nonnegative")


def _batch_result_fields(
    value: BatchResultField,
    precision: QueryPrecision,
    allowed: BatchResultField,
    family: str,
) -> int:
    bits = int(value)
    # HOST_MAPPED places a CUDA plan's columns and every family accepts it.
    unknown_or_disallowed = bits & ~int(allowed | BatchResultField.HOST_MAPPED)
    if unknown_or_disallowed:
        if unknown_or_disallowed & int(BatchResultField.QUERY_POSITION):
            raise ValueError(f"QUERY_POSITION is not valid for {family}")
        raise ValueError(f"result fields are not valid for {family}")
    if value & BatchResultField.SOURCE_FEATURE and precision not in (
        QueryPrecision.EXACT,
        QueryPrecision.EXACT_SINGLE,
    ):
        raise ValueError("SOURCE_FEATURE requires exact query precision")
    return bits


@dataclass(frozen=True)
class Device:
    device_type: int = raw.K_DL_CPU
    device_id: int = 0

    @classmethod
    def cpu(cls) -> "Device":
        return cls(raw.K_DL_CPU, 0)

    @classmethod
    def cuda(cls, index: int = 0) -> "Device":
        return cls(raw.K_DL_CUDA, index)

    @classmethod
    def from_raw(cls, value: raw.DLDevice) -> "Device":
        return cls(int(value.device_type), int(value.device_id))

    def to_raw(self) -> raw.DLDevice:
        return raw.DLDevice(self.device_type, self.device_id)


@dataclass(frozen=True)
class DeviceBudget:
    device: Device
    byte_budget: int

    def to_raw(self) -> raw.DeviceBudget:
        if self.byte_budget <= 0:
            raise ValueError("device byte budget must be positive")
        return raw.DeviceBudget(self.device.to_raw(), self.byte_budget)


@dataclass(frozen=True)
class InstanceConfig:
    operation_capacity: int = 8
    max_cpu_concurrency: int = 2
    host_byte_budget: int = 64 << 20
    device_budgets: tuple[DeviceBudget, ...] = ()

    def to_raw(self) -> tuple[raw.InstanceDesc, tuple[Any, ...]]:
        if min(self.operation_capacity, self.max_cpu_concurrency, self.host_byte_budget) <= 0:
            raise ValueError("instance capacities and host budget must be positive")
        values = tuple((item.to_raw() for item in self.device_budgets))
        storage = (raw.DeviceBudget * len(values))(*values) if values else None
        descriptor = raw.InstanceDesc(
            operation_capacity=self.operation_capacity,
            max_cpu_concurrency=self.max_cpu_concurrency,
            host_byte_budget=self.host_byte_budget,
            device_budgets=storage,
            device_budget_count=len(values),
        )
        return (descriptor, () if storage is None else (storage,))


@dataclass(frozen=True)
class PopulationConfig:
    retained_ordinal_capacity: int = 2
    # Device-prepared publications that may be pending at once; zero and one
    # both mean one (ADR-0021).
    pending_publication_capacity: int = 1

    def to_raw(self) -> raw.PopulationDesc:
        if self.retained_ordinal_capacity < 0:
            raise ValueError("retained ordinal capacity must be nonnegative")
        if self.pending_publication_capacity < 0:
            raise ValueError("pending publication capacity must be nonnegative")
        return raw.PopulationDesc(self.retained_ordinal_capacity, self.pending_publication_capacity)


@dataclass(frozen=True)
class OrdinalSelection:
    kind: raw.OrdinalSelectionKind
    source_ordinal: int = 0

    @classmethod
    def head(cls) -> "OrdinalSelection":
        return cls(raw.OrdinalSelectionKind.HEAD)

    @classmethod
    def retained_exact(cls, source_ordinal: int) -> "OrdinalSelection":
        return cls(raw.OrdinalSelectionKind.RETAINED_EXACT, source_ordinal)

    @classmethod
    def admitted_exact(cls, source_ordinal: int) -> "OrdinalSelection":
        """A completed retained ordinal or a device-prepared publication still being
        prepared; CUDA plan launches only. Experimental: may change in any minor release."""
        return cls(raw.OrdinalSelectionKind.ADMITTED_EXACT, source_ordinal)

    def to_raw(self) -> raw.OrdinalSelection:
        # Every query and launch builds one, so the int64 range is checked here rather than by
        # `raw._Structure`, whose per-field check costs about a microsecond a call.
        ordinal = self.source_ordinal
        if self.kind is not raw.OrdinalSelectionKind.HEAD and not 0 <= ordinal < 1 << 63:
            raise ValueError(f"source ordinal must be from 0 to 2**63 - 1; got {ordinal}")
        return raw.OrdinalSelection(int(self.kind), ordinal)


@dataclass(frozen=True)
class TargetRef:
    kind: raw.TargetKind
    handle: int

    def to_raw(self) -> raw.TargetRef:
        if self.handle == 0:
            raise ValueError("target handle must be nonzero")
        result = raw.TargetRef(int(self.kind))
        # Both handle union arms are one uint64 at offset 0.
        result.handle.population = self.handle
        return result

    def select(self, ordinal: OrdinalSelection | None = None) -> raw.QueryTarget:
        selected = OrdinalSelection.head() if ordinal is None else ordinal
        return raw.QueryTarget(self.to_raw(), selected.to_raw())


@dataclass(frozen=True)
class _DLPackLease:
    """One imported DLTensor descriptor and the objects retaining its pointer storage."""

    array: raw.InputArray
    keepalive: tuple[Any, ...]

    @classmethod
    def from_dlpack(
        cls,
        provider: Any,
        *,
        stream: int | None = None,
        ownership: raw.InputOwnership = raw.InputOwnership.BORROWED_UNTIL_TERMINAL,
    ) -> "_DLPackLease":
        streamed = {} if stream is None else {"stream": stream}
        try:
            # Only a versioned capsule can carry a read-only array; the library only reads.
            capsule = provider.__dlpack__(**streamed, max_version=(1, 0))
        except TypeError:
            # A producer older than DLPack 1.0 takes no max_version.
            capsule = provider.__dlpack__(**streamed)
        if _capsule_named(capsule, b"dltensor_versioned"):
            managed = _DLManagedTensorVersioned.from_address(
                _capsule_tensor(capsule, b"dltensor_versioned")
            )
            if managed.major != 1:
                raise ValueError(
                    f"provider returned DLPack {managed.major}.{managed.minor}; this package reads 1.x"
                )
            tensor = managed.dl_tensor
        else:
            address = _capsule_tensor(capsule, b"dltensor")
            if not address:
                raise ValueError("provider returned an invalid dltensor capsule")
            tensor = _DLManagedTensor.from_address(address).dl_tensor
        projected = raw.DLTensor.from_buffer_copy(tensor)
        projection_keepalive: list[Any] = [provider, capsule]
        if tensor.ndim > 0 and tensor.shape:
            shape = tensor.shape[: tensor.ndim]
            empty = 0 in shape
            if empty:
                projected.data = None
                projected.byte_offset = 0
            # A zero-size tensor's strides are arbitrary (NumPy reports 0); libraries
            # before ADR-0042 refuse any but compact ones.
            if empty or not tensor.strides:
                compact = [1] * tensor.ndim
                for dimension in range(tensor.ndim - 2, -1, -1):
                    compact[dimension] = compact[dimension + 1] * max(shape[dimension + 1], 1)
                stride_storage = (ctypes.c_int64 * tensor.ndim)(*compact)
                projected.strides = stride_storage
                projection_keepalive.append(stride_storage)
        return cls(raw.InputArray(projected, int(ownership)), tuple(projection_keepalive))


@dataclass(frozen=True)
class DataType:
    code: int
    bits: int
    lanes: int


class _ResultExports:
    """One operation's result storage: its readiness, its live DLPack exports, and its end."""

    def __init__(self) -> None:
        self.terminal_observed = False
        self.destroyed = False
        self.result_ready = 0
        self.live = 0

    def require_no_exports(self) -> None:
        if self.live:
            raise BufferError(
                f"{self.live} exported result column(s) are still alive; "
                "release or copy them before destroying the operation"
            )


# Each export's DLManagedTensor, shape and strides, kept alive until its deleter runs.
_exports_by_address: dict[int, tuple[_ResultExports, _DLManagedTensor, Any, Any]] = {}


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _release_export(address: int) -> None:
    entry = _exports_by_address.pop(address, None)
    if entry is not None:
        entry[0].live -= 1


# Independent function objects, so their argtypes do not clash with other pythonapi users.
_capsule_new = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p
)(("PyCapsule_New", ctypes.pythonapi))
_capsule_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_capsule_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _destroy_capsule(capsule: int) -> None:
    # A consumer renames the capsule it takes and calls the deleter itself.
    if _capsule_is_valid(capsule, b"dltensor"):
        _release_export(_capsule_pointer(capsule, b"dltensor"))


_cuda_driver: Any = None


def _cuda_wait(stream: int | None, event: int, device_id: int) -> None:
    """Queues `stream` behind `event`, with the device's primary context current."""
    if stream == 0:
        raise BufferError("stream 0 is ambiguous for CUDA; pass 1 for the legacy default stream")
    if stream == -1 or not event:
        return
    global _cuda_driver
    if _cuda_driver is None:
        # Loaded on the first CUDA export only, so CPU-only processes never load the driver.
        driver = (
            ctypes.CDLL("nvcuda.dll", winmode=0x800)
            if os.name == "nt"
            else ctypes.CDLL("libcuda.so.1")
        )
        pointer = ctypes.c_void_p
        for name, arguments in {
            "cuInit": [ctypes.c_uint],
            "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "cuDevicePrimaryCtxRetain": [ctypes.POINTER(pointer), ctypes.c_int],
            "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
            "cuCtxPushCurrent_v2": [pointer],
            "cuCtxPopCurrent_v2": [ctypes.POINTER(pointer)],
            "cuStreamWaitEvent": [pointer, pointer, ctypes.c_uint],
        }.items():
            function = getattr(driver, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        _cuda_driver = driver

    def call(name: str, *arguments: Any) -> None:
        status = getattr(_cuda_driver, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed with CUDA status {status}")

    device, instance, popped = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p()
    call("cuInit", 0)
    call("cuDeviceGet", ctypes.byref(device), device_id)
    call("cuDevicePrimaryCtxRetain", ctypes.byref(instance), device)
    try:
        call("cuCtxPushCurrent_v2", instance)
        try:
            # None and 1 name the legacy default stream, 2 the per-thread one; both are
            # also the driver's handles for them.
            call("cuStreamWaitEvent", ctypes.c_void_p(1 if stream is None else stream), event, 0)
        finally:
            call("cuCtxPopCurrent_v2", ctypes.byref(popped))
    finally:
        call("cuDevicePrimaryCtxRelease_v2", device)


@dataclass(frozen=True)
class TensorView:
    """One result column. Views an `Operation` returns are DLPack producers: pass one to
    `numpy.from_dlpack`, `torch.from_dlpack` or `cupy.from_dlpack` to read it without copying.
    The consumer borrows operation-owned, read-only storage: `Operation.close` and
    `close_after_cuda` raise `BufferError` while an export is alive, and an export after the
    operation is destroyed raises too. Copy what must outlive the operation; a CPU or
    `HOST_MAPPED` result's `to_numpy()` copies every column at once.
    """

    data: int
    device: Device
    ndim: int
    dtype: DataType
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    byte_offset: int
    _owner: _ResultExports | None = field(default=None, repr=False, compare=False)

    def __dlpack_device__(self) -> tuple[int, int]:
        return (self.device.device_type, self.device.device_id)

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        """Exports this column as a legacy `dltensor` capsule, whatever `max_version` asks.

        A CPU column, and the pinned host column of a `HOST_MAPPED` CUDA plan, is exported
        once its operation is terminal-observed, with no stream wait. A pinned column ignores
        `stream`, since the terminal state implies `result_ready` has completed, but refuses 0
        as for CUDA. A CUDA column's export first queues `stream` behind the operation's
        `result_ready` event, following the DLPack stream values: None or 1 the legacy
        default stream, 2 the per-thread default stream, another value a stream handle, and
        -1 no wait, which leaves the ordering to the consumer.
        """
        del max_version  # An unversioned capsule is a valid answer to any requested version.
        owner = self._owner
        if owner is None:
            raise BufferError("only result columns returned by an Operation can be exported")
        if copy:
            raise BufferError("result columns are exported without copying")
        if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
            raise BufferError("result columns are exported on their own device only")
        if owner.destroyed:
            raise BufferError("the operation that owned this result column was destroyed")
        if self.device.device_type in (raw.K_DL_CPU, raw.K_DL_CUDA_HOST):
            if not owner.terminal_observed:
                raise BufferError(
                    "host result columns are unreadable until the operation is terminal"
                )
            # A CUDA consumer of pinned memory passes its stream; there is nothing left to wait for.
            if self.device.device_type == raw.K_DL_CUDA_HOST and stream == 0:
                raise BufferError(
                    "stream 0 is ambiguous for CUDA; pass 1 for the legacy default stream"
                )
        elif self.device.device_type == raw.K_DL_CUDA:
            _cuda_wait(stream, owner.result_ready, self.device.device_id)
        else:
            raise BufferError("result columns are exported from host or CUDA devices only")

        compact = [1] * self.ndim
        for dimension in range(self.ndim - 2, -1, -1):
            compact[dimension] = compact[dimension + 1] * max(self.shape[dimension + 1], 1)
        shape = (ctypes.c_int64 * self.ndim)(*self.shape)
        strides = (ctypes.c_int64 * self.ndim)(*(self.strides or compact))
        managed = _DLManagedTensor(
            raw.DLTensor(
                data=self.data or None,
                device=self.device.to_raw(),
                ndim=self.ndim,
                dtype=raw.DLDataType(self.dtype.code, self.dtype.bits, self.dtype.lanes),
                shape=shape,
                strides=strides,
                byte_offset=self.byte_offset,
            ),
            None,
            ctypes.cast(_release_export, ctypes.c_void_p),
        )
        address = ctypes.addressof(managed)
        _exports_by_address[address] = (owner, managed, shape, strides)
        owner.live += 1
        try:
            return _capsule_new(
                address, b"dltensor", ctypes.cast(_destroy_capsule, ctypes.c_void_p)
            )
        except BaseException:
            _release_export(address)
            raise

    @classmethod
    def optional_from_raw(
        cls, value: raw.DLTensor, *, _owner: _ResultExports | None = None
    ) -> "TensorView | None":
        # The header's absent descriptor is all-zero; callers may test shape == NULL.
        return cls.from_raw(value, _owner=_owner) if value.shape else None

    @classmethod
    def from_raw(cls, value: raw.DLTensor, *, _owner: _ResultExports | None = None) -> "TensorView":
        # `_owner` is the operation's export record; only `Operation` passes it, so its views
        # are exportable from creation and nothing is rebuilt afterwards.
        if not value.shape:
            raise ValueError("required result tensor descriptor is absent")
        ndim = int(value.ndim)
        shape = tuple((int(value.shape[index]) for index in range(ndim)))
        strides = (
            tuple((int(value.strides[index]) for index in range(ndim))) if value.strides else None
        )
        return cls(
            int(value.data or 0),
            Device.from_raw(value.device),
            ndim,
            DataType(int(value.dtype.code), int(value.dtype.bits), int(value.dtype.lanes)),
            shape,
            strides,
            int(value.byte_offset),
            _owner,
        )


def _copy_columns(value: Any, numpy: Any) -> Any:
    """Returns `value` with every result column it holds copied into a NumPy array."""
    if isinstance(value, TensorView):
        if value.device.device_type not in (raw.K_DL_CPU, raw.K_DL_CUDA_HOST):
            raise ValueError(
                "to_numpy copies host results only, from a CPU or HOST_MAPPED plan; "
                f"this column is on {value.device}"
            )
        # The copy owns its memory, so the borrowed export is released before this returns.
        return numpy.from_dlpack(value).copy()
    if not is_dataclass(value):
        return value
    return replace(
        value,
        **{item.name: _copy_columns(getattr(value, item.name), numpy) for item in fields(value)},
    )


class _CopiesToNumpy:
    def to_numpy(self) -> Self:
        """Returns this CPU or `HOST_MAPPED` result with every column copied into a
        `numpy.ndarray`, read by the same field paths. The copy holds no DLPack export, so it
        outlives the operation. Raises `ValueError` for a result in device memory."""
        try:
            import numpy
        except ModuleNotFoundError as missing:
            raise ModuleNotFoundError(
                "to_numpy requires NumPy; install numpy or read the columns through DLPack",
                name="numpy",
            ) from missing
        return _copy_columns(self, numpy)


def _optional_tensor_group(
    values: tuple[raw.DLTensor, ...], label: str, owner: _ResultExports | None
) -> tuple[TensorView, ...] | None:
    projected = tuple(TensorView.optional_from_raw(value, _owner=owner) for value in values)
    present = tuple(value is not None for value in projected)
    if any(present) and not all(present):
        raise ValueError(f"{label} columns must be present or absent together")
    if not any(present):
        return None
    return tuple(value for value in projected if value is not None)


@dataclass(frozen=True)
class BatchResultCommon:
    result_slot: int
    query_count: int
    hit_capacity: int
    truth: HitTruth
    source_ordinal: int
    execution_device: Device
    result_ready: int

    @classmethod
    def from_raw(cls, value: raw.BatchResultCommon) -> "BatchResultCommon":
        return cls(
            int(value.result_slot),
            int(value.query_count),
            int(value.hit_capacity),
            HitTruth(value.truth),
            int(value.source_ordinal),
            Device.from_raw(value.execution_device),
            int(value.result_ready.value),
        )


@dataclass(frozen=True)
class BatchHitRanges:
    query_status: TensorView
    stored_hit_count: TensorView
    hit_offsets: TensorView
    hit_counts: TensorView
    truncated: TensorView

    @classmethod
    def from_raw(
        cls, value: raw.BatchHitRanges, *, _owner: _ResultExports | None = None
    ) -> "BatchHitRanges":
        return cls(
            *(
                TensorView.from_raw(getattr(value, name), _owner=_owner)
                for name, _ in value._fields_
            )
        )


@dataclass(frozen=True)
class GeometryIdentityColumns:
    types: TensorView
    values: TensorView

    @classmethod
    def optional_from_raw(
        cls, value: raw.GeometryIdentityColumns, *, _owner: _ResultExports | None = None
    ) -> "GeometryIdentityColumns | None":
        group = _optional_tensor_group((value.types, value.values), "geometry identity", _owner)
        return None if group is None else cls(*group)


@dataclass(frozen=True)
class ExactFeatureColumns:
    kinds: TensorView
    source_face_indices: TensorView
    source_corner_indices: TensorView
    barycentrics: TensorView

    @classmethod
    def optional_from_raw(
        cls, value: raw.ExactFeatureColumns, *, _owner: _ResultExports | None = None
    ) -> "ExactFeatureColumns | None":
        columns = tuple(getattr(value, name) for name, _ in value._fields_)
        group = _optional_tensor_group(columns, "exact feature", _owner)
        return None if group is None else cls(*group)

    def triangle_payload_for(
        self, kind: GeometryFeatureKind
    ) -> tuple[TensorView, TensorView, TensorView] | None:
        if kind is GeometryFeatureKind.NONE:
            return None
        if kind is not GeometryFeatureKind.TRIANGLE:
            raise ValueError("unknown exact feature kind")
        return (self.source_face_indices, self.source_corner_indices, self.barycentrics)


@dataclass(frozen=True)
class RaycastCandidateColumns:
    bounds_positions: TensorView | None
    bounds_normals: TensorView | None
    distances: TensorView | None


@dataclass(frozen=True)
class RaycastExactColumns:
    positions: TensorView | None
    normals: TensorView | None
    distances: TensorView | None
    features: ExactFeatureColumns | None


def _optional_columns(column_type: type, columns: Any, owner: _ResultExports | None) -> Any:
    return column_type(
        *(
            ExactFeatureColumns.optional_from_raw(columns.features, _owner=owner)
            if field.name == "features"
            else TensorView.optional_from_raw(getattr(columns, field.name), _owner=owner)
            for field in fields(column_type)
        )
    )


def _project_batch(
    value: Any, owner: _ResultExports | None, contact: tuple[type, type] | None = None
) -> tuple[Any, ...]:
    common = BatchResultCommon.from_raw(value.common)
    columns = value.columns
    geometry = GeometryIdentityColumns.optional_from_raw(columns.geometry, _owner=owner)
    projected = (
        common,
        BatchHitRanges.from_raw(columns.hit_ranges, _owner=owner),
        TensorView.from_raw(columns.object_ids, _owner=owner),
        geometry,
    )
    material_ids = TensorView.optional_from_raw(columns.material_ids, _owner=owner)
    if contact is None:
        return (*projected, material_ids)
    if common.truth is HitTruth.BOUNDS_CANDIDATE:
        candidate = _optional_columns(contact[0], columns.contact.candidate, owner)
        return (*projected, candidate, None, material_ids)
    exact = _optional_columns(contact[1], columns.contact.exact, owner)
    return (*projected, None, exact, material_ids)


@dataclass(frozen=True)
class RaycastBatchResult(_CopiesToNumpy):
    common: BatchResultCommon
    hit_ranges: BatchHitRanges
    object_ids: TensorView
    geometry: GeometryIdentityColumns | None
    candidate: RaycastCandidateColumns | None
    exact: RaycastExactColumns | None
    material_ids: TensorView | None = None

    @classmethod
    def from_raw(
        cls, value: raw.RaycastBatchResult, *, _owner: _ResultExports | None = None
    ) -> "RaycastBatchResult":
        if type(value) is not raw.RaycastBatchResult:
            raise TypeError("raycast terminal result requires raw RaycastBatchResult")
        return cls(*_project_batch(value, _owner, (RaycastCandidateColumns, RaycastExactColumns)))


@dataclass(frozen=True)
class RaycastBatchReservation:
    common: BatchResultCommon
    hit_ranges: BatchHitRanges
    object_ids: TensorView
    geometry: GeometryIdentityColumns | None
    candidate: RaycastCandidateColumns | None
    exact: RaycastExactColumns | None
    material_ids: TensorView | None = None

    @classmethod
    def from_raw(
        cls, value: raw.RaycastBatchReservation, *, _owner: _ResultExports | None = None
    ) -> "RaycastBatchReservation":
        if type(value) is not raw.RaycastBatchReservation:
            raise TypeError("raycast pending reservation requires raw RaycastBatchReservation")
        return cls(*_project_batch(value, _owner, (RaycastCandidateColumns, RaycastExactColumns)))


@dataclass(frozen=True)
class SphereSweepCandidateColumns:
    bounds_positions: TensorView | None
    bounds_normals: TensorView | None
    query_positions: TensorView | None
    distances: TensorView | None


@dataclass(frozen=True)
class SphereSweepExactColumns:
    positions: TensorView | None
    normals: TensorView | None
    query_positions: TensorView | None
    distances: TensorView | None
    features: ExactFeatureColumns | None


@dataclass(frozen=True)
class SphereSweepBatchResult(_CopiesToNumpy):
    common: BatchResultCommon
    hit_ranges: BatchHitRanges
    object_ids: TensorView
    geometry: GeometryIdentityColumns | None
    candidate: SphereSweepCandidateColumns | None
    exact: SphereSweepExactColumns | None
    material_ids: TensorView | None = None

    @classmethod
    def from_raw(
        cls, value: raw.SphereSweepBatchResult, *, _owner: _ResultExports | None = None
    ) -> "SphereSweepBatchResult":
        if type(value) is not raw.SphereSweepBatchResult:
            raise TypeError("sphere-sweep terminal result requires raw SphereSweepBatchResult")
        return cls(
            *_project_batch(value, _owner, (SphereSweepCandidateColumns, SphereSweepExactColumns))
        )


@dataclass(frozen=True)
class SphereSweepBatchReservation:
    common: BatchResultCommon
    hit_ranges: BatchHitRanges
    object_ids: TensorView
    geometry: GeometryIdentityColumns | None
    candidate: SphereSweepCandidateColumns | None
    exact: SphereSweepExactColumns | None
    material_ids: TensorView | None = None

    @classmethod
    def from_raw(
        cls, value: raw.SphereSweepBatchReservation, *, _owner: _ResultExports | None = None
    ) -> "SphereSweepBatchReservation":
        if type(value) is not raw.SphereSweepBatchReservation:
            raise TypeError(
                "sphere-sweep pending reservation requires raw SphereSweepBatchReservation"
            )
        return cls(
            *_project_batch(value, _owner, (SphereSweepCandidateColumns, SphereSweepExactColumns))
        )


BoxSweepCandidateColumns = SphereSweepCandidateColumns
BoxSweepExactColumns = SphereSweepExactColumns


@dataclass(frozen=True)
class BoxSweepBatchResult(SphereSweepBatchResult):
    pass


@dataclass(frozen=True)
class BoxSweepBatchReservation(SphereSweepBatchReservation):
    pass


@dataclass(frozen=True)
class OverlapBatchResult(_CopiesToNumpy):
    common: BatchResultCommon
    hit_ranges: BatchHitRanges
    object_ids: TensorView
    geometry: GeometryIdentityColumns | None
    material_ids: TensorView | None = None

    @classmethod
    def from_raw(
        cls, value: raw.OverlapBatchResult, *, _owner: _ResultExports | None = None
    ) -> "OverlapBatchResult":
        if type(value) is not raw.OverlapBatchResult:
            raise TypeError("overlap terminal result requires raw OverlapBatchResult")
        return cls(*_project_batch(value, _owner))


@dataclass(frozen=True)
class OverlapBatchReservation:
    common: BatchResultCommon
    hit_ranges: BatchHitRanges
    object_ids: TensorView
    geometry: GeometryIdentityColumns | None
    material_ids: TensorView | None = None

    @classmethod
    def from_raw(
        cls, value: raw.OverlapBatchReservation, *, _owner: _ResultExports | None = None
    ) -> "OverlapBatchReservation":
        if type(value) is not raw.OverlapBatchReservation:
            raise TypeError("overlap pending reservation requires raw OverlapBatchReservation")
        return cls(*_project_batch(value, _owner))


@dataclass(frozen=True)
class PopulationInfo:
    ready: bool
    head_ordinal: int | None
    object_count: int
    retained_ordinal_count: int
    update_pending: bool

    @classmethod
    def from_raw(cls, value: raw.PopulationInfo) -> "PopulationInfo":
        ready = bool(value.ready)
        return cls(
            ready,
            int(value.head_ordinal) if ready else None,
            int(value.object_count),
            int(value.retained_ordinal_count),
            bool(value.update_pending),
        )


# The exclusion columns are stdlib arrays so the package needs no third-party module; their
# typecodes are C unsigned long long and int, 8 and 4 bytes on every platform it supports.
_EXCLUDED_OBJECTS_TYPECODE = "Q"
_EXCLUSION_REASONS_TYPECODE = "i"
if (
    array.array(_EXCLUDED_OBJECTS_TYPECODE).itemsize != 8
    or array.array(_EXCLUSION_REASONS_TYPECODE).itemsize != 4
):
    raise ImportError("ovspatialquery needs 8-byte array('Q') and 4-byte array('i') items")


@dataclass(frozen=True, eq=False)
class PublicationResult:
    """Publication counts and owning copies of the complete sorted exclusion list.

    `excluded_objects` is an `array.array` of uint64 object IDs and `exclusion_reasons` one of
    int32 reason codes, the source adapter's enum values. Both remain readable after the
    operation closes and export the buffer protocol, so `numpy.asarray` views them without
    copying.
    """

    published_ordinal: int
    object_count: int
    excluded_object_count: int
    excluded_objects: array.array
    exclusion_reasons: array.array

    def __post_init__(self) -> None:
        # Equality and hashing use the values at construction, so editing a column later does
        # not change either.
        object.__setattr__(
            self,
            "_comparison_key",
            (
                self.published_ordinal,
                self.object_count,
                self.excluded_object_count,
                self.excluded_objects.tobytes(),
                self.exclusion_reasons.tobytes(),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PublicationResult):
            return NotImplemented
        return self._comparison_key == other._comparison_key

    def __hash__(self) -> int:
        return hash(self._comparison_key)

    @classmethod
    def from_raw(cls, value: raw.PopulationUpdateResult) -> "PublicationResult":
        count = int(value.excluded_object_count)
        columns = []
        for tensor, typecode in (
            (value.excluded_objects, _EXCLUDED_OBJECTS_TYPECODE),
            (value.exclusion_reasons, _EXCLUSION_REASONS_TYPECODE),
        ):
            column = array.array(typecode)
            if count:
                address = int(tensor.data or 0) + int(tensor.byte_offset)
                column.frombytes((ctypes.c_char * (count * column.itemsize)).from_address(address))
            columns.append(column)
        return cls(int(value.published_ordinal), int(value.object_count), count, *columns)


@dataclass(frozen=True)
class SceneHit:
    """One hit. `position` is the contact point on the object's bounds or surface; a sweep
    also sets `query_position`, the moving shape's center at that contact."""

    object_id: int
    geometry_type: GeometryType
    geometry_value: int
    truth: HitTruth
    position: tuple[float, float, float] | None = None
    normal: tuple[float, float, float] | None = None
    distance: float | None = None
    source_face_index: int | None = None
    source_face_corner_indices: tuple[int, int, int] | None = None
    barycentrics: tuple[float, float, float] | None = None
    query_position: tuple[float, float, float] | None = None
    material_id: int = 0
    """The material the population assigns this hit's surface; zero means none."""


def _floats(values: Any) -> tuple[float, ...] | None:
    return None if values is None else tuple(float(item) for item in values)


def _scene_hit(
    hit: Any,
    truth: HitTruth,
    position: Any = None,
    normal: Any = None,
    feature: Any = None,
    query_position: Any = None,
) -> SceneHit:
    triangle = None
    if feature is not None and feature.kind == GeometryFeatureKind.TRIANGLE:
        triangle = feature.data.triangle
    return SceneHit(
        int(hit.identity.object_id.value),
        GeometryType(hit.identity.geometry.type),
        int(hit.identity.geometry.value),
        truth,
        _floats(position),
        _floats(normal),
        None if position is None else float(hit.distance),
        None if triangle is None else int(triangle.source_face_index),
        None if triangle is None else tuple(int(i) for i in triangle.source_face_corner_indices),
        None if triangle is None else _floats(triangle.barycentrics),
        _floats(query_position),
        int(hit.identity.material_id),
    )


def _raycast_hit(value: raw.RaycastHit) -> SceneHit:
    truth = HitTruth(value.truth)
    if truth is HitTruth.BOUNDS_CANDIDATE:
        hit = value.data.bounds_candidate
        return _scene_hit(hit, truth, hit.bounds_position, hit.bounds_normal)
    hit = value.data.exact
    return _scene_hit(hit, truth, hit.position, hit.normal, hit.feature)


@dataclass(frozen=True)
class MultipleResult:
    """Every hit one multiple-hit query stored, in unspecified order."""

    source_ordinal: int
    hits: tuple[SceneHit, ...]
    truncated: bool


def _overlap_hit(value: raw.OverlapHit) -> SceneHit:
    return _scene_hit(value, HitTruth(value.truth))


@dataclass(frozen=True)
class RaycastResult:
    source_ordinal: int
    hit: SceneHit | None

    @classmethod
    def from_raw(cls, value: raw.RaycastSingleResult) -> "RaycastResult":
        return cls(int(value.source_ordinal), _raycast_hit(value.hit) if value.has_hit else None)


@dataclass(frozen=True)
class OverlapResult:
    source_ordinal: int
    hit: SceneHit | None

    @classmethod
    def from_raw(cls, value: raw.OverlapAnyResult) -> "OverlapResult":
        return cls(int(value.source_ordinal), _overlap_hit(value.hit) if value.has_hit else None)


def _sweep_hit(value: raw.SphereSweepHit) -> SceneHit:
    truth = HitTruth(value.truth)
    if truth is HitTruth.BOUNDS_CANDIDATE:
        hit = value.data.bounds_candidate
        return _scene_hit(
            hit, truth, hit.bounds_position, hit.bounds_normal, None, hit.query_position
        )
    hit = value.data.exact
    return _scene_hit(hit, truth, hit.object_position, hit.normal, hit.feature, hit.query_position)


@dataclass(frozen=True)
class SphereSweepResult:
    """The one hit a closest or any sweep returns, if any."""

    source_ordinal: int
    hit: SceneHit | None

    @classmethod
    def from_raw(cls, value: raw.SphereSweepClosestResult) -> "SphereSweepResult":
        return cls(int(value.source_ordinal), _sweep_hit(value.hit) if value.has_hit else None)


@dataclass(frozen=True)
class BoxSweepResult(SphereSweepResult):
    pass


@dataclass(frozen=True)
class PairSnapshotResult:
    first_source_ordinal: int
    second_source_ordinal: int
    rows: tuple[tuple[int, int], ...]
    truncated: bool


def _common(
    target: TargetRef,
    ordinal: OrdinalSelection,
    precision: QueryPrecision,
    query_bits: int,
    contact_tolerance: float,
    query_flags: QueryFlag = QueryFlag(0),
    both_sides: bool = False,
) -> raw.QueryCommon:
    return raw.QueryCommon(
        target.select(ordinal),
        int(precision),
        _participation(query_bits, "query bits"),
        _tolerance(contact_tolerance),
        _query_flags(query_flags, both_sides),
    )


def _query_flags(query_flags: QueryFlag, both_sides: bool) -> int:
    """The `query_flags` word: `query_flags`, with `MESH_BOTH_SIDES` added when `both_sides`."""
    if not isinstance(query_flags, QueryFlag):
        raise TypeError("query_flags must be a QueryFlag")
    if not isinstance(both_sides, bool):
        raise TypeError("both_sides must be a bool")
    value = int(query_flags | (QueryFlag.MESH_BOTH_SIDES if both_sides else QueryFlag(0)))
    # ctypes would truncate a wider value and drop the unknown high bits the library refuses.
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("query_flags must fit in 32 bits")
    return value


@dataclass(frozen=True)
class RaycastQuery:
    origin: tuple[float, float, float]
    unit_direction: tuple[float, float, float]
    max_distance: float
    min_distance: float = 0.0
    ordinal: OrdinalSelection = OrdinalSelection.head()
    precision: QueryPrecision = QueryPrecision.EXACT
    query_bits: int = 0
    contact_tolerance: float = 0.0
    query_flags: QueryFlag = QueryFlag(0)
    both_sides: bool = False

    def to_raw(self, target: TargetRef) -> raw.RaycastDesc:
        origin = _finite_vector(self.origin, 3, "ray origin")
        direction = _finite_vector(self.unit_direction, 3, "ray direction")
        _interval(self.min_distance, self.max_distance, "ray")
        return raw.RaycastDesc(
            _common(
                target,
                self.ordinal,
                self.precision,
                self.query_bits,
                self.contact_tolerance,
                self.query_flags,
                self.both_sides,
            ),
            raw.Ray(
                (ctypes.c_double * 3)(*origin),
                (ctypes.c_float * 3)(*direction),
                self.min_distance,
                self.max_distance,
            ),
        )


@dataclass(frozen=True)
class SphereSweepQuery:
    center: tuple[float, float, float]
    radius: float
    unit_direction: tuple[float, float, float]
    max_distance: float
    min_distance: float = 0.0
    ordinal: OrdinalSelection = OrdinalSelection.head()
    precision: QueryPrecision = QueryPrecision.EXACT
    query_bits: int = 0
    contact_tolerance: float = 0.0
    query_flags: QueryFlag = QueryFlag(0)
    both_sides: bool = False

    def to_raw(self, target: TargetRef) -> raw.SphereSweepDesc:
        center = _finite_vector(self.center, 3, "sweep center")
        direction = _finite_vector(self.unit_direction, 3, "sweep direction")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("sweep radius must be finite and positive")
        _interval(self.min_distance, self.max_distance, "sweep")
        sweep = raw.SphereSweep(
            (ctypes.c_double * 3)(*center),
            self.radius,
            (ctypes.c_float * 3)(*direction),
            self.min_distance,
            self.max_distance,
        )
        return raw.SphereSweepDesc(
            _common(
                target,
                self.ordinal,
                self.precision,
                self.query_bits,
                self.contact_tolerance,
                self.query_flags,
                self.both_sides,
            ),
            sweep,
        )


@dataclass(frozen=True)
class BoxSweepQuery:
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    unit_direction: tuple[float, float, float]
    max_distance: float
    min_distance: float = 0.0
    ordinal: OrdinalSelection = OrdinalSelection.head()
    precision: QueryPrecision = QueryPrecision.EXACT
    query_bits: int = 0
    contact_tolerance: float = 0.0
    query_flags: QueryFlag = QueryFlag(0)
    both_sides: bool = False

    def to_raw(self, target: TargetRef) -> raw.BoxSweepDesc:
        center = _finite_vector(self.center, 3, "box sweep center")
        half_extents = _finite_vector(self.half_extents, 3, "box sweep half extents")
        rotation = _finite_vector(self.rotation_xyzw, 4, "box sweep rotation")
        direction = _finite_vector(self.unit_direction, 3, "box sweep direction")
        if any(extent <= 0 for extent in half_extents):
            raise ValueError("box sweep half extents must be positive")
        _interval(self.min_distance, self.max_distance, "box sweep")
        return raw.BoxSweepDesc(
            _common(
                target,
                self.ordinal,
                self.precision,
                self.query_bits,
                self.contact_tolerance,
                self.query_flags,
                self.both_sides,
            ),
            raw.BoxSweep(
                (ctypes.c_double * 3)(*center),
                (ctypes.c_double * 3)(*half_extents),
                (ctypes.c_float * 4)(*rotation),
                (ctypes.c_float * 3)(*direction),
                self.min_distance,
                self.max_distance,
            ),
        )


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class OrientedBox:
    center: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    half_extents: tuple[float, float, float]


@dataclass(frozen=True)
class Aabb:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


@dataclass(frozen=True)
class OverlapQuery:
    shape: Sphere | OrientedBox | Aabb
    ordinal: OrdinalSelection = OrdinalSelection.head()
    precision: QueryPrecision = QueryPrecision.EXACT
    query_bits: int = 0
    contact_tolerance: float = 0.0

    def to_raw(self, target: TargetRef) -> raw.OverlapDesc:
        shape = raw.OverlapShape()
        if isinstance(self.shape, Sphere):
            center = _finite_vector(self.shape.center, 3, "sphere center")
            if not math.isfinite(self.shape.radius) or self.shape.radius <= 0:
                raise ValueError("sphere radius must be finite and positive")
            shape.kind = OverlapShapeKind.SPHERE
            shape.data.sphere = raw.OverlapSphere((ctypes.c_double * 3)(*center), self.shape.radius)
        elif isinstance(self.shape, OrientedBox):
            shape.kind = OverlapShapeKind.BOX
            shape.data.box = raw.OverlapBox(
                (ctypes.c_double * 3)(*_finite_vector(self.shape.center, 3, "box center")),
                (ctypes.c_float * 4)(*_finite_vector(self.shape.rotation_xyzw, 4, "box rotation")),
                (ctypes.c_double * 3)(
                    *_finite_vector(self.shape.half_extents, 3, "box half extents")
                ),
            )
        else:
            shape.kind = OverlapShapeKind.AABB
            shape.data.aabb = raw.OverlapAabb(
                (ctypes.c_double * 3)(*_finite_vector(self.shape.minimum, 3, "AABB minimum")),
                (ctypes.c_double * 3)(*_finite_vector(self.shape.maximum, 3, "AABB maximum")),
            )
        return raw.OverlapDesc(
            _common(target, self.ordinal, self.precision, self.query_bits, self.contact_tolerance),
            shape,
        )


@dataclass(frozen=True)
class RaycastPlanConfig:
    target: TargetRef
    hit_mode: HitMode
    maximum_query_count: int
    maximum_hits_per_query: int
    hit_capacity_per_slot: int
    precision: QueryPrecision = QueryPrecision.BOUNDS_CANDIDATE
    device: Device = Device()
    result_slot_count: int = 1
    result_fields: BatchResultField = BatchResultField(0)
    contact_tolerance: float = 0.0
    mesh_hits: MeshHits = MeshHits.ALL

    def to_raw(self) -> raw.RaycastPlanDesc:
        if (
            min(self.maximum_query_count, self.maximum_hits_per_query, self.hit_capacity_per_slot)
            < 0
        ):
            raise ValueError("raycast plan bounds must be nonnegative")
        if self.result_slot_count <= 0:
            raise ValueError("raycast plan result-slot count must be positive")
        fields = _batch_result_fields(
            self.result_fields,
            self.precision,
            BatchResultField.POSITION
            | BatchResultField.NORMAL
            | BatchResultField.DISTANCE
            | BatchResultField.SOURCE_FEATURE
            | BatchResultField.GEOMETRY_IDENTITY
            | BatchResultField.MATERIAL,
            "raycast",
        )
        return raw.RaycastPlanDesc(
            self.target.to_raw(),
            int(self.hit_mode),
            int(self.precision),
            self.device.to_raw(),
            self.maximum_query_count,
            self.maximum_hits_per_query,
            self.hit_capacity_per_slot,
            self.result_slot_count,
            fields,
            _tolerance(self.contact_tolerance),
            int(self.mesh_hits),
        )


@dataclass(frozen=True)
class SphereSweepPlanConfig:
    target: TargetRef
    radius: float
    maximum_query_count: int
    precision: QueryPrecision = QueryPrecision.BOUNDS_CANDIDATE
    device: Device = Device()
    result_slot_count: int = 1
    result_fields: BatchResultField = BatchResultField(0)
    contact_tolerance: float = 0.0
    # MULTIPLE keeps one hit per object reached, up to `maximum_hits_per_query`
    # per row and `hit_capacity_per_slot` per launch (`maximum_query_count` when
    # None, which CLOSEST and ANY require).
    hit_mode: HitMode = HitMode.CLOSEST
    maximum_hits_per_query: int = 1
    hit_capacity_per_slot: int | None = None

    def to_raw(self) -> raw.SphereSweepPlanDesc:
        if self.radius <= 0:
            raise ValueError("sphere-sweep plan radius must be positive")
        if min(self.maximum_query_count, self.maximum_hits_per_query, self._hit_capacity()) < 0:
            raise ValueError("sphere-sweep plan bounds must be nonnegative")
        if self.result_slot_count <= 0:
            raise ValueError("sphere-sweep plan result-slot count must be positive")
        fields = _batch_result_fields(
            self.result_fields,
            self.precision,
            BatchResultField.POSITION
            | BatchResultField.NORMAL
            | BatchResultField.DISTANCE
            | BatchResultField.SOURCE_FEATURE
            | BatchResultField.GEOMETRY_IDENTITY
            | BatchResultField.MATERIAL
            | BatchResultField.QUERY_POSITION,
            "sphere sweep",
        )
        return raw.SphereSweepPlanDesc(
            self.target.to_raw(),
            int(self.precision),
            self.radius,
            self.device.to_raw(),
            self.maximum_query_count,
            self.result_slot_count,
            fields,
            self._hit_capacity(),
            _tolerance(self.contact_tolerance),
            int(self.hit_mode),
            self.maximum_hits_per_query,
        )

    def _hit_capacity(self) -> int:
        return (
            self.maximum_query_count
            if self.hit_capacity_per_slot is None
            else self.hit_capacity_per_slot
        )


@dataclass(frozen=True)
class BoxSweepPlanConfig:
    target: TargetRef
    half_extents: tuple[float, float, float]
    maximum_query_count: int
    precision: QueryPrecision = QueryPrecision.BOUNDS_CANDIDATE
    device: Device = Device()
    result_slot_count: int = 1
    result_fields: BatchResultField = BatchResultField(0)
    contact_tolerance: float = 0.0
    # As for sphere-sweep plans.
    hit_mode: HitMode = HitMode.CLOSEST
    maximum_hits_per_query: int = 1
    hit_capacity_per_slot: int | None = None

    def to_raw(self) -> raw.BoxSweepPlanDesc:
        half_extents = _finite_vector(self.half_extents, 3, "box sweep plan half extents")
        if any(extent <= 0 for extent in half_extents):
            raise ValueError("box sweep plan half extents must be positive")
        if min(self.maximum_query_count, self.maximum_hits_per_query, self._hit_capacity()) < 0:
            raise ValueError("box sweep plan bounds must be nonnegative")
        if self.result_slot_count <= 0:
            raise ValueError("box sweep plan result-slot count must be positive")
        fields = _batch_result_fields(
            self.result_fields,
            self.precision,
            BatchResultField.POSITION
            | BatchResultField.NORMAL
            | BatchResultField.DISTANCE
            | BatchResultField.SOURCE_FEATURE
            | BatchResultField.GEOMETRY_IDENTITY
            | BatchResultField.MATERIAL
            | BatchResultField.QUERY_POSITION,
            "box sweep",
        )
        return raw.BoxSweepPlanDesc(
            self.target.to_raw(),
            int(self.precision),
            (ctypes.c_double * 3)(*half_extents),
            self.device.to_raw(),
            self.maximum_query_count,
            self.result_slot_count,
            fields,
            self._hit_capacity(),
            _tolerance(self.contact_tolerance),
            int(self.hit_mode),
            self.maximum_hits_per_query,
        )

    def _hit_capacity(self) -> int:
        return (
            self.maximum_query_count
            if self.hit_capacity_per_slot is None
            else self.hit_capacity_per_slot
        )


@dataclass(frozen=True)
class OverlapBatchShape:
    """Pose-free shape and default size every row of an overlap plan uses."""

    kind: OverlapShapeKind
    radius: float = 0.0
    half_extents: tuple[float, ...] = (0.0, 0.0, 0.0)

    @classmethod
    def sphere(cls, radius: float) -> "OverlapBatchShape":
        return cls(OverlapShapeKind.SPHERE, radius=float(radius))

    @classmethod
    def box(cls, half_extents: Iterable[float]) -> "OverlapBatchShape":
        return cls(
            OverlapShapeKind.BOX,
            half_extents=_finite_vector(tuple(half_extents), 3, "half extents"),
        )

    @classmethod
    def aabb(cls, half_extents: Iterable[float]) -> "OverlapBatchShape":
        return cls(
            OverlapShapeKind.AABB,
            half_extents=_finite_vector(tuple(half_extents), 3, "half extents"),
        )

    def to_raw(self) -> raw.OverlapBatchShape:
        sizes = (self.radius,) if self.kind is OverlapShapeKind.SPHERE else self.half_extents
        if len(sizes) != (1 if self.kind is OverlapShapeKind.SPHERE else 3) or not all(
            math.isfinite(size) and size > 0 for size in sizes
        ):
            raise ValueError(
                f"overlap plan {self.kind.name.lower()} size must be finite and positive"
            )
        value = raw.OverlapBatchShape()
        value.kind = int(self.kind)
        if self.kind is OverlapShapeKind.SPHERE:
            value.data.sphere.radius = self.radius
        elif self.kind is OverlapShapeKind.BOX:
            value.data.box.half_extents[:] = self.half_extents
        else:
            value.data.aabb.half_extents[:] = self.half_extents
        return value


@dataclass(frozen=True)
class OverlapPlanConfig:
    target: TargetRef
    shape: OverlapBatchShape
    hit_mode: HitMode
    maximum_query_count: int
    maximum_hits_per_query: int
    hit_capacity_per_slot: int
    precision: QueryPrecision = QueryPrecision.BOUNDS_CANDIDATE
    device: Device = Device()
    result_slot_count: int = 1
    result_fields: BatchResultField = BatchResultField(0)
    contact_tolerance: float = 0.0

    def to_raw(self) -> raw.OverlapPlanDesc:
        if (
            min(self.maximum_query_count, self.maximum_hits_per_query, self.hit_capacity_per_slot)
            < 0
        ):
            raise ValueError("overlap plan bounds must be nonnegative")
        if self.result_slot_count <= 0:
            raise ValueError("overlap plan result-slot count must be positive")
        fields = _batch_result_fields(
            self.result_fields,
            self.precision,
            BatchResultField.GEOMETRY_IDENTITY | BatchResultField.MATERIAL,
            "overlap; only GEOMETRY_IDENTITY and MATERIAL are allowed",
        )
        return raw.OverlapPlanDesc(
            self.target.to_raw(),
            int(self.hit_mode),
            int(self.precision),
            self.shape.to_raw(),
            self.device.to_raw(),
            self.maximum_query_count,
            self.maximum_hits_per_query,
            self.hit_capacity_per_slot,
            self.result_slot_count,
            fields,
            _tolerance(self.contact_tolerance),
        )


@dataclass(frozen=True)
class PairReporterConfig:
    first: TargetRef
    second: TargetRef
    active_pair_capacity: int
    delta_pair_capacity: int
    first_query_bits: int = 0
    second_query_bits: int = 0
    result_slot_count: int = 1
    contact_tolerance: float = 0.0
    result_device: Device = Device()

    def to_raw(self) -> raw.PairReporterDesc:
        if min(self.active_pair_capacity, self.delta_pair_capacity) < 0:
            raise ValueError("pair-reporter pair capacities must be nonnegative")
        if self.result_slot_count <= 0:
            raise ValueError("pair-reporter result-slot count must be positive")
        return raw.PairReporterDesc(
            self.first.to_raw(),
            _participation(self.first_query_bits, "first query bits"),
            self.second.to_raw(),
            _participation(self.second_query_bits, "second query bits"),
            self.active_pair_capacity,
            self.delta_pair_capacity,
            self.result_slot_count,
            _tolerance(self.contact_tolerance),
            self.result_device.to_raw(),
        )


@dataclass(frozen=True)
class PairReporterReservation:
    result_slot: int
    result_device: Device
    delta_pair_capacity: int
    result_ready: int
    # Columns of a CUDA reporter only; a CPU reporter's results are fetch-only.
    entered_pairs: TensorView | None
    exited_pairs: TensorView | None
    counts: TensorView | None

    @classmethod
    def from_raw(
        cls, value: raw.PairReporterReservation, *, _owner: _ResultExports | None = None
    ) -> "PairReporterReservation":
        return cls(
            int(value.result_slot),
            Device.from_raw(value.result_device),
            int(value.delta_pair_capacity),
            int(value.result_ready.value),
            TensorView.optional_from_raw(value.entered_pairs, _owner=_owner),
            TensorView.optional_from_raw(value.exited_pairs, _owner=_owner),
            TensorView.optional_from_raw(value.counts, _owner=_owner),
        )


@dataclass(frozen=True)
class PairReporterResult(_CopiesToNumpy):
    result_slot: int
    outcome: PairReporterOutcome
    first_source_ordinal: int
    second_source_ordinal: int
    accepted_active_pair_count: int
    result_ready: int
    entered_pairs: TensorView
    exited_pairs: TensorView

    @classmethod
    def from_raw(
        cls, value: raw.PairReporterResult, *, _owner: _ResultExports | None = None
    ) -> "PairReporterResult":
        return cls(
            int(value.result_slot),
            PairReporterOutcome(value.outcome),
            int(value.first_source_ordinal),
            int(value.second_source_ordinal),
            int(value.accepted_active_pair_count),
            int(value.result_ready.value),
            TensorView.from_raw(value.entered_pairs, _owner=_owner),
            TensorView.from_raw(value.exited_pairs, _owner=_owner),
        )


def flatten_keepalive(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple((value for value in values if value is not None))


__all__ = [
    "BoxSweepBatchResult",
    "BoxSweepBatchReservation",
    "BoxSweepCandidateColumns",
    "BoxSweepExactColumns",
    "BoxSweepPlanConfig",
    "BoxSweepQuery",
    "BoxSweepResult",
    "Aabb",
    "BatchHitRanges",
    "BatchResultCommon",
    "InstanceConfig",
    "DataType",
    "Device",
    "DeviceBudget",
    "OrdinalSelection",
    "OrientedBox",
    "OverlapPlanConfig",
    "OverlapBatchResult",
    "OverlapBatchReservation",
    "OverlapQuery",
    "OverlapResult",
    "PairReporterConfig",
    "PairReporterResult",
    "PairSnapshotResult",
    "PopulationConfig",
    "PopulationInfo",
    "PublicationResult",
    "RaycastPlanConfig",
    "RaycastBatchResult",
    "RaycastBatchReservation",
    "RaycastCandidateColumns",
    "RaycastExactColumns",
    "RaycastQuery",
    "RaycastResult",
    "SceneHit",
    "Sphere",
    "SphereSweepPlanConfig",
    "SphereSweepBatchResult",
    "SphereSweepBatchReservation",
    "SphereSweepCandidateColumns",
    "SphereSweepExactColumns",
    "SphereSweepQuery",
    "SphereSweepResult",
    "TargetRef",
    "TensorView",
    "flatten_keepalive",
]
