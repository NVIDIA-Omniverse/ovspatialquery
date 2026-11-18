# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""current API instance, target, plan, reporter, and operation lifetimes."""

from __future__ import annotations
import ctypes
import heapq
import itertools
import operator
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Iterable, Self, SupportsIndex
from . import raw
from .raw import ApiStatus, BatchResultField, OpState, QueryFlag
from .types import (
    _query_flags,
    BoxSweepPlanConfig,
    BoxSweepBatchResult,
    BoxSweepBatchReservation,
    BoxSweepQuery,
    BoxSweepResult,
    InstanceConfig,
    Device,
    _DLPackLease,
    _participation,
    MultipleResult,
    OrdinalSelection,
    OverlapPlanConfig,
    OverlapBatchResult,
    OverlapBatchReservation,
    OverlapQuery,
    OverlapResult,
    PairReporterConfig,
    PairReporterReservation,
    PairReporterResult,
    PairSnapshotResult,
    PopulationInfo,
    PublicationResult,
    RaycastPlanConfig,
    RaycastBatchResult,
    RaycastBatchReservation,
    RaycastQuery,
    RaycastResult,
    SphereSweepPlanConfig,
    SphereSweepBatchResult,
    SphereSweepBatchReservation,
    SphereSweepQuery,
    SphereSweepResult,
    TargetRef,
    _ResultExports,
    _overlap_hit,
    _raycast_hit,
    _sweep_hit,
)


def _decode_string(value: raw.String) -> str:
    if not value.ptr or not value.length:
        return ""
    return ctypes.string_at(value.ptr, value.length).decode("utf-8", errors="replace")


class ApiError(RuntimeError):
    def __init__(self, operation: str, status: ApiStatus, detail: str = "") -> None:
        self.operation = operation
        self.status = status
        self.detail = detail
        super().__init__(f"{operation}: {status.name}" + (f": {detail}" if detail else ""))


def _require(instance: "Instance", operation: str, result: raw.Result) -> None:
    status = ApiStatus(result.status)
    if status is not ApiStatus.SUCCESS:
        detail = _decode_string(instance.library.exports["ovspatialquery_get_last_error"]())
        raise ApiError(operation, status, detail)


def _supported(instance: "Instance", operation: str, result: raw.Result) -> bool:
    """True for SUCCESS, False for NOT_SUPPORTED; any other status raises `ApiError`."""
    if ApiStatus(result.status) is ApiStatus.NOT_SUPPORTED:
        return False
    _require(instance, operation, result)
    return True


def _require_enqueue(instance: "Instance", operation: str, result: raw.EnqueueResult) -> int:
    _require(instance, operation, raw.Result(result.status))
    if not result.op_id:
        raise ApiError(
            operation,
            ApiStatus.INTERNAL,
            "successful admission returned INVALID_OP",
        )
    return int(result.op_id)


class _Closeable:
    """A context manager that closes the object on exit. When the block raises, a failed close
    is suppressed so that exception propagates, and the object stays open for its instance.
    An instance closes what it still owns in ascending `_teardown_rank`, newest first."""

    _teardown_rank = 0

    def close(self) -> None:
        raise NotImplementedError

    def _exit(self) -> None:
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception is None:
            self._exit()
            return
        try:
            self._exit()
        except Exception:
            pass


class Instance:
    """Owns every object created through it. Closing it, or leaving its block, closes what is
    still open: operations (waiting for pending ones), plans and reporters (waiting for slots
    retired with `close_after_cuda`), views, populations, then geometry definitions, each newest
    first, and then destroys the instance. On a normal
    exit a failure raises. When the block raises, each failure is added to that exception as a
    note instead. Either way, an instance that is not destroyed is kept, with what it still
    owns, until a later close succeeds or the process exits."""

    def __init__(self, library: Any, handle: int, keepalive: Iterable[Any] = ()) -> None:
        self.library = library
        self._handle = ctypes.c_uint64(handle)
        self._keepalive = tuple(keepalive)
        self._owned: dict[Any, Any] = {}
        # While a close runs, what it has yet to visit, lowest rank and then newest first.
        self._closing: list[tuple[int, int, Any, Any]] | None = None
        self._closing_order = itertools.count()

    def _own(self, item: Any, key: Any = None) -> None:
        key = id(item) if key is None else key
        self._owned[key] = item
        if self._closing is not None:
            self._schedule(key, item)

    def _schedule(self, key: Any, item: Any) -> None:
        assert self._closing is not None
        heapq.heappush(self._closing, (item._teardown_rank, -next(self._closing_order), key, item))

    def _release(self, item: Any, key: Any = None) -> None:
        self._owned.pop(id(item) if key is None else key, None)

    @classmethod
    def create(cls, library: Any, config: InstanceConfig = InstanceConfig()) -> "Instance":
        """Creates an independent instance with the exact capacities in `config`."""
        descriptor, keepalive = config.to_raw()
        output = ctypes.c_uint64()
        result = library.exports["ovspatialquery_create_instance"](
            ctypes.byref(descriptor), ctypes.byref(output)
        )
        if int(result.status) != ApiStatus.SUCCESS:
            detail = _decode_string(library.exports["ovspatialquery_get_last_error"]())
            raise ApiError("ovspatialquery_create_instance", ApiStatus(result.status), detail)
        return cls(library, int(output.value), (descriptor, *keepalive))

    @property
    def handle(self) -> int:
        return int(self._handle.value)

    def memory_usage(self, device: Any) -> int:
        """Returns current instance-controlled bytes charged to one device budget."""
        output = ctypes.c_uint64()
        _require(
            self,
            "ovspatialquery_get_instance_memory_usage",
            self.library.exports["ovspatialquery_get_instance_memory_usage"](
                self.handle, device.to_raw(), ctypes.byref(output)
            ),
        )
        return int(output.value)

    def supports(self, capability: raw.Capability, device: Device = Device()) -> bool:
        """Whether this instance can use `capability` on `device`; it creates no object. The
        first question about a CUDA device for a capability the library knows tries to load the
        CUDA driver and start this library's CUDA backend on it, even when the answer is False.
        A capability a newer package names and the loaded library does not know is False."""
        return _supported(
            self,
            "ovspatialquery_query_capability",
            self.library.exports["ovspatialquery_query_capability"](
                self.handle, device.to_raw(), int(capability)
            ),
        )

    def create_raycast_plan(self, config: RaycastPlanConfig) -> "RaycastPlan":
        """Creates a reusable raycast plan and reserves its fixed result slots."""
        return RaycastPlan.create(self, config)

    def create_sphere_sweep_plan(self, config: SphereSweepPlanConfig) -> "SphereSweepPlan":
        """Creates a reusable sphere-sweep plan and reserves its fixed result slots."""
        return SphereSweepPlan.create(self, config)

    def create_box_sweep_plan(self, config: BoxSweepPlanConfig) -> "BoxSweepPlan":
        """Creates a reusable box-sweep plan and reserves its fixed result slots."""
        return BoxSweepPlan.create(self, config)

    def create_overlap_plan(self, config: OverlapPlanConfig) -> "OverlapPlan":
        """Creates a reusable overlap plan and reserves its fixed result slots."""
        return OverlapPlan.create(self, config)

    def create_pair_reporter(self, config: PairReporterConfig) -> "PairReporter":
        """Creates an empty-baseline, forward-only AABB-pair reporter. Experimental: pair
        discovery may change in any minor release."""
        return PairReporter.create(self, config)

    def explicit(self) -> Any:
        from .explicit import ExplicitAdapter

        return ExplicitAdapter(self)

    def query_pairs(
        self,
        first: "Target",
        second: "Target",
        *,
        capacity: int,
        first_ordinal: OrdinalSelection = OrdinalSelection.head(),
        second_ordinal: OrdinalSelection = OrdinalSelection.head(),
        first_query_bits: int = 0,
        second_query_bits: int = 0,
        contact_tolerance: float = 0.0,
    ) -> PairSnapshotResult:
        """Computes a stateless AABB-pair set between two selected immutable versions.
        Experimental: pair discovery may change in any minor release."""
        if capacity < 0:
            raise ValueError("pair capacity must be nonnegative")
        descriptor = raw.PairSnapshotDesc(
            first.reference.select(first_ordinal),
            _participation(first_query_bits, "first query bits"),
            second.reference.select(second_ordinal),
            _participation(second_query_bits, "second query bits"),
            contact_tolerance,
        )
        storage = (raw.Pair * capacity)() if capacity else None
        buffer = raw.PairBuffer(storage, capacity)
        output = raw.PairSnapshotResult()
        _require(
            self,
            "ovspatialquery_query_pairs_sync",
            self.library.exports["ovspatialquery_query_pairs_sync"](
                self.handle,
                ctypes.byref(descriptor),
                ctypes.byref(buffer),
                ctypes.byref(output),
            ),
        )
        rows = (
            tuple(
                (
                    (int(storage[index].first.value), int(storage[index].second.value))
                    for index in range(int(output.stored_pair_count))
                )
            )
            if storage is not None
            else ()
        )
        return PairSnapshotResult(
            int(output.first_source_ordinal),
            int(output.second_source_ordinal),
            rows,
            bool(output.truncated),
        )

    def ovstage(self, ovstage_module: Any, stage: Any) -> Any:
        from .ovstage import OvstageAdapter

        return OvstageAdapter(self, ovstage_module, stage)

    def close(self) -> None:
        self._close(None)

    def _close(self, unwinding: BaseException | None) -> None:
        if not self._handle.value:
            return
        # Kept until destroyed, so a failed teardown never frees what the library may still read.
        _abandoned.add(self)
        # Picked one at a time from a heap that `_own` extends: waiting for a registration can
        # hand the instance its definitions, and closing one item can release others.
        # Holding each visited item keeps its id from being reused by a new one.
        visited: dict[int, Any] = {}
        self._closing = []
        for key, item in self._owned.items():
            self._schedule(key, item)
        try:
            while self._closing:
                _, _, key, item = heapq.heappop(self._closing)
                if self._owned.get(key) is not item or id(item) in visited:
                    continue
                visited[id(item)] = item
                try:
                    item._exit()
                except Exception as failure:
                    if unwinding is None:
                        raise
                    unwinding.add_note(
                        f"ovspatialquery could not close a {type(item).__name__}: {failure}"
                    )
        finally:
            self._closing = None
        try:
            _require(
                self,
                "ovspatialquery_destroy_instance",
                self.library.exports["ovspatialquery_destroy_instance"](self.handle),
            )
        except ApiError as failure:
            if unwinding is None:
                raise
            unwinding.add_note(f"ovspatialquery kept the instance: {failure}")
            return
        self._handle.value = 0
        self._keepalive = ()
        _abandoned.discard(self)

    def __enter__(self) -> "Instance":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close(exception)


# Instances whose teardown has not finished, kept so nothing they lend the library is freed.
_abandoned: set[Instance] = set()


class Operation(_Closeable):
    """One enqueued operation. As a context manager, exit waits for it and closes it; a CUDA
    operation must be retired with `close_after_cuda` before the block ends, unless its plan
    keeps its results in host memory (`BatchResultField.HOST_MAPPED`)."""

    def __init__(
        self,
        instance: Instance,
        op_id: int,
        *,
        keepalive: Iterable[Any] = (),
        name: str = "operation",
        cuda: bool = False,
        owner: _SlotOwner | None = None,
    ) -> None:
        self.instance = instance
        self._op_id = ctypes.c_uint64(op_id)
        self.keepalive = tuple(keepalive)
        self._results = _ResultExports()
        self._name = name
        self._cuda = cuda
        self._on_completed: Any = None
        # The plan or reporter whose result slot this operation holds.
        self._owner = owner
        if owner is not None:
            owner._open_operations.add(self)
        instance._own(self)

    @property
    def op_id(self) -> int:
        return int(self._op_id.value)

    def wait(self, timeout: int = raw.TIMEOUT_INFINITE) -> OpState:
        """Waits until the operation is terminal or `timeout` expires and returns its state.
        `timeout` is an int number of nanoseconds; zero polls. When it expires first, returns
        `OpState.PENDING`. A timeout is not cancellation: the operation and all input-lifetime
        obligations remain live.
        """
        output = raw.OpWaitResult()
        result = self.instance.library.exports["ovspatialquery_wait_op"](
            self.instance.handle, self.op_id, timeout, ctypes.byref(output)
        )
        if ApiStatus(result.status) is ApiStatus.TIMEOUT:
            return OpState.PENDING
        _require(self.instance, "ovspatialquery_wait_op", result)
        state = OpState(output.state)
        self._observe(state)
        return state

    def status(self) -> tuple[OpState, ApiStatus, float]:
        """Polls the operation state and approximate progress without consuming it."""
        output = raw.OpStatus()
        _require(
            self.instance,
            "ovspatialquery_query_op_status",
            self.instance.library.exports["ovspatialquery_query_op_status"](
                self.instance.handle, self.op_id, ctypes.byref(output)
            ),
        )
        state = OpState(output.state)
        self._observe(state)
        return (state, ApiStatus(output.terminal_status), float(output.progress))

    def _observe(self, state: OpState) -> None:
        if state is OpState.PENDING:
            return
        self._results.terminal_observed = True
        if state is OpState.COMPLETED and self._on_completed is not None:
            completed, self._on_completed = self._on_completed, None
            completed()

    def finish(self, timeout: int = raw.TIMEOUT_INFINITE) -> None:
        """Waits for work whose only output is completion, such as a registration or a
        publication, and closes the operation. If it failed, raises `ApiError` with its terminal
        status after closing it. If `timeout` expires first, raises `ApiError` with TIMEOUT and
        leaves the operation open."""
        self._require_host_retirement()
        state = self.wait(timeout)
        if state is OpState.PENDING:
            raise ApiError(
                "ovspatialquery_wait_op", ApiStatus.TIMEOUT, "operation is still pending"
            )
        if state is OpState.FAILURE:
            detail = _decode_string(
                self.instance.library.exports["ovspatialquery_get_last_op_error"](
                    self.instance.handle, self.op_id
                )
            )
            status = self.status()[1]
            self.close()
            raise ApiError(self._name, status, detail)
        self.close()

    def _require_host_retirement(self) -> None:
        if self._cuda:
            raise RuntimeError(
                "a CUDA operation is retired only by close_after_cuda(consumer_done)"
            )

    def _exit(self) -> None:
        if not self.op_id:
            return
        self._require_host_retirement()
        self.wait()
        self.close()

    def fetch_population_update(self) -> PublicationResult:
        """Fetches the repeatable result of one successful population publication."""
        output = raw.PopulationUpdateResult()
        _require(
            self.instance,
            "ovspatialquery_fetch_population_update",
            self.instance.library.exports["ovspatialquery_fetch_population_update"](
                self.instance.handle, self.op_id, ctypes.byref(output)
            ),
        )
        return PublicationResult.from_raw(output)

    def fetch_dropped_triangles(self) -> dict[int, int]:
        """Fetches, for one successful ovstage publication, how many triangles without area
        each Mesh prim it read dropped, as object ID to count; Meshes that dropped none are
        absent."""
        output = raw.OvstageDroppedTriangles()
        _require(
            self.instance,
            "ovspatialquery_fetch_ovstage_dropped_triangles",
            self.instance.library.require_ovstage()[
                "ovspatialquery_fetch_ovstage_dropped_triangles"
            ](self.instance.handle, self.op_id, ctypes.byref(output)),
        )
        count = int(output.objects.shape[0]) if output.objects.shape else 0
        columns = [
            (ctypes.c_uint64 * count).from_address(int(tensor.data) + int(tensor.byte_offset))
            if count
            else ()
            for tensor in (output.objects, output.dropped_triangle_counts)
        ]
        return {int(prim): int(dropped) for prim, dropped in zip(*columns)}

    def fetch_pair_reporter(self) -> PairReporterResult:
        """Fetches repeatable entered/exited views and the typed advance outcome."""
        output = raw.PairReporterResult()
        _require(
            self.instance,
            "ovspatialquery_fetch_pair_reporter",
            self.instance.library.exports["ovspatialquery_fetch_pair_reporter"](
                self.instance.handle, self.op_id, ctypes.byref(output)
            ),
        )
        self._results.terminal_observed = True
        return self._project(PairReporterResult, output)

    def fetch_raycast_batch(self) -> RaycastBatchResult:
        """Fetches the repeatable terminal result views of one raycast-plan launch."""
        output = raw.RaycastBatchResult()
        _require(
            self.instance,
            "ovspatialquery_fetch_raycast_batch",
            self.instance.library.exports["ovspatialquery_fetch_raycast_batch"](
                self.instance.handle, self.op_id, ctypes.byref(output)
            ),
        )
        self._results.terminal_observed = True
        return self._project(RaycastBatchResult, output)

    def fetch_sphere_sweep_batch(self) -> SphereSweepBatchResult:
        """Fetches the repeatable terminal result views of one sphere-sweep launch."""
        output = raw.SphereSweepBatchResult()
        _require(
            self.instance,
            "ovspatialquery_fetch_sphere_sweep_batch",
            self.instance.library.exports["ovspatialquery_fetch_sphere_sweep_batch"](
                self.instance.handle, self.op_id, ctypes.byref(output)
            ),
        )
        self._results.terminal_observed = True
        return self._project(SphereSweepBatchResult, output)

    def fetch_box_sweep_batch(self) -> BoxSweepBatchResult:
        """Fetches repeatable terminal box-sweep contact columns."""
        output = raw.BoxSweepBatchResult()
        _require(
            self.instance,
            "ovspatialquery_fetch_box_sweep_batch",
            self.instance.library.exports["ovspatialquery_fetch_box_sweep_batch"](
                self.instance.handle, self.op_id, ctypes.byref(output)
            ),
        )
        self._results.terminal_observed = True
        return self._project(BoxSweepBatchResult, output)

    def fetch_overlap_batch(self) -> OverlapBatchResult:
        """Fetches the repeatable terminal result views of one overlap-plan launch."""
        output = raw.OverlapBatchResult()
        _require(
            self.instance,
            "ovspatialquery_fetch_overlap_batch",
            self.instance.library.exports["ovspatialquery_fetch_overlap_batch"](
                self.instance.handle, self.op_id, ctypes.byref(output)
            ),
        )
        self._results.terminal_observed = True
        return self._project(OverlapBatchResult, output)

    def _project(self, projection: Any, value: Any) -> Any:
        """Projects `value` with every result column exportable through this operation."""
        result = projection.from_raw(value, _owner=self._results)
        self._results.result_ready = getattr(result, "common", result).result_ready
        return result

    def close(self) -> None:
        """Destroys a terminal-observed CPU operation, a terminal-observed launch of a
        `HOST_MAPPED` CUDA plan whose columns no device work still reads, or one with no
        returned CUDA views. Raises `BufferError` while a DLPack export of its result columns
        is alive.
        """
        if not self._op_id.value:
            return
        self._results.require_no_exports()
        _require(
            self.instance,
            "ovspatialquery_destroy_op",
            self.instance.library.exports["ovspatialquery_destroy_op"](
                self.instance.handle, ctypes.byref(self._op_id)
            ),
        )
        self.keepalive = ()
        self._results.destroyed = True
        self.instance._release(self)
        if self._owner is not None:
            self._owner._open_operations.discard(self)

    def close_after_cuda(self, consumer_done: int) -> None:
        """Retires a terminal-observed CUDA operation after its consumers, given one event
        that covers all of them. Raises `BufferError` while a DLPack export of its result
        columns is alive.
        """
        self._results.require_no_exports()
        event = raw.CudaEvent(consumer_done)
        _require(
            self.instance,
            "ovspatialquery_destroy_op_after_cuda",
            self.instance.library.exports["ovspatialquery_destroy_op_after_cuda"](
                self.instance.handle, ctypes.byref(self._op_id), ctypes.byref(event)
            ),
        )
        self.keepalive = ()
        self._results.destroyed = True
        self.instance._release(self)
        if self._owner is not None:
            # The library releases the slot on its own thread after consumer_done.
            self._owner._open_operations.discard(self)
            self._owner._retired_after_cuda = True


class Target:
    def __init__(self, instance: Instance, reference: TargetRef) -> None:
        self.instance = instance
        self.reference = reference

    def raycast_closest(self, query: RaycastQuery) -> RaycastResult:
        """Runs one raycast on the caller thread and returns the closest hit, if any."""
        descriptor = query.to_raw(self.reference)
        output = raw.RaycastSingleResult()
        _require(
            self.instance,
            "ovspatialquery_raycast_closest_sync",
            self.instance.library.exports["ovspatialquery_raycast_closest_sync"](
                self.instance.handle, ctypes.byref(descriptor), ctypes.byref(output)
            ),
        )
        return RaycastResult.from_raw(output)

    def raycast_any(self, query: RaycastQuery) -> RaycastResult:
        """Runs one raycast on the caller thread and returns any admitted hit, if any. It makes no
        closest-hit or stable-choice promise.
        """
        descriptor = query.to_raw(self.reference)
        output = raw.RaycastSingleResult()
        _require(
            self.instance,
            "ovspatialquery_raycast_any_sync",
            self.instance.library.exports["ovspatialquery_raycast_any_sync"](
                self.instance.handle, ctypes.byref(descriptor), ctypes.byref(output)
            ),
        )
        return RaycastResult.from_raw(output)

    def raycast_multiple(
        self,
        query: RaycastQuery,
        capacity: SupportsIndex,
        mesh_hits: raw.MeshHits = raw.MeshHits.ALL,
    ) -> MultipleResult:
        """Stores up to `capacity` hits; `truncated` says more existed.

        A triangle mesh contributes one hit per triangle the ray crosses;
        `mesh_hits=MeshHits.NEAREST` keeps only each object's nearest contact.
        """
        return self._multiple(
            "ovspatialquery_raycast_multiple_sync",
            query,
            capacity,
            raw.RaycastHit,
            raw.RaycastHitBuffer,
            _raycast_hit,
            mesh_hits=int(mesh_hits),
        )

    def overlap_multiple(self, query: OverlapQuery, capacity: SupportsIndex) -> MultipleResult:
        """Stores up to `capacity` hits; `truncated` says more existed."""
        return self._multiple(
            "ovspatialquery_overlap_multiple_sync",
            query,
            capacity,
            raw.OverlapHit,
            raw.OverlapHitBuffer,
            _overlap_hit,
        )

    def _multiple(
        self,
        export: str,
        query: Any,
        capacity: SupportsIndex,
        row_type: type[ctypes.Structure],
        buffer_type: type[ctypes.Structure],
        project: Any,
        mesh_hits: int | None = None,
    ) -> MultipleResult:
        # Any integer, NumPy's included, as for every other count; a bool is not a count.
        try:
            capacity = -1 if isinstance(capacity, bool) else operator.index(capacity)
        except TypeError:
            capacity = -1
        if capacity < 0:
            raise ValueError("multiple-hit capacity must be a nonnegative integer")
        descriptor = query.to_raw(self.reference)
        if mesh_hits is not None:
            descriptor.mesh_hits = mesh_hits
        rows = (row_type * capacity)()
        buffer = buffer_type(rows if capacity else None, capacity)
        output = raw.MultipleResult()
        _require(
            self.instance,
            export,
            self.instance.library.exports[export](
                self.instance.handle,
                ctypes.byref(descriptor),
                ctypes.byref(buffer),
                ctypes.byref(output),
            ),
        )
        return MultipleResult(
            int(output.source_ordinal),
            tuple(project(rows[index]) for index in range(output.stored_hit_count)),
            bool(output.truncated),
        )

    def overlap_any(self, query: OverlapQuery) -> OverlapResult:
        """Runs one overlap and returns any overlapping object, if one exists."""
        descriptor = query.to_raw(self.reference)
        output = raw.OverlapAnyResult()
        _require(
            self.instance,
            "ovspatialquery_overlap_any_sync",
            self.instance.library.exports["ovspatialquery_overlap_any_sync"](
                self.instance.handle, ctypes.byref(descriptor), ctypes.byref(output)
            ),
        )
        return OverlapResult.from_raw(output)

    def sweep_sphere_closest(self, query: SphereSweepQuery) -> SphereSweepResult:
        """Runs one moving-sphere query and returns its closest hit, if any."""
        return self._sweep_single(
            "ovspatialquery_sweep_sphere_closest_sync", query, SphereSweepResult
        )

    def sweep_sphere_any(self, query: SphereSweepQuery) -> SphereSweepResult:
        """Returns a hit of any object the sphere reaches, if any; no closest promise."""
        return self._sweep_single("ovspatialquery_sweep_sphere_any_sync", query, SphereSweepResult)

    def sweep_sphere_multiple(
        self, query: SphereSweepQuery, capacity: SupportsIndex
    ) -> MultipleResult:
        """Stores up to `capacity` hits, one per object reached; `truncated` says more existed."""
        return self._multiple(
            "ovspatialquery_sweep_sphere_multiple_sync",
            query,
            capacity,
            raw.SphereSweepHit,
            raw.SphereSweepHitBuffer,
            _sweep_hit,
        )

    def sweep_box_closest(self, query: BoxSweepQuery) -> BoxSweepResult:
        """Runs one fixed-orientation box sweep and returns its closest hit, if any."""
        return self._sweep_single("ovspatialquery_sweep_box_closest_sync", query, BoxSweepResult)

    def sweep_box_any(self, query: BoxSweepQuery) -> BoxSweepResult:
        """Returns a hit of any object the box reaches, if any; no closest promise."""
        return self._sweep_single("ovspatialquery_sweep_box_any_sync", query, BoxSweepResult)

    def sweep_box_multiple(self, query: BoxSweepQuery, capacity: SupportsIndex) -> MultipleResult:
        """Stores up to `capacity` hits, one per object reached; `truncated` says more existed."""
        return self._multiple(
            "ovspatialquery_sweep_box_multiple_sync",
            query,
            capacity,
            raw.BoxSweepHit,
            raw.BoxSweepHitBuffer,
            _sweep_hit,
        )

    def _sweep_single(self, export: str, query: Any, projection: Any) -> Any:
        descriptor = query.to_raw(self.reference)
        output = raw.SphereSweepClosestResult()
        _require(
            self.instance,
            export,
            self.instance.library.exports[export](
                self.instance.handle, ctypes.byref(descriptor), ctypes.byref(output)
            ),
        )
        return projection.from_raw(output)


class Population(_Closeable):
    _teardown_rank = 3

    def __init__(self, instance: Instance, handle: int) -> None:
        self.instance = instance
        self._handle = ctypes.c_uint64(handle)
        instance._own(self)

    @property
    def handle(self) -> int:
        return int(self._handle.value)

    @property
    def target(self) -> Target:
        return Target(self.instance, TargetRef(raw.TargetKind.POPULATION, self.handle))

    def info(self) -> PopulationInfo:
        """Reads current population state."""
        output = raw.PopulationInfo()
        _require(
            self.instance,
            "ovspatialquery_get_population_info",
            self.instance.library.exports["ovspatialquery_get_population_info"](
                self.instance.handle, self.handle, ctypes.byref(output)
            ),
        )
        return PopulationInfo.from_raw(output)

    def close(self) -> None:
        """Destroys a population."""
        if not self._handle.value:
            return
        _require(
            self.instance,
            "ovspatialquery_destroy_population",
            self.instance.library.exports["ovspatialquery_destroy_population"](
                self.instance.handle, self.handle
            ),
        )
        self._handle.value = 0
        self.instance._release(self)


class View(_Closeable):
    _teardown_rank = 2

    def __init__(self, instance: Instance, handle: int, keepalive: Iterable[Any] = ()) -> None:
        self.instance = instance
        self._handle = ctypes.c_uint64(handle)
        self.keepalive = tuple(keepalive)
        instance._own(self)

    @property
    def handle(self) -> int:
        return int(self._handle.value)

    @property
    def target(self) -> Target:
        return Target(self.instance, TargetRef(raw.TargetKind.VIEW, self.handle))

    def close(self) -> None:
        """Destroys a stable view."""
        if not self._handle.value:
            return
        _require(
            self.instance,
            "ovspatialquery_destroy_view",
            self.instance.library.exports["ovspatialquery_destroy_view"](
                self.instance.handle, self.handle
            ),
        )
        self._handle.value = 0
        self.instance._release(self)
        self.keepalive = ()


_FLOAT32 = (raw.K_DL_FLOAT, 32)
_FLOAT64 = (raw.K_DL_FLOAT, 64)
_UINT64 = (raw.K_DL_UINT, 64)
_UINT32 = (raw.K_DL_UINT, 32)
_UINT8 = (raw.K_DL_UINT, 8)
_INT32 = (raw.K_DL_INT, 32)
_DTYPE_NAMES = {
    _FLOAT32: "float32",
    _FLOAT64: "float64",
    _UINT64: "uint64",
    _UINT32: "uint32",
    _UINT8: "uint8",
    _INT32: "int32",
}


@dataclass(frozen=True)
class _Column:
    """One input column: accepted dtypes, its row width (None for one value per row), and the
    name of its row count; columns sharing that name must have the same number of rows."""

    name: str
    dtypes: tuple[tuple[int, int], ...]
    width: int | None = None
    rows: str = "Q"


def _ownership(copy: bool) -> raw.InputOwnership:
    return (
        raw.InputOwnership.COPIED_ON_SUBMISSION
        if copy
        else raw.InputOwnership.BORROWED_UNTIL_TERMINAL
    )


def _cuda_handle(value: Any, label: str, attribute: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        handle = value
    elif label == "cuda_stream" and hasattr(value, "__cuda_stream__"):
        handle = value.__cuda_stream__()[1]
    elif hasattr(value, attribute):
        handle = getattr(value, attribute)
    else:
        raise TypeError(
            f"{label} must be an int handle or expose {attribute}; got {type(value).__name__}"
        )
    if isinstance(handle, bool) or not isinstance(handle, int):
        raise TypeError(f"{label} handle must be an int; got {type(handle).__name__}")
    if handle < 0:
        raise ValueError(f"{label} handle must be nonnegative")
    return int(handle)


def _cuda_sync(
    device: Device, cuda_stream: Any, cuda_wait_event: Any, subject: str
) -> tuple[int | None, int | None, int | None]:
    """Returns the stream and event handles, and the stream to pass to `__dlpack__`. A stream
    of None supplies none (`has_stream` 0); any handle, 0 for CUDA's legacy default stream
    included, is supplied as is."""
    cuda = device.device_type == raw.K_DL_CUDA
    if not cuda and (cuda_stream is not None or cuda_wait_event is not None):
        raise ValueError(f"a CPU {subject} takes no cuda_stream or cuda_wait_event")
    stream = _cuda_handle(cuda_stream, "cuda_stream", "cuda_stream")
    if cuda and stream is None:
        raise ValueError(f"a CUDA {subject} requires the cuda_stream its columns are ready on")
    event = _cuda_handle(cuda_wait_event, "cuda_wait_event", "cuda_event")
    # DLPack spells the legacy default stream 1; CUDA spells it 0.
    return (stream, event, (stream or 1) if cuda else None)


def _import_columns(
    columns: tuple[_Column, ...],
    values: dict[str, Any],
    device: Device,
    where: str,
    stream: int | None,
    ownership: raw.InputOwnership,
) -> tuple[dict[str, _DLPackLease], dict[str, int]]:
    """Imports each present column after checking it against its `_Column` and `device`;
    `where` ends a wrong-device message. Returns the leases and each row count by name."""
    expected = (device.device_type, device.device_id)
    leases: dict[str, _DLPackLease] = {}
    counts: dict[str, tuple[int, str]] = {}
    for column in columns:
        value = values[column.name]
        if value is None:
            continue
        if not hasattr(value, "__dlpack__"):
            raise TypeError(f"{column.name} must support DLPack; got {type(value).__name__}")
        # Check the device before export: a producer may refuse a stream for another device.
        if hasattr(value, "__dlpack_device__"):
            found = tuple(value.__dlpack_device__())
            if found != expected:
                raise ValueError(f"{column.name} is on {Device(*found)}; {where}")
        lease = _DLPackLease.from_dlpack(value, stream=stream, ownership=ownership)
        tensor = lease.array.tensor
        found = (tensor.device.device_type, tensor.device.device_id)
        if found != expected:
            raise ValueError(f"{column.name} is on {Device(*found)}; {where}")
        dtype = tensor.dtype
        if (dtype.code, dtype.bits) not in column.dtypes or dtype.lanes != 1:
            accepted = " or ".join(_DTYPE_NAMES[item] for item in column.dtypes)
            raise ValueError(
                f"{column.name} must be {accepted}; got DLPack dtype code {dtype.code}, "
                f"{dtype.bits} bits, {dtype.lanes} lanes"
            )
        shape = tuple(tensor.shape[: tensor.ndim])
        rows = column.rows if column.width is None else f"{column.rows}, {column.width}"
        if len(shape) != (1 if column.width is None else 2) or (
            column.width is not None and shape[1] != column.width
        ):
            raise ValueError(f"{column.name} must have shape ({rows}); got {shape}")
        count, first = counts.setdefault(column.rows, (shape[0], column.name))
        if shape[0] != count:
            raise ValueError(f"{column.name} has {shape[0]} rows; {first} has {count}")
        leases[column.name] = lease
    return (leases, {name: count for name, (count, _) in counts.items()})


# How long closing a plan or reporter waits for slots retired with close_after_cuda.
_RETIRED_SLOT_WAIT_SECONDS = 10.0


class _SlotOwner(_Closeable):
    """A plan or pair reporter, whose operations hold its result slots. Closing it waits up to
    `_RETIRED_SLOT_WAIT_SECONDS` for slots retired with `close_after_cuda` and released by the
    library once their consumer event completes; every other BUSY raises at once."""

    _teardown_rank = 1
    destroy_export = ""

    def _track_slots(self) -> None:
        self._open_operations: set[Operation] = set()
        self._retired_after_cuda = False

    def close(self) -> None:
        if not self._handle.value:
            return
        destroy = self.instance.library.exports[self.destroy_export]
        result = destroy(self.instance.handle, self._handle.value)
        deadline = time.monotonic() + _RETIRED_SLOT_WAIT_SECONDS
        delay = 0.001
        while (
            result.status == ApiStatus.BUSY
            and self._retired_after_cuda
            and not self._open_operations
            and time.monotonic() < deadline
        ):
            time.sleep(delay)
            delay = min(delay * 2, 0.01)
            result = destroy(self.instance.handle, self._handle.value)
        _require(self.instance, self.destroy_export, result)
        self._handle.value = 0
        self.instance._release(self)


class _Plan(_SlotOwner):
    create_export = ""
    enqueue_export = ""
    batch_type: type[ctypes.Structure]
    columns: tuple[_Column, ...]
    reservation_type: type[ctypes.Structure]
    reservation_projection: Any
    destroy_export = "ovspatialquery_destroy_batch_plan"

    def __init__(
        self, instance: Instance, handle: int, device: Device, *, host_mapped: bool = False
    ) -> None:
        self.instance = instance
        self._handle = ctypes.c_uint64(handle)
        self.device = device
        # A CUDA plan created with HOST_MAPPED keeps its results in pinned host memory, so
        # its operations are retired with close() like a CPU plan's.
        self.host_mapped = host_mapped and device.device_type == raw.K_DL_CUDA
        self._track_slots()
        instance._own(self)

    @classmethod
    def create(cls, instance: Instance, config: Any) -> Any:
        descriptor = config.to_raw()
        output = ctypes.c_uint64()
        _require(
            instance,
            cls.create_export,
            instance.library.exports[cls.create_export](
                instance.handle, ctypes.byref(descriptor), ctypes.byref(output)
            ),
        )
        return cls(
            instance,
            int(output.value),
            config.device,
            host_mapped=bool(config.result_fields & BatchResultField.HOST_MAPPED),
        )

    def _enqueue(
        self,
        values: dict[str, Any],
        ordinal: OrdinalSelection,
        cuda_stream: Any,
        cuda_wait_event: Any,
        copy: bool,
        **scalars: int,
    ) -> tuple[Operation, Any]:
        stream, event, export_stream = _cuda_sync(self.device, cuda_stream, cuda_wait_event, "plan")
        leases, counts = _import_columns(
            self.columns,
            values,
            self.device,
            f"the plan runs on {self.device}",
            export_stream,
            _ownership(copy),
        )
        batch = self.batch_type(
            ordinal=ordinal.to_raw(),
            query_count=counts.get("Q", 0),
            cuda_sync=raw.CudaSync(int(stream is not None), stream or 0, event or 0),
            **{name: ctypes.pointer(lease.array) for name, lease in leases.items()},
            **scalars,
        )
        reservation = self.reservation_type()
        result = self.instance.library.exports[self.enqueue_export](
            self.instance.handle,
            self._handle.value,
            ctypes.byref(batch),
            ctypes.byref(reservation),
        )
        op_id = _require_enqueue(self.instance, self.enqueue_export, result)
        # A borrowed column stays alive until the operation is closed; the library has
        # already copied a copied one. The stream and event must outlive the operation.
        retained = () if copy else tuple(leases.values())
        operation = Operation(
            self.instance,
            op_id,
            keepalive=(reservation, *retained, cuda_stream, cuda_wait_event),
            name=self.enqueue_export,
            cuda=self.device.device_type == raw.K_DL_CUDA and not self.host_mapped,
            owner=self,
        )
        return (operation, operation._project(self.reservation_projection, reservation))

    def close(self) -> None:
        """Destroys an idle plan. BUSY means an operation still owns a result slot; a slot retired
        with `close_after_cuda` is waited for, up to 10 seconds."""
        super().close()


class RaycastPlan(_Plan):
    create_export = "ovspatialquery_create_raycast_plan"
    enqueue_export = "ovspatialquery_enqueue_raycast_plan"
    batch_type = raw.RaycastBatch
    columns = (
        _Column("origins", (_FLOAT64,), 3),
        _Column("unit_directions", (_FLOAT32,), 3),
        _Column("max_distances", (_FLOAT64,)),
        _Column("min_distances", (_FLOAT64,)),
        _Column("query_bits", (_UINT64,), 2),
        _Column("maximum_hits", (_UINT64,)),
    )
    reservation_type = raw.RaycastBatchReservation
    reservation_projection = RaycastBatchReservation

    def enqueue(
        self,
        *,
        origins: Any,
        unit_directions: Any,
        max_distances: Any,
        min_distances: Any = None,
        query_bits: Any = None,
        maximum_hits: Any = None,
        query_flags: QueryFlag = QueryFlag(0),
        both_sides: bool = False,
        ordinal: OrdinalSelection = OrdinalSelection.head(),
        cuda_stream: Any = None,
        cuda_wait_event: Any = None,
        copy: bool = False,
    ) -> tuple[Operation, RaycastBatchReservation]:
        """Enqueues one launch of DLPack columns. Without `copy`, the operation borrows each
        column until it is closed, and the caller must not write to it before then.

        `query_flags` applies to every row of the launch; `both_sides=True` adds
        `QueryFlag.MESH_BOTH_SIDES`. Rows that need different options go in separate launches."""
        return self._enqueue(
            {
                "origins": origins,
                "unit_directions": unit_directions,
                "max_distances": max_distances,
                "min_distances": min_distances,
                "query_bits": query_bits,
                "maximum_hits": maximum_hits,
            },
            ordinal,
            cuda_stream,
            cuda_wait_event,
            copy,
            query_flags=_query_flags(query_flags, both_sides),
        )


class SphereSweepPlan(_Plan):
    create_export = "ovspatialquery_create_sphere_sweep_plan"
    enqueue_export = "ovspatialquery_enqueue_sphere_sweep_plan"
    batch_type = raw.SphereSweepBatch
    columns = (
        _Column("centers", (_FLOAT64,), 3),
        _Column("unit_directions", (_FLOAT32,), 3),
        _Column("max_distances", (_FLOAT64,)),
        _Column("radii", (_FLOAT64,)),
        _Column("min_distances", (_FLOAT64,)),
        _Column("query_bits", (_UINT64,), 2),
    )
    reservation_type = raw.SphereSweepBatchReservation
    reservation_projection = SphereSweepBatchReservation

    def enqueue(
        self,
        *,
        centers: Any,
        unit_directions: Any,
        max_distances: Any,
        radii: Any = None,
        min_distances: Any = None,
        query_bits: Any = None,
        query_flags: QueryFlag = QueryFlag(0),
        both_sides: bool = False,
        ordinal: OrdinalSelection = OrdinalSelection.head(),
        cuda_stream: Any = None,
        cuda_wait_event: Any = None,
        copy: bool = False,
    ) -> tuple[Operation, SphereSweepBatchReservation]:
        """Enqueues one launch of DLPack columns; lifetimes, `query_flags` and `both_sides` follow
        `RaycastPlan.enqueue`."""
        return self._enqueue(
            {
                "centers": centers,
                "unit_directions": unit_directions,
                "max_distances": max_distances,
                "radii": radii,
                "min_distances": min_distances,
                "query_bits": query_bits,
            },
            ordinal,
            cuda_stream,
            cuda_wait_event,
            copy,
            query_flags=_query_flags(query_flags, both_sides),
        )


class BoxSweepPlan(_Plan):
    create_export = "ovspatialquery_create_box_sweep_plan"
    enqueue_export = "ovspatialquery_enqueue_box_sweep_plan"
    batch_type = raw.BoxSweepBatch
    columns = (
        _Column("centers", (_FLOAT64,), 3),
        _Column("rotations_xyzw", (_FLOAT32,), 4),
        _Column("unit_directions", (_FLOAT32,), 3),
        _Column("max_distances", (_FLOAT64,)),
        _Column("half_extents", (_FLOAT64,), 3),
        _Column("min_distances", (_FLOAT64,)),
        _Column("query_bits", (_UINT64,), 2),
    )
    reservation_type = raw.BoxSweepBatchReservation
    reservation_projection = BoxSweepBatchReservation

    def enqueue(
        self,
        *,
        centers: Any,
        rotations_xyzw: Any,
        unit_directions: Any,
        max_distances: Any,
        half_extents: Any = None,
        min_distances: Any = None,
        query_bits: Any = None,
        query_flags: QueryFlag = QueryFlag(0),
        both_sides: bool = False,
        ordinal: OrdinalSelection = OrdinalSelection.head(),
        cuda_stream: Any = None,
        cuda_wait_event: Any = None,
        copy: bool = False,
    ) -> tuple[Operation, BoxSweepBatchReservation]:
        """Enqueues one launch of DLPack columns; lifetimes, `query_flags` and `both_sides` follow
        `RaycastPlan.enqueue`."""
        return self._enqueue(
            {
                "centers": centers,
                "rotations_xyzw": rotations_xyzw,
                "unit_directions": unit_directions,
                "max_distances": max_distances,
                "half_extents": half_extents,
                "min_distances": min_distances,
                "query_bits": query_bits,
            },
            ordinal,
            cuda_stream,
            cuda_wait_event,
            copy,
            query_flags=_query_flags(query_flags, both_sides),
        )


class OverlapPlan(_Plan):
    create_export = "ovspatialquery_create_overlap_plan"
    enqueue_export = "ovspatialquery_enqueue_overlap_plan"
    batch_type = raw.OverlapBatch
    columns = (
        _Column("translations", (_FLOAT64,), 3),
        _Column("rotations_xyzw", (_FLOAT32, _FLOAT64), 4),
        _Column("half_extents", (_FLOAT64,), 3),
        _Column("radii", (_FLOAT64,)),
        _Column("maximum_hits", (_UINT64,)),
        _Column("query_bits", (_UINT64,), 2),
    )
    reservation_type = raw.OverlapBatchReservation
    reservation_projection = OverlapBatchReservation

    def enqueue(
        self,
        *,
        translations: Any,
        rotations_xyzw: Any = None,
        half_extents: Any = None,
        radii: Any = None,
        maximum_hits: Any = None,
        query_bits: Any = None,
        ordinal: OrdinalSelection = OrdinalSelection.head(),
        cuda_stream: Any = None,
        cuda_wait_event: Any = None,
        copy: bool = False,
    ) -> tuple[Operation, OverlapBatchReservation]:
        """Enqueues one launch of DLPack columns; lifetimes follow `RaycastPlan.enqueue`."""
        return self._enqueue(
            {
                "translations": translations,
                "rotations_xyzw": rotations_xyzw,
                "half_extents": half_extents,
                "radii": radii,
                "maximum_hits": maximum_hits,
                "query_bits": query_bits,
            },
            ordinal,
            cuda_stream,
            cuda_wait_event,
            copy,
        )


class PairReporter(_SlotOwner):
    """Forward-only AABB-pair reporter. Experimental: may change in any minor release."""

    destroy_export = "ovspatialquery_destroy_pair_reporter"

    def __init__(self, instance: Instance, handle: int, result_device: Device = Device()) -> None:
        self.instance = instance
        self._handle = ctypes.c_uint64(handle)
        self.result_device = result_device
        self._track_slots()
        instance._own(self)

    @classmethod
    def create(cls, instance: Instance, config: PairReporterConfig) -> "PairReporter":
        """Creates an empty-baseline, forward-only AABB-pair reporter."""
        descriptor = config.to_raw()
        output = ctypes.c_uint64()
        _require(
            instance,
            "ovspatialquery_create_pair_reporter",
            instance.library.exports["ovspatialquery_create_pair_reporter"](
                instance.handle, ctypes.byref(descriptor), ctypes.byref(output)
            ),
        )
        return cls(instance, int(output.value), config.result_device)

    def advance(
        self,
        first: OrdinalSelection = OrdinalSelection.head(),
        second: OrdinalSelection = OrdinalSelection.head(),
    ) -> tuple[Operation, PairReporterReservation]:
        """Admits one forward advance of the reporter."""
        descriptor = raw.PairReporterAdvance(first.to_raw(), second.to_raw())
        reservation = raw.PairReporterReservation()
        result = self.instance.library.exports["ovspatialquery_enqueue_pair_reporter_advance"](
            self.instance.handle,
            self._handle.value,
            ctypes.byref(descriptor),
            ctypes.byref(reservation),
        )
        op_id = _require_enqueue(
            self.instance, "ovspatialquery_enqueue_pair_reporter_advance", result
        )
        operation = Operation(
            self.instance,
            op_id,
            keepalive=(descriptor, reservation),
            name="ovspatialquery_enqueue_pair_reporter_advance",
            cuda=self.result_device.device_type == raw.K_DL_CUDA,
            owner=self,
        )
        return (operation, operation._project(PairReporterReservation, reservation))

    def close(self) -> None:
        """Destroys an idle reporter. BUSY means an advance operation is still live; a slot
        retired with `close_after_cuda` is waited for, up to 10 seconds."""
        super().close()


__all__ = [
    "ApiError",
    "BoxSweepPlan",
    "Instance",
    "Operation",
    "OverlapPlan",
    "PairReporter",
    "Population",
    "RaycastPlan",
    "SphereSweepPlan",
    "Target",
    "View",
    "_decode_string",
    "_require",
]
