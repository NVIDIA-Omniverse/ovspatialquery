# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy, version-qualified projection of the current API Ovstage companion."""

from __future__ import annotations
import ctypes
import re
from typing import Any, Iterable, Mapping
from . import raw
from .raw import Preparation
from .instance import (
    Instance,
    Operation,
    Population,
    View,
    _cuda_sync,
    _require,
    _require_enqueue,
    _supported,
)
from .types import Device, PopulationConfig, _participation

SUPPORTED_OVSTAGE_RELEASE = "0.3.x, 0.4.x"


class OvstageUnavailable(RuntimeError):
    """The installed companion or source object cannot be qualified."""


def native_stage_pointer(ovstage_module: Any, stage: Any) -> int:
    """Return a live instance from a supported Ovstage release."""
    version = getattr(ovstage_module, "__version__", None)
    if not isinstance(version, str) or not re.fullmatch(
        r"0\.[34]\.\d+(?:\.\d+)*(?:(?:\.dev|rc)\d+)?(?:[+][a-zA-Z0-9.-]+)?", version
    ):
        raise OvstageUnavailable(
            f"Ovstage companion requires a supported release ({SUPPORTED_OVSTAGE_RELEASE}); observed {version!r}"
        )
    # 0.4 releases order ordinals as int64; pre-release 0.4 builds kept 0.3's uint64, and a
    # version string cannot tell them apart, so the bindings' ordinal type decides.
    bindings = getattr(getattr(ovstage_module, "_src", None), "bindings", None)
    if (
        version.startswith("0.4.")
        and getattr(bindings, "ovstage_ordinal_t", None) is not ctypes.c_int64
    ):
        raise OvstageUnavailable(
            f"Ovstage {version} is a pre-release 0.4 build with unsigned ordinals; "
            f"the companion requires a 0.4 release ({SUPPORTED_OVSTAGE_RELEASE})"
        )
    try:
        required = stage._require_inst()
        stored = stage._inst
        required_value = ctypes.cast(required, ctypes.c_void_p).value
        stored_value = ctypes.cast(stored, ctypes.c_void_p).value
    except (AttributeError, TypeError, ValueError) as failure:
        raise OvstageUnavailable(
            "Ovstage does not expose the expected private instance"
        ) from failure
    if not required_value or required_value != stored_value:
        raise OvstageUnavailable("Ovstage returned an invalid native instance")
    return int(required_value)


class OvstageAdapter:
    def __init__(self, instance: Instance, ovstage_module: Any, stage: Any) -> None:
        self.instance = instance
        self.ovstage_module = ovstage_module
        self.stage = stage
        self.stage_pointer = native_stage_pointer(ovstage_module, stage)
        self.exports = instance.library.require_ovstage()

    def _filter(self, filter: Any) -> tuple[raw.OvstageFilter, Any]:
        """Return the C view of an ovstage filter and the storage it borrows."""
        if not isinstance(filter, self.ovstage_module.Filter):
            raise TypeError(
                f"expected {self.ovstage_module.__name__}.Filter; got {type(filter).__name__}"
            )
        value, storage = filter.to_c()
        return raw.OvstageFilter(
            ctypes.cast(value.predicates, ctypes.c_void_p), int(value.count)
        ), storage

    def _path_value(self, path: Any, dictionary: Any, what: str = "prim paths") -> tuple[int, Any]:
        """Return a prim path's value in the stage's path dictionary and the dictionary used.
        Ovstage 0.3 interns a path as an int; 0.4 as a `PrimPath` whose `raw_id` is the value."""
        path = getattr(path, "raw_id", path)
        if isinstance(path, bool) or not isinstance(path, (int, str)):
            raise TypeError(f"{what} are prim path strings, PrimPaths or int tokens; got {path!r}")
        if isinstance(path, str):
            if dictionary is None:
                dictionary = self.ovstage_module.PathDictionary(self.stage)
            interned = dictionary.intern_path(path)
            path = int(getattr(interned, "raw_id", interned))
        if path < 0 or path > (1 << 64) - 1:
            raise ValueError(f"{what} must fit uint64")
        return path, dictionary

    def _scope(
        self, include_paths: Iterable[int | str], exclude_paths: Iterable[int | str]
    ) -> tuple[raw.OvstageScope, Any]:
        """Return the C scope and the arrays it borrows, interning path strings."""
        dictionary = None
        lists: list[list[int]] = []
        for paths in (include_paths, exclude_paths):
            if isinstance(paths, (str, bytes)):
                raise TypeError("include_paths and exclude_paths are sequences of prim paths")
            values: list[int] = []
            for path in paths:
                value, dictionary = self._path_value(path, dictionary)
                values.append(value)
            if len(values) > 0xFFFFFFFF:
                raise ValueError("scope path counts must fit uint32")
            lists.append(values)
        arrays = [(ctypes.c_uint64 * len(values))(*values) if values else None for values in lists]
        scope = raw.OvstageScope(arrays[0], arrays[1], len(lists[0]), len(lists[1]))
        return scope, arrays

    def supports(self, capability: raw.Capability, device: Device = Device()) -> bool:
        """`Instance.supports` for populations of this stage: PREPARE_ON_DEVICE also needs the
        stage's placement extension."""
        return _supported(
            self.instance,
            "ovspatialquery_query_ovstage_capability",
            self.exports["ovspatialquery_query_ovstage_capability"](
                self.instance.handle, self.stage_pointer, device.to_raw(), int(capability)
            ),
        )

    def create_population(
        self,
        filter: Any,
        config: PopulationConfig = PopulationConfig(),
        *,
        include_paths: Iterable[int | str] = (),
        exclude_paths: Iterable[int | str] = (),
    ) -> "OvstagePopulation":
        """Creates a NOT_READY population with no head and deep-copies the filter and the scope.
        The population holds the prims the filter selects within the scope: the nearest listed
        ancestor of a prim decides, an exclude wins at equal depth, and an empty include list
        stands for "/". Each path is a prim path string or its int path value."""
        filter_value, storage = self._filter(filter)
        scope, scope_storage = self._scope(include_paths, exclude_paths)
        descriptor = raw.OvstagePopulationDesc(
            self.stage_pointer, filter_value, scope, config.to_raw()
        )
        output = ctypes.c_uint64()
        _require(
            self.instance,
            "ovspatialquery_create_ovstage_population",
            self.exports["ovspatialquery_create_ovstage_population"](
                self.instance.handle, ctypes.byref(descriptor), ctypes.byref(output)
            ),
        )
        del storage, scope_storage
        return OvstagePopulation(self, int(output.value))


class OvstagePopulation(Population):
    def __init__(self, adapter: OvstageAdapter, handle: int) -> None:
        super().__init__(adapter.instance, handle)
        self.adapter = adapter

    def enqueue_set(
        self,
        target_ordinal: int,
        *,
        preparation: Preparation = Preparation.ON_HOST,
        device: Device | None = None,
        cuda_stream: Any = None,
        cuda_wait_event: Any = None,
    ) -> Operation:
        """Captures one sealed source snapshot before successful admission returns. With
        `preparation` ON_DEVICE or ON_DEVICE_HOST_READABLE the version is prepared on the CUDA
        `device`: poses are read on the device on `cuda_stream`, behind `cuda_wait_event` when
        given. Device preparation is experimental and may change in any minor release."""
        if target_ordinal < 0:
            raise ValueError("target ordinal must be nonnegative")
        execution = _execution(preparation, device, cuda_stream, cuda_wait_event)
        if execution is None:
            export = "ovspatialquery_enqueue_ovstage_population_set"
            result = self.adapter.exports[export](self.instance.handle, self.handle, target_ordinal)
        else:
            export = "ovspatialquery_enqueue_ovstage_population_set_with_execution"
            result = self.adapter.exports[export](
                self.instance.handle, self.handle, target_ordinal, ctypes.byref(execution)
            )
        return Operation(
            self.instance, _require_enqueue(self.instance, export, result), name=export
        )

    def enqueue_update(
        self,
        base_ordinal: int,
        target_ordinal: int,
        *,
        preparation: Preparation = Preparation.ON_HOST,
        device: Device | None = None,
        cuda_stream: Any = None,
        cuda_wait_event: Any = None,
    ) -> Operation:
        """Reacquires one complete target snapshot with the same synchronous capture as
        `enqueue_set`, prepared where `preparation` says."""
        if base_ordinal < 0 or target_ordinal < 0:
            raise ValueError("publication ordinals must be nonnegative")
        execution = _execution(preparation, device, cuda_stream, cuda_wait_event)
        if execution is None:
            export = "ovspatialquery_enqueue_ovstage_population_update"
            result = self.adapter.exports[export](
                self.instance.handle, self.handle, base_ordinal, target_ordinal
            )
        else:
            export = "ovspatialquery_enqueue_ovstage_population_update_with_execution"
            result = self.adapter.exports[export](
                self.instance.handle,
                self.handle,
                base_ordinal,
                target_ordinal,
                ctypes.byref(execution),
            )
        return Operation(
            self.instance, _require_enqueue(self.instance, export, result), name=export
        )

    def create_view(
        self,
        filter: Any,
        *,
        include_paths: Iterable[int | str] = (),
        exclude_paths: Iterable[int | str] = (),
    ) -> View:
        """Creates a filter view, narrowed by a path scope like `create_population`'s. It becomes
        ready on the next successful parent publication and then shares all parent versions and
        acceleration data.
        """
        filter_value, storage = self.adapter._filter(filter)
        scope, scope_storage = self.adapter._scope(include_paths, exclude_paths)
        descriptor = raw.OvstageViewDesc(filter_value, scope)
        output = ctypes.c_uint64()
        _require(
            self.instance,
            "ovspatialquery_create_ovstage_view",
            self.adapter.exports["ovspatialquery_create_ovstage_view"](
                self.instance.handle,
                self.handle,
                ctypes.byref(descriptor),
                ctypes.byref(output),
            ),
        )
        del storage, scope_storage
        return View(self.instance, int(output.value))

    def set_participation_map(
        self, default_bits: int, bits_by_path: Mapping[int | str, int]
    ) -> None:
        """Give each listed prim path its bits and every other object `default_bits`. A key is a
        prim path string or its path token (an object ID's value); bits are ints below 2**128."""
        dictionary = None
        paths: list[int] = []
        for path in bits_by_path:
            value, dictionary = self.adapter._path_value(path, dictionary, "participation keys")
            paths.append(value)
        bits = [_participation(value) for value in bits_by_path.values()]
        count = len(paths)
        if count > 0xFFFFFFFF:
            raise ValueError("participation table entry count must fit uint32")
        descriptor = raw.OvstageParticipationTable(
            _participation(default_bits, "default bits"),
            (ctypes.c_uint64 * count)(*paths) if count else None,
            (raw.ParticipationBits * count)(*bits) if count else None,
            count,
        )
        _require(
            self.instance,
            "ovspatialquery_set_ovstage_participation_map",
            self.adapter.exports["ovspatialquery_set_ovstage_participation_map"](
                self.instance.handle, self.handle, ctypes.byref(descriptor)
            ),
        )

    def clear_participation_map(self) -> None:
        _require(
            self.instance,
            "ovspatialquery_set_ovstage_participation_map",
            self.adapter.exports["ovspatialquery_set_ovstage_participation_map"](
                self.instance.handle, self.handle, None
            ),
        )


__all__ = [
    "OvstageAdapter",
    "OvstagePopulation",
    "OvstageUnavailable",
    "SUPPORTED_OVSTAGE_RELEASE",
    "native_stage_pointer",
]


def _execution(
    preparation: Preparation, device: Device | None, cuda_stream: Any, cuda_wait_event: Any
) -> raw.DynamicExecution | None:
    """None for the host call; otherwise the execution of a device-prepared publication."""
    preparation = Preparation(preparation)
    if preparation is Preparation.ON_HOST:
        if device is not None or cuda_stream is not None or cuda_wait_event is not None:
            raise ValueError("a host-prepared Ovstage publication takes no device, stream or event")
        return None
    if device is None or device.device_type != raw.K_DL_CUDA:
        raise ValueError(f"preparation {preparation.name} requires a CUDA device")
    stream, event, _ = _cuda_sync(device, cuda_stream, cuda_wait_event, "publication")
    return raw.DynamicExecution(
        device.to_raw(),
        raw.CudaSync(int(stream is not None), stream or 0, event or 0),
        int(preparation),
    )
