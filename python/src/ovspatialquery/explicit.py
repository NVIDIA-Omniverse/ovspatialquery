# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projection of the explicit-ingestion adapter: definitions and populations from DLPack columns."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

from . import raw
from .raw import ExplicitUpdateKind, GeometryType, Preparation
from .instance import (
    _FLOAT32,
    _FLOAT64,
    _INT32,
    _UINT8,
    _UINT32,
    _UINT64,
    Instance,
    Operation,
    Population,
    View,
    _Column,
    _cuda_sync,
    _import_columns,
    _ownership,
    _require,
    _require_enqueue,
)
from .types import Device, PopulationConfig, _DLPackLease

_REAL = (_FLOAT32, _FLOAT64)
_INSTANCE_COLUMNS = (
    _Column("object_values", (_UINT64,), rows="N"),
    _Column("geometry_types", (_INT32,), rows="N"),
    _Column("geometry_values", (_UINT64,), rows="N"),
    _Column("translations", _REAL, 3, "N"),
    _Column("rotations_xyzw", _REAL, 4, "N"),
    _Column("uniform_scales", _REAL, rows="N"),
    _Column("participation_bits", (_UINT64,), 2, rows="N"),
    _Column("material_ids", (_UINT64,), rows="N"),
)
_REMOVED_COLUMN = _Column("removed_object_values", (_UINT64,), rows="M")
_DEFINITION_VALUES = _Column("definition_values", (_UINT64,), rows="G")
_SPHERE_COLUMNS = (_DEFINITION_VALUES, _Column("radii", (_FLOAT64,), rows="G"))
_BOX_COLUMNS = (_DEFINITION_VALUES, _Column("half_extents", (_FLOAT64,), 3, "G"))
_CAPSULE_COLUMNS = (
    _DEFINITION_VALUES,
    _Column("radii", (_FLOAT64,), rows="G"),
    _Column("half_segments", (_FLOAT64,), rows="G"),
)
_CONVEX_TOPOLOGY = (
    "face_offsets",
    "face_counts",
    "face_vertex_offsets",
    "face_vertex_counts",
    "face_vertex_indices",
)
_CONVEX_COLUMNS = (
    _DEFINITION_VALUES,
    _Column("vertex_offsets", (_UINT64,), rows="G"),
    _Column("vertex_counts", (_UINT64,), rows="G"),
    _Column("face_offsets", (_UINT64,), rows="G"),
    _Column("face_counts", (_UINT64,), rows="G"),
    _Column("vertices", (_FLOAT32,), 3, "V"),
    _Column("face_vertex_offsets", (_UINT64,), rows="F"),
    _Column("face_vertex_counts", (_UINT32,), rows="F"),
    _Column("face_vertex_indices", (_UINT32,), rows="I"),
)
_TRIANGLE_MESH_COLUMNS = (
    _DEFINITION_VALUES,
    _Column("vertex_offsets", (_UINT64,), rows="G"),
    _Column("vertex_counts", (_UINT64,), rows="G"),
    _Column("triangle_offsets", (_UINT64,), rows="G"),
    _Column("triangle_counts", (_UINT64,), rows="G"),
    _Column("vertices", (_FLOAT32,), 3, "V"),
    _Column("triangle_indices", (_UINT32,), 3, "T"),
    _Column("double_sided", (_UINT8,), rows="G"),
    _Column("source_face_material_ids", (_UINT64,), rows="T"),
    _Column("source_face_indices", (_UINT64,), rows="T"),
    _Column("source_face_corner_indices", (_UINT32,), 3, "T"),
)


def _pointers(leases: dict[str, _DLPackLease]) -> dict[str, Any]:
    return {name: ctypes.pointer(lease.array) for name, lease in leases.items()}


def _columns_device(columns: tuple[_Column, ...], values: dict[str, Any]) -> tuple[Device, str]:
    """The device of the first present column, which every other column must share."""
    first = next(
        (column.name for column in columns if values[column.name] is not None), columns[0].name
    )
    value = values[first]
    if hasattr(value, "__dlpack_device__"):
        return (Device(*value.__dlpack_device__()), first)
    return (Device.cpu(), first)


def _uint64_values(lease: _DLPackLease) -> list[int]:
    """Reads a CPU uint64 column, which the caller may not change while this runs."""
    tensor = lease.array.tensor
    stride = tensor.strides[0] if tensor.strides else 1
    start = (tensor.data or 0) + tensor.byte_offset
    return [
        ctypes.c_uint64.from_address(start + 8 * stride * row).value
        for row in range(tensor.shape[0])
    ]


def _pointer(structure: ctypes.Structure | None) -> Any:
    return None if structure is None else ctypes.pointer(structure)


def _empty_column(device: Device) -> raw.InputArray:
    """A uint64[0] column on `device`, for an update that removes nothing."""
    shape = (ctypes.c_int64 * 1)(0)
    strides = (ctypes.c_int64 * 1)(1)
    tensor = raw.DLTensor(
        None, device.to_raw(), 1, raw.DLDataType(raw.K_DL_UINT, 64, 1), shape, strides, 0
    )
    # ctypes keeps the shape and stride arrays alive with the structure that points at them.
    return raw.InputArray(tensor, int(raw.InputOwnership.COPIED_ON_SUBMISSION))


@dataclass(frozen=True)
class GeometryDefinitionRef:
    geometry_type: GeometryType
    value: int

    def to_raw(self) -> raw.GeometryRef:
        try:
            geometry_type = GeometryType(self.geometry_type)
        except ValueError as failure:
            raise ValueError("geometry definition type must be built in") from failure
        if geometry_type is GeometryType.INVALID:
            raise ValueError("geometry definition type must be built in")
        if self.value <= 0 or self.value > (1 << 64) - 1:
            raise ValueError("geometry definition value must be nonzero uint64")
        return raw.GeometryRef(int(geometry_type), self.value)


class ExplicitAdapter:
    """Registers exact definitions from CPU DLPack columns named as in the definition tables of
    `ovspatialquery_explicit.h`. Without `copy`, the returned operation borrows each column
    until it is closed, and the caller must not write to it before then."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.exports = instance.library.require_explicit()

    def register_spheres(
        self, *, definition_values: Any, radii: Any, copy: bool = False
    ) -> Operation:
        """Captures and prepares an immutable batch of sphere definitions."""
        return self._register(
            "ovspatialquery_enqueue_sphere_registration",
            GeometryType.SPHERE,
            raw.SphereDefinitionTable,
            _SPHERE_COLUMNS,
            {"definition_values": definition_values, "radii": radii},
            copy,
        )

    def register_boxes(
        self, *, definition_values: Any, half_extents: Any, copy: bool = False
    ) -> Operation:
        """Captures and prepares an immutable batch of box definitions."""
        return self._register(
            "ovspatialquery_enqueue_box_registration",
            GeometryType.BOX,
            raw.BoxDefinitionTable,
            _BOX_COLUMNS,
            {"definition_values": definition_values, "half_extents": half_extents},
            copy,
        )

    def register_capsules(
        self, *, definition_values: Any, radii: Any, half_segments: Any, copy: bool = False
    ) -> Operation:
        """Captures and prepares an immutable batch of capsule definitions."""
        return self._register(
            "ovspatialquery_enqueue_capsule_registration",
            GeometryType.CAPSULE,
            raw.CapsuleDefinitionTable,
            _CAPSULE_COLUMNS,
            {
                "definition_values": definition_values,
                "radii": radii,
                "half_segments": half_segments,
            },
            copy,
        )

    def register_convexes(
        self,
        *,
        definition_values: Any,
        vertex_offsets: Any,
        vertex_counts: Any,
        vertices: Any,
        face_offsets: Any = None,
        face_counts: Any = None,
        face_vertex_offsets: Any = None,
        face_vertex_counts: Any = None,
        face_vertex_indices: Any = None,
        enclosure_tolerance: float = 0.0,
        hull_vertex_limit: int = 0,
        copy: bool = False,
    ) -> Operation:
        """Captures and prepares immutable convex definitions. The five face columns describe
        source topology and come all together or not at all.

        Both options apply to every definition in the call. `hull_vertex_limit` zero registers
        each definition's own hull; 8 to 64 registers the hull PhysX cooks for a convexHull
        collider under that limit, whose corners need not be points and which may leave points
        outside. That hull is cooked from the vertices as given and placed afterwards by each
        instance's pose and `uniform_scales`, as the simulation places a collider: register a
        stage's collider from its local points in stage units and place it with the prim's
        rotation, and its translation and uniform scale times the stage's meters per unit, because
        vertices with the placement already applied can give another hull.
        `enclosure_tolerance` is how far outside the own hull, as a fraction of the
        largest half extent of the points' box, a point may lie and still count as enclosed:
        zero is 2^-20, other values lie from 2^-20 to 2^-16.
        """
        values = {
            "definition_values": definition_values,
            "vertex_offsets": vertex_offsets,
            "vertex_counts": vertex_counts,
            "face_offsets": face_offsets,
            "face_counts": face_counts,
            "vertices": vertices,
            "face_vertex_offsets": face_vertex_offsets,
            "face_vertex_counts": face_vertex_counts,
            "face_vertex_indices": face_vertex_indices,
        }
        present = [values[name] is not None for name in _CONVEX_TOPOLOGY]
        if any(present) and not all(present):
            raise ValueError(f"convex topology columns {', '.join(_CONVEX_TOPOLOGY)} come together")
        return self._register(
            "ovspatialquery_enqueue_convex_registration",
            GeometryType.CONVEX,
            raw.ConvexDefinitionTable,
            _CONVEX_COLUMNS,
            values,
            copy,
            enclosure_tolerance=float(enclosure_tolerance),
            hull_vertex_limit=int(hull_vertex_limit),
        )

    def register_triangle_meshes(
        self,
        *,
        definition_values: Any,
        vertex_offsets: Any,
        vertex_counts: Any,
        triangle_offsets: Any,
        triangle_counts: Any,
        vertices: Any,
        triangle_indices: Any,
        double_sided: Any = None,
        source_face_material_ids: Any = None,
        source_face_indices: Any = None,
        source_face_corner_indices: Any = None,
        copy: bool = False,
    ) -> Operation:
        """Captures and prepares an immutable batch of triangle-mesh definitions.
        `source_face_material_ids` gives each triangle's material; zero lets it report its
        instance's material. `source_face_indices` and `source_face_corner_indices`, which come
        together, give each triangle the source face and the corners within it that its hits
        report; the triangles of one face then carry one material."""
        if (source_face_indices is None) != (source_face_corner_indices is None):
            raise ValueError("source_face_indices and source_face_corner_indices come together")
        return self._register(
            "ovspatialquery_enqueue_triangle_mesh_registration",
            GeometryType.TRIANGLE_MESH,
            raw.TriangleMeshDefinitionTable,
            _TRIANGLE_MESH_COLUMNS,
            {
                "definition_values": definition_values,
                "vertex_offsets": vertex_offsets,
                "vertex_counts": vertex_counts,
                "triangle_offsets": triangle_offsets,
                "triangle_counts": triangle_counts,
                "vertices": vertices,
                "triangle_indices": triangle_indices,
                "double_sided": double_sided,
                "source_face_material_ids": source_face_material_ids,
                "source_face_indices": source_face_indices,
                "source_face_corner_indices": source_face_corner_indices,
            },
            copy,
        )

    def _register(
        self,
        export: str,
        geometry_type: GeometryType,
        table_type: type[ctypes.Structure],
        columns: tuple[_Column, ...],
        values: dict[str, Any],
        copy: bool,
        **scalars: float | int,
    ) -> Operation:
        leases, _ = _import_columns(
            columns,
            values,
            Device.cpu(),
            "a registration reads CPU columns",
            None,
            _ownership(copy),
        )
        descriptor = table_type(**_pointers(leases), **scalars)
        result = self.exports[export](self.instance.handle, ctypes.byref(descriptor))
        op_id = _require_enqueue(self.instance, export, result)
        retained = () if copy else tuple(leases.values())
        operation = Operation(self.instance, op_id, keepalive=(descriptor, *retained), name=export)
        # The instance owns each definition once its registration is seen to complete.
        definitions = [
            GeometryDefinitionRef(geometry_type, value)
            for value in _uint64_values(leases["definition_values"])
        ]

        def own() -> None:
            for definition in definitions:
                _Definition(self, definition)

        operation._on_completed = own
        return operation

    def create_population(
        self, config: PopulationConfig = PopulationConfig()
    ) -> "ExplicitPopulation":
        """Creates a NOT_READY population with no head. Its first complete set may publish any
        nonnegative ordinal, including zero.
        """
        descriptor = config.to_raw()
        output = ctypes.c_uint64()
        _require(
            self.instance,
            "ovspatialquery_create_explicit_population",
            self.exports["ovspatialquery_create_explicit_population"](
                self.instance.handle, ctypes.byref(descriptor), ctypes.byref(output)
            ),
        )
        return ExplicitPopulation(self, int(output.value))

    def dropped_triangle_count(self, definition_value: int) -> int:
        """How many triangles registration dropped from a registered triangle-mesh definition
        because they have no area (collinear or repeated corners); zero when none."""
        output = ctypes.c_uint64()
        _require(
            self.instance,
            "ovspatialquery_get_dropped_triangle_count",
            self.exports["ovspatialquery_get_dropped_triangle_count"](
                self.instance.handle, definition_value, ctypes.byref(output)
            ),
        )
        return int(output.value)

    def destroy_definition(self, definition: GeometryDefinitionRef) -> None:
        """Destroys one explicit geometry definition."""
        _require(
            self.instance,
            "ovspatialquery_destroy_geometry_definition",
            self.exports["ovspatialquery_destroy_geometry_definition"](
                self.instance.handle, definition.to_raw()
            ),
        )
        self.instance._release(None, _Definition.key(definition))


class _Definition:
    """A registered definition its instance destroys if the caller has not."""

    _teardown_rank = 4

    def __init__(self, adapter: ExplicitAdapter, definition: GeometryDefinitionRef) -> None:
        self.adapter = adapter
        self.definition = definition
        adapter.instance._own(self, self.key(definition))

    @staticmethod
    def key(definition: GeometryDefinitionRef) -> tuple[str, int, int]:
        return ("definition", int(definition.geometry_type), int(definition.value))

    def _exit(self) -> None:
        self.adapter.destroy_definition(self.definition)


class ExplicitPopulation(Population):
    """Publishes instance rows from DLPack columns named as in
    `ovspatialquery_explicit_instance_table_t`, all on one device. CPU columns are captured on
    the host. CUDA columns require `cuda_stream`, which is forwarded to each column's
    `__dlpack__` and orders the library's reads after the writes queued on it; an optional
    `cuda_wait_event`, recorded once the columns are written, also orders writes from other
    streams. The library prepares the version as `preparation` says; ON_DEVICE and
    ON_DEVICE_HOST_READABLE are experimental and may change in any minor release. Without
    `copy`, the returned operation borrows each column until it is closed, and the caller must
    not write to it before then. With `copy`, the library queues its copy of CUDA columns on
    `cuda_stream` before returning, so a write from another stream must wait for an event the
    caller records there afterwards."""

    def __init__(self, adapter: ExplicitAdapter, handle: int) -> None:
        super().__init__(adapter.instance, handle)
        self.adapter = adapter

    def enqueue_set(
        self,
        target_ordinal: int,
        *,
        object_values: Any,
        geometry_types: Any,
        geometry_values: Any,
        translations: Any,
        rotations_xyzw: Any,
        uniform_scales: Any = None,
        participation_bits: Any = None,
        material_ids: Any = None,
        preparation: Preparation = Preparation.ON_HOST,
        slot_capacity: int = 0,
        cuda_stream: Any = None,
        cuda_wait_event: Any = None,
        copy: bool = False,
    ) -> Operation:
        """`slot_capacity` is meaningful for device-prepared sets only: the
        distinct objects the population may hold over its life; zero means these rows."""
        if target_ordinal < 0:
            raise ValueError("target ordinal must be nonnegative")
        if slot_capacity < 0:
            raise ValueError("slot capacity must be nonnegative")
        device, leases, instances, execution, keepalive = self._capture(
            _INSTANCE_COLUMNS,
            {
                "object_values": object_values,
                "geometry_types": geometry_types,
                "geometry_values": geometry_values,
                "translations": translations,
                "rotations_xyzw": rotations_xyzw,
                "uniform_scales": uniform_scales,
                "participation_bits": participation_bits,
                "material_ids": material_ids,
            },
            preparation,
            cuda_stream,
            cuda_wait_event,
            copy,
        )
        descriptor = raw.ExplicitPopulationSet(
            target_ordinal, instances, _pointer(execution), slot_capacity
        )
        return self._submit("ovspatialquery_enqueue_explicit_population_set", descriptor, keepalive)

    def enqueue_update(
        self,
        base_ordinal: int,
        target_ordinal: int,
        *,
        kind: ExplicitUpdateKind = ExplicitUpdateKind.COMPLETE_ROWS,
        object_values: Any,
        translations: Any,
        rotations_xyzw: Any,
        geometry_types: Any = None,
        geometry_values: Any = None,
        uniform_scales: Any = None,
        participation_bits: Any = None,
        material_ids: Any = None,
        removed_object_values: Any = None,
        preparation: Preparation = Preparation.ON_HOST,
        cuda_stream: Any = None,
        cuda_wait_event: Any = None,
        copy: bool = False,
    ) -> Operation:
        """Captures changed and removed rows against the exact current head, or against the
        newest admitted device-prepared update. COMPLETE_ROWS rows carry both geometry
        columns; POSES rows carry neither, nor `participation_bits`, which POSES_AND_BITS
        rows require. `material_ids` belongs to COMPLETE_ROWS rows; poses rows keep each
        object's material. None removes nothing.
        """
        if base_ordinal < 0:
            raise ValueError("base ordinal must be nonnegative")
        if target_ordinal <= base_ordinal:
            raise ValueError("target ordinal must be strictly greater than base ordinal")
        kind = ExplicitUpdateKind(kind)
        bindings = kind is ExplicitUpdateKind.COMPLETE_ROWS
        for name, value in (
            ("geometry_types", geometry_types),
            ("geometry_values", geometry_values),
        ):
            if (value is None) == bindings:
                raise ValueError(
                    f"a {kind.name} update {'requires' if bindings else 'takes no'} {name}"
                )
        if not bindings and material_ids is not None:
            raise ValueError(f"a {kind.name} update takes no material_ids")
        if kind is ExplicitUpdateKind.POSES and participation_bits is not None:
            raise ValueError("a POSES update takes no participation_bits; use POSES_AND_BITS")
        if kind is ExplicitUpdateKind.POSES_AND_BITS and participation_bits is None:
            raise ValueError("a POSES_AND_BITS update requires participation_bits")
        device, leases, instances, execution, keepalive = self._capture(
            (*_INSTANCE_COLUMNS, _REMOVED_COLUMN),
            {
                "object_values": object_values,
                "geometry_types": geometry_types,
                "geometry_values": geometry_values,
                "translations": translations,
                "rotations_xyzw": rotations_xyzw,
                "uniform_scales": uniform_scales,
                "participation_bits": participation_bits,
                "material_ids": material_ids,
                "removed_object_values": removed_object_values,
            },
            preparation,
            cuda_stream,
            cuda_wait_event,
            copy,
        )
        if "removed_object_values" in leases:
            removed = leases["removed_object_values"].array
        else:
            removed = _empty_column(device)
            keepalive.append(removed)
        descriptor = raw.ExplicitPopulationUpdate(
            base_ordinal,
            target_ordinal,
            int(kind),
            instances,
            ctypes.pointer(removed),
            _pointer(execution),
        )
        return self._submit(
            "ovspatialquery_enqueue_explicit_population_update", descriptor, keepalive
        )

    def _capture(
        self,
        columns: tuple[_Column, ...],
        values: dict[str, Any],
        preparation: Preparation,
        cuda_stream: Any,
        cuda_wait_event: Any,
        copy: bool,
    ) -> tuple[
        Device,
        dict[str, _DLPackLease],
        raw.ExplicitInstanceTable,
        raw.DynamicExecution | None,
        list[Any],
    ]:
        preparation = Preparation(preparation)
        device, first = _columns_device(columns, values)
        cuda = device.device_type == raw.K_DL_CUDA
        if not cuda and preparation is not Preparation.ON_HOST:
            raise ValueError(f"preparation {preparation.name} requires CUDA columns")
        stream, event, export_stream = _cuda_sync(
            device, cuda_stream, cuda_wait_event, "publication"
        )
        leases, _ = _import_columns(
            columns, values, device, f"{first} is on {device}", export_stream, _ownership(copy)
        )
        instances = raw.ExplicitInstanceTable(
            **_pointers(
                {name: lease for name, lease in leases.items() if name != _REMOVED_COLUMN.name}
            )
        )
        execution = (
            raw.DynamicExecution(
                device.to_raw(),
                raw.CudaSync(int(stream is not None), stream or 0, event or 0),
                int(preparation),
            )
            if cuda
            else None
        )
        # A borrowed column stays alive until the operation is closed; the library has
        # already copied a copied one. The stream and event must outlive the operation.
        retained = [] if copy else list(leases.values())
        return (
            device,
            leases,
            instances,
            execution,
            [execution, *retained, cuda_stream, cuda_wait_event],
        )

    def _submit(self, export: str, descriptor: ctypes.Structure, keepalive: list[Any]) -> Operation:
        result = self.adapter.exports[export](
            self.instance.handle, self.handle, ctypes.byref(descriptor)
        )
        op_id = _require_enqueue(self.instance, export, result)
        return Operation(self.instance, op_id, keepalive=(descriptor, *keepalive), name=export)

    def create_view(self, object_ids: Any) -> View:
        """Selects the objects named by a CPU uint64 column, which is copied during creation."""
        leases, _ = _import_columns(
            (_Column("object_ids", (_UINT64,), rows="N"),),
            {"object_ids": object_ids},
            Device.cpu(),
            "a view reads a CPU column",
            None,
            raw.InputOwnership.COPIED_ON_SUBMISSION,
        )
        descriptor = raw.ExplicitViewDesc(ctypes.pointer(leases["object_ids"].array))
        output = ctypes.c_uint64()
        _require(
            self.instance,
            "ovspatialquery_create_explicit_view",
            self.adapter.exports["ovspatialquery_create_explicit_view"](
                self.instance.handle,
                self.handle,
                ctypes.byref(descriptor),
                ctypes.byref(output),
            ),
        )
        return View(self.instance, int(output.value))


__all__ = [
    "ExplicitAdapter",
    "ExplicitPopulation",
    "GeometryDefinitionRef",
]
