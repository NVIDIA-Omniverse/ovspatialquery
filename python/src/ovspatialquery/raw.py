# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Literal ctypes projection of the installed current API C interfaces."""

from __future__ import annotations
import ctypes
import operator
from enum import IntEnum, IntFlag
from typing import Any, Final, NamedTuple


class _Structure(ctypes.Structure):
    """A structure whose integer fields refuse a Python integer outside their C range, which
    ctypes would silently wrap (2**32 into a uint32 field as 0). Every descriptor a caller's
    configuration fills uses it; ctypes sets constructor arguments through `__setattr__` too.
    Per-query and per-launch structures, the ordinal selection among them, skip the cost: the
    wrapper checks the values it fills them with."""

    _integer_bounds_: dict[str, tuple[int, int]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        bounds = {}
        for name, field_type, *_ in cls.__dict__.get("_fields_", ()):
            code = getattr(field_type, "_type_", None)
            if isinstance(code, str) and len(code) == 1 and code in "bBhHiIlLqQ":
                bits = 8 * ctypes.sizeof(field_type)
                signed = code.islower()
                bounds[name] = (
                    (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
                )
        cls._integer_bounds_ = bounds

    def __setattr__(self, name: str, value: Any) -> None:
        bounds = self._integer_bounds_.get(name)
        # ctypes takes an integer field's value through `__index__`, so the check does too.
        if bounds is not None and not isinstance(value, bool) and hasattr(value, "__index__"):
            value = operator.index(value)
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{name} must be from {bounds[0]} to {bounds[1]}; got {value}")
        ctypes.Structure.__setattr__(self, name, value)


class ApiStatus(IntEnum):
    SUCCESS = 0
    ERROR = 1
    TIMEOUT = 2
    INVALID_ARGUMENT = 3
    INVALID_HANDLE = 4
    INVALID_STATE = 5
    NOT_SUPPORTED = 6
    NOT_READY = 7
    BUSY = 8
    STALE_ORDINAL = 9
    ORDINAL_MISMATCH = 10
    ORDINAL_NOT_RETAINED = 11
    BACKWARD_ORDINAL = 12
    OPERATION_CAPACITY = 13
    RESULT_CAPACITY = 14
    HOST_BUDGET = 15
    DEVICE_BUDGET = 16
    OPERATION_FAILED = 17
    OUT_OF_MEMORY = 18
    INTERNAL = 19
    DEVICE_LOST = 20


class GeometryType(IntEnum):
    INVALID = 0
    SPHERE = 2
    BOX = 3
    CAPSULE = 4
    CONVEX = 5
    TRIANGLE_MESH = 6


class InputOwnership(IntEnum):
    BORROWED_UNTIL_TERMINAL = 0
    COPIED_ON_SUBMISSION = 1


class TargetKind(IntEnum):
    POPULATION = 0
    VIEW = 1


class OrdinalSelectionKind(IntEnum):
    HEAD = 0
    RETAINED_EXACT = 1
    ADMITTED_EXACT = 2


class QueryPrecision(IntEnum):
    BOUNDS_CANDIDATE = 0
    EXACT = 1
    EXACT_SINGLE = 2


class HitTruth(IntEnum):
    BOUNDS_CANDIDATE = 0
    EXACT_SURFACE = 1


class GeometryFeatureKind(IntEnum):
    NONE = 0
    TRIANGLE = 1


class OverlapShapeKind(IntEnum):
    SPHERE = 0
    BOX = 1
    AABB = 2


class HitMode(IntEnum):
    CLOSEST = 0
    ANY = 1
    MULTIPLE = 2


class QueryFlag(IntFlag):
    MESH_BOTH_SIDES = 1


class MeshHits(IntEnum):
    ALL = 0
    NEAREST = 1


class BatchRowStatus(IntEnum):
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    NOT_SUPPORTED = 2
    FAILED = 3


class BatchResultField(IntFlag):
    POSITION = 1
    NORMAL = 2
    DISTANCE = 4
    SOURCE_FEATURE = 8
    GEOMETRY_IDENTITY = 16
    QUERY_POSITION = 32
    MATERIAL = 64
    HOST_MAPPED = 128


class PairReporterOutcome(IntEnum):
    ADVANCED = 0
    ACTIVE_CAPACITY = 1
    DELTA_CAPACITY = 2
    FAILED = 3


class OpState(IntEnum):
    PENDING = 0
    COMPLETED = 1
    FAILURE = 2


class OvstageExclusionReason(IntEnum):
    NO_SPATIAL_SOURCE = 1
    UNSUPPORTED_TYPE = 2
    UNSUPPORTED_SHAPE = 3
    MALFORMED_GEOMETRY = 4
    RESHAPE_PENDING = 5


class Preparation(IntEnum):
    ON_HOST = 0
    ON_DEVICE = 1
    ON_DEVICE_HOST_READABLE = 2


class ExplicitUpdateKind(IntEnum):
    COMPLETE_ROWS = 0
    POSES = 1
    POSES_AND_BITS = 2


class Capability(IntEnum):
    RAYCAST_PLAN = 0
    RAYCAST_PLAN_EXACT = 1
    SPHERE_SWEEP_PLAN = 2
    SPHERE_SWEEP_PLAN_EXACT = 3
    BOX_SWEEP_PLAN = 4
    BOX_SWEEP_PLAN_EXACT = 5
    OVERLAP_PLAN = 6
    OVERLAP_PLAN_EXACT = 7
    PREPARE_ON_DEVICE = 8
    PAIR_REPORTER = 9


K_DL_CPU: Final = 1
K_DL_CUDA: Final = 2
K_DL_CUDA_HOST: Final = 3
K_DL_INT: Final = 0
K_DL_UINT: Final = 1
K_DL_FLOAT: Final = 2
INVALID_HANDLE: Final = 0
INVALID_OP: Final = 0
TIMEOUT_INFINITE: Final = (1 << 64) - 1


class DLDevice(_Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class Result(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int)]


class String(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("length", ctypes.c_size_t)]


class EnqueueResult(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int), ("op_id", ctypes.c_uint64)]


class CudaEvent(ctypes.Structure):
    _fields_ = [("value", ctypes.c_size_t)]


class CudaSync(ctypes.Structure):
    _fields_ = [
        ("has_stream", ctypes.c_uint32),
        ("stream", ctypes.c_size_t),
        ("wait_event", ctypes.c_size_t),
    ]


class ObjectId(ctypes.Structure):
    _fields_ = [("value", ctypes.c_uint64)]


class ParticipationBits(ctypes.Structure):
    _fields_ = [("words", ctypes.c_uint64 * 2)]


class GeometryRef(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("value", ctypes.c_uint64)]


class DeviceBudget(_Structure):
    _fields_ = [("device", DLDevice), ("byte_budget", ctypes.c_uint64)]


class InstanceDesc(_Structure):
    _fields_ = [
        ("operation_capacity", ctypes.c_uint32),
        ("max_cpu_concurrency", ctypes.c_uint32),
        ("host_byte_budget", ctypes.c_uint64),
        ("device_budgets", ctypes.POINTER(DeviceBudget)),
        ("device_budget_count", ctypes.c_uint32),
    ]


class PopulationDesc(_Structure):
    _fields_ = [
        ("retained_ordinal_capacity", ctypes.c_uint32),
        ("pending_publication_capacity", ctypes.c_uint32),
    ]


class PopulationInfo(ctypes.Structure):
    _fields_ = [
        ("ready", ctypes.c_uint32),
        ("head_ordinal", ctypes.c_int64),
        ("object_count", ctypes.c_uint64),
        ("retained_ordinal_count", ctypes.c_uint32),
        ("update_pending", ctypes.c_uint32),
    ]


class PopulationUpdateResult(ctypes.Structure):
    _fields_ = [
        ("published_ordinal", ctypes.c_int64),
        ("object_count", ctypes.c_uint64),
        ("excluded_object_count", ctypes.c_uint64),
        ("excluded_objects", DLTensor),
        ("exclusion_reasons", DLTensor),
    ]


class OvstageDroppedTriangles(ctypes.Structure):
    _fields_ = [("objects", DLTensor), ("dropped_triangle_counts", DLTensor)]


class InputArray(ctypes.Structure):
    _fields_ = [("tensor", DLTensor), ("ownership", ctypes.c_int)]


class DynamicExecution(ctypes.Structure):
    _fields_ = [
        ("device", DLDevice),
        ("cuda_sync", CudaSync),
        ("preparation", ctypes.c_int),
    ]


class ExplicitInstanceTable(ctypes.Structure):
    _fields_ = [
        ("object_values", ctypes.POINTER(InputArray)),
        ("geometry_types", ctypes.POINTER(InputArray)),
        ("geometry_values", ctypes.POINTER(InputArray)),
        ("translations", ctypes.POINTER(InputArray)),
        ("rotations_xyzw", ctypes.POINTER(InputArray)),
        ("uniform_scales", ctypes.POINTER(InputArray)),
        ("participation_bits", ctypes.POINTER(InputArray)),
        ("material_ids", ctypes.POINTER(InputArray)),
    ]


class ExplicitPopulationSet(ctypes.Structure):
    _fields_ = [
        ("target_ordinal", ctypes.c_int64),
        ("instances", ExplicitInstanceTable),
        ("dynamic_execution", ctypes.POINTER(DynamicExecution)),
        ("slot_capacity", ctypes.c_uint64),
    ]


class ExplicitPopulationUpdate(ctypes.Structure):
    _fields_ = [
        ("base_ordinal", ctypes.c_int64),
        ("target_ordinal", ctypes.c_int64),
        ("kind", ctypes.c_int),
        ("changed_instances", ExplicitInstanceTable),
        ("removed_object_values", ctypes.POINTER(InputArray)),
        ("dynamic_execution", ctypes.POINTER(DynamicExecution)),
    ]


class ExplicitViewDesc(_Structure):
    _fields_ = [("object_ids", ctypes.POINTER(InputArray))]


class SphereDefinitionTable(ctypes.Structure):
    _fields_ = [
        ("definition_values", ctypes.POINTER(InputArray)),
        ("radii", ctypes.POINTER(InputArray)),
    ]


class BoxDefinitionTable(ctypes.Structure):
    _fields_ = [
        ("definition_values", ctypes.POINTER(InputArray)),
        ("half_extents", ctypes.POINTER(InputArray)),
    ]


class CapsuleDefinitionTable(ctypes.Structure):
    _fields_ = [
        ("definition_values", ctypes.POINTER(InputArray)),
        ("radii", ctypes.POINTER(InputArray)),
        ("half_segments", ctypes.POINTER(InputArray)),
    ]


class ConvexDefinitionTable(_Structure):
    _fields_ = [
        ("definition_values", ctypes.POINTER(InputArray)),
        ("vertex_offsets", ctypes.POINTER(InputArray)),
        ("vertex_counts", ctypes.POINTER(InputArray)),
        ("face_offsets", ctypes.POINTER(InputArray)),
        ("face_counts", ctypes.POINTER(InputArray)),
        ("vertices", ctypes.POINTER(InputArray)),
        ("face_vertex_offsets", ctypes.POINTER(InputArray)),
        ("face_vertex_counts", ctypes.POINTER(InputArray)),
        ("face_vertex_indices", ctypes.POINTER(InputArray)),
        ("enclosure_tolerance", ctypes.c_double),
        ("hull_vertex_limit", ctypes.c_uint32),
    ]


class TriangleMeshDefinitionTable(ctypes.Structure):
    _fields_ = [
        ("definition_values", ctypes.POINTER(InputArray)),
        ("vertex_offsets", ctypes.POINTER(InputArray)),
        ("vertex_counts", ctypes.POINTER(InputArray)),
        ("triangle_offsets", ctypes.POINTER(InputArray)),
        ("triangle_counts", ctypes.POINTER(InputArray)),
        ("vertices", ctypes.POINTER(InputArray)),
        ("triangle_indices", ctypes.POINTER(InputArray)),
        ("double_sided", ctypes.POINTER(InputArray)),
        ("source_face_material_ids", ctypes.POINTER(InputArray)),
        ("source_face_indices", ctypes.POINTER(InputArray)),
        ("source_face_corner_indices", ctypes.POINTER(InputArray)),
    ]


class _TargetHandle(ctypes.Union):
    _fields_ = [("population", ctypes.c_uint64), ("view", ctypes.c_uint64)]


class TargetRef(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("handle", _TargetHandle)]


class OrdinalSelection(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("source_ordinal", ctypes.c_int64)]


class QueryTarget(ctypes.Structure):
    _fields_ = [("target", TargetRef), ("ordinal", OrdinalSelection)]


class QueryCommon(ctypes.Structure):
    _fields_ = [
        ("target", QueryTarget),
        ("precision", ctypes.c_int),
        ("query_bits", ParticipationBits),
        ("contact_tolerance", ctypes.c_double),
        ("query_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_void_p * 2),
    ]


class MultipleResult(ctypes.Structure):
    _fields_ = [
        ("source_ordinal", ctypes.c_int64),
        ("stored_hit_count", ctypes.c_uint64),
        ("truncated", ctypes.c_uint32),
    ]


class HitIdentity(ctypes.Structure):
    _fields_ = [
        ("object_id", ObjectId),
        ("geometry", GeometryRef),
        ("material_id", ctypes.c_uint64),
    ]


class TriangleFeature(ctypes.Structure):
    _fields_ = [
        ("source_face_index", ctypes.c_uint64),
        ("source_face_corner_indices", ctypes.c_uint32 * 3),
        ("barycentrics", ctypes.c_float * 3),
    ]


class _GeometryFeatureData(ctypes.Union):
    _fields_ = [("triangle", TriangleFeature), ("reserved", ctypes.c_uint64 * 6)]


class GeometryFeature(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("data", _GeometryFeatureData)]


class Ray(ctypes.Structure):
    _fields_ = [
        ("origin", ctypes.c_double * 3),
        ("unit_direction", ctypes.c_float * 3),
        ("min_distance", ctypes.c_double),
        ("max_distance", ctypes.c_double),
    ]


class RaycastDesc(ctypes.Structure):
    _fields_ = [("common", QueryCommon), ("ray", Ray), ("mesh_hits", ctypes.c_int)]


class RaycastBoundsCandidate(ctypes.Structure):
    _fields_ = [
        ("identity", HitIdentity),
        ("bounds_position", ctypes.c_double * 3),
        ("bounds_normal", ctypes.c_float * 3),
        ("distance", ctypes.c_double),
    ]


class RaycastExactHit(ctypes.Structure):
    _fields_ = [
        ("identity", HitIdentity),
        ("position", ctypes.c_double * 3),
        ("normal", ctypes.c_float * 3),
        ("distance", ctypes.c_double),
        ("feature", GeometryFeature),
    ]


class _RaycastHitData(ctypes.Union):
    _fields_ = [
        ("bounds_candidate", RaycastBoundsCandidate),
        ("exact", RaycastExactHit),
    ]


class RaycastHit(ctypes.Structure):
    _fields_ = [("truth", ctypes.c_int), ("data", _RaycastHitData)]


class RaycastHitBuffer(ctypes.Structure):
    _fields_ = [("hits", ctypes.POINTER(RaycastHit)), ("capacity", ctypes.c_uint64)]


class RaycastSingleResult(ctypes.Structure):
    _fields_ = [
        ("source_ordinal", ctypes.c_int64),
        ("has_hit", ctypes.c_uint32),
        ("hit", RaycastHit),
    ]


class SphereSweep(ctypes.Structure):
    _fields_ = [
        ("center", ctypes.c_double * 3),
        ("radius", ctypes.c_double),
        ("unit_direction", ctypes.c_float * 3),
        ("min_distance", ctypes.c_double),
        ("max_distance", ctypes.c_double),
    ]


class SphereSweepDesc(ctypes.Structure):
    _fields_ = [("common", QueryCommon), ("sweep", SphereSweep)]


class SphereSweepBoundsCandidate(ctypes.Structure):
    _fields_ = [
        ("identity", HitIdentity),
        ("query_position", ctypes.c_double * 3),
        ("bounds_position", ctypes.c_double * 3),
        ("bounds_normal", ctypes.c_float * 3),
        ("distance", ctypes.c_double),
    ]


class SphereSweepExactHit(ctypes.Structure):
    _fields_ = [
        ("identity", HitIdentity),
        ("query_position", ctypes.c_double * 3),
        ("object_position", ctypes.c_double * 3),
        ("normal", ctypes.c_float * 3),
        ("distance", ctypes.c_double),
        ("feature", GeometryFeature),
    ]


class _SphereSweepHitData(ctypes.Union):
    _fields_ = [
        ("bounds_candidate", SphereSweepBoundsCandidate),
        ("exact", SphereSweepExactHit),
    ]


class SphereSweepHit(ctypes.Structure):
    _fields_ = [("truth", ctypes.c_int), ("data", _SphereSweepHitData)]


class SphereSweepClosestResult(ctypes.Structure):
    _fields_ = [
        ("source_ordinal", ctypes.c_int64),
        ("has_hit", ctypes.c_uint32),
        ("hit", SphereSweepHit),
    ]


class SphereSweepHitBuffer(ctypes.Structure):
    _fields_ = [("hits", ctypes.POINTER(SphereSweepHit)), ("capacity", ctypes.c_uint64)]


class BoxSweep(ctypes.Structure):
    _fields_ = [
        ("center", ctypes.c_double * 3),
        ("half_extents", ctypes.c_double * 3),
        ("rotation_xyzw", ctypes.c_float * 4),
        ("unit_direction", ctypes.c_float * 3),
        ("min_distance", ctypes.c_double),
        ("max_distance", ctypes.c_double),
    ]


class BoxSweepDesc(ctypes.Structure):
    _fields_ = [("common", QueryCommon), ("sweep", BoxSweep)]


BoxSweepBoundsCandidate = SphereSweepBoundsCandidate
BoxSweepExactHit = SphereSweepExactHit
BoxSweepHit = SphereSweepHit
BoxSweepClosestResult = SphereSweepClosestResult
BoxSweepHitBuffer = SphereSweepHitBuffer


class OverlapSphere(ctypes.Structure):
    _fields_ = [("center", ctypes.c_double * 3), ("radius", ctypes.c_double)]


class OverlapBox(ctypes.Structure):
    _fields_ = [
        ("center", ctypes.c_double * 3),
        ("rotation_xyzw", ctypes.c_float * 4),
        ("half_extents", ctypes.c_double * 3),
    ]


class OverlapAabb(ctypes.Structure):
    _fields_ = [
        ("min_corner", ctypes.c_double * 3),
        ("max_corner", ctypes.c_double * 3),
    ]


class _OverlapShapeData(ctypes.Union):
    _fields_ = [("sphere", OverlapSphere), ("box", OverlapBox), ("aabb", OverlapAabb)]


class OverlapShape(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("data", _OverlapShapeData)]


class OverlapDesc(ctypes.Structure):
    _fields_ = [("common", QueryCommon), ("shape", OverlapShape)]


class OverlapHit(ctypes.Structure):
    _fields_ = [("truth", ctypes.c_int), ("identity", HitIdentity)]


class OverlapHitBuffer(ctypes.Structure):
    _fields_ = [("hits", ctypes.POINTER(OverlapHit)), ("capacity", ctypes.c_uint64)]


class OverlapAnyResult(ctypes.Structure):
    _fields_ = [
        ("source_ordinal", ctypes.c_int64),
        ("has_hit", ctypes.c_uint32),
        ("hit", OverlapHit),
    ]


class _OverlapBatchSphere(ctypes.Structure):
    _fields_ = [("radius", ctypes.c_double)]


class _OverlapBatchBox(ctypes.Structure):
    _fields_ = [("half_extents", ctypes.c_double * 3)]


class _OverlapBatchAabb(ctypes.Structure):
    _fields_ = [("half_extents", ctypes.c_double * 3)]


class _OverlapBatchShapeData(ctypes.Union):
    _fields_ = [
        ("sphere", _OverlapBatchSphere),
        ("box", _OverlapBatchBox),
        ("aabb", _OverlapBatchAabb),
    ]


class OverlapBatchShape(_Structure):
    _fields_ = [("kind", ctypes.c_int), ("data", _OverlapBatchShapeData)]


class RaycastPlanDesc(_Structure):
    _fields_ = [
        ("target", TargetRef),
        ("hit_mode", ctypes.c_int),
        ("precision", ctypes.c_int),
        ("device", DLDevice),
        ("maximum_query_count", ctypes.c_uint64),
        ("maximum_hits_per_query", ctypes.c_uint64),
        ("hit_capacity_per_slot", ctypes.c_uint64),
        ("result_slot_count", ctypes.c_uint32),
        ("result_fields", ctypes.c_uint32),
        ("contact_tolerance", ctypes.c_double),
        ("mesh_hits", ctypes.c_int),
    ]


class SphereSweepPlanDesc(_Structure):
    _fields_ = [
        ("target", TargetRef),
        ("precision", ctypes.c_int),
        ("radius", ctypes.c_double),
        ("device", DLDevice),
        ("maximum_query_count", ctypes.c_uint64),
        ("result_slot_count", ctypes.c_uint32),
        ("result_fields", ctypes.c_uint32),
        ("hit_capacity_per_slot", ctypes.c_uint64),
        ("contact_tolerance", ctypes.c_double),
        ("hit_mode", ctypes.c_int),
        ("maximum_hits_per_query", ctypes.c_uint64),
    ]


class BoxSweepPlanDesc(_Structure):
    _fields_ = [
        ("target", TargetRef),
        ("precision", ctypes.c_int),
        ("half_extents", ctypes.c_double * 3),
        ("device", DLDevice),
        ("maximum_query_count", ctypes.c_uint64),
        ("result_slot_count", ctypes.c_uint32),
        ("result_fields", ctypes.c_uint32),
        ("hit_capacity_per_slot", ctypes.c_uint64),
        ("contact_tolerance", ctypes.c_double),
        ("hit_mode", ctypes.c_int),
        ("maximum_hits_per_query", ctypes.c_uint64),
    ]


class OverlapPlanDesc(_Structure):
    _fields_ = [
        ("target", TargetRef),
        ("hit_mode", ctypes.c_int),
        ("precision", ctypes.c_int),
        ("shape", OverlapBatchShape),
        ("device", DLDevice),
        ("maximum_query_count", ctypes.c_uint64),
        ("maximum_hits_per_query", ctypes.c_uint64),
        ("hit_capacity_per_slot", ctypes.c_uint64),
        ("result_slot_count", ctypes.c_uint32),
        ("result_fields", ctypes.c_uint32),
        ("contact_tolerance", ctypes.c_double),
    ]


class RaycastBatch(ctypes.Structure):
    _fields_ = [
        ("ordinal", OrdinalSelection),
        ("query_count", ctypes.c_uint64),
        ("query_bits", ctypes.POINTER(InputArray)),
        ("origins", ctypes.POINTER(InputArray)),
        ("unit_directions", ctypes.POINTER(InputArray)),
        ("min_distances", ctypes.POINTER(InputArray)),
        ("max_distances", ctypes.POINTER(InputArray)),
        ("maximum_hits", ctypes.POINTER(InputArray)),
        ("cuda_sync", CudaSync),
        ("query_flags", ctypes.c_uint32),
        ("reserved", ctypes.POINTER(InputArray) * 2),
    ]


class SphereSweepBatch(ctypes.Structure):
    _fields_ = [
        ("ordinal", OrdinalSelection),
        ("query_count", ctypes.c_uint64),
        ("query_bits", ctypes.POINTER(InputArray)),
        ("centers", ctypes.POINTER(InputArray)),
        ("unit_directions", ctypes.POINTER(InputArray)),
        ("min_distances", ctypes.POINTER(InputArray)),
        ("max_distances", ctypes.POINTER(InputArray)),
        ("radii", ctypes.POINTER(InputArray)),
        ("cuda_sync", CudaSync),
        ("query_flags", ctypes.c_uint32),
        ("reserved", ctypes.POINTER(InputArray) * 2),
    ]


class BoxSweepBatch(ctypes.Structure):
    _fields_ = [
        ("ordinal", OrdinalSelection),
        ("query_count", ctypes.c_uint64),
        ("query_bits", ctypes.POINTER(InputArray)),
        ("centers", ctypes.POINTER(InputArray)),
        ("rotations_xyzw", ctypes.POINTER(InputArray)),
        ("unit_directions", ctypes.POINTER(InputArray)),
        ("min_distances", ctypes.POINTER(InputArray)),
        ("max_distances", ctypes.POINTER(InputArray)),
        ("cuda_sync", CudaSync),
        ("half_extents", ctypes.POINTER(InputArray)),
        ("query_flags", ctypes.c_uint32),
        ("reserved", ctypes.POINTER(InputArray) * 2),
    ]


class OverlapBatch(ctypes.Structure):
    _fields_ = [
        ("ordinal", OrdinalSelection),
        ("query_count", ctypes.c_uint64),
        ("query_bits", ctypes.POINTER(InputArray)),
        ("translations", ctypes.POINTER(InputArray)),
        ("rotations_xyzw", ctypes.POINTER(InputArray)),
        ("half_extents", ctypes.POINTER(InputArray)),
        ("maximum_hits", ctypes.POINTER(InputArray)),
        ("cuda_sync", CudaSync),
        ("radii", ctypes.POINTER(InputArray)),
        ("reserved", ctypes.POINTER(InputArray) * 2),
    ]


class BatchResultCommon(ctypes.Structure):
    _fields_ = [
        ("result_slot", ctypes.c_uint32),
        ("query_count", ctypes.c_uint64),
        ("hit_capacity", ctypes.c_uint64),
        ("truth", ctypes.c_int),
        ("source_ordinal", ctypes.c_int64),
        ("execution_device", DLDevice),
        ("result_ready", CudaEvent),
    ]


class BatchHitRanges(ctypes.Structure):
    _fields_ = [
        ("query_status", DLTensor),
        ("stored_hit_count", DLTensor),
        ("hit_offsets", DLTensor),
        ("hit_counts", DLTensor),
        ("truncated", DLTensor),
    ]


class GeometryIdentityColumns(ctypes.Structure):
    _fields_ = [("types", DLTensor), ("values", DLTensor)]


class ExactFeatureColumns(ctypes.Structure):
    _fields_ = [
        ("kinds", DLTensor),
        ("source_face_indices", DLTensor),
        ("source_corner_indices", DLTensor),
        ("barycentrics", DLTensor),
    ]


class _RaycastCandidateColumns(ctypes.Structure):
    _fields_ = [
        ("bounds_positions", DLTensor),
        ("bounds_normals", DLTensor),
        ("distances", DLTensor),
    ]


class _RaycastExactColumns(ctypes.Structure):
    _fields_ = [
        ("positions", DLTensor),
        ("normals", DLTensor),
        ("distances", DLTensor),
        ("features", ExactFeatureColumns),
    ]


class _RaycastContactColumns(ctypes.Union):
    _fields_ = [("candidate", _RaycastCandidateColumns), ("exact", _RaycastExactColumns)]


class RaycastBatchColumns(ctypes.Structure):
    _fields_ = [
        ("hit_ranges", BatchHitRanges),
        ("object_ids", DLTensor),
        ("geometry", GeometryIdentityColumns),
        ("material_ids", DLTensor),
        ("contact", _RaycastContactColumns),
        ("reserved", DLTensor * 4),
    ]


class _SphereSweepCandidateColumns(ctypes.Structure):
    _fields_ = [
        ("bounds_positions", DLTensor),
        ("bounds_normals", DLTensor),
        ("query_positions", DLTensor),
        ("distances", DLTensor),
    ]


class _SphereSweepExactColumns(ctypes.Structure):
    _fields_ = [
        ("positions", DLTensor),
        ("normals", DLTensor),
        ("query_positions", DLTensor),
        ("distances", DLTensor),
        ("features", ExactFeatureColumns),
    ]


class _SphereSweepContactColumns(ctypes.Union):
    _fields_ = [("candidate", _SphereSweepCandidateColumns), ("exact", _SphereSweepExactColumns)]


class SphereSweepBatchColumns(ctypes.Structure):
    _fields_ = [
        ("hit_ranges", BatchHitRanges),
        ("object_ids", DLTensor),
        ("geometry", GeometryIdentityColumns),
        ("material_ids", DLTensor),
        ("contact", _SphereSweepContactColumns),
        ("reserved", DLTensor * 4),
    ]


class OverlapBatchColumns(ctypes.Structure):
    _fields_ = [
        ("hit_ranges", BatchHitRanges),
        ("object_ids", DLTensor),
        ("geometry", GeometryIdentityColumns),
        ("material_ids", DLTensor),
        ("reserved", DLTensor * 4),
    ]


class RaycastBatchReservation(ctypes.Structure):
    _fields_ = [("common", BatchResultCommon), ("columns", RaycastBatchColumns)]


class SphereSweepBatchReservation(ctypes.Structure):
    _fields_ = [("common", BatchResultCommon), ("columns", SphereSweepBatchColumns)]


class OverlapBatchReservation(ctypes.Structure):
    _fields_ = [("common", BatchResultCommon), ("columns", OverlapBatchColumns)]


class RaycastBatchResult(ctypes.Structure):
    _fields_ = [("common", BatchResultCommon), ("columns", RaycastBatchColumns)]


class SphereSweepBatchResult(ctypes.Structure):
    _fields_ = [("common", BatchResultCommon), ("columns", SphereSweepBatchColumns)]


BoxSweepBatchColumns = SphereSweepBatchColumns
BoxSweepBatchReservation = SphereSweepBatchReservation
BoxSweepBatchResult = SphereSweepBatchResult


class OverlapBatchResult(ctypes.Structure):
    _fields_ = [("common", BatchResultCommon), ("columns", OverlapBatchColumns)]


class Pair(ctypes.Structure):
    _fields_ = [("first", ObjectId), ("second", ObjectId)]


class PairBuffer(ctypes.Structure):
    _fields_ = [("pairs", ctypes.POINTER(Pair)), ("capacity", ctypes.c_uint64)]


class PairSnapshotDesc(_Structure):
    _fields_ = [
        ("first", QueryTarget),
        ("first_query_bits", ParticipationBits),
        ("second", QueryTarget),
        ("second_query_bits", ParticipationBits),
        ("contact_tolerance", ctypes.c_double),
    ]


class PairSnapshotResult(ctypes.Structure):
    _fields_ = [
        ("first_source_ordinal", ctypes.c_int64),
        ("second_source_ordinal", ctypes.c_int64),
        ("stored_pair_count", ctypes.c_uint64),
        ("truncated", ctypes.c_uint32),
    ]


class PairReporterDesc(_Structure):
    _fields_ = [
        ("first", TargetRef),
        ("first_query_bits", ParticipationBits),
        ("second", TargetRef),
        ("second_query_bits", ParticipationBits),
        ("active_pair_capacity", ctypes.c_uint64),
        ("delta_pair_capacity", ctypes.c_uint64),
        ("result_slot_count", ctypes.c_uint32),
        ("contact_tolerance", ctypes.c_double),
        ("result_device", DLDevice),
    ]


class PairReporterAdvance(ctypes.Structure):
    _fields_ = [
        ("first_ordinal", OrdinalSelection),
        ("second_ordinal", OrdinalSelection),
        ("cuda_sync", CudaSync),
    ]


class PairReporterReservation(ctypes.Structure):
    _fields_ = [
        ("result_slot", ctypes.c_uint32),
        ("result_device", DLDevice),
        ("delta_pair_capacity", ctypes.c_uint64),
        ("result_ready", CudaEvent),
        ("entered_pairs", DLTensor),
        ("exited_pairs", DLTensor),
        ("counts", DLTensor),
    ]


class PairReporterResult(ctypes.Structure):
    _fields_ = [
        ("result_slot", ctypes.c_uint32),
        ("outcome", ctypes.c_int),
        ("first_source_ordinal", ctypes.c_int64),
        ("second_source_ordinal", ctypes.c_int64),
        ("accepted_active_pair_count", ctypes.c_uint64),
        ("result_ready", CudaEvent),
        ("entered_pairs", DLTensor),
        ("exited_pairs", DLTensor),
    ]


class OpWaitResult(ctypes.Structure):
    _fields_ = [("state", ctypes.c_int)]


class OpStatus(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int),
        ("terminal_status", ctypes.c_int),
        ("progress", ctypes.c_double),
    ]


class OvstageFilter(ctypes.Structure):
    _fields_ = [("predicates", ctypes.c_void_p), ("count", ctypes.c_size_t)]


class OvstageScope(ctypes.Structure):
    _fields_ = [
        ("include_paths", ctypes.POINTER(ctypes.c_uint64)),
        ("exclude_paths", ctypes.POINTER(ctypes.c_uint64)),
        ("include_count", ctypes.c_uint32),
        ("exclude_count", ctypes.c_uint32),
    ]


class OvstagePopulationDesc(_Structure):
    _fields_ = [
        ("stage", ctypes.c_void_p),
        ("filter", OvstageFilter),
        ("scope", OvstageScope),
        ("common", PopulationDesc),
    ]


class OvstageViewDesc(_Structure):
    _fields_ = [("filter", OvstageFilter), ("scope", OvstageScope)]


class OvstageParticipationTable(ctypes.Structure):
    _fields_ = [
        ("default_bits", ParticipationBits),
        ("paths", ctypes.POINTER(ctypes.c_uint64)),
        ("bits", ctypes.POINTER(ParticipationBits)),
        ("count", ctypes.c_uint32),
    ]


# Every public Structure above is an ABI type; aliases and underscored helpers are not.
TRANSPARENT_TYPES: Final = {
    name: value
    for name, value in tuple(globals().items())
    if isinstance(value, type)
    and issubclass(value, ctypes.Structure)
    and value.__name__ == name
    and not name.startswith("_")
}


class ExportSpec(NamedTuple):
    argtypes: tuple[type, ...]
    restype: type


P = ctypes.POINTER
U64 = ctypes.c_uint64
U32 = ctypes.c_uint32
I64 = ctypes.c_int64
CORE_EXPORT_SPECS: Final[dict[str, ExportSpec]] = {
    "ovspatialquery_get_version": ExportSpec((P(U32), P(U32), P(U32)), Result),
    "ovspatialquery_get_last_error": ExportSpec((), String),
    "ovspatialquery_get_last_op_error": ExportSpec((U64, U64), String),
    "ovspatialquery_create_instance": ExportSpec((P(InstanceDesc), P(U64)), Result),
    "ovspatialquery_get_instance_memory_usage": ExportSpec((U64, DLDevice, P(U64)), Result),
    "ovspatialquery_query_capability": ExportSpec((U64, DLDevice, ctypes.c_int), Result),
    "ovspatialquery_destroy_instance": ExportSpec((U64,), Result),
    "ovspatialquery_get_population_info": ExportSpec((U64, U64, P(PopulationInfo)), Result),
    "ovspatialquery_destroy_population": ExportSpec((U64, U64), Result),
    "ovspatialquery_fetch_population_update": ExportSpec(
        (U64, U64, P(PopulationUpdateResult)), Result
    ),
    "ovspatialquery_destroy_view": ExportSpec((U64, U64), Result),
    "ovspatialquery_raycast_closest_sync": ExportSpec(
        (U64, P(RaycastDesc), P(RaycastSingleResult)), Result
    ),
    "ovspatialquery_raycast_any_sync": ExportSpec(
        (U64, P(RaycastDesc), P(RaycastSingleResult)), Result
    ),
    "ovspatialquery_raycast_multiple_sync": ExportSpec(
        (U64, P(RaycastDesc), P(RaycastHitBuffer), P(MultipleResult)), Result
    ),
    "ovspatialquery_sweep_sphere_closest_sync": ExportSpec(
        (U64, P(SphereSweepDesc), P(SphereSweepClosestResult)), Result
    ),
    "ovspatialquery_sweep_sphere_any_sync": ExportSpec(
        (U64, P(SphereSweepDesc), P(SphereSweepClosestResult)), Result
    ),
    "ovspatialquery_sweep_sphere_multiple_sync": ExportSpec(
        (U64, P(SphereSweepDesc), P(SphereSweepHitBuffer), P(MultipleResult)), Result
    ),
    "ovspatialquery_sweep_box_closest_sync": ExportSpec(
        (U64, P(BoxSweepDesc), P(BoxSweepClosestResult)), Result
    ),
    "ovspatialquery_sweep_box_any_sync": ExportSpec(
        (U64, P(BoxSweepDesc), P(BoxSweepClosestResult)), Result
    ),
    "ovspatialquery_sweep_box_multiple_sync": ExportSpec(
        (U64, P(BoxSweepDesc), P(BoxSweepHitBuffer), P(MultipleResult)), Result
    ),
    "ovspatialquery_overlap_any_sync": ExportSpec(
        (U64, P(OverlapDesc), P(OverlapAnyResult)), Result
    ),
    "ovspatialquery_overlap_multiple_sync": ExportSpec(
        (U64, P(OverlapDesc), P(OverlapHitBuffer), P(MultipleResult)), Result
    ),
    "ovspatialquery_create_raycast_plan": ExportSpec((U64, P(RaycastPlanDesc), P(U64)), Result),
    "ovspatialquery_enqueue_raycast_plan": ExportSpec(
        (U64, U64, P(RaycastBatch), P(RaycastBatchReservation)), EnqueueResult
    ),
    "ovspatialquery_fetch_raycast_batch": ExportSpec((U64, U64, P(RaycastBatchResult)), Result),
    "ovspatialquery_create_sphere_sweep_plan": ExportSpec(
        (U64, P(SphereSweepPlanDesc), P(U64)), Result
    ),
    "ovspatialquery_enqueue_sphere_sweep_plan": ExportSpec(
        (U64, U64, P(SphereSweepBatch), P(SphereSweepBatchReservation)), EnqueueResult
    ),
    "ovspatialquery_fetch_sphere_sweep_batch": ExportSpec(
        (U64, U64, P(SphereSweepBatchResult)), Result
    ),
    "ovspatialquery_create_box_sweep_plan": ExportSpec((U64, P(BoxSweepPlanDesc), P(U64)), Result),
    "ovspatialquery_enqueue_box_sweep_plan": ExportSpec(
        (U64, U64, P(BoxSweepBatch), P(BoxSweepBatchReservation)), EnqueueResult
    ),
    "ovspatialquery_fetch_box_sweep_batch": ExportSpec((U64, U64, P(BoxSweepBatchResult)), Result),
    "ovspatialquery_create_overlap_plan": ExportSpec((U64, P(OverlapPlanDesc), P(U64)), Result),
    "ovspatialquery_enqueue_overlap_plan": ExportSpec(
        (U64, U64, P(OverlapBatch), P(OverlapBatchReservation)), EnqueueResult
    ),
    "ovspatialquery_fetch_overlap_batch": ExportSpec((U64, U64, P(OverlapBatchResult)), Result),
    "ovspatialquery_destroy_batch_plan": ExportSpec((U64, U64), Result),
    "ovspatialquery_query_pairs_sync": ExportSpec(
        (U64, P(PairSnapshotDesc), P(PairBuffer), P(PairSnapshotResult)), Result
    ),
    "ovspatialquery_create_pair_reporter": ExportSpec((U64, P(PairReporterDesc), P(U64)), Result),
    "ovspatialquery_grow_pair_reporter_capacity": ExportSpec((U64, U64, U64, U64), Result),
    "ovspatialquery_enqueue_pair_reporter_advance": ExportSpec(
        (U64, U64, P(PairReporterAdvance), P(PairReporterReservation)), EnqueueResult
    ),
    "ovspatialquery_fetch_pair_reporter": ExportSpec((U64, U64, P(PairReporterResult)), Result),
    "ovspatialquery_destroy_pair_reporter": ExportSpec((U64, U64), Result),
    "ovspatialquery_wait_op": ExportSpec((U64, U64, U64, P(OpWaitResult)), Result),
    "ovspatialquery_query_op_status": ExportSpec((U64, U64, P(OpStatus)), Result),
    "ovspatialquery_destroy_op": ExportSpec((U64, P(U64)), Result),
    "ovspatialquery_destroy_op_after_cuda": ExportSpec((U64, P(U64), P(CudaEvent)), Result),
}
EXPLICIT_EXPORT_SPECS: Final[dict[str, ExportSpec]] = {
    "ovspatialquery_create_explicit_population": ExportSpec(
        (U64, P(PopulationDesc), P(U64)), Result
    ),
    "ovspatialquery_enqueue_explicit_population_set": ExportSpec(
        (U64, U64, P(ExplicitPopulationSet)), EnqueueResult
    ),
    "ovspatialquery_enqueue_explicit_population_update": ExportSpec(
        (U64, U64, P(ExplicitPopulationUpdate)), EnqueueResult
    ),
    "ovspatialquery_create_explicit_view": ExportSpec(
        (U64, U64, P(ExplicitViewDesc), P(U64)), Result
    ),
    "ovspatialquery_enqueue_sphere_registration": ExportSpec(
        (U64, P(SphereDefinitionTable)), EnqueueResult
    ),
    "ovspatialquery_enqueue_box_registration": ExportSpec(
        (U64, P(BoxDefinitionTable)), EnqueueResult
    ),
    "ovspatialquery_enqueue_capsule_registration": ExportSpec(
        (U64, P(CapsuleDefinitionTable)), EnqueueResult
    ),
    "ovspatialquery_enqueue_convex_registration": ExportSpec(
        (U64, P(ConvexDefinitionTable)), EnqueueResult
    ),
    "ovspatialquery_enqueue_triangle_mesh_registration": ExportSpec(
        (U64, P(TriangleMeshDefinitionTable)), EnqueueResult
    ),
    "ovspatialquery_destroy_geometry_definition": ExportSpec((U64, GeometryRef), Result),
    "ovspatialquery_get_dropped_triangle_count": ExportSpec((U64, U64, P(U64)), Result),
}
OVSTAGE_EXPORT_SPECS: Final[dict[str, ExportSpec]] = {
    "ovspatialquery_create_ovstage_population": ExportSpec(
        (U64, P(OvstagePopulationDesc), P(U64)), Result
    ),
    "ovspatialquery_create_ovstage_view": ExportSpec(
        (U64, U64, P(OvstageViewDesc), P(U64)), Result
    ),
    "ovspatialquery_set_ovstage_participation_map": ExportSpec(
        (U64, U64, P(OvstageParticipationTable)), Result
    ),
    "ovspatialquery_enqueue_ovstage_population_set": ExportSpec((U64, U64, I64), EnqueueResult),
    "ovspatialquery_enqueue_ovstage_population_update": ExportSpec(
        (U64, U64, I64, I64), EnqueueResult
    ),
    "ovspatialquery_enqueue_ovstage_population_set_with_execution": ExportSpec(
        (U64, U64, I64, P(DynamicExecution)), EnqueueResult
    ),
    "ovspatialquery_enqueue_ovstage_population_update_with_execution": ExportSpec(
        (U64, U64, I64, I64, P(DynamicExecution)), EnqueueResult
    ),
    "ovspatialquery_fetch_ovstage_dropped_triangles": ExportSpec(
        (U64, U64, P(OvstageDroppedTriangles)), Result
    ),
    "ovspatialquery_query_ovstage_capability": ExportSpec(
        (U64, ctypes.c_void_p, DLDevice, ctypes.c_int), Result
    ),
}


def bind_exports(library: object, specs: dict[str, ExportSpec]) -> dict[str, Any]:
    bound = {}
    for name, spec in specs.items():
        function = getattr(library, name)
        function.argtypes = list(spec.argtypes)
        function.restype = spec.restype
        bound[name] = function
    return bound


__all__ = [
    "ApiStatus",
    "GeometryType",
    "InputOwnership",
    "TargetKind",
    "OrdinalSelectionKind",
    "QueryPrecision",
    "HitTruth",
    "GeometryFeatureKind",
    "OverlapShapeKind",
    "HitMode",
    "BatchRowStatus",
    "BatchResultField",
    "PairReporterOutcome",
    "OpState",
    "OvstageExclusionReason",
    "Capability",
    "CORE_EXPORT_SPECS",
    "EXPLICIT_EXPORT_SPECS",
    "OVSTAGE_EXPORT_SPECS",
    "TRANSPARENT_TYPES",
    "bind_exports",
    "BoxSweepBoundsCandidate",
    "BoxSweepExactHit",
    "BoxSweepHit",
    "BoxSweepClosestResult",
    "BoxSweepHitBuffer",
    "BoxSweepBatchColumns",
    "BoxSweepBatchReservation",
    "BoxSweepBatchResult",
    *TRANSPARENT_TYPES,
]
