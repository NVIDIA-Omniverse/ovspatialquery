# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed ctypes verification for the current API native package."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import subprocess
import sys


API_SUCCESS = 0
API_INVALID_ARGUMENT = 3
API_NOT_SUPPORTED = 6
INVALID_OP = 0
K_DL_CPU = 1


class Result(ctypes.Structure):
    _fields_ = [("status", ctypes.c_uint32)]


class String(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_char_p), ("length", ctypes.c_size_t)]


class DLDevice(ctypes.Structure):
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


class RaycastCandidateColumns(ctypes.Structure):
    _fields_ = [
        ("bounds_positions", DLTensor),
        ("bounds_normals", DLTensor),
        ("distances", DLTensor),
    ]


class RaycastExactColumns(ctypes.Structure):
    _fields_ = [
        ("positions", DLTensor),
        ("normals", DLTensor),
        ("distances", DLTensor),
        ("features", ExactFeatureColumns),
    ]


class RaycastContactColumns(ctypes.Union):
    _fields_ = [
        ("candidate", RaycastCandidateColumns),
        ("exact", RaycastExactColumns),
    ]


class RaycastBatchColumns(ctypes.Structure):
    _fields_ = [
        ("hit_ranges", BatchHitRanges),
        ("object_ids", DLTensor),
        ("geometry", GeometryIdentityColumns),
        ("material_ids", DLTensor),
        ("contact", RaycastContactColumns),
        ("reserved", DLTensor * 4),
    ]


class SphereSweepCandidateColumns(ctypes.Structure):
    _fields_ = [
        ("bounds_positions", DLTensor),
        ("bounds_normals", DLTensor),
        ("query_positions", DLTensor),
        ("distances", DLTensor),
    ]


class SphereSweepExactColumns(ctypes.Structure):
    _fields_ = [
        ("positions", DLTensor),
        ("normals", DLTensor),
        ("query_positions", DLTensor),
        ("distances", DLTensor),
        ("features", ExactFeatureColumns),
    ]


class SphereSweepContactColumns(ctypes.Union):
    _fields_ = [
        ("candidate", SphereSweepCandidateColumns),
        ("exact", SphereSweepExactColumns),
    ]


class SphereSweepBatchColumns(ctypes.Structure):
    _fields_ = [
        ("hit_ranges", BatchHitRanges),
        ("object_ids", DLTensor),
        ("geometry", GeometryIdentityColumns),
        ("material_ids", DLTensor),
        ("contact", SphereSweepContactColumns),
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


class CudaEvent(ctypes.Structure):
    _fields_ = [("value", ctypes.c_size_t)]


class BatchResultCommon(ctypes.Structure):
    _fields_ = [
        ("result_slot", ctypes.c_uint32),
        ("query_count", ctypes.c_uint64),
        ("hit_capacity", ctypes.c_uint64),
        ("truth", ctypes.c_int32),
        ("source_ordinal", ctypes.c_int64),
        ("execution_device", DLDevice),
        ("result_ready", CudaEvent),
    ]


class RaycastBatchReservation(ctypes.Structure):
    _fields_ = [
        ("common", BatchResultCommon),
        ("columns", RaycastBatchColumns),
    ]


class SphereSweepBatchReservation(ctypes.Structure):
    _fields_ = [
        ("common", BatchResultCommon),
        ("columns", SphereSweepBatchColumns),
    ]


class OverlapBatchReservation(ctypes.Structure):
    _fields_ = [
        ("common", BatchResultCommon),
        ("columns", OverlapBatchColumns),
    ]


class RaycastBatchResult(ctypes.Structure):
    _fields_ = RaycastBatchReservation._fields_


class SphereSweepBatchResult(ctypes.Structure):
    _fields_ = SphereSweepBatchReservation._fields_


class OverlapBatchResult(ctypes.Structure):
    _fields_ = OverlapBatchReservation._fields_


class PairReporterResult(ctypes.Structure):
    _fields_ = [
        ("result_slot", ctypes.c_uint32),
        ("outcome", ctypes.c_int32),
        ("first_source_ordinal", ctypes.c_int64),
        ("second_source_ordinal", ctypes.c_int64),
        ("accepted_active_pair_count", ctypes.c_uint64),
        ("result_ready", CudaEvent),
        ("entered_pairs", DLTensor),
        ("exited_pairs", DLTensor),
    ]


ABI_BATCH_TYPES = {
    layout.__name__: layout
    for layout in (
        DLDevice,
        DLDataType,
        DLTensor,
        BatchHitRanges,
        BatchResultCommon,
        GeometryIdentityColumns,
        ExactFeatureColumns,
        RaycastBatchColumns,
        SphereSweepBatchColumns,
        OverlapBatchColumns,
        RaycastBatchReservation,
        SphereSweepBatchReservation,
        OverlapBatchReservation,
        RaycastBatchResult,
        SphereSweepBatchResult,
        OverlapBatchResult,
        PairReporterResult,
    )
}

ABI_NESTED_FIELDS = {
    "RaycastBatchColumns": (
        "contact.candidate.bounds_positions",
        "contact.candidate.bounds_normals",
        "contact.candidate.distances",
        "contact.exact.positions",
        "contact.exact.normals",
        "contact.exact.distances",
        "contact.exact.features",
    ),
    "SphereSweepBatchColumns": (
        "contact.candidate.bounds_positions",
        "contact.candidate.bounds_normals",
        "contact.candidate.query_positions",
        "contact.candidate.distances",
        "contact.exact.positions",
        "contact.exact.normals",
        "contact.exact.query_positions",
        "contact.exact.distances",
        "contact.exact.features",
    ),
}


class DeviceBudget(ctypes.Structure):
    _fields_ = [("device", DLDevice), ("byte_budget", ctypes.c_uint64)]


class InstanceDesc(ctypes.Structure):
    _fields_ = [
        ("operation_capacity", ctypes.c_uint32),
        ("max_cpu_concurrency", ctypes.c_uint32),
        ("host_byte_budget", ctypes.c_uint64),
        ("device_budgets", ctypes.POINTER(DeviceBudget)),
        ("device_budget_count", ctypes.c_uint32),
    ]


def require(result: Result, expected: int, label: str) -> None:
    if result.status != expected:
        raise RuntimeError(f"{label}: expected {expected}, got {result.status}")


def bind(library: ctypes.CDLL) -> None:
    library.ovspatialquery_get_version.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.ovspatialquery_get_version.restype = Result
    library.ovspatialquery_get_last_error.argtypes = []
    library.ovspatialquery_get_last_error.restype = String
    library.ovspatialquery_get_last_op_error.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    library.ovspatialquery_get_last_op_error.restype = String
    library.ovspatialquery_create_instance.argtypes = [
        ctypes.POINTER(InstanceDesc),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.ovspatialquery_create_instance.restype = Result
    library.ovspatialquery_get_instance_memory_usage.argtypes = [
        ctypes.c_uint64,
        DLDevice,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.ovspatialquery_get_instance_memory_usage.restype = Result
    library.ovspatialquery_destroy_instance.argtypes = [ctypes.c_uint64]
    library.ovspatialquery_destroy_instance.restype = Result


def check_compiler_abi(witness_path: Path) -> None:
    completed = subprocess.run(
        [str(witness_path.resolve(strict=True))],
        check=True,
        capture_output=True,
        text=True,
    )
    observed_types: set[str] = set()
    observed_fields: set[tuple[str, str]] = set()
    for line in completed.stdout.splitlines():
        columns = line.split("\t")
        if columns[0] == "T" and len(columns) == 4:
            _, name, size, alignment = columns
            layout = ABI_BATCH_TYPES.get(name)
            if layout is None:
                raise RuntimeError(f"unknown compiler ABI type {name}")
            if ctypes.sizeof(layout) != int(size) or ctypes.alignment(layout) != int(alignment):
                raise RuntimeError(
                    f"{name}: ctypes size/alignment "
                    f"{ctypes.sizeof(layout)}/{ctypes.alignment(layout)}, "
                    f"compiler {size}/{alignment}"
                )
            observed_types.add(name)
        elif columns[0] == "F" and len(columns) == 4:
            _, name, field_name, offset = columns
            layout = ABI_BATCH_TYPES.get(name)
            if layout is None:
                raise RuntimeError(f"unknown compiler ABI field {name}.{field_name}")
            actual = 0
            field_layout = layout
            for field_part in field_name.split("."):
                if not hasattr(field_layout, field_part):
                    raise RuntimeError(f"unknown compiler ABI field {name}.{field_name}")
                actual += getattr(field_layout, field_part).offset
                field_layout = dict(field_layout._fields_)[field_part]
            if actual != int(offset):
                raise RuntimeError(
                    f"{name}.{field_name}: ctypes offset {actual}, compiler {offset}"
                )
            observed_fields.add((name, field_name))
        else:
            raise RuntimeError(f"unknown compiler ABI witness row: {line!r}")

    expected_types = set(ABI_BATCH_TYPES)
    expected_fields = {
        (name, field_name)
        for name, layout in ABI_BATCH_TYPES.items()
        for field_name, _ in layout._fields_
    }
    expected_fields.update(
        (name, field_name)
        for name, field_names in ABI_NESTED_FIELDS.items()
        for field_name in field_names
    )
    if observed_types != expected_types:
        raise RuntimeError(
            "compiler ABI type inventory mismatch: "
            f"missing={sorted(expected_types - observed_types)} "
            f"extra={sorted(observed_types - expected_types)}"
        )
    if observed_fields != expected_fields:
        raise RuntimeError(
            "compiler ABI field inventory mismatch: "
            f"missing={sorted(expected_fields - observed_fields)} "
            f"extra={sorted(observed_fields - expected_fields)}"
        )


def run_smoke(dll_path: Path) -> None:
    library = ctypes.CDLL(str(dll_path))
    bind(library)

    major = ctypes.c_uint32()
    minor = ctypes.c_uint32()
    patch = ctypes.c_uint32()
    require(
        library.ovspatialquery_get_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        ),
        API_SUCCESS,
        "get version",
    )
    if (major.value, minor.value, patch.value) == (0, 0, 0):
        raise RuntimeError("installed semantic version is unset")

    instance = ctypes.c_uint64()
    desc = InstanceDesc(4, 2, 4 << 20, None, 0)
    require(
        library.ovspatialquery_create_instance(ctypes.byref(desc), ctypes.byref(instance)),
        API_SUCCESS,
        "create instance",
    )
    if instance.value == 0:
        raise RuntimeError("create_instance returned an invalid handle")
    resident = ctypes.c_uint64((1 << 64) - 1)
    require(
        library.ovspatialquery_get_instance_memory_usage(
            instance, DLDevice(K_DL_CPU, 0), ctypes.byref(resident)
        ),
        API_SUCCESS,
        "host memory usage",
    )
    if resident.value == 0 or resident.value > desc.host_byte_budget:
        raise RuntimeError(f"invalid host ledger value {resident.value}")
    preserved = ctypes.c_uint64(123)
    require(
        library.ovspatialquery_get_instance_memory_usage(
            instance, DLDevice(K_DL_CPU, 1), ctypes.byref(preserved)
        ),
        API_NOT_SUPPORTED,
        "unconfigured device",
    )
    if preserved.value != 123:
        raise RuntimeError("failed memory query mutated caller output")
    if library.ovspatialquery_get_last_op_error(instance, INVALID_OP).length != 0:
        raise RuntimeError("invalid operation unexpectedly has diagnostics")
    require(library.ovspatialquery_destroy_instance(instance), API_SUCCESS, "destroy instance")
    invalid = ctypes.c_uint64(99)
    require(
        library.ovspatialquery_create_instance(None, ctypes.byref(invalid)),
        API_INVALID_ARGUMENT,
        "null instance descriptor",
    )
    if invalid.value != 0:
        raise RuntimeError("failed instance creation did not invalidate output")
    print(f"installed current API ctypes verification passed: {dll_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--abi-witness", type=Path)
    arguments = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required, got {sys.version}")
    if arguments.abi_witness is not None:
        check_compiler_abi(arguments.abi_witness)
    run_smoke(arguments.dll.resolve(strict=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
