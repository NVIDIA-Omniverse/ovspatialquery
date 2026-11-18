# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import array
import ctypes
from dataclasses import FrozenInstanceError
import gc
import importlib.util
import unittest
from unittest.mock import patch
import weakref
from ovspatialquery import (
    BatchResultField,
    GeometryFeatureKind,
    HitMode,
    MeshHits,
    OverlapShapeKind,
    HitTruth,
    PairReporterOutcome,
    raw,
)
from ovspatialquery.types import (
    BoxSweepPlanConfig,
    BoxSweepQuery,
    BoxSweepResult,
    BoxSweepBatchReservation,
    BoxSweepBatchResult,
    InstanceConfig,
    DataType,
    DeviceBudget,
    Device,
    ExactFeatureColumns,
    OrdinalSelection,
    PairReporterConfig,
    PairReporterResult,
    PopulationConfig,
    PopulationInfo,
    PublicationResult,
    RaycastPlanConfig,
    RaycastQuery,
    RaycastBatchResult,
    RaycastBatchReservation,
    SphereSweepPlanConfig,
    SphereSweepBatchReservation,
    SphereSweepBatchResult,
    OverlapBatchShape,
    OverlapPlanConfig,
    OverlapBatchReservation,
    OverlapBatchResult,
    TargetRef,
    TensorView,
    _DLPackLease,
)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", raw.DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class _CompactProvider:
    """A DLPack producer with the given strides, or none, as compact producers may omit them."""

    def __init__(
        self,
        data: ctypes.Array,
        shape: tuple[int, ...],
        dtype: raw.DLDataType,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        self.data = data
        self.shape = (ctypes.c_int64 * len(shape))(*shape)
        self.strides = None if strides is None else (ctypes.c_int64 * len(strides))(*strides)
        self.managed = _DLManagedTensor(
            raw.DLTensor(
                data=ctypes.cast(self.data, ctypes.c_void_p),
                device=raw.DLDevice(raw.K_DL_CPU, 0),
                ndim=len(shape),
                dtype=dtype,
                shape=self.shape,
                strides=self.strides,
            )
        )
        new_capsule = ctypes.pythonapi.PyCapsule_New
        new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        new_capsule.restype = ctypes.py_object
        self.capsule = new_capsule(ctypes.byref(self.managed), b"dltensor", None)

    def __dlpack__(self):
        return self.capsule


class TypeProjectionTests(unittest.TestCase):
    def test_optional_tensor_distinguishes_absent_from_present_zero_length(self) -> None:
        self.assertIsNone(TensorView.optional_from_raw(raw.DLTensor()))
        shape = (ctypes.c_int64 * 1)(0)
        present = raw.DLTensor(
            data=None,
            device=raw.DLDevice(raw.K_DL_CPU, 0),
            ndim=1,
            dtype=raw.DLDataType(raw.K_DL_UINT, 64, 1),
            shape=shape,
            strides=None,
            byte_offset=0,
        )
        projected = TensorView.optional_from_raw(present)
        self.assertIsNotNone(projected)
        self.assertEqual(projected.shape, (0,))
        self.assertEqual(projected.data, 0)

    def test_required_result_tensor_rejects_an_absent_descriptor(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent"):
            TensorView.from_raw(raw.DLTensor(ndim=1))

    def test_pair_reporter_result_preserves_device_tensor_descriptors(self) -> None:
        shape = (ctypes.c_int64 * 2)(0, 2)

        def empty_pairs() -> raw.DLTensor:
            return raw.DLTensor(
                data=None,
                device=raw.DLDevice(raw.K_DL_CUDA, 1),
                ndim=2,
                dtype=raw.DLDataType(raw.K_DL_UINT, 64, 1),
                shape=shape,
            )

        value = raw.PairReporterResult()
        value.outcome = PairReporterOutcome.ADVANCED
        value.first_source_ordinal = 10
        value.second_source_ordinal = 11
        value.result_ready = raw.CudaEvent(23)
        value.entered_pairs = empty_pairs()
        value.exited_pairs = empty_pairs()

        projected = PairReporterResult.from_raw(value)

        self.assertIsInstance(projected.entered_pairs, TensorView)
        self.assertIsInstance(projected.exited_pairs, TensorView)
        self.assertEqual(projected.entered_pairs.shape, (0, 2))
        self.assertEqual(projected.entered_pairs.device, Device.cuda(1))
        self.assertEqual(projected.result_ready, 23)

    def test_raycast_batch_wrapper_selects_truth_arm_without_host_projection(self) -> None:
        shape_q = (ctypes.c_int64 * 1)(2)
        shape_h = (ctypes.c_int64 * 1)(0)

        def tensor(shape, code=raw.K_DL_UINT, bits=64):
            return raw.DLTensor(
                data=ctypes.c_void_p(0x1234),
                device=raw.DLDevice(raw.K_DL_CUDA, 0),
                ndim=1,
                dtype=raw.DLDataType(code, bits, 1),
                shape=shape,
            )

        value = raw.RaycastBatchResult()
        value.common.query_count = 2
        value.common.hit_capacity = 4
        value.common.truth = HitTruth.BOUNDS_CANDIDATE
        value.common.source_ordinal = 11
        value.common.execution_device = raw.DLDevice(raw.K_DL_CUDA, 0)
        value.common.result_ready = raw.CudaEvent(77)
        value.columns.hit_ranges.query_status = tensor(shape_q, raw.K_DL_UINT, 32)
        value.columns.hit_ranges.stored_hit_count = tensor(shape_h)
        value.columns.hit_ranges.hit_offsets = tensor(shape_q)
        value.columns.hit_ranges.hit_counts = tensor(shape_q)
        value.columns.hit_ranges.truncated = tensor(shape_q, raw.K_DL_UINT, 8)
        value.columns.object_ids = tensor(shape_h)
        value.columns.contact.candidate.bounds_positions = tensor(shape_h, raw.K_DL_FLOAT, 64)

        projected = RaycastBatchResult.from_raw(value)
        self.assertEqual(projected.common.result_ready, 77)
        self.assertEqual(projected.hit_ranges.hit_counts.shape, (2,))
        self.assertEqual(projected.object_ids.shape, (0,))
        self.assertIsNotNone(projected.candidate)
        self.assertIsNone(projected.exact)
        self.assertIsNone(projected.geometry)
        self.assertEqual(projected.candidate.bounds_positions.data, 0x1234)

    def test_batch_atomic_optional_group_rejects_partial_presence(self) -> None:
        shape = (ctypes.c_int64 * 1)(0)
        value = raw.RaycastBatchResult()
        value.common.truth = HitTruth.EXACT_SURFACE
        value.columns.geometry.types = raw.DLTensor(
            device=raw.DLDevice(raw.K_DL_CPU, 0),
            ndim=1,
            dtype=raw.DLDataType(raw.K_DL_INT, 32, 1),
            shape=shape,
        )
        with self.assertRaisesRegex(ValueError, "geometry identity"):
            RaycastBatchResult.from_raw(value)

    def test_typed_reservations_are_not_terminal_result_types(self) -> None:
        for raw_reservation, raw_result, reservation, result in (
            (
                raw.RaycastBatchReservation,
                raw.RaycastBatchResult,
                RaycastBatchReservation,
                RaycastBatchResult,
            ),
            (
                raw.SphereSweepBatchReservation,
                raw.SphereSweepBatchResult,
                SphereSweepBatchReservation,
                SphereSweepBatchResult,
            ),
            (
                raw.BoxSweepBatchReservation,
                raw.BoxSweepBatchResult,
                BoxSweepBatchReservation,
                BoxSweepBatchResult,
            ),
            (
                raw.OverlapBatchReservation,
                raw.OverlapBatchResult,
                OverlapBatchReservation,
                OverlapBatchResult,
            ),
        ):
            with self.subTest(reservation=reservation.__name__):
                self.assertIsNot(reservation, result)
                self.assertFalse(issubclass(reservation, result))
                with self.assertRaisesRegex(TypeError, "terminal result"):
                    result.from_raw(raw_reservation())
                with self.assertRaisesRegex(TypeError, "pending reservation"):
                    reservation.from_raw(raw_result())

    def test_exact_feature_payload_is_masked_by_feature_kind(self) -> None:
        kinds, faces, corners, barycentrics = (
            TensorView(address, Device.cuda(), 1, DataType(raw.K_DL_UINT, 64, 1), (0,), None, 0)
            for address in (0x1000, 0x2000, 0x3000, 0x4000)
        )
        features = ExactFeatureColumns(kinds, faces, corners, barycentrics)
        self.assertIsNone(features.triangle_payload_for(GeometryFeatureKind.NONE))
        self.assertEqual(
            features.triangle_payload_for(GeometryFeatureKind.TRIANGLE),
            (faces, corners, barycentrics),
        )

    def test_dlpack_compact_layout_gets_explicit_strides_without_data_copy(self) -> None:
        provider = _CompactProvider(
            (ctypes.c_double * 6)(*range(6)), (2, 3), raw.DLDataType(raw.K_DL_FLOAT, 64, 1)
        )
        lease = _DLPackLease.from_dlpack(
            provider, ownership=raw.InputOwnership.COPIED_ON_SUBMISSION
        )

        tensor = lease.array.tensor
        self.assertEqual(int(tensor.data), ctypes.addressof(provider.data))
        self.assertEqual(tuple(tensor.strides[index] for index in range(2)), (3, 1))
        self.assertEqual(lease.array.ownership, raw.InputOwnership.COPIED_ON_SUBMISSION)
        self.assertIn(provider, lease.keepalive)

        def structure_types():
            gc.collect()
            return {
                value
                for value in gc.get_objects()
                if isinstance(value, type) and issubclass(value, ctypes.Structure)
            }

        before = structure_types()
        for _ in range(64):
            transient = _DLPackLease.from_dlpack(provider)
            del transient
        self.assertEqual(structure_types(), before)

        owner = weakref.ref(provider)
        del provider
        gc.collect()
        self.assertIsNotNone(owner())
        self.assertEqual(
            tuple(ctypes.cast(tensor.data, ctypes.POINTER(ctypes.c_double))[:6]),
            tuple(range(6)),
        )
        del lease
        gc.collect()
        self.assertIsNone(owner())

    def test_dlpack_empty_layout_gets_the_abi_required_null_data_pointer(self) -> None:
        empty = _CompactProvider(
            (ctypes.c_uint64 * 1)(99), (0,), raw.DLDataType(raw.K_DL_UINT, 64, 1)
        )
        lease = _DLPackLease.from_dlpack(empty)

        self.assertFalse(lease.array.tensor.data)
        self.assertEqual(lease.array.tensor.strides[0], 1)

    def test_dlpack_empty_layout_replaces_the_producers_arbitrary_strides(self) -> None:
        # NumPy exports an empty array with stride 0; the library accepts only compact strides.
        empty = _CompactProvider(
            (ctypes.c_uint64 * 1)(99), (0,), raw.DLDataType(raw.K_DL_UINT, 64, 1), strides=(0,)
        )
        lease = _DLPackLease.from_dlpack(empty)

        self.assertFalse(lease.array.tensor.data)
        self.assertEqual(lease.array.tensor.strides[0], 1)

    def test_ordinal_selection_is_explicit_and_has_no_source_version(self) -> None:
        self.assertEqual(OrdinalSelection.head().to_raw().kind, raw.OrdinalSelectionKind.HEAD)
        exact = OrdinalSelection.retained_exact(104).to_raw()
        self.assertEqual(exact.kind, raw.OrdinalSelectionKind.RETAINED_EXACT)
        self.assertEqual(exact.source_ordinal, 104)
        admitted = OrdinalSelection.admitted_exact(105).to_raw()
        self.assertEqual(admitted.kind, raw.OrdinalSelectionKind.ADMITTED_EXACT)
        self.assertEqual(admitted.source_ordinal, 105)
        with self.assertRaises(ValueError):
            OrdinalSelection.admitted_exact(-1).to_raw()

    def test_population_config_carries_pending_publication_capacity(self) -> None:
        default = PopulationConfig().to_raw()
        self.assertEqual(
            (default.retained_ordinal_capacity, default.pending_publication_capacity), (2, 1)
        )
        pipelined = PopulationConfig(3, pending_publication_capacity=2).to_raw()
        self.assertEqual(pipelined.pending_publication_capacity, 2)

    def test_integers_outside_their_c_field_are_refused(self) -> None:
        # ctypes would store 2**32 in a uint32 field as 0 and 2**32 + 1 as 1.
        target = TargetRef(raw.TargetKind.POPULATION, 17)
        for label, make in (
            ("retained_ordinal_capacity", lambda n: PopulationConfig(n)),
            (
                "result_slot_count",
                lambda n: RaycastPlanConfig(target, HitMode.CLOSEST, 8, 1, 8, result_slot_count=n),
            ),
        ):
            with self.subTest(label):
                self.assertEqual(getattr(make(2**32 - 1).to_raw(), label), 2**32 - 1)
                for value in (2**32, 2**32 + 1):
                    with self.assertRaisesRegex(ValueError, label):
                        make(value).to_raw()
        # Every query and launch builds an ordinal selection, so its wrapper checks the ordinal
        # instead of the descriptor.
        for select in (OrdinalSelection.retained_exact, OrdinalSelection.admitted_exact):
            self.assertEqual(select(2**63 - 1).to_raw().source_ordinal, 2**63 - 1)
            with self.assertRaisesRegex(ValueError, "source ordinal"):
                select(2**63).to_raw()
        # Every integer field of every descriptor a configuration object builds stores each value
        # its C type holds and refuses every other, whether set as an attribute or passed to the
        # constructor.
        descriptors = {
            type(config.to_raw())
            for config in (
                PopulationConfig(),
                Device.cuda(0),
                DeviceBudget(Device.cuda(0), 1),
                RaycastPlanConfig(target, HitMode.CLOSEST, 1, 1, 1),
                SphereSweepPlanConfig(target, 1.0, 1),
                BoxSweepPlanConfig(target, (1.0, 1.0, 1.0), 1),
                OverlapPlanConfig(target, OverlapBatchShape.sphere(1.0), HitMode.ANY, 1, 1, 1),
                OverlapBatchShape.sphere(1.0),
                PairReporterConfig(target, target, 1, 1),
            )
        } | {type(InstanceConfig().to_raw()[0])}
        probes = sorted(
            {
                sign * 2**bits + step
                for bits in (7, 8, 15, 16, 31, 32, 63, 64)
                for sign in (1, -1)
                for step in (-1, 0, 1)
            }
        )
        integer_types = (ctypes.c_int8, ctypes.c_int16, ctypes.c_int32, ctypes.c_int64)
        integer_types += (ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64)
        for structure in descriptors:
            for field, field_type, *_ in structure._fields_:
                if field_type not in integer_types:
                    continue
                with self.subTest(f"{structure.__name__}.{field}"):
                    for probe in probes:
                        # A bare ctypes value of the field's type says what the field can hold.
                        if field_type(probe).value == probe:
                            value = structure()
                            setattr(value, field, probe)
                            self.assertEqual(getattr(value, field), probe)
                            self.assertEqual(getattr(structure(**{field: probe}), field), probe)
                        else:
                            with self.assertRaisesRegex(ValueError, field):
                                setattr(structure(), field, probe)
                            with self.assertRaisesRegex(ValueError, field):
                                structure(**{field: probe})

    def test_instance_config_matches_public_api(self) -> None:
        descriptor, keepalive = InstanceConfig(
            operation_capacity=3, max_cpu_concurrency=1, host_byte_budget=4096
        ).to_raw()
        self.assertEqual(
            (
                descriptor.operation_capacity,
                descriptor.max_cpu_concurrency,
                descriptor.host_byte_budget,
            ),
            (3, 1, 4096),
        )
        self.assertEqual(keepalive, ())

    def test_family_plan_and_reporter_descriptors_are_distinct(self) -> None:
        population = TargetRef(raw.TargetKind.POPULATION, 17)
        view = TargetRef(raw.TargetKind.VIEW, 23)
        plan = RaycastPlanConfig(population, HitMode.CLOSEST, 8, 1, 8)
        self.assertEqual(plan.to_raw().target.handle.population, 17)
        reporter = PairReporterConfig(population, view, 16, 8)
        projected = reporter.to_raw()
        self.assertEqual(projected.second.handle.view, 23)
        with self.assertRaises(FrozenInstanceError):
            reporter.delta_pair_capacity = 9

    def test_participation_bits_are_an_int_below_2_to_the_128(self) -> None:
        target = TargetRef(raw.TargetKind.POPULATION, 17)
        # Bits 0-63 land in words[0] and bits 64-127 in words[1], as in the C struct.
        bits = (1 << 127) | (1 << 64) | 0x5
        common = RaycastQuery((0, 0, 0), (1, 0, 0), 1.0, query_bits=bits).to_raw(target).common
        self.assertEqual(list(common.query_bits.words), [0x5, (1 << 63) | 1])
        reporter = PairReporterConfig(target, target, 1, 1, second_query_bits=1 << 100).to_raw()
        self.assertEqual(list(reporter.second_query_bits.words), [0, 1 << 36])
        for invalid, error in (
            (1 << 128, ValueError),
            (-1, ValueError),
            (True, TypeError),
            (1.0, TypeError),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(error):
                RaycastQuery((0, 0, 0), (1, 0, 0), 1.0, query_bits=invalid).to_raw(target)

    def test_pair_reporter_allows_zero_pair_capacities_only(self) -> None:
        target = TargetRef(raw.TargetKind.POPULATION, 17)
        projected = PairReporterConfig(target, target, 0, 0).to_raw()
        self.assertEqual((projected.active_pair_capacity, projected.delta_pair_capacity), (0, 0))
        with self.assertRaises(ValueError):
            PairReporterConfig(target, target, 1, 1, result_slot_count=0).to_raw()

    def test_family_plans_allow_zero_bounded_work(self) -> None:
        target = TargetRef(raw.TargetKind.POPULATION, 17)
        raycast = RaycastPlanConfig(target, HitMode.MULTIPLE, 0, 0, 0).to_raw()
        sweep = SphereSweepPlanConfig(target, 1.0, 0).to_raw()
        box_sweep = BoxSweepPlanConfig(target, (0.5, 1.0, 1.5), 0).to_raw()
        overlap = OverlapPlanConfig(
            target, OverlapBatchShape.sphere(1.0), HitMode.MULTIPLE, 0, 0, 0
        ).to_raw()
        self.assertEqual(
            (
                raycast.maximum_query_count,
                raycast.hit_capacity_per_slot,
                sweep.maximum_query_count,
                sweep.hit_capacity_per_slot,
                box_sweep.maximum_query_count,
                box_sweep.hit_capacity_per_slot,
                overlap.maximum_query_count,
                overlap.hit_capacity_per_slot,
            ),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
        with self.assertRaises(ValueError):
            RaycastPlanConfig(target, HitMode.MULTIPLE, 0, 0, 0, result_slot_count=0).to_raw()
        # Every mesh crossing is the default; NEAREST asks for one hit per object.
        self.assertEqual(raycast.mesh_hits, MeshHits.ALL)
        nearest = RaycastPlanConfig(
            target, HitMode.MULTIPLE, 0, 0, 0, mesh_hits=MeshHits.NEAREST
        ).to_raw()
        self.assertEqual(nearest.mesh_hits, MeshHits.NEAREST)
        # Sweeps project CLOSEST, one hit per row, by default; MULTIPLE takes its own
        # per-query limit and slot capacity.
        self.assertEqual((sweep.hit_mode, box_sweep.hit_mode), (HitMode.CLOSEST, HitMode.CLOSEST))
        self.assertEqual((sweep.maximum_hits_per_query, box_sweep.maximum_hits_per_query), (1, 1))
        any_sweep = SphereSweepPlanConfig(target, 1.0, 0, hit_mode=HitMode.ANY).to_raw()
        self.assertEqual(any_sweep.hit_mode, HitMode.ANY)
        multiple = BoxSweepPlanConfig(
            target,
            (0.5, 1.0, 1.5),
            2,
            hit_mode=HitMode.MULTIPLE,
            maximum_hits_per_query=3,
            hit_capacity_per_slot=5,
        ).to_raw()
        self.assertEqual(
            (multiple.hit_mode, multiple.maximum_hits_per_query, multiple.hit_capacity_per_slot),
            (HitMode.MULTIPLE, 3, 5),
        )
        with self.assertRaises(ValueError):
            SphereSweepPlanConfig(target, 1.0, 0, maximum_hits_per_query=-1).to_raw()
        # ctypes would wrap 2**64 + 1 to 1; the configs refuse it instead.
        with self.assertRaises(ValueError):
            SphereSweepPlanConfig(target, 1.0, 0, maximum_hits_per_query=2**64 + 1).to_raw()
        with self.assertRaises(ValueError):
            BoxSweepPlanConfig(target, (1.0, 1.0, 1.0), 0, hit_capacity_per_slot=2**64 + 1).to_raw()

    def test_overlap_plan_shape_sizes_must_be_finite_and_positive(self) -> None:
        target = TargetRef(raw.TargetKind.POPULATION, 17)
        for shape in (
            OverlapBatchShape.sphere(0.0),
            OverlapBatchShape.sphere(float("nan")),
            OverlapBatchShape.box((0.5, -0.5, 0.5)),
            OverlapBatchShape(OverlapShapeKind.AABB, half_extents=(0.5, 0.5, 0.0)),
        ):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    OverlapPlanConfig(target, shape, HitMode.ANY, 1, 1, 1).to_raw()
        OverlapPlanConfig(target, OverlapBatchShape.sphere(0.5), HitMode.ANY, 1, 1, 1).to_raw()

    def test_family_plan_result_field_rules_fail_closed(self) -> None:
        target = TargetRef(raw.TargetKind.POPULATION, 17)
        with self.assertRaisesRegex(ValueError, "QUERY_POSITION"):
            RaycastPlanConfig(
                target,
                HitMode.CLOSEST,
                1,
                1,
                1,
                result_fields=BatchResultField.QUERY_POSITION,
            ).to_raw()
        with self.assertRaisesRegex(ValueError, "SOURCE_FEATURE"):
            SphereSweepPlanConfig(
                target,
                1.0,
                1,
                result_fields=BatchResultField.SOURCE_FEATURE,
            ).to_raw()
        with self.assertRaisesRegex(ValueError, "SOURCE_FEATURE"):
            BoxSweepPlanConfig(
                target,
                (0.5, 1.0, 1.5),
                1,
                result_fields=BatchResultField.SOURCE_FEATURE,
            ).to_raw()
        shape = OverlapBatchShape.aabb((1.0, 1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "GEOMETRY_IDENTITY"):
            OverlapPlanConfig(
                target,
                shape,
                HitMode.ANY,
                1,
                1,
                1,
                result_fields=BatchResultField.POSITION,
            ).to_raw()
        allowed = OverlapPlanConfig(
            target,
            shape,
            HitMode.ANY,
            1,
            1,
            1,
            result_fields=BatchResultField.GEOMETRY_IDENTITY,
        ).to_raw()
        self.assertEqual(allowed.result_fields, BatchResultField.GEOMETRY_IDENTITY)

    def test_box_sweep_projection_uses_fixed_pose_and_positive_half_extents(self) -> None:
        target = TargetRef(raw.TargetKind.POPULATION, 17)
        query = BoxSweepQuery(
            center=(1.0, 2.0, 3.0),
            half_extents=(0.5, 1.0, 1.5),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            unit_direction=(0.0, 0.0, -1.0),
            min_distance=0.25,
            max_distance=4.0,
        ).to_raw(target)
        self.assertEqual(tuple(query.sweep.half_extents), (0.5, 1.0, 1.5))
        self.assertEqual(tuple(query.sweep.rotation_xyzw), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual((query.sweep.min_distance, query.sweep.max_distance), (0.25, 4.0))
        plan = BoxSweepPlanConfig(target, (0.5, 1.0, 1.5), 3).to_raw()
        self.assertEqual(tuple(plan.half_extents), (0.5, 1.0, 1.5))
        self.assertEqual(plan.hit_capacity_per_slot, 3)
        self.assertIsInstance(BoxSweepResult.from_raw(raw.BoxSweepClosestResult()), BoxSweepResult)
        with self.assertRaisesRegex(ValueError, "positive"):
            BoxSweepQuery((0, 0, 0), (0.5, 0, 1.5), (0, 0, 0, 1), (0, 0, 1), 1).to_raw(target)
        with self.assertRaisesRegex(ValueError, "positive"):
            BoxSweepPlanConfig(target, (0.5, -1, 1.5), 3).to_raw()

    def test_publication_result_equality_and_hash_are_stable(self) -> None:
        objects = (ctypes.c_uint64 * 2)(37, 99)
        reasons = (ctypes.c_int32 * 2)(
            raw.OvstageExclusionReason.MALFORMED_GEOMETRY,
            raw.OvstageExclusionReason.UNSUPPORTED_SHAPE,
        )
        shape = (ctypes.c_int64 * 1)(2)
        result = raw.PopulationUpdateResult(
            110,
            8,
            2,
            raw.DLTensor(
                data=ctypes.cast(objects, ctypes.c_void_p),
                device=raw.DLDevice(raw.K_DL_CPU, 0),
                ndim=1,
                dtype=raw.DLDataType(raw.K_DL_UINT, 64, 1),
                shape=shape,
            ),
            raw.DLTensor(
                data=ctypes.cast(reasons, ctypes.c_void_p),
                device=raw.DLDevice(raw.K_DL_CPU, 0),
                ndim=1,
                dtype=raw.DLDataType(raw.K_DL_INT, 32, 1),
                shape=shape,
            ),
        )
        first = PublicationResult.from_raw(result)
        same = PublicationResult.from_raw(result)
        self.assertEqual(first, same)
        self.assertEqual(hash(first), hash(same))
        reasons[1] = raw.OvstageExclusionReason.NO_SPATIAL_SOURCE
        changed = PublicationResult.from_raw(result)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, object())
        self.assertEqual(first.excluded_objects.tolist(), [37, 99])
        self.assertEqual(
            first.exclusion_reasons.tolist(),
            [
                raw.OvstageExclusionReason.MALFORMED_GEOMETRY,
                raw.OvstageExclusionReason.UNSUPPORTED_SHAPE,
            ],
        )
        first.excluded_objects[0] = 0
        self.assertEqual(first, same)
        self.assertEqual(hash(first), hash(same))

    def test_population_values_copy_ordinal_truth(self) -> None:
        info = PopulationInfo.from_raw(raw.PopulationInfo(1, 104, 7, 2, 0))
        self.assertEqual((info.head_ordinal, info.retained_ordinal_count), (104, 2))
        shape = (ctypes.c_int64 * 1)(1)
        objects = (ctypes.c_uint64 * 1)(2**64 - 1)
        reasons = (ctypes.c_int32 * 1)(raw.OvstageExclusionReason.MALFORMED_GEOMETRY)
        original_import = __import__

        def without_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise AssertionError("reading a publication result imported NumPy")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_numpy):
            publication = PublicationResult.from_raw(
                raw.PopulationUpdateResult(
                    110,
                    8,
                    1,
                    raw.DLTensor(
                        data=ctypes.cast(objects, ctypes.c_void_p),
                        device=raw.DLDevice(raw.K_DL_CPU, 0),
                        ndim=1,
                        dtype=raw.DLDataType(raw.K_DL_UINT, 64, 1),
                        shape=shape,
                    ),
                    raw.DLTensor(
                        data=ctypes.cast(reasons, ctypes.c_void_p),
                        device=raw.DLDevice(raw.K_DL_CPU, 0),
                        ndim=1,
                        dtype=raw.DLDataType(raw.K_DL_INT, 32, 1),
                        shape=shape,
                    ),
                )
            )
            self.assertEqual(
                (publication.published_ordinal, publication.excluded_object_count), (110, 1)
            )
            # The columns own their values: overwriting the native storage changes neither.
            objects[0] = 99
            reasons[0] = raw.OvstageExclusionReason.UNSUPPORTED_SHAPE
            for column, values in (
                (publication.excluded_objects, [2**64 - 1]),
                (publication.exclusion_reasons, [raw.OvstageExclusionReason.MALFORMED_GEOMETRY]),
            ):
                self.assertEqual(len(column), 1)
                self.assertEqual(column[0], values[0])
                self.assertEqual(column.tolist(), values)
                self.assertIsInstance(column, array.array)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
    def test_publication_columns_are_zero_copy_numpy_inputs(self) -> None:
        import numpy

        shape = (ctypes.c_int64 * 1)(2)
        objects = (ctypes.c_uint64 * 2)(37, 2**64 - 1)
        reasons = (ctypes.c_int32 * 2)(-1, raw.OvstageExclusionReason.UNSUPPORTED_SHAPE)
        publication = PublicationResult.from_raw(
            raw.PopulationUpdateResult(
                110,
                8,
                2,
                raw.DLTensor(
                    data=ctypes.cast(objects, ctypes.c_void_p),
                    device=raw.DLDevice(raw.K_DL_CPU, 0),
                    ndim=1,
                    dtype=raw.DLDataType(raw.K_DL_UINT, 64, 1),
                    shape=shape,
                ),
                raw.DLTensor(
                    data=ctypes.cast(reasons, ctypes.c_void_p),
                    device=raw.DLDevice(raw.K_DL_CPU, 0),
                    ndim=1,
                    dtype=raw.DLDataType(raw.K_DL_INT, 32, 1),
                    shape=shape,
                ),
            )
        )
        for column, dtype, values in (
            (publication.excluded_objects, numpy.uint64, [37, 2**64 - 1]),
            (
                publication.exclusion_reasons,
                numpy.int32,
                [-1, raw.OvstageExclusionReason.UNSUPPORTED_SHAPE],
            ),
        ):
            view = numpy.asarray(column)
            self.assertEqual(view.dtype, numpy.dtype(dtype))
            self.assertEqual(view.tolist(), values)
            column[0] = 7
            self.assertEqual(view[0], 7)


if __name__ == "__main__":
    unittest.main()
