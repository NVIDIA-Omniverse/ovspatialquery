# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import array
import ctypes
import gc
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from tests.ovstage_population import populate
from ovspatialquery import (
    Aabb,
    ApiError,
    ApiStatus,
    Capability,
    GeometryDefinitionRef,
    GeometryType,
    Library,
    OverlapQuery,
    OvstageExclusionReason,
    RaycastQuery,
    Sphere,
)
from ovspatialquery.raw import Preparation, QueryPrecision
from ovspatialquery.types import InstanceConfig, Device, DeviceBudget, PairReporterConfig
from ovspatialquery.ovstage import OvstageUnavailable, native_stage_pointer

try:
    import numpy
    import ovstage
except ImportError:
    ovstage = None


class _Stage:
    def __init__(self, value: int) -> None:
        self._storage = ctypes.c_uint64(value)
        self._inst = ctypes.pointer(self._storage)

    def _require_inst(self):
        return self._inst


def _path_int(path) -> int:
    """A prim path's value: Ovstage 0.3 interns a path as an int, 0.4 as a PrimPath."""
    return int(getattr(path, "raw_id", path))


def _prim_path(value):
    """The path dictionary's handle for an object ID's value."""
    prim_path = getattr(ovstage, "PrimPath", None)
    return value if prim_path is None or isinstance(value, prim_path) else prim_path(value)


def _ovstage(version, ordinal):
    """A stand-in ovstage module whose bindings declare `ordinal` as ovstage_ordinal_t."""
    bindings = SimpleNamespace() if ordinal is None else SimpleNamespace(ovstage_ordinal_t=ordinal)
    return SimpleNamespace(__version__=version, _src=SimpleNamespace(bindings=bindings))


class OvstagePointerQualificationTests(unittest.TestCase):
    def test_listed_families_and_consistent_private_pointer_are_required(self) -> None:
        stage = _Stage(9)
        for version in (
            "0.3.0.1791986420",
            "0.3.9+local",
            "0.4.0.1794520352",
            "0.4.0rc1",
            "0.4.2",
        ):
            with self.subTest(version=version):
                self.assertEqual(
                    native_stage_pointer(_ovstage(version, ctypes.c_int64), stage),
                    ctypes.cast(stage._inst, ctypes.c_void_p).value,
                )

    def test_pre_release_04_builds_with_unsigned_ordinals_fail_closed(self) -> None:
        for version in ("0.4.0.1793320890", "0.4.0.dev1793837579+main.39a7302f"):
            for ordinal in (ctypes.c_uint64, None):
                with self.subTest(version=version, ordinal=ordinal):
                    with self.assertRaisesRegex(OvstageUnavailable, "pre-release 0.4"):
                        native_stage_pointer(_ovstage(version, ordinal), _Stage(4))

    def test_other_families_and_malformed_versions_fail_closed(self) -> None:
        for version in (
            "0.1.1.355824",
            "0.2.1.385922",
            "0.5.0",
            "0.5.0.dev1793837579+main.39a7302f",
            "1.3.0",
            "0.30.0",
            "0.3",
            "0.3.invalid",
            None,
        ):
            with self.subTest(version=version):
                with self.assertRaisesRegex(OvstageUnavailable, "0.3.x, 0.4.x"):
                    native_stage_pointer(SimpleNamespace(__version__=version), _Stage(3))

    def test_inconsistent_private_pointer_fails_closed(self) -> None:
        module = SimpleNamespace(__version__="0.3.0.1791986420")
        stage = _Stage(5)
        other = ctypes.c_uint64(8)
        stage._inst = ctypes.pointer(other)
        stage._require_inst = lambda: ctypes.pointer(stage._storage)
        with self.assertRaisesRegex(OvstageUnavailable, "invalid"):
            native_stage_pointer(module, stage)


def _installed_library():
    """The library OVSPATIALQUERY_LIBRARY names, else the bootstrap install, else the default."""
    install = Path(__file__).resolve().parents[2] / "_build/bootstrap/native-stable/install"
    artifact = install / (
        "bin/ovspatialquery.dll" if sys.platform == "win32" else "lib/libovspatialquery.so"
    )
    return os.environ.get("OVSPATIALQUERY_LIBRARY") or (artifact if artifact.is_file() else None)


def _enqueue_set_or_skip_unavailable_cuda(population, ordinal, execution):
    try:
        return population.enqueue_set(ordinal, **execution)
    except ApiError as refused:
        if (
            execution.get("preparation", Preparation.ON_HOST) != Preparation.ON_HOST
            and refused.status == ApiStatus.NOT_SUPPORTED
            and refused.detail.startswith("CUDA is unavailable")
        ):
            raise unittest.SkipTest(refused.detail) from refused
        raise


@unittest.skipIf(ovstage is None, "Ovstage is an optional dependency")
class OvstageFilterAndParticipationTests(unittest.TestCase):
    def test_unsupported_capsule_axis_has_shape_reason_on_host_and_device(self) -> None:
        cuda = _cuda_device_present()
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        source = ovstage.Stage("unsupported-capsule-axis")
        try:
            populate(
                source,
                '#usda 1.0\ndef Xform "World" {\n def Capsule "Bad" {\n uniform token axis = "Q"\n double radius = 0.5\n double height = 2\n }\n}\n',
                1,
                0.0,
            )
            source.advance_write_floor(ordinal=1).wait()
            source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
            selected = ovstage.Filter(
                [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ["/World/Bad"])]
            )
            with source.query(filter=selected) as query:
                query.wait()
                source.write_attribute(
                    query,
                    "_worldExtent",
                    2,
                    numpy.array([[-1, -2, -1, 1, 2, 1]], dtype=numpy.float64),
                    is_array=False,
                ).wait()
            source.advance_write_floor(ordinal=2).wait()
            expected = _path_int(ovstage.PathDictionary(source).intern_path("/World/Bad"))
            with (
                Library.open(_installed_library(), require_ovstage=True) as library,
                library.instance(config) as instance,
            ):
                adapter = instance.ovstage(ovstage, source)
                for execution in executions:
                    with self.subTest(preparation=execution["preparation"]):
                        with adapter.create_population(selected) as population:
                            with _enqueue_set_or_skip_unavailable_cuda(
                                population, 2, execution
                            ) as operation:
                                operation.wait()
                                result = operation.fetch_population_update()
                                self.assertEqual(result.object_count, 0)
                                numpy.testing.assert_array_equal(
                                    result.excluded_objects, [expected]
                                )
                                numpy.testing.assert_array_equal(
                                    result.exclusion_reasons,
                                    [OvstageExclusionReason.UNSUPPORTED_SHAPE],
                                )
        finally:
            source.destroy()

    def test_unsupported_types_precede_missing_extent_on_host_and_device(self) -> None:
        cuda = _cuda_device_present()
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        source = ovstage.Stage("unsupported-type-without-extent")
        try:
            populate(
                source,
                '#usda 1.0\ndef Scope "S" {}\ndef Material "M" {}\ndef Cube "C" {\n double size = 1\n}\ndef Cube "D" {\n double size = 1\n}\n',
                1,
                0.0,
            )
            source.advance_write_floor(ordinal=1).wait()
            source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
            cube = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ["/C"])])
            with source.query(filter=cube) as query:
                query.wait()
                source.write_attribute(
                    query,
                    "omni:fabric:worldMatrix",
                    2,
                    numpy.eye(4, dtype=numpy.float64).reshape(1, 16),
                    is_array=False,
                ).wait()
            missing = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ["/D"])])
            extreme = float(numpy.finfo(numpy.float32).max)
            with source.query(filter=missing) as query:
                query.wait()
                source.write_attribute(
                    query,
                    "_worldExtent",
                    2,
                    numpy.array([[extreme] * 3 + [-extreme] * 3], dtype=numpy.float64),
                    is_array=False,
                ).wait()
            source.advance_write_floor(ordinal=2).wait()
            selected = ovstage.Filter(
                [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ["/S", "/M", "/C", "/D"])]
            )
            paths = ovstage.PathDictionary(source)
            expected = {
                _path_int(paths.intern_path("/S")): OvstageExclusionReason.UNSUPPORTED_TYPE,
                _path_int(paths.intern_path("/M")): OvstageExclusionReason.UNSUPPORTED_TYPE,
                _path_int(paths.intern_path("/D")): OvstageExclusionReason.NO_SPATIAL_SOURCE,
            }
            with (
                Library.open(_installed_library(), require_ovstage=True) as library,
                library.instance(config) as instance,
            ):
                adapter = instance.ovstage(ovstage, source)
                for execution in executions:
                    with self.subTest(preparation=execution["preparation"]):
                        with adapter.create_population(selected) as population:
                            with _enqueue_set_or_skip_unavailable_cuda(
                                population, 2, execution
                            ) as operation:
                                operation.wait()
                                update = operation.fetch_population_update()
                                self.assertEqual(
                                    (update.object_count, update.excluded_object_count), (1, 3)
                                )
                                self.assertEqual(
                                    dict(
                                        zip(
                                            update.excluded_objects.tolist(),
                                            update.exclusion_reasons.tolist(),
                                        )
                                    ),
                                    expected,
                                )
        finally:
            source.destroy()

    def test_present_extent_values_do_not_control_publication_on_host_or_device(self) -> None:
        cuda = _cuda_device_present()
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        source = ovstage.Stage("present-extent-values")
        try:
            populate(source, '#usda 1.0\ndef Cube "Shape" {\n double size = 2\n}\n', 1, 0.0)
            source.advance_write_floor(ordinal=1).wait()
            source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
            selected = ovstage.Filter(
                [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ["/Shape"])]
            )
            with (
                Library.open(_installed_library(), require_ovstage=True) as library,
                library.instance(config) as instance,
            ):
                adapter = instance.ovstage(ovstage, source)
                populations = [
                    (adapter.create_population(selected), execution) for execution in executions
                ]
                device_supported = False
                extreme = float(numpy.finfo(numpy.float32).max)
                for ordinal, extent, present in (
                    (2, [float("inf"), -1, -1, 1, 1, 1], True),
                    (3, [2, -1, -1, 1, 1, 1], True),
                    (4, [extreme] * 3 + [-extreme] * 3, False),
                    (5, [-1, -1, -1, 1, 1, 1], True),
                    (6, [extreme] * 3 + [-extreme] * 3, False),
                ):
                    with source.query(filter=selected) as query:
                        query.wait()
                        source.write_attribute(
                            query,
                            "_worldExtent",
                            ordinal,
                            numpy.array([extent], dtype=numpy.float64),
                            is_array=False,
                        ).wait()
                    source.advance_write_floor(ordinal=ordinal).wait()
                    for population, execution in populations:
                        if (
                            ordinal > 2
                            and execution["preparation"] != Preparation.ON_HOST
                            and not device_supported
                        ):
                            continue
                        with self.subTest(ordinal=ordinal, preparation=execution["preparation"]):
                            enqueue = (
                                _enqueue_set_or_skip_unavailable_cuda(
                                    population, ordinal, execution
                                )
                                if ordinal == 2
                                else population.enqueue_update(ordinal - 1, ordinal, **execution)
                            )
                            if execution["preparation"] != Preparation.ON_HOST:
                                device_supported = True
                            with enqueue as operation:
                                operation.wait()
                                update = operation.fetch_population_update()
                                self.assertEqual(
                                    (update.object_count, update.excluded_object_count),
                                    (1, 0) if present else (0, 1),
                                )
                                if not present:
                                    self.assertEqual(
                                        update.exclusion_reasons.tolist(),
                                        [OvstageExclusionReason.NO_SPATIAL_SOURCE],
                                    )
                            self.assertEqual(
                                population.target.overlap_any(
                                    OverlapQuery(Aabb((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))
                                ).hit
                                is not None,
                                present,
                            )
                            self.assertIsNone(
                                population.target.overlap_any(
                                    OverlapQuery(Aabb((10, 10, 10), (11, 11, 11)))
                                ).hit
                            )
                for population, _ in populations:
                    population.close()
        finally:
            source.destroy()

    def test_affine_homogeneous_w_precedes_domain_and_shape_tests(self) -> None:
        cuda = _cuda_device_present()
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        for execution in executions:
            source = ovstage.Stage("affine-homogeneous-w")
            try:
                populate(source, '#usda 1.0\ndef Cube "Shape" {\n double size = 1\n}\n', 1, 0.0)
                source.advance_write_floor(ordinal=1).wait()
                source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
                source.advance_write_floor(ordinal=2).wait()
                selected = ovstage.Filter(
                    [ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Cube"])]
                )
                dictionary = ovstage.PathDictionary(source)
                expected_id = _path_int(dictionary.intern_path("/Shape"))
                with (
                    Library.open(_installed_library(), require_ovstage=True) as library,
                    library.instance(config) as instance,
                ):
                    adapter = instance.ovstage(ovstage, source)
                    with adapter.create_population(selected) as population:
                        with _enqueue_set_or_skip_unavailable_cuda(population, 2, execution) as op:
                            op.wait()
                            self.assertEqual(op.fetch_population_update().object_count, 1)
                        base = 2
                        cases = [
                            (w, 0.0, 3.0, True)
                            for w in (2.0, -2.0, 1.0 - 2.0**-53, 2.0**61, 2.0**-61)
                        ]
                        cases += [
                            (1.0, 2.0**-100, 3.0, False),
                            (0.0, 0.0, 3.0, False),
                            (float("inf"), 0.0, 3.0, False),
                            (float("nan"), 0.0, 3.0, False),
                            (0.5, 0.0, 2.0**61, False),
                        ]
                        for w, perspective, x, valid in cases:
                            target = base + 1
                            matrix = numpy.eye(4, dtype=numpy.float64)
                            if numpy.isfinite(w):
                                matrix *= w
                            matrix[3, 3] = w
                            matrix[3, 0] = x * w if numpy.isfinite(w) else x
                            matrix[0, 3] = perspective
                            with source.query(filter=selected) as query:
                                query.wait()
                                source.write_attribute(
                                    query,
                                    "omni:fabric:worldMatrix",
                                    target,
                                    matrix.reshape(1, 16),
                                    is_array=False,
                                ).wait()
                                source.write_attribute(
                                    query,
                                    "_worldExtent",
                                    target,
                                    numpy.array(
                                        [[2.5, -0.5, -0.5, 3.5, 0.5, 0.5]], dtype=numpy.float64
                                    ),
                                    is_array=False,
                                ).wait()
                            source.advance_write_floor(ordinal=target).wait()
                            with self.subTest(
                                execution=execution, w=w, perspective=perspective, x=x
                            ):
                                with adapter.create_population(selected) as fresh:
                                    for op in (
                                        population.enqueue_update(base, target, **execution),
                                        fresh.enqueue_set(target, **execution),
                                    ):
                                        with op:
                                            op.wait()
                                            value = op.fetch_population_update()
                                            self.assertEqual(
                                                (
                                                    value.object_count,
                                                    value.excluded_objects.tolist(),
                                                    value.exclusion_reasons.tolist(),
                                                ),
                                                (1, [], [])
                                                if valid
                                                else (
                                                    0,
                                                    [expected_id],
                                                    [
                                                        int(
                                                            OvstageExclusionReason.MALFORMED_GEOMETRY
                                                        )
                                                    ],
                                                ),
                                            )
                                    if valid:
                                        for published in (population, fresh):
                                            hit = published.target.raycast_closest(
                                                RaycastQuery(
                                                    origin=(1.0, 0.0, 0.0),
                                                    unit_direction=(1.0, 0.0, 0.0),
                                                    max_distance=4.0,
                                                )
                                            )
                                            self.assertIsNotNone(hit.hit)
                                            self.assertEqual(hit.hit.object_id, expected_id)
                                            self.assertAlmostEqual(hit.hit.distance, 1.5, places=10)
                            base = target
            finally:
                source.destroy()

    def test_publication_exclusions_are_sorted_copies_for_host_and_device(self) -> None:
        usda = """#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Sphere "A"
    {
        double radius = 1
        double3 xformOp:scale = (2, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
    def Sphere "B"
    {
        double radius = 1
        double3 xformOp:scale = (2, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
    def Sphere "C"
    {
        double radius = 1
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
}
"""
        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )

        def selected(*paths):
            return ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])

        def check_result(result, ids, objects):
            self.assertEqual(result.object_count, objects)
            self.assertEqual(result.excluded_object_count, len(ids))
            self.assertIsInstance(result.excluded_objects, array.array)
            self.assertIsInstance(result.exclusion_reasons, array.array)
            self.assertEqual(numpy.asarray(result.excluded_objects).dtype, numpy.uint64)
            self.assertEqual(numpy.asarray(result.exclusion_reasons).dtype, numpy.int32)
            self.assertEqual(result.excluded_objects.tolist(), ids)
            self.assertEqual(
                result.exclusion_reasons.tolist(),
                [OvstageExclusionReason.UNSUPPORTED_SHAPE] * len(ids),
            )

        with (
            Library.open(_installed_library(), require_ovstage=True) as library,
            library.instance(config) as instance,
        ):
            for execution in executions:
                with self.subTest(preparation=execution["preparation"]):
                    source = ovstage.Stage("publication-exclusion-copies")
                    try:
                        populate(source, usda, 1, 0.0)
                        source.advance_write_floor(ordinal=1).wait()
                        source.compute_hierarchy(
                            1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU
                        )
                        source.advance_write_floor(ordinal=2).wait()
                        paths = ovstage.PathDictionary(source)
                        ids = sorted(
                            _path_int(paths.intern_path(path)) for path in ("/World/A", "/World/B")
                        )
                        adapter = instance.ovstage(ovstage, source)
                        with adapter.create_population(selected("/World/C")) as clean:
                            with _enqueue_set_or_skip_unavailable_cuda(
                                clean, 2, execution
                            ) as operation:
                                operation.wait()
                                check_result(operation.fetch_population_update(), [], 1)
                        with adapter.create_population(
                            selected("/World/B", "/World/A", "/World/C")
                        ) as population:
                            population.set_participation_map(1 << 100, {})
                            high_bit_probe = OverlapQuery(
                                Aabb((-3.0,) * 3, (3.0,) * 3), query_bits=1 << 100
                            )
                            with population.enqueue_set(2, **execution) as operation:
                                operation.wait()
                                saved = operation.fetch_population_update()
                                check_result(saved, ids, 1)
                                self.assertEqual(saved.published_ordinal, 2)
                                self.assertEqual(
                                    population.target.overlap_any(high_bit_probe).hit.object_id,
                                    _path_int(paths.intern_path("/World/C")),
                                )
                                repeated = operation.fetch_population_update()
                                check_result(repeated, ids, 1)
                                numpy.testing.assert_array_equal(
                                    repeated.excluded_objects, saved.excluded_objects
                                )
                                numpy.testing.assert_array_equal(
                                    repeated.exclusion_reasons, saved.exclusion_reasons
                                )
                            source.compute_hierarchy(
                                2, 3, ovstage.HierarchyComputationModel.DEFAULT_CPU
                            )
                            source.advance_write_floor(ordinal=3).wait()
                            with population.enqueue_update(2, 3, **execution) as operation:
                                operation.wait()
                                later = operation.fetch_population_update()
                                check_result(later, ids, 1)
                                self.assertEqual(later.published_ordinal, 3)
                            check_result(saved, ids, 1)
                            nonuniform = numpy.diag([2.0, 1.0, 1.0, 1.0])
                            with paths.create_path_list_from_strings(["/World/C"]) as listed:
                                with source.query_from_path_list(listed) as query:
                                    query.wait()
                                    source.write_attribute(
                                        query,
                                        "omni:xform",
                                        4,
                                        nonuniform.reshape(1, 16),
                                        is_array=False,
                                        semantic=ovstage.AttributeSemantic.MATRIX,
                                    ).wait()
                            source.compute_hierarchy(
                                4, 4, ovstage.HierarchyComputationModel.DEFAULT_CPU
                            )
                            source.advance_write_floor(ordinal=4).wait()
                            with population.enqueue_update(3, 4, **execution) as operation:
                                operation.wait()
                                excluded = operation.fetch_population_update()
                                check_result(
                                    excluded,
                                    sorted([*ids, _path_int(paths.intern_path("/World/C"))]),
                                    0,
                                )
                                self.assertEqual(excluded.published_ordinal, 4)
                            check_result(saved, ids, 1)
                            # Restore every slot, then exclude all of them through
                            # a pose-only device update, exercising the full list.
                            all_paths = ["/World/A", "/World/B", "/World/C"]
                            all_ids = sorted(
                                _path_int(paths.intern_path(path)) for path in all_paths
                            )
                            for target, matrix, excluded_ids, count in (
                                (5, numpy.eye(4), [], 3),
                                (6, nonuniform, all_ids, 0),
                            ):
                                with paths.create_path_list_from_strings(all_paths) as listed:
                                    with source.query_from_path_list(listed) as query:
                                        query.wait()
                                        source.write_attribute(
                                            query,
                                            "omni:xform",
                                            target,
                                            numpy.tile(matrix.reshape(1, 16), (3, 1)),
                                            is_array=False,
                                            semantic=ovstage.AttributeSemantic.MATRIX,
                                        ).wait()
                                source.compute_hierarchy(
                                    target, target, ovstage.HierarchyComputationModel.DEFAULT_CPU
                                )
                                source.advance_write_floor(ordinal=target).wait()
                                with population.enqueue_update(
                                    target - 1, target, **execution
                                ) as operation:
                                    operation.wait()
                                    check_result(
                                        operation.fetch_population_update(), excluded_ids, count
                                    )
                                    if count == 0:
                                        self.assertIsNone(
                                            population.target.overlap_any(high_bit_probe).hit
                                        )
                            check_result(saved, ids, 1)
                    finally:
                        source.destroy()

    def test_filters_select_prims_and_listed_paths_take_their_bits(self) -> None:
        usda = """#usda 1.0
def Cube "A"
{
    double size = 2
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
def Cube "B"
{
    double size = 2
    double3 xformOp:translate = (5, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
"""
        source = ovstage.Stage("participation-table-test")
        populate(source, usda, 1, 0.0)
        source.advance_write_floor(ordinal=1).wait()
        source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        source.advance_write_floor(ordinal=2).wait()
        paths = ovstage.PathDictionary(source)
        selected = _installed_library()
        with (
            Library.open(selected, require_ovstage=True) as library,
            library.instance() as instance,
        ):
            adapter = instance.ovstage(ovstage, source)

            def paths_filter(*selected_paths: str):
                return ovstage.Filter(
                    [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, selected_paths)]
                )

            with self.assertRaisesRegex(TypeError, "Filter"):
                adapter.create_population(paths_filter("/A", "/B").to_c()[0])
            # The library copies each filter, so neither filter object is retained.
            population = adapter.create_population(paths_filter("/A", "/B"))
            view = population.create_view(paths_filter("/A"))
            try:
                gc.collect()
                for key in (1.5, True):
                    with self.assertRaisesRegex(TypeError, "participation keys"):
                        population.set_participation_map(0x4, {key: 0x1})
                with self.assertRaisesRegex(ValueError, "128 bits"):
                    population.set_participation_map(0x4, {"/A": 1 << 128})
                # /B also holds bit 100, which only the second word of its entry carries.
                population.set_participation_map(
                    0x4, {_path_int(paths.intern_path("/A")): 0x1, "/B": 0x2 | (1 << 100)}
                )
                operation = population.enqueue_set(2)
                try:
                    operation.wait()
                    self.assertEqual(operation.fetch_population_update().object_count, 2)
                finally:
                    operation.close()
                around = {
                    "/A": Aabb((-0.5,) * 3, (0.5,) * 3),
                    "/B": Aabb((4.5, -0.5, -0.5), (5.5, 0.5, 0.5)),
                }
                for target, path, query_bits, admitted in (
                    (population.target, "/A", 0x1, True),
                    (population.target, "/A", 0x2, False),
                    (population.target, "/B", 0x1, False),
                    (population.target, "/B", 0x2, True),
                    (population.target, "/A", 1 << 100, False),
                    (population.target, "/B", 1 << 100, True),
                    (view.target, "/A", 0x1, True),
                    (view.target, "/B", 0x2, False),
                ):
                    with self.subTest(target=target, path=path, query_bits=query_bits):
                        hit = target.overlap_any(
                            OverlapQuery(around[path], query_bits=query_bits)
                        ).hit
                        self.assertEqual(hit is not None, admitted)
                        if hit is not None:
                            self.assertEqual(paths.path_to_string(_prim_path(hit.object_id)), path)
            finally:
                view.close()
                population.close()

    def test_scopes_narrow_populations_and_views_by_path(self) -> None:
        def cube(name: str, x: float) -> str:
            return (
                f'def Cube "{name}"\n{{\n    double size = 1\n'
                f"    double3 xformOp:translate = ({x}, 0, 0)\n"
                '    uniform token[] xformOpOrder = ["xformOp:translate"]\n}\n'
            )

        usda = (
            '#usda 1.0\ndef Xform "World"\n{\ndef Xform "A"\n{\n'
            + cube("c1", 0)
            + 'def Xform "B"\n{\n'
            + cube("c2", 2)
            + '}\n}\ndef Xform "Env"\n{\n'
            + cube("c4", 6)
            + '}\ndef Xform "Environment"\n{\n'
            + cube("c5", 8)
            + "}\n}\n"
        )
        source = ovstage.Stage("path-scope-test")
        populate(source, usda, 1, 0.0)
        source.advance_write_floor(ordinal=1).wait()
        source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        source.advance_write_floor(ordinal=2).wait()
        paths = ovstage.PathDictionary(source)
        everything = Aabb((-100.0,) * 3, (100.0,) * 3)
        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        with (
            Library.open(_installed_library(), require_ovstage=True) as library,
            library.instance(config) as instance,
        ):
            adapter = instance.ovstage(ovstage, source)
            empty = ovstage.Filter([])
            for include, exclude, error in (
                ("/World", (), TypeError),
                ((1.5,), (), TypeError),
                ((), (True,), TypeError),
                ((-1,), (), ValueError),
            ):
                with self.subTest(include=include, exclude=exclude):
                    with self.assertRaises(error):
                        adapter.create_population(
                            empty, include_paths=include, exclude_paths=exclude
                        )
            with self.assertRaises(ApiError) as raised:
                adapter.create_population(empty, include_paths=(0,))
            self.assertEqual(raised.exception.status, ApiStatus.INVALID_ARGUMENT)

            # Excludes alone mean everything else; paths match by element, so excluding
            # /World/Env keeps /World/Environment. Entries mix strings and path values.
            population = adapter.create_population(
                empty, exclude_paths=("/World/A/B", _path_int(paths.intern_path("/World/Env")))
            )
            view = population.create_view(empty, include_paths=("/World/A", "/World/Env"))
            try:
                operation = population.enqueue_set(2)
                try:
                    operation.wait()
                    update = operation.fetch_population_update()
                    self.assertEqual((update.object_count, update.excluded_object_count), (2, 0))
                finally:
                    operation.close()
                for target, expected in (
                    (population.target, {"/World/A/c1", "/World/Environment/c5"}),
                    (view.target, {"/World/A/c1"}),
                ):
                    result = target.overlap_multiple(OverlapQuery(everything), 8)
                    self.assertFalse(result.truncated)
                    self.assertEqual(
                        {paths.path_to_string(_prim_path(hit.object_id)) for hit in result.hits},
                        expected,
                    )
            finally:
                view.close()
                population.close()

            # A device-prepared publication holds the same prims as the host one.
            if cuda:
                with adapter.create_population(empty, exclude_paths=("/World/A/B",)) as population:
                    with _enqueue_set_or_skip_unavailable_cuda(
                        population,
                        2,
                        dict(
                            preparation=Preparation.ON_DEVICE_HOST_READABLE,
                            device=Device.cuda(0),
                            cuda_stream=1,
                        ),
                    ) as operation:
                        operation.wait()
                        update = operation.fetch_population_update()
                        self.assertEqual(
                            (update.object_count, update.excluded_object_count), (3, 0)
                        )


_INSTANCED = """#usda 1.0
def Xform "Proto"
{
    double3 xformOp:translate = (0, 100, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Cube "Body"%(collision)s
    {
        double size = 1
    }
}
def Xform "Panel"
{
    double3 xformOp:translate = (0, 100, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "Quad"
    {
        point3f[] points = [(0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 3, 2, 1]
    }
}
def Xform "Hull"
{
    double3 xformOp:translate = (0, 100, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "Shape" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        point3f[] points = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
                            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
        uniform token physics:approximation = "convexHull"
    }
}
def Xform "Outer"
{
    double3 xformOp:translate = (0, 100, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Sphere "Ball"
    {
        double radius = 0.5
    }
    def Xform "Inner" (
        instanceable = true
        references = </Proto>
    )
    {
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
def Xform "World"
{
%(instances)s}
"""


def _instance(name: str, prototype: str, x: float, scale: float = 1.0) -> str:
    return f"""    def Xform "{name}" (
        instanceable = true
        references = </{prototype}>
    )
    {{
        double3 xformOp:translate = ({x}, 0, 0)
        double3 xformOp:scale = ({scale}, {scale}, {scale})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
"""


def _instanced_stage(name: str, domain: str, collision: str = "") -> object:
    """Native instances on the +x axis; the authored prototypes sit at y = 100, off it."""
    instances = (
        _instance("I1", "Proto", 5.0)
        + _instance("I2", "Proto", 8.0, 2.0)
        + _instance("P1", "Panel", 12.0)
        + _instance("P2", "Panel", 14.0)
        + _instance("N1", "Outer", 20.0)
    )
    instances += _instance("H1", "Hull", 30.0) + _instance("H2", "Hull", 33.0)
    source = ovstage.Stage(name)
    usda = _INSTANCED % {"collision": collision, "instances": instances}
    populate(source, usda, 1, 0.0, domain)
    source.advance_write_floor(ordinal=1).wait()
    source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
    source.advance_write_floor(ordinal=2).wait()
    return source


@unittest.skipIf(ovstage is None, "Ovstage and OpenUSD are optional dependencies")
class OvstageNativeInstancingTests(unittest.TestCase):
    """ADR-0050: each instance proxy is an object at its own path and pose with its prototype
    prim's geometry; Ovstage's prototype copies under /__Prototype_* are never published."""

    RAY = RaycastQuery(origin=(-50.0, 0.0, 0.0), unit_direction=(1.0, 0.0, 0.0), max_distance=200.0)

    def _hits(self, target, paths) -> dict[str, tuple[float, int]]:
        result = target.raycast_multiple(self.RAY, 32)
        self.assertFalse(result.truncated)
        return {
            paths.path_to_string(_prim_path(hit.object_id)): (
                round(hit.distance, 6),
                hit.geometry_value,
            )
            for hit in result.hits
        }

    def test_distinct_scaled_proxies_keep_all_homogeneous_affine_rows(self) -> None:
        # Original distinct-shape pilot: the pinned SDK emits w=1-2^-53 on
        # 148 rows. Check every object and surface rather than pinning SDK bits.
        n = 1000
        source_text = """#usda 1.0
 def Xform "Proto" {
     def Mesh "Tri" {
         uniform token orientation = "rightHanded"
         uniform bool doubleSided = true
         uniform token subdivisionScheme = "none"
         point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
         int[] faceVertexCounts = [3]
         int[] faceVertexIndices = [0,1,2]
         double3 xformOp:translate = (0, 0, 0)
         uniform token[] xformOpOrder = ["xformOp:translate"]
     }
 }
 def Xform "World" {
"""
        for i in range(n):
            source_text += f"""def Xform "I{i}" (\n instanceable = true\n references = </Proto>\n) {{
                double3 xformOp:translate = ({3 * i}, 0, 0)
                double3 xformOp:scale = ({1 + i / n:.17g}, 1, 1)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }}\n"""
        source_text += "}\n"
        source = ovstage.Stage("distinct-proxy-homogeneous-w")
        try:
            populate(source, source_text, 1, 0.0)
            source.advance_write_floor(ordinal=1).wait()
            source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
            source.advance_write_floor(ordinal=2).wait()
            cuda = _cuda_device_present()
            executions = [dict(preparation=Preparation.ON_HOST)]
            if cuda:
                executions.append(
                    dict(
                        preparation=Preparation.ON_DEVICE_HOST_READABLE,
                        device=Device.cuda(0),
                        cuda_stream=1,
                    )
                )
            config = InstanceConfig(
                device_budgets=(DeviceBudget(Device.cuda(0), 128 << 20),) if cuda else ()
            )
            selected = ovstage.Filter(
                [ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["MeshInstance"])]
            )
            dictionary = ovstage.PathDictionary(source)
            ids = [_path_int(dictionary.intern_path(f"/World/I{i}/Tri")) for i in range(n)]
            with (
                Library.open(_installed_library(), require_ovstage=True) as library,
                library.instance(config) as instance,
            ):
                adapter = instance.ovstage(ovstage, source)
                for execution in executions:
                    with (
                        self.subTest(execution=execution),
                        adapter.create_population(selected) as population,
                    ):
                        with _enqueue_set_or_skip_unavailable_cuda(population, 2, execution) as op:
                            op.wait()
                            value = op.fetch_population_update()
                            self.assertEqual(
                                (
                                    value.object_count,
                                    value.excluded_objects.tolist(),
                                    value.exclusion_reasons.tolist(),
                                ),
                                (n, [], []),
                            )
                        result = population.target.overlap_multiple(
                            OverlapQuery(
                                Aabb((-1.0, -1.0, -1.0), (3.0 * n, 2.0, 1.0)),
                                precision=QueryPrecision.BOUNDS_CANDIDATE,
                            ),
                            n + 1,
                        )
                        self.assertFalse(result.truncated)
                        self.assertEqual(sorted(hit.object_id for hit in result.hits), sorted(ids))
                        references = []
                        for i, expected_id in enumerate(ids):
                            hit = population.target.raycast_closest(
                                RaycastQuery(
                                    origin=(3.0 * i + 0.25 * (1 + i / n), 0.25, 1.0),
                                    unit_direction=(0.0, 0.0, -1.0),
                                    max_distance=2.0,
                                )
                            )
                            self.assertIsNotNone(hit.hit)
                            self.assertEqual(hit.hit.object_id, expected_id)
                            # SDK-authored instance hierarchy matrices carry float rounding.
                            self.assertAlmostEqual(hit.hit.distance, 1.0, delta=1.0e-6)
                            references.append(hit.hit.geometry_value)
                        self.assertEqual(len(set(references)), n)
        finally:
            source.destroy()

    def test_refit_validates_authored_points_before_shape_baking(self) -> None:
        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )
        mesh = """def Mesh "Shape"
        {
            uniform bool doubleSided = 1
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
            float3[] extent = [(0,0,0), (1,1,0)]
        }"""
        for proxy in (False, True):
            if proxy:
                usda = f"""#usda 1.0
(defaultPrim = "World")
def Xform "Proto"
{{
    {mesh}
}}
def Xform "World"
{{
    def Xform "Instance" (
        instanceable = true
        references = </Proto>
    )
    {{
        double3 xformOp:scale = (0.25, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
    def Xform "Other" (
        instanceable = true
        references = </Proto>
    )
    {{
        double3 xformOp:scale = (0.25, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
"""
                path = "/World/Instance/Shape"
                points_filter = ovstage.Filter(
                    [
                        ovstage.Predicate(
                            "usd-schemas", ovstage.FilterOp.CONTAINS, ["GeomPrototypeAPI"]
                        ),
                        ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Mesh"]),
                    ]
                )
            else:
                usda = f"""#usda 1.0
(defaultPrim = "World")
def Xform "World"
{{
    double3 xformOp:scale = (0.25, 2, 2)
    uniform token[] xformOpOrder = ["xformOp:scale"]
    {mesh}
}}
"""
                path = "/World/Shape"
                points_filter = ovstage.Filter(
                    [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, [path])]
                )
            for execution in executions:
                with self.subTest(proxy=proxy, preparation=execution["preparation"]):
                    source = ovstage.Stage("authored-refit-domain")
                    try:
                        populate(source, usda, 1, 0.0)
                        source.advance_write_floor(ordinal=1).wait()
                        source.compute_hierarchy(
                            1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU
                        )
                        source.advance_write_floor(ordinal=2).wait()
                        selected = ovstage.Filter(
                            [
                                ovstage.Predicate(
                                    "usd-prim-type",
                                    ovstage.FilterOp.IN,
                                    ["MeshInstance" if proxy else "Mesh"],
                                )
                            ]
                        )
                        with (
                            Library.open(_installed_library(), require_ovstage=True) as library,
                            library.instance(config) as instance,
                        ):
                            adapter = instance.ovstage(ovstage, source)
                            with adapter.create_population(selected) as population:
                                with _enqueue_set_or_skip_unavailable_cuda(
                                    population, 2, execution
                                ) as operation:
                                    operation.wait()
                                    self.assertEqual(
                                        operation.fetch_population_update().object_count,
                                        2 if proxy else 1,
                                    )
                                # The baked X coordinate is 2^59, within the input domain,
                                # but its authored 2^61 value is outside the 2^60 limit.
                                with source.query(filter=points_filter) as query:
                                    query.wait()
                                    source.write_attribute(
                                        query,
                                        "points",
                                        3,
                                        [
                                            ovstage.make_dltensor(
                                                numpy.array(
                                                    [[0, 0, 0], [2.0**61, 0, 0], [0, 1, 0]],
                                                    dtype=numpy.float32,
                                                ),
                                                dtype=ovstage.DLDataType(2, 32, 3),
                                                shape=[3],
                                            )
                                        ],
                                        is_array=True,
                                    ).wait()
                                source.compute_hierarchy(
                                    3, 3, ovstage.HierarchyComputationModel.DEFAULT_CPU
                                )
                                # Supply enclosing, in-domain bounds independently of the
                                # SDK's cached authored extent; only point admission differs.
                                with source.query(filter=selected) as query:
                                    query.wait()
                                    source.write_attribute(
                                        query,
                                        "_worldExtent",
                                        3,
                                        [
                                            numpy.array(
                                                [[-(2.0**60), -4, -4, 2.0**60, 4, 4]]
                                                * (2 if proxy else 1),
                                                dtype=numpy.float64,
                                            )
                                        ],
                                        is_array=False,
                                    ).wait()
                                source.advance_write_floor(ordinal=3).wait()
                                dictionary = ovstage.PathDictionary(source)
                                ids = sorted(
                                    _path_int(dictionary.intern_path(p))
                                    for p in ((path, "/World/Other/Shape") if proxy else (path,))
                                )
                                with adapter.create_population(selected) as fresh:
                                    for operation in (
                                        fresh.enqueue_set(3, **execution),
                                        population.enqueue_update(2, 3, **execution),
                                    ):
                                        with operation:
                                            operation.wait()
                                            result = operation.fetch_population_update()
                                            self.assertEqual(
                                                (
                                                    result.object_count,
                                                    result.excluded_objects.tolist(),
                                                    result.exclusion_reasons.tolist(),
                                                ),
                                                (
                                                    0,
                                                    ids,
                                                    [int(OvstageExclusionReason.MALFORMED_GEOMETRY)]
                                                    * len(ids),
                                                ),
                                            )
                    finally:
                        source.destroy()

    def test_a_triangle_without_area_is_dropped_and_its_mesh_published(self) -> None:
        """A triangle whose corners are collinear or repeated has no area (ADR-0073). The set
        drops it, publishes the rest of its Mesh with authored face ordinals and names it in
        the calling thread's diagnostic. Each update judges the points it reads: a triangle
        that loses its area leaves, one that regains it returns, and a Mesh left with no
        triangle is MALFORMED_GEOMETRY. The same holds for a device-prepared publication."""
        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )

        def mesh(name, x, counts, indices):
            return f"""    def Mesh "{name}"
    {{
        uniform token orientation = "rightHanded"
        int[] faceVertexCounts = {list(counts)}
        int[] faceVertexIndices = {list(indices)}
        point3f[] points = [({x},0,0), ({x + 1},0,0), ({x + 1},1,0), ({x},1,0)]
        float3[] extent = [({x},0,0), ({x + 1},1,0)]
    }}
"""

        # Repeated's face 0 is a quad whose fan holds (0, 2, 2), and its face 1 covers the
        # upper-left half of its square. Partial and Collapsed are whole squares until an
        # update moves Partial's corner 3 onto its diagonal and Collapsed onto one line.
        usda = (
            '#usda 1.0\ndef Xform "World"\n{\n'
            + mesh("Repeated", 0, (4, 3), (0, 1, 2, 2, 0, 2, 3))
            + mesh("Partial", 3, (4,), (0, 1, 2, 3))
            + mesh("Collapsed", 6, (4,), (0, 1, 2, 3))
            + "}\n"
        )
        selected = ovstage.Filter(
            [ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Mesh"])]
        )

        def down(target, x, y):
            hit = target.raycast_closest(
                RaycastQuery(origin=(x, y, 1.0), unit_direction=(0.0, 0.0, -1.0), max_distance=2.0)
            ).hit
            return None if hit is None else hit.source_face_index

        def move(source, name, points, ordinal):
            path = ovstage.Filter(
                [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, [f"/World/{name}"])]
            )
            with source.query(filter=path) as query:
                query.wait()
                source.write_attribute(
                    query,
                    "points",
                    ordinal,
                    [
                        ovstage.make_dltensor(
                            numpy.array(points, dtype=numpy.float32),
                            dtype=ovstage.DLDataType(2, 32, 3),
                            shape=[len(points)],
                        )
                    ],
                    is_array=True,
                ).wait()

        def publish(source, ordinal):
            source.compute_hierarchy(
                ordinal, ordinal, ovstage.HierarchyComputationModel.DEFAULT_CPU
            )
            source.advance_write_floor(ordinal=ordinal).wait()

        for execution in executions:
            with self.subTest(preparation=execution["preparation"]):
                source = ovstage.Stage("triangle-without-area")
                try:
                    populate(source, usda, 1, 0.0)
                    source.advance_write_floor(ordinal=1).wait()
                    source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
                    source.advance_write_floor(ordinal=2).wait()
                    dictionary = ovstage.PathDictionary(source)
                    repeated, partial, collapsed = (
                        _path_int(dictionary.intern_path(f"/World/{name}"))
                        for name in ("Repeated", "Partial", "Collapsed")
                    )
                    with (
                        Library.open(_installed_library(), require_ovstage=True) as library,
                        library.instance(config) as instance,
                    ):

                        def last_error():
                            detail = library.exports["ovspatialquery_get_last_error"]()
                            return (
                                ctypes.string_at(detail.ptr, detail.length).decode()
                                if detail.length
                                else ""
                            )

                        def outcome(operation):
                            operation.wait()
                            value = operation.fetch_population_update()
                            return value.object_count, value.excluded_objects.tolist()

                        adapter = instance.ovstage(ovstage, source)
                        with adapter.create_population(selected) as population:
                            target = population.target
                            with _enqueue_set_or_skip_unavailable_cuda(
                                population, 2, execution
                            ) as operation:
                                report = last_error()
                                self.assertEqual(outcome(operation), (3, []))
                                self.assertEqual(operation.fetch_dropped_triangles(), {repeated: 1})
                            self.assertRegex(
                                report, r'"/World/Repeated"[^;]*\b1 of its 3 triangles[^;]*face 0\)'
                            )
                            self.assertNotIn("Partial", report)
                            self.assertEqual(
                                [down(target, 0.75, 0.25), down(target, 0.25, 0.75)], [0, 1]
                            )

                            move(
                                source,
                                "Partial",
                                [[3, 0, 0], [4, 0, 0], [4, 1, 0], [3.5, 0.5, 0]],
                                3,
                            )
                            move(
                                source, "Collapsed", [[6, 0, 0], [7, 0, 0], [8, 0, 0], [9, 0, 0]], 3
                            )
                            publish(source, 3)
                            with population.enqueue_update(2, 3, **execution) as operation:
                                report = last_error()
                                self.assertEqual(outcome(operation), (2, [collapsed]))
                                dropped = operation.fetch_dropped_triangles()
                                self.assertEqual(dropped.get(partial), 1)
                                self.assertNotIn(collapsed, dropped)
                            self.assertRegex(report, r'"/World/Partial"[^;]*\b1 of its 2 triangles')
                            self.assertEqual(
                                [down(target, 3.75, 0.25), down(target, 3.25, 0.75)], [0, None]
                            )

                            move(source, "Partial", [[3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]], 4)
                            move(
                                source, "Collapsed", [[6, 0, 0], [7, 0, 0], [7, 1, 0], [6, 1, 0]], 4
                            )
                            publish(source, 4)
                            with population.enqueue_update(3, 4, **execution) as operation:
                                self.assertEqual(outcome(operation), (3, []))
                                self.assertNotIn(partial, operation.fetch_dropped_triangles())
                            self.assertEqual(
                                [down(target, 3.25, 0.75), down(target, 6.25, 0.75)], [0, 0]
                            )
                finally:
                    source.destroy()

    def test_a_convex_hull_publishes_as_the_hull_physx_simulates(self) -> None:
        """A convexHull Mesh publishes as the hull PhysX cooks for it under its hull vertex
        limit, as Omni Physics simulates it (ADR-0078): physxConvexHullCollision:hullVertexLimit
        when the stage carries it, else 64, clamped to 8 to 64. Four copies of the larger
        carton, whose own hull has 255 faces and was excluded, each translated, author none,
        8, 3 and 200; the 128-gon prism, excluded before for 256 vertices, authors none. Each
        publishes, and a sphere of 1e-6 of the half extent on each of its points overlaps it
        exactly when it overlaps the explicit registration of the same points under the limit
        expected: a library that ignored the attribute or did not clamp it would differ on some
        points. Ovstage 0.3's population domains do not carry the attribute, so there every prim
        answers under 64."""
        from tests.test_explicit import _CARTON_BEYOND_THE_FACE_LIMIT
        import base64

        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )
        carton = numpy.frombuffer(
            base64.b64decode("".join(_CARTON_BEYOND_THE_FACE_LIMIT)), "<f4"
        ).reshape(-1, 3)
        angle = 2 * numpy.pi * numpy.arange(128) / 128
        ring = numpy.column_stack([numpy.cos(angle), numpy.sin(angle)])
        prism = numpy.vstack([numpy.column_stack([ring, numpy.full(128, z)]) for z in (-1.0, 1.0)])
        prism = prism.astype(numpy.float32)
        # name, points, x offset, API applied, authored limit, limit expected
        # Ovstage 0.3 has an ALL domain and no property selectors (tests.ovstage_population).
        carried = not hasattr(ovstage.PopulationDomain, "ALL")
        prims = (
            ("Default", carton, 0, True, None, 64),
            ("Eight", carton, 2, True, 8, 8),
            ("Three", carton, 4, False, 3, 8),
            ("TwoHundred", carton, 6, True, 200, 64),
            ("Prism", prism, 10, False, None, 64),
        )

        def mesh(name, points, x, api, limit):
            schemas = '"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"' + (
                ', "PhysxConvexHullCollisionAPI"' if api else ""
            )
            text = ", ".join(f"({float(a)!r}, {float(b)!r}, {float(c)!r})" for a, b, c in points)
            low, high = points.min(0), points.max(0)
            authored = (
                f"        int physxConvexHullCollision:hullVertexLimit = {limit}\n"
                if limit is not None
                else ""
            )
            return f"""    def Mesh "{name}" (
        prepend apiSchemas = [{schemas}]
    )
    {{
        point3f[] points = [{text}]
        float3[] extent = [({low[0]}, {low[1]}, {low[2]}), ({high[0]}, {high[1]}, {high[2]})]
        uniform token physics:approximation = "convexHull"
{authored}        double3 xformOp:translate = ({x}, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
"""

        usda = (
            '#usda 1.0\ndef Xform "World"\n{\n'
            + "".join(mesh(name, points, x, api, limit) for name, points, x, api, limit, _ in prims)
            + "}\n"
        )
        selected = ovstage.Filter(
            [ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Mesh"])]
        )

        def overlapped(target, points, x):
            half = 0.5 * float((points.max(0) - points.min(0)).max())
            return [
                target.overlap_any(
                    OverlapQuery(Sphere((float(a) + x, float(b), float(c)), 1e-6 * half))
                ).hit
                is not None
                for a, b, c in points
            ]

        for execution in executions:
            with self.subTest(preparation=execution["preparation"]):
                source = ovstage.Stage("convex-hull-under-its-vertex-limit")
                try:
                    populate(source, usda, 1, 0.0)
                    source.advance_write_floor(ordinal=1).wait()
                    source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
                    source.advance_write_floor(ordinal=2).wait()
                    with (
                        Library.open(_installed_library(), require_ovstage=True) as library,
                        library.instance(config) as instance,
                    ):
                        adapter = instance.ovstage(ovstage, source)
                        explicit = instance.explicit()
                        with adapter.create_population(selected) as population:
                            with _enqueue_set_or_skip_unavailable_cuda(
                                population, 2, execution
                            ) as operation:
                                operation.wait()
                                value = operation.fetch_population_update()
                                self.assertEqual(
                                    (value.object_count, value.excluded_objects.tolist()),
                                    (len(prims), []),
                                )
                            for index, (name, points, x, _, _, limit) in enumerate(prims, start=1):
                                limit = limit if carried else 64
                                explicit.register_convexes(
                                    definition_values=numpy.array([index], numpy.uint64),
                                    vertex_offsets=numpy.array([0], numpy.uint64),
                                    vertex_counts=numpy.array([len(points)], numpy.uint64),
                                    vertices=points,
                                    hull_vertex_limit=limit,
                                ).finish()
                                with explicit.create_population() as reference:
                                    reference.enqueue_set(
                                        1,
                                        object_values=numpy.array([1], numpy.uint64),
                                        geometry_types=numpy.array(
                                            [GeometryType.CONVEX], numpy.int32
                                        ),
                                        geometry_values=numpy.array([index], numpy.uint64),
                                        translations=numpy.array([(x, 0.0, 0.0)]),
                                        rotations_xyzw=numpy.array([(0.0, 0.0, 0.0, 1.0)]),
                                    ).finish()
                                    expected = overlapped(reference.target, points, x)
                                self.assertIn(False, expected, name)
                                if name == "Default":
                                    # The same hull placed where Eight is.
                                    with explicit.create_population() as shifted:
                                        shifted.enqueue_set(
                                            1,
                                            object_values=numpy.array([1], numpy.uint64),
                                            geometry_types=numpy.array(
                                                [GeometryType.CONVEX], numpy.int32
                                            ),
                                            geometry_values=numpy.array([index], numpy.uint64),
                                            translations=numpy.array([(2.0, 0.0, 0.0)]),
                                            rotations_xyzw=numpy.array([(0.0, 0.0, 0.0, 1.0)]),
                                        ).finish()
                                        default_hits = overlapped(shifted.target, points, 2)
                                self.assertEqual(
                                    overlapped(population.target, points, x), expected, name
                                )
                                explicit.destroy_definition(
                                    GeometryDefinitionRef(GeometryType.CONVEX, index)
                                )
                            # An edit of the limit alone republishes the prim's hull: Eight
                            # authors 64 at ordinal 3 and answers as Default's carton does.
                            if carried:
                                eight = ovstage.Filter(
                                    [
                                        ovstage.Predicate(
                                            "usd-path", ovstage.FilterOp.IN, ["/World/Eight"]
                                        )
                                    ]
                                )
                                with source.query(filter=eight) as query:
                                    query.wait()
                                    source.write_attribute(
                                        query,
                                        "physxConvexHullCollision:hullVertexLimit",
                                        3,
                                        numpy.array([64], dtype=numpy.int32),
                                        is_array=False,
                                    ).wait()
                                source.advance_write_floor(ordinal=3).wait()
                                with population.enqueue_update(2, 3, **execution) as operation:
                                    operation.wait()
                                    self.assertEqual(
                                        operation.fetch_population_update().object_count,
                                        len(prims),
                                    )
                                self.assertEqual(
                                    overlapped(population.target, carton, 2),
                                    default_hits,
                                )
                finally:
                    source.destroy()

    def test_proxy_geometry_sharing_uses_normalized_shape_and_determinant_sign(self) -> None:
        families = (
            ("Cube", "Cube", "", "double size = 1", True),
            ("Sphere", "Sphere", "", "double radius = 0.5", False),
            (
                "Capsule",
                "Capsule",
                "",
                'double radius = 0.5\n        double height = 1\n        uniform token axis = "Y"',
                False,
            ),
            (
                "Mesh",
                "Mesh",
                "",
                """uniform bool doubleSided = 0
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(-.5, -.5, 0), (.5, -.5, 0), (0, .5, 0)]
        float3[] extent = [(-.5, -.5, 0), (.5, .5, 0)]""",
                True,
            ),
            (
                "convexHull",
                "Mesh",
                '(prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])',
                """uniform token physics:approximation = "convexHull"
        uniform bool doubleSided = 1
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3, 3, 3, 3]
        int[] faceVertexIndices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (.5, 0, 0), (0, .5, 0), (0, 0, .5)]
        float3[] extent = [(0, 0, 0), (.5, .5, .5)]""",
                True,
            ),
        )
        scales = ((1, 1, 1), (2, 2, 2), (-1, -1, -1), (-2, -2, -2), (4, 2, 1), (8, 4, 2), (8, 2, 1))
        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )
        with (
            Library.open(_installed_library(), require_ovstage=True) as library,
            library.instance(config) as instance,
        ):
            for family, prim_type, metadata, attributes, nonuniform in families:
                instances = "".join(
                    f"""    def Xform "I{index}" (
        instanceable = true
        references = </Proto>
    )
    {{
        double3 xformOp:translate = ({10 * index}, 0, 0)
        double3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
"""
                    for index, (sx, sy, sz) in enumerate(scales)
                )
                usda = f"""#usda 1.0
(defaultPrim = "World")
def Xform "Proto"
{{
    def {prim_type} "Shape" {metadata}
    {{
        {attributes}
    }}
}}
def Xform "World"
{{
{instances}}}
"""
                for execution in executions:
                    with self.subTest(family=family, preparation=execution["preparation"]):
                        source = ovstage.Stage("normalized-proxy-sharing")
                        try:
                            populate(source, usda, 1, 0.0)
                            source.advance_write_floor(ordinal=1).wait()
                            source.compute_hierarchy(
                                1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU
                            )
                            source.advance_write_floor(ordinal=2).wait()
                            paths = ovstage.PathDictionary(source)
                            ids = [
                                _path_int(paths.intern_path(f"/World/I{index}/Shape"))
                                for index in range(7)
                            ]
                            proxy_filter = ovstage.Filter(
                                [
                                    ovstage.Predicate(
                                        "usd-prim-type",
                                        ovstage.FilterOp.IN,
                                        [f"{prim_type}Instance"],
                                    )
                                ]
                            )
                            adapter = instance.ovstage(ovstage, source)
                            with adapter.create_population(proxy_filter) as population:
                                with _enqueue_set_or_skip_unavailable_cuda(
                                    population, 2, execution
                                ) as operation:
                                    operation.wait()
                                    publication = operation.fetch_population_update()
                                    self.assertEqual(
                                        publication.object_count, 7 if nonuniform else 4
                                    )
                                    excluded = [] if nonuniform else sorted(ids[4:])
                                    self.assertEqual(
                                        publication.excluded_object_count, len(excluded)
                                    )
                                    numpy.testing.assert_array_equal(
                                        publication.excluded_objects, excluded
                                    )
                                    numpy.testing.assert_array_equal(
                                        publication.exclusion_reasons,
                                        [OvstageExclusionReason.UNSUPPORTED_SHAPE] * len(excluded),
                                    )
                                references = {}
                                for index, object_id in enumerate(ids):
                                    center = 10.0 * index
                                    around = Aabb(
                                        (center - 0.1, -0.1, -0.1), (center + 0.1, 0.1, 0.1)
                                    )
                                    result = population.target.overlap_multiple(
                                        OverlapQuery(
                                            around, precision=QueryPrecision.BOUNDS_CANDIDATE
                                        ),
                                        2,
                                    )
                                    self.assertFalse(result.truncated)
                                    admitted = nonuniform or index < 4
                                    self.assertEqual(len(result.hits), 1 if admitted else 0)
                                    if admitted:
                                        self.assertEqual(result.hits[0].object_id, object_id)
                                        references[index] = result.hits[0].geometry_value
                                # Positive uniform scale preserves one shape per sign.
                                self.assertEqual(references[0], references[1])
                                self.assertEqual(references[2], references[3])
                                self.assertNotEqual(references[0], references[2])
                                if nonuniform:
                                    # det(4,2,1)=8 and det(8,4,2)=64 give s=2 and4;
                                    # their normalized metrics are both diag(4,1,.25).
                                    self.assertEqual(references[4], references[5])
                                    self.assertNotEqual(references[4], references[6])
                                    for index in (4, 6):
                                        self.assertNotEqual(references[index], references[0])
                                        self.assertNotEqual(references[index], references[2])
                                if family == "Mesh":
                                    # A points-only edit must preserve each proxy's authored
                                    # front even when the prototype has both determinant signs.
                                    prototypes = ovstage.Filter(
                                        [
                                            ovstage.Predicate(
                                                "usd-schemas",
                                                ovstage.FilterOp.CONTAINS,
                                                ["GeomPrototypeAPI"],
                                            ),
                                            ovstage.Predicate(
                                                "usd-prim-type", ovstage.FilterOp.IN, ["Mesh"]
                                            ),
                                        ]
                                    )
                                    with source.query(filter=prototypes) as query:
                                        query.wait()
                                        source.write_attribute(
                                            query,
                                            "points",
                                            3,
                                            [
                                                ovstage.make_dltensor(
                                                    numpy.array(
                                                        [
                                                            [-0.25, -0.25, 0],
                                                            [0.25, -0.25, 0],
                                                            [0, 0.25, 0],
                                                        ],
                                                        dtype=numpy.float32,
                                                    ),
                                                    dtype=ovstage.DLDataType(2, 32, 3),
                                                    shape=[3],
                                                )
                                            ],
                                            is_array=True,
                                        ).wait()
                                    source.compute_hierarchy(
                                        3, 3, ovstage.HierarchyComputationModel.DEFAULT_CPU
                                    )
                                    source.advance_write_floor(ordinal=3).wait()
                                    with population.enqueue_update(2, 3, **execution) as operation:
                                        operation.wait()
                                    for index, (_, _, sz) in enumerate(scales):
                                        direction = -1.0 if sz > 0 else 1.0
                                        ray = RaycastQuery(
                                            origin=(10.0 * index, 0, -2 * direction),
                                            unit_direction=(0, 0, direction),
                                            max_distance=4,
                                        )
                                        hit = population.target.raycast_any(ray).hit
                                        self.assertIsNotNone(
                                            hit, f"proxy {index}, scale {scales[index]}"
                                        )
                                        self.assertEqual(hit.object_id, ids[index])
                                        self.assertAlmostEqual(hit.distance, 2.0)
                                        self.assertAlmostEqual(hit.normal[2], -direction)
                        finally:
                            source.destroy()

    def test_instance_proxies_publish_and_follow_their_prototype(self) -> None:
        source = _instanced_stage("instancing-test", "ALL")
        paths = ovstage.PathDictionary(source)
        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        device = dict(
            preparation=Preparation.ON_DEVICE_HOST_READABLE, device=Device.cuda(0), cuda_stream=1
        )
        try:
            with (
                Library.open(_installed_library(), require_ovstage=True) as library,
                library.instance(config) as instance,
            ):
                adapter = instance.ovstage(ovstage, source)
                # The empty filter also matches the prototype copies; the device population's
                # filter matches only proxies, so only their prototype prims carry definitions.
                proxies = ovstage.Predicate(
                    "usd-prim-type",
                    ovstage.FilterOp.IN,
                    ["SphereInstance", "CubeInstance", "MeshInstance"],
                )
                populations = [adapter.create_population(ovstage.Filter([]))]
                if cuda:
                    populations.append(adapter.create_population(ovstage.Filter([proxies])))
                for index, population in enumerate(populations):
                    try:
                        operation = population.enqueue_set(2, **(device if index else {}))
                    except ApiError as refused:
                        if (
                            index
                            and refused.status == ApiStatus.NOT_SUPPORTED
                            and refused.detail.startswith("CUDA is unavailable")
                        ):
                            population.close()
                            populations.pop()
                            continue
                        raise
                    with operation:
                        operation.wait()
                        # Eight proxies on the ray, the proxy /Outer/Inner/Body at
                        # y = 100 and, for the empty filter, the four authored prims there.
                        proxy_count = 9
                        self.assertEqual(
                            operation.fetch_population_update().object_count,
                            proxy_count if index else proxy_count + 4,
                        )
                    hits = self._hits(population.target, paths)
                    expected = {
                        "/World/I1/Body": 54.5,
                        "/World/I2/Body": 57.0,
                        "/World/P1/Quad": 62.0,
                        "/World/P2/Quad": 64.0,
                        "/World/N1/Ball": 69.5,
                        "/World/N1/Inner/Body": 72.5,
                    }
                    expected.update({"/World/H1/Shape": 79.5, "/World/H2/Shape": 82.5})
                    self.assertEqual(hits["/World/H1/Shape"][1], hits["/World/H2/Shape"][1])
                    self.assertEqual(
                        {path: distance for path, (distance, _) in hits.items()}, expected
                    )
                    # The proxies of one prototype prim share its definition.
                    self.assertEqual(hits["/World/I1/Body"][1], hits["/World/N1/Inner/Body"][1])
                    self.assertEqual(hits["/World/P1/Quad"][1], hits["/World/P2/Quad"][1])

                # Shrinking the prototype's cube in Ovstage changes every proxy of it.
                # The prototype copy the proxies name; ovstage marks it with GeomPrototypeAPI.
                prototype_cube = ovstage.Filter(
                    [
                        ovstage.Predicate(
                            "usd-schemas", ovstage.FilterOp.CONTAINS, ["GeomPrototypeAPI"]
                        ),
                        ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Cube"]),
                    ]
                )
                with source.query(filter=prototype_cube) as query:
                    query.wait()
                    source.write_attribute(
                        query, "size", 3, [numpy.array([0.5])], is_array=False
                    ).wait()
                    # The hidden copy's bounds never take part, malformed or not.
                    source.write_attribute(
                        query,
                        "_worldExtent",
                        3,
                        [numpy.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])],
                        is_array=False,
                    ).wait()
                source.advance_write_floor(ordinal=3).wait()
                for index, population in enumerate(populations):
                    with population.enqueue_update(2, 3, **(device if index else {})) as operation:
                        operation.wait()
                    hits = self._hits(population.target, paths)
                    self.assertEqual(hits["/World/I1/Body"][0], 54.75)
                    self.assertEqual(hits["/World/I2/Body"][0], 57.5)
                    self.assertEqual(hits["/World/N1/Inner/Body"][0], 72.75)
                    self.assertEqual(hits["/World/P1/Quad"][0], 62.0)

                # Explicit filters keep Ovstage's vocabulary; neither selects a prototype copy.
                for predicate, expected in (
                    (ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Cube"]), 1),
                    (ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ["/World/I1/Body"]), 1),
                    (ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["CubeInstance"]), 4),
                ):
                    with self.subTest(predicate=predicate):
                        population = adapter.create_population(ovstage.Filter([predicate]))
                        with population.enqueue_set(3) as operation:
                            operation.wait()
                            update = operation.fetch_population_update()
                            self.assertEqual(update.object_count, expected)
                            self.assertEqual(update.excluded_object_count, 0)
                        population.close()
                # A rigid move changes the binding even while prototype content,
                # normalized shape and geometry reference remain unchanged.
                moved_panel = ovstage.Filter(
                    [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ["/World/P1/Quad"])]
                )
                matrix = numpy.eye(4, dtype=numpy.float64)
                matrix[3, 0] = 16.0
                with source.query(filter=moved_panel) as query:
                    query.wait()
                    source.write_attribute(
                        query, "omni:fabric:worldMatrix", 4, matrix.reshape(1, 16), is_array=False
                    ).wait()
                    source.write_attribute(
                        query,
                        "_worldExtent",
                        4,
                        numpy.array([[15, -2, -2, 17, 2, 2]], dtype=numpy.float64),
                        is_array=False,
                    ).wait()
                source.advance_write_floor(ordinal=4).wait()
                for index, population in enumerate(populations):
                    with population.enqueue_update(3, 4, **(device if index else {})) as operation:
                        operation.wait()
                        self.assertEqual(
                            operation.fetch_population_update().object_count,
                            proxy_count if index else proxy_count + 4,
                        )
                    hits = self._hits(population.target, paths)
                    self.assertEqual(hits["/World/P1/Quad"][0], 66.0)
                    self.assertEqual(hits["/World/P2/Quad"][0], 64.0)
                    self.assertEqual(hits["/World/P1/Quad"][1], hits["/World/P2/Quad"][1])
                for population in populations:
                    population.close()
        finally:
            source.destroy()

    def test_a_stage_without_proxy_rows_refuses_its_prototype_copies(self) -> None:
        # PHYSICS alone keeps the prototype copies and makes no proxy rows.
        source = _instanced_stage(
            "instancing-physics-test",
            "PHYSICS",
            ' (\n        prepend apiSchemas = ["PhysicsCollisionAPI"]\n    )',
        )
        try:
            with (
                Library.open(_installed_library(), require_ovstage=True) as library,
                library.instance() as instance,
            ):
                population = instance.ovstage(ovstage, source).create_population(ovstage.Filter([]))
                with self.assertRaises(ApiError) as refused:
                    population.enqueue_set(2)
                self.assertEqual(refused.exception.status, ApiStatus.NOT_SUPPORTED)
                self.assertIn("/__Prototype_", refused.exception.detail)
                self.assertIn("RENDERING", refused.exception.detail)
                population.close()
        finally:
            source.destroy()

    def test_point_instancer_prototypes_are_never_published_as_themselves(self) -> None:
        # RENDERING copies each prototype prim under the instancer (/World/PI/proto0,
        # /World/PI/proto1/A, /World/PI/proto1/B) at the origin with a _worldExtent; ALL
        # also keeps the authored prototypes at their own paths. Neither is an object, and
        # neither is an exclusion: only /World/Plain publishes.
        usda = """#usda 1.0
def Xform "World" {
  def PointInstancer "PI" {
    rel prototypes = [</World/PI/Protos/Box>, </World/PI/Protos/Pair>]
    int[] protoIndices = [0, 1, 0]
    point3f[] positions = [(0,0,0), (5,0,0), (10,0,0)]
    def Scope "Protos" {
      def Cube "Box" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double size = 1
      }
      def Xform "Pair" {
        def Sphere "A" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
          double radius = 0.5
        }
        def Sphere "B" {
          double radius = 0.25
        }
      }
    }
  }
  def Cube "Plain" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
    double size = 1
    double3 xformOp:translate = (-5, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
  }
}
"""
        cuda = _cuda_device_present()
        executions = [dict(preparation=Preparation.ON_HOST)]
        if cuda:
            executions.append(
                dict(
                    preparation=Preparation.ON_DEVICE_HOST_READABLE,
                    device=Device.cuda(0),
                    cuda_stream=1,
                )
            )
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        for domain in ("RENDERING", "ALL"):
            source = ovstage.Stage(f"point-instancer-{domain}")
            try:
                populate(source, usda, 1, 0.0, domain)
                source.advance_write_floor(ordinal=1).wait()
                source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
                paths = ovstage.PathDictionary(source)
                if domain == "ALL":
                    # The authored Box has no _worldExtent; giving it one shows it is dropped
                    # as a prototype, not left out for want of a spatial source.
                    authored = ovstage.Filter(
                        [
                            ovstage.Predicate(
                                "usd-path", ovstage.FilterOp.IN, ["/World/PI/Protos/Box"]
                            )
                        ]
                    )
                    with source.query(filter=authored) as query:
                        query.wait()
                        self.assertEqual(query.result().total_prim_count, 1)
                        source.write_attribute(
                            query,
                            "_worldExtent",
                            2,
                            numpy.array([[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]], dtype=numpy.float64),
                            is_array=False,
                        ).wait()
                source.advance_write_floor(ordinal=2).wait()
                plain = _path_int(paths.intern_path("/World/Plain"))
                # A device publication refuses a usd-path IN name without a prim (BH-035), and
                # only ALL keeps the authored prototype.
                named = ["/World/PI/proto0", "/World/PI/proto1/A", "/World/Plain"]
                named += ["/World/PI/Protos/Box"] if domain == "ALL" else []
                filters = {
                    "empty": [],
                    "Cube": [ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Cube"])],
                    "paths": [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, named)],
                }
                with (
                    Library.open(_installed_library(), require_ovstage=True) as library,
                    library.instance(config) as instance,
                ):
                    adapter = instance.ovstage(ovstage, source)
                    for (name, predicates), execution in (
                        (item, execution) for item in filters.items() for execution in executions
                    ):
                        with (
                            self.subTest(
                                domain=domain, filter=name, preparation=execution["preparation"]
                            ),
                            adapter.create_population(ovstage.Filter(predicates)) as population,
                        ):
                            with _enqueue_set_or_skip_unavailable_cuda(
                                population, 2, execution
                            ) as operation:
                                operation.wait()
                                update = operation.fetch_population_update()
                            self.assertEqual(
                                (update.object_count, update.excluded_objects.tolist()), (1, [])
                            )
                            result = population.target.overlap_multiple(
                                OverlapQuery(Aabb((-100.0,) * 3, (100.0,) * 3)), 16
                            )
                            self.assertEqual([hit.object_id for hit in result.hits], [plain])
                    if domain == "ALL" and cuda:
                        # Shearing Plain is a shape change the device leaves pending; the
                        # next update rebuilds it on the host and must still read poses only
                        # for Plain, since the authored prototypes carry no Fabric matrix.
                        device = executions[1]
                        with adapter.create_population(ovstage.Filter([])) as population:
                            with _enqueue_set_or_skip_unavailable_cuda(
                                population, 2, device
                            ) as operation:
                                operation.wait()
                            matrix = numpy.eye(4, dtype=numpy.float64)
                            matrix[1, 0], matrix[3, 0] = 1.0, -5.0
                            moved = ovstage.Filter(
                                [
                                    ovstage.Predicate(
                                        "usd-path", ovstage.FilterOp.IN, ["/World/Plain"]
                                    )
                                ]
                            )
                            with source.query(filter=moved) as query:
                                query.wait()
                                source.write_attribute(
                                    query, "omni:xform", 3, matrix.reshape(1, 16), is_array=False
                                ).wait()
                            source.compute_hierarchy(
                                3, 3, ovstage.HierarchyComputationModel.DEFAULT_CPU
                            )
                            source.advance_write_floor(ordinal=3).wait()
                            with population.enqueue_update(2, 3, **device) as operation:
                                operation.wait()
                                pending = operation.fetch_population_update()
                            self.assertEqual(
                                (
                                    pending.excluded_objects.tolist(),
                                    pending.exclusion_reasons.tolist(),
                                ),
                                ([plain], [OvstageExclusionReason.RESHAPE_PENDING]),
                            )
                            source.compute_hierarchy(
                                4, 4, ovstage.HierarchyComputationModel.DEFAULT_CPU
                            )
                            source.advance_write_floor(ordinal=4).wait()
                            with population.enqueue_update(3, 4, **device) as operation:
                                operation.wait()
                                recovered = operation.fetch_population_update()
                            self.assertEqual(
                                (recovered.object_count, recovered.excluded_objects.tolist()),
                                (1, []),
                            )
                            # Retargeting the instancer onto Plain changes no matched row,
                            # yet makes Plain a prototype the device must drop too.
                            instancer = ovstage.Filter(
                                [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ["/World/PI"])]
                            )
                            targets = numpy.array(
                                [
                                    _path_int(paths.intern_path(target))
                                    for target in (
                                        "/World/PI/Protos/Box",
                                        "/World/PI/Protos/Pair",
                                        "/World/Plain",
                                    )
                                ],
                                dtype=numpy.uint64,
                            )
                            with source.query(filter=instancer) as query:
                                query.wait()
                                source.write_attribute(
                                    query,
                                    "prototypes:usd-targets",
                                    5,
                                    [
                                        ovstage.make_dltensor(
                                            targets,
                                            dtype=ovstage.DLDataType(1, 64, 1),
                                            shape=[len(targets)],
                                        )
                                    ],
                                    is_array=True,
                                ).wait()
                            source.compute_hierarchy(
                                5, 5, ovstage.HierarchyComputationModel.DEFAULT_CPU
                            )
                            source.advance_write_floor(ordinal=5).wait()
                            with population.enqueue_update(4, 5, **device) as operation:
                                operation.wait()
                                retargeted = operation.fetch_population_update()
                            self.assertEqual(
                                (retargeted.object_count, retargeted.excluded_objects.tolist()),
                                (0, []),
                            )
            finally:
                source.destroy()


def _cuda_device_present() -> bool:
    try:
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    except OSError:
        return False
    count = ctypes.c_int()
    return (
        driver.cuInit(0) == 0
        and driver.cuDeviceGetCount(ctypes.byref(count)) == 0
        and count.value > 0
    )


@unittest.skipIf(ovstage is None, "Ovstage and OpenUSD are optional dependencies")
class OvstageDevicePreparationTests(unittest.TestCase):
    """ADR-0047 through the Python wrapper: a device-prepared set and a poses-only update on
    a runtime with the placement extension and a CUDA device; NOT_SUPPORTED otherwise."""

    def test_device_prepared_publications_follow_the_stage(self) -> None:
        usda = """#usda 1.0
def Cube "A"
{
    double size = 2
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
"""
        source = ovstage.Stage("device-preparation-test")
        populate(source, usda, 1, 0.0)
        source.advance_write_floor(ordinal=1).wait()
        source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        source.advance_write_floor(ordinal=2).wait()
        selected = _installed_library()
        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        with (
            Library.open(selected, require_ovstage=True) as library,
            library.instance(config) as instance,
        ):
            adapter = instance.ovstage(ovstage, source)
            population = adapter.create_population(
                ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ("/A",))])
            )
            device = dict(
                preparation=Preparation.ON_DEVICE_HOST_READABLE,
                device=Device.cuda(0),
                cuda_stream=1,
            )
            with self.assertRaisesRegex(ValueError, "CUDA device"):
                population.enqueue_set(2, preparation=Preparation.ON_DEVICE)
            # The capability query predicts the admission below without creating anything.
            predicted = adapter.supports(Capability.PREPARE_ON_DEVICE, Device.cuda(0))
            try:
                operation = population.enqueue_set(2, **device)
            except ApiError as refused:
                self.assertFalse(predicted)
                # Without a CUDA backend, driver or device the library says CUDA is unavailable.
                self.assertEqual(refused.status, ApiStatus.NOT_SUPPORTED)
                self.assertTrue(refused.detail.startswith("CUDA is unavailable"), refused.detail)
                population.close()
                return
            self.assertTrue(cuda)
            self.assertTrue(predicted)
            with operation:
                operation.wait()
                self.assertEqual(operation.fetch_population_update().object_count, 1)
            around = Aabb((4.5, -0.5, -0.5), (5.5, 0.5, 0.5))
            self.assertIsNone(population.target.overlap_any(OverlapQuery(around)).hit)
            # Move /A to x = 5 through its local transform; the device reads the new pose.
            import numpy

            moved = numpy.eye(4, dtype=numpy.float64)
            moved[3, 0] = 5.0
            paths = ovstage.PathDictionary(source)
            with paths.create_path_list_from_strings(["/A"]) as listed:
                with source.query_from_path_list(listed) as query:
                    source.write_attribute(
                        query,
                        "omni:xform",
                        3,
                        moved.reshape(1, 16),
                        is_array=False,
                        semantic=ovstage.AttributeSemantic.MATRIX,
                    ).wait()
            source.compute_hierarchy(3, 3, ovstage.HierarchyComputationModel.DEFAULT_CPU)
            source.advance_write_floor(ordinal=3).wait()
            with population.enqueue_update(2, 3, **device) as operation:
                operation.wait()
                self.assertEqual(operation.fetch_population_update().object_count, 1)
            hit = population.target.overlap_any(OverlapQuery(around)).hit
            self.assertIsNotNone(hit)
            self.assertEqual(paths.path_to_string(_prim_path(hit.object_id)), "/A")
            population.close()

    def test_an_excluded_prim_returns_with_its_participation_bits(self) -> None:
        # A pose update that moves /A out of the input domain excludes it; the next one
        # brings it back with its bits, including bit 100 in the second word.
        import numpy

        usda = """#usda 1.0
def Cube "A"
{
    double size = 2
}
"""
        source = ovstage.Stage("excluded-prim-bits-test")
        populate(source, usda, 1, 0.0)
        source.advance_write_floor(ordinal=1).wait()
        source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        source.advance_write_floor(ordinal=2).wait()
        cuda = _cuda_device_present()
        config = InstanceConfig(
            device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),) if cuda else ()
        )
        paths = ovstage.PathDictionary(source)
        ordinal = 2

        def move_to(x: float) -> None:
            matrix = numpy.eye(4, dtype=numpy.float64)
            matrix[3, 0] = x
            with paths.create_path_list_from_strings(["/A"]) as listed:
                with source.query_from_path_list(listed) as query:
                    source.write_attribute(
                        query,
                        "omni:xform",
                        ordinal,
                        matrix.reshape(1, 16),
                        is_array=False,
                        semantic=ovstage.AttributeSemantic.MATRIX,
                    ).wait()
            source.compute_hierarchy(
                ordinal, ordinal, ovstage.HierarchyComputationModel.DEFAULT_CPU
            )
            source.advance_write_floor(ordinal=ordinal).wait()

        around = Aabb((-0.5,) * 3, (0.5,) * 3)
        bit = 1 << 100
        with (
            Library.open(_installed_library(), require_ovstage=True) as library,
            library.instance(config) as instance,
        ):
            adapter = instance.ovstage(ovstage, source)
            executions = [dict(preparation=Preparation.ON_HOST)]
            # A CPU-only build on a GPU host has a device but no CUDA backend.
            if cuda and adapter.supports(Capability.PREPARE_ON_DEVICE, Device.cuda(0)):
                executions.append(
                    dict(
                        preparation=Preparation.ON_DEVICE_HOST_READABLE,
                        device=Device.cuda(0),
                        cuda_stream=1,
                    )
                )
            for execution in executions:
                with self.subTest(preparation=execution["preparation"].name):
                    population = adapter.create_population(
                        ovstage.Filter(
                            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, ("/A",))]
                        )
                    )
                    try:
                        population.set_participation_map(0x1, {"/A": bit})
                        with population.enqueue_set(ordinal, **execution) as operation:
                            operation.wait()
                            self.assertEqual(operation.fetch_population_update().object_count, 1)
                        for x, excluded in ((2.0**61, True), (0.0, False)):
                            ordinal += 1
                            move_to(x)
                            with population.enqueue_update(
                                ordinal - 1, ordinal, **execution
                            ) as operation:
                                operation.wait()
                                update = operation.fetch_population_update()
                            self.assertEqual(
                                update.excluded_objects.tolist(),
                                [_path_int(paths.intern_path("/A"))] if excluded else [],
                            )
                            hit = population.target.overlap_any(
                                OverlapQuery(around, query_bits=bit)
                            ).hit
                            self.assertEqual(hit is None, excluded, x)
                    finally:
                        population.close()


# ADR-0052: a moving cube, an animated arm with a static child, a prim moved by an `omni:xform`
# write, a prim that leaves its neighbour, and the still cubes they enter or leave.
_TIME_UPDATE_USDA = """#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 1
    timeCodesPerSecond = 1
)
def Xform "World"
{
    def Cube "Moving"
    {
        double size = 2
        double3 xformOp:translate.timeSamples = { 0: (1, 0, 0), 1: (4, 0, 0) }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Still"
    {
        double size = 2
        double3 xformOp:translate = (5.5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Xform "Arm"
    {
        double3 xformOp:translate.timeSamples = { 0: (0, 10, 0), 1: (0, 13, 0) }
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Child"
        {
            double size = 2
        }
    }
    def Cube "ArmStop"
    {
        double size = 2
        double3 xformOp:translate = (0, 14.5, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Pushed"
    {
        double size = 2
        double3 xformOp:translate = (0, -10, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "PushStop"
    {
        double size = 2
        double3 xformOp:translate = (0, -14.5, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Leaver"
    {
        double size = 2
        double3 xformOp:translate.timeSamples = { 0: (20, 0, 0), 1: (30, 0, 0) }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Anchor"
    {
        double size = 2
        double3 xformOp:translate = (21, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""
# Where each moving prim's centre is once the step's writers ran.
_TIME_MOVES = {
    "/World/Moving": (4.0, 0.0, 0.0),
    "/World/Arm/Child": (0.0, 13.0, 0.0),
    "/World/Leaver": (30.0, 0.0, 0.0),
}
_XFORM_MOVES = {"/World/Pushed": (0.0, -13.0, 0.0)}


def _copy_from_device(address: int, count: int) -> list[int]:
    driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    host = (ctypes.c_uint64 * count)()
    if count:
        status = driver.cuMemcpyDtoH_v2(host, ctypes.c_uint64(address), ctypes.c_size_t(8 * count))
        if status != 0:
            raise RuntimeError(f"cuMemcpyDtoH failed with {status}")
    return list(host)


@unittest.skipIf(ovstage is None, "Ovstage and OpenUSD are optional dependencies")
class OvstageDeviceTimeUpdateTests(unittest.TestCase):
    """ADR-0052 and BH-014: a device-prepared update equals the ON_HOST update of the same
    ordinal whichever writer moved the poses and whichever hierarchy model derived them, in its
    query results and in the pairs a CUDA reporter patches from its changed slots."""

    def _step(self, writer, model):
        """Publishes a host and a device population of every cube at 2, applies `writer` at 3
        and the hierarchy into 4, updates both to 4, and returns what each sees."""
        stage = ovstage.Stage("device-time-update-test")
        try:
            populate(stage, _TIME_UPDATE_USDA, 1, 0.0)
            stage.advance_write_floor(ordinal=1).wait()
            stage.compute_hierarchy(1, 2, model)
            stage.advance_write_floor(ordinal=2).wait()
            config = InstanceConfig(device_budgets=(DeviceBudget(Device.cuda(0), 64 << 20),))
            device = dict(
                preparation=Preparation.ON_DEVICE_HOST_READABLE,
                device=Device.cuda(0),
                cuda_stream=1,
            )
            with (
                Library.open(_installed_library(), require_ovstage=True) as library,
                library.instance(config) as instance,
            ):
                adapter = instance.ovstage(ovstage, stage)
                cubes = ovstage.Filter(
                    [ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, ["Cube"])]
                )
                host = adapter.create_population(cubes)
                on_device = adapter.create_population(cubes)
                host.enqueue_set(2).finish()
                try:
                    on_device.enqueue_set(2, **device).finish()
                except ApiError as refused:
                    host.close()
                    on_device.close()
                    if refused.status == ApiStatus.NOT_SUPPORTED:
                        self.skipTest(refused.detail)
                    raise
                reporters = {
                    "host": instance.create_pair_reporter(
                        PairReporterConfig(host.target.reference, host.target.reference, 64, 64)
                    ),
                    "device": instance.create_pair_reporter(
                        PairReporterConfig(
                            on_device.target.reference,
                            on_device.target.reference,
                            64,
                            64,
                            result_device=Device.cuda(0),
                        )
                    ),
                }
                baseline = {name: self._advance(r) for name, r in reporters.items()}
                if writer in ("time", "mixed"):
                    ovstage.population.update_from_usd_time(stage, 3, 1.0)
                if writer in ("xform", "mixed"):
                    pushed = numpy.eye(4, dtype=numpy.float64)
                    pushed[3, :3] = _XFORM_MOVES["/World/Pushed"]
                    paths = ovstage.PathDictionary(stage)
                    with paths.create_path_list_from_strings(["/World/Pushed"]) as listed:
                        with stage.query_from_path_list(listed) as query:
                            stage.write_attribute(
                                query,
                                "omni:xform",
                                3,
                                pushed.reshape(1, 16),
                                is_array=False,
                                semantic=ovstage.AttributeSemantic.MATRIX,
                            ).wait()
                stage.advance_write_floor(ordinal=3).wait()
                stage.compute_hierarchy(3, 4, model)
                stage.advance_write_floor(ordinal=4).wait()
                host.enqueue_update(2, 4).finish()
                on_device.enqueue_update(2, 4, **device).finish()
                seen = {}
                paths = ovstage.PathDictionary(stage)
                for name, population in (("host", host), ("device", on_device)):
                    hits = {}
                    for path, centre in {**_TIME_MOVES, **_XFORM_MOVES}.items():
                        probe = Aabb(tuple(c - 0.1 for c in centre), tuple(c + 0.1 for c in centre))
                        hit = population.target.overlap_any(OverlapQuery(probe)).hit
                        hits[path] = (
                            None if hit is None else paths.path_to_string(_prim_path(hit.object_id))
                        )
                    entered, exited = self._advance(reporters[name])
                    named = lambda pairs: {  # noqa: E731
                        tuple(sorted(paths.path_to_string(_prim_path(i)) for i in pair))
                        for pair in pairs
                    }
                    seen[name] = (hits, named(entered), named(exited), named(baseline[name][0]))
                for reporter in reporters.values():
                    reporter.close()
                host.close()
                on_device.close()
            return seen
        finally:
            stage.destroy()

    @staticmethod
    def _advance(reporter):
        operation, reservation = reporter.advance()
        operation.wait()
        if reservation.counts is None:
            result = operation.fetch_pair_reporter().to_numpy()
            pairs = (result.entered_pairs.tolist(), result.exited_pairs.tolist())
            operation.close()
            return pairs
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        # The copies and the consumers' event need the device's primary context current here.
        context = ctypes.c_void_p()
        if (
            driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), 0) != 0
            or driver.cuCtxSetCurrent(context) != 0
        ):
            raise RuntimeError("could not make the CUDA primary context current")
        if driver.cuEventSynchronize(ctypes.c_void_p(reservation.result_ready)) != 0:
            raise RuntimeError("cuEventSynchronize failed")
        counts = _copy_from_device(reservation.counts.data + reservation.counts.byte_offset, 4)
        columns = []
        for view, count in (
            (reservation.entered_pairs, counts[1]),
            (reservation.exited_pairs, counts[2]),
        ):
            flat = _copy_from_device(view.data + view.byte_offset, 2 * count)
            columns.append([flat[2 * row : 2 * row + 2] for row in range(count)])
        # The copies above were synchronous, so an event recorded now covers them.
        done = ctypes.c_void_p()
        if (
            driver.cuEventCreate(ctypes.byref(done), 0) != 0
            or driver.cuEventRecord(done, None) != 0
            or driver.cuDevicePrimaryCtxRelease(0) != 0
        ):
            raise RuntimeError("could not record the consumers' CUDA event")
        operation.close_after_cuda(done.value)
        return tuple(columns)

    def test_device_update_matches_the_host_update_for_every_pose_writer(self) -> None:
        if not _cuda_device_present():
            self.skipTest("no CUDA device")
        models = ovstage.HierarchyComputationModel
        for writer in ("time", "xform", "mixed"):
            for model in (models.DEFAULT_CPU, models.GPU_INCREMENTAL):
                with self.subTest(writer=writer, model=model.name):
                    seen = self._step(writer, model)
                    moved = {
                        **(_TIME_MOVES if writer != "xform" else {}),
                        **(_XFORM_MOVES if writer != "time" else {}),
                    }
                    hits, entered, exited, initial = seen["host"]
                    # The host update follows every writer: each moved prim is at its new centre.
                    for path in moved:
                        self.assertEqual(hits[path], path)
                    self.assertEqual(
                        entered,
                        {
                            pair
                            for pair, mover in (
                                (("/World/Moving", "/World/Still"), "/World/Moving"),
                                (("/World/Arm/Child", "/World/ArmStop"), "/World/Arm/Child"),
                                (("/World/PushStop", "/World/Pushed"), "/World/Pushed"),
                            )
                            if mover in moved
                        },
                    )
                    self.assertEqual(
                        exited,
                        {("/World/Anchor", "/World/Leaver")} if "/World/Leaver" in moved else set(),
                    )
                    self.assertEqual(seen["device"], seen["host"])


if __name__ == "__main__":
    unittest.main()
