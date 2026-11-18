# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure installed Python result-observation paths on a small exact fixture."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np

import ovspatialquery
from ovspatialquery import (
    GeometryDefinitionRef,
    InstanceConfig,
    Library,
    PopulationConfig,
    RaycastPlanConfig,
    RaycastQuery,
)
from ovspatialquery import raw

ORDINAL = 1
BOX_OBJECT, SPHERE_OBJECT = 1, 2
BOX_DEFINITION = SPHERE_DEFINITION = 1
BOX_HALF = (0.5, 0.375, 0.5)
SPHERE_RADIUS = 0.5
SPHERE_HEIGHT = math.sqrt(SPHERE_RADIUS**2 - 0.125**2 - 0.0625**2)
BOX_ROTATION = (0.0, 0.0, math.sin(0.25), math.cos(0.25))
FIELDS = (
    raw.BatchResultField.POSITION
    | raw.BatchResultField.NORMAL
    | raw.BatchResultField.DISTANCE
    | raw.BatchResultField.GEOMETRY_IDENTITY
)


def _finish(operation: Any, fetch: bool = False) -> Any:
    if operation.wait() is not raw.OpState.COMPLETED:
        raise RuntimeError("fixture operation did not complete")
    result = operation.fetch_population_update() if fetch else None
    operation.close()
    return result


def _fixture(instance: Any) -> tuple[Any, Any, tuple[GeometryDefinitionRef, ...]]:
    adapter = instance.explicit()
    _finish(
        adapter.register_boxes(
            definition_values=np.array([BOX_DEFINITION], dtype=np.uint64),
            half_extents=np.array([BOX_HALF], dtype=np.float64),
            copy=True,
        )
    )
    _finish(
        adapter.register_spheres(
            definition_values=np.array([SPHERE_DEFINITION], dtype=np.uint64),
            radii=np.array([SPHERE_RADIUS], dtype=np.float64),
            copy=True,
        )
    )
    population = adapter.create_population(PopulationConfig())
    publication = _finish(
        population.enqueue_set(
            ORDINAL,
            object_values=np.array([BOX_OBJECT, SPHERE_OBJECT], dtype=np.uint64),
            geometry_types=np.array(
                [raw.GeometryType.BOX, raw.GeometryType.SPHERE], dtype=np.int32
            ),
            geometry_values=np.array([BOX_DEFINITION, SPHERE_DEFINITION], dtype=np.uint64),
            translations=np.array([(0.0, 0.0, 0.5), (2.0, 0.0, 0.5)], dtype=np.float64),
            rotations_xyzw=np.array([BOX_ROTATION, (0.0, 0.0, 0.0, 1.0)], dtype=np.float32),
            participation_bits=np.array([(1, 0), (2, 0)], dtype=np.uint64),
            copy=True,
        ),
        fetch=True,
    )
    if (publication.published_ordinal, publication.object_count) != (ORDINAL, 2):
        raise AssertionError(f"unexpected fixture publication: {publication}")
    definitions = (
        GeometryDefinitionRef(raw.GeometryType.BOX, BOX_DEFINITION),
        GeometryDefinitionRef(raw.GeometryType.SPHERE, SPHERE_DEFINITION),
    )
    return adapter, population, definitions


def _origin(row: int) -> tuple[float, float, float]:
    object_x = 0.0 if row % 2 == 0 else 2.0
    return (object_x + (0.875 if row % 5 == 4 else 0.125), 0.0625, 4.0)


def _queries(count: int) -> tuple[RaycastQuery, ...]:
    return tuple(
        RaycastQuery(
            origin=_origin(row),
            unit_direction=(0.0, 0.0, -1.0),
            min_distance=0.125,
            max_distance=8.0,
        )
        for row in range(count)
    )


def _add(digest: list[float], row: int, identity: tuple[Any, ...]) -> None:
    object_id, geometry_type, geometry_value, position, normal, distance = identity
    weight = row + 1
    digest[0] += 1
    digest[1] += weight * int(object_id)
    digest[2] += weight * int(geometry_type)
    digest[3] += weight * int(geometry_value)
    for index, value in enumerate((*position, *normal, distance), 4):
        digest[index] += weight * float(value)


def _expected_row(row: int) -> tuple[Any, ...] | None:
    if row % 5 == 4:
        return None
    if row % 2 == 0:
        return (
            ORDINAL,
            raw.HitTruth.EXACT_SURFACE,
            BOX_OBJECT,
            raw.GeometryType.BOX,
            BOX_DEFINITION,
            (0.125, 0.0625, 1.0),
            (0.0, 0.0, 1.0),
            3.0,
        )
    return (
        ORDINAL,
        raw.HitTruth.EXACT_SURFACE,
        SPHERE_OBJECT,
        raw.GeometryType.SPHERE,
        SPHERE_DEFINITION,
        (2.125, 0.0625, 0.5 + SPHERE_HEIGHT),
        (0.25, 0.125, ctypes.c_float(2.0 * SPHERE_HEIGHT).value),
        3.5 - SPHERE_HEIGHT,
    )


def _expected_rows(count: int) -> tuple[Any, ...]:
    return tuple(_expected_row(row) for row in range(count))


def _expected(count: int) -> tuple[float, ...]:
    digest = [0.0] * 11
    for row, hit in enumerate(_expected_rows(count)):
        if hit is not None:
            _add(digest, row, hit[2:])
    return tuple(digest)


def _typed_runner(population: Any, count: int) -> Callable[..., Any]:
    queries = _queries(count)
    target = population.target

    def execute(capture: bool = False) -> tuple[tuple[float, ...], tuple[Any, ...] | None]:
        digest = [0.0] * 11
        rows = [] if capture else None
        for row, query in enumerate(queries):
            result = target.raycast_closest(query)
            hit = result.hit
            if hit is not None:
                _add(
                    digest,
                    row,
                    (
                        hit.object_id,
                        hit.geometry_type,
                        hit.geometry_value,
                        hit.position,
                        hit.normal,
                        hit.distance,
                    ),
                )
            if rows is not None:
                rows.append(
                    None
                    if hit is None
                    else (
                        result.source_ordinal,
                        hit.truth,
                        hit.object_id,
                        hit.geometry_type,
                        hit.geometry_value,
                        hit.position,
                        hit.normal,
                        hit.distance,
                    )
                )
        return tuple(digest), None if rows is None else tuple(rows)

    return execute


def _raw_runner(instance: Any, population: Any, count: int) -> Callable[..., Any]:
    descriptors = tuple(query.to_raw(population.target.reference) for query in _queries(count))
    function = instance.library.exports["ovspatialquery_raycast_closest_sync"]

    def execute(capture: bool = False) -> tuple[tuple[float, ...], tuple[Any, ...] | None]:
        digest = [0.0] * 11
        rows = [] if capture else None
        for row, descriptor in enumerate(descriptors):
            output = raw.RaycastSingleResult()
            result = function(instance.handle, ctypes.byref(descriptor), ctypes.byref(output))
            if result.status != raw.ApiStatus.SUCCESS:
                raise RuntimeError(f"raw scalar query failed: {result.status}")
            if output.has_hit:
                hit = output.hit.data.exact
                identity = (
                    hit.identity.object_id.value,
                    hit.identity.geometry.type,
                    hit.identity.geometry.value,
                    hit.position,
                    hit.normal,
                    hit.distance,
                )
                _add(digest, row, identity)
            if rows is not None:
                rows.append(
                    None
                    if not output.has_hit
                    else (
                        output.source_ordinal,
                        output.hit.truth,
                        identity[0],
                        identity[1],
                        identity[2],
                        tuple(hit.position),
                        tuple(hit.normal),
                        hit.distance,
                    )
                )
        return tuple(digest), None if rows is None else tuple(rows)

    return execute


def _values(view: Any) -> list[int | float]:
    return np.from_dlpack(view).ravel().tolist()


def _batch_runner(instance: Any, population: Any, count: int) -> tuple[Callable[..., Any], Any]:
    origins = [coordinate for row in range(count) for coordinate in _origin(row)]
    columns = {
        "origins": np.array(origins, dtype=np.float64).reshape(count, 3),
        "unit_directions": np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (count, 1)),
        "min_distances": np.full(count, 0.125),
        "max_distances": np.full(count, 8.0),
        "query_bits": np.zeros((count, 2), dtype=np.uint64),
    }
    plan = instance.create_raycast_plan(
        RaycastPlanConfig(
            target=population.target.reference,
            hit_mode=raw.HitMode.CLOSEST,
            precision=raw.QueryPrecision.EXACT,
            maximum_query_count=count,
            maximum_hits_per_query=1,
            hit_capacity_per_slot=count,
            result_slot_count=1,
            result_fields=FIELDS,
        )
    )

    def execute(capture: bool = False) -> tuple[tuple[float, ...], tuple[Any, ...] | None]:
        operation, _ = plan.enqueue(**columns, copy=True)
        if operation.wait() is not raw.OpState.COMPLETED:
            raise RuntimeError("CPU batch did not complete")
        result = operation.fetch_raycast_batch()
        counts = _values(result.hit_ranges.hit_counts)
        offsets = _values(result.hit_ranges.hit_offsets)
        statuses = _values(result.hit_ranges.query_status)
        truncated = _values(result.hit_ranges.truncated)
        if (
            any(statuses)
            or any(truncated)
            or result.exact is None
            or result.geometry is None
            or result.common.source_ordinal != ORDINAL
            or result.common.truth is not raw.HitTruth.EXACT_SURFACE
        ):
            raise AssertionError("CPU batch result metadata changed")
        ids = _values(result.object_ids)
        types = _values(result.geometry.types)
        values = _values(result.geometry.values)
        positions = _values(result.exact.positions)
        normals = _values(result.exact.normals)
        distances = _values(result.exact.distances)
        digest = [0.0] * 11
        rows = [None] * count if capture else None
        for row, hit_count in enumerate(counts):
            if hit_count not in (0, 1):
                raise AssertionError("closest batch returned more than one hit")
            if hit_count:
                hit = int(offsets[row])
                identity = (
                    ids[hit],
                    types[hit],
                    values[hit],
                    positions[hit * 3 : hit * 3 + 3],
                    normals[hit * 3 : hit * 3 + 3],
                    distances[hit],
                )
                _add(digest, row, identity)
                if rows is not None:
                    rows[row] = (
                        result.common.source_ordinal,
                        result.common.truth,
                        identity[0],
                        identity[1],
                        identity[2],
                        tuple(identity[3]),
                        tuple(identity[4]),
                        identity[5],
                    )
        operation.close()
        return tuple(digest), None if rows is None else tuple(rows)

    return execute, plan


def _validate(actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
    if any(
        not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)
        for left, right in zip(actual, expected)
    ):
        raise AssertionError(f"fixture parity failed: {actual} != {expected}")


def _validate_rows(actual: tuple[Any, ...] | None, expected: tuple[Any, ...]) -> None:
    if actual is None or len(actual) != len(expected):
        raise AssertionError("fixture validation did not return every row")
    for index, (observed, wanted) in enumerate(zip(actual, expected)):
        if observed is None or wanted is None:
            if observed is not wanted:
                raise AssertionError(f"row {index} hit mask changed")
            continue
        if observed[:5] != wanted[:5]:
            raise AssertionError(f"row {index} identity changed: {observed} != {wanted}")
        left = (*observed[5], *observed[6], observed[7])
        right = (*wanted[5], *wanted[6], wanted[7])
        if any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6) for a, b in zip(left, right)):
            raise AssertionError(f"row {index} contact changed: {observed} != {wanted}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("typed", "raw", "batch"), required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    options = parser.parse_args()
    if options.queries < 1 or options.frames < 1 or options.warmup < 0:
        parser.error("queries and frames must be positive; warmup must be nonnegative")

    library = Library.open(require_explicit=True)
    instance = library.instance(
        InstanceConfig(operation_capacity=4, max_cpu_concurrency=1, host_byte_budget=1 << 30)
    )
    adapter = population = plan = None
    definitions: tuple[GeometryDefinitionRef, ...] = ()
    try:
        adapter, population, definitions = _fixture(instance)
        if options.mode == "typed":
            execute = _typed_runner(population, options.queries)
        elif options.mode == "raw":
            execute = _raw_runner(instance, population, options.queries)
        else:
            execute, plan = _batch_runner(instance, population, options.queries)
        expected = _expected(options.queries)
        digest, rows = execute(True)
        _validate(digest, expected)
        _validate_rows(rows, _expected_rows(options.queries))
        for _ in range(options.warmup):
            _validate(execute()[0], expected)
        samples = []
        for _ in range(options.frames):
            start = time.perf_counter_ns()
            observed, _ = execute()
            elapsed = time.perf_counter_ns() - start
            _validate(observed, expected)
            samples.append(elapsed)
        ordered = sorted(samples)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": options.mode,
                    "scope": "sync call and selected result observation"
                    if options.mode != "batch"
                    else "enqueue, wait, fetch, selected result observation, and operation retirement",
                    "observed_fields": [
                        "object_id",
                        "geometry_identity",
                        "position",
                        "normal",
                        "distance",
                    ],
                    "fixture": {
                        "native_case": "--workload rays --objects 2",
                        "source_ordinal": ORDINAL,
                        "objects": [
                            {
                                "kind": "box",
                                "object_id": BOX_OBJECT,
                                "definition_value": BOX_DEFINITION,
                                "translation": [0, 0, 0.5],
                                "half_extents": list(BOX_HALF),
                                "rotation_z_radians": 0.5,
                            },
                            {
                                "kind": "sphere",
                                "object_id": SPHERE_OBJECT,
                                "definition_value": SPHERE_DEFINITION,
                                "translation": [2, 0, 0.5],
                                "radius": SPHERE_RADIUS,
                            },
                        ],
                        "ray_generation": {
                            "object_index": "(query_index * 37) % 2",
                            "origin_offset_x": "0.875 every fifth query, otherwise 0.125",
                            "origin_offset_y": 0.0625,
                            "origin_z": 4,
                            "direction": [0, 0, -1],
                            "min_distance": 0.125,
                            "max_distance": 8,
                        },
                        "precision": "exact",
                    },
                    "run": {
                        "queries_per_frame": options.queries,
                        "frames": options.frames,
                        "warmup_frames": options.warmup,
                    },
                    "timing_ns": {
                        "samples": samples,
                        "median": statistics.median(samples),
                        "p95": p95,
                        "min": min(samples),
                        "max": max(samples),
                        "median_per_query": statistics.median(samples) / options.queries,
                    },
                    "installed": {
                        "python_module": str(Path(ovspatialquery.__file__).resolve()),
                        "library": str(library.path),
                        "version": list(library.version),
                    },
                },
                sort_keys=True,
            )
        )
    finally:
        if plan is not None:
            plan.close()
        if population is not None:
            population.close()
        if adapter is not None:
            for definition in definitions:
                adapter.destroy_definition(definition)
        instance.close()
        library.close()


if __name__ == "__main__":
    main()
