# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for using the ovspatialquery Python package with Ovstage.

Copy this file next to your program and `import ovsq`. The package itself takes
`ovstage.Filter`s, defaults single queries to exact precision, returns all hits of a ray
or overlap (`raycast_multiple`, `overlap_multiple`), gives batch results as NumPy
(`to_numpy()`), and closes everything an instance owns; these helpers cover
what it leaves to the caller: Ovstage ordinals and time units, choosing prims, naming
the prims a publication left out, unit directions and per-ray closest hits from a batch.
Needs `ovspatialquery`, `ovstage` and `numpy`.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import ovstage
from ovspatialquery import HitMode, OvstageExclusionReason, QueryPrecision, RaycastPlanConfig

# The only USD prim types an Ovstage population publishes. A prim of any other type that
# a filter matches (Xform, Cylinder, a light; a Scope or Material) is left out of the
# publication and listed as UNSUPPORTED_TYPE (NO_SPATIAL_SOURCE is for a prim of these
# types that has no world bounds);
# an empty filter selects exactly these types and their instance proxies.
SUPPORTED_TYPES = ("Cube", "Sphere", "Capsule", "Mesh")
# Ovstage's `usd-prim-type` of a prim inside a native instance (`instanceable = true`).
INSTANCE_TYPES = tuple(name + "Instance" for name in SUPPORTED_TYPES)




class _ColumnView:
    """One attribute column of an Ovstage 0.4 read group, with the group's prim rows."""

    __slots__ = ("_group", "_column")

    def __init__(self, group, column):
        self._group = group
        self._column = column

    def __getattr__(self, name):
        try:
            return getattr(self._column, name)
        except AttributeError:
            return getattr(self._group, name)


def _read_columns(group):
    """A 0.3 read group holds one attribute; a 0.4 group one column per attribute."""
    columns = getattr(group, "columns", None)
    return (group,) if columns is None else tuple(_ColumnView(group, c) for c in columns)


def _handle_value(handle):
    """An Ovstage token or path as an int: 0.3 returns ints, 0.4 handles."""
    return int(getattr(handle, "raw_id", handle))

def physics_desc(ovstage):
    """What ovspatialquery reads from a stage on Ovstage 0.3 and 0.4: 0.3's ALL domain, or
    0.4's RENDERING, which has no PHYSICS domain, plus every prim's physics,
    physxConvexHullCollision and material:binding properties for convex hulls, their vertex
    limits and physics materials."""
    population = ovstage.population
    domain = ovstage.PopulationDomain
    physics = population.Selector(
        prim_predicate=population.PrimPredicate.all(),
        property_predicate=population.PropertyPredicate.in_namespace(
            "physics", "physxConvexHullCollision", "material:binding"
        ),
    )
    return population.Desc(domains=getattr(domain, "ALL", domain.RENDERING), selectors=[physics])

class Scene:
    """An Ovstage stage opened from USD, with the ordinal bookkeeping done for you.

    Ovstage writes are numbered by ordinals; a population can publish only the newest
    sealed ordinal. `Scene` opens at ordinal 1, computes world transforms into ordinal 2,
    and every `set_time` writes and seals the next ordinal. `scene.ordinal` is always
    the ordinal to publish.

    Times are USD time codes (the numbers keying `timeSamples`). Ovstage itself takes
    seconds; `Scene` divides by the layer's `timeCodesPerSecond` (USD default 24).
    """

    def __init__(self, usd: str | Path, *, time_code: float = 0.0, name: str = "scene") -> None:
        text = Path(usd).read_text(encoding="utf-8") if _is_path(usd) else str(usd)
        self.usda = text
        self.time_codes_per_second = _time_codes_per_second(text)
        self.stage = ovstage.Stage(name)
        # Opening from a string resolves relative asset paths against the cwd; open the
        # file with `ovstage.population.open_usd` instead if the layer references files.
        ovstage.population.open_usd_from_string_with_desc(
            self.stage, text, 1, self.seconds(time_code), physics_desc(ovstage))
        self.stage.advance_write_floor(ordinal=1).wait()
        self.stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        self.stage.advance_write_floor(ordinal=2).wait()
        self.ordinal = 2
        self._paths = ovstage.PathDictionary(self.stage)

    def seconds(self, time_code: float) -> float:
        return float(time_code) / self.time_codes_per_second

    def set_time(self, time_code: float) -> int:
        """Re-evaluate animated prims at `time_code` into a new sealed ordinal; returns it."""
        self.ordinal += 1
        ovstage.population.update_from_usd_time(self.stage, self.ordinal, self.seconds(time_code))
        self.stage.advance_write_floor(ordinal=self.ordinal).wait()
        return self.ordinal

    def path(self, object_id: int) -> str:
        """Prim path of a hit's `object_id` (Ovstage objects are identified by path id)."""
        return str(self._paths.path_to_string(_prim_path(object_id)))

    def close(self) -> None:
        """Destroy the stage; close every population built on it first."""
        self.stage.destroy()

    def __enter__(self) -> "Scene":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _prim_path(value: Any) -> Any:
    """The path dictionary's handle for a path value: an int on Ovstage 0.3, a PrimPath on 0.4."""
    prim_path = getattr(ovstage, "PrimPath", None)
    return int(value) if prim_path is None else prim_path(int(value))


def _is_path(value: str | Path) -> bool:
    return isinstance(value, Path) or ("\n" not in str(value) and Path(str(value)).is_file())


def _time_codes_per_second(usda: str) -> float:
    match = re.search(r"\btimeCodesPerSecond\s*=\s*([0-9.eE+-]+)", usda)
    if match is None:
        match = re.search(r"\bframesPerSecond\s*=\s*([0-9.eE+-]+)", usda)
    return float(match.group(1)) if match else 24.0


def gprim_paths(
    usd: str | Path, roots: Iterable[str] = ("/",), types: Iterable[str] = SUPPORTED_TYPES
) -> list[str]:
    """Sorted paths of every supported gprim under `roots`, read with Ovstage.

    Use them in `ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])`
    to select exactly those prims. Prims inside native instances (`instanceable = true`) are
    included at their instance-proxy paths.
    """
    wanted = set(types)
    if not wanted:
        return []
    names = sorted(wanted | {name + "Instance" for name in wanted})
    prefixes = tuple(root.rstrip("/") + "/" for root in roots)
    stage = ovstage.Stage("gprim-paths")
    try:
        if _is_path(usd):
            ovstage.population.open_usd_with_desc(
                stage, str(usd), 1, 0.0, physics_desc(ovstage))
        else:
            ovstage.population.open_usd_from_string_with_desc(
                stage, str(usd), 1, 0.0, physics_desc(ovstage))
        stage.advance_write_floor(ordinal=1).wait()
        stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        stage.advance_write_floor(ordinal=2).wait()
        paths = ovstage.PathDictionary(stage)
        path_token = paths.intern_token("usd-path")

        def read_paths(selected: ovstage.Filter) -> set[str]:
            found = set()
            with stage.query(filter=selected, attrs=[path_token]) as query:
                query.wait()
                if query.result().total_prim_count == 0:
                    return found
                with stage.read_attributes(
                    query, [path_token], ovstage.OrdinalRange.latest(2)
                ) as read:
                    read.wait()
                    for read_group in read.groups():
                        try:
                            for group in _read_columns(read_group):
                                if not group.is_delete:
                                    found.update(
                                        paths.path_to_string(_prim_path(token))
                                        for token in np.asarray(group.array(0)).reshape(-1)
                                    )
                        finally:
                            stage.release_group(read_group)
            return found

        prototypes = read_paths(
            ovstage.Filter(
                [ovstage.Predicate("usd-schemas", ovstage.FilterOp.CONTAINS, ["PrototypeRootAPI"])]
            )
        )
        selected = ovstage.Filter(
            [ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, names)]
        )
        return sorted(
            path
            for path in read_paths(selected)
            if (path + "/").startswith(prefixes)
            and not any(path == root or path.startswith(root + "/") for root in prototypes)
        )
    finally:
        stage.destroy()


def gprims_under(*roots: str) -> ovstage.Filter:
    """Filter for every publishable prim below `roots`, instance proxies included.

    A `usd-path PREFIX` alone also matches the groups (Xform, XformInstance) below a root,
    which publication leaves out as UNSUPPORTED_TYPE; this ANDs it with every published type.
    """
    prefixes = [root.rstrip("/") + "/" for root in roots]
    return ovstage.Filter(
        [
            ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, prefixes),
            ovstage.Predicate(
                "usd-prim-type", ovstage.FilterOp.IN, [*SUPPORTED_TYPES, *INSTANCE_TYPES]
            ),
        ]
    )


def publish(
    population: Any, ordinal: int, base: int | None = None, *, allow_exclusions: bool = False
) -> Any:
    """Publish `ordinal` (set, or update from `base`); returns the update record.

    `ordinal` must be the newest sealed Ovstage ordinal (`Scene.ordinal`); `base` the
    population's previous published ordinal. Stage content never fails a publication:
    a prim that cannot be published exactly (wrong type, a Sphere or Capsule under
    non-uniform scale, broken geometry) is left out and listed in the result. This
    raises naming each such prim and its reason unless `allow_exclusions`. Check
    `.object_count` too: a filter that matches nothing publishes zero objects.
    """
    if base is None:
        operation = population.enqueue_set(ordinal)
    else:
        operation = population.enqueue_update(base, ordinal)
    with operation:
        operation.wait()
        result = operation.fetch_population_update()
    if result.excluded_object_count and not allow_exclusions:
        listed = exclusions(population, result)
        left_out = ", ".join(f"{path} {reason}" for path, reason in listed[:10])
        left_out += f" and {len(listed) - 10} more" if len(listed) > 10 else ""
        raise RuntimeError(
            f"publishing ordinal {ordinal} left out {result.excluded_object_count} prim(s): "
            f"{left_out}. Fix the filter or the prims, or pass allow_exclusions=True."
        )
    return result


def exclusions(population: Any, result: Any) -> list[tuple[str, str]]:
    """(prim path, reason name) of every prim a publication left out, sorted by object id.

    Reasons: NO_SPATIAL_SOURCE (a Cube/Sphere/Capsule/Mesh without world bounds),
    UNSUPPORTED_TYPE (any other type, Scope and Material included),
    UNSUPPORTED_SHAPE (Sphere or Capsule under non-uniform scale or shear),
    MALFORMED_GEOMETRY, RESHAPE_PENDING (device-prepared versions only).
    """
    paths = ovstage.PathDictionary(population.adapter.stage)
    return [
        (str(paths.path_to_string(_prim_path(object_id))), OvstageExclusionReason(int(reason)).name)
        for object_id, reason in zip(result.excluded_objects, result.exclusion_reasons)
    ]


def unit(vector: Sequence[float]) -> tuple[float, float, float]:
    """Normalize in double precision; queries reject directions that are not unit length."""
    length = math.sqrt(sum(float(v) * float(v) for v in vector))
    if length == 0.0:
        raise ValueError("zero-length direction")
    return tuple(float(v) / length for v in vector)  # type: ignore[return-value]


def yaw_xyzw(degrees: float) -> tuple[float, float, float, float]:
    """Unit quaternion (x, y, z, w) for a rotation about +Z; the API's quaternion order."""
    half = math.radians(degrees) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class RaycastBatch:
    """A reusable closest-hit raycast plan: create once, `run` every frame.

    `run` takes origins [N,3], directions [N,3] (normalized here, sent as float32) and
    max distances, and returns per-ray arrays: `hit` bool, `distance` float64 (inf on a
    miss) and `object_id` uint64 (0 on a miss). Plans default to bounds precision, so
    this one asks for EXACT. Use `with RaycastBatch(...) as batch:` or `close()`.
    """

    def __init__(
        self, instance: Any, target: Any, max_rays: int, precision=QueryPrecision.EXACT
    ) -> None:
        from ovspatialquery import BatchResultField

        self.max_rays = int(max_rays)
        self.plan = instance.create_raycast_plan(
            RaycastPlanConfig(
                target=target.reference,
                hit_mode=HitMode.CLOSEST,
                precision=precision,
                maximum_query_count=self.max_rays,
                maximum_hits_per_query=1,
                hit_capacity_per_slot=self.max_rays,
                result_slot_count=1,
                result_fields=BatchResultField.DISTANCE,
            )
        )

    def run(
        self, origins: Any, directions: Any, max_distance: Any, min_distance: Any = 0.0
    ) -> dict:
        origins = np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 3)
        count = len(origins)
        if count > self.max_rays:
            raise ValueError(f"{count} rays exceed the plan's {self.max_rays}")
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        operation, _ = self.plan.enqueue(
            origins=origins,
            unit_directions=np.ascontiguousarray(directions, dtype=np.float32),
            min_distances=np.broadcast_to(np.float64(min_distance), (count,)).copy(),
            max_distances=np.broadcast_to(np.float64(max_distance), (count,)).copy(),
            copy=True,
        )
        with operation:
            operation.wait()
            result = operation.fetch_raycast_batch().to_numpy()
        ranges = result.hit_ranges
        if np.any(ranges.query_status != 0):
            raise RuntimeError(
                f"raycast rows failed with statuses {np.unique(ranges.query_status)}"
            )
        offsets = ranges.hit_offsets.astype(np.int64)
        hit = ranges.hit_counts > 0
        distance = np.full(count, np.inf)
        object_id = np.zeros(count, dtype=np.uint64)
        distance[hit] = result.exact.distances[offsets[hit]]
        object_id[hit] = result.object_ids[offsets[hit]]
        return {"hit": hit, "distance": distance, "object_id": object_id}

    def close(self) -> None:
        self.plan.close()

    def __enter__(self) -> "RaycastBatch":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
