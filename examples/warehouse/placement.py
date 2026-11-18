# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded placement waves: query one snapshot, accept one box, publish, repeat.

This is a source-grounded randomizer analogue, not arbitrary mesh containment or
physics. All occupied objects are analytical boxes. Acceptance slots are allocated
at setup and parked outside the sampling area until moved into an accepted pose.
"""

from time import perf_counter

import numpy as np
from ovspatialquery import (
    ApiStatus,
    BatchRowStatus,
    GeometryDefinitionRef,
    GeometryType,
    HitMode,
    OpState,
    OverlapPlanConfig,
    OrdinalSelection,
    OverlapBatchShape,
    PopulationConfig,
    QueryPrecision,
)

from sources import QueryStage, population

HALF = np.array((0.20, 0.35, 0.20))
SHAPES = np.array(
    [(0.16 + 0.025 * (i % 4), 0.28 + 0.09 * (i % 3), 0.17 + 0.025 * (i // 2)) for i in range(8)]
)
DEFINITIONS = np.arange(740000, 740009, dtype=np.uint64)


def obstacle_data(count=256, seed=47):
    """Return deterministic centers, half-extents and a square sampling domain."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    extent = max(20.0, np.sqrt(count) * 0.72)
    side = int(np.ceil(np.sqrt(count * 1.4)))
    cells = rng.permutation(side * side)[:count]
    pitch = extent / side
    xy = np.column_stack((cells % side, cells // side))
    xy = (xy + rng.uniform(0.3, 0.7, (count, 2))) * pitch
    halves = SHAPES[np.arange(count) % len(SHAPES)].copy()
    centers = np.column_stack((xy[:, 0] + 15, halves[:, 1], xy[:, 1] + 60))
    return centers, halves, ((15.0, 60.0), (15.0 + extent, 60.0 + extent))


def parked_centers(capacity):
    return np.column_stack(
        (np.arange(capacity) * 2.0 + 15, np.full(capacity, -100.0), np.full(capacity, 70.0))
    )


def occupancy_oracle(candidates, centers, halves):
    """Independent complete Boolean vector; inclusive solid AABB overlap.

    The old client uses a 1e-4 touching band; this oracle deliberately matches
    the query contract instead of claiming exact client epsilon-policy parity.
    """
    return np.array(
        [np.any(np.all(np.abs(centers - c) <= halves + HALF, axis=1)) for c in candidates]
    )


def ledger_any(candidates, centers, halves):
    """Matching any-hit linear AABB baseline, with early exit per candidate."""
    bounds = [(tuple(c - h), tuple(c + h)) for c, h in zip(centers, halves)]
    results = []
    for center in candidates:
        lo, hi = center - HALF, center + HALF
        results.append(
            any(
                all(hi[a] >= lower[a] and lo[a] <= upper[a] for a in range(3))
                for lower, upper in bounds
            )
        )
    return np.array(results)


class Placement:
    def __init__(self, instance, source="ovstage", count=256, candidates=256, seed=47, capacity=32):
        if source not in ("explicit", "ovstage"):
            raise ValueError("source must be explicit or ovstage")
        self.instance, self.source, self.capacity = instance, source, capacity
        self.plan = self.population = self.stage_source = None
        self.registered = False
        self.reset(count, candidates, seed)

    def reset(self, count=None, candidates=None, seed=None):
        self._close_queries()
        self.count = int(count if count is not None else self.count)
        self.candidates = int(candidates if candidates is not None else self.candidates)
        self.seed = int(seed if seed is not None else self.seed)
        if self.candidates < 1 or self.capacity < 1:
            raise ValueError("candidate count and capacity must be positive")
        setup_start = perf_counter()
        start = setup_start
        self.centers, self.halves, self.bounds = obstacle_data(self.count, self.seed)
        self.accepted_centers = []
        self.candidate_centers = np.empty((0, 3))
        self.candidate_hits = np.empty(0, dtype=bool)
        self.accepted_index = None
        self.wave = 0
        self.query_ms = self.publish_ms = self.loop_ms = 0.0
        self.query_ordinal = 0
        self.rng = np.random.default_rng(self.seed + 1)
        self.paths = [f"/Placement/Cargo{i:05}" for i in range(self.count + self.capacity)]
        self.all_centers = np.concatenate((self.centers, parked_centers(self.capacity)))
        self.all_halves = np.concatenate((self.halves, np.tile(HALF, (self.capacity, 1))))
        self.prepare_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        self.usd_author_ms = self.source_load_ms = 0.0
        if self.source == "ovstage":
            rows = ["#usda 1.0", 'def "Placement"', "{"]
            for path, center, half in zip(self.paths, self.all_centers, self.all_halves):
                size = [format(float(value), ".17g") for value in half * 2]
                position = [format(float(value), ".17g") for value in center]
                rows.extend(
                    [
                        f'    def Cube "{path.rsplit("/", 1)[1]}"',
                        "    {",
                        "        double size = 1",
                        "        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]",
                        "        matrix4d xformOp:transform = "
                        f"(({size[0]}, 0, 0, 0), (0, {size[1]}, 0, 0), "
                        f"(0, 0, {size[2]}, 0), ({', '.join(position)}, 1))",
                        '        uniform token[] xformOpOrder = ["xformOp:transform"]',
                        "    }",
                    ]
                )
            rows.extend(("}", ""))
            scene_text = "\n".join(rows)
            self.usd_author_ms = (perf_counter() - start) * 1000
            loading = perf_counter()
            self.stage_source = QueryStage(scene_text)
            self.source_load_ms = (perf_counter() - loading) * 1000
            self.stage_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            self.population, receipt, _ = population(
                self.instance, self.stage_source.stage, self.paths
            )
            self.ordinal = self.stage_source.ordinal
            self._check_receipt(receipt, self.ordinal)
        else:
            adapter = self.instance.explicit()
            # Geometry identities live for the instance lifetime, even after destruction.
            # Reset replaces the population while retaining these unchanged definitions.
            if not self.registered:
                op = adapter.register_boxes(
                    definition_values=np.asarray(DEFINITIONS, dtype=np.uint64),
                    half_extents=np.concatenate((SHAPES, HALF[None])).astype(np.float64),
                )
                with op:
                    op.wait()
                    state, status, _ = op.status()
                    if state != OpState.COMPLETED or status != ApiStatus.SUCCESS:
                        raise RuntimeError(
                            f"Placement geometry registration failed: {state.name}/{status.name}"
                        )
                self.registered = True
            self.stage_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            self.population = adapter.create_population(PopulationConfig(2))
            self.ordinal = 1
            op = self.population.enqueue_set(
                self.ordinal, **self._instances(np.arange(len(self.paths)))
            )
            self._finish_publication(op, self.ordinal)
        self.initial_publish_ms = (perf_counter() - start) * 1000
        self._create_plan()
        self.initial_plan_setup_ms = self.plan_setup_ms
        self.setup_ms = (perf_counter() - setup_start) * 1000
        return self.state()

    def _check_receipt(self, receipt, ordinal):
        if (
            receipt.published_ordinal != ordinal
            or receipt.object_count != len(self.paths)
            or receipt.excluded_object_count
        ):
            raise AssertionError(
                f"Incomplete placement publication: {receipt}; expected ordinal {ordinal}, {len(self.paths)} objects"
            )

    def _finish_publication(self, op, ordinal):
        with op:
            op.wait()
            state, status, _ = op.status()
            if state != OpState.COMPLETED or status != ApiStatus.SUCCESS:
                raise RuntimeError(
                    f"Placement publication {ordinal} failed: {state.name}/{status.name}"
                )
            self._check_receipt(op.fetch_population_update(), ordinal)

    def _create_plan(self):
        if self.plan is not None:
            self.plan.close()
        start = perf_counter()
        shape = OverlapBatchShape.box(HALF)
        self.plan = self.instance.create_overlap_plan(
            OverlapPlanConfig(
                target=self.population.target.reference,
                shape=shape,
                hit_mode=HitMode.ANY,
                precision=QueryPrecision.EXACT,
                maximum_query_count=self.candidates,
                maximum_hits_per_query=1,
                hit_capacity_per_slot=self.candidates,
            )
        )
        self.plan_setup_ms = (perf_counter() - start) * 1000

    def _instances(self, indices):
        kinds = np.where(indices < self.count, indices % 8, 8)
        return {
            "object_values": (indices + 1).astype(np.uint64),
            "geometry_types": np.full(len(indices), GeometryType.BOX, dtype=np.int32),
            "geometry_values": np.asarray(DEFINITIONS[kinds], dtype=np.uint64),
            "translations": self.all_centers[indices].astype(np.float64),
            "rotations_xyzw": np.tile(np.array([0, 0, 0, 1.0]), (len(indices), 1)),
        }

    def rewind(self):
        """Restore acceptance slots without reloading or reparsing the source scene."""
        self.all_centers[self.count :] = parked_centers(self.capacity)
        base = self.ordinal
        if self.source == "ovstage":
            matrices = np.tile(np.eye(4), (self.capacity, 1, 1))
            matrices[:, range(3), range(3)] = HALF * 2
            matrices[:, 3, :3] = self.all_centers[self.count :]
            self.ordinal = self.stage_source.transforms(self.paths[self.count :], matrices)
            op = self.population.enqueue_update(base, self.ordinal)
        else:
            self.ordinal += 1
            op = self.population.enqueue_update(
                base,
                self.ordinal,
                **self._instances(np.arange(self.count, self.count + self.capacity)),
            )
        self._finish_publication(op, self.ordinal)
        self.accepted_centers = []
        self.candidate_centers = np.empty((0, 3))
        self.candidate_hits = np.empty(0, dtype=bool)
        self.accepted_index = None
        self.wave = 0
        self.query_ms = self.publish_ms = self.loop_ms = 0.0
        self.query_ordinal = 0
        self.rng = np.random.default_rng(self.seed + 1)
        self._create_plan()

    def next_candidates(self):
        lo, hi = np.array(self.bounds)
        xz = self.rng.uniform(lo + HALF[[0, 2]], hi - HALF[[0, 2]], (self.candidates, 2))
        return np.column_stack((xz[:, 0], np.full(self.candidates, HALF[1]), xz[:, 1]))

    def query(self, centers, ordinal=None):
        expected_ordinal = self.ordinal if ordinal is None else ordinal
        start = perf_counter()
        op, _ = self.plan.enqueue(
            translations=np.asarray(centers, dtype=np.float64),
            rotations_xyzw=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(centers), 1)),
            ordinal=(
                OrdinalSelection.head()
                if ordinal is None
                else OrdinalSelection.retained_exact(ordinal)
            ),
        )
        with op:
            op.wait()
            result = op.fetch_overlap_batch()
            ranges = result.hit_ranges

            statuses = np.from_dlpack(ranges.query_status)
            if np.any(statuses != BatchRowStatus.SUCCESS) or np.any(
                np.from_dlpack(ranges.truncated)
            ):
                raise RuntimeError("Placement overlap batch failed or truncated")
            hits = np.from_dlpack(ranges.hit_counts) != 0
            del statuses
            ordinal = int(result.common.source_ordinal)
            if ordinal != expected_ordinal:
                raise AssertionError(
                    f"Placement query captured ordinal {ordinal}, expected {expected_ordinal}; head {self.population.info()}"
                )
        return hits, ordinal, (perf_counter() - start) * 1000

    def step(self, validate=False):
        if len(self.accepted_centers) >= self.capacity:
            return self.state()
        start = perf_counter()
        self.candidate_centers = self.next_candidates()
        self.candidate_hits, self.query_ordinal, self.query_ms = self.query(self.candidate_centers)
        if validate:
            expected = occupancy_oracle(self.candidate_centers, self.all_centers, self.all_halves)
            np.testing.assert_array_equal(self.candidate_hits, expected)
        clear = np.flatnonzero(~self.candidate_hits)
        self.accepted_index = int(clear[0]) if len(clear) else None
        self.publish_ms = 0.0
        if self.accepted_index is not None:
            center = self.candidate_centers[self.accepted_index].copy()
            index = self.count + len(self.accepted_centers)
            self.all_centers[index] = center
            publish = perf_counter()
            base = self.ordinal
            if self.source == "ovstage":
                matrix = np.eye(4)
                matrix[range(3), range(3)] = HALF * 2
                matrix[3, :3] = center
                next_ordinal = self.stage_source.transforms([self.paths[index]], [matrix])
                op = self.population.enqueue_update(base, next_ordinal)
            else:
                next_ordinal = self.ordinal + 1
                op = self.population.enqueue_update(
                    base, next_ordinal, **self._instances(np.array([index]))
                )
            self._finish_publication(op, next_ordinal)
            self.ordinal = next_ordinal
            self.publish_ms = (perf_counter() - publish) * 1000
            self.accepted_centers.append(center)
        self.wave += 1
        self.loop_ms = (perf_counter() - start) * 1000
        return self.state()

    def state(self):
        return dict(
            source=self.source,
            count=self.count,
            candidates=self.candidates,
            seed=self.seed,
            wave=self.wave,
            accepted=len(self.accepted_centers),
            capacity=self.capacity,
            blocked=int(np.count_nonzero(self.candidate_hits)),
            accepted_index=self.accepted_index,
            query_ordinal=self.query_ordinal,
            ordinal=self.ordinal,
            query_ms=self.query_ms,
            publish_ms=self.publish_ms,
            loop_ms=self.loop_ms,
            complete=len(self.accepted_centers) >= self.capacity,
            bounds=self.bounds,
            half=HALF.tolist(),
            accepted_centers=[c.tolist() for c in self.accepted_centers],
            candidate_centers=self.candidate_centers.tolist(),
            candidate_hits=self.candidate_hits.tolist(),
        )

    def _close_queries(self):
        for name in ("plan", "population", "stage_source"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.close()
                setattr(self, name, None)

    def close(self):
        self._close_queries()
        if self.registered:
            for definition in DEFINITIONS:
                self.instance.explicit().destroy_definition(
                    GeometryDefinitionRef(GeometryType.BOX, int(definition))
                )
            self.registered = False
