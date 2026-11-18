# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object pairs between two populations: one snapshot, then entered/exited tracking.

Two animated pucks (Spheres) slide over two static pads (Cubes). The layer authors
timeCodesPerSecond = 4; ovsq.Scene converts time codes to Ovstage seconds.
Pair tests compare world axis-aligned bounds (closed intervals), not exact shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import ovsq  # noqa: E402
import ovstage  # noqa: E402
from ovspatialquery import (  # noqa: E402
    ApiError,
    ApiStatus,
    Library,
    PairReporterConfig,
    PairReporterOutcome,
)

USDA = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 8
    timeCodesPerSecond = 4
)
def Xform "World"
{
    def Xform "Pucks"
    {
        def Sphere "A"
        {
            double radius = 0.5
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double3 xformOp:translate.timeSamples = {
                0: (-3, 0, 0),
                8: (3, 0, 0),
            }
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Sphere "B"
        {
            double radius = 0.5
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double3 xformOp:translate.timeSamples = {
                0: (0, 5, 0),
                4: (0, 5, 0),
                8: (3, 0, 0),
            }
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def Xform "Pads"
    {
        def Cube "Left"
        {
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (-3, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Cube "Right"
        {
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (3, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
}
"""

A, B = "/World/Pucks/A", "/World/Pucks/B"
LEFT, RIGHT = "/World/Pads/Left", "/World/Pads/Right"
# Hand answers per time code: pucks' bounds against pads' bounds.
EXPECTED = {
    0: {
        "pairs": {(A, LEFT)},
        "entered": {(A, LEFT)},
        "exited": set(),
    },  # first advance: all entered
    4: {"pairs": set(), "entered": set(), "exited": {(A, LEFT)}},  # A at x=0
    5: {"pairs": set(), "entered": set(), "exited": set()},  # moved, no change
    8: {"pairs": {(A, RIGHT), (B, RIGHT)}, "entered": {(A, RIGHT), (B, RIGHT)}, "exited": set()},
}


def named(scene: ovsq.Scene, rows) -> set[tuple[str, str]]:
    return {(scene.path(first), scene.path(second)) for first, second in rows}


def path_filter(paths: list[str]) -> ovstage.Filter:
    return ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])


def advance(reporter):
    """One reporter step; entered/exited are [K, 2] uint64 NumPy arrays of (first, second) ids."""
    operation, _reservation = reporter.advance()
    with operation:  # waits, then closes; to_numpy copies the deltas so they outlive it
        operation.wait()
        return operation.fetch_pair_reporter().to_numpy()


def main() -> dict:
    out: dict = {"steps": {}}
    # The scene outlives the instance: populations close before their stage is destroyed.
    with (
        ovsq.Scene(USDA, time_code=0) as scene,
        Library.open(require_ovstage=True) as library,
        library.instance() as instance,
    ):
        adapter = instance.ovstage(ovstage, scene.stage)
        pucks = adapter.create_population(path_filter([A, B]))
        pads = adapter.create_population(path_filter([LEFT, RIGHT]))
        assert ovsq.publish(pucks, scene.ordinal).object_count == 2
        assert ovsq.publish(pads, scene.ordinal).object_count == 2

        # Snapshot: rows are (first-population id, second-population id).
        snapshot = instance.query_pairs(pucks.target, pads.target, capacity=8)
        assert not snapshot.truncated
        assert named(scene, snapshot.rows) == EXPECTED[0]["pairs"], snapshot
        out["snapshot_t0"] = sorted(named(scene, snapshot.rows))

        # Capacities: active >= max pairs alive at once, delta >= max entered (or exited) per step.
        reporter = instance.create_pair_reporter(
            PairReporterConfig(
                first=pucks.target.reference,
                second=pads.target.reference,
                active_pair_capacity=2 * 2,
                delta_pair_capacity=2 * 2,
            )
        )
        pucks_base = scene.ordinal
        for time_code, expected in EXPECTED.items():
            if time_code != 0:
                scene.set_time(time_code)
                # Republish before every advance, even when no pair changes (time code 5).
                ovsq.publish(pucks, scene.ordinal, pucks_base)
                pucks_base = scene.ordinal
            result = advance(reporter)
            assert result.outcome is PairReporterOutcome.ADVANCED, result.outcome
            assert result.first_source_ordinal == scene.ordinal  # pads stay at ordinal 2
            entered, exited = named(scene, result.entered_pairs), named(scene, result.exited_pairs)
            assert entered == expected["entered"], (time_code, entered)
            assert exited == expected["exited"], (time_code, exited)
            assert result.accepted_active_pair_count == len(expected["pairs"])
            out["steps"][time_code] = {
                "entered": sorted(entered),
                "exited": sorted(exited),
                "active": result.accepted_active_pair_count,
            }

        # Advancing again with no new publication on either side is rejected.
        try:
            reporter.advance()
            raise AssertionError("advance without a new publication should fail")
        except ApiError as error:
            assert error.status is ApiStatus.STALE_ORDINAL, error
            out["advance_without_publish"] = error.status.name

        # Too-small capacities are outcomes, not exceptions, and the deltas are empty.
        snapshot = instance.query_pairs(pucks.target, pads.target, capacity=1)
        assert snapshot.truncated and len(snapshot.rows) == 1
        out["outcomes"] = {"snapshot_capacity_1_truncated": snapshot.truncated}
        for active, delta, outcome in (
            (1, 4, PairReporterOutcome.ACTIVE_CAPACITY),
            (4, 1, PairReporterOutcome.DELTA_CAPACITY),
        ):
            config = PairReporterConfig(
                first=pucks.target.reference,
                second=pads.target.reference,
                active_pair_capacity=active,
                delta_pair_capacity=delta,
            )
            with instance.create_pair_reporter(config) as small:
                result = advance(small)  # 2 pairs enter at once
            assert result.outcome is outcome, result.outcome
            assert len(result.entered_pairs) == 0 and len(result.exited_pairs) == 0
            out["outcomes"][f"active_{active}_delta_{delta}"] = result.outcome.name
    return out


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
