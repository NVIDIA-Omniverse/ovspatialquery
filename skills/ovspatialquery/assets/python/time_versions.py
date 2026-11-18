# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Animated USD: publish one version per time code, query the head and older versions.

A ball rises along +y (time samples), a static ceiling sits above it. The layer authors
timeCodesPerSecond = 10, so time code 10 is 1 second for Ovstage (ovsq.Scene converts).
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
    OrdinalSelection,
    PopulationConfig,
    RaycastQuery,
)

USDA = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 20
    timeCodesPerSecond = 10
)
def Xform "World"
{
    def Sphere "Ball"
    {
        double radius = 0.5
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        double3 xformOp:translate.timeSamples = {
            0: (0, 1, 0),
            20: (0, 9, 0),
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Ceiling"
    {
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        double3 xformOp:translate = (0, 12, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""


def ball_distance(time_code: float) -> float:
    """Hand answer: ray from y = -2 along +y hits the ball's bottom at y = 1 + 0.4 * tc - 0.5."""
    return (1.0 + 0.4 * time_code - 0.5) - (-2.0)


def main() -> dict:
    out: dict = {"ordinals": {}, "head": {}, "retained": {}}
    # The scene outlives the instance: populations close before their stage is destroyed.
    with (
        ovsq.Scene(USDA, time_code=0) as scene,  # open at a time code: time samples need it
        Library.open(require_ovstage=True) as library,
        library.instance() as instance,
    ):
        assert scene.seconds(10) == 1.0
        paths = ovsq.gprim_paths(USDA, roots=["/World"])
        assert paths == ["/World/Ball", "/World/Ceiling"], paths
        selected = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])
        # Single queries are EXACT by default.
        ray = dict(origin=(0.0, -2.0, 0.0), unit_direction=ovsq.unit((0, 1, 0)), max_distance=50.0)

        adapter = instance.ovstage(ovstage, scene.stage)
        # Keeps 2 prior versions besides the head (also the default).
        population = adapter.create_population(
            selected, PopulationConfig(retained_ordinal_capacity=2)
        )
        # Keeps only 1 prior version: shows what happens when a version falls out.
        short = adapter.create_population(selected, PopulationConfig(retained_ordinal_capacity=1))

        published = {}  # time code -> ordinal
        base = None
        for time_code in (0, 10, 20):
            if time_code != 0:
                scene.set_time(time_code)  # writes and seals the next ordinal
            for target in (population, short):
                update = ovsq.publish(target, scene.ordinal, base)
                assert update.published_ordinal == scene.ordinal
                assert update.object_count == 2, update
            base = scene.ordinal  # the next update's base is the previous published ordinal
            published[time_code] = scene.ordinal
            head = population.target.raycast_closest(RaycastQuery(**ray))
            assert head.source_ordinal == scene.ordinal
            assert scene.path(head.hit.object_id) == "/World/Ball"
            assert abs(head.hit.distance - ball_distance(time_code)) < 1e-4, head
            out["head"][time_code] = round(head.hit.distance, 6)
        out["ordinals"] = published
        assert published == {0: 2, 10: 3, 20: 4}

        # Older versions stay queryable by their source ordinal.
        for time_code in (0, 10):
            old = population.target.raycast_closest(
                RaycastQuery(**ray, ordinal=OrdinalSelection.retained_exact(published[time_code]))
            )
            assert old.source_ordinal == published[time_code]  # not the head (4)
            assert abs(old.hit.distance - ball_distance(time_code)) < 1e-4, old
            out["retained"][time_code] = round(old.hit.distance, 6)

        info = population.info()
        assert (info.head_ordinal, info.retained_ordinal_count) == (4, 2), info
        out["info"] = {
            "head_ordinal": info.head_ordinal,
            "retained_ordinal_count": info.retained_ordinal_count,
        }

        # capacity 1: ordinal 3 is retained, ordinal 2 was dropped.
        short.target.raycast_closest(
            RaycastQuery(**ray, ordinal=OrdinalSelection.retained_exact(3))
        )
        try:
            short.target.raycast_closest(
                RaycastQuery(**ray, ordinal=OrdinalSelection.retained_exact(2))
            )
            raise AssertionError("ordinal 2 should no longer be retained")
        except ApiError as error:
            assert error.status is ApiStatus.ORDINAL_NOT_RETAINED, error
            out["dropped_version_error"] = error.status.name
    return out


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
