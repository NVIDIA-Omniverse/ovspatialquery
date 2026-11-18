# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planar range scan: one reusable raycast plan (ovsq.RaycastBatch), one launch per frame.

The sensor slides along -Z past two cubes and sweeps a 161-ray fan in the XZ plane
(the scene is Y-up). Every ray is checked against an independent slab test.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import ovsq  # noqa: E402
import ovstage  # noqa: E402
from ovspatialquery import Library  # noqa: E402

SCENE = """#usda 1.0
(
    upAxis = "Y"
)
def Xform "World"
{
    def Xform "Targets"
    {
        def Cube "Hit"
        {
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Cube "OffRay"
        {
            double size = 1.5
            float3[] extent = [(-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)]
            double3 xformOp:translate = (3, 0, -3)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
}
"""
# World boxes (center, half size) of the cubes above, for the independent check.
BOXES = {
    "/World/Targets/Hit": ((1.0, 0.0, 0.0), 1.0),
    "/World/Targets/OffRay": ((3.0, 0.0, -3.0), 0.75),
}
ANGLES = np.radians(np.arange(-40.0, 40.0 + 0.25, 0.5))  # 161 rays about +X
MAX_RANGE = 12.0
SENSOR_Z = (0.0, -1.5, -3.0)  # one frame per pose


def slab_closest(origin, directions, max_range):
    """Closest (distance, path) per ray against BOXES; (inf, None) on a miss."""
    best = np.full(len(directions), np.inf)
    names = np.full(len(directions), None, dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / directions
        for path, (center, half) in BOXES.items():
            low = (np.asarray(center) - half - origin) * inverse
            high = (np.asarray(center) + half - origin) * inverse
            near = np.nanmax(np.minimum(low, high), axis=1)
            far = np.nanmin(np.maximum(low, high), axis=1)
            hit = (near <= far) & (far >= 0.0) & (near <= max_range) & (near < best)
            best[hit] = np.maximum(near[hit], 0.0)
            names[hit] = path
    return best, names


def main() -> dict:
    paths = ovsq.gprim_paths(SCENE, roots=["/World/Targets"])
    selected = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])
    directions = np.stack([np.cos(ANGLES), np.zeros_like(ANGLES), np.sin(ANGLES)], axis=1)
    frames = []
    # The stage outlives the instance, so the instance closes the population and plan first.
    with (
        ovsq.Scene(SCENE) as scene,
        Library.open(require_ovstage=True) as library,
        library.instance() as instance,
    ):
        population = instance.ovstage(ovstage, scene.stage).create_population(selected)
        assert ovsq.publish(population, scene.ordinal).object_count == len(paths) == 2
        # Create the plan once; it is sized for the largest frame and reused.
        with ovsq.RaycastBatch(instance, population.target, max_rays=len(ANGLES)) as scan:
            for z in SENSOR_Z:
                origin = np.array([-4.0, 0.0, z])
                origins = np.repeat(origin[None, :], len(ANGLES), axis=0)
                started = time.perf_counter()
                result = scan.run(origins, directions, MAX_RANGE)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                hit_paths = [
                    scene.path(i) if h else None for h, i in zip(result["hit"], result["object_id"])
                ]

                expected, expected_paths = slab_closest(origin, directions, MAX_RANGE)
                assert list(expected_paths) == hit_paths, (z, hit_paths)
                assert np.allclose(
                    result["distance"][result["hit"]], expected[result["hit"]], atol=1e-5
                )
                center = len(ANGLES) // 2  # the ray straight along +X
                frames.append(
                    {
                        "sensor": origin.tolist(),
                        "hits": int(result["hit"].sum()),
                        "center_ray": [hit_paths[center], float(result["distance"][center])],
                        "ms": round(elapsed_ms, 3),
                    }
                )

    # Hand-computed: +X from (-4,0,0) meets Hit's face x=0 at 4; from (-4,0,-3) it misses
    # Hit (z in [-1,1]) and meets OffRay's face x=2.25 at 6.25; from z=-1.5 it misses both.
    assert frames[0]["center_ray"] == ["/World/Targets/Hit", 4.0]
    assert frames[1]["center_ray"][0] is None and math.isinf(frames[1]["center_ray"][1])
    assert frames[2]["center_ray"] == ["/World/Targets/OffRay", 6.25]
    for frame in frames:
        frame["center_ray"][1] = (
            None if math.isinf(frame["center_ray"][1]) else frame["center_ray"][1]
        )
    return {"rays_per_frame": len(ANGLES), "frames": frames}


if __name__ == "__main__":
    print(json.dumps(main()))
