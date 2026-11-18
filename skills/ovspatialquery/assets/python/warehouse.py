# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warehouse of natively instanced racks: 3D scan from a vehicle, check pallet placements.

The scene is generated in code: 12 rows x 10 bays, each bay an `instanceable` reference
to one rack prototype (2 uprights + 4 shelves, all Cubes), plus a floor. Z is up.
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
from ovspatialquery import Library, OrientedBox, OverlapQuery  # noqa: E402

ROWS, BAYS = 12, 10
BAY_PITCH, ROW_PITCH = 2.5, 4.0  # bay i of row j is centred at (2.5 i, 4 j, 0)
# Prototype parts in rack-local space: name -> (center, half extents). The rack spans
# x -1.2..1.2, y -0.5..0.5, z 0..3; shelves leave clear bays between z levels.
PARTS = {
    "Upright_L": ((-1.15, 0.0, 1.5), (0.05, 0.5, 1.5)),
    "Upright_R": ((1.15, 0.0, 1.5), (0.05, 0.5, 1.5)),
    **{f"Shelf_{k}": ((0.0, 0.0, z), (1.1, 0.5, 0.03)) for k, z in enumerate((0.4, 1.1, 1.8, 2.5))},
}


def cube(name: str, center, half) -> str:
    # A size-2 Cube scaled by the half extents is exactly the box center +/- half.
    return f"""
        def Cube "{name}"
        {{
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})
            float3 xformOp:scale = ({half[0]}, {half[1]}, {half[2]})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}"""


def build_scene() -> str:
    rows = []
    for j in range(ROWS):
        bays = "".join(
            f"""
            def Xform "Bay_{i:02d}" (
                instanceable = true
                references = </RackPrototype>
            )
            {{
                double3 xformOp:translate = ({i * BAY_PITCH}, {j * ROW_PITCH}, 0)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }}"""
            for i in range(BAYS)
        )
        rows.append(f'\n        def Xform "Row_{j:02d}"\n        {{{bays}\n        }}')
    parts = "".join(cube(name, c, h) for name, (c, h) in PARTS.items())
    floor = cube("Floor", (12.0, 22.0, -0.05), (20.0, 30.0, 0.05))
    return f"""#usda 1.0
(
    upAxis = "Z"
)
class Xform "RackPrototype"
{{{parts}
}}
def Xform "World"
{{{floor}
    def Xform "Racks"
    {{{"".join(rows)}
    }}
}}
"""


def world_boxes() -> list[tuple[str, np.ndarray, np.ndarray]]:
    boxes = []
    for j in range(ROWS):
        for i in range(BAYS):
            offset = np.array([i * BAY_PITCH, j * ROW_PITCH, 0.0])
            for name, (center, half) in PARTS.items():
                c = offset + center
                boxes.append((f"/World/Racks/Row_{j:02d}/Bay_{i:02d}/{name}", c - half, c + half))
    return boxes


def slab_closest(origin, directions, boxes, max_range):
    """Closest (distance, box index) per ray; independent check of the library."""
    low = np.stack([b[1] for b in boxes]) - origin  # [B,3]
    high = np.stack([b[2] for b in boxes]) - origin
    distance, index = [], []
    for chunk in np.array_split(directions, max(1, len(directions) // 256)):
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = 1.0 / chunk  # [R,3]
            t0 = low[None, :, :] * inverse[:, None, :]
            t1 = high[None, :, :] * inverse[:, None, :]
            near = np.nanmax(np.minimum(t0, t1), axis=2)
            far = np.nanmin(np.maximum(t0, t1), axis=2)
        near = np.where((near <= far) & (far >= 0) & (near <= max_range), near, np.inf)
        distance.append(near.min(axis=1))
        index.append(near.argmin(axis=1))
    return np.concatenate(distance), np.concatenate(index)


def rack_population(instance, stage, prim_types):
    # Prefix AND prim type: the prefix alone would also match the Row Xforms and the
    # Bay instance roots (left out as UNSUPPORTED_TYPE), and leaving the prefix out would
    # pull in the floor.
    racks = ovstage.Filter(
        [
            ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, ["/World/Racks/"]),
            ovstage.Predicate("usd-prim-type", ovstage.FilterOp.IN, list(prim_types)),
        ]
    )
    return instance.ovstage(ovstage, stage).create_population(racks)


def scan_directions(azimuths: int, elevations_deg) -> np.ndarray:
    """Unit directions for a 3D scan, row-major [elevation, azimuth]; yaw 0 is +X."""
    az = np.radians(np.arange(azimuths) * (360.0 / azimuths))
    el = np.radians(np.asarray(elevations_deg, dtype=np.float64))
    e, a = np.meshgrid(el, az, indexing="ij")
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1).reshape(
        -1, 3
    )


def main() -> dict:
    usda = build_scene()
    boxes = world_boxes()
    assert ovsq.gprim_paths(usda, roots=["/World/Racks"]) == sorted(path for path, _, _ in boxes)
    named_like_prototype = usda + '\ndef Cube "__Prototype_Box"\n{\n double size = 1\n}\n'
    paths = ovsq.gprim_paths(named_like_prototype)
    assert "/__Prototype_Box" in paths
    assert not any(path.startswith("/__Prototype_") and path != "/__Prototype_Box" for path in paths)
    out: dict = {}
    with Library.open(require_ovstage=True) as library, library.instance() as instance:
        with ovsq.Scene(usda, name="warehouse") as scene:
            # Every bay's parts are instance proxies, typed CubeInstance, at their own
            # paths. A filter naming only "Cube" selects authored cubes: zero here, no error.
            # The population's block ends before the stage's: it must not outlive it.
            with rack_population(instance, scene.stage, ["Cube"]) as authored:
                out["objects_cube_only"] = ovsq.publish(authored, scene.ordinal).object_count
            assert out["objects_cube_only"] == 0
            started = time.perf_counter()
            with rack_population(instance, scene.stage, ovsq.INSTANCE_TYPES) as population:
                out["objects"] = ovsq.publish(population, scene.ordinal).object_count
                out["load_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
                assert out["objects"] == len(boxes) == ROWS * BAYS * len(PARTS) == 720
                out["scans"] = vehicle_scans(instance, population, scene, boxes)
                out["placements"] = placements(population, scene)
    return out


def vehicle_scans(instance, population, scene, boxes) -> list[dict]:
    elevations = np.linspace(-20.0, 20.0, 16)
    local = scan_directions(360, elevations)  # 5760 rays, reused every pose
    max_range = 30.0
    results = []
    with ovsq.RaycastBatch(instance, population.target, max_rays=len(local)) as scan:
        # The vehicle drives down the aisle between rows 3 (y=12) and 4 (y=16), sensor
        # 1.0 m up, stopping beside an upright at x = 2.5 i + 1.15.
        for i, yaw_deg in ((2, 0.0), (4, 30.0), (6, 90.0)):
            sensor = np.array([i * BAY_PITCH + 1.15, 14.0, 1.0])
            c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
            directions = local @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            origins = np.repeat(sensor[None, :], len(directions), axis=0)
            started = time.perf_counter()
            hits = scan.run(origins, directions, max_range)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            expected, index = slab_closest(sensor, directions, boxes, max_range)
            assert np.array_equal(np.isfinite(expected), hits["hit"])
            assert np.allclose(hits["distance"][hits["hit"]], expected[hits["hit"]], atol=1e-4)
            hit_paths = {scene.path(o) for o in hits["object_id"][hits["hit"]]}
            assert hit_paths == {boxes[k][0] for k in index[hits["hit"]]}

            # Hand check: horizontal rays to +Y and -Y meet the uprights' aisle faces
            # (y = 15.5 and y = 12.5) at 1.5 m. Which azimuth that is depends on the yaw.
            plus_y = scan.run([sensor, sensor], [(0, 1, 0), (0, -1, 0)], max_range)
            assert plus_y["hit"].all() and np.allclose(plus_y["distance"], 1.5)
            assert (
                scene.path(plus_y["object_id"][0]) == f"/World/Racks/Row_04/Bay_{i:02d}/Upright_R"
            )
            results.append(
                {
                    "sensor": sensor.tolist(),
                    "yaw_deg": yaw_deg,
                    "rays": len(directions),
                    "hits": int(hits["hit"].sum()),
                    "ms": round(elapsed_ms, 3),
                }
            )
    return results


def placements(population, scene) -> list[dict]:
    """Which rack parts would a pallet-sized box touch? Touching counts as overlap."""
    bay = "/World/Racks/Row_04/Bay_04/"  # centred at (10, 16, 0); clear bay z 1.13..1.77
    candidates = [
        # name, center, half extents, yaw degrees, expected blockers
        (
            "aisle floor",
            (10.0, 14.0, 0.3),
            (0.6, 0.4, 0.3),
            0.0,
            [],
        ),  # floor is not in the population
        ("bay between shelves", (10.0, 16.0, 1.45), (0.6, 0.4, 0.3), 0.0, []),
        ("bay too low", (10.0, 16.0, 1.3), (0.6, 0.4, 0.3), 0.0, [bay + "Shelf_1"]),
        ("long box along Y", (10.0, 16.0, 1.45), (0.3, 1.2, 0.3), 0.0, []),
        (
            "long box turned 90",
            (10.0, 16.0, 1.45),
            (0.3, 1.2, 0.3),
            90.0,
            [bay + "Upright_L", bay + "Upright_R"],
        ),
    ]
    rows = []
    for name, center, half, yaw, expected in candidates:
        # Single queries default to EXACT; overlap_any would return only one blocker.
        query = OverlapQuery(
            OrientedBox(center=center, rotation_xyzw=ovsq.yaw_xyzw(yaw), half_extents=half)
        )
        started = time.perf_counter()
        result = population.target.overlap_multiple(query, capacity=16)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        assert not result.truncated
        blockers = sorted(scene.path(hit.object_id) for hit in result.hits)
        assert blockers == expected, (name, blockers)
        rows.append({"candidate": name, "blocked_by": blockers, "ms": round(elapsed_ms, 3)})
    return rows


if __name__ == "__main__":
    print(json.dumps(main()))
