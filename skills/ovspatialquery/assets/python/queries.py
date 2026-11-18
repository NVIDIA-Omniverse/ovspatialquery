# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single synchronous queries on one Ovstage population.

Raycast closest/any/multiple (exact and bounds), sphere sweep, oriented box sweep,
overlap any/multiple with Sphere, OrientedBox and Aabb, a triangle-mesh face hit, two
ways to query a subset of one population (a view and participation bits), the prims
a publication leaves out, and path scopes ("everything except" a subtree).
Prints one JSON object; every answer is asserted against a hand calculation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import ovsq  # noqa: E402
import ovstage  # noqa: E402
from ovspatialquery import (  # noqa: E402
    Aabb,
    BoxSweepQuery,
    GeometryType,
    HitTruth,
    Library,
    OrientedBox,
    OverlapQuery,
    QueryPrecision,
    RaycastQuery,
    Sphere,
    SphereSweepQuery,
)

BOUNDS = QueryPrecision.BOUNDS_CANDIDATE  # single queries default to EXACT

# Crate: box x in [4, 6], y and z in [-1, 1].  Ball: sphere r=1 at (0, 5, 0).
# Panel: mesh in the plane x = -4, front face toward +x; face 0 is a triangle,
# face 1 a quad over y, z in [-1, 1]. doubleSided = 0 (the USD default) means rays
# arriving from behind (-x side) pass through it. subdivisionScheme is unauthored (USD
# falls back to catmullClark): the mesh is still queried as its authored faces.
# Floor is not selected.
USDA = """#usda 1.0
(
    upAxis = "Z"
)
def Xform "World"
{
    def Xform "Props"
    {
        def Cube "Crate"
        {
            double size = 2
            double3 xformOp:translate = (5, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Sphere "Ball"
        {
            double radius = 1
            double3 xformOp:translate = (0, 5, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Mesh "Panel"
        {
            uniform bool doubleSided = 0
            int[] faceVertexCounts = [3, 4]
            int[] faceVertexIndices = [4, 5, 6, 0, 1, 2, 3]
            point3f[] points = [(0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1), (0, -3, -1), (0, -2, -1), (0, -2, 1)]
            float3[] extent = [(0, -3, -1), (0, 1, 1)]
            double3 xformOp:translate = (-4, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def Cube "Floor"
    {
        double3 xformOp:translate = (0, 0, -3)
        double3 xformOp:scale = (40, 40, 0.2)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
}
"""
CRATE, BALL, PANEL = "/World/Props/Crate", "/World/Props/Ball", "/World/Props/Panel"


def close(a, b, tol=1e-5) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def paths_filter(paths: list[str]) -> ovstage.Filter:
    return ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])


def main() -> dict:
    paths = ovsq.gprim_paths(USDA, roots=["/World/Props"])  # Ball, Crate, Panel; not Floor
    out: dict = {}
    # Scene outermost: the instance closes the population and view before the stage dies.
    with (
        ovsq.Scene(USDA) as scene,
        Library.open(require_ovstage=True) as library,
        library.instance() as instance,
    ):
        population = instance.ovstage(ovstage, scene.stage).create_population(paths_filter(paths))
        # Both must exist before the publication that should include them.
        view = population.create_view(paths_filter([CRATE, BALL]))
        population.set_participation_map(default_bits=0b01, bits_by_path={PANEL: 1 << 100})
        update = ovsq.publish(population, scene.ordinal)
        assert update.object_count == 3, update
        target = population.target
        name = scene.path

        # 1. Closest raycast (exact by default): box face, outward normal, geometry type.
        r = target.raycast_closest(RaycastQuery((0, 0, 0), (1, 0, 0), 20.0))
        h = r.hit
        assert r.source_ordinal == scene.ordinal and name(h.object_id) == CRATE
        assert h.truth == HitTruth.EXACT_SURFACE and h.geometry_type == GeometryType.BOX
        assert (
            close([h.distance], [4.0])
            and close(h.position, (4, 0, 0))
            and close(h.normal, (-1, 0, 0))
        )
        out["crate_ray"] = {"path": name(h.object_id), "distance": h.distance}

        # 2. Exact vs bounds on a sphere, ray offset 0.6 from the centre:
        #    exact y = 5 - sqrt(1 - 0.6^2) = 4.2; bounds stops at the AABB face y = 4.
        ray = dict(origin=(0.6, 0, 0), unit_direction=(0, 1, 0), max_distance=20.0)
        exact = target.raycast_closest(RaycastQuery(**ray)).hit
        bounds = target.raycast_closest(RaycastQuery(**ray, precision=BOUNDS)).hit
        assert close([exact.distance, bounds.distance], [4.2, 4.0])
        assert close(exact.normal, (0.6, -0.8, 0)) and bounds.truth == HitTruth.BOUNDS_CANDIDATE
        out["ball_exact_vs_bounds"] = [exact.distance, bounds.distance]

        # 3. Any hit: a yes/no visibility test. max_distance shorter than the hit -> None.
        assert target.raycast_any(RaycastQuery((0, 0, 0), (1, 0, 0), 20.0)).hit is not None
        assert target.raycast_any(RaycastQuery((0, 0, 0), (1, 0, 0), 3.9)).hit is None

        # 4. Triangle mesh: face index is the authored USD face; corner indices are corners
        #    of that face (a quad is fanned (0,1,2), (0,2,3)); barycentrics weight them.
        h = target.raycast_closest(RaycastQuery((0, 0.5, -0.25), (-1, 0, 0), 20.0)).hit
        assert name(h.object_id) == PANEL and h.geometry_type == GeometryType.TRIANGLE_MESH
        assert h.source_face_index == 1 and h.source_face_corner_indices == (0, 1, 2)
        assert close(h.barycentrics, (0.25, 0.375, 0.375)) and close(h.normal, (1, 0, 0))
        h = target.raycast_closest(RaycastQuery((0, -2.2, 0), (-1, 0, 0), 20.0)).hit
        assert h.source_face_index == 0 and close(h.barycentrics, (0.2, 0.3, 0.5))
        out["panel_face"] = {"face": 1, "barycentrics": [0.25, 0.375, 0.375]}

        # 5. Single-sided mesh seen from behind: exact passes through, bounds does not.
        ray = dict(origin=(-6, 0.5, -0.25), unit_direction=(1, 0, 0), max_distance=20.0)
        assert name(target.raycast_closest(RaycastQuery(**ray)).hit.object_id) == CRATE
        bounds = target.raycast_closest(RaycastQuery(**ray, precision=BOUNDS)).hit
        assert name(bounds.object_id) == PANEL

        # 6. Every hit along a ray (unspecified order): one per box and one per mesh
        #    triangle crossed; the single-sided Panel is crossed once, through its front.
        #    Crate entry at 2, Panel front at 12. `truncated` says capacity was too small.
        many = target.raycast_multiple(RaycastQuery((8, 0.5, -0.25), (-1, 0, 0), 20.0), 16)
        assert not many.truncated and many.source_ordinal == scene.ordinal
        rows = sorted((round(h.distance, 6), name(h.object_id)) for h in many.hits)
        assert rows == [(2.0, CRATE), (12.0, PANEL)], rows
        out["multiple"] = [list(row) for row in rows]

        # 7. Sphere sweep r=0.5 from (1.2, 0, 0) along +y toward the Ball:
        #    centres 1.5 apart, lateral 1.2 -> y = 5 - 0.9 = 4.1. Bounds: AABBs touch at 3.5.
        #    hit.position is the contact ON THE OBJECT; hit.query_position is the swept
        #    sphere's centre at contact (center + distance * direction).
        sweep = dict(center=(1.2, 0, 0), radius=0.5, unit_direction=(0, 1, 0), max_distance=10.0)
        h = target.sweep_sphere_closest(SphereSweepQuery(**sweep)).hit
        assert name(h.object_id) == BALL and close([h.distance], [4.1])
        assert close(h.position, (0.8, 4.4, 0)) and close(h.normal, (0.8, -0.6, 0))
        assert close(h.query_position, (1.2, 4.1, 0))
        b = target.sweep_sphere_closest(SphereSweepQuery(**sweep, precision=BOUNDS)).hit
        assert close([b.distance], [3.5]) and close(b.query_position, (1.2, 3.5, 0))
        out["sphere_sweep"] = {"path": name(h.object_id), "distance": h.distance}

        # 8. Box sweep: cube of half extent 0.5 yawed 45 deg (corner leads, reach 0.5*sqrt2)
        #    along +x into the Crate face x = 4 -> distance 4 - 0.7071 = 3.2929.
        h = target.sweep_box_closest(
            BoxSweepQuery(
                center=(0, 0, 0),
                half_extents=(0.5, 0.5, 0.5),
                rotation_xyzw=ovsq.yaw_xyzw(45),
                unit_direction=(1, 0, 0),
                max_distance=10.0,
            )
        ).hit
        assert name(h.object_id) == CRATE and close([h.distance], [4 - 0.5 * math.sqrt(2)])
        out["box_sweep"] = {"path": name(h.object_id), "distance": h.distance}

        # 9. Overlap any: sphere r=0.2 at (0.9, 4.1, 0) is 0.073 outside the Ball but inside
        #    its AABB, so exact says none and bounds says Ball.
        probe = Sphere((0.9, 4.1, 0), 0.2)
        assert target.overlap_any(OverlapQuery(probe)).hit is None
        assert name(target.overlap_any(OverlapQuery(probe, precision=BOUNDS)).hit.object_id) == BALL

        # 10. Overlap multiple with an Aabb and an OrientedBox (yaw 90: the long axis turns
        #     from x to y and reaches the Ball; unrotated it would hit nothing).
        def overlapping(target, query) -> list[str]:
            result = target.overlap_multiple(query, 64)
            assert not result.truncated
            return sorted(name(h.object_id) for h in result.hits)

        assert overlapping(target, OverlapQuery(Aabb((-5, -1, -1), (6, 1, 1)))) == [CRATE, PANEL]
        rotated = OrientedBox(
            center=(0, 3, 0), rotation_xyzw=ovsq.yaw_xyzw(90), half_extents=(1.5, 0.2, 0.2)
        )
        assert overlapping(target, OverlapQuery(rotated)) == [BALL]
        out["overlap_aabb"] = [CRATE, PANEL]

        # 11. Subset by view (Crate, Ball): the Panel is invisible to view.target.
        everything = Aabb((-10, -10, -2), (10, 10, 2))
        assert overlapping(view.target, OverlapQuery(everything)) == [BALL, CRATE]
        panel_ray = RaycastQuery((0, 0.5, -0.25), (-1, 0, 0), 20.0)
        assert view.target.raycast_closest(panel_ray).hit is None

        # 12. Subset by participation bits: query_bits=0 accepts all; otherwise an object
        #     matches when (query_bits & object_bits) != 0. Panel has bit 100 (bits are 128
        #     wide), the rest bit 0.
        def panel_ray_hit(bits):
            hit = target.raycast_closest(
                RaycastQuery((0, 0.5, -0.25), (-1, 0, 0), 20.0, query_bits=bits)
            ).hit
            return None if hit is None else name(hit.object_id)

        hits = (panel_ray_hit(0), panel_ray_hit(1 << 100), panel_ray_hit(0b01))
        assert hits == (PANEL, PANEL, None), hits
        assert overlapping(target, OverlapQuery(everything, query_bits=0b01)) == [BALL, CRATE]
        out["participation"] = {"bits0": PANEL, "bit100": PANEL, "bit0": None}

        # 13. Content never fails a publication. A bare prefix also matches the Props Xform,
        #     which is left out and listed; ovsq.publish raises naming it unless allowed.
        loose = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, ["/World/Props"])]
        )
        props = instance.ovstage(ovstage, scene.stage).create_population(loose)
        try:
            ovsq.publish(props, scene.ordinal)
            raise AssertionError("an excluded Xform must raise")
        except RuntimeError as error:
            assert "/World/Props UNSUPPORTED_TYPE" in str(error), error
        republish = instance.ovstage(ovstage, scene.stage).create_population(loose)
        update = ovsq.publish(republish, scene.ordinal, allow_exclusions=True)
        assert update.object_count == 3 and update.excluded_object_count == 1, update
        out["excluded"] = ovsq.exclusions(republish, update)
        assert out["excluded"] == [("/World/Props", "UNSUPPORTED_TYPE")], out["excluded"]

        # 14. Path scopes say "everything except", which a filter cannot. Walking up from a
        #     prim, the nearest path in either list decides (an exclude wins a tie), by path
        #     element. The empty filter selects every gprim; the scope drops the Floor, which
        #     is then not selected at all, so it is not an exclusion. The view excludes Props
        #     and re-includes the Crate below it. Create both before the publication.
        scoped = instance.ovstage(ovstage, scene.stage).create_population(
            ovstage.Filter([]), exclude_paths=["/World/Floor"]
        )
        crate_only = scoped.create_view(
            ovstage.Filter([]), include_paths=[CRATE], exclude_paths=["/World/Props"]
        )
        update = ovsq.publish(scoped, scene.ordinal)
        assert update.object_count == 3 and update.excluded_object_count == 0, update
        floor_too = Aabb((-30, -30, -4), (30, 30, 4))
        assert overlapping(scoped.target, OverlapQuery(floor_too)) == [BALL, CRATE, PANEL]
        assert overlapping(crate_only.target, OverlapQuery(floor_too)) == [CRATE]
        out["scoped"] = overlapping(crate_only.target, OverlapQuery(floor_too))
    return out


if __name__ == "__main__":
    print(json.dumps(main()))
