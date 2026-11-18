# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish prims under any transform and read the prims a publication leaves out."""

import ovstage

from ovspatialquery import Library, OvstageExclusionReason, RaycastQuery

SCENE = """#usda 1.0
def Xform "World"
{
    def Cube "Crate"
    {
        double size = 2
        double3 xformOp:translate = (4, 0, 0)
        float3 xformOp:scale = (0.5, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Sphere "Egg"
    {
        double radius = 0.5
        double3 xformOp:translate = (4, 3, 0)
        float3 xformOp:scale = (1, 2, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Cylinder "Post"
    {
        double3 xformOp:translate = (8, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Ramp"
    {
        double size = 1
        matrix4d xformOp:transform = (
            (1, 0, 0, 0), (0.5, 1, 0, 0), (0, 0, 1, 0), (7, 3, 0, 1)
        )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
"""

stage = ovstage.Stage("exclusions")
RENDERING = ovstage.PopulationDomain.RENDERING
ovstage.population.open_usd_from_string(stage, SCENE, 1, 0.0, RENDERING)
stage.advance_write_floor(ordinal=1).wait()
stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
stage.advance_write_floor(ordinal=2).wait()
paths = ovstage.PathDictionary(stage)

world = ovstage.Filter(
    [ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, ["/World/"])]
)

with Library.open(require_ovstage=True) as library, library.instance() as instance:
    population = instance.ovstage(ovstage, stage).create_population(world)

    # The publication succeeds and lists every prim it left out.
    with population.enqueue_set(2) as publication:
        publication.wait()
        result = publication.fetch_population_update()
    print("published", result.object_count, "excluded", result.excluded_object_count)
    for object_id, reason in zip(result.excluded_objects, result.exclusion_reasons):
        name = OvstageExclusionReason(reason).name
        print("  excluded", paths.path_to_string(ovstage.PrimPath(int(object_id))), name)

    # One ray along +x past the Crate, one past the Egg to the sheared Ramp.
    for y in (0.0, 3.25):
        ray = population.target.raycast_closest(
            RaycastQuery(origin=(0, y, 0), unit_direction=(1, 0, 0), max_distance=20)
        )
        path = paths.path_to_string(ovstage.PrimPath(ray.hit.object_id))
        print("ray at y", y, "hit", path, "at distance", round(ray.hit.distance, 3))

# Destroy the stage once nothing uses it, rather than at interpreter exit.
stage.destroy()
