# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the prims under /World from an ovstage stage and raycast them."""

import ovstage

from ovspatialquery import Library, RaycastQuery

SCENE = """#usda 1.0
def Xform "World"
{
    def Cube "Crate"
    {
        double size = 1
        double3 xformOp:translate = (4, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Sphere "Ball"
    {
        double radius = 0.5
        double3 xformOp:translate = (4, 3, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""

# ovstage seals each edit at an ordinal. Ordinal 1 holds the authored scene
# (read at time 0), ordinal 2 its computed world transforms.
stage = ovstage.Stage("first-query")
RENDERING = ovstage.PopulationDomain.RENDERING
ovstage.population.open_usd_from_string(stage, SCENE, 1, 0.0, RENDERING)
stage.advance_write_floor(ordinal=1).wait()
stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
stage.advance_write_floor(ordinal=2).wait()

# A population is bound to one stage and one ovstage filter:
# here, every prim under /World.
world = ovstage.Filter(
    [ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, ["/World/"])]
)

with Library.open(require_ovstage=True) as library, library.instance() as instance:
    population = instance.ovstage(ovstage, stage).create_population(world)

    # Publish sealed ordinal 2. The version keeps that number.
    with population.enqueue_set(2) as publication:
        publication.wait()
        count = publication.fetch_population_update().object_count
    print("published", count, "prims")

    ray = population.target.raycast_closest(
        RaycastQuery(origin=(0, 0, 0), unit_direction=(1, 0, 0), max_distance=20)
    )
    # An ovstage object ID is the prim's path token.
    prim_path = ovstage.PrimPath(ray.hit.object_id)
    path = ovstage.PathDictionary(stage).path_to_string(prim_path)
    print("ray hit", path, "at distance", round(ray.hit.distance, 3), end=" ")
    print("ordinal", ray.source_ordinal)

# Destroy the stage once nothing uses it, rather than at interpreter exit.
stage.destroy()
