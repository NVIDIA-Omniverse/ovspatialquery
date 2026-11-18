/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// An Ovstage convexHull Mesh, on the library's PhysX backend, answers for the hull a PhysX
// simulation of the stage collides with (ADR-0078): the hull PhysX cooks from the Mesh's points
// as Omni Physics does, every authored point in its order at its own coordinates, duplicates and
// points no face references included.

#include "../../support/native/ovstage_companion_fixtures.h"
#include "../../support/native/physx_convex_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

using physx_convex_oracle::Vector;

int testConvexHullMeshesAnswerForTheSimulatedHull()
{
    // A 128-gon prism of radius 0.1 and height 0.1 about (5, 2, 1), whose faces are its two caps,
    // then three of its points again and an apex 0.15 above it that no face references. About
    // (5, 2, 1) PhysX's hull of these points differs from its hull of the same points moved to
    // the origin, under the default limit 64 and under 16.
    std::vector<float> points;
    for (const double z : {0.95, 1.05})
        for (int side = 0; side < 128; ++side)
            points.insert(points.end(), {static_cast<float>(5.0 + 0.1 * std::cos(6.283185307179586 * side / 128.0)),
                                         static_cast<float>(2.0 + 0.1 * std::sin(6.283185307179586 * side / 128.0)),
                                         static_cast<float>(z)});
    for (const size_t again : {0u, 64u, 200u})
        points.insert(points.end(), {points[3u * again], points[3u * again + 1u], points[3u * again + 2u]});
    points.insert(points.end(), {5.0f, 2.0f, 1.2f});
    const size_t count = points.size() / 3u;

    std::string text, caps;
    char        number[96];
    for (size_t point = 0u; point < count; ++point)
    {
        std::snprintf(number, sizeof(number), "%s(%.9g, %.9g, %.9g)", point == 0u ? "" : ", ", points[3u * point],
                      points[3u * point + 1u], points[3u * point + 2u]);
        text += number;
    }
    for (int corner = 0; corner < 256; ++corner)
    {
        std::snprintf(number, sizeof(number), "%s%d", corner == 0 ? "" : ", ", corner < 128 ? 127 - corner : corner);
        caps += number;
    }
    struct Prim
    {
        const char* name;
        uint32_t    limit;
        double      x;
    };
    const Prim  prims[] = {{"DefaultLimit", 64u, 0.0}, {"SixteenLimit", 16u, 10.0}};
    std::string usda    = "#usda 1.0\ndef Xform \"World\"\n{\n";
    for (const Prim& prim : prims)
    {
        usda += std::string("    def Mesh \"") + prim.name +
                "\" (\n        prepend apiSchemas = [\"PhysicsCollisionAPI\", \"PhysicsMeshCollisionAPI\", "
                "\"PhysxConvexHullCollisionAPI\"]\n    )\n    {\n"
                "        uniform token physics:approximation = \"convexHull\"\n";
        if (prim.limit != 64u)
            usda += "        int physxConvexHullCollision:hullVertexLimit = " + std::to_string(prim.limit) + "\n";
        usda += "        int[] faceVertexCounts = [128, 128]\n        int[] faceVertexIndices = [" + caps +
                "]\n        point3f[] points = [" + text +
                "]\n        float3[] extent = [(4.9, 1.9, 0.95), (5.1, 2.1, 1.2)]\n"
                "        double3 xformOp:translate = (" +
                std::to_string(prim.x) +
                ", 0, 0)\n        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n";
    }
    usda += "}\n";

    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-physx-convex-hull-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK && stage != nullptr);
    const ovstage_population_enqueue_result_t opened =
        openUsdWithPhysics(stage, ovx_string_t{usda.data(), usda.size()}, 1u, 0.0);
    ovstage_population_op_wait_result_t openedWait{};
    CHECK(opened.status == OVSTAGE_OK &&
          ovstage_population_wait_op(stage, opened.op_index, OVSTAGE_TIMEOUT_INFINITE, &openedWait) == OVSTAGE_OK);
    const ovstage_write_floor_desc_t floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorOne)));
    CHECK(waitStageOperation(
        stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u)));
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorTwo)));

    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));
    const ovx_string_t  meshType[] = {{"Mesh", 4u}};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = ovx_string_t{"usd-prim-type", sizeof("usd-prim-type") - 1u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = meshType;
    predicate.value_count      = 1u;
    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                            = stage;
    populationDesc.filter                           = ovstage_filter_t{&predicate, 1u};
    populationDesc.common.retained_ordinal_capacity = 2u;
    ovspatialquery_population_handle_t population   = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2), 2,
                                 2u, 0u));

    uint64_t   state   = UINT64_C(0x9E3779B97F4A7C15);
    const auto uniform = [&state](double low, double high)
    {
        state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return low + (high - low) * static_cast<double>(state >> 11u) * (1.0 / 9007199254740992.0);
    };
    const Vector centre{5.0, 2.0, 1.075};
    for (const Prim& prim : prims)
    {
        const std::vector<Vector> corners = physx_convex_oracle::cookAsOmniPhysics(points.data(), count, prim.limit);
        // The simulated hull reaches the apex, 0.15 above the caps, so a library that left out
        // points no face references would answer for another hull.
        CHECK(std::any_of(corners.begin(), corners.end(), [](const Vector& corner) { return corner[2] > 1.19; }));
        const std::vector<physx_convex_oracle::Face> faces = physx_convex_oracle::hullFaces(corners, 0.125);
        ovspatialquery_raycast_desc_t                ray{};
        ray.common.target    = populationTarget(population);
        ray.common.precision = OVSPATIALQUERY_QUERY_EXACT;
        ray.ray.max_distance = 0.75;
        size_t hits          = 0u;
        for (int index = 0; index < 256; ++index)
        {
            const Vector from = physx_convex_oracle::unit({uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)});
            const Vector origin{centre[0] + 0.375 * from[0], centre[1] + 0.375 * from[1], centre[2] + 0.375 * from[2]};
            const Vector aim{centre[0] + uniform(-0.1, 0.1), centre[1] + uniform(-0.1, 0.1),
                             centre[2] + uniform(-0.125, 0.125)};
            Vector direction = physx_convex_oracle::unit({aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]});
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                ray.ray.origin[axis]         = origin[axis] + (axis == 0u ? prim.x : 0.0);
                ray.ray.unit_direction[axis] = static_cast<float>(direction[axis]);
                direction[axis]              = ray.ray.unit_direction[axis];
            }
            double     entry = 0.0, exit = 0.0;
            const bool expectedHit =
                physx_convex_oracle::clipRay(faces, origin, direction, ray.ray.max_distance, entry, exit);
            if (expectedHit && exit - entry < 1e-6)
                continue; // grazing
            ovspatialquery_raycast_single_result_t result{};
            CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &result)));
            CHECK((result.has_hit != 0u) == expectedHit);
            CHECK(!expectedHit || std::abs(result.hit.data.exact.distance - entry) <= 1e-6);
            hits += expectedHit ? 1u : 0u;
        }
        CHECK(hits != 0u);
    }
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

} // namespace

int main() { return testConvexHullMeshesAnswerForTheSimulatedHull(); }
