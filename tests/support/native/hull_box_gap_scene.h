/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_HULL_BOX_GAP_SCENE_H
#define OVSPATIALQUERY_TEST_HULL_BOX_GAP_SCENE_H

// The scene the CPU and CUDA hull box-gap tests share. A convex hull (object
// and definition kHull) at the origin is a square prism turned 45 degrees: its
// vertical edges stand at distance kReach along +-X and +-Y, its caps at
// z = +-0.5, so near those edges it is narrower than its AABB. Each row's query
// box, of half extent 0.5, is yawed by theta (below 45 degrees) about Z, so its
// -X face has outward normal m = (-cos, -sin, 0) and tangent t = (-sin, cos, 0).
// The box is placed so that the hull's edge at (kReach, 0), the hull's
// farthest point along -m, lies `gap` outside that face at offset tau along t:
// c = (kReach, 0, 0) - (0.5 + gap) m - tau t. The exact separation is then
// `gap`, so rows 1e-3 and 1e-4 apart overlap nothing and rows 1e-4 deep overlap
// the hull. The float32 and float64 rotation columns hold the same values.
// Each test brings its own plan and assertions.

#include "public_api_inputs.h"

#include <ovspatialquery/ovspatialquery_explicit.h>

#include <cmath>
#include <cstdint>

struct HullBoxGapScene
{
    static constexpr uint32_t kYaws       = 8u;
    static constexpr uint32_t kRows       = kYaws * 2u * 3u;
    static constexpr double   kHalfExtent = 0.5;
    static constexpr uint64_t kHull       = 40u;
    static constexpr float    kReach      = 0.70710677f; // 0.5 sqrt(2), rounded to float

    // Registers the hull, publishes the population and fills the rows.
    explicit HullBoxGapScene(ovspatialquery_instance_handle_t instanceHandle) : instance(instanceHandle)
    {
        uint64_t    hullValue   = kHull;
        uint64_t    offset      = 0u;
        uint64_t    vertexCount = 8u;
        const float edges[4][2]{{kReach, 0.0f}, {0.0f, kReach}, {-kReach, 0.0f}, {0.0f, -kReach}};
        float       vertices[24]{};
        for (uint32_t corner = 0u; corner < 8u; ++corner)
        {
            vertices[corner * 3u]      = edges[corner / 2u][0];
            vertices[corner * 3u + 1u] = edges[corner / 2u][1];
            vertices[corner * 3u + 2u] = corner % 2u != 0u ? 0.5f : -0.5f;
        }
        InputArray                                     values(&hullValue, 1, kDLUInt);
        InputArray                                     offsets(&offset, 1, kDLUInt);
        InputArray                                     counts(&vertexCount, 1, kDLUInt);
        InputArray                                     points(vertices, 8, 3, kDLFloat);
        const ovspatialquery_convex_definition_table_t hulls{
            &values.value, &offsets.value, &counts.value, nullptr, nullptr, &points.value,
            nullptr,       nullptr,        nullptr,       0.0,     0u,
        };
        waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &hulls));
        population = createExplicitPopulation(instance, 1u);
        InstanceRows rows;
        rows.add(kHull, OVSPATIALQUERY_GEOMETRY_CONVEX, kHull, 0.0, 0.0, 0.0);
        const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

        static constexpr double kTaus[2]{-0.3, 0.3};
        static constexpr double kGaps[3]{1.0e-3, 1.0e-4, -1.0e-4};
        uint32_t                row = 0u;
        for (uint32_t yaw = 0u; yaw < kYaws; ++yaw)
        {
            const double theta = 0.1 + 0.075 * yaw;
            const float  z = static_cast<float>(std::sin(0.5 * theta)), w = static_cast<float>(std::cos(0.5 * theta));
            // The library normalizes the quaternion it reads.
            const double norm   = double(z) * z + double(w) * w;
            const double cosine = (double(w) * w - double(z) * z) / norm, sine = 2.0 * double(z) * w / norm;
            for (const double tau : kTaus)
            {
                for (const double gap : kGaps)
                {
                    translations[row * 3u]      = double(kReach) + (0.5 + gap) * cosine + tau * sine;
                    translations[row * 3u + 1u] = (0.5 + gap) * sine - tau * cosine;
                    rotations32[row * 4u + 2u]  = z;
                    rotations32[row * 4u + 3u]  = w;
                    rotations64[row * 4u + 2u]  = z;
                    rotations64[row * 4u + 3u]  = w;
                    overlaps[row]               = gap < 0.0 ? 1u : 0u;
                    ++row;
                }
            }
        }
    }

    // Destroys the population and the hull; the caller destroys its plans
    // before and the instance after.
    void destroy()
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, kHull}).status ==
                OVSPATIALQUERY_API_SUCCESS);
    }

    ovspatialquery_instance_handle_t   instance;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    double                             translations[kRows * 3u]{};
    double                             rotations64[kRows * 4u]{};
    float                              rotations32[kRows * 4u]{};
    uint64_t                           overlaps[kRows]{}; // 1 when the row's box overlaps the hull
};

#endif
