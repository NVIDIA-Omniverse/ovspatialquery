/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_FLOAT64_BOX_ROW_SCENE_H
#define OVSPATIALQUERY_TEST_FLOAT64_BOX_ROW_SCENE_H

// The scene the CPU and CUDA float64 BOX-row tests share (ADR-0026). Four
// objects published at ordinal 7, 20 apart on Z, each present a face or side
// at x = 1: an upright box (30), a capsule of radius 1 about world Y (31), a
// convex hull of the same box (32) and a triangle-mesh quad (33). The query
// box kHalfExtents yawed 87 degrees about Z reaches 0.3|cos| + 0.25|sin| along
// X; its float32 quaternion reaches 2.2e-8 further. Rows 2k and 2k + 1 sit
// 2e-9 beyond and inside object k's face, with the same yaw as a float64 and
// a float32 rotation column. Each test brings its own plan and assertions.

#include "public_api_inputs.h"

#include <ovspatialquery/ovspatialquery_explicit.h>

#include <cmath>
#include <cstdint>

struct Float64BoxRowScene
{
    static constexpr uint32_t kObjects        = 4u;
    static constexpr uint32_t kRows           = kObjects * 2u;
    static constexpr double   kHalfExtents[3] = {0.3, 0.25, 0.1};
    static constexpr uint64_t kBox            = 30u;
    static constexpr uint64_t kCapsule        = 31u;
    static constexpr uint64_t kConvex         = 32u;
    static constexpr uint64_t kMesh           = 33u;
    // Object k is definition kValues[k] of kTypes[k]; the object id is the definition value.
    static constexpr uint64_t                       kValues[kObjects] = {kBox, kCapsule, kConvex, kMesh};
    static constexpr ovspatialquery_geometry_type_t kTypes[kObjects]  = {
        OVSPATIALQUERY_GEOMETRY_BOX, OVSPATIALQUERY_GEOMETRY_CAPSULE, OVSPATIALQUERY_GEOMETRY_CONVEX,
        OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH};

    // Registers the four definitions, publishes the population and fills the rows.
    explicit Float64BoxRowScene(ovspatialquery_instance_handle_t instanceHandle) : instance(instanceHandle)
    {
        uint64_t                                    boxValue      = kBox;
        double                                      boxExtents[3] = {1.0, 1.0, 0.05};
        InputArray                                  boxValues(&boxValue, 1, kDLUInt);
        InputArray                                  boxExtentArray(boxExtents, 1, 3, kDLFloat);
        const ovspatialquery_box_definition_table_t boxes{&boxValues.value, &boxExtentArray.value};
        waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));

        uint64_t                                        capsuleValue   = kCapsule;
        double                                          capsuleRadius  = 1.0;
        double                                          capsuleSegment = 2.0;
        InputArray                                      capsuleValues(&capsuleValue, 1, kDLUInt);
        InputArray                                      capsuleRadii(&capsuleRadius, 1, kDLFloat);
        InputArray                                      capsuleSegments(&capsuleSegment, 1, kDLFloat);
        const ovspatialquery_capsule_definition_table_t capsules{&capsuleValues.value, &capsuleRadii.value,
                                                                 &capsuleSegments.value};
        waitAndDestroy(instance, ovspatialquery_enqueue_capsule_registration(instance, &capsules));

        uint64_t convexValue        = kConvex;
        uint64_t convexVertexOffset = 0u;
        uint64_t convexVertexCount  = 8u;
        float    convexVertices[24]{};
        for (uint32_t corner = 0u; corner < 8u; ++corner)
        {
            convexVertices[corner * 3u]      = (corner & 1u) != 0u ? 1.0f : -1.0f;
            convexVertices[corner * 3u + 1u] = (corner & 2u) != 0u ? 1.0f : -1.0f;
            convexVertices[corner * 3u + 2u] = (corner & 4u) != 0u ? 0.05f : -0.05f;
        }
        InputArray                                     convexValues(&convexValue, 1, kDLUInt);
        InputArray                                     convexOffsets(&convexVertexOffset, 1, kDLUInt);
        InputArray                                     convexCounts(&convexVertexCount, 1, kDLUInt);
        InputArray                                     convexPoints(convexVertices, 8, 3, kDLFloat);
        const ovspatialquery_convex_definition_table_t convexes{
            &convexValues.value,
            &convexOffsets.value,
            &convexCounts.value,
            nullptr,
            nullptr,
            &convexPoints.value,
            nullptr,
            nullptr,
            nullptr,
            0.0,
            0u,
        };
        waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &convexes));

        uint64_t   meshValue         = kMesh;
        uint64_t   meshZero          = 0u;
        uint64_t   meshVertexCount   = 4u;
        uint64_t   meshTriangleCount = 2u;
        float      meshVertices[12]  = {1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
        uint32_t   meshTriangles[6]  = {0u, 1u, 2u, 0u, 2u, 3u};
        InputArray meshValues(&meshValue, 1, kDLUInt);
        InputArray meshZeros(&meshZero, 1, kDLUInt);
        InputArray meshVertexCounts(&meshVertexCount, 1, kDLUInt);
        InputArray meshTriangleCounts(&meshTriangleCount, 1, kDLUInt);
        InputArray meshVertexData(meshVertices, 4, 3, kDLFloat);
        InputArray meshTriangleData(meshTriangles, 2, 3, kDLUInt);
        const ovspatialquery_triangle_mesh_definition_table_t meshes{
            &meshValues.value,
            &meshZeros.value,
            &meshVertexCounts.value,
            &meshZeros.value,
            &meshTriangleCounts.value,
            &meshVertexData.value,
            &meshTriangleData.value,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
        };
        waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));

        population = createExplicitPopulation(instance, 1u);
        InstanceRows rows;
        for (uint32_t object = 0u; object < kObjects; ++object)
        {
            rows.add(kValues[object], kTypes[object], kValues[object], 0.0, 0.0, 20.0 * object);
        }
        const ovspatialquery_explicit_population_set_t set{7, rows.table(), nullptr, 0u};
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

        const double yaw   = 87.0 * 3.14159265358979323846 / 180.0;
        const double reach = kHalfExtents[0] * std::abs(std::cos(yaw)) + kHalfExtents[1] * std::abs(std::sin(yaw));
        for (uint32_t row = 0u; row < kRows; ++row)
        {
            translations[row * 3u]      = 1.0 + reach + (row % 2u == 0u ? 2.0e-9 : -2.0e-9);
            translations[row * 3u + 2u] = 20.0 * (row / 2u);
            rotations64[row * 4u + 2u]  = std::sin(0.5 * yaw);
            rotations64[row * 4u + 3u]  = std::cos(0.5 * yaw);
            rotations32[row * 4u + 2u]  = static_cast<float>(rotations64[row * 4u + 2u]);
            rotations32[row * 4u + 3u]  = static_cast<float>(rotations64[row * 4u + 3u]);
        }
    }

    // Destroys the population and the four definitions; the caller destroys
    // its plans before and the instance after.
    void destroy()
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        for (uint32_t object = 0u; object < kObjects; ++object)
        {
            REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {kTypes[object], kValues[object]}).status ==
                    OVSPATIALQUERY_API_SUCCESS);
        }
    }

    ovspatialquery_instance_handle_t   instance;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    double                             translations[kRows * 3u]{};
    double                             rotations64[kRows * 4u]{};
    float                              rotations32[kRows * 4u]{};
};

#endif
