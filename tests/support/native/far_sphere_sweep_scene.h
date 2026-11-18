/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_FAR_SPHERE_SWEEP_SCENE_H
#define OVSPATIALQUERY_TEST_FAR_SPHERE_SWEEP_SCENE_H

// The scene the CPU and CUDA far sphere-sweep tests share: spheres of radius
// 0.75 that meet a face face-on more than a thousand units from their start,
// where the rebuilt center rounds by ulps of the path and the coordinates, not
// of the radius.
//
// - Mesh rows: one single-sided triangle (-2000, 1000, -2000), (4000, 1000,
//   -2000), (-2000, 1000, 4000) in y = 1000, facing -y. Sweeps start at the
//   origin along the float32 normalize(a, b, c), a and c in -2 to 2 and b in
//   3 to 7, up to 4000. The center reaches y = 999.25 at t = 999.25 / d_y, at
//   least 600 units inside every edge.
// - Cube rows: a convex of the corners of a cube of half extent 1000 at the
//   origin. Sweeps start at (0, 0, -3000) along the float32 normalize(a, b,
//   10), a and b in 1 to 5, up to 10000. The center reaches z = -1000.75 at
//   t = 1999.25 / d_z, inside the face z = -1000 (no more than 999.7 off its
//   axis, so 0.94 from the nearest edge).
//
// - Grazing rows: spheres starting 1e-10 beyond their radius below the
//   triangle's face, moving parallel to it, so they never touch it. The gap
//   is within the slack a far contact's rounding needs but is no rounding at
//   the start, so they must miss.
//
// Each expected distance is that plane contact along the float32 direction
// the library reads. Each test brings its own queries and assertions.

#include "public_api_inputs.h"

#include <ovspatialquery/ovspatialquery_explicit.h>

#include <cmath>
#include <cstdint>
#include <vector>

struct FarSphereSweepScene
{
    static constexpr double   kRadius = 0.75;
    static constexpr uint64_t kMesh   = 73u;
    static constexpr uint64_t kCube   = 74u;

    struct Rows
    {
        std::vector<double> origins;    // [Q, 3]
        std::vector<float>  directions; // [Q, 3]
        std::vector<double> maximumDistances;
        std::vector<double> expected;

        // Appends a row along the float32 normalize(direction) whose contact is
        // where coordinate `axis` of the center reaches `plane`.
        void add(const double origin[3], const double direction[3], double maximum, unsigned int axis, double plane)
        {
            const double length =
                std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
            float narrowed[3]{};
            for (unsigned int k = 0u; k < 3u; ++k)
            {
                narrowed[k] = float(direction[k] / length);
                origins.push_back(origin[k]);
                directions.push_back(narrowed[k]);
            }
            maximumDistances.push_back(maximum);
            expected.push_back(double(((long double)plane - origin[axis]) / (long double)narrowed[axis]));
        }
    };

    // Registers the triangle and the cube, publishes one population for each
    // and fills the rows.
    explicit FarSphereSweepScene(ovspatialquery_instance_handle_t instanceHandle) : instance(instanceHandle)
    {
        uint64_t   meshValue = kMesh, zero = 0u, three = 3u, one = 1u;
        float      corners[9]  = {-2000.f, 1000.f, -2000.f, 4000.f, 1000.f, -2000.f, -2000.f, 1000.f, 4000.f};
        uint32_t   triangle[3] = {0u, 1u, 2u};
        InputArray meshValues(&meshValue, 1, kDLUInt), vertexOffsets(&zero, 1, kDLUInt),
            vertexCounts(&three, 1, kDLUInt), triangleOffsets(&zero, 1, kDLUInt), triangleCounts(&one, 1, kDLUInt),
            vertexRows(corners, 3, 3, kDLFloat), triangleRows(triangle, 1, 3, kDLUInt);
        const ovspatialquery_triangle_mesh_definition_table_t meshes{&meshValues.value,
                                                                     &vertexOffsets.value,
                                                                     &vertexCounts.value,
                                                                     &triangleOffsets.value,
                                                                     &triangleCounts.value,
                                                                     &vertexRows.value,
                                                                     &triangleRows.value,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr};
        waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));

        uint64_t cubeValue = kCube, eight = 8u;
        float    cube[24]{};
        for (unsigned int corner = 0u; corner < 8u; ++corner)
        {
            for (unsigned int axis = 0u; axis < 3u; ++axis)
            {
                cube[corner * 3u + axis] = (corner >> axis) % 2u != 0u ? 1000.f : -1000.f;
            }
        }
        InputArray cubeValues(&cubeValue, 1, kDLUInt), cubeOffsets(&zero, 1, kDLUInt), cubeCounts(&eight, 1, kDLUInt),
            cubePoints(cube, 8, 3, kDLFloat);
        const ovspatialquery_convex_definition_table_t hulls{
            &cubeValues.value,
            &cubeOffsets.value,
            &cubeCounts.value,
            nullptr,
            nullptr,
            &cubePoints.value,
            nullptr,
            nullptr,
            nullptr,
            0.0,
            0u,
        };
        waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &hulls));

        meshPopulation = publish(OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMesh);
        cubePopulation = publish(OVSPATIALQUERY_GEOMETRY_CONVEX, kCube);

        const double origin[3] = {0.0, 0.0, 0.0};
        for (int a = -2; a <= 2; ++a)
        {
            for (int b = 3; b <= 7; ++b)
            {
                for (int c = -2; c <= 2; ++c)
                {
                    const double direction[3] = {double(a), double(b), double(c)};
                    meshRows.add(origin, direction, 4000.0, 1u, 1000.0 - kRadius);
                }
            }
        }
        for (int a = -2; a <= 2; ++a)
        {
            grazingRows.origins.insert(grazingRows.origins.end(), {500.0 * a, 1000.0 - kRadius - 1.0e-10, 0.0});
            grazingRows.directions.insert(grazingRows.directions.end(), {1.f, 0.f, 0.f});
            grazingRows.maximumDistances.push_back(500.0);
        }
        const double start[3] = {0.0, 0.0, -3000.0};
        for (int a = 1; a <= 5; ++a)
        {
            for (int b = 1; b <= 5; ++b)
            {
                const double direction[3] = {double(a), double(b), 10.0};
                cubeRows.add(start, direction, 10000.0, 2u, -1000.0 - kRadius);
            }
        }
    }

    // Destroys the populations and definitions; the caller destroys its plans
    // before and the instance after.
    void destroy()
    {
        REQUIRE(ovspatialquery_destroy_population(instance, meshPopulation).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_population(instance, cubePopulation).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMesh})
                    .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, kCube}).status ==
                OVSPATIALQUERY_API_SUCCESS);
    }

    ovspatialquery_instance_handle_t   instance;
    ovspatialquery_population_handle_t meshPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_population_handle_t cubePopulation = OVSPATIALQUERY_INVALID_POPULATION;
    Rows                               meshRows;
    Rows                               cubeRows;
    Rows                               grazingRows; // against meshPopulation; no expected contact

  private:
    ovspatialquery_population_handle_t publish(int32_t type, uint64_t definition)
    {
        const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
        InstanceRows                             rows;
        rows.add(definition, type, definition, 0.0, 0.0, 0.0);
        const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
        return population;
    }
};

#endif
