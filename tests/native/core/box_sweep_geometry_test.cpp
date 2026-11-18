/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/portable_convex_hull.h"
#include "../../support/native/test_failure.h"
#include "ovspatialquery_box_sweep_geometry.h"
#include "ovspatialquery_box_sweep_math.h"
#include "ovspatialquery_instance_host_ledger.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace
{
using namespace ovspatialquery::detail;
// Divides by the quaternion length, unlike makeQueryBoxGeometry; the cases below were built from these axes.
box_sweep::Box makeBox(box_sweep::Vec3 center, const float q[4], box_sweep::Vec3 half)
{
    double length = sqrt(double(q[0]) * q[0] + double(q[1]) * q[1] + double(q[2]) * q[2] + double(q[3]) * q[3]);
    double x = q[0] / length, y = q[1] / length, z = q[2] / length, w = q[3] / length;
    return {center,
            {{1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)},
             {2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)},
             {2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)}},
            half};
}
BoxGeometrySource queryBox(double x = 0)
{
    BoxGeometrySource b{};
    b.center      = {x, 0, 0};
    b.axes        = {std::array<double, 3>{1, 0, 0}, std::array<double, 3>{0, 1, 0}, std::array<double, 3>{0, 0, 1}};
    b.halfExtents = {.5, .5, .5};
    return b;
}
ExactGeometryHit sweep(const ExactGeometryDescription& description, const BoxGeometrySource& box, double minimum,
                       double maximum, bool expected)
{
    const float      d[3]  = {1, 0, 0};
    bool             found = false;
    ExactGeometryHit hit{};
    REQUIRE(sweepBoxGeometry(description, box, d, minimum, maximum, found, hit) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(found == expected);
    return hit;
}
void primitives()
{
    auto                           q = queryBox();
    SphereExactGeometryDescription sphere{{5, 0, 0}, 1};
    auto                           h = sweep(sphere, q, 0, 10, true);
    REQUIRE(std::abs(h.distance - 3.5) < 1e-12);
    REQUIRE(std::abs(h.position[0] - 4) < 1e-12);
    REQUIRE(h.normal[0] == -1);
    sweep(sphere, q, 0, 3.49, false);
    h = sweep(sphere, q, 4, 10, true);
    REQUIRE(h.distance == 4);
    BoxExactGeometryDescription target{{5, 0, 0}, q.axes, {1, 1, 1}};
    h = sweep(target, q, 0, 10, true);
    REQUIRE(std::abs(h.distance - 3.5) < 1e-12);
    REQUIRE(std::abs(h.position[0] - 4) < 1e-12);
    q.halfExtents[0] += .125;
    h = sweep(target, q, 0, 10, true);
    REQUIRE(h.distance == 3.375);
    q = queryBox();
    CapsuleExactGeometryDescription capsule{{5, -4, 0}, {5, 4, 0}, .25};
    h = sweep(capsule, q, 0, 10, true);
    REQUIRE(std::abs(h.distance - 4.25) < 1e-12);
    REQUIRE(std::abs(h.position[0] - 4.75) < 1e-12);
    // The closest capsule feature is a segment interior, not either endpoint.
    q.center[2] = .7;
    h           = sweep(capsule, q, 0, 10, true);
    REQUIRE(std::abs(h.distance - (4.5 - std::sqrt(.25 * .25 - .2 * .2))) < 1e-12);
    q.center[2] = .8;
    sweep(capsule, q, 0, 10, false);
    q = queryBox(5);
    h = sweep(capsule, q, 0, 10, true);
    REQUIRE(h.distance == 0);
    // Translated coordinates remain double; neither box is rounded to world float.
    q             = queryBox(1e10);
    target.center = {1e10 + 5, 0, 0};
    h             = sweep(target, q, 0, 10, true);
    REQUIRE(h.distance == 3.5);
}
void meshAndConvex()
{
    TriangleMeshSource mesh{};
    mesh.key           = 73;
    mesh.vertices      = {0, -2, -2, 0, -2, 2, 0, 2, 2, 0, 2, -2};
    mesh.triangles     = {0, 1, 2, 0, 2, 3};
    mesh.sourceFaces   = {71, 71};
    mesh.sourceCorners = {3, 7, 9, 3, 9, 12};
    std::shared_ptr<const PreparedTriangleMesh> prepared;
    REQUIRE(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS);
    // A uniform scale exercises the scaled gather and world test.
    std::array<double, 16>                        matrix = {2, 0, 0, 5, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
    std::shared_ptr<const TriangleObjectGeometry> object;
    REQUIRE(bindPreparedTriangleGeometry(prepared, matrix, true, object) == OVSPATIALQUERY_API_SUCCESS);
    TriangleGeometryDescriptionView description{};
    REQUIRE(exportTriangleGeometryDescription(*object, description));
    auto h = sweep(description, queryBox(), 0, 10, true);
    REQUIRE(h.distance == 4.5);
    REQUIRE(h.position[0] == 5);
    REQUIRE(h.hasTriangleSource && h.sourceFace == 71);
    REQUIRE(h.generatedTriangle < 2);
    for (unsigned i = 0; i < 3; ++i)
        REQUIRE(h.sourceCorners[i] == mesh.sourceCorners[3 * h.generatedTriangle + i]);
    REQUIRE(h.uv[0] >= 0 && h.uv[1] >= 0 && h.uv[0] + h.uv[1] <= 1.00001);
    auto q      = queryBox();
    q.center[1] = 40;
    sweep(description, q, 0, 10, false);
    h = sweep(description, queryBox(5), 0, 10, true);
    REQUIRE(h.distance == 0);
    ConvexGeometrySource convex{};
    convex.vertices          = {-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1};
    convex.faceVertexCounts  = {4, 4, 4, 4, 4, 4};
    convex.faceVertexIndices = {0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5};
    convex.worldFromLocal    = {1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::shared_ptr<const ExactGeometry> geometry;
    REQUIRE(createConvexExactGeometry(convex, geometry) == ConvexConstructionOutcome::success);
    ExactGeometryDescription exact;
    REQUIRE(geometry->exportDescription(exact));
    h = sweep(exact, queryBox(), 0, 10, true);
    REQUIRE(h.distance == 3.5);
    h = sweep(exact, queryBox(5), 0, 10, true);
    REQUIRE(h.distance == 0);
}
void rotatedMeshCornerGather()
{
    TriangleMeshSource mesh{};
    mesh.key = 174;
    // Contact is near a box corner, outside its inscribed sphere. Rotating the
    // instance also turns the world sweep into a diagonal local path.
    mesh.vertices      = {5, .45f, .45f, 5, .5f, .45f, 5, .45f, .5f};
    mesh.triangles     = {0, 1, 2};
    mesh.sourceFaces   = {174};
    mesh.sourceCorners = {0, 1, 2};
    std::shared_ptr<const PreparedTriangleMesh> prepared;
    REQUIRE(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS);
    const double c = std::cos(.6), s = std::sin(.6);
    for (double scale : {.25, 2.0})
    {
        const std::array<double, 16> matrix = {scale * c, -scale * s, 0,     0, scale * s, scale * c, 0, 0,
                                               0,         0,          scale, 0, 0,         0,         0, 1};
        std::shared_ptr<const TriangleObjectGeometry> object;
        REQUIRE(bindPreparedTriangleGeometry(prepared, matrix, true, object) == OVSPATIALQUERY_API_SUCCESS);
        TriangleGeometryDescriptionView gathered{};
        REQUIRE(exportTriangleGeometryDescription(*object, gathered));
        auto scanned             = gathered;
        scanned.owner            = nullptr;
        auto q                   = queryBox();
        q.halfExtents            = {.5 * scale, .5 * scale, .5 * scale};
        const float direction[3] = {1, 0, 0};
        // Aim the world-axis sweep at the rotated corner triangle, while the
        // box axes follow the instance. Compare against the complete SAT scan.
        q.axes   = {{{c, s, 0}, {-s, c, 0}, {0, 0, 1}}};
        q.center = {0, scale * (5 * s + .47 * c), 0};
        ExactGeometryHit accelerated{}, exhaustive{};
        bool             hasAccelerated = false, hasExhaustive = false;
        REQUIRE(sweepBoxGeometry(gathered, q, direction, 0, 20, hasAccelerated, accelerated) ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(sweepBoxGeometry(scanned, q, direction, 0, 20, hasExhaustive, exhaustive) ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(hasAccelerated && hasExhaustive);
        REQUIRE(accelerated.distance == exhaustive.distance);
        REQUIRE(accelerated.sourceFace == 174 && accelerated.generatedTriangle == exhaustive.generatedTriangle);
        for (unsigned axis = 0; axis < 3; ++axis)
            REQUIRE(accelerated.position[axis] == exhaustive.position[axis]);
    }
}
// Deterministic across platforms, unlike the standard distributions.
struct Random
{
    uint64_t state;
    double   next()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return double(state >> 11) * 0x1.0p-53;
    }
    double between(double low, double high) { return low + (high - low) * next(); }
};
// `cells` per side of the plane: 10 keeps the mesh below the size at which a
// box sweep gathers its path in two parts, 50 takes it above.
void acceleratedMatchesExhaustive(uint32_t cells)
{
    // A tessellated plane gives shared edges and equal-time ties; a random
    // soup adds slivers, initial overlaps and triangles beside each path.
    Random             random{0x9e3779b97f4a7c15ull};
    TriangleMeshSource mesh{};
    mesh.key = 171 + cells;
    for (uint32_t y = 0; y <= cells; ++y)
        for (uint32_t x = 0; x <= cells; ++x)
            mesh.vertices.insert(mesh.vertices.end(), {float(-2 + 4. * x / cells), float(-2 + 4. * y / cells), 0.f});
    for (uint32_t y = 0; y < cells; ++y)
        for (uint32_t x = 0; x < cells; ++x)
        {
            const uint32_t corner = y * (cells + 1) + x;
            mesh.triangles.insert(mesh.triangles.end(), {corner, corner + 1, corner + cells + 2, corner,
                                                         corner + cells + 2, corner + cells + 1});
        }
    for (int t = 0; t < 400; ++t)
    {
        const auto   first   = uint32_t(mesh.vertices.size() / 3);
        const double size    = t % 4 ? .5 : .002;
        const double base[3] = {random.between(-2, 2), random.between(-2, 2), random.between(-2, 2)};
        for (int v = 0; v < 3; ++v)
            for (int a = 0; a < 3; ++a)
                mesh.vertices.push_back(float(base[a] + random.between(-size, size)));
        mesh.triangles.insert(mesh.triangles.end(), {first, first + 1, first + 2});
    }
    for (uint32_t t = 0; t < mesh.triangles.size() / 3; ++t)
    {
        mesh.sourceFaces.push_back(1000 + t);
        mesh.sourceCorners.insert(mesh.sourceCorners.end(), {3 * t, 3 * t + 1, 3 * t + 2});
    }
    std::shared_ptr<const PreparedTriangleMesh> prepared;
    REQUIRE(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS);
    const double c = std::cos(.7), s = std::sin(.7), c4 = std::cos(.4), s4 = std::sin(.4);
    // Identity, then rigid placements with uniform scale about each axis, one far away.
    const std::array<std::array<double, 16>, 5> placements = {{
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
        {.37 * c, -.37 * s, 0, 3, .37 * s, .37 * c, 0, -2, 0, 0, .37, 1, 0, 0, 0, 1},
        {2.5 * c, 0, 2.5 * s, 1e6, 0, 2.5, 0, -2e6, -2.5 * s, 0, 2.5 * c, 5e5, 0, 0, 0, 1},
        {2, 0, 0, 5, 0, 2 * c4, -2 * s4, 0, 0, 2 * s4, 2 * c4, 0, 0, 0, 0, 1},
        {1.5 * c, -1.5 * s, 0, 0, 1.5 * s, 1.5 * c, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 1},
    }};
    size_t                                      hits = 0, starts = 0, still = 0;
    for (size_t placement = 0; placement < placements.size(); ++placement)
    {
        const auto&                                   matrix = placements[placement];
        std::shared_ptr<const TriangleObjectGeometry> object;
        REQUIRE(bindPreparedTriangleGeometry(prepared, matrix, placement % 2 == 0, object) ==
                OVSPATIALQUERY_API_SUCCESS);
        TriangleGeometryDescriptionView accelerated{};
        REQUIRE(exportTriangleGeometryDescription(*object, accelerated));
        auto exhaustive  = accelerated;
        exhaustive.owner = nullptr;
        for (int i = 0; i < 300; ++i)
        {
            const bool                   aligned  = i % 4 == 0;
            const double                 local[3] = {random.between(-3, 3), random.between(-3, 3),
                                     aligned ? random.between(.05, 1) : random.between(-3, 3)};
            ovspatialquery_overlap_box_t box{};
            for (int a = 0; a < 3; ++a)
                box.center[a] = matrix[4 * a] * local[0] + matrix[4 * a + 1] * local[1] + matrix[4 * a + 2] * local[2] +
                                matrix[4 * a + 3];
            for (int a = 0; a < 3; ++a)
                box.half_extents[a] = float(random.between(.05, .8));
            if (aligned)
                box.rotation_xyzw[3] = 1;
            else
            {
                double q[4], length = 0;
                for (double& value : q)
                {
                    value = random.between(-1, 1);
                    length += value * value;
                }
                for (int a = 0; a < 4; ++a)
                    box.rotation_xyzw[a] = float(q[a] / std::sqrt(length));
            }
            float direction[3] = {0, 0, 0};
            if (aligned)
                direction[i / 4 % 3] = i % 8 ? -1.f : 1.f;
            else
            {
                double d[3], length = 0;
                for (double& value : d)
                {
                    value = random.between(-1, 1);
                    length += value * value;
                }
                for (int a = 0; a < 3; ++a)
                    direction[a] = float(d[a] / std::sqrt(length));
            }
            const double minimum = i % 5 == 0 ? random.between(0, 1) : 0;
            // Some sweeps have no length: a closest query asks that after a contact at its minimum.
            const double     length   = random.between(0, 8);
            const double     maximum  = i % 9 == 4 ? minimum : minimum + length;
            const auto       query    = makeQueryBoxGeometry(box);
            bool             found[2] = {};
            ExactGeometryHit hit[2]{};
            REQUIRE(sweepBoxGeometry(accelerated, query, direction, minimum, maximum, found[0], hit[0]) ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(sweepBoxGeometry(exhaustive, query, direction, minimum, maximum, found[1], hit[1]) ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(found[0] == found[1]);
            if (!found[0])
                continue;
            ++hits;
            starts += hit[0].distance == minimum;
            still += maximum == minimum;
            REQUIRE(hit[0].distance == hit[1].distance);
            REQUIRE(hit[0].generatedTriangle == hit[1].generatedTriangle && hit[0].sourceFace == hit[1].sourceFace);
            for (unsigned a = 0; a < 3; ++a)
                REQUIRE(hit[0].position[a] == hit[1].position[a] && hit[0].normal[a] == hit[1].normal[a] &&
                        hit[0].sourceCorners[a] == hit[1].sourceCorners[a]);
            REQUIRE(hit[0].uv[0] == hit[1].uv[0] && hit[0].uv[1] == hit[1].uv[1]);
        }
    }
    // Enough contacts, including touching, initial overlap and sweeps without length, for the
    // comparison to mean something.
    REQUIRE(hits > 600 && starts > 50 && still > 10);
}
void exactTouching()
{
    // Axis-aligned sweeps leave two face axes with zero velocity. A box whose
    // side face lies exactly on the grid's boundary, or which slides on the
    // plane, touches the mesh only there; pruning must keep those triangles.
    TriangleMeshSource mesh{};
    mesh.key = 182;
    for (int y = 0; y <= 4; ++y)
        for (int x = 0; x <= 4; ++x)
            mesh.vertices.insert(mesh.vertices.end(), {float(-2 + x), float(-2 + y), 0.f});
    for (uint32_t y = 0; y < 4; ++y)
        for (uint32_t x = 0; x < 4; ++x)
        {
            const uint32_t corner = y * 5 + x;
            mesh.triangles.insert(mesh.triangles.end(),
                                  {corner, corner + 1, corner + 6, corner, corner + 6, corner + 5});
        }
    std::shared_ptr<const PreparedTriangleMesh> prepared;
    REQUIRE(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS);
    for (double translation : {0.0, 1e6})
    {
        const std::array<double, 16> matrix = {1, 0, 0, translation, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        std::shared_ptr<const TriangleObjectGeometry> object;
        REQUIRE(bindPreparedTriangleGeometry(prepared, matrix, true, object) == OVSPATIALQUERY_API_SUCCESS);
        TriangleGeometryDescriptionView accelerated{};
        REQUIRE(exportTriangleGeometryDescription(*object, accelerated));
        auto exhaustive  = accelerated;
        exhaustive.owner = nullptr;
        struct Case
        {
            double center[3];
            float  direction[3];
            double distance;
        };
        const Case cases[] = {
            {{.5, -2.25, 1}, {0, 0, -1}, .75}, {{.5, 2.25, 1}, {0, 0, -1}, .75}, {{-2.25, .5, -1}, {0, 0, 1}, .75},
            {{2.25, .5, 1}, {0, 0, -1}, .75},  {{-3, .5, .25}, {1, 0, 0}, 0},    {{3, -1.5, -.25}, {-1, 0, 0}, 0},
            {{.5, -2.25, .25}, {0, 1, 0}, 0},
        };
        for (const Case& c : cases)
        {
            BoxGeometrySource query = queryBox();
            query.halfExtents       = {.25, .25, .25};
            for (unsigned a = 0; a < 3; ++a)
                query.center[a] = c.center[a] + (a == 0 ? translation : 0);
            if (c.distance == 0)
                query.center[0] += c.direction[0] * 1.25;
            bool             found[2] = {};
            ExactGeometryHit hit[2]{};
            REQUIRE(sweepBoxGeometry(accelerated, query, c.direction, 0, 5, found[0], hit[0]) ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(sweepBoxGeometry(exhaustive, query, c.direction, 0, 5, found[1], hit[1]) ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(found[0] && found[1]);
            REQUIRE(hit[0].distance == c.distance && hit[1].distance == c.distance);
            REQUIRE(hit[0].generatedTriangle == hit[1].generatedTriangle);
            for (unsigned a = 0; a < 3; ++a)
                REQUIRE(hit[0].position[a] == hit[1].position[a]);
        }
    }
}
void largeFloorAndDenseGrid()
{
    // PhysX tests triangles in float relative to the gather box, so its rounding
    // grows with the vertices, not the box: a box resting on a floor that spans
    // 1e4 must keep the floor. A fine grid under a sliding box gathers hundreds of
    // triangles; a sweep that ends exactly at its contact must still report it.
    TriangleMeshSource floor{};
    floor.key = 183;
    // Tilted (z = 0.3 y) so float differences against 3e3-sized coordinates round.
    floor.vertices  = {-1e4f, -1e4f, -3e3f, 1e4f, -1e4f, -3e3f, 1e4f, 1e4f, 3e3f, -1e4f, 1e4f, 3e3f};
    floor.triangles = {0, 1, 2, 0, 2, 3};
    TriangleMeshSource grid{};
    grid.key = 184;
    for (int y = 0; y <= 60; ++y)
        for (int x = 0; x <= 60; ++x)
            grid.vertices.insert(grid.vertices.end(), {float(-3 + .1 * x), float(-3 + .1 * y), 0.f});
    for (uint32_t y = 0; y < 60; ++y)
        for (uint32_t x = 0; x < 60; ++x)
        {
            const uint32_t corner = y * 61 + x;
            grid.triangles.insert(grid.triangles.end(),
                                  {corner, corner + 1, corner + 62, corner, corner + 62, corner + 61});
        }
    const double c = std::cos(.4), s = std::sin(.4);
    // The identity and a rotated half-scale placement both take the box gather.
    const std::array<std::array<double, 16>, 2> placements = {{
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
        {.5 * c, -.5 * s, 0, 7, .5 * s, .5 * c, 0, -3, 0, 0, .5, 2, 0, 0, 0, 1},
    }};
    for (const TriangleMeshSource* source : {&floor, &grid})
    {
        std::shared_ptr<const PreparedTriangleMesh> prepared;
        REQUIRE(prepareTriangleMesh(*source, prepared) == OVSPATIALQUERY_API_SUCCESS);
        for (const auto& matrix : placements)
        {
            std::shared_ptr<const TriangleObjectGeometry> object;
            REQUIRE(bindPreparedTriangleGeometry(prepared, matrix, true, object) == OVSPATIALQUERY_API_SUCCESS);
            TriangleGeometryDescriptionView accelerated{};
            REQUIRE(exportTriangleGeometryDescription(*object, accelerated));
            auto exhaustive    = accelerated;
            exhaustive.owner   = nullptr;
            const double scale = matrix[10];
            // Resting on the surface and sliding along it, then falling onto it
            // with the range ending exactly at the contact.
            BoxGeometrySource query = queryBox();
            query.halfExtents       = {.25 * scale, .25 * scale, .25 * scale};
            query.axes              = {{{c, s, 0}, {-s, c, 0}, {0, 0, 1}}};
            const float slide[3]    = {float(c), float(s), 0};
            const float fall[3]     = {0, 0, -1};
            // Start 2.5 box-widths back in mesh-local x. On the tilted floor, sink
            // the box 1 um into the plane z = 0.3 y (an initial overlap) by its
            // support along (0, -0.3, 1) in the mesh frame; on the grid, rest on it.
            const bool tilted  = source == &floor;
            double     support = 0;
            for (unsigned j = 0; j < 3; ++j)
            {
                double local[3]{};
                for (unsigned a = 0; a < 3; ++a)
                    for (unsigned b = 0; b < 3; ++b)
                        local[a] += matrix[4 * b + a] * query.axes[j][b] / scale;
                support += .25 * std::abs(tilted ? local[2] - .3 * local[1] : local[2]);
            }
            struct Case
            {
                double       lift, maximum, distance;
                const float* direction;
            };
            for (const Case& test : {Case{0, 5, 0, slide}, Case{1, 1, 1, fall}})
            {
                const double local[3] = {-2.5, 0, support - (tilted ? 1e-6 : 0) + test.lift / scale};
                for (unsigned a = 0; a < 3; ++a)
                    query.center[a] = matrix[4 * a] * local[0] + matrix[4 * a + 1] * local[1] +
                                      matrix[4 * a + 2] * local[2] + matrix[4 * a + 3];
                const double     maximum  = test.maximum * (tilted ? 2 : 1);
                bool             found[2] = {};
                ExactGeometryHit hit[2]{};
                REQUIRE(sweepBoxGeometry(accelerated, query, test.direction, 0, maximum, found[0], hit[0]) ==
                        OVSPATIALQUERY_API_SUCCESS);
                REQUIRE(sweepBoxGeometry(exhaustive, query, test.direction, 0, maximum, found[1], hit[1]) ==
                        OVSPATIALQUERY_API_SUCCESS);
                REQUIRE(found[0] && found[1]);
                REQUIRE(hit[0].distance == hit[1].distance && hit[0].generatedTriangle == hit[1].generatedTriangle);
                // Flat grid: exactly the lift; tilted floor: the lift less the 1 um sink.
                REQUIRE(std::abs(hit[0].distance -
                                 (test.distance == 0 ? 0 : test.lift - (tilted ? 1e-6 * scale : 0))) <= 1e-9);
                for (unsigned a = 0; a < 3; ++a)
                    REQUIRE(hit[0].position[a] == hit[1].position[a]);
                if (tilted || test.distance == 0)
                    continue;
                // The falling box gathers few of the grid's 7,200 triangles, but through a mesh
                // this large it reserves room for all of them, refused one byte short.
                const uint64_t     whole = uint64_t(grid.triangles.size() / 3) * sizeof(uint32_t);
                InstanceHostLedger refused(whole - 1), admitted(whole);
                bool               budgetedFound = false;
                ExactGeometryHit   budgeted{};
                REQUIRE(sweepBoxGeometry(accelerated, query, test.direction, 0, maximum, budgetedFound, budgeted,
                                         &refused) == OVSPATIALQUERY_API_HOST_BUDGET);
                REQUIRE(refused.total() == 0);
                REQUIRE(sweepBoxGeometry(accelerated, query, test.direction, 0, maximum, budgetedFound, budgeted,
                                         &admitted) == OVSPATIALQUERY_API_SUCCESS);
                REQUIRE(admitted.total() == 0 && budgetedFound);
                REQUIRE(budgeted.distance == hit[1].distance && budgeted.generatedTriangle == hit[1].generatedTriangle);
                for (unsigned a = 0; a < 3; ++a)
                    REQUIRE(budgeted.position[a] == hit[1].position[a] && budgeted.normal[a] == hit[1].normal[a]);
            }
        }
    }
}
void grazingTriangleContact()
{
    namespace m                   = box_sweep;
    const float   rotation[4]     = {0, 0, float(std::sin(.25)), float(std::cos(.25))};
    const m::Vec3 centers[2]      = {{0, -.28951570224266532, .24722860985581585},
                                     {0, .077349551358697255, .66588319187086609}};
    const m::Vec3 triangles[2][3] = {{{3.5258660142336429, -.72488159483864334, .46166136452893403},
                                      {2.5350921812976899, 1.1555529619333864, -.81365317980533947},
                                      {3.7939050952037414, 1.121456063207769, -1.8950745450444653}},
                                     {{3.1396519714981901, .87654331037931721, .26319547034634105},
                                      {3.6480168685694774, -1.8196424649899892, 1.4138757937619966},
                                      {3.3587803584985823, -.28617839037001369, 1.9650420579560981}}};
    for (unsigned i = 0; i < 2; ++i)
    {
        auto   box = makeBox(centers[i], rotation, {.25, .2, .2});
        m::Hit hit{};
        REQUIRE(m::boxTriangle(box, triangles[i][0], triangles[i][1], triangles[i][2], true, {1, 0, 0}, 0, 10, hit));
        box.center.x += hit.distance;
        const auto local = m::localPoint(box, hit.position);
        for (unsigned axis = 0; axis < 3; ++axis)
            REQUIRE(std::abs(local[axis]) <= box.half[axis] + 1e-10);
        REQUIRE(hit.u >= -1e-12 && hit.v >= -1e-12 && hit.u + hit.v <= 1 + 1e-12);
        const auto reconstructed =
            m::add(triangles[i][0], m::add(m::mul(m::sub(triangles[i][1], triangles[i][0]), hit.u),
                                           m::mul(m::sub(triangles[i][2], triangles[i][0]), hit.v)));
        REQUIRE(m::maximumComponent(m::sub(reconstructed, hit.position)) < 1e-10);
    }
}

void translatedPreparedFeatures()
{
    const std::vector<float>                  vertices = {-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
                                                          -1, -1, 1,  1, -1, 1,  1, 1, 1,  -1, 1, 1};
    std::shared_ptr<const PreparedConvexHull> hull;
    REQUIRE(preparePortableConvexHull(vertices, 8, UINT64_MAX, hull) == ConvexHullPreparationOutcome::success);
    TriangleMeshSource mesh{};
    mesh.key       = 79;
    mesh.vertices  = {-2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0};
    mesh.triangles = {0, 1, 2, 0, 2, 3};
    std::shared_ptr<const PreparedTriangleMesh> prepared;
    REQUIRE(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS);
    const float direction[3] = {0, 0, -1};
    double      distances[3][2]{};
    for (int pose = 0; pose < 2; ++pose)
    {
        const double                 translation = pose ? 1e20 : 0;
        ovspatialquery_overlap_box_t query{};
        query.center[0]                                = translation;
        query.center[2]                                = 5;
        query.half_extents[0]                          = .25;
        query.half_extents[1]                          = .25;
        query.half_extents[2]                          = .5;
        query.rotation_xyzw[1]                         = float(std::sin(.3926990816987241548));
        query.rotation_xyzw[3]                         = float(std::cos(.3926990816987241548));
        const BoxGeometrySource              box       = makeQueryBoxGeometry(query);
        const std::array<double, 16>         transform = {2, 0, 0, translation, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
        std::shared_ptr<const ExactGeometry> convex;
        REQUIRE(createConvexExactGeometry(hull, transform, convex) == ConvexConstructionOutcome::success);
        ExactGeometryDescription description;
        REQUIRE(convex->exportDescription(description));
        bool             found = false;
        ExactGeometryHit hit{};
        REQUIRE(sweepBoxGeometry(description, box, direction, 0, 10, found, hit) == OVSPATIALQUERY_API_SUCCESS &&
                found);
        distances[0][pose] = hit.distance;
        const CapsuleExactGeometryDescription capsule{{translation, -1, 0}, {translation, 1, 0}, 1};
        ExactGeometryHit                      capsuleHit{};
        REQUIRE(sweepBoxGeometry(capsule, box, direction, 0, 10, found, capsuleHit) == OVSPATIALQUERY_API_SUCCESS &&
                found);
        distances[2][pose]   = capsuleHit.distance;
        const double cornerX = box.axes[0][0] * .25 - box.axes[2][0] * .5;
        const double cornerZ = box.axes[0][2] * .25 - box.axes[2][2] * .5;
        REQUIRE(std::abs(capsuleHit.distance - (5 + cornerZ - std::sqrt(1 - cornerX * cornerX))) < 1e-12);

        const double support = std::abs(box.axes[0][2]) * .25 + std::abs(box.axes[2][2]) * .5;
        REQUIRE(std::abs(hit.distance - (3 - support)) < 1e-12);
        std::shared_ptr<const TriangleObjectGeometry> object;
        REQUIRE(bindPreparedTriangleGeometry(prepared, transform, true, object) == OVSPATIALQUERY_API_SUCCESS);
        TriangleGeometryDescriptionView meshDescription{};
        REQUIRE(exportTriangleGeometryDescription(*object, meshDescription));
        REQUIRE(sweepBoxGeometry(meshDescription, box, direction, 0, 10, found, hit) == OVSPATIALQUERY_API_SUCCESS &&
                found);
        distances[1][pose] = hit.distance;
        REQUIRE(std::abs(hit.distance - (5 - support)) < 1e-12);
        REQUIRE(hit.hasTriangleSource && std::isfinite(hit.uv[0]) && std::isfinite(hit.uv[1]));
    }
    REQUIRE(distances[0][0] == distances[0][1] && distances[1][0] == distances[1][1] &&
            distances[2][0] == distances[2][1]);
}

void continuousTriangleAndRotation()
{
    namespace m          = box_sweep;
    float  quaternion[4] = {0, 0, float(std::sin(.3)), float(std::cos(.3))};
    m::Box box           = makeBox({0, 0, 0}, quaternion, {1, .25, .5});
    m::Hit hit{};
    REQUIRE(m::boxTriangle(box, {5, -10, -10}, {5, -10, 10}, {5, 10, 10}, true, {1, 0, 0}, 0, 10, hit));
    double extent = std::abs(box.axis[0].x) + .25 * std::abs(box.axis[1].x);
    REQUIRE(std::abs(hit.distance - (5 - extent)) < 1e-12);
    REQUIRE(std::abs(hit.position.x - 5) < 1e-12);
    for (double scale : {1e-90, 1e90})
    {
        const float identity[4] = {0, 0, 0, 1};
        const auto  scaled      = makeBox({0, 0, 0}, identity, {.5 * scale, .5 * scale, .5 * scale});
        REQUIRE(m::boxTriangle(scaled, {5 * scale, -scale, -scale}, {5 * scale, scale, -scale}, {5 * scale, 0, scale},
                               true, {1, 0, 0}, 0, 10 * scale, hit));
        REQUIRE(std::abs(hit.distance / scale - 4.5) < 1e-12);
        REQUIRE(std::abs(hit.normal.x + 1) < 1e-12);
        REQUIRE(std::isfinite(hit.u) && std::isfinite(hit.v));
        REQUIRE(std::abs(hit.u - .25) < 1e-12 && std::abs(hit.v - .5) < 1e-12);
    }
    REQUIRE(!m::boxTriangle(box, {5, -10, -10}, {5, 10, 10}, {5, -10, 10}, false, {1, 0, 0}, 0, 10, hit));
    REQUIRE(m::boxTriangle(box, {5, -10, -10}, {5, -10, 10}, {5, 10, 10}, false, {1, 0, 0}, 0, 10, hit));
}
} // namespace
int main()
{
    primitives();
    meshAndConvex();
    continuousTriangleAndRotation();
    translatedPreparedFeatures();
    grazingTriangleContact();
    rotatedMeshCornerGather();
    acceleratedMatchesExhaustive(10);
    acceleratedMatchesExhaustive(50);
    exactTouching();
    largeFloorAndDenseGrid();
    return 0;
}
