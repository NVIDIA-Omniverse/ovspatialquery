/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/test_failure.h"
#include "ovspatialquery_geometry.h"
#include "sphere_sweep_oracle.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

// CPU sphere sweeps against triangle meshes checked by an independent oracle, not by
// another implementation of the library's rule. A direction is float and unit only to
// within rounding: the library accepts squared lengths within 1e-5 of one and places the
// center at center + t d with the direction as given, so every contact must be the first
// parameter at which that moving sphere touches the mesh. Rows cover directions shorter
// and longer than unit, vertex, edge and face contacts, touching and grazing paths,
// initial overlap, clipping at both ends, tiny radii, ray-tube radii, sweeps a kilometre
// long, and rigid placements with uniform scale, near and far. Single-sided placements
// follow ADR-0051: a triangle whose front normal points along the motion is ignored, and
// every other one is met at its first touch from either side.
//
// Placements ask PhysX for the candidate triangles and evaluate the rule on them. PhysX can
// miss a contact outright (a named CPU limitation, owned elsewhere): such a row is counted,
// and the count must stay a small share of the rows.
//
// Long rows run up to about 120 m for ordinary radii and a kilometre for tiny and tube
// radii; contacts thousands of units from the start, where the rule's slack grows with the
// path and the coordinates, are checked through the public API on CPU and CUDA against
// far_sphere_sweep_scene.h.

namespace
{
using namespace ovspatialquery::detail;
namespace oracle = sphere_sweep_oracle;
using oracle::Wide;

void require(bool condition, const char* message, size_t row)
{
    if (!condition)
    {
        std::fprintf(stderr, "sphere sweep contact failed at row %zu: %s\n", row, message);
        OVSPATIALQUERY_TEST_EXIT_FAILURE();
    }
}

uint64_t state = 0x9e3779b97f4a7c15ull;
double   between(double low, double high)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return low + (high - low) * (double(state >> 11) * 0x1.0p-53);
}

struct Placement
{
    std::array<double, 16> matrix;
    bool                   doubleSided;
};

// World corners exactly as the CPU path reconstructs them.
void worldTriangle(const TriangleMeshSource& mesh, const Placement& placement, uint32_t triangle, double corner[3][3])
{
    const auto& m = placement.matrix;
    for (int k = 0; k < 3; ++k)
    {
        const float* local = mesh.vertices.data() + 3 * mesh.triangles[3 * triangle + k];
        for (int a = 0; a < 3; ++a)
            corner[k][a] =
                std::fma(m[4 * a + 2], double(local[2]),
                         std::fma(m[4 * a + 1], double(local[1]), std::fma(m[4 * a], double(local[0]), m[4 * a + 3])));
    }
}

void worldPoint(const Placement& placement, const double local[3], double world[3])
{
    const auto& m = placement.matrix;
    for (int a = 0; a < 3; ++a)
        world[a] = m[4 * a] * local[0] + m[4 * a + 1] * local[1] + m[4 * a + 2] * local[2] + m[4 * a + 3];
}

void worldVector(const Placement& placement, const double local[3], double world[3])
{
    const auto& m = placement.matrix;
    for (int a = 0; a < 3; ++a)
        world[a] = m[4 * a] * local[0] + m[4 * a + 1] * local[1] + m[4 * a + 2] * local[2];
}

// A float direction whose squared length differs from one by about `excess`.
void floatDirection(const double world[3], double excess, float direction[3])
{
    const double length = std::sqrt(world[0] * world[0] + world[1] * world[1] + world[2] * world[2]);
    const double scale  = std::sqrt(1 + excess) / length;
    for (int a = 0; a < 3; ++a)
        direction[a] = float(world[a] * scale);
}

struct Row
{
    size_t placement = 0;
    double center[3]{};
    float  direction[3]{};
    double radius  = 0;
    double minimum = 0;
    double maximum = 0;
};

// n . d within this of zero, for a unit normal and a direction unit to within 1e-5, is
// rounding of the world triangle and the double dot product, not a side of the triangle.
const Wide kParallel = Wide(1e-12);

struct Coverage
{
    size_t hits = 0, starts = 0, vertices = 0, edges = 0, faces = 0, tiny = 0, tubes = 0, longSweeps = 0, shorter = 0,
           longer = 0, clippedStarts = 0, grazing = 0, oneSidedBack = 0, discoveryMisses = 0;
};

// The oracle's verdict on one row: the contact the library reported (or its absence)
// must be real, not penetrating unless the sweep started in contact, and no later than
// the first parameter at which any triangle comes clearly within the radius.
// Returns whether the library missed a clearly touched contact before the one it reports.
bool checkRow(size_t index, const Row& row, const TriangleMeshSource& mesh, const Placement& placement,
              const TriangleObjectGeometry& object, Coverage& coverage)
{
    PreciseTriangleHit          hit{};
    ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
    const bool found = sweepSphereTriangleObject(object, row.center, row.radius, row.direction, row.minimum,
                                                 row.maximum, hit, nullptr, &status);
    require(status == OVSPATIALQUERY_API_SUCCESS, "sweep status", index);
    const double squared   = row.radius * row.radius;
    const double tolerance = std::max(1.0, squared) * std::numeric_limits<double>::epsilon() * 256.0;
    // The library accepts a center within `accept`; `slack` is far above double rounding of
    // these coordinates and far below the (|d|^2 - 1) t^2 error this test guards against.
    const Wide     accept = std::sqrt(Wide(squared) + tolerance);
    const Wide     slack  = Wide(1e-9) * std::max(1.0, row.radius);
    const Wide     lower = row.minimum, upper = row.maximum;
    bool           required     = false;
    Wide           firstClear   = 0;
    bool           reportedKept = true;
    const uint32_t triangles    = uint32_t(mesh.triangles.size() / 3);
    for (uint32_t triangle = 0; triangle < triangles; ++triangle)
    {
        double corner[3][3];
        worldTriangle(mesh, placement, triangle, corner);
        const oracle::Path path  = oracle::makePath(corner, row.center, row.direction);
        Wide               touch = 0;
        if (!oracle::mayReach(path, accept + slack, lower, upper) ||
            !oracle::firstWithin(path, accept + slack, lower, upper, touch))
            continue;
        // A single-sided triangle the sphere moves along (n . d > 0) is ignored whole; every
        // other one counts from either side. The library takes the sign of n . d in double, so
        // a triangle within rounding of parallel may go either way, except one exactly parallel.
        const Wide approach = oracle::dot(path.normal, path.direction);
        const bool kept     = placement.doubleSided || approach <= kParallel;
        const bool counts   = placement.doubleSided || approach < -kParallel || approach == 0;
        Wide       clear    = 0;
        const bool reaches  = oracle::firstWithin(path, Wide(row.radius) - slack, lower, upper, clear);
        if (found && triangle == hit.generatedTriangle)
            reportedKept = kept;
        if (counts && reaches && (!required || clear < firstClear))
        {
            required   = true;
            firstClear = clear;
        }
    }
    const bool missed =
        required && !(found && Wide(hit.distance) <= firstClear + Wide(1e-12) * std::max(Wide(1), firstClear));
    if (!found)
        return missed;
    ++coverage.hits;
    require(hit.generatedTriangle < triangles && hit.faceIndex == 1000u + hit.generatedTriangle, "reported face",
            index);
    double corner[3][3];
    worldTriangle(mesh, placement, hit.generatedTriangle, corner);
    const oracle::Path path     = oracle::makePath(corner, row.center, row.direction);
    const Wide         t        = hit.distance;
    const Wide         distance = oracle::distanceAt(path, t);
    require(t >= lower && t <= upper, "contact outside the clip", index);
    require(distance <= accept + slack, "reported contact does not touch its face", index);
    require(reportedKept, "reported a single-sided triangle the sphere moves along", index);
    const bool start = hit.distance == row.minimum;
    require(start || distance >= Wide(row.radius) - slack, "reported contact penetrates its face", index);
    // The contact point lies on the face, as far from the moved center as the face is.
    Wide center[3]{}, gap[3]{}, scale = 1;
    oracle::centerAt(path, t, center);
    for (int a = 0; a < 3; ++a)
    {
        const Wide point = Wide(hit.position[a]) - Wide(corner[0][a]);
        gap[a]           = center[a] - point;
        scale            = std::max({scale, std::abs(Wide(hit.position[a]))});
    }
    require(std::abs(std::sqrt(oracle::dot(gap, gap)) - distance) <= slack + Wide(1e-15) * scale,
            "contact point is not the face's closest point", index);
    coverage.starts += start;
    coverage.clippedStarts += start && row.minimum > 0;
    coverage.oneSidedBack += !placement.doubleSided && oracle::heightAt(path, t) < -slack;
    const double w = 1.0 - double(hit.uv[0]) - double(hit.uv[1]);
    const int    zeros =
        (std::abs(w) < 1e-6) + (std::abs(double(hit.uv[0])) < 1e-6) + (std::abs(double(hit.uv[1])) < 1e-6);
    coverage.vertices += zeros >= 2;
    coverage.edges += zeros == 1;
    coverage.faces += zeros == 0;
    coverage.tiny += row.radius < 1e-6;
    coverage.tubes += row.radius == 0.02;
    coverage.longSweeps += row.maximum > 40;
    double squaredLength = 0;
    for (float value : row.direction)
        squaredLength += double(value) * value;
    coverage.shorter += squaredLength < 1 - 1e-7;
    coverage.longer += squaredLength > 1 + 1e-7;
    return missed;
}

void testSphereSweepsMeetTheOracle()
{
    // A tessellated plane with shared edges and a soup of ordinary triangles and millimetre
    // slivers around it, spanning many cooked leaves.
    TriangleMeshSource soup{};
    soup.key                = 880;
    constexpr uint32_t side = 16;
    for (uint32_t y = 0; y <= side; ++y)
        for (uint32_t x = 0; x <= side; ++x)
            soup.vertices.insert(soup.vertices.end(), {float(-3 + 6.0 * x / side), float(-3 + 6.0 * y / side), 0.f});
    for (uint32_t y = 0; y < side; ++y)
        for (uint32_t x = 0; x < side; ++x)
        {
            const uint32_t corner = y * (side + 1) + x;
            soup.triangles.insert(soup.triangles.end(), {corner, corner + 1, corner + side + 2, corner,
                                                         corner + side + 2, corner + side + 1});
        }
    for (int t = 0; t < 400; ++t)
    {
        const auto   first   = uint32_t(soup.vertices.size() / 3);
        const double size    = t % 4 ? .5 : .002;
        const double base[3] = {between(-3, 3), between(-3, 3), between(-3, 3)};
        for (int v = 0; v < 3; ++v)
            for (int a = 0; a < 3; ++a)
                soup.vertices.push_back(float(base[a] + between(-size, size)));
        soup.triangles.insert(soup.triangles.end(), {first, first + 1, first + 2});
    }
    for (uint32_t t = 0; t < soup.triangles.size() / 3; ++t)
    {
        soup.sourceFaces.push_back(1000 + t);
        soup.sourceCorners.insert(soup.sourceCorners.end(), {3 * t, 3 * t + 1, 3 * t + 2});
    }
    std::shared_ptr<const PreparedTriangleMesh> prepared;
    require(prepareTriangleMesh(soup, prepared) == OVSPATIALQUERY_API_SUCCESS, "prepare", 0);
    const double c = std::cos(.7), s = std::sin(.7), c4 = std::cos(.4), s4 = std::sin(.4);
    // Identity, then rigid placements with uniform scale about each axis, one far away.
    const std::vector<Placement> placements = {
        {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, true},
        {{.37 * c, -.37 * s, 0, 3, .37 * s, .37 * c, 0, -2, 0, 0, .37, 1, 0, 0, 0, 1}, false},
        {{2.5 * c, 0, 2.5 * s, 1e6, 0, 2.5, 0, -2e6, -2.5 * s, 0, 2.5 * c, 5e5, 0, 0, 0, 1}, true},
        {{2, 0, 0, 5, 0, 2 * c4, -2 * s4, 0, 0, 2 * s4, 2 * c4, 0, 0, 0, 0, 1}, false},
        {{1.5 * c, -1.5 * s, 0, 0, 1.5 * s, 1.5 * c, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 1}, false},
    };
    std::vector<std::shared_ptr<const TriangleObjectGeometry>> objects;
    for (const Placement& placement : placements)
    {
        std::shared_ptr<const TriangleObjectGeometry> object;
        require(bindPreparedTriangleGeometry(prepared, placement.matrix, placement.doubleSided, object) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "placement", 0);
        objects.push_back(object);
    }
    // Squared-length excess of the float direction: as rounded, shorter and longer by up to
    // nearly the accepted 1e-5.
    const double     excesses[] = {0, 2e-7, -2e-7, 1e-6, -1e-6, 9e-6, -9e-6};
    Coverage         coverage;
    constexpr size_t rows = 1400;
    for (size_t index = 0; index < rows; ++index)
    {
        Row row;
        row.placement          = index % placements.size();
        const size_t i         = index / placements.size();
        const auto&  placement = placements[row.placement];
        const size_t kind      = i % 7;
        const double excess    = excesses[(i / 7) % 7];
        row.radius             = i % 9 == 4 ? 1e-7 : (i % 11 == 6 ? 0.02 : between(.05, .6));
        double local[3]{}, localDirection[3]{}, world[3]{};
        double reach = 0;
        if (kind == 0)
        {
            // Slide or drop along a mesh axis above the plane: axis-aligned only for the identity.
            local[0]                   = between(-4, 4);
            local[1]                   = between(-4, 4);
            local[2]                   = between(.35, 1.2);
            localDirection[i / 49 % 3] = i % 2 ? -1 : 1;
            reach                      = between(0, 6);
        }
        else if (kind == 1)
        {
            // Resting on or sunk into the plane: contact at the start, possibly clipped.
            local[0]          = between(-2, 2);
            local[1]          = between(-2, 2);
            local[2]          = between(-.05, .05);
            localDirection[0] = between(-1, 1);
            localDirection[1] = between(-1, 1);
            localDirection[2] = between(-.2, .2);
            reach             = 2;
        }
        else if (kind == 6)
        {
            // A long approach toward a point inside the soup.
            const double target[3] = {between(-2.5, 2.5), between(-2.5, 2.5), between(-2.5, 2.5)};
            const double away[3]   = {between(-1, 1), between(-1, 1), between(-1, 1)};
            const double length    = std::sqrt(away[0] * away[0] + away[1] * away[1] + away[2] * away[2]);
            const double distance  = row.radius < .05 ? 400 : (i % 2 ? 40 : 20);
            for (int a = 0; a < 3; ++a)
            {
                local[a]          = target[a] + away[a] / length * distance;
                localDirection[a] = target[a] - local[a];
            }
            reach = distance * between(.9, 1.1);
        }
        else
        {
            // From outside the soup toward a point inside it: moving first contacts.
            const double target[3] = {between(-2.5, 2.5), between(-2.5, 2.5), between(-2.5, 2.5)};
            for (int a = 0; a < 3; ++a)
            {
                local[a]          = between(-5, 5);
                localDirection[a] = target[a] - local[a];
            }
            reach = std::sqrt(localDirection[0] * localDirection[0] + localDirection[1] * localDirection[1] +
                              localDirection[2] * localDirection[2]) *
                    between(.5, 1.5);
        }
        worldPoint(placement, local, row.center);
        worldVector(placement, localDirection, world);
        // Placement scale changes path length; keep ranges in world units.
        const double stretch = std::sqrt(world[0] * world[0] + world[1] * world[1] + world[2] * world[2]) /
                               std::sqrt(localDirection[0] * localDirection[0] + localDirection[1] * localDirection[1] +
                                         localDirection[2] * localDirection[2]);
        floatDirection(world, excess, row.direction);
        row.minimum = kind == 3 ? between(0, 1) * stretch : (kind == 1 && i % 2 ? between(0, .5) : 0);
        row.maximum = row.minimum + reach * stretch;
        if (kind == 5)
        {
            // Graze a vertex: pass it at the radius, measured from the float path itself.
            double corner[3][3];
            worldTriangle(soup, placement, uint32_t(between(0, double(soup.triangles.size() / 3) - 1)), corner);
            double offset[3]{}, along = 0, squaredLength = 0;
            for (int a = 0; a < 3; ++a)
            {
                offset[a] = row.center[a] - corner[0][a];
                along += offset[a] * row.direction[a];
                squaredLength += double(row.direction[a]) * row.direction[a];
            }
            double perpendicular[3]{}, length = 0;
            for (int a = 0; a < 3; ++a)
            {
                perpendicular[a] = offset[a] - row.direction[a] * along / squaredLength;
                length += perpendicular[a] * perpendicular[a];
            }
            length = std::sqrt(length);
            if (length > 0)
            {
                for (int a = 0; a < 3; ++a)
                    row.center[a] = corner[0][a] + perpendicular[a] / length * row.radius -
                                    row.direction[a] * (4 * stretch) / std::sqrt(squaredLength);
                row.minimum = 0;
                row.maximum = 8 * stretch;
                ++coverage.grazing;
            }
        }
        // PhysX did not offer the contact.
        if (checkRow(index, row, soup, placement, *objects[row.placement], coverage))
            ++coverage.discoveryMisses;
    }
    std::printf("sphere sweep contacts: %zu hits (%zu vertex, %zu edge, %zu face), %zu at the start (%zu clipped), "
                "%zu tiny, %zu tube, %zu long, %zu shorter and %zu longer directions, %zu grazing rows, "
                "%zu single-sided contacts from behind; %zu contacts PhysX did not offer\n",
                coverage.hits, coverage.vertices, coverage.edges, coverage.faces, coverage.starts,
                coverage.clippedStarts, coverage.tiny, coverage.tubes, coverage.longSweeps, coverage.shorter,
                coverage.longer, coverage.grazing, coverage.oneSidedBack, coverage.discoveryMisses);
    // The rigid placements' excused rows stay a small share: a regression specific to the
    // PhysX-backed path would otherwise pass as discovery misses (25 when recorded).
    require(coverage.discoveryMisses < 50, "PhysX-backed rows excused as discovery misses", rows);
    require(coverage.hits > 700 && coverage.vertices > 20 && coverage.edges > 50 && coverage.faces > 100 &&
                coverage.starts > 50 && coverage.clippedStarts > 10 && coverage.tiny > 20 && coverage.tubes > 20 &&
                coverage.longSweeps > 50 && coverage.shorter > 100 && coverage.longer > 100,
            "coverage", rows);
}
} // namespace

int main()
{
    testSphereSweepsMeetTheOracle();
    std::puts("sphere sweep contact test passed");
    return 0;
}
