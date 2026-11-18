/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_STORED_SPHERE_PATHS_H
#define OVSPATIALQUERY_TEST_STORED_SPHERE_PATHS_H

#include "sphere_sweep_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Rays, tolerance tubes and sphere sweeps against stored spheres, shared by the CPU and CUDA
// tests and judged by the independent oracle: a sphere is a triangle whose three corners
// coincide at its center, so the oracle's distance is the distance to that point. Distances
// are path parameters of the float direction as given; rows take its squared length up to
// 9e-6 from one either way, as rounded, and exactly one. They cover a placement far from the
// origin, a micrometre sphere, tiny and ordinary sweep radii, paths a kilometre long, passes
// near tangency, starts inside, touching and just outside, and clipping at both ends,
// including minimums inside the sphere (ADR-0018: a solid reports such a start there).
namespace stored_sphere_paths
{
using Wide = sphere_sweep_oracle::Wide;

struct Target
{
    double center[3];
    double radius;
};

// Participation bit k selects target k.
inline const Target kTargets[3] = {
    {{0.0, 0.0, 0.0}, 0.5},
    {{1234.5, -987.25, 40.125}, 0.3},
    {{3.0, -4.0, 5.0}, 1e-6},
};

// Every tube row shares this tolerance, so one raycast plan can carry it.
constexpr double kTube = 0.01;

struct Row
{
    uint32_t target = 0;
    double   origin[3]{};
    float    direction[3]{};
    double   radius  = 0; // sweep radius, or kTube for a tube; zero for a plain ray
    bool     tube    = false;
    double   minimum = 0;
    double   maximum = 0;
};

inline std::vector<Row> rows()
{
    std::vector<Row> out;
    uint64_t         state   = 0x2545f4914f6cdd1dull;
    const auto       uniform = [&state]
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return double(state >> 11) * 0x1.0p-53;
    };
    // A random unit path direction and a unit side vector across it.
    const auto frame = [&](double unit[3], double side[3])
    {
        for (double* v : {unit, side})
            for (;;)
            {
                double squared = 0;
                for (int a = 0; a < 3; ++a)
                {
                    v[a] = 2 * uniform() - 1;
                    squared += v[a] * v[a];
                }
                if (squared > 0.05 && squared < 1)
                {
                    for (int a = 0; a < 3; ++a)
                        v[a] /= std::sqrt(squared);
                    break;
                }
            }
        const double along   = side[0] * unit[0] + side[1] * unit[1] + side[2] * unit[2];
        double       squared = 0;
        for (int a = 0; a < 3; ++a)
        {
            side[a] -= along * unit[a];
            squared += side[a] * side[a];
        }
        for (int a = 0; a < 3; ++a)
            side[a] /= std::sqrt(squared);
    };
    // Passing `fraction` of the reach from the center, `along` before the closest approach.
    const auto place = [&](uint32_t target, double radius, bool tube, double excess, double along, double fraction)
    {
        Row row{};
        row.target = target;
        row.radius = radius;
        row.tube   = tube;
        double unit[3], side[3];
        frame(unit, side);
        const Target& sphere = kTargets[target];
        const double  reach  = sphere.radius + radius;
        for (int a = 0; a < 3; ++a)
        {
            row.direction[a] = float(unit[a] * std::sqrt(1 + excess));
            row.origin[a]    = sphere.center[a] + side[a] * fraction * reach - unit[a] * along;
        }
        row.maximum = 2 * along + 10;
        return row;
    };
    struct Shape
    {
        double radius;
        bool   tube;
    };
    const Shape shapes[] = {{0, false}, {kTube, true}, {1e-6, false}, {0.25, false}};
    for (uint32_t target = 0; target < 3; ++target)
        for (const Shape& shape : shapes)
        {
            for (double excess : {-9e-6, 0.0, 9e-6})
                for (double along : {0.3, 10.0, 1000.0})
                    for (double fraction : {0.0, 0.7, 0.999, 1.001, 1.2})
                        out.push_back(place(target, shape.radius, shape.tube, excess, along, fraction));
            // Clipping: the minimum before the entry, inside, after the exit; the maximum
            // before the entry, just after it, beyond the exit; and a start inside.
            for (double excess : {-9e-6, 9e-6})
            {
                const Row    base  = place(target, shape.radius, shape.tube, excess, 2.0, 0.5);
                const double reach = kTargets[target].radius + shape.radius;
                double       speed = 0;
                for (float component : base.direction)
                    speed += double(component) * component;
                speed              = std::sqrt(speed);
                const double entry = (2.0 - std::sqrt(0.75) * reach) / speed;
                const double exit = (2.0 + std::sqrt(0.75) * reach) / speed, span = exit - entry;
                const double ranges[][2] = {{entry - 0.1 * span, exit + 1},
                                            {entry + 0.25 * span, exit + 1},
                                            {exit + 0.01 * span, exit + 1},
                                            {0, entry - 0.01 * span},
                                            {0, entry + 0.01 * span},
                                            {entry + 0.25 * span, exit - 0.1 * span},
                                            {entry + 0.25 * span, entry + 0.5 * span}};
                for (const auto& range : ranges)
                {
                    Row row     = base;
                    row.minimum = range[0];
                    row.maximum = range[1];
                    out.push_back(row);
                }
                Row inside     = place(target, shape.radius, shape.tube, excess, 0.0, 0.5);
                inside.maximum = 1;
                out.push_back(inside);
            }
        }
    // Starts on the surface of the sphere at the origin, rounded to either side of it by
    // about eps times its radius, moving obliquely inward: the contact stays at the start
    // however the entry rounds, never at the exit.
    for (const Shape& shape : shapes)
        for (int k = 0; k < 256; ++k)
        {
            Row row{};
            row.radius = shape.radius;
            row.tube   = shape.tube;
            double unit[3], side[3];
            frame(unit, side);
            const double reach  = kTargets[0].radius + shape.radius;
            const double inward = 0.1 + 0.9 * uniform(); // cosine of the angle to the inward normal
            for (int a = 0; a < 3; ++a)
            {
                row.direction[a] = float(unit[a] * std::sqrt(1 + (k % 2 ? 9e-6 : 0.0)));
                row.origin[a]    = reach * (std::sqrt(1 - inward * inward) * side[a] - inward * unit[a]);
            }
            row.maximum = 1;
            out.push_back(row);
        }
    // Exact starts on the unit sphere's surface moving inward, one ulp outside it, and an exact
    // tangent pass; the start rows must report zero or a tiny positive entry, never the exit.
    for (const Shape& shape : shapes)
    {
        const double reach = 0.5 + shape.radius;
        for (double start : {-reach, std::nextafter(-reach, -1.0)})
            for (float x : {1.0f, 1.0f + 0x1.0p-23f})
            {
                Row row{};
                row.radius       = shape.radius;
                row.tube         = shape.tube;
                row.origin[0]    = start;
                row.direction[0] = x;
                row.maximum      = 1;
                out.push_back(row);
            }
        Row tangent{};
        tangent.radius       = shape.radius;
        tangent.tube         = shape.tube;
        tangent.origin[0]    = -10;
        tangent.origin[1]    = reach;
        tangent.direction[0] = 1;
        tangent.maximum      = 20;
        out.push_back(tangent);
    }
    return out;
}

// Why an answer fails the oracle, or null. The contact must touch the moving sphere at the
// first parameter the oracle finds within reach, to a slack of 1e-9 of the reach (at least
// 1e-9); a path already inside the sphere at its minimum reports that minimum. The contact
// point lies on the stored sphere and the unit normal points from its center to the moved one.
inline const char* verdict(const Row& row, bool hit, double distance, const double position[3], const float normal[3])
{
    namespace oracle           = sphere_sweep_oracle;
    const Target&      sphere  = kTargets[row.target];
    const double       c[3][3] = {{sphere.center[0], sphere.center[1], sphere.center[2]},
                                  {sphere.center[0], sphere.center[1], sphere.center[2]},
                                  {sphere.center[0], sphere.center[1], sphere.center[2]}};
    const oracle::Path path    = oracle::makePath(c, row.origin, row.direction);
    const Wide         reach   = Wide(sphere.radius) + row.radius;
    const Wide         slack   = Wide(1e-9) * std::max<Wide>(1, reach);
    const Wide         start   = oracle::distanceAt(path, row.minimum);
    Wide               clear = 0, touch = 0;
    const bool         reaches = oracle::firstWithin(path, reach - slack, row.minimum, row.maximum, clear);
    const bool         nears   = oracle::firstWithin(path, reach + slack, row.minimum, row.maximum, touch);
    if (!hit)
        return reaches ? "missed a contact the oracle finds" : nullptr;
    if (!nears)
        return "reported a contact the path never comes within reach of";
    if (!(distance >= row.minimum && distance <= row.maximum))
        return "contact outside the distance interval";
    const Wide gap = oracle::distanceAt(path, distance);
    Wide       moved[3]{};
    oracle::centerAt(path, distance, moved);
    if (start < reach - slack)
    {
        if (distance != row.minimum)
            return "a start inside did not report the minimum";
    }
    else if (std::abs(gap - reach) > slack)
        return "contact does not touch the moving sphere";
    else if (reaches && Wide(distance) > clear + Wide(1e-12) * std::max<Wide>(1, clear))
        return "contact later than the first the oracle finds";
    Wide surface[3]{}, normalLength = 0, alignment = 0;
    for (int a = 0; a < 3; ++a)
    {
        surface[a] = Wide(position[a]) - sphere.center[a];
        normalLength += Wide(normal[a]) * normal[a];
        alignment += Wide(normal[a]) * moved[a];
    }
    if (std::abs(std::sqrt(oracle::dot(surface, surface)) - sphere.radius) > Wide(1e-9) * std::max(1.0, sphere.radius))
        return "contact point off the stored sphere";
    if (std::abs(normalLength - 1) > 1e-6 || (gap > slack && alignment < gap * (1 - Wide(1e-6))))
        return "normal does not point to the moved center";
    return nullptr;
}
} // namespace stored_sphere_paths

#endif
