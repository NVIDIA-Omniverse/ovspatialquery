/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_SPHERE_SWEEP_ORACLE_H
#define OVSPATIALQUERY_TEST_SPHERE_SWEEP_ORACLE_H

#include <algorithm>
#include <cmath>
#include <limits>

// An independent first contact of a sphere whose center moves along center + t d against
// one triangle. It shares no formula with the library's candidate roots: the distance from
// that moving point to a triangle is convex in t, so the parameters at which it is within
// a reach form one interval, found by a golden-section search for the least distance and a
// bisection before it. Closest points come from the triangle's plane and its three edge
// segments in long double. Distances are path parameters of the float direction as given,
// the library's contract, so a direction that is not exactly unit needs no special case.
namespace sphere_sweep_oracle
{
using Wide = long double;

struct Path
{
    Wide origin[3]{};    // start center relative to the first corner
    Wide direction[3]{}; // the float direction, widened
    Wide edge[2][3]{};   // second and third corners relative to the first
    Wide normal[3]{};    // unit normal of the winding; zero for a degenerate triangle
};

inline Wide dot(const Wide a[3], const Wide b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline Path makePath(const double corner[3][3], const double center[3], const float direction[3])
{
    Path path{};
    for (int axis = 0; axis < 3; ++axis)
    {
        path.origin[axis]    = Wide(center[axis]) - Wide(corner[0][axis]);
        path.direction[axis] = Wide(direction[axis]);
        path.edge[0][axis]   = Wide(corner[1][axis]) - Wide(corner[0][axis]);
        path.edge[1][axis]   = Wide(corner[2][axis]) - Wide(corner[0][axis]);
    }
    const Wide* u = path.edge[0];
    const Wide* v = path.edge[1];
    Wide        n[3]{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    const Wide  length = std::sqrt(dot(n, n));
    if (length > 0)
        for (int axis = 0; axis < 3; ++axis)
            path.normal[axis] = n[axis] / length;
    return path;
}

inline void centerAt(const Path& path, Wide t, Wide point[3])
{
    for (int axis = 0; axis < 3; ++axis)
        point[axis] = path.origin[axis] + path.direction[axis] * t;
}

inline Wide segmentDistance(const Wide point[3], const Wide start[3], const Wide end[3])
{
    Wide edge[3]{}, offset[3]{};
    for (int axis = 0; axis < 3; ++axis)
    {
        edge[axis]   = end[axis] - start[axis];
        offset[axis] = point[axis] - start[axis];
    }
    const Wide length = dot(edge, edge);
    const Wide s      = length > 0 ? std::min<Wide>(1, std::max<Wide>(0, dot(offset, edge) / length)) : 0;
    Wide       gap[3]{};
    for (int axis = 0; axis < 3; ++axis)
        gap[axis] = offset[axis] - edge[axis] * s;
    return std::sqrt(dot(gap, gap));
}

// Distance from the center at t to the triangle: to the plane over the face's interior,
// otherwise to the nearest edge segment.
inline Wide distanceAt(const Path& path, Wide t)
{
    Wide p[3]{};
    centerAt(path, t, p);
    const Wide* u = path.edge[0];
    const Wide* v = path.edge[1];
    const Wide  n[3]{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    const Wide  squared = dot(n, n);
    if (squared > 0)
    {
        const Wide pv[3]{p[1] * v[2] - p[2] * v[1], p[2] * v[0] - p[0] * v[2], p[0] * v[1] - p[1] * v[0]};
        const Wide up[3]{u[1] * p[2] - u[2] * p[1], u[2] * p[0] - u[0] * p[2], u[0] * p[1] - u[1] * p[0]};
        const Wide b = dot(pv, n) / squared, c = dot(up, n) / squared;
        if (b >= 0 && c >= 0 && b + c <= 1)
            return std::abs(dot(p, n)) / std::sqrt(squared);
    }
    const Wide zero[3]{};
    return std::min({segmentDistance(p, zero, u), segmentDistance(p, u, v), segmentDistance(p, v, zero)});
}

// Signed distance of the center at t from the triangle's plane along its winding normal.
inline Wide heightAt(const Path& path, Wide t)
{
    Wide p[3]{};
    centerAt(path, t, p);
    return dot(p, path.normal);
}

// First t in [minimum, maximum] at which the center comes within `reach` of the triangle.
inline bool firstWithin(const Path& path, Wide reach, Wide minimum, Wide maximum, Wide& first)
{
    const auto within = [&](Wide t) { return distanceAt(path, t) <= reach; };
    if (within(minimum))
    {
        first = minimum;
        return true;
    }
    // Any parameter within reach, from a golden-section search for the least distance.
    Wide       low = minimum, high = maximum, inside = maximum;
    bool       found = within(maximum);
    const Wide ratio = (std::sqrt(Wide(5)) - 1) / 2;
    Wide       a = high - ratio * (high - low), b = low + ratio * (high - low);
    Wide       fa = distanceAt(path, a), fb = distanceAt(path, b);
    for (int step = 0; step < 400 && !found && low < a && a < b && b < high; ++step)
    {
        if (fa <= reach || fb <= reach)
        {
            inside = fa <= reach ? a : b;
            found  = true;
        }
        else if (fa < fb)
        {
            high = b;
            b    = a;
            fb   = fa;
            a    = high - ratio * (high - low);
            fa   = distanceAt(path, a);
        }
        else
        {
            low = a;
            a   = b;
            fa  = fb;
            b   = low + ratio * (high - low);
            fb  = distanceAt(path, b);
        }
    }
    if (!found)
        return false;
    // Outside at the minimum and within at `inside`: the interval begins between them.
    low = minimum;
    for (int step = 0; step < 400; ++step)
    {
        const Wide middle = low + (inside - low) / 2;
        if (middle <= low || middle >= inside)
            break;
        (within(middle) ? inside : low) = middle;
    }
    first = inside;
    return true;
}

// Whether the path segment can come within `reach` of the triangle at all: a cheap
// bounding-sphere test so exhaustive checks need not search every far triangle.
inline bool mayReach(const Path& path, Wide reach, Wide minimum, Wide maximum)
{
    Wide centroid[3]{}, radius = 0;
    for (int axis = 0; axis < 3; ++axis)
        centroid[axis] = (path.edge[0][axis] + path.edge[1][axis]) / 3;
    const Wide zero[3]{};
    for (const Wide* corner : {zero, static_cast<const Wide*>(path.edge[0]), static_cast<const Wide*>(path.edge[1])})
    {
        Wide gap[3]{};
        for (int axis = 0; axis < 3; ++axis)
            gap[axis] = corner[axis] - centroid[axis];
        radius = std::max(radius, std::sqrt(dot(gap, gap)));
    }
    Wide start[3]{}, end[3]{};
    centerAt(path, minimum, start);
    centerAt(path, maximum, end);
    return segmentDistance(centroid, start, end) <= radius + reach + reach * Wide(1e-6) + Wide(1e-9);
}
} // namespace sphere_sweep_oracle

#endif
