/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_box_sweep_geometry.h"
#include "ovspatialquery_box_sweep_math.h"

#include <cmath>
#include <type_traits>

namespace ovspatialquery
{
namespace detail
{
namespace
{
using box_sweep::Box;
using box_sweep::Hit;
using box_sweep::Vec3;
Vec3 vector(const std::array<double, 3>& v) { return {v[0], v[1], v[2]}; }
template <class T>
Box box(const T& b)
{
    return {vector(b.center), {vector(b.axes[0]), vector(b.axes[1]), vector(b.axes[2])}, vector(b.halfExtents)};
}
Vec3 transform(Vec3 p, const double* m)
{
    if (!m)
        return p;
    return {std::fma(m[0], p.x, std::fma(m[1], p.y, m[2] * p.z)), std::fma(m[4], p.x, std::fma(m[5], p.y, m[6] * p.z)),
            std::fma(m[8], p.x, std::fma(m[9], p.y, m[10] * p.z))};
}
Vec3 convexVertex(const ConvexExactGeometryDescription& d, uint32_t i)
{
    const double* p = d.vertices + 3ull * i;
    return transform({p[0], p[1], p[2]}, d.worldFromLocal);
}
Vec3 triangleVertex(const TriangleMeshExactGeometryDescription& d, uint32_t i)
{
    const float* p = d.localVertices + 3ull * i;
    return transform({p[0], p[1], p[2]}, d.worldFromLocal.data());
}
bool convexContains(const ConvexExactGeometryDescription& d, Vec3 p)
{
    if (d.worldFromLocal)
    {
        const double* m   = d.worldFromLocal;
        Vec3          v   = p;
        double        inv = 1 / (d.uniformScale * d.uniformScale);
        p                 = {box_sweep::dot(v, {m[0], m[4], m[8]}) * inv, box_sweep::dot(v, {m[1], m[5], m[9]}) * inv,
                             box_sweep::dot(v, {m[2], m[6], m[10]}) * inv};
    }
    for (uint64_t i = 0; i < d.planeCount; ++i)
        if (box_sweep::dot(vector(d.planes[i].normal), p) > d.planes[i].offset)
            return false;
    return d.planeCount != 0;
}
void copy(const Hit& hit, ExactGeometryHit& out)
{
    out          = {};
    out.distance = hit.distance;
    for (int i = 0; i < 3; ++i)
    {
        out.position[i] = hit.position[i];
        out.normal[i]   = static_cast<float>(hit.normal[i]);
    }
    out.uv[0] = static_cast<float>(hit.u);
    out.uv[1] = static_cast<float>(hit.v);
}
} // namespace

bool sweepBoxBoxInterval(const BoxExactGeometryDescription& target, const BoxGeometrySource& query,
                         const float unitDirection[3], double minimumDistance, double maximumDistance,
                         BoxSweepInterval& out) noexcept
{
    box_sweep::Interval s{};
    if (!box_sweep::boxBoxInterval(box(query), box(target), {unitDirection[0], unitDirection[1], unitDirection[2]},
                                   minimumDistance, maximumDistance, s))
        return false;
    out = {s.enter, s.exit, {s.normal.x, s.normal.y, s.normal.z}};
    return true;
}

void sweepBoxBoxContact(const BoxExactGeometryDescription& target, const BoxGeometrySource& query,
                        const float unitDirection[3], const BoxSweepInterval& interval,
                        ExactGeometryHit& outHit) noexcept
{
    const box_sweep::Interval s{
        interval.enter, interval.exit, {interval.normal[0], interval.normal[1], interval.normal[2]}};
    Hit hit{};
    box_sweep::boxBoxContact(box(query), box(target), {unitDirection[0], unitDirection[1], unitDirection[2]}, s, hit);
    copy(hit, outHit);
}

ovspatialquery_api_status_t sweepBoxGeometry(const ExactGeometryDescription& target, const BoxGeometrySource& query,
                                             const float unitDirection[3], double minimumDistance,
                                             double maximumDistance, bool& hasHit, ExactGeometryHit& outHit,
                                             InstanceHostLedger* ledger) noexcept
{
    if (const auto* mesh = std::get_if<TriangleMeshExactGeometryDescription>(&target); mesh && mesh->owner)
    {
        try
        {
            return sweepBoxTriangleObject(*mesh->owner, query, unitDirection, minimumDistance, maximumDistance, hasHit,
                                          outHit, ledger, mesh->doubleSided);
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (...)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
    }
    const Box  q         = box(query);
    const Vec3 direction = {unitDirection[0], unitDirection[1], unitDirection[2]};
    hasHit               = false;
    std::visit(
        [&](const auto& description)
        {
            using T = std::decay_t<decltype(description)>;
            Hit hit{};
            if constexpr (std::is_same_v<T, SphereExactGeometryDescription>)
            {
                hasHit = box_sweep::boxSphere(q, vector(description.center), description.radius, direction,
                                              minimumDistance, maximumDistance, hit);
                if (hasHit)
                    copy(hit, outHit);
            }
            else if constexpr (std::is_same_v<T, BoxExactGeometryDescription>)
            {
                hasHit = box_sweep::boxBox(q, box(description), direction, minimumDistance, maximumDistance, hit);
                if (hasHit)
                    copy(hit, outHit);
            }
            else if constexpr (std::is_same_v<T, CapsuleExactGeometryDescription>)
            {
                hasHit = box_sweep::boxCapsule(q, vector(description.endpointA), vector(description.endpointB),
                                               description.radius, direction, minimumDistance, maximumDistance, hit);
                if (hasHit)
                    copy(hit, outHit);
            }
            else
            {
                // Local shape features survive large world translations. The
                // contact position is the only value converted back to world.
                const double* placement = nullptr;
                if constexpr (std::is_same_v<T, ConvexExactGeometryDescription>)
                    placement = description.worldFromLocal;
                else
                    placement = description.worldFromLocal.data();
                const Vec3 translation   = placement ? Vec3{placement[3], placement[7], placement[11]} : Vec3{};
                Box        relativeQuery = q;
                relativeQuery.center     = box_sweep::sub(q.center, translation);
                double   best            = maximumDistance;
                uint64_t bestTriangle    = 0;
                for (uint64_t i = 0; i < description.triangleCount; ++i)
                {
                    const uint32_t* index = description.triangleVertexIndices + 3 * i;
                    Vec3            a, b, c;
                    bool            doubleSided = true;
                    if constexpr (std::is_same_v<T, ConvexExactGeometryDescription>)
                    {
                        a = convexVertex(description, index[0]);
                        b = convexVertex(description, index[1]);
                        c = convexVertex(description, index[2]);
                    }
                    else
                    {
                        a           = triangleVertex(description, index[0]);
                        b           = triangleVertex(description, index[1]);
                        c           = triangleVertex(description, index[2]);
                        doubleSided = description.doubleSided;
                    }
                    if (box_sweep::boxTriangle(relativeQuery, a, b, c, doubleSided, direction, minimumDistance, best,
                                               hit) &&
                        (!hasHit || hit.distance < best))
                    {
                        hasHit       = true;
                        best         = hit.distance;
                        bestTriangle = i;
                        copy(hit, outHit);
                    }
                }
                if constexpr (std::is_same_v<T, ConvexExactGeometryDescription>)
                {
                    Box moved    = relativeQuery;
                    moved.center = box_sweep::at(relativeQuery.center, direction, minimumDistance);
                    for (int i = 0; i < 8; ++i)
                        if (convexContains(description, box_sweep::vertex(moved, i)))
                        {
                            // Entirely enclosed boxes have no surface intersection. Select the closest
                            // target-surface point as the initial-overlap witness.
                            double closest = 1.7976931348623157e308;
                            Hit    initial{minimumDistance, {}, box_sweep::mul(direction, -1), 0, 0};
                            for (uint64_t j = 0; j < description.triangleCount; ++j)
                            {
                                const uint32_t* index = description.triangleVertexIndices + 3 * j;
                                double          u, v;
                                Vec3            point = box_sweep::triangleClosest(
                                    moved.center, convexVertex(description, index[0]),
                                    convexVertex(description, index[1]), convexVertex(description, index[2]), u, v);
                                Vec3   delta    = box_sweep::sub(moved.center, point);
                                double distance = box_sweep::dot(delta, delta);
                                if (distance < closest)
                                {
                                    closest          = distance;
                                    initial.position = point;
                                }
                            }
                            hasHit = true;
                            copy(initial, outHit);
                            break;
                        }
                }
                else if (hasHit)
                {
                    outHit.generatedTriangle = static_cast<uint32_t>(bestTriangle);
                    outHit.sourceFace = description.sourceFaces ? description.sourceFaces[bestTriangle] : bestTriangle;
                    for (int i = 0; i < 3; ++i)
                        outHit.sourceCorners[i] = description.sourceCorners
                                                      ? description.sourceCorners[3 * bestTriangle + i]
                                                      : static_cast<uint32_t>(i);
                    outHit.hasTriangleSource = true;
                }
                if (hasHit)
                    for (int axis = 0; axis < 3; ++axis)
                        outHit.position[axis] += translation[axis];
            }
        },
        target);
    return OVSPATIALQUERY_API_SUCCESS;
}
} // namespace detail
} // namespace ovspatialquery
