/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "portable_convex_hull.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
namespace
{

using Point = std::array<double, 3u>;

struct HullFace
{
    uint32_t first  = 0u;
    uint32_t second = 0u;
    uint32_t third  = 0u;
    Point    normal{};
    double   offset  = 0.0;
    bool     removed = false;
};

struct HorizonEdge
{
    uint32_t first    = 0u;
    uint32_t second   = 0u;
    uint32_t useCount = 0u;
};

double dot(const Point& first, const Point& second)
{
    double result = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        result = std::fma(first[axis], second[axis], result);
    }
    return result;
}

Point subtract(const Point& first, const Point& second)
{
    return {
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    };
}

Point cross(const Point& first, const Point& second)
{
    return {
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    };
}

double lengthSquared(const Point& value) { return dot(value, value); }

bool normalize(Point& value)
{
    double maximumComponent = 0.0;
    for (double component : value)
    {
        maximumComponent = std::max(maximumComponent, std::abs(component));
    }
    if (!(maximumComponent > 0.0) || !std::isfinite(maximumComponent))
    {
        return false;
    }
    Point scaled{};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        scaled[axis] = value[axis] / maximumComponent;
    }
    const double squared = lengthSquared(scaled);
    if (!std::isfinite(squared) || squared <= 0.0)
    {
        return false;
    }
    const double inverseLength = 1.0 / std::sqrt(squared);
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        value[axis] = scaled[axis] * inverseLength;
    }
    return true;
}

double signedDistance(const HullFace& face, const Point& point) { return dot(face.normal, point) - face.offset; }

bool makeHullFace(uint32_t first, uint32_t second, uint32_t third, const std::vector<Point>& points,
                  const Point& knownInterior, HullFace& outFace)
{
    Point normal = cross(subtract(points[second], points[first]), subtract(points[third], points[first]));
    if (!normalize(normal))
    {
        return false;
    }
    double offset = dot(normal, points[first]);
    if (dot(normal, knownInterior) > offset)
    {
        std::swap(second, third);
        for (double& component : normal)
        {
            component = -component;
        }
        offset = -offset;
    }
    outFace = HullFace{first, second, third, normal, offset, false};
    return true;
}

bool makeInitialHull(const std::vector<Point>& points, std::array<uint32_t, 4u>& outTetrahedron, Point& outInterior,
                     std::vector<HullFace>& outFaces)
{
    const uint32_t first            = 0u;
    uint32_t       second           = first;
    double         furthestDistance = 0.0;
    for (uint32_t index = 1u; index < points.size(); ++index)
    {
        const double distance = lengthSquared(subtract(points[index], points[first]));
        if (distance > furthestDistance)
        {
            furthestDistance = distance;
            second           = index;
        }
    }
    if (!(furthestDistance > 0.0))
    {
        return false;
    }

    const Point line                 = subtract(points[second], points[first]);
    uint32_t    third                = first;
    double      furthestLineDistance = 0.0;
    for (uint32_t index = 0u; index < points.size(); ++index)
    {
        const double distance = lengthSquared(cross(line, subtract(points[index], points[first])));
        if (distance > furthestLineDistance)
        {
            furthestLineDistance = distance;
            third                = index;
        }
    }
    if (!(furthestLineDistance > 0.0))
    {
        return false;
    }

    Point planeNormal = cross(subtract(points[second], points[first]), subtract(points[third], points[first]));
    if (!normalize(planeNormal))
    {
        return false;
    }
    const double planeOffset           = dot(planeNormal, points[first]);
    uint32_t     fourth                = first;
    double       furthestPlaneDistance = 0.0;
    for (uint32_t index = 0u; index < points.size(); ++index)
    {
        const double distance = std::abs(dot(planeNormal, points[index]) - planeOffset);
        if (distance > furthestPlaneDistance)
        {
            furthestPlaneDistance = distance;
            fourth                = index;
        }
    }
    if (!(furthestPlaneDistance > 0.0))
    {
        return false;
    }

    outTetrahedron = {first, second, third, fourth};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        outInterior[axis] =
            (points[first][axis] + points[second][axis] + points[third][axis] + points[fourth][axis]) * 0.25;
    }
    const uint32_t faceVertices[4u][3u] = {
        {first, second, third},
        {first, fourth, second},
        {second, fourth, third},
        {third, fourth, first},
    };
    for (const auto& faceVerticesValue : faceVertices)
    {
        HullFace face{};
        if (!makeHullFace(faceVerticesValue[0], faceVerticesValue[1], faceVerticesValue[2], points, outInterior, face))
        {
            return false;
        }
        outFaces.push_back(face);
    }
    return true;
}

bool buildTriangulatedHull(const ConvexPointSet& pointSet, std::vector<double>& outVertices,
                           std::vector<uint32_t>& outFaceVertexCounts, std::vector<uint32_t>& outFaceVertexIndices)
{
    std::array<uint32_t, 4u> tetrahedron{};
    Point                    interior{};
    std::vector<HullFace>    faces;
    faces.reserve(pointSet.normalized.size() * 2u);
    if (!makeInitialHull(pointSet.normalized, tetrahedron, interior, faces))
    {
        return false;
    }

    constexpr double kVisibilityTolerance = 1024.0 * std::numeric_limits<double>::epsilon();
    for (uint32_t pointIndex = 0u; pointIndex < pointSet.normalized.size(); ++pointIndex)
    {
        if (std::find(tetrahedron.begin(), tetrahedron.end(), pointIndex) != tetrahedron.end())
        {
            continue;
        }
        std::vector<size_t> visibleFaces;
        for (size_t faceIndex = 0u; faceIndex < faces.size(); ++faceIndex)
        {
            if (!faces[faceIndex].removed &&
                signedDistance(faces[faceIndex], pointSet.normalized[pointIndex]) > kVisibilityTolerance)
            {
                visibleFaces.push_back(faceIndex);
            }
        }
        if (visibleFaces.empty())
        {
            continue;
        }

        std::map<std::pair<uint32_t, uint32_t>, HorizonEdge> horizon;
        for (size_t faceIndex : visibleFaces)
        {
            HullFace&      face          = faces[faceIndex];
            const uint32_t edges[3u][2u] = {
                {face.first, face.second},
                {face.second, face.third},
                {face.third, face.first},
            };
            for (const auto& edge : edges)
            {
                const std::pair<uint32_t, uint32_t> key = std::minmax(edge[0], edge[1]);
                HorizonEdge&                        use = horizon[key];
                if (use.useCount == 0u)
                {
                    use.first  = edge[0];
                    use.second = edge[1];
                }
                ++use.useCount;
            }
            face.removed = true;
        }
        for (const auto& entry : horizon)
        {
            if (entry.second.useCount == 1u)
            {
                HullFace face{};
                if (!makeHullFace(entry.second.first, entry.second.second, pointIndex, pointSet.normalized, interior,
                                  face))
                {
                    return false;
                }
                faces.push_back(face);
            }
            else if (entry.second.useCount != 2u)
            {
                return false;
            }
        }
    }

    std::vector<bool> used(pointSet.authored.size(), false);
    for (const HullFace& face : faces)
    {
        if (!face.removed)
        {
            used[face.first]  = true;
            used[face.second] = true;
            used[face.third]  = true;
        }
    }
    std::vector<uint32_t> remap(pointSet.authored.size(), 0u);
    for (uint32_t index = 0u; index < used.size(); ++index)
    {
        if (used[index])
        {
            remap[index] = static_cast<uint32_t>(outVertices.size() / 3u);
            outVertices.insert(outVertices.end(), pointSet.authored[index].begin(), pointSet.authored[index].end());
        }
    }
    for (const HullFace& face : faces)
    {
        if (!face.removed)
        {
            outFaceVertexCounts.push_back(3u);
            outFaceVertexIndices.push_back(remap[face.first]);
            outFaceVertexIndices.push_back(remap[face.second]);
            outFaceVertexIndices.push_back(remap[face.third]);
        }
    }
    return outVertices.size() >= 12u && outFaceVertexCounts.size() >= 4u;
}

} // namespace

ConvexHullPreparationOutcome preparePortableConvexHull(const std::vector<float>& authoredPoints,
                                                       uint32_t maximumVertexCount, uint64_t maximumResidentBytes,
                                                       std::shared_ptr<const PreparedConvexHull>& outHull,
                                                       const ConvexHullPreparationBudget*         budget,
                                                       const ConvexHullCookingOptions&            options,
                                                       double*                                    outFarthestOutside)
{
    return prepareConvexHullFrom(
        authoredPoints, maximumResidentBytes, outHull, budget,
        [maximumVertexCount](const ConvexPointSet& points, uint32_t vertexLimit, std::vector<double>& vertices,
                             std::vector<uint32_t>& faceVertexCounts, std::vector<uint32_t>& faceVertexIndices,
                             std::shared_ptr<physx::PxConvexMesh>&, CookedFrame&)
        {
            if (!buildTriangulatedHull(points, vertices, faceVertexCounts, faceVertexIndices))
                return ConvexHullPreparationOutcome::topologyRejected;
            // Under a vertex limit PhysX keeps a hull already within it; only PhysX reduces a
            // larger one, so the oracle refuses that instead.
            const uint32_t limit = vertexLimit != 0u ? vertexLimit : maximumVertexCount;
            return vertices.size() / 3u <= limit ? ConvexHullPreparationOutcome::success
                                                 : ConvexHullPreparationOutcome::beyondVertexLimit;
        },
        options, outFarthestOutside);
}

} // namespace detail
} // namespace ovspatialquery
