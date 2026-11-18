/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_typed_geometry.h"

#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_physx_backend.h"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{
namespace
{

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
std::atomic<bool> gFailNextPlacedConvex{false};
#endif

using Point = std::array<double, 3u>;
using Plane = ConvexPlaneDescription;
using Edge  = ConvexEdgeDescription;

struct EdgeUse
{
    uint32_t count          = 0u;
    bool     firstDirection = false;
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

template <typename Scalar>
Point readVertex(const std::vector<Scalar>& vertices, uint32_t index)
{
    const size_t offset = static_cast<size_t>(index) * 3u;
    return {vertices[offset], vertices[offset + 1u], vertices[offset + 2u]};
}

// Center points on their bounds and scale the largest offset to one. Fails when
// every point coincides or an offset overflows.
template <typename Read>
bool normalizeAboutBounds(size_t count, const Read& read, Point& minimum, Point& maximum,
                          std::vector<Point>& normalized)
{
    minimum = read(0u);
    maximum = minimum;
    for (size_t index = 1u; index < count; ++index)
    {
        const Point point = read(index);
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], point[axis]);
            maximum[axis] = std::max(maximum[axis], point[axis]);
        }
    }
    Point center{};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
        center[axis] = minimum[axis] * 0.5 + maximum[axis] * 0.5;
    double scale = 0.0;
    for (size_t index = 0u; index < count; ++index)
        for (double component : subtract(read(index), center))
            scale = std::max(scale, std::abs(component));
    if (!std::isfinite(scale) || scale <= 0.0)
        return false;
    normalized.resize(count);
    for (size_t index = 0u; index < count; ++index)
    {
        const Point delta = subtract(read(index), center);
        for (uint32_t axis = 0u; axis < 3u; ++axis)
            normalized[index][axis] = delta[axis] / scale;
    }
    return true;
}

// Some four points span a volume: a second point away from the first, a third off their line
// and a fourth off their plane.
bool spansVolume(const std::vector<Point>& points)
{
    size_t second = 0u, third = 0u;
    double furthest = 0.0;
    for (size_t index = 1u; index < points.size(); ++index)
        if (lengthSquared(subtract(points[index], points[0])) > furthest)
        {
            furthest = lengthSquared(subtract(points[index], points[0]));
            second   = index;
        }
    if (!(furthest > 0.0))
        return false;
    const Point line = subtract(points[second], points[0]);
    furthest         = 0.0;
    for (size_t index = 0u; index < points.size(); ++index)
        if (lengthSquared(cross(line, subtract(points[index], points[0]))) > furthest)
        {
            furthest = lengthSquared(cross(line, subtract(points[index], points[0])));
            third    = index;
        }
    if (!(furthest > 0.0))
        return false;
    Point normal = cross(line, subtract(points[third], points[0]));
    if (!normalize(normal))
        return false;
    for (const Point& point : points)
        if (std::abs(dot(normal, point) - dot(normal, points[0])) > 0.0)
            return true;
    return false;
}

bool qualifyPointSet(const std::vector<float>& authoredPoints, ConvexPointSet& outPoints)
{
    if (authoredPoints.size() < 12u || authoredPoints.size() % 3u != 0u)
    {
        return false;
    }

    ConvexPointSet candidate{};
    const size_t   pointCount = authoredPoints.size() / 3u;
    candidate.authored.reserve(pointCount);
    for (size_t index = 0u; index < pointCount; ++index)
    {
        Point point{};
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            const double value = authoredPoints[index * 3u + axis];
            if (!inInputDomain(value))
            {
                return false;
            }
            point[axis] = value == 0.0 ? 0.0 : value;
        }
        candidate.authored.push_back(point);
    }
    std::sort(candidate.authored.begin(), candidate.authored.end());
    candidate.authored.erase(std::unique(candidate.authored.begin(), candidate.authored.end()),
                             candidate.authored.end());
    if (candidate.authored.size() < 4u ||
        candidate.authored.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    if (!normalizeAboutBounds(
            candidate.authored.size(), [&](size_t index) { return candidate.authored[index]; }, candidate.minimum,
            candidate.maximum, candidate.normalized) ||
        !spansVolume(candidate.normalized))
    {
        return false;
    }
    outPoints = std::move(candidate);
    return true;
}

// A face's corners and every vertex may lie this far outside its plane, in coordinates
// normalized about the vertices' bounds (ADR-0030): PhysX merges nearly coplanar hull triangles
// into polygons that are planar only to float precision.
constexpr double kPolygonTolerance = 0x1.0p-20;

/* Splits each polygon whose corners lie further off one plane than kPolygonTolerance, the test
 * hull validation applies, into triangles that fold outward only, and returns whether it split
 * any (ADR-0076). PhysX merges hull triangles into such polygons; the corners are still hull
 * vertices, so the hull's faces there are triangles of them. Starting from a fan, an inner edge
 * whose two triangles fold inward is flipped to the other diagonal of their quadrilateral, which
 * is convex because the polygon is, until no edge folds inward; in exact arithmetic that takes
 * fewer than count^2 flips, which bounds the work when rounding makes a nearly flat
 * quadrilateral fold both ways. Corners that admit no such triangulation keep an inward fold,
 * which hull validation then refuses. */
bool splitNonplanarPolygons(const std::vector<double>& vertices, std::vector<uint32_t>& faceVertexCounts,
                            std::vector<uint32_t>& faceVertexIndices)
{
    using Triangle    = std::array<uint32_t, 3u>;
    const auto vertex = [&vertices](uint32_t index) {
        return Point{vertices[3u * index], vertices[3u * index + 1u], vertices[3u * index + 2u]};
    };
    Point              minimum{};
    Point              maximum{};
    std::vector<Point> normalized;
    if (!normalizeAboutBounds(
            vertices.size() / 3u, [&](size_t index) { return vertex(static_cast<uint32_t>(index)); }, minimum, maximum,
            normalized))
        return false;
    // The same plane test as hull validation: the plane of the first three corners.
    const auto planar = [&normalized](const uint32_t* polygon, uint32_t count)
    {
        Point normal = cross(subtract(normalized[polygon[1]], normalized[polygon[0]]),
                             subtract(normalized[polygon[2]], normalized[polygon[0]]));
        if (!normalize(normal))
            return true; // validation refuses it as malformed; splitting cannot help
        const double offset = dot(normal, normalized[polygon[0]]);
        for (uint32_t corner = 3u; corner < count; ++corner)
            if (std::abs(dot(normal, normalized[polygon[corner]]) - offset) > kPolygonTolerance)
                return false;
        return true;
    };
    // Positive when `point` lies outside the plane of `triangle`, whose corners turn outward.
    const auto outside = [&vertex](const Triangle& triangle, uint32_t point)
    {
        const Point first = vertex(triangle[0]);
        return dot(cross(subtract(vertex(triangle[1]), first), subtract(vertex(triangle[2]), first)),
                   subtract(vertex(point), first));
    };
    std::vector<uint32_t> counts;
    std::vector<uint32_t> indices;
    counts.reserve(faceVertexCounts.size());
    indices.reserve(faceVertexIndices.size() * 3u);
    bool   split  = false;
    size_t offset = 0u;
    for (const uint32_t count : faceVertexCounts)
    {
        const uint32_t* polygon = faceVertexIndices.data() + offset;
        offset += count;
        if (count == 3u || planar(polygon, count))
        {
            counts.push_back(count);
            indices.insert(indices.end(), polygon, polygon + count);
            continue;
        }
        split = true;
        // Directed edge -> the triangle it turns in; an inner edge appears in both directions.
        std::vector<Triangle>                           triangles;
        std::map<std::pair<uint32_t, uint32_t>, size_t> edgeOwner;
        std::vector<std::pair<uint32_t, uint32_t>>      pending;
        const auto                                      own = [&](size_t triangle)
        {
            for (uint32_t edge = 0u; edge < 3u; ++edge)
                edgeOwner[{triangles[triangle][edge], triangles[triangle][(edge + 1u) % 3u]}] = triangle;
        };
        for (uint32_t corner = 1u; corner + 1u < count; ++corner)
        {
            triangles.push_back({polygon[0], polygon[corner], polygon[corner + 1u]});
            own(triangles.size() - 1u);
            if (corner + 2u < count)
                pending.push_back({polygon[0], polygon[corner + 1u]});
        }
        for (size_t flips = 0u; !pending.empty() && flips < size_t(count) * count;)
        {
            const auto [from, to] = pending.back();
            pending.pop_back();
            const auto firstOwner  = edgeOwner.find({from, to});
            const auto secondOwner = edgeOwner.find({to, from});
            if (firstOwner == edgeOwner.end() || secondOwner == edgeOwner.end())
                continue;
            Triangle&  first  = triangles[firstOwner->second];
            Triangle&  second = triangles[secondOwner->second];
            const auto corner = [](const Triangle& triangle, uint32_t a, uint32_t b)
            {
                for (const uint32_t index : triangle)
                    if (index != a && index != b)
                        return index;
                return a;
            };
            const uint32_t apex = corner(first, from, to);
            const uint32_t far  = corner(second, from, to);
            if (!(outside(first, far) > 0.0))
                continue;
            const size_t firstIndex = firstOwner->second, secondIndex = secondOwner->second;
            edgeOwner.erase({from, to});
            edgeOwner.erase({to, from});
            first  = {far, to, apex};
            second = {apex, from, far};
            own(firstIndex);
            own(secondIndex);
            ++flips;
            pending.insert(pending.end(), {{to, apex}, {apex, from}, {from, far}, {far, to}});
        }
        for (const Triangle& triangle : triangles)
        {
            counts.push_back(3u);
            indices.insert(indices.end(), triangle.begin(), triangle.end());
        }
    }
    faceVertexCounts  = std::move(counts);
    faceVertexIndices = std::move(indices);
    return split;
}

bool qualifyAffineTransform(const std::array<double, 16u>& matrix, double& outDeterminant)
{
    for (double value : matrix)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    if (matrix[12] != 0.0 || matrix[13] != 0.0 || matrix[14] != 0.0 || matrix[15] != 1.0)
    {
        return false;
    }
    const Point      first                     = {matrix[0], matrix[4], matrix[8]};
    const Point      second                    = {matrix[1], matrix[5], matrix[9]};
    const Point      third                     = {matrix[2], matrix[6], matrix[10]};
    const double     firstLength               = std::sqrt(lengthSquared(first));
    const double     secondLength              = std::sqrt(lengthSquared(second));
    const double     thirdLength               = std::sqrt(lengthSquared(third));
    const double     scale                     = firstLength * secondLength * thirdLength;
    const double     determinant               = dot(first, cross(second, third));
    constexpr double kRelativeDeterminantFloor = 256.0 * std::numeric_limits<double>::epsilon();
    if (!std::isfinite(scale) || scale <= 0.0 || !std::isfinite(determinant) ||
        std::abs(determinant) <= scale * kRelativeDeterminantFloor)
    {
        return false;
    }
    outDeterminant = determinant;
    return true;
}

bool transformVertices(const ConvexGeometrySource& source, std::vector<double>& outVertices, double& outDeterminant)
{
    if (!qualifyAffineTransform(source.worldFromLocal, outDeterminant))
    {
        return false;
    }
    outVertices.resize(source.vertices.size());
    const size_t vertexCount = source.vertices.size() / 3u;
    for (size_t vertex = 0u; vertex < vertexCount; ++vertex)
    {
        const Point local = readVertex(source.vertices, static_cast<uint32_t>(vertex));
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            const double value = std::fma(source.worldFromLocal[axis * 4u], local[0],
                                          std::fma(source.worldFromLocal[axis * 4u + 1u], local[1],
                                                   std::fma(source.worldFromLocal[axis * 4u + 2u], local[2],
                                                            source.worldFromLocal[axis * 4u + 3u])));
            if (!std::isfinite(value))
            {
                return false;
            }
            outVertices[vertex * 3u + axis] = value;
        }
    }
    return true;
}

template <typename Position, typename Normal>
ExactGeometryHit contactHit(double distance, const Position& position, const Normal& normal)
{
    ExactGeometryHit hit{};
    hit.distance = distance;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        hit.position[axis] = position[axis];
        hit.normal[axis]   = static_cast<float>(normal[axis]);
    }
    return hit;
}

template <typename Outcome, typename Body>
Outcome guardAllocation(const Body& body)
{
    try
    {
        return body();
    }
    catch (const std::bad_alloc&)
    {
        return Outcome::outOfMemory;
    }
    catch (const std::length_error&)
    {
        return Outcome::outOfMemory;
    }
    catch (...)
    {
        return Outcome::internalFailure;
    }
}

bool validQueryBox(const BoxGeometrySource& box)
{
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (!std::isfinite(box.center[axis]) || !std::isfinite(box.halfExtents[axis]) || box.halfExtents[axis] < 0.0)
        {
            return false;
        }
        for (double component : box.axes[axis])
        {
            if (!std::isfinite(component))
            {
                return false;
            }
        }
    }
    return true;
}

class ConvexExactGeometry final : public ExactGeometry
{
  public:
    ConvexExactGeometry(std::vector<double> vertices, std::vector<uint32_t> triangles, std::vector<Plane> planes,
                        std::vector<Edge> edges, std::shared_ptr<const TriangleObjectGeometry> surface,
                        const ovspatialquery_aabb_t& bounds, std::shared_ptr<const PreparedConvexHull> preparedHull,
                        double faceSlack, std::vector<uint32_t> planeTriangles)
        : mPreparedHull(std::move(preparedHull)), mVertices(std::move(vertices)), mTriangles(std::move(triangles)),
          mPlanes(std::move(planes)), mPlaneTriangles(std::move(planeTriangles)), mEdges(std::move(edges)),
          mSurface(std::move(surface)), mBounds(bounds), mFaceSlack(faceSlack),
          mVertexMagnitude(
              [this]
              {
                  double magnitude = 0.0;
                  for (const double coordinate : mVertices)
                      magnitude = std::max(magnitude, std::abs(coordinate));
                  return magnitude;
              }()),
          mPlaneProjections(
              [this]
              {
                  std::vector<std::array<double, 2u>> projections;
                  projections.reserve(mPlanes.size());
                  for (const Plane& plane : mPlanes)
                      projections.push_back(projectVertices(plane.normal));
                  return projections;
              }()),
          mPlaneSlack(
              [this]
              {
                  double slack = 0.0;
                  for (size_t plane = 0u; plane < mPlanes.size(); ++plane)
                      slack = std::max(slack, mPlaneProjections[plane][1] - mPlanes[plane].offset);
                  return slack;
              }())
    {
        // A ball about the vertices' centroid inside every face plane.
        const size_t vertexCount = mVertices.size() / 3u;
        for (size_t vertex = 0u; vertex < vertexCount; ++vertex)
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                mInnerCenter[axis] += mVertices[vertex * 3u + axis] / static_cast<double>(vertexCount);
        double inner = std::numeric_limits<double>::infinity();
        for (const Plane& plane : mPlanes)
            inner = std::min(inner, plane.offset - dot(plane.normal, mInnerCenter));
        mInnerRadiusSquared = inner > 0.0 ? inner * inner : -1.0;
    }

    ExactGeometryFamily family() const noexcept override { return ExactGeometryFamily::convex; }

    const std::vector<Edge>& edges() const noexcept { return mEdges; }

    bool exportDescription(ExactGeometryDescription& outDescription) const noexcept override
    {
        ConvexExactGeometryDescription description{};
        description.vertices              = mVertices.data();
        description.vertexCount           = static_cast<uint64_t>(mVertices.size() / 3u);
        description.triangleVertexIndices = mTriangles.data();
        description.triangleCount         = static_cast<uint64_t>(mTriangles.size() / 3u);
        description.planes                = mPlanes.data();
        description.planeCount            = static_cast<uint64_t>(mPlanes.size());
        description.edges                 = mEdges.data();
        description.edgeCount             = static_cast<uint64_t>(mEdges.size());
        outDescription                    = description;
        return true;
    }

    const ovspatialquery_aabb_t& worldBounds() const noexcept override { return mBounds; }

    bool raycast(const double origin[3], const float unitDirection[3], double minimumDistance, double maximumDistance,
                 bool, ExactGeometryHit& outHit, InstanceHostLedger*, ovspatialquery_api_status_t*, bool) const override
    {
        return raycastLocal({origin[0], origin[1], origin[2]}, {unitDirection[0], unitDirection[1], unitDirection[2]},
                            minimumDistance, maximumDistance, outHit);
    }

    // The direction may include inverse instance scale; distances remain in the
    // caller's parameterization. Keep the transformed direction in double precision.
    bool raycastLocal(const Point& origin, const Point& unitDirection, double minimumDistance, double maximumDistance,
                      ExactGeometryHit& outHit) const
    {
        double       entry      = minimumDistance;
        double       exit       = maximumDistance;
        const Plane* entryPlane = nullptr;
        for (const Plane& plane : mPlanes)
        {
            double signedOrigin = -plane.offset;
            double velocity     = 0.0;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                signedOrigin = std::fma(plane.normal[axis], origin[axis], signedOrigin);
                velocity     = std::fma(plane.normal[axis], static_cast<double>(unitDirection[axis]), velocity);
            }
            if (velocity == 0.0)
            {
                if (signedOrigin > 0.0)
                {
                    return false;
                }
                continue;
            }
            const double distance = -signedOrigin / velocity;
            if (velocity < 0.0)
            {
                if (distance > entry)
                {
                    entry      = distance;
                    entryPlane = &plane;
                }
            }
            else
            {
                exit = std::min(exit, distance);
            }
            if (entry > exit)
            {
                return false;
            }
        }

        Point point = {
            origin[0] + static_cast<double>(unitDirection[0]) * entry,
            origin[1] + static_cast<double>(unitDirection[1]) * entry,
            origin[2] + static_cast<double>(unitDirection[2]) * entry,
        };
        if (entryPlane == nullptr)
        {
            double smallestGap = std::numeric_limits<double>::infinity();
            for (const Plane& plane : mPlanes)
            {
                const double gap = plane.offset - dot(plane.normal, point);
                if (gap < smallestGap)
                {
                    smallestGap = gap;
                    entryPlane  = &plane;
                }
            }
            if (entryPlane == nullptr || !std::isfinite(smallestGap))
            {
                return false;
            }
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                point[axis] += entryPlane->normal[axis] * smallestGap;
            }
        }
        outHit = contactHit(entry, point, entryPlane->normal);
        return true;
    }

    bool sweepSphere(const double center[3], double radius, const float unitDirection[3], double minimumDistance,
                     double maximumDistance, ExactGeometryHit& outHit, InstanceHostLedger* ledger,
                     ovspatialquery_api_status_t* outStatus, bool, bool) const override
    {
        const Point start = {
            center[0] + static_cast<double>(unitDirection[0]) * minimumDistance,
            center[1] + static_cast<double>(unitDirection[1]) * minimumDistance,
            center[2] + static_cast<double>(unitDirection[2]) * minimumDistance,
        };
        Point closest{};
        Point normal{};
        if (sphereOverlap(start, radius, &closest, &normal))
        {
            outHit = contactHit(minimumDistance, closest, normal);
            if (outStatus != nullptr)
                *outStatus = OVSPATIALQUERY_API_SUCCESS;
            return true;
        }

        PreciseTriangleHit          sourceHit{};
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        const bool hasHit = sweepSphereTriangleObject(*mSurface, center, radius, unitDirection, minimumDistance,
                                                      maximumDistance, sourceHit, ledger, &status);
        if (outStatus != nullptr)
            *outStatus = status;
        if (status != OVSPATIALQUERY_API_SUCCESS || !hasHit)
        {
            return false;
        }
        outHit = contactHit(sourceHit.distance, sourceHit.position, sourceHit.normal);
        return true;
    }

    bool overlaps(const ovspatialquery_overlap_shape_t& geometry, const double (*boxRotation)[3], InstanceHostLedger*,
                  ovspatialquery_api_status_t*) const override
    {
        // A sphere or box takes the exact test below, as on CUDA; PhysX is not asked.
        if (geometry.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
        {
            const Point center = {
                geometry.data.sphere.center[0],
                geometry.data.sphere.center[1],
                geometry.data.sphere.center[2],
            };
            return overlapsSphere(center, geometry.data.sphere.radius);
        }

        BoxGeometrySource box{};
        return queryBoxGeometry(geometry, boxRotation, box) && overlapsBox(box);
    }

    // The exact sphere test, as on CUDA: the centre inside every face plane or a triangle within
    // the radius. The centre inside a ball within every face plane proves an overlap, and beyond
    // a face plane by more than the radius, the plane slack and the rounding proves it apart.
    // PhysX's float32 sphere test is GJK, which can stall short of its own convergence test and
    // then reports contact up to about 2e-4 of the hull's size away, and which misses a sphere
    // centred within a few float32 steps of some hull vertices, so it is not asked.
    bool overlapsSphere(const Point& center, double radius) const
    {
        if (!std::isfinite(radius) || radius < 0.0)
        {
            return false;
        }
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            if (center[axis] < mBounds.minimum[axis] - radius || center[axis] > mBounds.maximum[axis] + radius)
                return false;
        }
        const double radiusSquared = radius * radius;
        // Unfused products keep these loops inline; their rounding is within the margins below.
        const auto product = [](const Point& first, const Point& second)
        { return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]; };
        const auto squaredDistance = [&](const Point& point)
        {
            const Point delta = subtract(center, point);
            return product(delta, delta);
        };
        if (squaredDistance(mInnerCenter) <= mInnerRadiusSquared)
        {
            return true;
        }
        // The faces the centre lies beyond, the farthest first; a hull of at most 255 vertices
        // has at most 506.
        std::array<uint32_t, 512u> beyondFaces;
        size_t                     beyondCount = 0u;
        double                     separation  = -std::numeric_limits<double>::infinity();
        for (size_t plane = 0u; plane < mPlanes.size(); ++plane)
        {
            const double beyond = product(mPlanes[plane].normal, center) - mPlanes[plane].offset;
            if (beyond > 0.0 && beyondCount < beyondFaces.size())
            {
                beyondFaces[beyondCount++] = static_cast<uint32_t>(plane);
                if (beyond > separation)
                    std::swap(beyondFaces[0], beyondFaces[beyondCount - 1u]);
            }
            separation = std::max(separation, beyond);
        }
        if (separation <= 0.0)
        {
            return true;
        }
        const double magnitude = std::max({std::abs(center[0]), std::abs(center[1]), std::abs(center[2])});
        const double rounding = 16.0 * std::numeric_limits<double>::epsilon() * (magnitude + radius + mVertexMagnitude);
        if (separation > radius + mPlaneSlack + rounding)
        {
            return false;
        }
        const auto reaches = [&](size_t triangle)
        {
            double corners[3][3]{};
            for (uint32_t corner = 0u; corner < 3u; ++corner)
            {
                const Point point = readVertex(mVertices, mTriangles[triangle * 3u + corner]);
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                    corners[corner][axis] = point[axis];
            }
            Point  candidate{};
            double u = 0.0, v = 0.0;
            closestPointOnTriangle(corners, center.data(), candidate.data(), u, v);
            return squaredDistance(candidate) <= radiusSquared;
        };
        // An outside centre's nearest point lies on a face it is beyond, most often the farthest.
        const size_t triangleCount = mTriangles.size() / 3u;
        for (size_t face = 0u; face < beyondCount; ++face)
        {
            const size_t plane = beyondFaces[face];
            const size_t end   = plane + 1u < mPlaneTriangles.size() ? mPlaneTriangles[plane + 1u] : triangleCount;
            for (size_t triangle = mPlaneTriangles[plane]; triangle < end; ++triangle)
            {
                if (reaches(triangle))
                {
                    return true;
                }
            }
        }
        for (size_t triangle = 0u; triangle < triangleCount; ++triangle)
        {
            if (reaches(triangle))
            {
                return true;
            }
        }
        return false;
    }

    // The exact separating-axis test, as on CUDA. A box corner inside the hull (when its planes
    // are its faces) or a hull vertex inside the box proves an overlap first. The face axes
    // read the vertices' projections kept per plane, and each other axis stops reading
    // vertices once they straddle the box's interval, which changes no answer. PhysX's
    // float32 box test is GJK, which reports a box within a tenth of the shapes' margins as
    // overlapping and misses a small box pushed into an edge, so it is not asked.
    bool overlapsBox(const BoxGeometrySource& box) const
    {
        if (!validQueryBox(box))
        {
            return false;
        }
        if (sharesAVertex(box))
        {
            return true;
        }
        for (size_t plane = 0u; plane < mPlanes.size(); ++plane)
        {
            if (separatedOnAxis(mPlanes[plane].normal, box, &mPlaneProjections[plane]))
            {
                return false;
            }
        }
        for (const Point& axis : box.axes)
        {
            if (separatedOnAxis(axis, box))
            {
                return false;
            }
        }
        for (const Edge& edge : mEdges)
        {
            const Point edgeDirection = subtract(readVertex(mVertices, edge.second), readVertex(mVertices, edge.first));
            for (const Point& boxAxis : box.axes)
            {
                const Point axis = cross(edgeDirection, boxAxis);
                if (lengthSquared(axis) > 0.0 && separatedOnAxis(axis, box))
                {
                    return false;
                }
            }
        }
        return true;
    }

    uint64_t residentByteCount() const noexcept override
    {
        uint64_t total = sizeof(ConvexExactGeometry);
        total          = saturatingResidentByteAdd(total, static_cast<uint64_t>(mVertices.capacity()), sizeof(double));
        total = saturatingResidentByteAdd(total, static_cast<uint64_t>(mTriangles.capacity()), sizeof(uint32_t));
        total = saturatingResidentByteAdd(total, static_cast<uint64_t>(mPlanes.capacity()), sizeof(Plane));
        total = saturatingResidentByteAdd(total, static_cast<uint64_t>(mPlaneProjections.capacity()),
                                          sizeof(std::array<double, 2u>));
        total = saturatingResidentByteAdd(total, static_cast<uint64_t>(mPlaneTriangles.capacity()), sizeof(uint32_t));
        total = saturatingResidentByteAdd(total, static_cast<uint64_t>(mEdges.capacity()), sizeof(Edge));
        total = saturatingResidentByteAdd(total, 1u, sizeof(TriangleObjectGeometry));
        const bool definitionOwnsSurface = mPreparedHull != nullptr && mPreparedHull->preparedSurface() != nullptr &&
                                           mPreparedHull->preparedSurface()->mesh == mSurface->mesh;
        return definitionOwnsSurface
                   ? total
                   : saturatingResidentByteAdd(total, 1u, triangleGeometryResidentByteCount(*mSurface));
    }

    void accumulateResidentAllocations(ExactGeometryResidentAccumulator& accumulator, bool) const noexcept override
    {
        (void)accumulator.add(this, residentByteCount());
    }

    const TriangleObjectGeometry* triangleSurface() const noexcept override { return mSurface.get(); }

    bool sphereOverlap(const Point& center, double radius, Point* outClosest, Point* outNormal) const
    {
        if (!std::isfinite(radius) || radius < 0.0)
        {
            return false;
        }
        // A sweep's broad-phase bounds include its entire path. Its starting
        // sphere can still be far from this hull; avoid constructing a surface
        // contact when even the hull's bounds cannot overlap it.
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            if (center[axis] < mBounds.minimum[axis] - radius || center[axis] > mBounds.maximum[axis] + radius)
                return false;
        }
        double closestDistanceSquared = std::numeric_limits<double>::infinity();
        Point  closest{};
        Point  closestNormal{};
        for (size_t triangle = 0u; triangle < mTriangles.size() / 3u; ++triangle)
        {
            const Point  first         = readVertex(mVertices, mTriangles[triangle * 3u]);
            const Point  second        = readVertex(mVertices, mTriangles[triangle * 3u + 1u]);
            const Point  third         = readVertex(mVertices, mTriangles[triangle * 3u + 2u]);
            const double corners[3][3] = {
                {first[0], first[1], first[2]}, {second[0], second[1], second[2]}, {third[0], third[1], third[2]}};
            Point  candidate{};
            double u = 0.0, v = 0.0;
            closestPointOnTriangle(corners, center.data(), candidate.data(), u, v);
            const Point  delta           = subtract(center, candidate);
            const double distanceSquared = lengthSquared(delta);
            if (distanceSquared < closestDistanceSquared)
            {
                closestDistanceSquared = distanceSquared;
                closest                = candidate;
                closestNormal          = cross(subtract(second, first), subtract(third, first));
                normalize(closestNormal);
                if (distanceSquared > 0.0 && !contains(center))
                {
                    closestNormal = delta;
                    normalize(closestNormal);
                }
            }
        }
        const bool inside = contains(center);
        if (!inside && closestDistanceSquared > radius * radius)
        {
            return false;
        }
        if (outClosest != nullptr)
        {
            *outClosest = closest;
        }
        if (outNormal != nullptr)
        {
            *outNormal = closestNormal;
        }
        return true;
    }

  private:
    bool contains(const Point& point) const
    {
        for (const Plane& plane : mPlanes)
        {
            if (dot(plane.normal, point) > plane.offset)
            {
                return false;
            }
        }
        return true;
    }

    // A box corner inside every face plane, or a hull vertex inside the box, proves an overlap
    // when it is inside by more than the test's own rounding, which follows the magnitudes of
    // the box and the hull; anything closer is left to the separating-axis test.
    bool sharesAVertex(const BoxGeometrySource& box) const
    {
        double magnitude = 0.0;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            magnitude += box.halfExtents[axis] + std::abs(box.center[axis]);
        }
        const double rounding = 16.0 * std::numeric_limits<double>::epsilon() * (magnitude + mVertexMagnitude);
        for (uint32_t corner = 0u; corner < (mFaceSlack >= 0.0 ? 8u : 0u); ++corner)
        {
            Point point = box.center;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                const double extent = ((corner >> axis) & 1u) != 0u ? box.halfExtents[axis] : -box.halfExtents[axis];
                for (uint32_t component = 0u; component < 3u; ++component)
                {
                    point[component] = std::fma(extent, box.axes[axis][component], point[component]);
                }
            }
            if (std::all_of(mPlanes.begin(), mPlanes.end(),
                            [&](const Plane& plane)
                            { return dot(plane.normal, point) <= plane.offset - mFaceSlack - rounding; }))
            {
                return true;
            }
        }
        for (size_t vertex = 0u; vertex < mVertices.size() / 3u; ++vertex)
        {
            const Point offset = subtract(readVertex(mVertices, static_cast<uint32_t>(vertex)), box.center);
            bool        inside = true;
            for (uint32_t axis = 0u; axis < 3u && inside; ++axis)
            {
                inside = std::abs(dot(offset, box.axes[axis])) <= box.halfExtents[axis] - rounding;
            }
            if (inside)
            {
                return true;
            }
        }
        return false;
    }

    // The smallest and largest projection of the vertices on an axis.
    std::array<double, 2u> projectVertices(const Point& axis) const
    {
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t vertex = 0u; vertex < mVertices.size() / 3u; ++vertex)
        {
            const double projection = dot(axis, readVertex(mVertices, static_cast<uint32_t>(vertex)));
            minimum                 = std::min(minimum, projection);
            maximum                 = std::max(maximum, projection);
        }
        return {minimum, maximum};
    }

    // Whether the vertices' projections on the axis all lie below or all above the box's
    // interval. `projections` holds them when known; otherwise the vertices are read only
    // until some lie at or above its lower end and some at or below its upper end, after
    // which the extremes cannot separate.
    bool separatedOnAxis(const Point& axis, const BoxGeometrySource& box,
                         const std::array<double, 2u>* projections = nullptr) const
    {
        const double centerProjection = dot(axis, box.center);
        double       radius           = 0.0;
        for (uint32_t boxAxis = 0u; boxAxis < 3u; ++boxAxis)
        {
            radius = std::fma(std::abs(dot(axis, box.axes[boxAxis])), box.halfExtents[boxAxis], radius);
        }
        const double lower = centerProjection - radius;
        const double upper = centerProjection + radius;
        if (projections != nullptr)
        {
            return (*projections)[1] < lower || (*projections)[0] > upper;
        }
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t vertex = 0u; vertex < mVertices.size() / 3u; ++vertex)
        {
            const double projection = dot(axis, readVertex(mVertices, static_cast<uint32_t>(vertex)));
            minimum                 = std::min(minimum, projection);
            maximum                 = std::max(maximum, projection);
            if (!(maximum < lower) && !(minimum > upper))
            {
                return false;
            }
        }
        return maximum < lower || minimum > upper;
    }

    const std::shared_ptr<const PreparedConvexHull> mPreparedHull;
    const std::vector<double>                       mVertices;
    const std::vector<uint32_t>                     mTriangles;
    const std::vector<Plane>                        mPlanes;
    // The first triangle of each plane's face; a face's triangles run up to the next one's.
    const std::vector<uint32_t>                         mPlaneTriangles;
    const std::vector<Edge>                             mEdges;
    const std::shared_ptr<const TriangleObjectGeometry> mSurface;
    const ovspatialquery_aabb_t                         mBounds{};
    // When every face's corners lie on its plane and every vertex inside every plane, to
    // rounding, the planes are the vertex hull's faces to within this distance, so a point
    // inside every plane by more lies inside the hull; negative otherwise. A polygon accepted
    // as planar within kPolygonTolerance keeps the plane of its first corners, which can
    // enclose a sliver outside the hull.
    const double mFaceSlack;
    // The largest vertex coordinate, which scales the rounding of tests against the vertices.
    const double mVertexMagnitude;
    // The vertices' smallest and largest projection on each plane's normal.
    const std::vector<std::array<double, 2u>> mPlaneProjections;
    // How far the farthest vertex lies outside any face plane: a point beyond a plane by more
    // than this and its rounding lies outside the hull by the difference.
    const double mPlaneSlack;
    // A centre within this ball lies inside every face plane; the squared radius is negative
    // when there is none.
    Point  mInnerCenter{};
    double mInnerRadiusSquared = -1.0;
};

ConvexConstructionOutcome prepareWorldConvex(const std::shared_ptr<const PreparedConvexHull>& hull,
                                             const std::array<double, 16u>&                   pose,
                                             std::shared_ptr<const ExactGeometry>&            out);

// A definition owns the validated identity contact object; instances only bind
// its surface and pose. The world-array implementation remains the fallback for
// poses orthonormal only to the adapters' tolerance and for a query radius or
// extent below the local predicates' range.
class PlacedConvexExactGeometry final : public ExactGeometry
{
  public:
    PlacedConvexExactGeometry(std::shared_ptr<const PreparedConvexHull>     hull,
                              std::shared_ptr<const TriangleObjectGeometry> surface, double scale,
                              const ovspatialquery_aabb_t& bounds)
        : mHull(std::move(hull)), mSurface(std::move(surface)), mScale(scale), mBounds(bounds)
    {}

    ExactGeometryFamily          family() const noexcept override { return ExactGeometryFamily::convex; }
    const ovspatialquery_aabb_t& worldBounds() const noexcept override { return mBounds; }

    bool exportDescription(ExactGeometryDescription& out) const noexcept override
    {
        if (!local().exportDescription(out))
            return false;
        auto& description          = std::get<ConvexExactGeometryDescription>(out);
        description.worldFromLocal = mSurface->worldFromLocal.data();
        description.uniformScale   = mScale;
        return true;
    }

    bool raycast(const double origin[3], const float direction[3], double minimum, double maximum, bool,
                 ExactGeometryHit& out, InstanceHostLedger*, ovspatialquery_api_status_t*, bool) const override
    {
        const Point      localOrigin    = localPoint({origin[0], origin[1], origin[2]});
        const Point      localDirection = localVector({direction[0], direction[1], direction[2]});
        ExactGeometryHit hit{};
        if (!local().raycastLocal(localOrigin, localDirection, minimum, maximum, hit))
            return false;
        // Exterior ray contacts already lie on the submitted world ray. Avoid
        // transforming that point back through the pose; inside contacts still
        // need their projected surface position.
        const bool projected = hit.distance <= minimum;
        transformContact(hit, projected);
        if (!projected)
            for (uint32_t axis = 0; axis < 3; ++axis)
                hit.position[axis] = origin[axis] + direction[axis] * hit.distance;
        out = hit;
        return true;
    }

    bool sweepSphere(const double center[3], double radius, const float direction[3], double minimum, double maximum,
                     ExactGeometryHit& out, InstanceHostLedger* ledger, ovspatialquery_api_status_t* outStatus, bool,
                     bool) const override
    {
        const Point localCenter    = localPoint({center[0], center[1], center[2]});
        const Point localDirection = localVector({direction[0], direction[1], direction[2]});
        Point       localStart{};
        for (uint32_t axis = 0; axis < 3; ++axis)
            localStart[axis] = std::fma(localDirection[axis], minimum, localCenter[axis]);
        const double localRadius = radius / mScale;
        if (tinyRadius(localRadius))
            return queryWorld(
                [&](const ExactGeometry& geometry)
                {
                    const Point shifted = relativePoint({center[0], center[1], center[2]});
                    if (!geometry.sweepSphere(shifted.data(), radius, direction, minimum, maximum, out, ledger,
                                              outStatus))
                        return false;
                    translateContact(out);
                    return true;
                },
                ledger, outStatus);
        Point closest{}, normal{};
        if (local().sphereOverlap(localStart, localRadius, &closest, &normal))
        {
            ExactGeometryHit hit = contactHit(minimum, closest, normal);
            transformContact(hit);
            out = hit;
            if (outStatus)
                *outStatus = OVSPATIALQUERY_API_SUCCESS;
            return true;
        }
        // Reconstruct contacts without adding a large translation to each triangle.
        // Keep world axes and scale so direction and distance need no conversion.
        TriangleObjectGeometry shiftedSurface = *mSurface;
        for (uint32_t axis = 0; axis < 3; ++axis)
            shiftedSurface.worldFromLocal[axis * 4 + 3] = 0.0;
        const Point                 shiftedCenter = relativePoint({center[0], center[1], center[2]});
        PreciseTriangleHit          hit{};
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        const bool hasHit = sweepSphereTriangleObject(shiftedSurface, shiftedCenter.data(), radius, direction, minimum,
                                                      maximum, hit, ledger, &status);
        if (outStatus)
            *outStatus = status;
        if (status != OVSPATIALQUERY_API_SUCCESS || !hasHit)
            return false;
        ExactGeometryHit result = contactHit(hit.distance, hit.position, hit.normal);
        translateContact(result);
        out = result;
        return true;
    }

    bool overlaps(const ovspatialquery_overlap_shape_t& shape, const double (*boxRotation)[3],
                  InstanceHostLedger* ledger, ovspatialquery_api_status_t* outStatus) const override
    {
        // As above: a sphere or box takes the exact test, without PhysX.
        const auto worldOverlap = [&]
        {
            auto shifted = shape;
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                const double translation = mSurface->worldFromLocal[axis * 4 + 3];
                if (shape.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
                    shifted.data.sphere.center[axis] -= translation;
                else if (shape.kind == OVSPATIALQUERY_OVERLAP_BOX)
                    shifted.data.box.center[axis] -= translation;
                else if (shape.kind == OVSPATIALQUERY_OVERLAP_AABB)
                {
                    shifted.data.aabb.min_corner[axis] -= translation;
                    shifted.data.aabb.max_corner[axis] -= translation;
                }
            }
            return queryWorld([&](const ExactGeometry& geometry)
                              { return geometry.overlaps(shifted, boxRotation, ledger, outStatus); },
                              ledger, outStatus);
        };
        if (shape.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
        {
            const auto&  sphere = shape.data.sphere;
            const Point  center = localPoint({sphere.center[0], sphere.center[1], sphere.center[2]});
            const double radius = sphere.radius / mScale;
            if (tinyRadius(radius))
                return worldOverlap();
            return local().overlapsSphere(center, radius);
        }
        BoxGeometrySource box{};
        if (!queryBoxGeometry(shape, boxRotation, box))
            return false;
        box.center = localPoint(box.center);
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            box.axes[axis] = localVector(box.axes[axis]);
            for (double& component : box.axes[axis])
                component *= mScale;
            const double extent = box.halfExtents[axis] / mScale;
            if (box.halfExtents[axis] != 0.0 && tinyRadius(extent))
                return worldOverlap();
            box.halfExtents[axis] = extent;
        }
        return local().overlapsBox(box);
    }

    uint64_t residentByteCount() const noexcept override
    {
        // The definition already charges its local contacts and shared mesh.
        return sizeof(PlacedConvexExactGeometry) + sizeof(TriangleObjectGeometry);
    }
    void accumulateResidentAllocations(ExactGeometryResidentAccumulator& accumulator, bool) const noexcept override
    {
        (void)accumulator.add(this, residentByteCount());
    }

    const TriangleObjectGeometry* triangleSurface() const noexcept override { return mSurface.get(); }

  private:
    // Local contact predicates contain products of dot products, which a
    // local radius this small underflows; such a query uses the world path.
    // Inside the input domain local values stay below 2^123 (ADR-0032).
    static bool tinyRadius(double radius) { return radius < 1.0e-150; }

    template <typename Query>
    bool queryWorld(const Query& query, InstanceHostLedger* ledger, ovspatialquery_api_status_t* outStatus) const
    {
        HostLease workspace;
        if (ledger)
        {
            // Bounds source copies, normalized work arrays and the temporary
            // world contact object; no fallback allocation survives this query.
            HostTransaction transaction(*ledger);
            if (!transaction.reserve(saturatingResidentByteAdd(0u, 3u, mHull->residentByteCount())) ||
                !transaction.commit(workspace))
            {
                if (outStatus)
                    *outStatus = OVSPATIALQUERY_API_HOST_BUDGET;
                return false;
            }
        }
        std::shared_ptr<const ExactGeometry> geometry;
        auto                                 pose = mSurface->worldFromLocal;
        for (uint32_t axis = 0; axis < 3; ++axis)
            pose[axis * 4 + 3] = 0.0;
        const auto outcome = prepareWorldConvex(mHull, pose, geometry);
        if (outcome != ConvexConstructionOutcome::success)
        {
            if (outStatus)
                *outStatus = outcome == ConvexConstructionOutcome::outOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                                                               : OVSPATIALQUERY_API_INTERNAL;
            return false;
        }
        return query(*geometry);
    }

    const ConvexExactGeometry& local() const noexcept
    {
        // Only createPreparedConvexHull supplies this retained identity geometry.
        return static_cast<const ConvexExactGeometry&>(*mHull->localGeometry());
    }
    Point localVector(const Point& vector) const
    {
        Point result{};
        for (uint32_t row = 0; row < 3; ++row)
            for (uint32_t column = 0; column < 3; ++column)
                result[row] = std::fma(mSurface->localFromWorldLinear[row * 3 + column], vector[column], result[row]);
        return result;
    }
    Point relativePoint(const Point& point) const
    {
        const auto& m = mSurface->worldFromLocal;
        return {point[0] - m[3], point[1] - m[7], point[2] - m[11]};
    }
    Point localPoint(const Point& point) const { return localVector(relativePoint(point)); }
    void  translateContact(ExactGeometryHit& hit) const
    {
        for (uint32_t axis = 0; axis < 3; ++axis)
            hit.position[axis] += mSurface->worldFromLocal[axis * 4 + 3];
    }
    void transformContact(ExactGeometryHit& hit, bool transformPosition = true) const
    {
        const auto& m = mSurface->worldFromLocal;
        const Point position{hit.position[0], hit.position[1], hit.position[2]};
        const Point normal{hit.normal[0], hit.normal[1], hit.normal[2]};
        for (uint32_t row = 0; row < 3; ++row)
        {
            if (transformPosition)
                hit.position[row] = std::fma(
                    m[row * 4], position[0],
                    std::fma(m[row * 4 + 1], position[1], std::fma(m[row * 4 + 2], position[2], m[row * 4 + 3])));
            double rotated = 0.0;
            for (uint32_t column = 0; column < 3; ++column)
                rotated = std::fma(m[row * 4 + column], normal[column], rotated);
            hit.normal[row] = static_cast<float>(rotated / mScale);
        }
    }
    const std::shared_ptr<const PreparedConvexHull>     mHull;
    const std::shared_ptr<const TriangleObjectGeometry> mSurface;
    const double                                        mScale;
    const ovspatialquery_aabb_t                         mBounds;
};

// A rigid pose preserves the validated local hull. Compute finite world bounds
// without reconstructing or revalidating a rounded world polyhedron.
bool placedConvexBounds(const PreparedConvexHull& hull, const TriangleObjectGeometry& surface, double scale,
                        ovspatialquery_aabb_t& bounds)
{
    constexpr double tolerance = 32.0 * std::numeric_limits<double>::epsilon();
    const auto&      matrix    = surface.worldFromLocal;
    for (uint32_t a = 0; a < 3; ++a)
        for (uint32_t b = a; b < 3; ++b)
        {
            double product = 0.0;
            for (uint32_t row = 0; row < 3; ++row)
                product = std::fma(matrix[row * 4 + a] / scale, matrix[row * 4 + b] / scale, product);
            if (std::abs(product - (a == b ? 1.0 : 0.0)) > tolerance)
                return false;
        }
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        bounds.minimum[axis] = std::numeric_limits<double>::infinity();
        bounds.maximum[axis] = -std::numeric_limits<double>::infinity();
    }
    for (size_t vertex = 0; vertex < hull.vertices().size() / 3; ++vertex)
    {
        const Point local = readVertex(hull.vertices(), static_cast<uint32_t>(vertex));
        Point       world{};
        for (uint32_t row = 0; row < 3; ++row)
        {
            world[row] = std::fma(
                matrix[row * 4], local[0],
                std::fma(matrix[row * 4 + 1], local[1], std::fma(matrix[row * 4 + 2], local[2], matrix[row * 4 + 3])));
            if (!std::isfinite(world[row]))
                return false;
            bounds.minimum[row] = std::min(bounds.minimum[row], world[row]);
            bounds.maximum[row] = std::max(bounds.maximum[row], world[row]);
        }
    }
    return true;
}

ConvexConstructionOutcome prepareConvex(const ConvexGeometrySource&                    source,
                                        std::shared_ptr<const ExactGeometry>&          outGeometry,
                                        std::shared_ptr<const PreparedConvexHull>      preparedHull       = {},
                                        std::shared_ptr<const TriangleObjectGeometry>* outPreparedSurface = nullptr)
{
    if (source.vertices.size() < 12u || source.vertices.size() % 3u != 0u || source.faceVertexCounts.size() < 4u)
    {
        return ConvexConstructionOutcome::malformed;
    }
    size_t expectedIndexCount = 0u;
    for (uint32_t count : source.faceVertexCounts)
    {
        if (count < 3u || expectedIndexCount > std::numeric_limits<size_t>::max() - count)
        {
            return ConvexConstructionOutcome::malformed;
        }
        expectedIndexCount += count;
    }
    if (expectedIndexCount != source.faceVertexIndices.size())
    {
        return ConvexConstructionOutcome::malformed;
    }
    for (double value : source.vertices)
    {
        if (!std::isfinite(value))
        {
            return ConvexConstructionOutcome::malformed;
        }
    }

    std::vector<double> vertices;
    double              determinant = 0.0;
    if (!transformVertices(source, vertices, determinant) || determinant < 0.0)
    {
        return ConvexConstructionOutcome::malformed;
    }
    const size_t       vertexCount = vertices.size() / 3u;
    Point              minimum{};
    Point              maximum{};
    std::vector<Point> normalized;
    if (!normalizeAboutBounds(
            vertexCount, [&](size_t index) { return readVertex(vertices, static_cast<uint32_t>(index)); }, minimum,
            maximum, normalized))
    {
        return ConvexConstructionOutcome::malformed;
    }
    constexpr double kConstructionTolerance = 256.0 * std::numeric_limits<double>::epsilon();
    for (size_t first = 0u; first < vertexCount; ++first)
    {
        for (size_t second = first + 1u; second < vertexCount; ++second)
        {
            if (lengthSquared(subtract(normalized[first], normalized[second])) <=
                kConstructionTolerance * kConstructionTolerance)
            {
                return ConvexConstructionOutcome::malformed;
            }
        }
    }

    // A placement keeps orientation and connectivity. Keep the world-space numerical
    // checks below: finite transforms can still collapse faces.
    const std::vector<uint32_t>&                     orientedIndices = source.faceVertexIndices;
    const auto*                                      localContacts   = preparedHull && preparedHull->localGeometry()
                                                                           ? static_cast<const ConvexExactGeometry*>(preparedHull->localGeometry().get())
                                                                           : nullptr;
    const bool                                       reuseTopology   = localContacts != nullptr;
    std::map<std::pair<uint32_t, uint32_t>, EdgeUse> edgeUses;
    std::vector<Plane>                               planes;
    std::vector<uint32_t>                            planeTriangles;
    std::vector<uint32_t>                            triangles;
    // Each validated face has at least three indices, so twice the face count
    // cannot overflow or exceed expectedIndexCount.
    const size_t expectedTriangleCount = expectedIndexCount - 2u * source.faceVertexCounts.size();
    planes.reserve(source.faceVertexCounts.size());
    planeTriangles.reserve(source.faceVertexCounts.size());
    triangles.reserve(expectedTriangleCount * 3u);
    size_t indexOffset          = 0u;
    double signedVolumeTimesSix = 0.0;
    bool   nonConvex            = false;
    bool   planesAreFaces       = true;
    for (uint32_t count : source.faceVertexCounts)
    {
        if (!reuseTopology)
        {
            for (uint32_t corner = 0u; corner < count; ++corner)
            {
                const uint32_t first  = orientedIndices[indexOffset + corner];
                const uint32_t second = orientedIndices[indexOffset + (corner + 1u) % count];
                if (first >= vertexCount || second >= vertexCount || first == second)
                {
                    return ConvexConstructionOutcome::malformed;
                }
                const std::pair<uint32_t, uint32_t> key       = std::minmax(first, second);
                EdgeUse&                            use       = edgeUses[key];
                const bool                          direction = first < second;
                if (use.count == 0u)
                {
                    use.firstDirection = direction;
                }
                else if (use.count != 1u || use.firstDirection == direction)
                {
                    return ConvexConstructionOutcome::malformed;
                }
                ++use.count;
            }
        }

        const uint32_t firstIndex  = orientedIndices[indexOffset];
        const uint32_t secondIndex = orientedIndices[indexOffset + 1u];
        const uint32_t thirdIndex  = orientedIndices[indexOffset + 2u];
        Point          normal      = cross(subtract(normalized[secondIndex], normalized[firstIndex]),
                                           subtract(normalized[thirdIndex], normalized[firstIndex]));
        if (!normalize(normal))
        {
            return ConvexConstructionOutcome::malformed;
        }
        const double normalizedOffset = dot(normal, normalized[firstIndex]);
        for (uint32_t corner = 0u; corner < count; ++corner)
        {
            const uint32_t index    = orientedIndices[indexOffset + corner];
            const double   offPlane = std::abs(dot(normal, normalized[index]) - normalizedOffset);
            if (offPlane > kPolygonTolerance)
            {
                return ConvexConstructionOutcome::malformed;
            }
            planesAreFaces               = planesAreFaces && offPlane <= kConstructionTolerance;
            const uint32_t previousIndex = orientedIndices[indexOffset + (corner + count - 1u) % count];
            const uint32_t nextIndex     = orientedIndices[indexOffset + (corner + 1u) % count];
            const double   turn          = dot(cross(subtract(normalized[index], normalized[previousIndex]),
                                                     subtract(normalized[nextIndex], normalized[index])),
                                               normal);
            if (std::abs(turn) <= kConstructionTolerance)
            {
                return ConvexConstructionOutcome::malformed;
            }
            if (turn < 0.0)
            {
                nonConvex = true;
            }
        }
        for (const Point& point : normalized)
        {
            const double outside = dot(normal, point) - normalizedOffset;
            if (outside > kPolygonTolerance)
            {
                nonConvex = true;
            }
            planesAreFaces = planesAreFaces && outside <= kConstructionTolerance;
        }

        const Point worldFirst = readVertex(vertices, firstIndex);
        planes.push_back(Plane{normal, dot(normal, worldFirst)});
        planeTriangles.push_back(static_cast<uint32_t>(triangles.size() / 3u));
        for (uint32_t corner = 1u; corner + 1u < count; ++corner)
        {
            const uint32_t second = orientedIndices[indexOffset + corner];
            const uint32_t third  = orientedIndices[indexOffset + corner + 1u];
            triangles.push_back(firstIndex);
            triangles.push_back(second);
            triangles.push_back(third);
            signedVolumeTimesSix += dot(normalized[firstIndex], cross(normalized[second], normalized[third]));
        }
        indexOffset += count;
    }
    for (const auto& entry : edgeUses)
    {
        if (entry.second.count != 2u)
        {
            return ConvexConstructionOutcome::malformed;
        }
    }
    if (!std::isfinite(signedVolumeTimesSix) || signedVolumeTimesSix <= kConstructionTolerance)
    {
        return ConvexConstructionOutcome::malformed;
    }
    if (nonConvex)
    {
        return ConvexConstructionOutcome::nonConvex;
    }

    std::vector<Edge> edges;
    if (reuseTopology)
    {
        edges = localContacts->edges();
    }
    else
    {
        edges.reserve(edgeUses.size());
        for (const auto& entry : edgeUses)
        {
            edges.push_back(Edge{entry.first.first, entry.first.second});
        }
    }

    ovspatialquery_aabb_t bounds{};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        bounds.minimum[axis] = minimum[axis];
        bounds.maximum[axis] = maximum[axis];
    }

    std::shared_ptr<const TriangleObjectGeometry> surface;
    ovspatialquery_api_status_t                   surfaceStatus = OVSPATIALQUERY_API_SUCCESS;
    if (preparedHull != nullptr && preparedHull->preparedSurface() != nullptr)
    {
        surfaceStatus =
            bindPreparedTriangleGeometry(preparedHull->preparedSurface()->mesh, source.worldFromLocal, true, surface);
    }
    else
    {
        TriangleMeshSource mesh{};
        mesh.key       = 1u;
        mesh.vertices  = source.vertices;
        mesh.triangles = triangles;
        mesh.sourceFaces.assign(triangles.size() / 3u, 0u);
        std::shared_ptr<const PreparedTriangleMesh> prepared;
        surfaceStatus = prepareTriangleMesh(mesh, prepared);
        if (surfaceStatus == OVSPATIALQUERY_API_SUCCESS)
        {
            surfaceStatus = bindPreparedTriangleGeometry(prepared, source.worldFromLocal, true, surface);
        }
    }
    if (surfaceStatus == OVSPATIALQUERY_API_OUT_OF_MEMORY)
    {
        return ConvexConstructionOutcome::outOfMemory;
    }
    if (surfaceStatus == OVSPATIALQUERY_API_INVALID_ARGUMENT)
    {
        return ConvexConstructionOutcome::malformed;
    }
    if (surfaceStatus != OVSPATIALQUERY_API_SUCCESS || surface == nullptr)
    {
        return ConvexConstructionOutcome::internalFailure;
    }

    // Normalized coordinates divide by half the widest extent; twice the tolerance in them
    // bounds how far a face plane strays from the hull.
    double halfExtent = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
        halfExtent = std::max(halfExtent, 0.5 * (maximum[axis] - minimum[axis]));
    const double                         faceSlack = planesAreFaces ? 2.0 * kConstructionTolerance * halfExtent : -1.0;
    std::shared_ptr<const ExactGeometry> geometry  = std::make_shared<const ConvexExactGeometry>(
        std::move(vertices), std::move(triangles), std::move(planes), std::move(edges), surface, bounds,
        std::move(preparedHull), faceSlack, std::move(planeTriangles));
    if (outPreparedSurface != nullptr)
    {
        *outPreparedSurface = std::move(surface);
    }
    outGeometry = std::move(geometry);
    return ConvexConstructionOutcome::success;
}

ConvexConstructionOutcome prepareWorldConvex(const std::shared_ptr<const PreparedConvexHull>& preparedHull,
                                             const std::array<double, 16u>&                   worldFromLocal,
                                             std::shared_ptr<const ExactGeometry>&            outGeometry)
{
    return guardAllocation<ConvexConstructionOutcome>(
        [&]
        {
            ConvexGeometrySource source{};
            source.vertices.reserve(preparedHull->vertices().size());
            for (double value : preparedHull->vertices())
                source.vertices.push_back(static_cast<float>(value));
            source.faceVertexCounts  = preparedHull->faceVertexCounts();
            source.faceVertexIndices = preparedHull->faceVertexIndices();
            source.worldFromLocal    = worldFromLocal;
            return prepareConvex(source, outGeometry, preparedHull);
        });
}

// Shared by the budget charge taken before a hull exists and the hull's own count.
uint64_t preparedHullResidentBytes(const std::vector<double>& vertices, const std::vector<uint32_t>& faceVertexCounts,
                                   const std::vector<uint32_t>& faceVertexIndices, bool cooked,
                                   const ExactGeometry* localGeometry)
{
    uint64_t total = sizeof(PreparedConvexHull);
    total          = saturatingResidentByteAdd(total, static_cast<uint64_t>(vertices.capacity()), sizeof(double));
    total = saturatingResidentByteAdd(total, static_cast<uint64_t>(faceVertexCounts.capacity()), sizeof(uint32_t));
    total = saturatingResidentByteAdd(total, static_cast<uint64_t>(faceVertexIndices.capacity()), sizeof(uint32_t));
    if (cooked)
        total = saturatingResidentByteAdd(total, 1, 4096 + (vertices.size() / 3) * 64 + faceVertexCounts.size() * 64);
    return saturatingResidentByteAdd(total, 1u, localGeometry ? localGeometry->residentByteCount() : 0u);
}

} // namespace

PreparedConvexHull::PreparedConvexHull(std::vector<double> vertices, std::vector<uint32_t> faceVertexCounts,
                                       std::vector<uint32_t> faceVertexIndices, std::shared_ptr<void> residentCharge,
                                       std::shared_ptr<physx::PxConvexMesh>          cookedMesh,
                                       std::shared_ptr<const TriangleObjectGeometry> preparedSurface,
                                       std::shared_ptr<const ExactGeometry> localGeometry, CookedFrame cookedFrame)
    : mResidentCharge(std::move(residentCharge)), mVertices(std::move(vertices)),
      mFaceVertexCounts(std::move(faceVertexCounts)), mFaceVertexIndices(std::move(faceVertexIndices)),
      mCookedMesh(std::move(cookedMesh)), mPreparedSurface(std::move(preparedSurface)),
      mLocalGeometry(std::move(localGeometry)), mCookedFrame(cookedFrame)
{}

const std::vector<double>& PreparedConvexHull::vertices() const noexcept { return mVertices; }

const std::vector<uint32_t>& PreparedConvexHull::faceVertexCounts() const noexcept { return mFaceVertexCounts; }

const std::vector<uint32_t>& PreparedConvexHull::faceVertexIndices() const noexcept { return mFaceVertexIndices; }

uint64_t PreparedConvexHull::vertexCount() const noexcept { return static_cast<uint64_t>(mVertices.size() / 3u); }

uint64_t PreparedConvexHull::residentByteCount() const noexcept
{
    return preparedHullResidentBytes(mVertices, mFaceVertexCounts, mFaceVertexIndices, mCookedMesh != nullptr,
                                     mLocalGeometry.get());
}

uint64_t saturatingResidentByteAdd(uint64_t total, uint64_t elementCount, uint64_t elementSize) noexcept
{
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (elementCount != 0u && elementSize > maximum / elementCount)
    {
        return maximum;
    }
    const uint64_t bytes = elementCount * elementSize;
    return bytes > maximum - total ? maximum : total + bytes;
}

ConvexConstructionOutcome createConvexExactGeometry(const ConvexGeometrySource&           source,
                                                    std::shared_ptr<const ExactGeometry>& outGeometry)
{
    return guardAllocation<ConvexConstructionOutcome>([&] { return prepareConvex(source, outGeometry); });
}

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
void failNextPlacedConvexForTest() { gFailNextPlacedConvex.store(true); }
#endif

ConvexConstructionOutcome createConvexExactGeometry(const std::shared_ptr<const PreparedConvexHull>& preparedHull,
                                                    const std::array<double, 16u>&                   worldFromLocal,
                                                    std::shared_ptr<const ExactGeometry>&            outGeometry)
{
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    if (gFailNextPlacedConvex.exchange(false))
        return ConvexConstructionOutcome::outOfMemory;
#endif
    if (preparedHull == nullptr)
    {
        return ConvexConstructionOutcome::malformed;
    }
    return guardAllocation<ConvexConstructionOutcome>(
        [&]
        {
            if (preparedHull->localGeometry() && preparedHull->preparedSurface())
            {
                std::shared_ptr<const TriangleObjectGeometry> surface;
                const auto                                    status =
                    bindPreparedTriangleGeometry(preparedHull->preparedSurface()->mesh, worldFromLocal, true, surface);
                if (status == OVSPATIALQUERY_API_OUT_OF_MEMORY)
                    return ConvexConstructionOutcome::outOfMemory;
                if (status != OVSPATIALQUERY_API_SUCCESS)
                    return ConvexConstructionOutcome::malformed;
                const double          scale = surface->uniformScale;
                ovspatialquery_aabb_t bounds{};
                if (placedConvexBounds(*preparedHull, *surface, scale, bounds))
                {
                    outGeometry = std::make_shared<const PlacedConvexExactGeometry>(preparedHull, std::move(surface),
                                                                                    scale, bounds);
                    return ConvexConstructionOutcome::success;
                }
            }
            return prepareWorldConvex(preparedHull, worldFromLocal, outGeometry);
        });
}

namespace
{

/* createPreparedConvexHull's body. It moves the arrays and the cooked mesh into the hull only on
 * success, so after topologyRejected the caller still holds them and can split the hull's
 * polygons and try again without having copied them first (ADR-0076). */
ConvexHullPreparationOutcome createPreparedConvexHullKeepingInputs(
    std::vector<double>& vertices, std::vector<uint32_t>& faceVertexCounts, std::vector<uint32_t>& faceVertexIndices,
    std::shared_ptr<const PreparedConvexHull>& outHull, const ConvexHullPreparationBudget* budget,
    std::shared_ptr<physx::PxConvexMesh>& cookedMesh, CookedFrame cookedFrame)
{
    return guardAllocation<ConvexHullPreparationOutcome>(
        [&]
        {
            ConvexGeometrySource validationSource{};
            validationSource.vertices.reserve(vertices.size());
            for (double value : vertices)
                validationSource.vertices.push_back(static_cast<float>(value));
            validationSource.faceVertexCounts  = faceVertexCounts;
            validationSource.faceVertexIndices = faceVertexIndices;
            validationSource.worldFromLocal    = {
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            };
            std::shared_ptr<const ExactGeometry>          validationGeometry;
            std::shared_ptr<const TriangleObjectGeometry> preparedSurface;
            const ConvexConstructionOutcome               validation =
                prepareConvex(validationSource, validationGeometry, {}, &preparedSurface);
            if (validation == ConvexConstructionOutcome::outOfMemory)
            {
                return ConvexHullPreparationOutcome::outOfMemory;
            }
            if (validation == ConvexConstructionOutcome::malformed ||
                validation == ConvexConstructionOutcome::nonConvex)
            {
                return ConvexHullPreparationOutcome::topologyRejected;
            }
            if (validation != ConvexConstructionOutcome::success)
            {
                return ConvexHullPreparationOutcome::internalFailure;
            }
            std::shared_ptr<void> residentCharge;
            if (budget != nullptr)
            {
                const uint64_t residentBytes = preparedHullResidentBytes(
                    vertices, faceVertexCounts, faceVertexIndices, cookedMesh != nullptr, validationGeometry.get());
                const ConvexHullPreparationOutcome chargeOutcome =
                    budget->acquire(budget->state, residentBytes, residentCharge);
                if (chargeOutcome != ConvexHullPreparationOutcome::success)
                {
                    return chargeOutcome;
                }
            }
            std::shared_ptr<const PreparedConvexHull> candidate = std::make_shared<const PreparedConvexHull>(
                std::move(vertices), std::move(faceVertexCounts), std::move(faceVertexIndices),
                std::move(residentCharge), std::move(cookedMesh), std::move(preparedSurface),
                std::move(validationGeometry), cookedFrame);
            outHull = std::move(candidate);
            return ConvexHullPreparationOutcome::success;
        });
}

} // namespace

ConvexHullPreparationOutcome createPreparedConvexHull(
    std::vector<double> vertices, std::vector<uint32_t> faceVertexCounts, std::vector<uint32_t> faceVertexIndices,
    std::shared_ptr<const PreparedConvexHull>& outHull, const ConvexHullPreparationBudget* budget,
    std::shared_ptr<physx::PxConvexMesh> cookedMesh, CookedFrame cookedFrame)
{
    return createPreparedConvexHullKeepingInputs(vertices, faceVertexCounts, faceVertexIndices, outHull, budget,
                                                 cookedMesh, cookedFrame);
}

namespace
{

ConvexHullPreparationOutcome commitPreparedConvexHull(const std::shared_ptr<const PreparedConvexHull>& hull,
                                                      uint64_t                                   maximumResidentBytes,
                                                      std::shared_ptr<const PreparedConvexHull>& outHull)
{
    if (hull->residentByteCount() > maximumResidentBytes)
    {
        return ConvexHullPreparationOutcome::resourceExhausted;
    }
    outHull = hull;
    return ConvexHullPreparationOutcome::success;
}

/* How far the farthest authored point lies outside the hull's face planes, as a fraction of the
 * points' largest half extent; zero when every point is enclosed. The hull encloses its points
 * when that is at most the call's enclosure tolerance, 2^-20 by default, as far as hull
 * validation lets the hull's own vertices lie (ADR-0030, ADR-0078). PhysX's quickhull takes a
 * point within its plane thickness of a face, 3 float epsilons of half the sum of the cooked
 * extents, as on the face and leaves it out of the hull. Float32 points in a face's plane, as on
 * a box rotated off the axes, lie off it by a few float steps, so PhysX leaves some of them
 * outside, and an exact test refused their hull. */
ConvexHullPreparationOutcome measurePreparedConvexHullEnclosure(const PreparedConvexHull& preparedHull,
                                                                const ConvexPointSet&     points,
                                                                double&                   outFarthestOutside)
{
    outFarthestOutside = 0.0;
    ExactGeometryDescription description;
    if (!preparedHull.localGeometry() || !preparedHull.localGeometry()->exportDescription(description) ||
        !std::holds_alternative<ConvexExactGeometryDescription>(description))
        return ConvexHullPreparationOutcome::internalFailure;
    const auto& convex     = std::get<ConvexExactGeometryDescription>(description);
    double      halfExtent = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
        halfExtent = std::max(halfExtent, 0.5 * (points.maximum[axis] - points.minimum[axis]));
    std::vector<Point> vertices(preparedHull.vertexCount());
    for (uint32_t vertex = 0u; vertex < vertices.size(); ++vertex)
        vertices[vertex] = readVertex(preparedHull.vertices(), vertex);
    std::sort(vertices.begin(), vertices.end());
    double farthest = 0.0;
    for (const Point& point : points.authored)
    {
        if (std::binary_search(vertices.begin(), vertices.end(), point))
            continue;
        for (uint64_t plane = 0u; plane < convex.planeCount; ++plane)
            farthest = std::max(farthest, dot(convex.planes[plane].normal, point) - convex.planes[plane].offset);
    }
    outFarthestOutside = farthest / halfExtent;
    return ConvexHullPreparationOutcome::success;
}

} // namespace

ConvexHullPreparationOutcome prepareConvexHullFrom(const std::vector<float>&                  authoredPoints,
                                                   uint64_t                                   maximumResidentBytes,
                                                   std::shared_ptr<const PreparedConvexHull>& outHull,
                                                   const ConvexHullPreparationBudget*         budget,
                                                   const ConvexHullBuilder&                   build,
                                                   const ConvexHullCookingOptions& options, double* outFarthestOutside)
{
    if (outFarthestOutside != nullptr)
        *outFarthestOutside = 0.0;
    return guardAllocation<ConvexHullPreparationOutcome>(
        [&]
        {
            ConvexPointSet pointSet{};
            if (!qualifyPointSet(authoredPoints, pointSet))
                return ConvexHullPreparationOutcome::invalidArgument;
            std::vector<double>                       vertices;
            std::vector<uint32_t>                     faceVertexCounts;
            std::vector<uint32_t>                     faceVertexIndices;
            std::shared_ptr<physx::PxConvexMesh>      cookedMesh;
            CookedFrame                               cookedFrame;
            std::shared_ptr<const PreparedConvexHull> hull;
            const ConvexHullPreparationOutcome built = build(pointSet, options.vertexLimit, vertices, faceVertexCounts,
                                                             faceVertexIndices, cookedMesh, cookedFrame);
            if (built == ConvexHullPreparationOutcome::topologyRejected)
                return ConvexHullPreparationOutcome::noEnclosingHull;
            if (built != ConvexHullPreparationOutcome::success)
                return built;
            ConvexHullPreparationOutcome outcome = createPreparedConvexHullKeepingInputs(
                vertices, faceVertexCounts, faceVertexIndices, hull, budget, cookedMesh, cookedFrame);
            // A polygon whose corners lie off one plane beyond ADR-0030's tolerance is split into
            // triangles of its corners (ADR-0076). The split hull keeps no cooked mesh, whose
            // polygon planes stay off the corners, so every reader uses the triangles' planes.
            if (outcome == ConvexHullPreparationOutcome::topologyRejected &&
                splitNonplanarPolygons(vertices, faceVertexCounts, faceVertexIndices))
                outcome = createPreparedConvexHull(std::move(vertices), std::move(faceVertexCounts),
                                                   std::move(faceVertexIndices), hull, budget);
            if (outcome == ConvexHullPreparationOutcome::topologyRejected)
                return ConvexHullPreparationOutcome::noEnclosingHull;
            if (outcome != ConvexHullPreparationOutcome::success)
                return outcome;
            // Under a vertex limit the hull is PhysX's, which may leave points outside (ADR-0078).
            if (options.vertexLimit != 0u)
                return commitPreparedConvexHull(hull, maximumResidentBytes, outHull);
            double farthestOutside = 0.0;
            outcome                = measurePreparedConvexHullEnclosure(*hull, pointSet, farthestOutside);
            if (outcome != ConvexHullPreparationOutcome::success)
                return outcome;
            // A hull that leaves a point outside is refused, never replaced by another shape: a
            // backend that stopped at the vertex limit reached it (ADR-0074); any other hull
            // means the points have no hull of their own within the tolerance (ADR-0076).
            if (farthestOutside > options.enclosureTolerance)
            {
                if (outFarthestOutside != nullptr)
                    *outFarthestOutside = farthestOutside;
                const bool reachedVertexLimit = pointSet.authored.size() > kMaximumConvexHullVertices &&
                                                hull->vertexCount() >= kMaximumConvexHullVertices;
                return reachedVertexLimit ? ConvexHullPreparationOutcome::beyondVertexLimit
                                          : ConvexHullPreparationOutcome::noEnclosingHull;
            }
            return commitPreparedConvexHull(hull, maximumResidentBytes, outHull);
        });
}

} // namespace detail
} // namespace ovspatialquery
