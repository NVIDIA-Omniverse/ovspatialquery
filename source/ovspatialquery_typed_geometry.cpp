/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_typed_geometry.h"

#include "ovspatialquery_box_sweep_math.h"
#include "ovspatialquery_float_bounds.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory_resource>
#include <new>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace ovspatialquery
{
namespace detail
{

static_assert(std::is_nothrow_copy_assignable<ExactGeometryDescription>::value,
              "Exact geometry description export must remain allocation-free");

struct ExactGeometryResidentAccumulator::Impl
{
    // Deduplication nodes live only for this accounting pass. Destroy owners
    // before its arena so all temporary blocks are released together.
    std::pmr::monotonic_buffer_resource  scratch;
    std::pmr::unordered_set<const void*> owners;
    uint64_t                             residentBytes = 0u;
    bool                                 failed        = false;

    Impl() : owners(&scratch) {}
};

ExactGeometryResidentAccumulator::ExactGeometryResidentAccumulator() : mImpl(new Impl()) {}

ExactGeometryResidentAccumulator::~ExactGeometryResidentAccumulator() = default;

bool ExactGeometryResidentAccumulator::add(const void* owner, uint64_t bytes) noexcept
{
    if (mImpl == nullptr || owner == nullptr || mImpl->failed)
    {
        return false;
    }
    try
    {
        if (!mImpl->owners.insert(owner).second)
        {
            return true;
        }
    }
    catch (...)
    {
        mImpl->failed        = true;
        mImpl->residentBytes = std::numeric_limits<uint64_t>::max();
        return false;
    }
    if (bytes > std::numeric_limits<uint64_t>::max() - mImpl->residentBytes)
    {
        mImpl->failed        = true;
        mImpl->residentBytes = std::numeric_limits<uint64_t>::max();
        return false;
    }
    mImpl->residentBytes += bytes;
    return true;
}

uint64_t ExactGeometryResidentAccumulator::residentByteCount() const noexcept
{
    return mImpl == nullptr ? std::numeric_limits<uint64_t>::max() : mImpl->residentBytes;
}

namespace
{

box_sweep::Vec3 vec(const std::array<double, 3u>& v) { return {v[0], v[1], v[2]}; }

std::array<double, 3u> toArray(const box_sweep::Vec3& v) { return {v.x, v.y, v.z}; }

double dot(const std::array<double, 3u>& first, const std::array<double, 3u>& second)
{
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

std::array<double, 3u> subtract(const std::array<double, 3u>& first, const std::array<double, 3u>& second)
{
    return {
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    };
}

std::array<double, 3u> boxLocalPoint(const BoxGeometrySource& box, const std::array<double, 3u>& worldPoint)
{
    const std::array<double, 3u> delta = subtract(worldPoint, box.center);
    return {
        dot(delta, box.axes[0]),
        dot(delta, box.axes[1]),
        dot(delta, box.axes[2]),
    };
}

std::array<double, 3u> boxLocalVector(const BoxGeometrySource& box, const std::array<double, 3u>& worldVector)
{
    return {
        dot(worldVector, box.axes[0]),
        dot(worldVector, box.axes[1]),
        dot(worldVector, box.axes[2]),
    };
}

std::array<double, 3u> boxWorldPoint(const BoxGeometrySource& box, const std::array<double, 3u>& localPoint)
{
    std::array<double, 3u> result = box.center;
    for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
    {
        for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
        {
            result[worldAxis] = std::fma(box.axes[localAxis][worldAxis], localPoint[localAxis], result[worldAxis]);
        }
    }
    return result;
}

void setBoxPathHit(const BoxGeometrySource& box, const std::array<double, 3u>& localOrigin,
                   const std::array<double, 3u>& localDirection, double distance, ExactGeometryHit& outHit)
{
    const box_sweep::Vec3 point = box_sweep::at(vec(localOrigin), vec(localDirection), distance);
    box_sweep::Vec3       closest{};
    const double          distanceSquared   = box_sweep::pointBoxDistanceSquared(point, vec(box.halfExtents), &closest);
    const std::array<double, 3u> localPoint = toArray(point);
    std::array<double, 3u>       localClosest = toArray(closest);
    std::array<double, 3u>       localNormal{};
    if (distanceSquared > 0.0)
    {
        const double inverseLength = 1.0 / std::sqrt(distanceSquared);
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            localNormal[axis] = (localPoint[axis] - localClosest[axis]) * inverseLength;
        }
    }
    else
    {
        uint32_t nearestAxis = 0u;
        double   nearestFace = std::numeric_limits<double>::max();
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            const double faceDistance = std::abs(box.halfExtents[axis] - std::abs(localPoint[axis]));
            if (faceDistance < nearestFace)
            {
                nearestFace = faceDistance;
                nearestAxis = axis;
            }
        }
        localNormal[nearestAxis]  = localPoint[nearestAxis] < 0.0 ? -1.0 : 1.0;
        localClosest[nearestAxis] = localNormal[nearestAxis] * box.halfExtents[nearestAxis];
    }
    const std::array<double, 3u> worldPoint = boxWorldPoint(box, localClosest);
    ExactGeometryHit             hit{};
    hit.distance = distance;
    for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
    {
        double worldNormal = 0.0;
        for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
        {
            worldNormal = std::fma(box.axes[localAxis][worldAxis], localNormal[localAxis], worldNormal);
        }
        hit.position[worldAxis] = worldPoint[worldAxis];
        hit.normal[worldAxis]   = static_cast<float>(worldNormal);
    }
    outHit = hit;
}

} // namespace

BoxGeometrySource makeQueryBoxGeometry(const ovspatialquery_overlap_box_t& source, const double (*doubleRotation)[3])
{
    double quaternionRotation[3][3];
    if (doubleRotation == nullptr)
    {
        const double quaternion[4] = {
            source.rotation_xyzw[0],
            source.rotation_xyzw[1],
            source.rotation_xyzw[2],
            source.rotation_xyzw[3],
        };
        const double inverseLength = 1.0 / std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                                                     quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
        quaternionRotationMatrix(quaternion, inverseLength, quaternionRotation);
    }
    const double(*rotation)[3] = doubleRotation != nullptr ? doubleRotation : quaternionRotation;
    BoxGeometrySource box{};
    box.center = {source.center[0], source.center[1], source.center[2]};
    // axes[k] is the k-th column of the rotation matrix.
    box.axes = {
        std::array<double, 3u>{rotation[0][0], rotation[1][0], rotation[2][0]},
        std::array<double, 3u>{rotation[0][1], rotation[1][1], rotation[2][1]},
        std::array<double, 3u>{rotation[0][2], rotation[1][2], rotation[2][2]},
    };
    box.halfExtents = {
        source.half_extents[0],
        source.half_extents[1],
        source.half_extents[2],
    };
    return box;
}

bool queryBoxGeometry(const ovspatialquery_overlap_shape_t& shape, const double (*rotation)[3],
                      BoxGeometrySource&                    outBox)
{
    if (shape.kind == OVSPATIALQUERY_OVERLAP_BOX)
        outBox = makeQueryBoxGeometry(shape.data.box, rotation);
    else if (shape.kind == OVSPATIALQUERY_OVERLAP_AABB)
        outBox = makeQueryBoxGeometry(shape.data.aabb);
    else
        return false;
    return true;
}

BoxGeometrySource makeQueryBoxGeometry(const ovspatialquery_overlap_aabb_t& source)
{
    BoxGeometrySource box{};
    box.center = {
        source.min_corner[0] * 0.5 + source.max_corner[0] * 0.5,
        source.min_corner[1] * 0.5 + source.max_corner[1] * 0.5,
        source.min_corner[2] * 0.5 + source.max_corner[2] * 0.5,
    };
    box.axes = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    box.halfExtents = {
        source.max_corner[0] * 0.5 - source.min_corner[0] * 0.5,
        source.max_corner[1] * 0.5 - source.min_corner[1] * 0.5,
        source.max_corner[2] * 0.5 - source.min_corner[2] * 0.5,
    };
    return box;
}

namespace
{

bool boxesOverlap(const BoxGeometrySource& first, const BoxGeometrySource& second)
{
    double rotation[3][3]         = {};
    double absoluteRotation[3][3] = {};
    for (uint32_t firstAxis = 0u; firstAxis < 3u; ++firstAxis)
    {
        for (uint32_t secondAxis = 0u; secondAxis < 3u; ++secondAxis)
        {
            rotation[firstAxis][secondAxis]         = dot(first.axes[firstAxis], second.axes[secondAxis]);
            absoluteRotation[firstAxis][secondAxis] = std::abs(rotation[firstAxis][secondAxis]) + 1.0e-15;
        }
    }
    const std::array<double, 3u> worldDelta  = subtract(second.center, first.center);
    const std::array<double, 3u> translation = {
        dot(worldDelta, first.axes[0]),
        dot(worldDelta, first.axes[1]),
        dot(worldDelta, first.axes[2]),
    };
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        double secondRadius = 0.0;
        for (uint32_t secondAxis = 0u; secondAxis < 3u; ++secondAxis)
        {
            secondRadius = std::fma(second.halfExtents[secondAxis], absoluteRotation[axis][secondAxis], secondRadius);
        }
        if (std::abs(translation[axis]) > first.halfExtents[axis] + secondRadius)
        {
            return false;
        }
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        double firstRadius          = 0.0;
        double projectedTranslation = 0.0;
        for (uint32_t firstAxis = 0u; firstAxis < 3u; ++firstAxis)
        {
            firstRadius = std::fma(first.halfExtents[firstAxis], absoluteRotation[firstAxis][axis], firstRadius);
            projectedTranslation = std::fma(translation[firstAxis], rotation[firstAxis][axis], projectedTranslation);
        }
        if (std::abs(projectedTranslation) > firstRadius + second.halfExtents[axis])
        {
            return false;
        }
    }
    for (uint32_t firstAxis = 0u; firstAxis < 3u; ++firstAxis)
    {
        const uint32_t firstNext = (firstAxis + 1u) % 3u;
        const uint32_t firstLast = (firstAxis + 2u) % 3u;
        for (uint32_t secondAxis = 0u; secondAxis < 3u; ++secondAxis)
        {
            const uint32_t secondNext  = (secondAxis + 1u) % 3u;
            const uint32_t secondLast  = (secondAxis + 2u) % 3u;
            const double   firstRadius = first.halfExtents[firstNext] * absoluteRotation[firstLast][secondAxis] +
                                       first.halfExtents[firstLast] * absoluteRotation[firstNext][secondAxis];
            const double secondRadius = second.halfExtents[secondNext] * absoluteRotation[firstAxis][secondLast] +
                                        second.halfExtents[secondLast] * absoluteRotation[firstAxis][secondNext];
            const double projectedTranslation = std::abs(translation[firstLast] * rotation[firstNext][secondAxis] -
                                                         translation[firstNext] * rotation[firstLast][secondAxis]);
            if (projectedTranslation > firstRadius + secondRadius)
            {
                return false;
            }
        }
    }
    return true;
}

void setCapsulePathHit(const CapsuleGeometrySource& capsule, const std::array<double, 3u>& origin,
                       const std::array<double, 3u>& direction, double distance, ExactGeometryHit& outHit)
{
    const std::array<double, 3u> movingCenter = {
        std::fma(direction[0], distance, origin[0]),
        std::fma(direction[1], distance, origin[1]),
        std::fma(direction[2], distance, origin[2]),
    };
    box_sweep::Vec3              closest{};
    const std::array<double, 3u> normal = toArray(box_sweep::capsuleContactNormal(
        vec(capsule.endpointA), vec(capsule.endpointB), vec(movingCenter), vec(direction), closest));
    const std::array<double, 3u> foot   = toArray(closest);
    ExactGeometryHit             hit{};
    hit.distance = distance;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        hit.normal[axis]   = static_cast<float>(normal[axis]);
        hit.position[axis] = std::fma(normal[axis], capsule.radius, foot[axis]);
    }
    outHit = hit;
}

// Distances are path parameters: the center moves to origin + t d with the caller's float
// direction as given, whose squared length is one only to within 1e-5. The roots of
// |d|^2 t^2 + 2 b t + c therefore keep |d|^2; taking it as one leaves the sphere about
// (|d|^2 - 1) t^2 off touching at the root. The half chord comes from the path's closest
// approach, as for capsules and the triangle entries, because the textbook discriminant
// loses about eps L^2 on long paths. A sphere is solid, so a path already touching or inside
// it at the minimum reports the minimum, as the header's distance-interval rule states; that
// also absorbs an entry rounded behind a start on the surface.
bool spherePathHit(const std::array<double, 3u>& targetCenter, double targetRadius, const double origin[3],
                   double expansionRadius, const float unitDirection[3], double minimumDistance, double maximumDistance,
                   ExactGeometryHit& outHit)
{
    const double expandedRadius = targetRadius + expansionRadius;
    double       offset[3]      = {};
    double       direction[3]   = {};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        offset[axis]    = origin[axis] - targetCenter[axis];
        direction[axis] = static_cast<double>(unitDirection[axis]);
    }
    const double inverseDirectionSquare = 1.0 / dot3(direction, direction);
    const double closest                = -dot3(offset, direction) * inverseDirectionSquare;
    double       gapSquared             = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        const double gap = std::fma(direction[axis], closest, offset[axis]);
        gapSquared += gap * gap;
    }
    const double halfChordSquared = (expandedRadius * expandedRadius - gapSquared) * inverseDirectionSquare;
    if (!(halfChordSquared >= 0.0))
    {
        return false;
    }
    const double halfChord = std::sqrt(halfChordSquared);
    if (!(closest + halfChord >= minimumDistance))
    {
        return false;
    }
    const double distance = std::max(minimumDistance, closest - halfChord);
    if (!std::isfinite(distance) || distance > maximumDistance)
    {
        return false;
    }
    double movingCenter[3]     = {};
    double normalLengthSquared = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        movingCenter[axis]     = std::fma(direction[axis], distance, origin[axis]);
        const double component = movingCenter[axis] - targetCenter[axis];
        normalLengthSquared    = std::fma(component, component, normalLengthSquared);
    }
    const double     normalLength = std::sqrt(normalLengthSquared);
    ExactGeometryHit hit{};
    hit.distance = distance;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        const double normal =
            normalLength == 0.0 ? -direction[axis] : (movingCenter[axis] - targetCenter[axis]) / normalLength;
        hit.normal[axis]   = static_cast<float>(normal);
        hit.position[axis] = std::fma(normal, targetRadius, targetCenter[axis]);
    }
    outHit = hit;
    return true;
}

class SphereExactGeometry final : public ExactGeometry
{
  public:
    explicit SphereExactGeometry(const SphereGeometrySource& source) : mSource(source)
    {
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            mBounds.minimum[axis] = mSource.center[axis] - mSource.radius;
            mBounds.maximum[axis] = mSource.center[axis] + mSource.radius;
        }
    }

    ExactGeometryFamily family() const noexcept override { return ExactGeometryFamily::sphere; }

    bool exportDescription(ExactGeometryDescription& outDescription) const noexcept override
    {
        outDescription = mSource;
        return true;
    }

    const ovspatialquery_aabb_t& worldBounds() const noexcept override { return mBounds; }

    bool raycast(const double origin[3], const float unitDirection[3], double minimumDistance, double maximumDistance,
                 bool, ExactGeometryHit& outHit, InstanceHostLedger*, ovspatialquery_api_status_t*, bool) const override
    {
        return spherePathHit(mSource.center, mSource.radius, origin, 0.0, unitDirection, minimumDistance,
                             maximumDistance, outHit);
    }

    bool sweepSphere(const double center[3], double radius, const float unitDirection[3], double minimumDistance,
                     double maximumDistance, ExactGeometryHit& outHit, InstanceHostLedger*,
                     ovspatialquery_api_status_t*, bool, bool) const override
    {
        return spherePathHit(mSource.center, mSource.radius, center, radius, unitDirection, minimumDistance,
                             maximumDistance, outHit);
    }

    bool overlaps(const ovspatialquery_overlap_shape_t& geometry, const double (*boxRotation)[3], InstanceHostLedger*,
                  ovspatialquery_api_status_t*) const override
    {
        if (geometry.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
        {
            double distanceSquared = 0.0;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                const double delta = geometry.data.sphere.center[axis] - mSource.center[axis];
                distanceSquared    = std::fma(delta, delta, distanceSquared);
            }
            const double radius = mSource.radius + geometry.data.sphere.radius;
            return distanceSquared <= radius * radius;
        }
        BoxGeometrySource box{};
        if (!queryBoxGeometry(geometry, boxRotation, box))
            return false;
        const std::array<double, 3u> localCenter = boxLocalPoint(box, mSource.center);
        return box_sweep::pointBoxDistanceSquared(vec(localCenter), vec(box.halfExtents), nullptr) <=
               mSource.radius * mSource.radius;
    }

    uint64_t residentByteCount() const noexcept override { return sizeof(SphereExactGeometry); }

    void accumulateResidentAllocations(ExactGeometryResidentAccumulator& accumulator, bool) const noexcept override
    {
        (void)accumulator.add(this, sizeof(SphereExactGeometry));
    }

  private:
    const SphereGeometrySource mSource;
    ovspatialquery_aabb_t      mBounds{};
};

class BoxExactGeometry final : public ExactGeometry
{
  public:
    explicit BoxExactGeometry(const BoxGeometrySource& source) : mSource(source)
    {
        for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
        {
            double radius = 0.0;
            for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
            {
                radius = std::fma(std::abs(mSource.axes[localAxis][worldAxis]), mSource.halfExtents[localAxis], radius);
            }
            mBounds.minimum[worldAxis] = mSource.center[worldAxis] - radius;
            mBounds.maximum[worldAxis] = mSource.center[worldAxis] + radius;
        }
    }

    ExactGeometryFamily family() const noexcept override { return ExactGeometryFamily::box; }

    bool exportDescription(ExactGeometryDescription& outDescription) const noexcept override
    {
        outDescription = mSource;
        return true;
    }

    const ovspatialquery_aabb_t& worldBounds() const noexcept override { return mBounds; }

    bool raycast(const double origin[3], const float unitDirection[3], double minimumDistance, double maximumDistance,
                 bool, ExactGeometryHit& outHit, InstanceHostLedger*, ovspatialquery_api_status_t*, bool) const override
    {
        return tracePath(origin, 0.0, unitDirection, minimumDistance, maximumDistance, outHit);
    }

    bool sweepSphere(const double center[3], double radius, const float unitDirection[3], double minimumDistance,
                     double maximumDistance, ExactGeometryHit& outHit, InstanceHostLedger*,
                     ovspatialquery_api_status_t*, bool, bool) const override
    {
        return tracePath(center, radius, unitDirection, minimumDistance, maximumDistance, outHit);
    }

    bool overlaps(const ovspatialquery_overlap_shape_t& geometry, const double (*boxRotation)[3], InstanceHostLedger*,
                  ovspatialquery_api_status_t*) const override
    {
        if (geometry.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
        {
            const std::array<double, 3u> center = {
                geometry.data.sphere.center[0],
                geometry.data.sphere.center[1],
                geometry.data.sphere.center[2],
            };
            const std::array<double, 3u> localCenter = boxLocalPoint(mSource, center);
            const double                 radius      = geometry.data.sphere.radius;
            return box_sweep::pointBoxDistanceSquared(vec(localCenter), vec(mSource.halfExtents), nullptr) <=
                   radius * radius;
        }
        BoxGeometrySource other{};
        return queryBoxGeometry(geometry, boxRotation, other) && boxesOverlap(mSource, other);
    }

    uint64_t residentByteCount() const noexcept override { return sizeof(BoxExactGeometry); }

    void accumulateResidentAllocations(ExactGeometryResidentAccumulator& accumulator, bool) const noexcept override
    {
        (void)accumulator.add(this, sizeof(BoxExactGeometry));
    }

  private:
    bool tracePath(const double origin[3], double radius, const float unitDirection[3], double minimumDistance,
                   double maximumDistance, ExactGeometryHit& outHit) const
    {
        const std::array<double, 3u> worldOrigin    = {origin[0], origin[1], origin[2]};
        const std::array<double, 3u> worldDirection = {unitDirection[0], unitDirection[1], unitDirection[2]};
        const std::array<double, 3u> localOrigin    = boxLocalPoint(mSource, worldOrigin);
        const std::array<double, 3u> localDirection = boxLocalVector(mSource, worldDirection);
        double                       distance       = 0.0;
        if (!box_sweep::roundedBoxPath(vec(localOrigin), vec(localDirection), vec(mSource.halfExtents), radius,
                                       minimumDistance, maximumDistance, distance))
        {
            return false;
        }
        setBoxPathHit(mSource, localOrigin, localDirection, distance, outHit);
        return true;
    }

    const BoxGeometrySource mSource;
    ovspatialquery_aabb_t   mBounds{};
};

class CapsuleExactGeometry final : public ExactGeometry
{
  public:
    explicit CapsuleExactGeometry(const CapsuleGeometrySource& source) : mSource(source)
    {
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            mBounds.minimum[axis] = std::min(mSource.endpointA[axis], mSource.endpointB[axis]) - mSource.radius;
            mBounds.maximum[axis] = std::max(mSource.endpointA[axis], mSource.endpointB[axis]) + mSource.radius;
        }
    }

    ExactGeometryFamily family() const noexcept override { return ExactGeometryFamily::capsule; }

    bool exportDescription(ExactGeometryDescription& outDescription) const noexcept override
    {
        outDescription = mSource;
        return true;
    }

    const ovspatialquery_aabb_t& worldBounds() const noexcept override { return mBounds; }

    bool raycast(const double origin[3], const float unitDirection[3], double minimumDistance, double maximumDistance,
                 bool, ExactGeometryHit& outHit, InstanceHostLedger*, ovspatialquery_api_status_t*, bool) const override
    {
        return tracePath(origin, 0.0, unitDirection, minimumDistance, maximumDistance, outHit);
    }

    bool sweepSphere(const double center[3], double radius, const float unitDirection[3], double minimumDistance,
                     double maximumDistance, ExactGeometryHit& outHit, InstanceHostLedger*,
                     ovspatialquery_api_status_t*, bool, bool) const override
    {
        return tracePath(center, radius, unitDirection, minimumDistance, maximumDistance, outHit);
    }

    bool overlaps(const ovspatialquery_overlap_shape_t& geometry, const double (*boxRotation)[3], InstanceHostLedger*,
                  ovspatialquery_api_status_t*) const override
    {
        if (geometry.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
        {
            const std::array<double, 3u> center = {
                geometry.data.sphere.center[0],
                geometry.data.sphere.center[1],
                geometry.data.sphere.center[2],
            };
            const double radius = mSource.radius + geometry.data.sphere.radius;
            return box_sweep::segmentDistanceSquared(vec(mSource.endpointA), vec(mSource.endpointB), vec(center)) <=
                   radius * radius;
        }
        BoxGeometrySource box{};
        if (!queryBoxGeometry(geometry, boxRotation, box))
            return false;
        const std::array<double, 3u> localA = boxLocalPoint(box, mSource.endpointA);
        const std::array<double, 3u> localB = boxLocalPoint(box, mSource.endpointB);
        return box_sweep::segmentBoxDistanceSquared(vec(localA), vec(localB), vec(box.halfExtents)) <=
               mSource.radius * mSource.radius;
    }

    uint64_t residentByteCount() const noexcept override { return sizeof(CapsuleExactGeometry); }

    void accumulateResidentAllocations(ExactGeometryResidentAccumulator& accumulator, bool) const noexcept override
    {
        (void)accumulator.add(this, sizeof(CapsuleExactGeometry));
    }

  private:
    bool tracePath(const double origin[3], double expansionRadius, const float unitDirection[3], double minimumDistance,
                   double maximumDistance, ExactGeometryHit& outHit) const
    {
        const std::array<double, 3u> worldOrigin    = {origin[0], origin[1], origin[2]};
        const std::array<double, 3u> worldDirection = {unitDirection[0], unitDirection[1], unitDirection[2]};
        double                       distance       = 0.0;
        if (!box_sweep::capsulePath(vec(mSource.endpointA), vec(mSource.endpointB), vec(worldOrigin),
                                    vec(worldDirection), mSource.radius + expansionRadius, minimumDistance,
                                    maximumDistance, distance))
        {
            return false;
        }
        setCapsulePathHit(mSource, worldOrigin, worldDirection, distance, outHit);
        return true;
    }

    const CapsuleGeometrySource mSource;
    ovspatialquery_aabb_t       mBounds{};
};

class TriangleMeshExactGeometry final : public ExactGeometry
{
  public:
    TriangleMeshExactGeometry(std::shared_ptr<const TriangleObjectGeometry> geometry) : mGeometry(std::move(geometry))
    {}

    ExactGeometryFamily family() const noexcept override { return ExactGeometryFamily::triangleMesh; }

    bool exportDescription(ExactGeometryDescription& outDescription) const noexcept override
    {
        TriangleMeshExactGeometryDescription description{};
        if (!exportTriangleGeometryDescription(*mGeometry, description))
        {
            return false;
        }
        outDescription = description;
        return true;
    }

    const ovspatialquery_aabb_t& worldBounds() const noexcept override { return mGeometry->worldBounds; }

    bool raycast(const double origin[3], const float direction[3], double minimumDistance, double maximumDistance,
                 bool anyHit, ExactGeometryHit& outHit, InstanceHostLedger* ledger,
                 ovspatialquery_api_status_t* outStatus, bool bothSides) const override
    {
        bool               hasHit = false;
        PreciseTriangleHit hit{};
        const auto status = raycastTriangleObject(*mGeometry, origin, direction, minimumDistance, maximumDistance,
                                                  anyHit, ledger, hasHit, hit, bothSides);
        if (outStatus != nullptr)
            *outStatus = status;
        if (status == OVSPATIALQUERY_API_SUCCESS && hasHit)
            copyHit(hit, outHit);
        return status == OVSPATIALQUERY_API_SUCCESS && hasHit;
    }

    ovspatialquery_api_status_t raycastAll(const double origin[3], const float unitDirection[3], double minimumDistance,
                                           double maximumDistance, HitSink emit, void* context,
                                           InstanceHostLedger* ledger, bool bothSides) const override
    {
        struct Forward
        {
            HitSink emit;
            void*   context;
        } forward{emit, context};
        return raycastTriangleObjectAll(
            *mGeometry, origin, unitDirection, minimumDistance, maximumDistance, ledger,
            [](void* opaque, const PreciseTriangleHit& source)
            {
                const auto&      target = *static_cast<Forward*>(opaque);
                ExactGeometryHit hit{};
                copyHit(source, hit);
                return target.emit(target.context, hit);
            },
            &forward, bothSides);
    }

    bool sweepSphere(const double center[3], double radius, const float unitDirection[3], double minimumDistance,
                     double maximumDistance, ExactGeometryHit& outHit, InstanceHostLedger* ledger,
                     ovspatialquery_api_status_t* outStatus, bool bothSides, bool rayNormals) const override
    {
        bool                        hasHit = false;
        PreciseTriangleHit          sourceHit{};
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        hasHit = sweepSphereTriangleObject(*mGeometry, center, radius, unitDirection, minimumDistance, maximumDistance,
                                           sourceHit, ledger, &status, false, bothSides, rayNormals);
        if (outStatus != nullptr)
            *outStatus = status;
        if (status == OVSPATIALQUERY_API_SUCCESS && hasHit)
            copyHit(sourceHit, outHit);
        return status == OVSPATIALQUERY_API_SUCCESS && hasHit;
    }

    bool overlaps(const ovspatialquery_overlap_shape_t& geometry, const double (*boxRotation)[3],
                  InstanceHostLedger* ledger, ovspatialquery_api_status_t* outStatus) const override
    {
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        const bool hasOverlap              = overlapTriangleObject(*mGeometry, geometry, boxRotation, ledger, &status);
        if (outStatus != nullptr)
            *outStatus = status;
        return status == OVSPATIALQUERY_API_SUCCESS && hasOverlap;
    }

    uint64_t residentByteCount() const noexcept override
    {
        uint64_t total =
            saturatingResidentByteAdd(sizeof(TriangleMeshExactGeometry), 1u, sizeof(TriangleObjectGeometry));
        return mGeometry == nullptr
                   ? total
                   : saturatingResidentByteAdd(total, 1u, triangleGeometryResidentByteCount(*mGeometry));
    }

    const TriangleObjectGeometry* triangleSurface() const noexcept override { return mGeometry.get(); }

    void accumulateResidentAllocations(ExactGeometryResidentAccumulator& accumulator,
                                       bool includeSharedTriangleMesh) const noexcept override
    {
        (void)accumulator.add(this, sizeof(TriangleMeshExactGeometry));
        if (mGeometry == nullptr)
        {
            return;
        }
        (void)accumulator.add(mGeometry.get(), sizeof(TriangleObjectGeometry));
        if (includeSharedTriangleMesh && mGeometry->mesh != nullptr)
        {
            (void)accumulator.add(mGeometry->mesh.get(), triangleGeometryResidentByteCount(*mGeometry));
        }
    }

  private:
    static void copyHit(const PreciseTriangleHit& source, ExactGeometryHit& destination)
    {
        ExactGeometryHit hit{};
        hit.distance          = source.distance;
        hit.generatedTriangle = source.generatedTriangle;
        hit.sourceFace        = source.faceIndex;
        hit.sourceCorners[0]  = source.sourceCorners[0];
        hit.sourceCorners[1]  = source.sourceCorners[1];
        hit.sourceCorners[2]  = source.sourceCorners[2];
        hit.uv[0]             = source.uv[0];
        hit.uv[1]             = source.uv[1];
        hit.hasTriangleSource = source.hasTriangleSource;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            hit.position[axis] = source.position[axis];
            hit.normal[axis]   = source.normal[axis];
        }
        destination = hit;
    }

    const std::shared_ptr<const TriangleObjectGeometry> mGeometry;
};

// Sets `outGeometry` only once the geometry is built.
template <typename Geometry, typename Source>
ovspatialquery_api_status_t makeExactGeometry(Source&& source, std::shared_ptr<const ExactGeometry>& outGeometry)
{
    try
    {
        outGeometry = std::make_shared<const Geometry>(std::forward<Source>(source));
        return OVSPATIALQUERY_API_SUCCESS;
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

} // namespace

ovspatialquery_api_status_t convexHullPreparationStatus(ConvexHullPreparationOutcome outcome) noexcept
{
    switch (outcome)
    {
    case ConvexHullPreparationOutcome::success: return OVSPATIALQUERY_API_SUCCESS;
    case ConvexHullPreparationOutcome::invalidArgument:
    case ConvexHullPreparationOutcome::beyondVertexLimit:
    case ConvexHullPreparationOutcome::beyondFaceLimit:
    case ConvexHullPreparationOutcome::noEnclosingHull: return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    case ConvexHullPreparationOutcome::resourceExhausted: return OVSPATIALQUERY_API_HOST_BUDGET;
    case ConvexHullPreparationOutcome::outOfMemory: return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    default: return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t exportExactGeometryDescription(const std::shared_ptr<const ExactGeometry>& geometry,
                                                           ExactGeometryDescription& outDescription) noexcept
{
    if (geometry == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    ExactGeometryDescription description{};
    if (!geometry->exportDescription(description))
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    outDescription = description;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t qualifyPrimitiveTransform(const std::array<double, 16u>& worldFromLocal,
                                                      PrimitiveScaleRule             scaleRule,
                                                      QualifiedPrimitiveTransform&   outTransform)
{
    for (double component : worldFromLocal)
    {
        if (!std::isfinite(component))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    if (worldFromLocal[12] != 0.0 || worldFromLocal[13] != 0.0 || worldFromLocal[14] != 0.0 ||
        worldFromLocal[15] != 1.0)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    double columnLengthSquared[3] = {};
    double columnDot[3]           = {};
    for (uint32_t column = 0u; column < 3u; ++column)
    {
        for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
        {
            const double component      = worldFromLocal[worldAxis * 4u + column];
            columnLengthSquared[column] = std::fma(component, component, columnLengthSquared[column]);
        }
    }
    for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
    {
        columnDot[0] = std::fma(worldFromLocal[worldAxis * 4u], worldFromLocal[worldAxis * 4u + 1u], columnDot[0]);
        columnDot[1] = std::fma(worldFromLocal[worldAxis * 4u], worldFromLocal[worldAxis * 4u + 2u], columnDot[1]);
        columnDot[2] = std::fma(worldFromLocal[worldAxis * 4u + 1u], worldFromLocal[worldAxis * 4u + 2u], columnDot[2]);
    }
    const double maximumLengthSquared =
        std::max(columnLengthSquared[0], std::max(columnLengthSquared[1], columnLengthSquared[2]));
    constexpr double kRelativeTolerance = 1.0e-12;
    if (!std::isfinite(maximumLengthSquared) || columnLengthSquared[0] <= 0.0 || columnLengthSquared[1] <= 0.0 ||
        columnLengthSquared[2] <= 0.0)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const double columnLengths[3] = {
        std::sqrt(columnLengthSquared[0]),
        std::sqrt(columnLengthSquared[1]),
        std::sqrt(columnLengthSquared[2]),
    };
    const double normalizedDots[3] = {
        std::abs(columnDot[0]) / columnLengths[0] / columnLengths[1],
        std::abs(columnDot[1]) / columnLengths[0] / columnLengths[2],
        std::abs(columnDot[2]) / columnLengths[1] / columnLengths[2],
    };
    if (normalizedDots[0] > kRelativeTolerance || normalizedDots[1] > kRelativeTolerance ||
        normalizedDots[2] > kRelativeTolerance)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (scaleRule == PrimitiveScaleRule::uniform &&
        (std::abs(columnLengthSquared[1] - columnLengthSquared[0]) / maximumLengthSquared > kRelativeTolerance ||
         std::abs(columnLengthSquared[2] - columnLengthSquared[0]) / maximumLengthSquared > kRelativeTolerance))
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    QualifiedPrimitiveTransform qualified{};
    qualified.center = {worldFromLocal[3], worldFromLocal[7], worldFromLocal[11]};
    for (uint32_t column = 0u; column < 3u; ++column)
    {
        qualified.scales[column] = columnLengths[column];
        for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
        {
            qualified.axes[column][worldAxis] = worldFromLocal[worldAxis * 4u + column] / qualified.scales[column];
        }
    }
    outTransform = qualified;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t createSphereExactGeometry(const SphereGeometrySource&           source,
                                                      std::shared_ptr<const ExactGeometry>& outGeometry)
{
    if (!std::isfinite(source.radius) || source.radius <= 0.0)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    for (double component : source.center)
    {
        if (!std::isfinite(component) ||
            component - source.radius < -static_cast<double>(std::numeric_limits<float>::max()) ||
            component + source.radius > static_cast<double>(std::numeric_limits<float>::max()))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    return makeExactGeometry<SphereExactGeometry>(source, outGeometry);
}

ovspatialquery_api_status_t createBoxExactGeometry(const BoxGeometrySource&              source,
                                                   std::shared_ptr<const ExactGeometry>& outGeometry)
{
    for (double component : source.center)
    {
        if (!std::isfinite(component))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (!std::isfinite(source.halfExtents[axis]) || source.halfExtents[axis] <= 0.0)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        const double lengthSquared = dot(source.axes[axis], source.axes[axis]);
        if (!std::isfinite(lengthSquared) || std::abs(lengthSquared - 1.0) > 1.0e-12)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        for (uint32_t other = axis + 1u; other < 3u; ++other)
        {
            if (std::abs(dot(source.axes[axis], source.axes[other])) > 1.0e-12)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
        }
    }
    for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
    {
        double radius = 0.0;
        for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
        {
            radius = std::fma(std::abs(source.axes[localAxis][worldAxis]), source.halfExtents[localAxis], radius);
        }
        if (!std::isfinite(radius) ||
            source.center[worldAxis] - radius < -static_cast<double>(std::numeric_limits<float>::max()) ||
            source.center[worldAxis] + radius > static_cast<double>(std::numeric_limits<float>::max()))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    return makeExactGeometry<BoxExactGeometry>(source, outGeometry);
}

ovspatialquery_api_status_t createCapsuleExactGeometry(const CapsuleGeometrySource&          source,
                                                       std::shared_ptr<const ExactGeometry>& outGeometry)
{
    if (!std::isfinite(source.radius) || source.radius <= 0.0)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    double segmentLengthSquared = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (!std::isfinite(source.endpointA[axis]) || !std::isfinite(source.endpointB[axis]))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        const double delta   = source.endpointB[axis] - source.endpointA[axis];
        segmentLengthSquared = std::fma(delta, delta, segmentLengthSquared);
        const double minimum = std::min(source.endpointA[axis], source.endpointB[axis]) - source.radius;
        const double maximum = std::max(source.endpointA[axis], source.endpointB[axis]) + source.radius;
        if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
            minimum < -static_cast<double>(std::numeric_limits<float>::max()) ||
            maximum > static_cast<double>(std::numeric_limits<float>::max()))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    // A zero-length segment is a sphere of the capsule's radius; the capsule math handles it.
    if (!std::isfinite(segmentLengthSquared))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    return makeExactGeometry<CapsuleExactGeometry>(source, outGeometry);
}

ovspatialquery_api_status_t createTriangleMeshExactGeometry(
    std::shared_ptr<const TriangleObjectGeometry> triangleGeometry, std::shared_ptr<const ExactGeometry>& outGeometry)
{
    if (triangleGeometry == nullptr || triangleGeometry->mesh == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    return makeExactGeometry<TriangleMeshExactGeometry>(std::move(triangleGeometry), outGeometry);
}

} // namespace detail
} // namespace ovspatialquery
