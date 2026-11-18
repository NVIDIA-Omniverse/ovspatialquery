/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery_query_evaluation.h>

#include "ovspatialquery_box_sweep_geometry.h"
#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_instance_host_ledger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{
namespace
{

struct QueryLocationCandidate
{
    ovspatialquery_object_id_t objectId{};
    GeometryReference          geometry{};
    uint64_t                   materialId        = 0u;
    double                     distance          = 0.0;
    double                     queryPosition[3]  = {};
    double                     boundsPosition[3] = {};
    double                     position[3]       = {};
    float                      normal[3]         = {};
    uint64_t                   faceIndex         = 0u;
    uint32_t                   sourceCorners[3]  = {};
    float                      uv[2]             = {};
    bool                       hasTriangleSource = false;
};

enum class QueryTraversalKind : uint32_t
{
    ray,
    sphereSweep,
    sweptAabb,
};

struct PreparedPathQuery
{
    InstanceHostLedger* hostLedger       = nullptr;
    double              origin[3]        = {};
    float               unitDirection[3] = {};
    double              minimumDistance  = 0.0;
    double              maximumDistance  = 0.0;
    double              expansionRadius  = 0.0;
    // Every mesh triangle counts from both sides for this query (ADR-0065).
    bool bothSides = false;
    // A raycast, whose contact tolerance makes it a thin sweep that keeps the ray's
    // back-face normal rule; sphere sweeps leave it false (ADR-0065).
    bool                                          rayNormals    = false;
    QueryTraversalKind                            traversalKind = QueryTraversalKind::sweptAabb;
    ovspatialquery::detail::FloatRaycastQuery     ray{};
    ovspatialquery::detail::FloatSphereSweepQuery sweep{};
    ovspatialquery::detail::FloatAabb             sweptAabb{};
};

bool isValidUnitDirection(const float direction[3])
{
    double lengthSquared = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (!std::isfinite(direction[axis]))
        {
            return false;
        }
        lengthSquared += static_cast<double>(direction[axis]) * direction[axis];
    }
    return std::abs(lengthSquared - 1.0) <= 1.0e-5;
}

// Inside the input domain (ADR-0032) every path coordinate, its expansion by
// a radius and a tolerance, and every distance stay far inside float range.
double pathCoordinate(double origin, float direction, double distance)
{
    return std::fma(static_cast<double>(direction), distance, origin);
}

void prepareSweptAabb(PreparedPathQuery& query)
{
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        const double first  = pathCoordinate(query.origin[axis], query.unitDirection[axis], query.minimumDistance);
        const double second = pathCoordinate(query.origin[axis], query.unitDirection[axis], query.maximumDistance);
        query.sweptAabb.minimum[axis] = conservativeFloatBound(std::min(first, second) - query.expansionRadius, true);
        query.sweptAabb.maximum[axis] = conservativeFloatBound(std::max(first, second) + query.expansionRadius, false);
    }
    query.traversalKind = QueryTraversalKind::sweptAabb;
}

float roundUpToFloat(double value)
{
    float converted = static_cast<float>(value);
    if (static_cast<double>(converted) < value)
    {
        converted = std::nextafter(converted, std::numeric_limits<float>::infinity());
    }
    return converted;
}

void prepareConservativeTraversal(PreparedPathQuery& query, double originTolerance)
{
    double centerErrorSquared = 0.0;
    double maximumAxisError   = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        const double shifted   = pathCoordinate(query.origin[axis], query.unitDirection[axis], query.minimumDistance);
        const float  converted = static_cast<float>(shifted);
        const double error     = shifted - static_cast<double>(converted);
        centerErrorSquared += error * error;
        maximumAxisError                = std::max(maximumAxisError, std::abs(error));
        query.ray.origin[axis]          = converted;
        query.ray.unitDirection[axis]   = query.unitDirection[axis];
        query.sweep.center[axis]        = converted;
        query.sweep.unitDirection[axis] = query.unitDirection[axis];
    }
    // A path of zero length is the box around its expanded point.
    const float traversalDistance = roundUpToFloat(query.maximumDistance - query.minimumDistance);
    if (traversalDistance == 0.0f)
    {
        prepareSweptAabb(query);
        return;
    }
    query.ray.maxDistance   = traversalDistance;
    query.sweep.maxDistance = traversalDistance;

    // The backend absorbs origin rounding up to its tolerance in its tree
    // bounds; a ray within it needs no rounding-radius sweep.
    const double conservativeRadius = query.expansionRadius + std::sqrt(centerErrorSquared);
    if (query.expansionRadius == 0.0 && maximumAxisError <= originTolerance)
    {
        query.traversalKind = QueryTraversalKind::ray;
        return;
    }

    // Positive here: a nonzero expansion, or a rounding error above the tolerance.
    query.sweep.radius  = roundUpToFloat(conservativeRadius);
    query.traversalKind = QueryTraversalKind::sphereSweep;
}

ovspatialquery_api_status_t preparePathQuery(const double origin[3], const float unitDirection[3],
                                             double minimumDistance, double maximumDistance, double expansionRadius,
                                             PreparedPathQuery& destination, double originTolerance = 0.0)
{
    if (!isValidUnitDirection(unitDirection) || !inInputDomain(minimumDistance) || !inInputDomain(maximumDistance) ||
        minimumDistance < 0.0 || maximumDistance < minimumDistance || !std::isfinite(expansionRadius) ||
        expansionRadius < 0.0)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (!inInputDomain(origin[axis]))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        destination.origin[axis]        = origin[axis];
        destination.unitDirection[axis] = unitDirection[axis];
    }
    destination.minimumDistance = minimumDistance;
    destination.maximumDistance = maximumDistance;
    destination.expansionRadius = expansionRadius;
    prepareConservativeTraversal(destination, originTolerance);
    return OVSPATIALQUERY_API_SUCCESS;
}

// The distance at which the path enters the object's bounds grown by the
// expansion radius, the axis it enters through (-1 when it starts inside), and
// that face's coordinate and outward normal sign; false when it misses them.
bool pathAabbEntry(const PreparedPathQuery& query, const ovspatialquery_aabb_t& objectBounds, double& entry,
                   int32_t& entryAxis, double& entryBoundary, float& entryNormal)
{
    entry         = query.minimumDistance;
    double exit   = query.maximumDistance;
    entryAxis     = -1;
    entryBoundary = 0.0;
    entryNormal   = 0.0f;

    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        const double minimum   = objectBounds.minimum[axis] - query.expansionRadius;
        const double maximum   = objectBounds.maximum[axis] + query.expansionRadius;
        const double origin    = query.origin[axis];
        const double direction = static_cast<double>(query.unitDirection[axis]);
        if (direction == 0.0)
        {
            if (origin < minimum || origin > maximum)
            {
                return false;
            }

            continue;
        }

        double nearDistance = (minimum - origin) / direction;
        double farDistance  = (maximum - origin) / direction;
        double nearBoundary = minimum;
        float  nearNormal   = -1.0f;
        if (direction < 0.0) // by sign: a zero-thickness slab has equal distances
        {
            std::swap(nearDistance, farDistance);
            nearBoundary = maximum;
            nearNormal   = 1.0f;
        }
        if (nearDistance > entry || (nearDistance == entry && entryAxis < 0))
        {
            entry         = nearDistance;
            entryAxis     = static_cast<int32_t>(axis);
            entryBoundary = nearBoundary;
            entryNormal   = nearNormal;
        }
        exit = std::min(exit, farDistance);
        if (entry > exit)
        {
            return false;
        }
    }
    return std::isfinite(entry) && entry >= query.minimumDistance && entry <= query.maximumDistance;
}

bool exactPathAabbIntersection(const PreparedPathQuery& query, const ovspatialquery_aabb_t& objectBounds,
                               QueryLocationCandidate& candidate)
{
    double  entry         = 0.0;
    int32_t entryAxis     = -1;
    double  entryBoundary = 0.0;
    float   entryNormal   = 0.0f;
    if (!pathAabbEntry(query, objectBounds, entry, entryAxis, entryBoundary, entryNormal))
    {
        return false;
    }

    candidate.distance = entry;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        candidate.queryPosition[axis] = pathCoordinate(query.origin[axis], query.unitDirection[axis], entry);
        candidate.position[axis]      = candidate.queryPosition[axis];
        candidate.boundsPosition[axis] =
            std::max(objectBounds.minimum[axis], std::min(candidate.queryPosition[axis], objectBounds.maximum[axis]));
        candidate.normal[axis] = 0.0f;
    }
    if (entryAxis >= 0)
    {
        const uint32_t axis      = static_cast<uint32_t>(entryAxis);
        candidate.position[axis] = entryBoundary;
        candidate.normal[axis]   = entryNormal;
    }
    if (query.expansionRadius > 0.0 && entry == 0.0)
    {
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            candidate.normal[axis] = -query.unitDirection[axis];
        }
    }
    return true;
}

bool objectParticipates(const PopulationObject& object, const ParticipationBits& queryBits)
{
    return participationAccepts(queryBits, object.participationBits);
}

ovspatialquery_api_status_t traverseBackendPathCandidates(const GenerationLease&   generation,
                                                          const PreparedPathQuery& query, BackendCandidateSink& sink)
{
    if (query.traversalKind == QueryTraversalKind::sweptAabb)
    {
        return generation.backend().aabbOverlapCandidates(query.sweptAabb, sink);
    }
    return query.traversalKind == QueryTraversalKind::ray
               ? generation.backend().raycastCandidates(query.ray, sink)
               : generation.backend().sphereSweepCandidates(query.sweep, sink);
}

void exactPathCandidate(const PopulationObject& object, const PreparedPathQuery& query, const ExactGeometryHit& hit,
                        QueryLocationCandidate& outCandidate)
{
    outCandidate.objectId = object.id;
    outCandidate.geometry = object.geometryReference;
    outCandidate.distance = hit.distance;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        outCandidate.queryPosition[axis] = pathCoordinate(query.origin[axis], query.unitDirection[axis], hit.distance);
        outCandidate.position[axis]      = hit.position[axis];
        outCandidate.normal[axis]        = hit.normal[axis];
    }
    outCandidate.faceIndex = hit.sourceFace;
    std::copy(hit.sourceCorners, hit.sourceCorners + 3u, outCandidate.sourceCorners);
    outCandidate.uv[0]             = hit.uv[0];
    outCandidate.uv[1]             = hit.uv[1];
    outCandidate.hasTriangleSource = hit.hasTriangleSource;
    outCandidate.materialId        = hitMaterialId(object, hit.hasTriangleSource, hit.sourceFace);
}

bool preciseRaycastCandidate(const PopulationObject& object, const PreparedPathQuery& query, bool anyHit,
                             QueryLocationCandidate& outCandidate, ovspatialquery_api_status_t& status)
{
    if (object.exactGeometry == nullptr)
    {
        return false;
    }
    ExactGeometryHit hit{};
    bool             hasHit = false;
    status                  = OVSPATIALQUERY_API_SUCCESS;
    if (query.expansionRadius == 0.0)
    {
        hasHit = object.exactGeometry->raycast(query.origin, query.unitDirection, query.minimumDistance,
                                               query.maximumDistance, anyHit, hit, query.hostLedger, &status,
                                               query.bothSides);
    }
    else
    {
        hasHit = object.exactGeometry->sweepSphere(query.origin, query.expansionRadius, query.unitDirection,
                                                   query.minimumDistance, query.maximumDistance, hit, query.hostLedger,
                                                   &status, query.bothSides, query.rayNormals);
    }
    if (status != OVSPATIALQUERY_API_SUCCESS || !hasHit)
    {
        return false;
    }
    exactPathCandidate(object, query, hit, outCandidate);
    return true;
}

// The one candidate producer behind every path query: bounds precision tests
// the conservative object bounds, exact precision asks the object geometry.
// outCandidate is scratch: it is meaningful only when the call returns true.
bool pathCandidate(QueryPrecision precision, const PopulationObject& object, const PreparedPathQuery& query,
                   HitMode mode, QueryLocationCandidate& outCandidate, ovspatialquery_api_status_t& status)
{
    if (precision == QueryPrecision::exact)
    {
        return preciseRaycastCandidate(object, query, mode == HitMode::any, outCandidate, status);
    }
    outCandidate.objectId   = object.id;
    outCandidate.geometry   = object.geometryReference;
    outCandidate.materialId = object.materialId;
    return exactPathAabbIntersection(query, object.bounds, outCandidate);
}

// Accumulation policies for PathCandidateSink: accept() answers false only
// when the candidate could not be stored.
class ClosestCandidateAccumulator
{
  public:
    // Whether accept would make `candidate` the running best.
    bool beats(const QueryLocationCandidate& candidate) const
    {
        return !mHasHit || candidate.distance < mCandidate.distance ||
               (candidate.distance == mCandidate.distance && objectIdLess(candidate.objectId, mCandidate.objectId));
    }

    bool accept(const QueryLocationCandidate& candidate)
    {
        if (beats(candidate))
        {
            mCandidate = candidate;
            mHasHit    = true;
        }
        return true;
    }

    bool hasHit() const { return mHasHit; }

    const QueryLocationCandidate& candidate() const { return mCandidate; }
    QueryLocationCandidate&       candidate() { return mCandidate; }

  private:
    QueryLocationCandidate mCandidate{};
    bool                   mHasHit = false;
};

// Later candidates are evaluated up to this bound, a few float roundoffs beyond
// the running best, so a contact that rounding placed just past it can still
// tie or beat it.
double closestSweepClipBound(double best)
{
    return best + 8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) * std::max(1.0, best);
}

// The reach, along a traversal that starts at `minimum`, that keeps every
// object a closest sweep may still accept: the clip bound of its running best.
// The tree's leaves absorb float rounding at the population's magnitude, and
// the bound's relative slack the rounding of a start farther out.
float closestSweepReach(double minimum, double maximum, double best)
{
    return roundUpToFloat(std::min(maximum, closestSweepClipBound(best)) - minimum);
}

// One broadphase candidate of a closest exact sweep, keyed by the distance at
// which the sweep enters its conservative object bounds.
// Deliberately without default initializers: the sink's buffer must not be
// zeroed on every sub-microsecond sweep; entries are written before they are
// read.
struct OrderedSweepCandidate
{
    double   entryDistance;
    uint32_t objectIndex;
};

// A ray's mesh contacts come from PhysX's float raycast in the cooked mesh's frame,
// centred on the mesh, whose distance error grows with the distance to the mesh and
// the mesh's size (5.5e-5 m twelve metres from a rotated instance two metres wide).
double closestRayClipBound(double best) { return best + 1.0e-4 * (best + 1.0); }

// How far beyond its running best a closest query keeps evaluating: `bound` of the
// best, plus `boundsSlack` times a candidate's largest bounds extent. A closest ray's
// slack exceeds any float error that could bring a candidate entered beyond its
// bound back to the best, so its result is the unordered evaluation's.
struct ClosestClip
{
    double (*bound)(double best);
    double boundsSlack;
};
constexpr ClosestClip kClosestRayClip{closestRayClipBound, 1.0e-4};

double largestExtent(const ovspatialquery_aabb_t& bounds)
{
    return std::max(std::max(bounds.maximum[0] - bounds.minimum[0], bounds.maximum[1] - bounds.minimum[1]),
                    bounds.maximum[2] - bounds.minimum[2]);
}

// The broadphase candidates of one closest exact sweep with the distance at
// which the sweep enters each candidate's conservative bounds. Small sweeps
// stay allocation-free; a candidate the double bounds test cannot place keeps
// the earliest entry so it is evaluated exactly as before.
class OrderedSweepCandidateSink final : public BackendCandidateSink
{
  public:
    OrderedSweepCandidateSink(const GenerationLease& generation, const ParticipationBits& queryBits)
        : mGeneration(generation), mQueryBits(queryBits)
    {}

    bool reportCandidate(uint32_t objectIndex) override
    {
        const ObjectTable& objects = mGeneration.objects();
        if (objectIndex >= objects.size())
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
            return false;
        }
        if (!mGeneration.includesObject(objectIndex) || !objectParticipates(objects[objectIndex], mQueryBits))
        {
            return true;
        }
        // Entry distances are only needed to order two or more candidates;
        // the caller fills them in then.
        const OrderedSweepCandidate candidate{0.0, objectIndex};
        if (mCount < kSmall)
        {
            mSmall[mCount] = candidate;
        }
        else
        {
            try
            {
                if (mCount == kSmall)
                {
                    mLarge.assign(mSmall, mSmall + kSmall);
                }
                mLarge.push_back(candidate);
            }
            catch (const std::bad_alloc&)
            {
                mStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
                return false;
            }
        }
        ++mCount;
        return true;
    }

    ovspatialquery_api_status_t status() const override { return mStatus; }

    uint32_t               size() const { return mCount; }
    OrderedSweepCandidate* begin() { return mLarge.empty() ? mSmall : mLarge.data(); }
    OrderedSweepCandidate* end() { return begin() + mCount; }

  private:
    static constexpr uint32_t          kSmall = 64u;
    const GenerationLease&             mGeneration;
    ParticipationBits                  mQueryBits{};
    OrderedSweepCandidate              mSmall[kSmall];
    std::vector<OrderedSweepCandidate> mLarge;
    uint32_t                           mCount  = 0u;
    ovspatialquery_api_status_t        mStatus = OVSPATIALQUERY_API_SUCCESS;
};

// Closest exact rays evaluate the broadphase candidates nearest first and
// clip every later exact test to the running best: a candidate entered beyond
// the best contact cannot beat or tie it, and a mesh sweep bounded by the best
// contact traverses and gathers far fewer triangles. The candidate set, the
// exact predicates and the tie rule (distance, then lower object id) are the
// same as for the unordered evaluation.
ovspatialquery_api_status_t evaluateOrderedClosestSweep(const GenerationLease&       generation,
                                                        const PreparedPathQuery&     query,
                                                        const ParticipationBits&     queryBits,
                                                        ClosestCandidateAccumulator& accumulate,
                                                        const ClosestClip&           clip)
{
    OrderedSweepCandidateSink         sink(generation, queryBits);
    const ovspatialquery_api_status_t collected = traverseBackendPathCandidates(generation, query, sink);
    if (collected != OVSPATIALQUERY_API_SUCCESS)
    {
        return collected;
    }
    const ObjectTable& objects = generation.objects();
    if (sink.size() > 1u)
    {
        // A candidate the double bounds test cannot place keeps the earliest
        // entry, so it is evaluated exactly as before.
        for (OrderedSweepCandidate& candidate : sink)
        {
            QueryLocationCandidate entry{};
            candidate.entryDistance = exactPathAabbIntersection(query, objects[candidate.objectIndex].bounds, entry)
                                          ? entry.distance
                                          : query.minimumDistance;
        }
        std::sort(sink.begin(), sink.end(),
                  [](const OrderedSweepCandidate& left, const OrderedSweepCandidate& right)
                  {
                      return left.entryDistance < right.entryDistance ||
                             (left.entryDistance == right.entryDistance && left.objectIndex < right.objectIndex);
                  });
    }
    PreparedPathQuery clipped = query;
    for (const OrderedSweepCandidate& next : sink)
    {
        if (accumulate.hasHit())
        {
            const double slack = clip.boundsSlack * largestExtent(objects[next.objectIndex].bounds);
            const double bound = std::min(query.maximumDistance, clip.bound(accumulate.candidate().distance) + slack);
            if (next.entryDistance > bound)
            {
                // Entries are sorted; without a per-candidate slack no later one is within its bound.
                if (clip.boundsSlack == 0.0)
                {
                    break;
                }
                continue;
            }
            clipped.maximumDistance = bound;
        }
        QueryLocationCandidate      candidate{};
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        if (preciseRaycastCandidate(objects[next.objectIndex], clipped, false, candidate, status))
        {
            accumulate.accept(candidate);
        }
        else if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// Closest exact sweeps evaluate their broadphase candidates nearest first and
// clip every later exact test to the running best: a candidate entered beyond
// the best contact cannot beat or tie it, and a mesh sweep bounded by the best
// contact gathers far fewer triangles. Candidates are evaluated in chunks as
// the traversal reports them, and after each chunk the traversal's reach falls
// to the best, so a long sweep stops near its first contact; a sweep with at
// most one chunk of candidates evaluates them in the order a complete sort
// gives. The candidate set, the exact predicates and the tie rule (distance,
// then lower object id) do not depend on that order.
class OrderedClosestSweepSink final : public BackendCandidateSink
{
  public:
    OrderedClosestSweepSink(const GenerationLease& generation, const PreparedPathQuery& query,
                            const ParticipationBits& queryBits, ClosestCandidateAccumulator& accumulate)
        : mGeneration(generation), mQuery(query), mClipped(query), mQueryBits(queryBits), mAccumulate(accumulate)
    {}

    bool reportCandidate(uint32_t objectIndex) override
    {
        const ObjectTable& objects = mGeneration.objects();
        if (objectIndex >= objects.size())
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
            return false;
        }
        if (!mGeneration.includesObject(objectIndex) || !objectParticipates(objects[objectIndex], mQueryBits))
        {
            return true;
        }
        mChunk[mCount++] = {0.0, objectIndex};
        return mCount < kChunk || flush();
    }

    ovspatialquery_api_status_t status() const override { return mStatus; }

    // Evaluates the candidates held so far, nearest first; false when one
    // failed. It runs inside the backend's traversal, so it reports a failed
    // allocation as a status instead of throwing through it.
    bool flush() noexcept
    {
        try
        {
            return evaluateChunk();
        }
        catch (const std::bad_alloc&)
        {
            mStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (...)
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
        }
        return false;
    }

  private:
    bool evaluateChunk()
    {
        const ObjectTable& objects = mGeneration.objects();
        // Entries are only needed to order two or more candidates or to skip
        // one beyond the best. A candidate the double bounds test cannot place
        // keeps the earliest entry, so it is evaluated.
        if (mCount > 1u || mAccumulate.hasHit())
        {
            for (uint32_t index = 0u; index < mCount; ++index)
            {
                OrderedSweepCandidate& candidate = mChunk[index];
                int32_t                axis      = -1;
                double                 boundary  = 0.0;
                float                  normal    = 0.0f;
                if (!pathAabbEntry(mQuery, objects[candidate.objectIndex].bounds, candidate.entryDistance, axis,
                                   boundary, normal))
                {
                    candidate.entryDistance = mQuery.minimumDistance;
                }
            }
            std::sort(mChunk, mChunk + mCount,
                      [](const OrderedSweepCandidate& left, const OrderedSweepCandidate& right)
                      {
                          return left.entryDistance < right.entryDistance ||
                                 (left.entryDistance == right.entryDistance && left.objectIndex < right.objectIndex);
                      });
        }
        const uint32_t count = mCount;
        mCount               = 0u;
        for (uint32_t index = 0u; index < count; ++index)
        {
            const OrderedSweepCandidate& next = mChunk[index];
            if (mAccumulate.hasHit())
            {
                const double bound =
                    std::min(mQuery.maximumDistance, closestSweepClipBound(mAccumulate.candidate().distance));
                if (next.entryDistance > bound)
                {
                    break;
                }
                mClipped.maximumDistance = bound;
            }
            QueryLocationCandidate candidate{};
            if (preciseRaycastCandidate(objects[next.objectIndex], mClipped, false, candidate, mStatus))
            {
                mAccumulate.accept(candidate);
                lowerReach(closestSweepReach(mQuery.minimumDistance, mQuery.maximumDistance,
                                             mAccumulate.candidate().distance));
            }
            else if (mStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                return false;
            }
        }
        return true;
    }

    static constexpr uint32_t    kChunk = 64u;
    const GenerationLease&       mGeneration;
    const PreparedPathQuery&     mQuery;
    PreparedPathQuery            mClipped;
    ParticipationBits            mQueryBits{};
    ClosestCandidateAccumulator& mAccumulate;
    OrderedSweepCandidate        mChunk[kChunk];
    uint32_t                     mCount  = 0u;
    ovspatialquery_api_status_t  mStatus = OVSPATIALQUERY_API_SUCCESS;
};

class AllCandidatesAccumulator
{
  public:
    explicit AllCandidatesAccumulator(std::vector<QueryLocationCandidate>& candidates) : mCandidates(candidates) {}

    bool accept(const QueryLocationCandidate& candidate)
    {
        try
        {
            mCandidates.push_back(candidate);
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        return true;
    }

  private:
    std::vector<QueryLocationCandidate>& mCandidates;
};

// Every crossing orders by distance, then object id, then source face, so the
// retained hits are the same whatever order the backend and PhysX report them in.
bool crossingLess(const QueryLocationCandidate& left, const QueryLocationCandidate& right)
{
    if (left.distance != right.distance)
    {
        return left.distance < right.distance;
    }
    if (!objectIdEqual(left.objectId, right.objectId))
    {
        return objectIdLess(left.objectId, right.objectId);
    }
    return left.faceIndex < right.faceIndex;
}

// Grows `values` to hold `needed` elements. The row's execution peak already
// holds `prepaid` elements (one per object, as a per-object row needs); each
// new capacity beyond them is charged to `ledger`, with `extraBytes` per
// element for what the caller derives from it, and the old and new buffers'
// overlap is charged while reserve copies. `charged` counts the elements
// charged so far; the leases stay until the evaluation ends. False when the
// ledger refuses.
template <typename T>
bool reserveCharged(std::vector<T>& values, size_t needed, size_t limit, size_t extraBytes, size_t prepaid,
                    size_t& charged, InstanceHostLedger* ledger, std::vector<HostLease>& leases)
{
    if (needed <= values.capacity())
    {
        return true;
    }
    const size_t capacity  = std::min(std::max({needed, size_t{16u}, 2u * values.capacity()}), std::max(needed, limit));
    const size_t covered   = prepaid + charged;
    const size_t uncovered = capacity > covered ? capacity - covered : 0u;
    // While reserve copies, the old buffer lives beside the new one: charge
    // that overlap for the copy, then keep only the new capacity's charge.
    const size_t   peak    = values.capacity() + capacity > covered ? values.capacity() + capacity - covered : 0u;
    const uint64_t lasting = uint64_t(uncovered) * (sizeof(T) + extraBytes);
    HostLease      lease;
    if (ledger != nullptr && peak != 0u)
    {
        HostTransaction transaction(*ledger);
        if (!transaction.reserve(lasting + uint64_t(peak - uncovered) * sizeof(T)) || !transaction.commit(lease))
        {
            return false;
        }
    }
    values.reserve(capacity);
    if (lease && lasting == 0u)
    {
        lease.reset();
    }
    else if (lease)
    {
        lease.shrink(lasting);
        leases.push_back(std::move(lease));
    }
    charged += uncovered;
    return true;
}

// Keeps the `retained` nearest crossings of a row: it holds at most twice that
// many and drops the farthest half when full. Its storage grows with the
// crossings found, so a generous row limit costs nothing up front; growth
// beyond the per-object storage the execution peak holds is charged.
class NearestCrossingsAccumulator
{
  public:
    NearestCrossingsAccumulator(std::vector<QueryLocationCandidate>& candidates, size_t retained, size_t objects,
                                InstanceHostLedger* ledger)
        : mCandidates(candidates), mRetained(retained), mObjects(objects), mLedger(ledger)
    {}

    bool accept(const QueryLocationCandidate& candidate)
    {
        try
        {
            if (mCandidates.size() >= 2u * mRetained)
            {
                std::nth_element(mCandidates.begin(), mCandidates.begin() + static_cast<ptrdiff_t>(mRetained),
                                 mCandidates.end(), crossingLess);
                mCandidates.resize(mRetained);
            }
            // Each kept crossing later becomes one projected hit.
            if (!reserveCharged(mCandidates, mCandidates.size() + 1u, 2u * mRetained, sizeof(RaycastSemanticHit),
                                mObjects, mChargedCandidates, mLedger, mLeases))
            {
                mBudgetRefused = true;
                return false;
            }
            mCandidates.push_back(candidate);
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        return true;
    }

    // The tree may report an object more than once: `first` says whether this
    // is its first report. False only when the visit could not be recorded.
    bool visit(uint32_t objectIndex, bool& first)
    {
        first = false;
        try
        {
            const auto position = std::lower_bound(mVisited.begin(), mVisited.end(), objectIndex);
            if (position != mVisited.end() && *position == objectIndex)
            {
                return true;
            }
            const ptrdiff_t offset = position - mVisited.begin();
            // The per-object peak prepays a candidate and a projected hit per object,
            // both of which this row uses, so the visited list is charged in full.
            if (!reserveCharged(mVisited, mVisited.size() + 1u, mObjects, 0u, 0u, mChargedVisited, mLedger, mLeases))
            {
                mBudgetRefused = true;
                return false;
            }
            mVisited.insert(mVisited.begin() + offset, objectIndex);
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        first = true;
        return true;
    }

    // The status of a refused accept() or visit().
    ovspatialquery_api_status_t refusal() const
    {
        return mBudgetRefused ? OVSPATIALQUERY_API_HOST_BUDGET : OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }

  private:
    std::vector<QueryLocationCandidate>& mCandidates;
    size_t                               mRetained;
    size_t                               mObjects;
    InstanceHostLedger*                  mLedger;
    std::vector<uint32_t>                mVisited;
    size_t                               mChargedCandidates = 0u;
    size_t                               mChargedVisited    = 0u;
    std::vector<HostLease>               mLeases;
    bool                                 mBudgetRefused = false;
};

template <typename Accumulate>
class PathCandidateSink final : public BackendCandidateSink
{
  public:
    PathCandidateSink(const GenerationLease& generation, const PreparedPathQuery& query, QueryPrecision precision,
                      HitMode mode, const ParticipationBits& queryBits, Accumulate& accumulate)
        : mGeneration(generation), mQuery(query), mPrecision(precision), mMode(mode), mQueryBits(queryBits),
          mAccumulate(accumulate)
    {}

    bool reportCandidate(uint32_t objectIndex) override
    {
        if (objectIndex >= mGeneration.objects().size())
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
            return false;
        }
        if (!mGeneration.includesObject(objectIndex))
        {
            return true;
        }
        const PopulationObject& object = mGeneration.objects()[objectIndex];
        if (!objectParticipates(object, mQueryBits))
        {
            return true;
        }
        if constexpr (std::is_same_v<Accumulate, NearestCrossingsAccumulator>)
        {
            // Expand each object once, however often the tree reports it.
            bool first = false;
            if (!mAccumulate.visit(objectIndex, first))
            {
                mStatus = mAccumulate.refusal();
                return false;
            }
            if (!first)
            {
                return true;
            }
            if (mPrecision == QueryPrecision::exact)
            {
                return reportCrossings(object);
            }
        }
        QueryLocationCandidate candidate{};
        if (!pathCandidate(mPrecision, object, mQuery, mMode, candidate, mStatus))
        {
            return mStatus == OVSPATIALQUERY_API_SUCCESS;
        }
        if (!mAccumulate.accept(candidate))
        {
            mStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
            if constexpr (std::is_same_v<Accumulate, NearestCrossingsAccumulator>)
            {
                mStatus = mAccumulate.refusal();
            }
            return false;
        }
        if constexpr (std::is_same_v<Accumulate, ClosestCandidateAccumulator>)
        {
            if (mClipToClosest)
            {
                lowerReach(closestSweepReach(mQuery.minimumDistance, mQuery.maximumDistance,
                                             mAccumulate.candidate().distance));
            }
        }
        return mMode != HitMode::any;
    }

    ovspatialquery_api_status_t status() const override { return mStatus; }

    // A closest bounds sweep lets the traversal stop at its running best's clip
    // bound, so its cost follows the distance to its first contact rather than
    // the path's length.
    void clipToClosest() { mClipToClosest = true; }

  private:
    // Every surface crossing of an exact object; admission refuses a contact
    // tolerance here, so the path is a ray.
    bool reportCrossings(const PopulationObject& object)
    {
        if (object.exactGeometry == nullptr)
        {
            return true;
        }
        struct Emit
        {
            PathCandidateSink&      sink;
            const PopulationObject& object;
        } emit{*this, object};
        mStatus = object.exactGeometry->raycastAll(
            mQuery.origin, mQuery.unitDirection, mQuery.minimumDistance, mQuery.maximumDistance,
            [](void* opaque, const ExactGeometryHit& hit)
            {
                const auto&            target = *static_cast<Emit*>(opaque);
                QueryLocationCandidate candidate{};
                exactPathCandidate(target.object, target.sink.mQuery, hit, candidate);
                return target.sink.mAccumulate.accept(candidate);
            },
            &emit, mQuery.hostLedger, mQuery.bothSides);
        if constexpr (std::is_same_v<Accumulate, NearestCrossingsAccumulator>)
        {
            // raycastAll reports a refused hit as OUT_OF_MEMORY; the accumulator knows why.
            if (mStatus == OVSPATIALQUERY_API_OUT_OF_MEMORY)
            {
                mStatus = mAccumulate.refusal();
            }
        }
        return mStatus == OVSPATIALQUERY_API_SUCCESS;
    }

    const GenerationLease&      mGeneration;
    const PreparedPathQuery&    mQuery;
    QueryPrecision              mPrecision;
    HitMode                     mMode;
    ParticipationBits           mQueryBits{};
    Accumulate&                 mAccumulate;
    ovspatialquery_api_status_t mStatus        = OVSPATIALQUERY_API_SUCCESS;
    bool                        mClipToClosest = false;
};

struct PreparedOverlapQuery
{
    ovspatialquery_overlap_shape_t geometry{};
    double                         boxRotation[3][3] = {};
    // Whether the row gave its BOX rotation in double precision (ADR-0026).
    // A float32 row's `boxRotation` is the matrix of its float32 quaternion,
    // computed as makeQueryBoxGeometry computes it.
    bool                              exactBoxRotation     = false;
    double                            broadphaseMinimum[3] = {};
    double                            broadphaseMaximum[3] = {};
    ovspatialquery::detail::FloatAabb broadphase{};
};

// `boxRotation`, when given, orients a BOX in double precision in place of
// the shape's float32 quaternion.
ovspatialquery_api_status_t prepareQueryOverlap(const ovspatialquery_overlap_shape_t& source, double contactTolerance,
                                                PreparedOverlapQuery& destination, const double* boxRotation = nullptr)
{
    destination.geometry = source;
    double minimum[3]    = {};
    double maximum[3]    = {};
    if (source.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
    {
        if (!inInputDomain(source.data.sphere.radius) || source.data.sphere.radius <= 0.0)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            if (!inInputDomain(source.data.sphere.center[axis]))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            const double radius                     = source.data.sphere.radius + contactTolerance;
            destination.geometry.data.sphere.radius = radius;
            minimum[axis]                           = source.data.sphere.center[axis] - radius;
            maximum[axis]                           = source.data.sphere.center[axis] + radius;
        }
    }
    else if (source.kind == OVSPATIALQUERY_OVERLAP_AABB)
    {
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            const double sourceMinimum = source.data.aabb.min_corner[axis];
            const double sourceMaximum = source.data.aabb.max_corner[axis];
            if (!inInputDomain(sourceMinimum) || !inInputDomain(sourceMaximum) || sourceMinimum > sourceMaximum)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            minimum[axis]                                   = sourceMinimum - contactTolerance;
            maximum[axis]                                   = sourceMaximum + contactTolerance;
            destination.geometry.data.aabb.min_corner[axis] = minimum[axis];
            destination.geometry.data.aabb.max_corner[axis] = maximum[axis];
        }
    }
    else if (source.kind == OVSPATIALQUERY_OVERLAP_BOX)
    {
        double quaternion[4] = {};
        double lengthSquared = 0.0;
        for (uint32_t component = 0u; component < 4u; ++component)
        {
            quaternion[component] = boxRotation != nullptr
                                        ? boxRotation[component]
                                        : static_cast<double>(source.data.box.rotation_xyzw[component]);
            if (!std::isfinite(quaternion[component]))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            lengthSquared += quaternion[component] * quaternion[component];
        }
        if (!std::isfinite(lengthSquared) || std::abs(lengthSquared - 1.0) > 1.0e-5)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        const double inverseLength  = 1.0 / std::sqrt(lengthSquared);
        double       rotation[3][3] = {};
        quaternionRotationMatrix(quaternion, inverseLength, rotation);
        destination.exactBoxRotation = boxRotation != nullptr;
        for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
        {
            if (!inInputDomain(source.data.box.half_extents[localAxis]) ||
                source.data.box.half_extents[localAxis] <= 0.0)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            destination.geometry.data.box.half_extents[localAxis] =
                source.data.box.half_extents[localAxis] + contactTolerance;
        }
        for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
        {
            if (!inInputDomain(source.data.box.center[worldAxis]))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            double worldExtent = 0.0;
            for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
            {
                destination.boxRotation[worldAxis][localAxis] = rotation[worldAxis][localAxis];
                worldExtent +=
                    std::abs(rotation[worldAxis][localAxis]) * destination.geometry.data.box.half_extents[localAxis];
            }
            minimum[worldAxis] = source.data.box.center[worldAxis] - worldExtent;
            maximum[worldAxis] = source.data.box.center[worldAxis] + worldExtent;
        }
    }
    else
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        destination.broadphaseMinimum[axis]  = minimum[axis];
        destination.broadphaseMaximum[axis]  = maximum[axis];
        destination.broadphase.minimum[axis] = conservativeFloatBound(minimum[axis], true);
        destination.broadphase.maximum[axis] = conservativeFloatBound(maximum[axis], false);
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

bool aabbsOverlap(const double minimum[3], const double maximum[3], const ovspatialquery_aabb_t& objectBounds)
{
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (maximum[axis] < objectBounds.minimum[axis] || minimum[axis] > objectBounds.maximum[axis])
        {
            return false;
        }
    }
    return true;
}

bool sphereOverlapsAabb(const ovspatialquery_overlap_sphere_t& sphere, const ovspatialquery_aabb_t& objectBounds)
{
    double normalizedDistanceSquared = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        double distance = 0.0;
        if (sphere.center[axis] < objectBounds.minimum[axis])
        {
            distance = objectBounds.minimum[axis] - sphere.center[axis];
        }
        else if (sphere.center[axis] > objectBounds.maximum[axis])
        {
            distance = sphere.center[axis] - objectBounds.maximum[axis];
        }
        if (distance > sphere.radius)
        {
            return false;
        }
        const double normalized = distance / sphere.radius;
        normalizedDistanceSquared += normalized * normalized;
        if (normalizedDistanceSquared > 1.0)
        {
            return false;
        }
    }
    return true;
}

bool boxesOverlapOnAxis(const double axis[3], const double scaledCenterDelta[3], const double scaledAabbExtents[3],
                        const double scaledBoxHalfExtents[3], const PreparedOverlapQuery& query)
{
    const double axisLengthSquared = dot3(axis, axis);
    if (axisLengthSquared < 1.0e-30)
    {
        return true;
    }
    const double centerDistance = std::abs(dot3(scaledCenterDelta, axis));
    double       aabbRadius     = 0.0;
    double       boxRadius      = 0.0;
    for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
    {
        aabbRadius += scaledAabbExtents[worldAxis] * std::abs(axis[worldAxis]);
    }
    for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
    {
        const double boxAxis[3] = {
            query.boxRotation[0][localAxis],
            query.boxRotation[1][localAxis],
            query.boxRotation[2][localAxis],
        };
        boxRadius += scaledBoxHalfExtents[localAxis] * std::abs(dot3(boxAxis, axis));
    }
    return centerDistance <= aabbRadius + boxRadius;
}

bool boxOverlapsAabb(const PreparedOverlapQuery& query, const ovspatialquery_aabb_t& objectBounds)
{
    double clippedMinimum[3] = {};
    double clippedMaximum[3] = {};
    double aabbCenter[3]     = {};
    double aabbExtents[3]    = {};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        clippedMinimum[axis] = std::max(objectBounds.minimum[axis], query.broadphaseMinimum[axis]);
        clippedMaximum[axis] = std::min(objectBounds.maximum[axis], query.broadphaseMaximum[axis]);
        if (clippedMinimum[axis] > clippedMaximum[axis])
        {
            return false;
        }
        aabbCenter[axis]  = clippedMinimum[axis] * 0.5 + clippedMaximum[axis] * 0.5;
        aabbExtents[axis] = clippedMaximum[axis] * 0.5 - clippedMinimum[axis] * 0.5;
    }
    double scale = 1.0;
    for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
    {
        scale = std::max(scale, std::abs(aabbCenter[worldAxis]));
        scale = std::max(scale, std::abs(query.geometry.data.box.center[worldAxis]));
        scale = std::max(scale, aabbExtents[worldAxis]);
    }
    for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
    {
        scale = std::max(scale, query.geometry.data.box.half_extents[localAxis]);
    }
    double scaledCenterDelta[3]    = {};
    double scaledAabbExtents[3]    = {};
    double scaledBoxHalfExtents[3] = {};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        scaledCenterDelta[axis]    = query.geometry.data.box.center[axis] / scale - aabbCenter[axis] / scale;
        scaledAabbExtents[axis]    = aabbExtents[axis] / scale;
        scaledBoxHalfExtents[axis] = query.geometry.data.box.half_extents[axis] / scale;
    }

    const double worldAxes[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
    {
        if (!boxesOverlapOnAxis(worldAxes[worldAxis], scaledCenterDelta, scaledAabbExtents, scaledBoxHalfExtents,
                                query))
        {
            return false;
        }
    }
    for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
    {
        const double boxAxis[3] = {
            query.boxRotation[0][localAxis],
            query.boxRotation[1][localAxis],
            query.boxRotation[2][localAxis],
        };
        if (!boxesOverlapOnAxis(boxAxis, scaledCenterDelta, scaledAabbExtents, scaledBoxHalfExtents, query))
        {
            return false;
        }
        for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
        {
            const double crossAxis[3] = {
                worldAxes[worldAxis][1] * boxAxis[2] - worldAxes[worldAxis][2] * boxAxis[1],
                worldAxes[worldAxis][2] * boxAxis[0] - worldAxes[worldAxis][0] * boxAxis[2],
                worldAxes[worldAxis][0] * boxAxis[1] - worldAxes[worldAxis][1] * boxAxis[0],
            };
            if (!boxesOverlapOnAxis(crossAxis, scaledCenterDelta, scaledAabbExtents, scaledBoxHalfExtents, query))
            {
                return false;
            }
        }
    }
    return true;
}

bool exactOverlap(const PreparedOverlapQuery& query, const ovspatialquery_aabb_t& objectBounds)
{
    if (query.geometry.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
    {
        return sphereOverlapsAabb(query.geometry.data.sphere, objectBounds);
    }
    if (query.geometry.kind == OVSPATIALQUERY_OVERLAP_AABB)
    {
        return aabbsOverlap(query.geometry.data.aabb.min_corner, query.geometry.data.aabb.max_corner, objectBounds);
    }
    return boxOverlapsAabb(query, objectBounds);
}

bool exactGeometryIsAvailable(const GenerationLease& generation, const ParticipationBits& queryBits)
{
    if (generation.hasOnlyStoredExactGeometry())
    {
        return true;
    }
    for (size_t objectIndex = 0u; objectIndex < generation.objects().size(); ++objectIndex)
    {
        if (!generation.includesObject(objectIndex))
        {
            continue;
        }
        const PopulationObject& object = generation.objects()[objectIndex];
        if (objectParticipates(object, queryBits) && object.exactGeometry == nullptr)
        {
            return false;
        }
    }
    return true;
}

SemanticEvaluationOutcome semanticOutcome(ovspatialquery_api_status_t status)
{
    switch (status)
    {
    case OVSPATIALQUERY_API_SUCCESS: return SemanticEvaluationOutcome::success;
    case OVSPATIALQUERY_API_INVALID_ARGUMENT: return SemanticEvaluationOutcome::invalidArgument;
    case OVSPATIALQUERY_API_NOT_SUPPORTED: return SemanticEvaluationOutcome::notSupported;
    case OVSPATIALQUERY_API_OUT_OF_MEMORY: return SemanticEvaluationOutcome::outOfMemory;
    case OVSPATIALQUERY_API_HOST_BUDGET: return SemanticEvaluationOutcome::hostBudget;
    default: return SemanticEvaluationOutcome::internalFailure;
    }
}

SemanticHitIdentity semanticIdentity(const QueryLocationCandidate& candidate)
{
    return SemanticHitIdentity{candidate.objectId, candidate.geometry, candidate.materialId};
}

GeometryFeature semanticFeature(const QueryLocationCandidate& candidate)
{
    GeometryFeature feature{};
    if (!candidate.hasTriangleSource)
    {
        return feature;
    }
    feature.kind                = GeometryFeatureKind::triangle;
    feature.triangle.sourceFace = static_cast<uint64_t>(candidate.faceIndex);
    std::copy(candidate.sourceCorners, candidate.sourceCorners + 3u, feature.triangle.sourceCorners.begin());
    feature.triangle.barycentrics = {1.0f - candidate.uv[0] - candidate.uv[1], candidate.uv[0], candidate.uv[1]};
    return feature;
}

RaycastSemanticHit semanticRaycastHit(const QueryLocationCandidate& candidate, HitTruth truth)
{
    RaycastSemanticHit hit{};
    hit.truth = truth;
    if (truth == HitTruth::boundsCandidate)
    {
        hit.boundsCandidate.identity = semanticIdentity(candidate);
        std::copy(candidate.boundsPosition, candidate.boundsPosition + 3u, hit.boundsCandidate.boundsPosition.begin());
        std::copy(candidate.normal, candidate.normal + 3u, hit.boundsCandidate.boundsNormal.begin());
        hit.boundsCandidate.distance = candidate.distance;
    }
    else
    {
        hit.exact.identity = semanticIdentity(candidate);
        std::copy(candidate.position, candidate.position + 3u, hit.exact.position.begin());
        std::copy(candidate.normal, candidate.normal + 3u, hit.exact.normal.begin());
        hit.exact.distance = candidate.distance;
        hit.exact.feature  = semanticFeature(candidate);
    }
    return hit;
}

SphereSweepSemanticHit semanticSphereSweepHit(const QueryLocationCandidate& candidate, HitTruth truth)
{
    SphereSweepSemanticHit hit{};
    hit.truth = truth;
    if (truth == HitTruth::boundsCandidate)
    {
        hit.boundsCandidate.identity = semanticIdentity(candidate);
        std::copy(candidate.queryPosition, candidate.queryPosition + 3u, hit.boundsCandidate.queryPosition.begin());
        std::copy(candidate.boundsPosition, candidate.boundsPosition + 3u, hit.boundsCandidate.boundsPosition.begin());
        std::copy(candidate.normal, candidate.normal + 3u, hit.boundsCandidate.boundsNormal.begin());
        hit.boundsCandidate.distance = candidate.distance;
    }
    else
    {
        hit.exact.identity = semanticIdentity(candidate);
        std::copy(candidate.queryPosition, candidate.queryPosition + 3u, hit.exact.queryPosition.begin());
        std::copy(candidate.position, candidate.position + 3u, hit.exact.objectPosition.begin());
        std::copy(candidate.normal, candidate.normal + 3u, hit.exact.normal.begin());
        hit.exact.distance = candidate.distance;
        hit.exact.feature  = semanticFeature(candidate);
    }
    return hit;
}

struct OverlapEvaluationCandidate
{
    ovspatialquery_object_id_t objectId{};
    GeometryReference          geometry{};
    uint64_t                   materialId = 0u;
};

// An overlapped object as the tree reports it: its id, to order and deduplicate by, and its
// index, to read the rest from once. A giant overlap sorts thousands of these.
struct OverlappedObject
{
    ovspatialquery_object_id_t objectId{};
    uint32_t                   objectIndex = 0u;
};

// Orders per-object candidates nearest first by distance, then object id, and
// keeps one per object: the tree may report an object more than once.
void keepNearestPerObject(std::vector<QueryLocationCandidate>& candidates)
{
    std::sort(candidates.begin(), candidates.end(),
              [](const QueryLocationCandidate& left, const QueryLocationCandidate& right)
              {
                  return left.distance < right.distance ||
                         (left.distance == right.distance && objectIdLess(left.objectId, right.objectId));
              });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                 [](const QueryLocationCandidate& left, const QueryLocationCandidate& right)
                                 { return objectIdEqual(left.objectId, right.objectId); }),
                     candidates.end());
}

// Adds a MULTIPLE sweep's candidate within storage reserved for one per object,
// the peak its row prepays. The tree may report an object more than once, so a
// full vector first keeps each object's nearest candidate. If it is still full,
// every object already holds the candidate a repeated report evaluates to again.
void appendPerObjectCandidate(std::vector<QueryLocationCandidate>& candidates, const QueryLocationCandidate& candidate)
{
    if (candidates.size() == candidates.capacity())
    {
        keepNearestPerObject(candidates);
        if (candidates.size() == candidates.capacity())
        {
            return;
        }
    }
    candidates.push_back(candidate);
}

class PerObjectCandidatesAccumulator
{
  public:
    explicit PerObjectCandidatesAccumulator(std::vector<QueryLocationCandidate>& candidates) : mCandidates(candidates)
    {}

    bool accept(const QueryLocationCandidate& candidate)
    {
        try
        {
            appendPerObjectCandidate(mCandidates, candidate);
        }
        catch (const std::bad_alloc&)
        {
            return false;
        }
        return true;
    }

  private:
    std::vector<QueryLocationCandidate>& mCandidates;
};

// Emits the nearest `retainedHits` of a MULTIPLE sweep's candidates, one per
// object; the caller reads one more than its limit as truncation. The output is
// sized once, so it never holds more than one projected hit per object.
template <typename Emit, typename Reserve>
void emitNearestPerObject(std::vector<QueryLocationCandidate>& candidates, uint64_t retainedHits,
                          const Reserve& reserve, const Emit& emit)
{
    keepNearestPerObject(candidates);
    const size_t count = static_cast<size_t>(std::min<uint64_t>(candidates.size(), retainedHits));
    reserve(count);
    for (size_t index = 0u; index < count; ++index)
    {
        emit(candidates[index]);
    }
}

// Path evaluators deliver their final candidates, in result order, through
// emit(const QueryLocationCandidate&). Single-hit modes carry the one
// candidate on the stack; only complete multiple results collect a vector
// first.
template <typename Emit, typename Reserve = void (*)(size_t)>
SemanticEvaluationOutcome evaluateRaycastCandidates(
    const GenerationLease& generation, const RaycastQuery& source, QueryPrecision precision, HitMode mode,
    const ParticipationBits& queryBits, double contactTolerance, InstanceHostLedger* hostLedger, const Emit& emit,
    bool allCrossings = false, uint64_t retainedHits = UINT64_MAX, const Reserve& reserve = [](size_t) {}) noexcept
{
    PreparedPathQuery                 query{};
    const ovspatialquery_api_status_t preparationStatus =
        preparePathQuery(source.origin.data(), source.unitDirection.data(), source.minimumDistance,
                         source.maximumDistance, contactTolerance, query, generation.backend().rayOriginTolerance());
    if (preparationStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return semanticOutcome(preparationStatus);
    }
    query.hostLedger = hostLedger;
    query.bothSides  = source.bothSides;
    query.rayNormals = true;
    if (precision == QueryPrecision::exact && !exactGeometryIsAvailable(generation, queryBits))
    {
        return SemanticEvaluationOutcome::notSupported;
    }
    try
    {
        if (mode != HitMode::multiple)
        {
            // A closest exact ray, like a closest sweep, evaluates its candidates
            // nearest first and skips those entered beyond its running best's clip bound.
            ClosestCandidateAccumulator accumulate;
            ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
            if (precision == QueryPrecision::exact && mode == HitMode::closest)
            {
                status = evaluateOrderedClosestSweep(generation, query, queryBits, accumulate, kClosestRayClip);
            }
            else
            {
                PathCandidateSink<ClosestCandidateAccumulator> sink(generation, query, precision, mode, queryBits,
                                                                    accumulate);
                status = traverseBackendPathCandidates(generation, query, sink);
            }
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return semanticOutcome(status);
            }
            if (accumulate.hasHit())
            {
                emit(accumulate.candidate());
            }
            return SemanticEvaluationOutcome::success;
        }
        if (allCrossings)
        {
            // The nearest retainedHits crossings in order; the caller reads one
            // more than its limit as truncation.
            const size_t retained = static_cast<size_t>(std::min<uint64_t>(retainedHits, SIZE_MAX / 4u));
            std::vector<QueryLocationCandidate>            crossings;
            NearestCrossingsAccumulator                    accumulate(crossings, std::max<size_t>(retained, 1u),
                                                                      generation.objects().size(), hostLedger);
            PathCandidateSink<NearestCrossingsAccumulator> sink(generation, query, precision, mode, queryBits,
                                                                accumulate);
            const ovspatialquery_api_status_t status = traverseBackendPathCandidates(generation, query, sink);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return semanticOutcome(status);
            }
            // The accumulator expands each object once and PhysX reports each crossed
            // triangle once, so there is nothing to merge: as in PhysX, a ray
            // through a shared edge keeps one row per triangle.
            std::sort(crossings.begin(), crossings.end(), crossingLess);
            if (crossings.size() > retained)
            {
                crossings.resize(retained);
            }
            // Size the output once: the accumulator charged one projected hit per
            // candidate, which a growing output would exceed while it reallocates.
            reserve(crossings.size());
            for (const QueryLocationCandidate& candidate : crossings)
            {
                emit(candidate);
            }
            return SemanticEvaluationOutcome::success;
        }
        std::vector<QueryLocationCandidate> candidates;
        candidates.reserve(generation.objects().size());
        AllCandidatesAccumulator                    accumulate(candidates);
        PathCandidateSink<AllCandidatesAccumulator> sink(generation, query, precision, mode, queryBits, accumulate);
        const ovspatialquery_api_status_t           status = traverseBackendPathCandidates(generation, query, sink);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return semanticOutcome(status);
        }
        keepNearestPerObject(candidates);
        for (const QueryLocationCandidate& candidate : candidates)
        {
            emit(candidate);
        }
        return SemanticEvaluationOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return SemanticEvaluationOutcome::outOfMemory;
    }
    catch (...)
    {
        return SemanticEvaluationOutcome::internalFailure;
    }
}

// A closest box sweep clips each later object to the running best; an ANY
// sweep stops at its first contact; a MULTIPLE sweep keeps every object's.
class BoxSweepCandidateSink final : public BackendCandidateSink
{
  public:
    BoxSweepCandidateSink(const GenerationLease& generation, const SphereSweepQuery& source,
                          const BoxGeometrySource& box, QueryPrecision precision, HitMode mode,
                          const ParticipationBits& bits, InstanceHostLedger* ledger,
                          ClosestCandidateAccumulator& accumulate, std::vector<QueryLocationCandidate>& all)
        : mGeneration(generation), mSource(source), mBox(box), mPrecision(precision), mMode(mode), mBits(bits),
          mLedger(ledger), mAccumulate(accumulate), mAll(all)
    {}

    bool reportCandidate(uint32_t index) override
    {
        if (index >= mGeneration.objects().size())
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
            return false;
        }
        const auto& object = mGeneration.objects()[index];
        if (!mGeneration.includesObject(index) || !objectParticipates(object, mBits))
            return true;
        ExactGeometryDescription geometry;
        if (mPrecision == QueryPrecision::exact)
        {
            if (!object.exactGeometry || !object.exactGeometry->exportDescription(geometry))
            {
                mStatus = OVSPATIALQUERY_API_NOT_SUPPORTED;
                return false;
            }
            // A both-sides query sweeps every mesh as double-sided (ADR-0065).
            if (auto* mesh = std::get_if<TriangleMeshExactGeometryDescription>(&geometry); mesh && mSource.bothSides)
            {
                mesh->doubleSided = true;
            }
        }
        else
        {
            BoxExactGeometryDescription bounds{};
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                bounds.center[axis]      = object.bounds.minimum[axis] * 0.5 + object.bounds.maximum[axis] * 0.5;
                bounds.halfExtents[axis] = object.bounds.maximum[axis] * 0.5 - object.bounds.minimum[axis] * 0.5;
                bounds.axes[axis][axis]  = 1.0;
            }
            geometry = bounds;
        }
        ExactGeometryHit hit{};
        bool             hasHit          = false;
        const double     maximumDistance = mMode == HitMode::closest && mAccumulate.hasHit()
                                               ? std::min(mSource.maximumDistance, mAccumulate.candidate().distance)
                                               : mSource.maximumDistance;
        const auto*      targetBox       = std::get_if<BoxExactGeometryDescription>(&geometry);
        BoxSweepInterval interval{};
        // A closest sweep through many boxes completes only its winner's contact (finish).
        const bool deferContact = mMode == HitMode::closest && targetBox != nullptr;
        if (deferContact)
        {
            if (!sweepBoxBoxInterval(*targetBox, mBox, mSource.unitDirection.data(), mSource.minimumDistance,
                                     maximumDistance, interval))
                return true;
            hit.distance = interval.enter;
        }
        else
        {
            mStatus = sweepBoxGeometry(geometry, mBox, mSource.unitDirection.data(), mSource.minimumDistance,
                                       maximumDistance, hasHit, hit, mLedger);
            if (mStatus != OVSPATIALQUERY_API_SUCCESS)
                return false;
            if (!hasHit)
                return true;
        }
        QueryLocationCandidate candidate{};
        candidate.objectId = object.id;
        candidate.geometry = object.geometryReference;
        candidate.distance = hit.distance;
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            candidate.queryPosition[axis] =
                pathCoordinate(mSource.center[axis], mSource.unitDirection[axis], hit.distance);
            candidate.position[axis]       = hit.position[axis];
            candidate.boundsPosition[axis] = hit.position[axis];
            candidate.normal[axis]         = hit.normal[axis];
        }
        candidate.faceIndex = hit.sourceFace;
        std::copy(std::begin(hit.sourceCorners), std::end(hit.sourceCorners), candidate.sourceCorners);
        std::copy(std::begin(hit.uv), std::end(hit.uv), candidate.uv);
        candidate.hasTriangleSource = hit.hasTriangleSource;
        candidate.materialId        = hitMaterialId(object, hit.hasTriangleSource, hit.sourceFace);
        if (mMode != HitMode::multiple)
        {
            if (mMode == HitMode::closest && mAccumulate.beats(candidate))
            {
                mPendingBox = deferContact;
                if (deferContact)
                {
                    mPendingTarget   = *targetBox;
                    mPendingInterval = interval;
                }
            }
            mAccumulate.accept(candidate);
            // A closest sweep's traversal stops a few roundoffs past its running best.
            if (mMode == HitMode::closest)
                lowerReach(closestSweepReach(mSource.minimumDistance, mSource.maximumDistance,
                                             mAccumulate.candidate().distance));
            return mMode == HitMode::closest;
        }
        try
        {
            appendPerObjectCandidate(mAll, candidate);
        }
        catch (const std::bad_alloc&)
        {
            mStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
            return false;
        }
        return true;
    }

    ovspatialquery_api_status_t status() const override { return mStatus; }

    // The winner's contact, when it is a box whose contact reportCandidate deferred.
    void finish()
    {
        if (!mPendingBox)
            return;
        ExactGeometryHit hit{};
        sweepBoxBoxContact(mPendingTarget, mBox, mSource.unitDirection.data(), mPendingInterval, hit);
        QueryLocationCandidate& candidate = mAccumulate.candidate();
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            candidate.position[axis]       = hit.position[axis];
            candidate.boundsPosition[axis] = hit.position[axis];
            candidate.normal[axis]         = hit.normal[axis];
        }
        std::copy(std::begin(hit.uv), std::end(hit.uv), candidate.uv);
    }

  private:
    const GenerationLease&               mGeneration;
    const SphereSweepQuery&              mSource;
    const BoxGeometrySource&             mBox;
    QueryPrecision                       mPrecision;
    HitMode                              mMode;
    ParticipationBits                    mBits;
    InstanceHostLedger*                  mLedger;
    ClosestCandidateAccumulator&         mAccumulate;
    std::vector<QueryLocationCandidate>& mAll;
    ovspatialquery_api_status_t          mStatus     = OVSPATIALQUERY_API_SUCCESS;
    bool                                 mPendingBox = false;
    BoxExactGeometryDescription          mPendingTarget{};
    BoxSweepInterval                     mPendingInterval{};
};

template <typename Emit, typename Reserve>
SemanticEvaluationOutcome evaluateBoxSweepCandidates(const GenerationLease& generation, const SphereSweepQuery& source,
                                                     QueryPrecision precision, HitMode mode,
                                                     const ParticipationBits& queryBits, double contactTolerance,
                                                     InstanceHostLedger* hostLedger, uint64_t retainedHits,
                                                     const Reserve& reserve, const Emit& emit) noexcept
{
    PreparedPathQuery path{};
    auto status = preparePathQuery(source.center.data(), source.unitDirection.data(), source.minimumDistance,
                                   source.maximumDistance, 0.0, path);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return semanticOutcome(status);
    ovspatialquery_overlap_shape_t shape{};
    shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
    std::copy(source.center.begin(), source.center.end(), shape.data.box.center);
    std::copy(source.halfExtents.begin(), source.halfExtents.end(), shape.data.box.half_extents);
    std::copy(source.rotationXyzw.begin(), source.rotationXyzw.end(), shape.data.box.rotation_xyzw);
    PreparedOverlapQuery prepared{};
    status = prepareQueryOverlap(shape, contactTolerance, prepared);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return semanticOutcome(status);
    if (precision == QueryPrecision::exact && !exactGeometryIsAvailable(generation, queryBits))
        return SemanticEvaluationOutcome::notSupported;
    try
    {
        const auto                          box = makeQueryBoxGeometry(prepared.geometry.data.box);
        ClosestCandidateAccumulator         accumulate;
        std::vector<QueryLocationCandidate> all;
        if (mode == HitMode::multiple)
            all.reserve(generation.objects().size());
        BoxSweepCandidateSink sink(generation, source, box, precision, mode, queryBits, hostLedger, accumulate, all);
        // The AABB of the whole sweep, and the volume the query box's broadphase
        // AABB sweeps along the path. A path along an axis, or one no longer than
        // the box, sweeps most of that AABB, which the tree tests most cheaply. A
        // longer oblique path follows its swept box instead, so candidates follow
        // the path; so does a closest sweep longer than four box widths, whose
        // traversal stops at its running best.
        const double length = source.maximumDistance - source.minimumDistance;
        FloatAabb    swept{};
        double       half[3]{}, pathBox = 1.0, faces = 0.0;
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const double first = source.unitDirection[axis] * source.minimumDistance;
            const double last  = source.unitDirection[axis] * source.maximumDistance;
            swept.minimum[axis] =
                conservativeFloatBound(prepared.broadphaseMinimum[axis] + std::min(first, last), true);
            swept.maximum[axis] =
                conservativeFloatBound(prepared.broadphaseMaximum[axis] + std::max(first, last), false);
            half[axis] = 0.5 * (prepared.broadphaseMaximum[axis] - prepared.broadphaseMinimum[axis]);
            pathBox *= 2.0 * half[axis] + length * std::abs(double(source.unitDirection[axis]));
        }
        for (uint32_t axis = 0; axis < 3; ++axis)
            faces += half[(axis + 1) % 3] * half[(axis + 2) % 3] * std::abs(double(source.unitDirection[axis]));
        const double sweptVolume = 8.0 * half[0] * half[1] * half[2] + 4.0 * length * faces;
        const bool   follow      = pathBox > 2.0 * sweptVolume ||
                            (mode == HitMode::closest && length > 8.0 * std::max(std::max(half[0], half[1]), half[2]));
        if (!follow)
            status = generation.backend().aabbOverlapCandidates(swept, sink);
        else
        {
            // The box at the path's start: its float box contains the double
            // one, widened by four float roundoffs of the largest coordinate or
            // length the traversal computes with.
            FloatBoxSweepQuery traversal{};
            traversal.maxDistance = roundUpToFloat(length);
            double reach          = traversal.maxDistance;
            float  low[3]{}, high[3]{};
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                const double offset = source.unitDirection[axis] * source.minimumDistance;
                low[axis]           = conservativeFloatBound(prepared.broadphaseMinimum[axis] + offset, true);
                high[axis]          = conservativeFloatBound(prepared.broadphaseMaximum[axis] + offset, false);
                reach = std::max(std::max(reach, std::abs(double(low[axis]))), std::abs(double(high[axis])));
            }
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                const float center            = static_cast<float>(0.5 * double(low[axis]) + 0.5 * double(high[axis]));
                traversal.center[axis]        = center;
                traversal.unitDirection[axis] = source.unitDirection[axis];
                traversal.halfExtents[axis]   = roundUpToFloat(
                    std::max(double(high[axis]) - center, center - double(low[axis])) + std::ldexp(reach, -21));
            }
            status = generation.backend().boxSweepCandidates(traversal, sink);
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return semanticOutcome(status);
        sink.finish();
        if (mode == HitMode::multiple)
            emitNearestPerObject(all, retainedHits, reserve, emit);
        else if (accumulate.hasHit())
            emit(accumulate.candidate());
        return SemanticEvaluationOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return SemanticEvaluationOutcome::outOfMemory;
    }
    catch (...)
    {
        return SemanticEvaluationOutcome::internalFailure;
    }
}

template <typename Emit, typename Reserve>
SemanticEvaluationOutcome evaluateSphereSweepCandidates(const GenerationLease&  generation,
                                                        const SphereSweepQuery& source, QueryPrecision precision,
                                                        HitMode mode, const ParticipationBits& queryBits,
                                                        double contactTolerance, InstanceHostLedger* hostLedger,
                                                        uint64_t retainedHits, const Reserve& reserve,
                                                        const Emit& emit) noexcept
{
    if (source.box)
        return evaluateBoxSweepCandidates(generation, source, precision, mode, queryBits, contactTolerance, hostLedger,
                                          retainedHits, reserve, emit);
    if (!inInputDomain(source.radius) || source.radius <= 0.0)
    {
        return SemanticEvaluationOutcome::invalidArgument;
    }
    PreparedPathQuery                 query{};
    const ovspatialquery_api_status_t preparationStatus =
        preparePathQuery(source.center.data(), source.unitDirection.data(), source.minimumDistance,
                         source.maximumDistance, source.radius + contactTolerance, query);
    if (preparationStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return semanticOutcome(preparationStatus);
    }
    query.hostLedger = hostLedger;
    query.bothSides  = source.bothSides;
    if (precision == QueryPrecision::exact && !exactGeometryIsAvailable(generation, queryBits))
    {
        return SemanticEvaluationOutcome::notSupported;
    }
    try
    {
        if (mode == HitMode::multiple)
        {
            std::vector<QueryLocationCandidate> candidates;
            candidates.reserve(generation.objects().size());
            PerObjectCandidatesAccumulator                    accumulate(candidates);
            PathCandidateSink<PerObjectCandidatesAccumulator> sink(generation, query, precision, mode, queryBits,
                                                                   accumulate);
            const ovspatialquery_api_status_t status = traverseBackendPathCandidates(generation, query, sink);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return semanticOutcome(status);
            }
            emitNearestPerObject(candidates, retainedHits, reserve, emit);
            return SemanticEvaluationOutcome::success;
        }
        ClosestCandidateAccumulator accumulate;
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        if (precision == QueryPrecision::exact && mode == HitMode::closest)
        {
            OrderedClosestSweepSink sink(generation, query, queryBits, accumulate);
            status = traverseBackendPathCandidates(generation, query, sink);
            if (status == OVSPATIALQUERY_API_SUCCESS && !sink.flush())
            {
                status = sink.status();
            }
        }
        else
        {
            PathCandidateSink<ClosestCandidateAccumulator> sink(generation, query, precision, mode, queryBits,
                                                                accumulate);
            if (mode == HitMode::closest)
            {
                sink.clipToClosest();
            }
            status = traverseBackendPathCandidates(generation, query, sink);
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return semanticOutcome(status);
        }
        if (accumulate.hasHit())
        {
            emit(accumulate.candidate());
        }
        return SemanticEvaluationOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return SemanticEvaluationOutcome::outOfMemory;
    }
    catch (...)
    {
        return SemanticEvaluationOutcome::internalFailure;
    }
}

class OverlapCandidateSink final : public BackendCandidateSink
{
  public:
    OverlapCandidateSink(const GenerationLease& generation, const PreparedOverlapQuery& query, QueryPrecision precision,
                         HitMode mode, const ParticipationBits& queryBits, InstanceHostLedger* hostLedger,
                         std::vector<OverlappedObject>& result)
        : mGeneration(generation), mQuery(query), mPrecision(precision), mMode(mode), mQueryBits(queryBits),
          mHostLedger(hostLedger), mResult(result)
    {}

    bool reportCandidate(uint32_t objectIndex) override
    {
        if (objectIndex >= mGeneration.objects().size())
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
            return false;
        }
        if (!mGeneration.includesObject(objectIndex))
        {
            return true;
        }
        const PopulationObject& object = mGeneration.objects()[objectIndex];
        if (!objectParticipates(object, mQueryBits))
        {
            return true;
        }
        try
        {
            bool overlaps = false;
            if (mPrecision == QueryPrecision::exact)
            {
                // The analytic families take the BOX query's prepared rotation
                // for every row instead of rebuilding the same matrix for every
                // candidate. Hulls, meshes and user geometry take it only for a
                // float64 row: it moves them off PhysX's float32 test.
                const ExactGeometryFamily family = object.geometryReference.family;
                const bool analytic = family == ExactGeometryFamily::sphere || family == ExactGeometryFamily::box ||
                                      family == ExactGeometryFamily::capsule;
                const bool oriented =
                    mQuery.geometry.kind == OVSPATIALQUERY_OVERLAP_BOX && (mQuery.exactBoxRotation || analytic);
                mStatus  = OVSPATIALQUERY_API_SUCCESS;
                overlaps = object.exactGeometry->overlaps(mQuery.geometry, oriented ? mQuery.boxRotation : nullptr,
                                                          mHostLedger, &mStatus);
                if (mStatus != OVSPATIALQUERY_API_SUCCESS)
                    return false;
            }
            else
            {
                overlaps = exactOverlap(mQuery, object.bounds);
            }
            if (!overlaps)
            {
                return true;
            }
            mResult.push_back({object.id, objectIndex});
            return mMode != HitMode::any;
        }
        catch (const std::bad_alloc&)
        {
            mStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (...)
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
        }
        return false;
    }

    ovspatialquery_api_status_t status() const override { return mStatus; }

  private:
    const GenerationLease&         mGeneration;
    const PreparedOverlapQuery&    mQuery;
    QueryPrecision                 mPrecision;
    HitMode                        mMode;
    ParticipationBits              mQueryBits;
    InstanceHostLedger*            mHostLedger;
    std::vector<OverlappedObject>& mResult;
    ovspatialquery_api_status_t    mStatus = OVSPATIALQUERY_API_SUCCESS;
};

template <typename Emit>
SemanticEvaluationOutcome evaluateOverlapCandidates(const GenerationLease& generation, const OverlapQuery& source,
                                                    QueryPrecision precision, HitMode mode,
                                                    const ParticipationBits& queryBits, double contactTolerance,
                                                    InstanceHostLedger* hostLedger, const Emit& emit) noexcept
{
    PreparedOverlapQuery              query{};
    const ovspatialquery_api_status_t preparationStatus = prepareQueryOverlap(
        source.geometry, contactTolerance, query, source.hasBoxRotation ? source.boxRotationXyzw.data() : nullptr);
    if (preparationStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return semanticOutcome(preparationStatus);
    }
    if (precision == QueryPrecision::exact && !exactGeometryIsAvailable(generation, queryBits))
    {
        return SemanticEvaluationOutcome::notSupported;
    }
    try
    {
        std::vector<OverlappedObject>     result;
        OverlapCandidateSink              sink(generation, query, precision, mode, queryBits, hostLedger, result);
        const ovspatialquery_api_status_t status = generation.backend().aabbOverlapCandidates(query.broadphase, sink);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return semanticOutcome(status);
        }
        std::sort(result.begin(), result.end(),
                  [](const OverlappedObject& left, const OverlappedObject& right)
                  { return objectIdLess(left.objectId, right.objectId); });
        result.erase(std::unique(result.begin(), result.end(),
                                 [](const OverlappedObject& left, const OverlappedObject& right)
                                 { return objectIdEqual(left.objectId, right.objectId); }),
                     result.end());
        for (const OverlappedObject& overlapped : result)
        {
            const PopulationObject& object = generation.objects()[overlapped.objectIndex];
            emit(OverlapEvaluationCandidate{object.id, object.geometryReference, object.materialId});
        }
        return SemanticEvaluationOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return SemanticEvaluationOutcome::outOfMemory;
    }
    catch (...)
    {
        return SemanticEvaluationOutcome::internalFailure;
    }
}

} // namespace

// Each entry clears outHits first and holds the projected hits on success; a
// failed evaluation leaves it cleared or partially filled, never a stale prior
// result. Reusing the caller's storage keeps a warm single-hit row
// allocation-free. Every evaluator emits inside its own try block.
SemanticEvaluationOutcome evaluateSemanticRaycast(const GenerationLease& generation, const RaycastQuery& source,
                                                  QueryPrecision precision, HitMode mode,
                                                  const ParticipationBits&         queryBits,
                                                  std::vector<RaycastSemanticHit>& outHits, double contactTolerance,
                                                  InstanceHostLedger* hostLedger, bool allMeshHits,
                                                  uint64_t retainedHits) noexcept
{
    const HitTruth truth = precision == QueryPrecision::exact ? HitTruth::exactSurface : HitTruth::boundsCandidate;
    outHits.clear();
    return evaluateRaycastCandidates(
        generation, source, precision, mode, queryBits, contactTolerance, hostLedger,
        [&](const QueryLocationCandidate& candidate) { outHits.push_back(semanticRaycastHit(candidate, truth)); },
        allMeshHits && mode == HitMode::multiple, retainedHits, [&](size_t count) { outHits.reserve(count); });
}

SemanticEvaluationOutcome evaluateSemanticSphereSweep(const GenerationLease& generation, const SphereSweepQuery& source,
                                                      QueryPrecision precision, HitMode mode,
                                                      const ParticipationBits&             queryBits,
                                                      std::vector<SphereSweepSemanticHit>& outHits,
                                                      double contactTolerance, InstanceHostLedger* hostLedger,
                                                      uint64_t retainedHits) noexcept
{
    const HitTruth truth = precision == QueryPrecision::exact ? HitTruth::exactSurface : HitTruth::boundsCandidate;
    outHits.clear();
    return evaluateSphereSweepCandidates(
        generation, source, precision, mode, queryBits, contactTolerance, hostLedger, retainedHits,
        [&](size_t count)
        {
            // Free a reused row's smaller storage before sizing it, so the old
            // and new storage are never held together.
            if (count > outHits.capacity())
            {
                std::vector<SphereSweepSemanticHit>().swap(outHits);
                outHits.reserve(count);
            }
        },
        [&](const QueryLocationCandidate& candidate) { outHits.push_back(semanticSphereSweepHit(candidate, truth)); });
}

SemanticEvaluationOutcome evaluateSemanticOverlap(const GenerationLease& generation, const OverlapQuery& source,
                                                  QueryPrecision precision, HitMode mode,
                                                  const ParticipationBits&         queryBits,
                                                  std::vector<OverlapSemanticHit>& outHits, double contactTolerance,
                                                  InstanceHostLedger* hostLedger) noexcept
{
    const HitTruth truth = precision == QueryPrecision::exact ? HitTruth::exactSurface : HitTruth::boundsCandidate;
    outHits.clear();
    return evaluateOverlapCandidates(
        generation, source, precision, mode, queryBits, contactTolerance, hostLedger,
        [&](const OverlapEvaluationCandidate& candidate)
        {
            outHits.push_back(OverlapSemanticHit{
                truth, SemanticHitIdentity{candidate.objectId, candidate.geometry, candidate.materialId}});
        });
}

// Peak bytes per object: one candidate plus the larger of a broadphase index
// and a projected hit.
bool estimateSemanticRaycastPeakBytes(uint64_t objectCount, uint64_t& outBytes) noexcept
{
    outBytes = 0u;
    return addHostCharge(
        objectCount, sizeof(QueryLocationCandidate) + std::max(sizeof(uint32_t), sizeof(RaycastSemanticHit)), outBytes);
}

bool estimateSemanticSphereSweepPeakBytes(uint64_t objectCount, HitMode mode, uint64_t& outBytes) noexcept
{
    if (mode == HitMode::multiple)
    {
        // A multiple sweep holds at most one candidate per object, however often
        // the tree reports one, and projects at most one hit per object into
        // storage sized once.
        outBytes = 0u;
        return addHostCharge(
            objectCount, sizeof(QueryLocationCandidate) + std::max(sizeof(uint32_t), sizeof(SphereSweepSemanticHit)),
            outBytes);
    }
    // A single-hit sweep holds the closest candidate and its hit.
    outBytes = sizeof(QueryLocationCandidate) + sizeof(SphereSweepSemanticHit);
    return true;
}

bool estimateSemanticOverlapPeakBytes(uint64_t objectCount, uint64_t& outBytes) noexcept
{
    outBytes = 0u;
    return addHostCharge(objectCount, sizeof(OverlappedObject) + std::max(sizeof(uint32_t), sizeof(OverlapSemanticHit)),
                         outBytes);
}

} // namespace detail
} // namespace ovspatialquery
