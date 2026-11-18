/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_physx_backend.h"
#include "ovspatialquery_backend.h"
#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_geometry.h"
#include "ovspatialquery_instance_host_ledger.h"
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
#include "ovspatialquery_test_gate.h"
#endif

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

std::mutex                        gDefaultRuntimeMutex;
ovspatialquery_backend_runtime_t* gDefaultRuntime               = nullptr;
uint64_t                          gDefaultRuntimeReferenceCount = 0u;

struct PxConvexMeshReleaser
{
    void operator()(physx::PxConvexMesh* mesh) const noexcept
    {
        if (mesh != nullptr)
        {
            mesh->release();
        }
    }
};

} // namespace

struct ovspatialquery_backend_runtime_t
{
    std::shared_ptr<ovspatialquery::detail::PhysXInfrastructure> infrastructure;

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    ovspatialquery::detail::TestGate rayGate;
#endif
};

namespace ovspatialquery
{
namespace detail
{

ConvexHullPreparationOutcome prepareConvexHull(ovspatialquery_backend_runtime_t* backendRuntime,
                                               const std::vector<float>& authoredPoints, uint64_t maximumResidentBytes,
                                               std::shared_ptr<const PreparedConvexHull>& outHull,
                                               const ConvexHullPreparationBudget*         budget,
                                               const ConvexHullCookingOptions& options, double* outFarthestOutside)
{
    if (backendRuntime == nullptr)
    {
        return ConvexHullPreparationOutcome::invalidArgument;
    }
    return prepareConvexHullFrom(
        authoredPoints, maximumResidentBytes, outHull, budget,
        [backendRuntime, &authoredPoints](const ConvexPointSet& points, uint32_t vertexLimit,
                                          std::vector<double>& vertices, std::vector<uint32_t>& faceVertexCounts,
                                          std::vector<uint32_t>&                faceVertexIndices,
                                          std::shared_ptr<physx::PxConvexMesh>& cookedMesh, CookedFrame& cookedFrame)
        {
            // Without a limit PhysX cooks the authored floats, deduplicated. Under its largest
            // limits PhysX's hull has at most 255 vertices and 254 polygons, the limits a hull may
            // have (ADR-0074). Past the polygon limit it reports ePOLYGONS_LIMIT_REACHED. Past the
            // vertex limit, plane shifting stops its quickhull at the limit and leaves the
            // remaining points outside, which prepareConvexHullFrom's enclosure check refuses;
            // PhysX's default algorithm instead crops an oriented box, and a Debug PhysX asserts
            // there on hulls such as a 128-gon prism.
            // PhysX's hull tolerances are absolute, so the library cooks those points moved to
            // their box's centre and scaled by the power of two that brings the largest coordinate into [0.5, 1);
            // where it judges points coplanar then depends on their extent, not on how far they
            // sit from the origin (ADR-0076). On each axis the centre is a float that is at least
            // half and at most twice every coordinate, so subtracting it is exact (Sterbenz); an
            // axis whose box is near zero keeps origin 0, where the largest coordinate is already
            // within 1.5 times the extent. Both steps are exact, so the hull's vertices map back
            // to the authored points exactly.
            // Under a vertex limit PhysX cooks the authored floats as given, every point in its
            // order and at its own coordinates, as Omni Physics cooks a convexHull collider
            // (ADR-0078): which points the quickhull keeps depends on their order, and its
            // tolerances' absolute floors on their frame. The cooked vertices are then the hull's.
            const bool limited = vertexLimit != 0u;
            if (!limited)
            {
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    const double minimum = points.minimum[axis], maximum = points.maximum[axis];
                    const double centre      = float(0.5 * (minimum + maximum));
                    const bool   exact       = centre > 0.0
                                                   ? 0.5 * centre <= minimum && maximum <= 2.0 * centre
                                                   : centre < 0.0 && 0.5 * centre >= maximum && minimum >= 2.0 * centre;
                    cookedFrame.origin[axis] = exact ? centre : 0.0;
                }
                double largest = 0.0;
                for (const auto& point : points.authored)
                    for (uint32_t axis = 0u; axis < 3u; ++axis)
                        largest = std::max(largest, std::abs(point[axis] - cookedFrame.origin[axis]));
                int exponent = 0;
                std::frexp(largest, &exponent);
                cookedFrame.scale = std::ldexp(1.0, -exponent);
            }
            const double               cookedFromLocal = cookedFrame.scale;
            const auto&                origin          = cookedFrame.origin;
            std::vector<physx::PxVec3> localVertices;
            if (!limited)
            {
                localVertices.reserve(points.authored.size());
                for (const auto& point : points.authored)
                    localVertices.emplace_back(float((point[0] - origin[0]) * cookedFromLocal),
                                               float((point[1] - origin[1]) * cookedFromLocal),
                                               float((point[2] - origin[2]) * cookedFromLocal));
            }
            physx::PxConvexMeshDesc descriptor{};
            descriptor.points.count =
                static_cast<physx::PxU32>(limited ? authoredPoints.size() / 3u : localVertices.size());
            descriptor.points.stride = sizeof(physx::PxVec3);
            descriptor.points.data   = limited ? static_cast<const void*>(authoredPoints.data())
                                               : static_cast<const void*>(localVertices.data());
            physx::PxTolerancesScale               tolerances{};
            physx::PxCookingParams                 parameters(tolerances);
            physx::PxConvexMeshCookingResult::Enum condition = physx::PxConvexMeshCookingResult::eFAILURE;
            if (limited)
            {
                // The hull PhysX simulates for a convexHull collider (ADR-0078): Omni Physics cooks
                // with eCOMPUTE_CONVEX alone, its default plane tolerance and GPU data, under the
                // authored vertex limit. Past the limit PhysX crops an oriented box with the faces
                // of a quickhull stopped there, so its corners need not be points and points may
                // lie outside it. Its tolerances scale with the points' extent but have absolute
                // floors. The simulation's tolerance scale, 1 / metersPerUnit, enters only PhysX's
                // replacement of points whose box is thinner than 1e-6 of it by a box, a mesh the
                // simulation thickens first, which ADR-0078 leaves out.
                descriptor.flags        = physx::PxConvexFlag::eCOMPUTE_CONVEX;
                descriptor.vertexLimit  = static_cast<physx::PxU16>(vertexLimit);
                parameters.buildGPUData = true;
            }
            else
            {
                descriptor.flags       = physx::PxConvexFlag::eCOMPUTE_CONVEX | physx::PxConvexFlag::ePLANE_SHIFTING;
                descriptor.vertexLimit = static_cast<physx::PxU16>(kMaximumConvexHullVertices);
                // The default plane tolerance drops points near the hull and merges faces up to 1e-2
                // of the extent off their plane; zero keeps every hull point, so polygons are planar
                // to float precision and the authored points stay enclosed.
                parameters.planeTolerance = 0.0f;
            }
            descriptor.polygonLimit = static_cast<physx::PxU16>(255u);
            std::unique_ptr<physx::PxConvexMesh, PxConvexMeshReleaser> mesh(PxCreateConvexMesh(
                parameters, descriptor, *physxInsertionCallback(backendRuntime->infrastructure), &condition));
            // Under a vertex limit PhysX reports the polygon limit when its quickhull reached 255
            // polygons before cropping to the limit, and the cropped hull is the collider.
            if ((!limited && condition == physx::PxConvexMeshCookingResult::ePOLYGONS_LIMIT_REACHED) ||
                (mesh != nullptr && mesh->getNbPolygons() > kMaximumConvexHullFaces))
                return ConvexHullPreparationOutcome::beyondFaceLimit;
            if (mesh == nullptr)
                return ConvexHullPreparationOutcome::topologyRejected;
            const physx::PxU32   vertexCount    = mesh->getNbVertices();
            const physx::PxVec3* cookedVertices = mesh->getVertices();
            const physx::PxU8*   cookedIndices  = mesh->getIndexBuffer();
            const physx::PxU32   polygonCount   = mesh->getNbPolygons();
            if (vertexCount < 4u || cookedVertices == nullptr || cookedIndices == nullptr || polygonCount < 4u)
                return ConvexHullPreparationOutcome::topologyRejected;
            const bool beyondVertexLimit = localVertices.size() > kMaximumConvexHullVertices;
            if (!limited && (vertexCount == 8u || beyondVertexLimit))
            {
                // A cooked corner that is not an input point is PhysX's thickened box for points it
                // judges coplanar, which it cooks from 8 corners (ADR-0076), or, past the vertex
                // limit, a plane-shifted corner: this PhysX keeps input points there, but its
                // documentation allows otherwise. Plane shifting acts only past the limit, so other
                // hulls skip the check and its sort.
                const auto lexicographic = [](const physx::PxVec3& a, const physx::PxVec3& b)
                { return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z); };
                std::sort(localVertices.begin(), localVertices.end(), lexicographic);
                for (physx::PxU32 index = 0u; index < vertexCount; ++index)
                    if (!std::binary_search(localVertices.begin(), localVertices.end(), cookedVertices[index],
                                            lexicographic))
                        return beyondVertexLimit ? ConvexHullPreparationOutcome::beyondVertexLimit
                                                 : ConvexHullPreparationOutcome::topologyRejected;
            }
            // Without a limit the cooked vertices map back to input points exactly; under one they
            // are the hull's own float32 coordinates.
            vertices.reserve(static_cast<size_t>(vertexCount) * 3u);
            for (physx::PxU32 index = 0u; index < vertexCount; ++index)
                vertices.insert(vertices.end(), {cookedVertices[index].x / cookedFromLocal + origin[0],
                                                 cookedVertices[index].y / cookedFromLocal + origin[1],
                                                 cookedVertices[index].z / cookedFromLocal + origin[2]});
            faceVertexCounts.reserve(polygonCount);
            for (physx::PxU32 polygonIndex = 0u; polygonIndex < polygonCount; ++polygonIndex)
            {
                physx::PxHullPolygon polygon{};
                if (!mesh->getPolygonData(polygonIndex, polygon) || polygon.mNbVerts < 3u)
                    return ConvexHullPreparationOutcome::topologyRejected;
                const size_t first = faceVertexIndices.size();
                for (physx::PxU16 corner = 0u; corner < polygon.mNbVerts; ++corner)
                {
                    const uint32_t vertex = cookedIndices[static_cast<size_t>(polygon.mIndexBase) + corner];
                    if (vertex >= vertexCount)
                        return ConvexHullPreparationOutcome::topologyRejected;
                    faceVertexIndices.push_back(vertex);
                }
                const physx::PxVec3 a = cookedVertices[faceVertexIndices[first]];
                const physx::PxVec3 b = cookedVertices[faceVertexIndices[first + 1u]];
                const physx::PxVec3 c = cookedVertices[faceVertexIndices[first + 2u]];
                if ((b - a).cross(c - a).dot(physx::PxVec3(polygon.mPlane[0], polygon.mPlane[1], polygon.mPlane[2])) <
                    0.0f)
                    std::reverse(faceVertexIndices.begin() + static_cast<std::ptrdiff_t>(first),
                                 faceVertexIndices.end());
                faceVertexCounts.push_back(polygon.mNbVerts);
            }
            cookedMesh = std::shared_ptr<physx::PxConvexMesh>(
                mesh.release(),
                [infrastructure = backendRuntime->infrastructure](physx::PxConvexMesh* retained)
                {
                    (void)infrastructure; // Keep the SDK alive until its cooked hull is released.
                    retained->release();
                });
            return ConvexHullPreparationOutcome::success;
        },
        options, outFarthestOutside);
}

ovspatialquery_api_status_t acquireDefaultBackendRuntime(ovspatialquery_backend_runtime_t** outRuntime)
{
    if (outRuntime == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    *outRuntime = nullptr;

    std::lock_guard<std::mutex> lock(gDefaultRuntimeMutex);
    if (gDefaultRuntime == nullptr)
    {
        std::unique_ptr<ovspatialquery_backend_runtime_t> runtime(new (std::nothrow)
                                                                      ovspatialquery_backend_runtime_t());
        if (runtime == nullptr)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        const ovspatialquery_api_status_t status = acquirePhysXInfrastructure(runtime->infrastructure);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        gDefaultRuntime = runtime.release();
    }
    ++gDefaultRuntimeReferenceCount;
    *outRuntime = gDefaultRuntime;
    return OVSPATIALQUERY_API_SUCCESS;
}

void releaseDefaultBackendRuntime(ovspatialquery_backend_runtime_t* runtime)
{
    std::lock_guard<std::mutex> lock(gDefaultRuntimeMutex);
    if (runtime == nullptr || runtime != gDefaultRuntime || gDefaultRuntimeReferenceCount == 0u)
    {
        return;
    }
    --gDefaultRuntimeReferenceCount;
    if (gDefaultRuntimeReferenceCount == 0u)
    {
        delete gDefaultRuntime;
        gDefaultRuntime = nullptr;
    }
}

} // namespace detail
} // namespace ovspatialquery

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
void ovspatialquery_physx_test_block_rays(ovspatialquery_backend_runtime_t* runtime, uint32_t expected_rays)
{
    ovspatialquery::detail::configureTestGate(runtime->rayGate, expected_rays);
}

uint32_t ovspatialquery_physx_test_wait_for_blocked_rays(ovspatialquery_backend_runtime_t* runtime)
{
    return ovspatialquery::detail::waitForTestGate(runtime->rayGate);
}

void ovspatialquery_physx_test_release_rays(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::releaseTestGate(runtime->rayGate);
}
#endif

namespace ovspatialquery
{
namespace detail
{
namespace
{

physx::PxBounds3 toPxBounds(const FloatAabb& bounds)
{
    return physx::PxBounds3(physx::PxVec3(bounds.minimum[0], bounds.minimum[1], bounds.minimum[2]),
                            physx::PxVec3(bounds.maximum[0], bounds.maximum[1], bounds.maximum[2]));
}

FloatAabb makePopulationProxy(const ovspatialquery_aabb_t& bounds)
{
    FloatAabb proxy{};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        proxy.minimum[axis] = conservativeFloatBound(bounds.minimum[axis], true);
        proxy.maximum[axis] = conservativeFloatBound(bounds.maximum[axis], false);
    }
    return proxy;
}

float proxyMagnitude(const physx::PxBounds3& proxy)
{
    float magnitude = 0.0f;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        magnitude = std::max(magnitude, std::max(std::abs(proxy.minimum[axis]), std::abs(proxy.maximum[axis])));
    }
    return magnitude;
}

// A float query origin rounds by at most one ulp per axis of its own
// magnitude. Inflating the tree bounds by that margin for every coordinate
// the population reaches lets a plain ray traversal stay conservative for
// origins inside twice that range, instead of sweeping a sphere of the
// rounding radius.
float rayOriginMargin(float maximumMagnitude) { return std::ldexp(maximumMagnitude, -23); }

// A tree leaf: the exact proxy grown by twice the margin in float arithmetic,
// whose own rounding is below the extra margin; safe proxies stay finite.
physx::PxBounds3 treeLeaf(const physx::PxBounds3& proxy, float margin)
{
    const physx::PxVec3 inflation(2.0f * margin);
    return physx::PxBounds3(proxy.minimum - inflation, proxy.maximum + inflation);
}

// A tree cooked once and loaded by every version derived from the version
// that cooked it (ADR-0019): a PxBVH can be neither cloned nor share nodes,
// so each version owns a loaded copy refitted at the objects changed since
// the cook. A derived version keeps the cook's margin while no changed
// object reaches beyond the cook's largest coordinate.
struct CookedTree
{
    std::vector<uint8_t> bytes;
    float                maximumMagnitude = 0.0f;
    float                margin           = 0.0f;
};

physx::PxBVH* createBvh(const std::vector<physx::PxBounds3>& bounds, physx::PxInsertionCallback& insertionCallback)
{
    if (bounds.empty())
    {
        return nullptr;
    }
    physx::PxBVHDesc descriptor;
    descriptor.enlargement   = 0.0f;
    descriptor.bounds.count  = static_cast<physx::PxU32>(bounds.size());
    descriptor.bounds.stride = sizeof(physx::PxBounds3);
    descriptor.bounds.data   = bounds.data();
    return PxCreateBVH(descriptor, insertionCallback);
}

// Every object is a tree leaf at its own index (ADR-0032 keeps every
// population bound inside PhysX's float range).
class PairCallback final : public physx::PxReportCallback<physx::PxGeomIndexPair>
{
  public:
    PairCallback(size_t firstObjectCount, size_t secondObjectCount, BackendPairSink& sink)
        : mFirstObjectCount(firstObjectCount), mSecondObjectCount(secondObjectCount), mSink(sink)
    {}

    bool flushResults(physx::PxU32 count, const physx::PxGeomIndexPair* items) override
    {
        for (physx::PxU32 index = 0; index < count; ++index)
        {
            if (items[index].id0 >= mFirstObjectCount || items[index].id1 >= mSecondObjectCount)
            {
                mValid = false;
                return false;
            }
            if (!mSink.reportPair(items[index].id0, items[index].id1))
            {
                return false;
            }
        }
        return true;
    }

    bool valid() const { return mValid; }

  private:
    size_t           mFirstObjectCount;
    size_t           mSecondObjectCount;
    BackendPairSink& mSink;
    bool             mValid = true;
};

// The candidate raycast and overlap callbacks differ only in the PhysX
// report signature; the bounds-index check and the sink handshake are one.
template <typename PhysXCallback>
class CandidateCallback : public PhysXCallback
{
  public:
    CandidateCallback(size_t objectCount, BackendCandidateSink& sink) : mObjectCount(objectCount), mSink(sink) {}

    bool stopped() const { return mStopped; }

    ovspatialquery_api_status_t status() const
    {
        return mStatus == OVSPATIALQUERY_API_SUCCESS ? mSink.status() : mStatus;
    }

  protected:
    float traversalReach() const { return mSink.traversalReach(); }

    bool reportCandidate(physx::PxU32 boundsIndex)
    {
        if (boundsIndex >= mObjectCount)
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
            return false;
        }
        if (!mSink.reportCandidate(boundsIndex))
        {
            mStopped = true;
            mStatus  = mSink.status();
            return false;
        }
        return true;
    }

  private:
    size_t                      mObjectCount;
    BackendCandidateSink&       mSink;
    ovspatialquery_api_status_t mStatus  = OVSPATIALQUERY_API_SUCCESS;
    bool                        mStopped = false;
};

class CandidateRaycastCallback final : public CandidateCallback<physx::PxBVH::RaycastCallback>
{
  public:
    using CandidateCallback::CandidateCallback;

    // `distance` is the traversal's current reach; lowering it skips what lies beyond.
    bool reportHit(physx::PxU32 boundsIndex, physx::PxReal& distance) override
    {
        if (!reportCandidate(boundsIndex))
        {
            return false;
        }
        distance = std::min(distance, traversalReach());
        return true;
    }
};

class CandidateOverlapCallback final : public CandidateCallback<physx::PxBVH::OverlapCallback>
{
  public:
    using CandidateCallback::CandidateCallback;

    bool reportHit(physx::PxU32 boundsIndex) override { return reportCandidate(boundsIndex); }
};

class PhysXPopulation final : public BackendPopulation
{
  public:
    explicit PhysXPopulation(ovspatialquery_backend_runtime_t* runtime) : mRuntime(runtime) {}

    ~PhysXPopulation() override
    {
        if (mBvh != nullptr)
        {
            mBvh->release();
        }
    }

    ovspatialquery_api_status_t update(const ObjectTable& objects) override
    {
        if (objects.size() > static_cast<size_t>(std::numeric_limits<physx::PxU32>::max()))
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        std::vector<physx::PxBounds3>           candidateBounds;
        std::vector<ovspatialquery_object_id_t> candidateIds;
        candidateBounds.reserve(objects.size());
        candidateIds.reserve(objects.size());
        float maximumMagnitude = 0.0f;
        for (const PopulationObject& object : objects)
        {
            candidateIds.push_back(object.id);
            candidateBounds.push_back(toPxBounds(makePopulationProxy(object.bounds)));
            maximumMagnitude = std::max(maximumMagnitude, proxyMagnitude(candidateBounds.back()));
        }

        // mBounds keeps the exact proxies; the tree holds them grown by the margin.
        const float                   candidateMargin = rayOriginMargin(maximumMagnitude);
        std::vector<physx::PxBounds3> treeBounds;
        treeBounds.reserve(candidateBounds.size());
        for (const physx::PxBounds3& bounds : candidateBounds)
        {
            treeBounds.push_back(treeLeaf(bounds, candidateMargin));
        }
        physx::PxBVH* candidateBvh = createBvh(treeBounds, *physxInsertionCallback(mRuntime->infrastructure));
        if (!candidateBounds.empty() && candidateBvh == nullptr)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        if (mBvh != nullptr)
        {
            mBvh->release();
        }
        mBvh                = candidateBvh;
        mRayOriginTolerance = static_cast<double>(candidateMargin);
        mBounds.swap(candidateBounds);
        mIds.swap(candidateIds);
        mCooked.reset();
        mDirty.reset();
        return OVSPATIALQUERY_API_SUCCESS;
    }

    // Reuses the base's cooked tree when the objects kept their count and
    // identities: a loaded copy refitted at every object
    // changed since the cook. More than a quarter changed since the cook, or a
    // changed object beyond the cook's reach, cooks this version's tree for
    // its successors; anything else rebuilds as update() does.
    ovspatialquery_api_status_t derive(const BackendPopulation& base, const ObjectTable& objects,
                                       const std::vector<uint32_t>& changedPositions) override
    {
        const PhysXPopulation* previous = static_cast<const PhysXPopulation*>(&base);
        if (previous->mBvh == nullptr || objects.size() != previous->mIds.size())
        {
            return update(objects);
        }
        std::vector<physx::PxBounds3> candidateBounds(previous->mBounds);
        float                         changedMagnitude = 0.0f;
        for (const uint32_t position : changedPositions)
        {
            if (position >= objects.size() || !objectIdEqual(objects[position].id, previous->mIds[position]))
            {
                return update(objects);
            }
            candidateBounds[position] = toPxBounds(makePopulationProxy(objects[position].bounds));
            changedMagnitude          = std::max(changedMagnitude, proxyMagnitude(candidateBounds[position]));
        }
        // Everything that allocates happens before this version holds a tree.
        std::vector<ovspatialquery_object_id_t> candidateIds(previous->mIds);
        std::shared_ptr<std::vector<uint32_t>>  dirty  = std::make_shared<std::vector<uint32_t>>();
        std::shared_ptr<const CookedTree>       cooked = previous->mCooked;
        physx::PxBVH*                           tree   = nullptr;
        if (cooked != nullptr && changedMagnitude <= cooked->maximumMagnitude)
        {
            std::set_union(previous->mDirty->begin(), previous->mDirty->end(), changedPositions.begin(),
                           changedPositions.end(), std::back_inserter(*dirty));
            if (dirty->size() <= objects.size() / 4u)
            {
                tree = loadTree(*cooked);
                if (tree == nullptr)
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
                for (const uint32_t position : *dirty)
                {
                    if (!tree->updateBounds(position, treeLeaf(candidateBounds[position], cooked->margin)))
                    {
                        tree->release();
                        return OVSPATIALQUERY_API_INTERNAL;
                    }
                }
                tree->partialRefit();
            }
        }
        if (tree == nullptr)
        {
            dirty->clear();
            const ovspatialquery_api_status_t status = cookTree(candidateBounds, cooked, tree);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
        }
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
        requireDerivedTreeMatches(objects, candidateBounds, *tree);
#endif
        if (mBvh != nullptr)
        {
            mBvh->release();
        }
        mBvh                = tree;
        mRayOriginTolerance = static_cast<double>(cooked->margin);
        mBounds.swap(candidateBounds);
        mIds.swap(candidateIds);
        mCooked = std::move(cooked);
        mDirty  = std::move(dirty);
        return OVSPATIALQUERY_API_SUCCESS;
    }

    ovspatialquery_api_status_t findPairs(const BackendPopulation& other, BackendPairSink& sink) const override
    {
        const PhysXPopulation& otherPopulation = static_cast<const PhysXPopulation&>(other);
        if (mBvh != nullptr && otherPopulation.mBvh != nullptr)
        {
            PairCallback callback(mIds.size(), otherPopulation.mIds.size(), sink);
            (void)PxFindOverlap(callback, *mBvh, *otherPopulation.mBvh);
            if (!callback.valid())
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
        }
        return sink.status();
    }

    ovspatialquery_api_status_t raycastCandidates(const FloatRaycastQuery& query,
                                                  BackendCandidateSink&    sink) const override
    {
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
        enterTestGate(mRuntime->rayGate);
#endif
        if (mBvh == nullptr)
        {
            return sink.status();
        }
        const physx::PxVec3      origin(query.origin[0], query.origin[1], query.origin[2]);
        const physx::PxVec3      direction(query.unitDirection[0], query.unitDirection[1], query.unitDirection[2]);
        CandidateRaycastCallback callback(mIds.size(), sink);
        const bool               completed = mBvh->raycast(origin, direction, query.maxDistance, callback);
        if (callback.status() != OVSPATIALQUERY_API_SUCCESS)
        {
            return callback.status();
        }
        return completed || callback.stopped() ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INTERNAL;
    }

    ovspatialquery_api_status_t sphereSweepCandidates(const FloatSphereSweepQuery& query,
                                                      BackendCandidateSink&        sink) const override
    {
        if (mBvh == nullptr)
        {
            return sink.status();
        }
        const physx::PxSphereGeometry sphere(query.radius);
        const physx::PxVec3           center(query.center[0], query.center[1], query.center[2]);
        const physx::PxTransform      pose(center);
        const physx::PxVec3           direction(query.unitDirection[0], query.unitDirection[1], query.unitDirection[2]);
        CandidateRaycastCallback      callback(mIds.size(), sink);
        const bool                    completed = mBvh->sweep(sphere, pose, direction, query.maxDistance, callback);
        if (callback.status() != OVSPATIALQUERY_API_SUCCESS)
        {
            return callback.status();
        }
        return completed || callback.stopped() ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INTERNAL;
    }

    // PhysX sweeps the box's AABB through the tree.
    ovspatialquery_api_status_t boxSweepCandidates(const FloatBoxSweepQuery& query,
                                                   BackendCandidateSink&     sink) const override
    {
        if (mBvh == nullptr)
        {
            return sink.status();
        }
        const physx::PxBoxGeometry box(query.halfExtents[0], query.halfExtents[1], query.halfExtents[2]);
        const physx::PxTransform   pose(physx::PxVec3(query.center[0], query.center[1], query.center[2]));
        const physx::PxVec3        direction(query.unitDirection[0], query.unitDirection[1], query.unitDirection[2]);
        CandidateRaycastCallback   callback(mIds.size(), sink);
        const bool                 completed = mBvh->sweep(box, pose, direction, query.maxDistance, callback);
        if (callback.status() != OVSPATIALQUERY_API_SUCCESS)
        {
            return callback.status();
        }
        return completed || callback.stopped() ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INTERNAL;
    }

    ovspatialquery_api_status_t aabbOverlapCandidates(const FloatAabb& query, BackendCandidateSink& sink) const override
    {
        if (mBvh == nullptr)
        {
            return sink.status();
        }
        CandidateOverlapCallback callback(mIds.size(), sink);
        const bool               completed = mBvh->overlap(toPxBounds(query), callback);
        if (callback.status() != OVSPATIALQUERY_API_SUCCESS)
        {
            return callback.status();
        }
        return completed || callback.stopped() ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INTERNAL;
    }

    double rayOriginTolerance() const noexcept override { return mRayOriginTolerance; }

  private:
    physx::PxBVH* loadTree(const CookedTree& cooked) const
    {
        physx::PxDefaultMemoryInputData input(const_cast<physx::PxU8*>(cooked.bytes.data()),
                                              static_cast<physx::PxU32>(cooked.bytes.size()));
        return physxPhysics(mRuntime->infrastructure)->createBVH(input);
    }

    // Cooks a tree over `proxies` and loads this version's copy of it.
    ovspatialquery_api_status_t cookTree(const std::vector<physx::PxBounds3>& proxies,
                                         std::shared_ptr<const CookedTree>& outCooked, physx::PxBVH*& outTree) const
    {
        std::shared_ptr<CookedTree> cooked = std::make_shared<CookedTree>();
        for (const physx::PxBounds3& proxy : proxies)
        {
            cooked->maximumMagnitude = std::max(cooked->maximumMagnitude, proxyMagnitude(proxy));
        }
        cooked->margin = rayOriginMargin(cooked->maximumMagnitude);
        std::vector<physx::PxBounds3> leaves;
        leaves.reserve(proxies.size());
        for (const physx::PxBounds3& proxy : proxies)
        {
            leaves.push_back(treeLeaf(proxy, cooked->margin));
        }
        physx::PxBVHDesc descriptor;
        descriptor.enlargement   = 0.0f;
        descriptor.bounds.count  = static_cast<physx::PxU32>(leaves.size());
        descriptor.bounds.stride = sizeof(physx::PxBounds3);
        descriptor.bounds.data   = leaves.data();
        physx::PxDefaultMemoryOutputStream output;
        if (!PxCookBVH(descriptor, output))
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        cooked->bytes.assign(output.getData(), output.getData() + output.getSize());
        physx::PxBVH* tree = loadTree(*cooked);
        if (tree == nullptr)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        outCooked = std::move(cooked);
        outTree   = tree;
        return OVSPATIALQUERY_API_SUCCESS;
    }

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    // Test builds check that a derived version holds every object's exact
    // proxy and a tree leaf containing the leaf a rebuild would make.
    static void requireDerivedTreeMatches(const ObjectTable& objects, const std::vector<physx::PxBounds3>& proxies,
                                          const physx::PxBVH& tree)
    {
        bool  matches   = proxies.size() == objects.size() && tree.getNbBounds() == objects.size();
        float magnitude = 0.0f;
        for (size_t index = 0u; matches && index < objects.size(); ++index)
        {
            const physx::PxBounds3 expected = toPxBounds(makePopulationProxy(objects[index].bounds));
            matches                         = std::memcmp(&expected, &proxies[index], sizeof(expected)) == 0;
            magnitude                       = std::max(magnitude, proxyMagnitude(expected));
        }
        for (size_t index = 0u; matches && index < objects.size(); ++index)
        {
            matches = treeLeaf(proxies[index], rayOriginMargin(magnitude)).isInside(tree.getBounds()[index]);
        }
        if (!matches)
        {
            std::fprintf(stderr, "a derived PhysX population differs from a rebuild\n");
            std::fflush(stderr);
            std::_Exit(EXIT_FAILURE);
        }
    }
#endif

    ovspatialquery_backend_runtime_t*       mRuntime;
    std::vector<physx::PxBounds3>           mBounds;
    std::vector<ovspatialquery_object_id_t> mIds;
    physx::PxBVH*                           mBvh                = nullptr;
    double                                  mRayOriginTolerance = 0.0;
    // The tree this version was cooked from or loaded from, and the tree
    // positions whose bounds differ from it; empty when built directly.
    std::shared_ptr<const CookedTree>            mCooked;
    std::shared_ptr<const std::vector<uint32_t>> mDirty;
};

} // namespace

ovspatialquery_api_status_t createBackendPopulation(ovspatialquery_backend_runtime_t*   backendRuntime,
                                                    std::unique_ptr<BackendPopulation>& population)
{
    if (backendRuntime == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    population.reset(new PhysXPopulation(backendRuntime));
    return OVSPATIALQUERY_API_SUCCESS;
}

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
// The PhysX backend has no deterministic test checkpoints.
void reachPopulationBeforePublish(ovspatialquery_backend_runtime_t*, ovspatialquery_ordinal_t) {}
void reachPopulationAfterPublish(ovspatialquery_backend_runtime_t*, ovspatialquery_ordinal_t) {}
void reachOperationBeforeCompletion(ovspatialquery_backend_runtime_t*, ovspatialquery_ordinal_t) {}
#endif

} // namespace detail
} // namespace ovspatialquery
