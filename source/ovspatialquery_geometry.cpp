/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery_geometry.h>

#include "ovspatialquery_box_sweep_geometry.h"
#include "ovspatialquery_box_sweep_math.h"
#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_physx_backend.h"
#include <atomic>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
namespace
{

double vertexComponent(const std::vector<float>& vertices, uint32_t vertex, uint32_t axis)
{
    return vertices[static_cast<size_t>(vertex) * 3u + axis];
}

bool triangleHasArea(const std::vector<float>& vertices, const uint32_t* triangle)
{
    // Float inputs fit safely in double differences and cross products, including
    // subnormal inputs. No additional coordinate normalization is necessary.
    double first[3]{}, second[3]{};
    for (uint32_t a = 0; a < 3; ++a)
    {
        first[a]  = vertexComponent(vertices, triangle[1], a) - vertexComponent(vertices, triangle[0], a);
        second[a] = vertexComponent(vertices, triangle[2], a) - vertexComponent(vertices, triangle[0], a);
    }
    return first[1] * second[2] != first[2] * second[1] || first[2] * second[0] != first[0] * second[2] ||
           first[0] * second[1] != first[1] * second[0];
}

// Where a mesh query gathers its candidate triangles: the local box always,
// or the local swept sphere when the sweep is known exactly. The capsule runs
// from `start` along `direction` (local units per unit of the sweep parameter).
struct TriangleGatherVolume
{
    double minimum[3]{};
    double maximum[3]{};
    bool   capsule = false;
    double start[3]{};
    double end[3]{};
    double direction[3]{};
    double radius = 0.0;
};

struct TriangleCandidates
{
    HostLease                   charge;
    std::unique_ptr<uint32_t[]> large;
    uint32_t                    small[128];
    uint32_t                    count = 0;

    const uint32_t* data() const { return large ? large.get() : small; }
};

// A gather expected to be large skips the inline attempt, whose traversal until it
// fills can cost as much as a complete one.
template <typename Gather>
ovspatialquery_api_status_t gatherTriangleCandidatesWith(const Gather& gather, uint32_t triangleCount,
                                                         InstanceHostLedger* ledger, TriangleCandidates& candidates,
                                                         bool large = false)
{
    bool overflow = false;
    if (!large)
    {
        candidates.count = gather(candidates.small, 128u, overflow);
        if (!overflow)
            return OVSPATIALQUERY_API_SUCCESS;
    }

    // The packaged BVH34 box path ignores startIndex. Like PxMeshOverlapUtil,
    // retry once with room for every triangle instead of attempting pagination.
    if (ledger)
    {
        HostTransaction transaction(*ledger);
        if (!transaction.reserve(uint64_t(triangleCount) * sizeof(uint32_t)) || !transaction.commit(candidates.charge))
            return OVSPATIALQUERY_API_HOST_BUDGET;
    }
    // Uninitialized: the gather writes every reported entry.
    candidates.large.reset(new uint32_t[triangleCount]);
    candidates.count = gather(candidates.large.get(), triangleCount, overflow);
    return overflow ? OVSPATIALQUERY_API_INTERNAL : OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t gatherTriangleCandidates(const PhysXTriangleMesh& mesh, const double minimum[3],
                                                     const double maximum[3], uint32_t triangleCount,
                                                     InstanceHostLedger* ledger, TriangleCandidates& candidates)
{
    return gatherTriangleCandidatesWith(
        [&](uint32_t* results, uint32_t capacity, bool& overflow)
        { return mesh.overlapTriangles(minimum, maximum, results, capacity, overflow); },
        triangleCount, ledger, candidates);
}

// A diagonal sweep's box holds many triangles the swept sphere never reaches;
// gathering by the capsule itself leaves them out.
ovspatialquery_api_status_t gatherTriangleCandidates(const PhysXTriangleMesh& mesh, const TriangleGatherVolume& volume,
                                                     uint32_t triangleCount, InstanceHostLedger* ledger,
                                                     TriangleCandidates& candidates, bool large = false)
{
    if (!volume.capsule)
        return gatherTriangleCandidates(mesh, volume.minimum, volume.maximum, triangleCount, ledger, candidates);
    return gatherTriangleCandidatesWith(
        [&](uint32_t* results, uint32_t capacity, bool& overflow)
        { return mesh.overlapTrianglesCapsule(volume.start, volume.end, volume.radius, results, capacity, overflow); },
        triangleCount, ledger, candidates, large);
}

// Bounds of a mesh's PhysX query structure (ADR-0075), measured with an allocation
// counter against the pinned cooker on grid, soup and random meshes of 1 to 524288
// triangles: what the built structure keeps, and the peak its build adds beyond that.
// Both stay at least 7 % above every measurement, a template cook and load included;
// the scratch's constant covers about 52 KiB the cooker allocates for any mesh above a
// few triangles.
uint64_t structureResidentBound(uint64_t vertexCount, uint64_t triangleCount) noexcept
{
    return saturatingResidentByteAdd(saturatingResidentByteAdd(4096u, vertexCount, 16u), triangleCount, 32u);
}

uint64_t structureScratchBound(uint64_t vertexCount, uint64_t triangleCount) noexcept
{
    return saturatingResidentByteAdd(saturatingResidentByteAdd(65536u, vertexCount, 16u), triangleCount, 192u);
}

// A source triangle without area: its place among the source's triangles, its vertex
// indices, source face and face-local corners.
struct DroppedTriangle
{
    uint64_t sourceIndex = 0u;
    uint64_t sourceFace  = 0u;
    uint32_t vertices[3]{};
    uint32_t corners[3]{};

    bool operator==(const DroppedTriangle& other) const noexcept
    {
        return sourceIndex == other.sourceIndex && sourceFace == other.sourceFace &&
               std::equal(vertices, vertices + 3, other.vertices) && std::equal(corners, corners + 3, other.corners);
    }
};

// The index arrays of a prepared mesh, shared by the meshes that differ from it only
// in vertex positions, with the cooked PhysX template a refit starts from. The template
// is a non-quantized BVH34 mesh over these triangles, cooked once when a derived mesh
// first needs its cooked mesh; every derived mesh loads a fresh mesh from the bytes and
// refits it to its own vertices instead of cooking, and the meshes the earlier
// publications hold are never touched.
struct TriangleMeshTopology
{
    std::vector<uint32_t> triangles;
    std::vector<uint64_t> sourceFaces;
    std::vector<uint32_t> sourceCorners;
    uint64_t              contentHash = 0u; // Of the three arrays; see triangleMeshContentHash.
    // Sum of the triangles' local box areas in the mesh that was read in full and
    // started this chain of topologies. A derived mesh whose triangles have stretched far
    // beyond it (a crumpled sheet, not a folded or waving one) would leave a refitted
    // tree with overlapping nodes and slow traversal, so it starts a topology of its own
    // and cooks a fresh tree; the rest-state area stays the reference along the chain,
    // so content that stays crumpled keeps cooking rather than refitting a poor tree.
    double baseBoxArea = 0.0;
    // The source's triangles without area, left out of the arrays above (ADR-0073) and in
    // source order. A derived mesh checks them under its own vertices, and the source's
    // arrays are rebuilt from both lists when one of them regains its area.
    std::vector<DroppedTriangle> dropped;
    mutable std::mutex           templateMutex;
    mutable std::vector<uint8_t> cookedTemplate;

    bool sameArrays(const TriangleMeshTopology& other) const noexcept
    {
        return triangles == other.triangles && sourceFaces == other.sourceFaces &&
               sourceCorners == other.sourceCorners && dropped == other.dropped;
    }

    // The index arrays only: each derived mesh charges the cooked template with its own
    // structure (admitTriangleMeshStructures).
    uint64_t residentByteCount() const noexcept
    {
        uint64_t total = sizeof(TriangleMeshTopology);
        total          = saturatingResidentByteAdd(total, triangles.capacity(), sizeof(uint32_t));
        total          = saturatingResidentByteAdd(total, sourceFaces.capacity(), sizeof(uint64_t));
        total          = saturatingResidentByteAdd(total, dropped.capacity(), sizeof(DroppedTriangle));
        return saturatingResidentByteAdd(total, sourceCorners.capacity(), sizeof(uint32_t));
    }
};

// FNV-1a over 64-bit words; the element counts separate the arrays.
struct ContentHasher
{
    uint64_t hash = UINT64_C(14695981039346656037);
    void     mix(uint64_t word) noexcept
    {
        hash ^= word;
        hash *= UINT64_C(1099511628211);
    }
    void mixBytes(const void* data, size_t bytes) noexcept
    {
        const uint8_t* cursor = static_cast<const uint8_t*>(data);
        size_t         offset = 0u;
        for (; offset + sizeof(uint64_t) <= bytes; offset += sizeof(uint64_t))
        {
            uint64_t word = 0u;
            std::memcpy(&word, cursor + offset, sizeof(word));
            mix(word);
        }
        if (offset < bytes)
        {
            uint64_t tail = 0u;
            std::memcpy(&tail, cursor + offset, bytes - offset);
            mix(tail);
        }
    }
    template <typename Values>
    void mixArray(const Values& values) noexcept
    {
        mix(static_cast<uint64_t>(values.size()));
        mixBytes(values.data(), values.size() * sizeof(values[0]));
    }
};

uint64_t hashTopologyArrays(const std::vector<uint32_t>& triangles, const std::vector<uint64_t>& sourceFaces,
                            const std::vector<uint32_t>& sourceCorners) noexcept
{
    ContentHasher hasher;
    hasher.mixArray(triangles);
    hasher.mixArray(sourceFaces);
    hasher.mixArray(sourceCorners);
    return hasher.hash;
}

// Triangles stretched beyond this multiple of the topology's base box area recook.
constexpr double kRefitStretchLimit = 8.0;

// Checks `count` triangles against `vertexCount` vertices: INVALID_ARGUMENT for an index
// outside them. Adds the local box areas of those with area to `boxArea` and appends the
// positions of those without (ADR-0073) to `withoutArea`. The one caller of
// triangleHasArea, so the check stays inlined in this per-triangle loop.
ovspatialquery_api_status_t scanTriangles(const std::vector<float>& vertices, const uint32_t* triangles, size_t count,
                                          double& boxArea, std::vector<uint64_t>& withoutArea)
{
    const size_t vertexCount = vertices.size() / 3u;
    for (size_t face = 0u; face < count; ++face)
    {
        const uint32_t* triangle = triangles + face * 3u;
        if (triangle[0] >= vertexCount || triangle[1] >= vertexCount || triangle[2] >= vertexCount)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        if (!triangleHasArea(vertices, triangle))
        {
            withoutArea.push_back(face);
            continue;
        }
        double extent[3] = {};
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            const double first  = vertexComponent(vertices, triangle[0], axis);
            const double second = vertexComponent(vertices, triangle[1], axis);
            const double third  = vertexComponent(vertices, triangle[2], axis);
            extent[axis]        = std::max(first, std::max(second, third)) - std::min(first, std::min(second, third));
        }
        boxArea += extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0];
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// Splits the source's triangles under `vertices` into the topology's arrays, those with
// area, and its dropped list: INVALID_ARGUMENT for an index outside the vertices or when
// no triangle has area. The arrays are moved in whole when nothing is dropped.
ovspatialquery_api_status_t keepTrianglesWithArea(const std::vector<float>& vertices, std::vector<uint32_t> triangles,
                                                  std::vector<uint64_t> faces, std::vector<uint32_t> corners,
                                                  TriangleMeshTopology& topology, double& outBoxArea)
{
    const size_t          count = triangles.size() / 3u;
    std::vector<uint64_t> withoutArea;
    outBoxArea = 0.0;
    const ovspatialquery_api_status_t status =
        scanTriangles(vertices, triangles.data(), count, outBoxArea, withoutArea);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    if (withoutArea.size() == count)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (withoutArea.empty())
    {
        topology.triangles     = std::move(triangles);
        topology.sourceFaces   = std::move(faces);
        topology.sourceCorners = std::move(corners);
        topology.dropped.clear();
        return OVSPATIALQUERY_API_SUCCESS;
    }
    // Copies the runs of kept triangles between dropped ones as blocks into arrays sized once.
    const size_t kept = count - withoutArea.size();
    topology.dropped.resize(withoutArea.size());
    topology.triangles.resize(3u * kept);
    topology.sourceFaces.resize(kept);
    topology.sourceCorners.resize(3u * kept);
    size_t runStart = 0u;
    size_t written  = 0u;
    for (size_t next = 0u; next <= withoutArea.size(); ++next)
    {
        const size_t runEnd = next < withoutArea.size() ? static_cast<size_t>(withoutArea[next]) : count;
        const size_t run    = runEnd - runStart;
        if (run != 0u)
        {
            std::memcpy(topology.triangles.data() + 3u * written, triangles.data() + 3u * runStart,
                        3u * run * sizeof(uint32_t));
            std::memcpy(topology.sourceFaces.data() + written, faces.data() + runStart, run * sizeof(uint64_t));
            std::memcpy(topology.sourceCorners.data() + 3u * written, corners.data() + 3u * runStart,
                        3u * run * sizeof(uint32_t));
            written += run;
        }
        if (next == withoutArea.size())
        {
            break;
        }
        DroppedTriangle& entry = topology.dropped[next];
        entry.sourceIndex      = runEnd;
        entry.sourceFace       = faces[runEnd];
        std::copy(triangles.data() + 3u * runEnd, triangles.data() + 3u * runEnd + 3u, entry.vertices);
        std::copy(corners.data() + 3u * runEnd, corners.data() + 3u * runEnd + 3u, entry.corners);
        runStart = runEnd + 1u;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// The source's arrays of a topology, its dropped triangles back in their places.
void sourceArrays(const TriangleMeshTopology& topology, std::vector<uint32_t>& outTriangles,
                  std::vector<uint64_t>& outFaces, std::vector<uint32_t>& outCorners)
{
    const size_t count = topology.sourceFaces.size() + topology.dropped.size();
    outTriangles.clear();
    outTriangles.reserve(3u * count);
    outFaces.clear();
    outFaces.reserve(count);
    outCorners.clear();
    outCorners.reserve(3u * count);
    size_t kept = 0u;
    size_t next = 0u;
    for (size_t triangle = 0u; triangle < count; ++triangle)
    {
        if (next < topology.dropped.size() && topology.dropped[next].sourceIndex == triangle)
        {
            const DroppedTriangle& entry = topology.dropped[next++];
            outTriangles.insert(outTriangles.end(), entry.vertices, entry.vertices + 3);
            outFaces.push_back(entry.sourceFace);
            outCorners.insert(outCorners.end(), entry.corners, entry.corners + 3);
            continue;
        }
        outTriangles.insert(outTriangles.end(), topology.triangles.begin() + kept * 3u,
                            topology.triangles.begin() + kept * 3u + 3u);
        outFaces.push_back(topology.sourceFaces[kept]);
        outCorners.insert(outCorners.end(), topology.sourceCorners.begin() + kept * 3u,
                          topology.sourceCorners.begin() + kept * 3u + 3u);
        ++kept;
    }
}

// Whether a topology holds exactly the given source arrays, its dropped triangles included.
bool sameSourceArrays(const TriangleMeshTopology& topology, const std::vector<uint32_t>& triangles,
                      const std::vector<uint64_t>& faces, const std::vector<uint32_t>& corners)
{
    if (topology.dropped.empty())
    {
        return topology.triangles == triangles && topology.sourceFaces == faces && topology.sourceCorners == corners;
    }
    std::vector<uint32_t> storedTriangles;
    std::vector<uint64_t> storedFaces;
    std::vector<uint32_t> storedCorners;
    sourceArrays(topology, storedTriangles, storedFaces, storedCorners);
    return storedTriangles == triangles && storedFaces == faces && storedCorners == corners;
}

} // namespace

class PreparedTriangleMesh
{
  public:
    static ovspatialquery_api_status_t create(const TriangleMeshSource&                    source,
                                              std::shared_ptr<const PreparedTriangleMesh>& outMesh);
    /* A mesh over `base`'s topology with the source's vertices; `source.vertices` must
     * hold as many vertices as the base. */
    static ovspatialquery_api_status_t createDerived(const std::shared_ptr<const PreparedTriangleMesh>& base,
                                                     const TriangleMeshSource&                          source,
                                                     std::shared_ptr<const PreparedTriangleMesh>&       outMesh);

    uint64_t residentByteCount() const noexcept;
    bool     sweepSphere(const std::array<double, 16u>& worldFromLocal, const double center[3], double radius,
                         const float unitDirection[3], double minimumDistance, double maximumDistance, bool doubleSided,
                         bool flipBackFaces, PreciseTriangleHit& outHit, const PhysXTriangleMesh& cooked,
                         const TriangleGatherVolume& volume, bool raycast, InstanceHostLedger* ledger,
                         ovspatialquery_api_status_t* outStatus, const uint32_t* onlyTriangle = nullptr) const;
    bool     overlap(const std::array<double, 16u>& worldFromLocal, const ovspatialquery_overlap_shape_t& geometry,
                     const double (*boxRotation)[3], const PhysXTriangleMesh& cooked, const double localMinimum[3],
                     const double localMaximum[3], InstanceHostLedger* ledger,
                     ovspatialquery_api_status_t* outStatus) const;

    void exportDescription(TriangleGeometryDescriptionView& outDescription) const noexcept
    {
        TriangleGeometryDescriptionView description{};
        description.localVertices         = mVertices.data();
        description.vertexCount           = static_cast<uint64_t>(mVertices.size() / 3u);
        description.triangleVertexIndices = mTopology->triangles.data();
        description.triangleCount         = static_cast<uint64_t>(mTopology->triangles.size() / 3u);
        description.sourceFaces           = mTopology->sourceFaces.data();
        description.sourceCorners         = mTopology->sourceCorners.data();
        description.refitTree             = mDerived;
        outDescription                    = description;
    }

    // Builds the query structure on the first exact reader. The admission that created
    // the mesh reserved it and claimed the scratch of building it (ADR-0075), so the
    // build charges nothing; claimed builds take the instance's build lock one at a time.
    ovspatialquery_api_status_t physxMesh(const PhysXTriangleMesh*& result) const
    {
        result = mPublishedPhysX.load(std::memory_order_acquire);
        if (result != nullptr)
            return OVSPATIALQUERY_API_SUCCESS;
        std::lock_guard<std::mutex> lock(mPhysXMutex);
        if (!mPhysXMesh)
        {
            InstanceHostLedger* claimed = nullptr;
            {
                std::lock_guard<std::mutex> claimLock(mClaimMutex);
                claimed = mBuildClaim.ledger();
            }
            std::unique_lock<std::mutex> build;
            if (claimed != nullptr)
                build = claimed->lockBuild();
            TriangleGeometryDescriptionView description{};
            exportDescription(description);
            const auto status =
                mDerived ? refitPhysXMesh(description) : PhysXTriangleMesh::create(description, mPhysXMesh);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            mPublishedPhysX.store(mPhysXMesh.get(), std::memory_order_release);
            if (build.owns_lock())
                build.unlock();
            BuildScratchClaim           ended;
            std::lock_guard<std::mutex> claimLock(mClaimMutex);
            ended = std::move(mBuildClaim);
        }
        result = mPhysXMesh.get();
        return OVSPATIALQUERY_API_SUCCESS;
    }

    // What admitting the mesh charges while no admission has: its structure, with the
    // template a derived mesh loads from as each derived mesh may cook it, and the build
    // scratch while unbuilt. Zero once admitted.
    void structureAdmission(uint64_t& outResident, uint64_t& outScratch) const noexcept
    {
        outResident = outScratch = 0u;
        std::lock_guard<std::mutex> claimLock(mClaimMutex);
        if (mStructureLease)
            return;
        const uint64_t vertices = mVertices.size() / 3u, triangles = mTopology->triangles.size() / 3u;
        outResident = saturatingResidentByteAdd(0u, mDerived ? 2u : 1u, structureResidentBound(vertices, triangles));
        if (mPublishedPhysX.load(std::memory_order_acquire) == nullptr)
            outScratch = structureScratchBound(vertices, triangles);
    }

    // Keeps the charges unless another admission's arrived first; the claim ends at the build.
    void adoptStructure(HostLease&& structure, BuildScratchClaim&& claim) const noexcept
    {
        HostLease                   unusedStructure;
        BuildScratchClaim           unusedClaim;
        std::lock_guard<std::mutex> claimLock(mClaimMutex);
        if (mStructureLease)
        {
            unusedStructure = std::move(structure);
            unusedClaim     = std::move(claim);
            return;
        }
        mStructureLease = std::move(structure);
        if (mPublishedPhysX.load(std::memory_order_acquire) == nullptr)
            mBuildClaim = std::move(claim);
        else
            unusedClaim = std::move(claim);
    }

    const ovspatialquery_aabb_t& localBounds() const { return mLocalBounds; }

    bool sameContent(const TriangleMeshSource& source) const noexcept
    {
        const auto same = [](const auto& stored, const auto& candidate)
        {
            return !candidate.empty() && stored.size() == candidate.size() &&
                   (stored.empty() ||
                    std::memcmp(stored.data(), candidate.data(), stored.size() * sizeof(stored[0])) == 0);
        };
        // The triangles kept follow from the vertices and the source's triangles, so the
        // same vertices over the same source arrays make the same mesh.
        if (source.topologyBase != nullptr)
        {
            const TriangleMeshTopology& other = *source.topologyBase->mTopology;
            if (!same(mVertices, source.vertices))
                return false;
            if (&other == mTopology.get())
                return true;
            if (other.contentHash != mTopology->contentHash)
                return false;
            if (mTopology->sameArrays(other))
                return true;
            if (other.dropped.empty() && mTopology->dropped.empty())
                return false;
            try
            {
                std::vector<uint32_t> triangles;
                std::vector<uint64_t> faces;
                std::vector<uint32_t> corners;
                sourceArrays(other, triangles, faces, corners);
                return sameSourceArrays(*mTopology, triangles, faces, corners);
            }
            catch (...)
            {
                return false; // Not shared; the caller prepares its own mesh.
            }
        }
        if (!same(mVertices, source.vertices) || source.sourceFaces.empty() || source.sourceCorners.empty())
            return false;
        if (mTopology->dropped.empty())
            return same(mTopology->triangles, source.triangles) && same(mTopology->sourceFaces, source.sourceFaces) &&
                   same(mTopology->sourceCorners, source.sourceCorners);
        try
        {
            return sameSourceArrays(*mTopology, source.triangles, source.sourceFaces, source.sourceCorners);
        }
        catch (...)
        {
            return false;
        }
    }

    // The source triangles left out for having no area, in source order (ADR-0073).
    const std::vector<DroppedTriangle>& droppedTriangles() const noexcept { return mTopology->dropped; }
    uint64_t sourceTriangleCount() const noexcept { return mTopology->sourceFaces.size() + mTopology->dropped.size(); }

    uint64_t topologyContentHash() const noexcept { return mTopology->contentHash; }

    uint64_t sourceFace(uint32_t triangle) const { return mTopology->sourceFaces[triangle]; }

    const uint32_t* sourceCorners(uint32_t triangle) const
    {
        return mTopology->sourceCorners.data() + static_cast<size_t>(triangle) * 3u;
    }

  private:
    // Loads a mesh from the topology's cooked template (cooked here on first use) and
    // refits it to this mesh's vertices; the base's own cooked mesh stays untouched.
    ovspatialquery_api_status_t refitPhysXMesh(const TriangleGeometryDescriptionView& description) const
    {
        std::lock_guard<std::mutex> templateLock(mTopology->templateMutex);
        if (mTopology->cookedTemplate.empty())
        {
            std::vector<uint8_t> cooked;
            const auto           status = PhysXTriangleMesh::cookTemplate(description, cooked);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            mTopology->cookedTemplate = std::move(cooked);
        }
        return PhysXTriangleMesh::createFromTemplate(mTopology->cookedTemplate, description, mPhysXMesh);
    }
    // Checks the vertices against the input domain and records their local bounds.
    static ovspatialquery_api_status_t finishVertices(PreparedTriangleMesh& mesh);

    mutable std::mutex                            mPhysXMutex;
    mutable std::mutex                            mClaimMutex;
    mutable HostLease                             mStructureLease; // Guarded by mClaimMutex.
    mutable BuildScratchClaim                     mBuildClaim;     // Until built; guarded by mClaimMutex.
    mutable std::unique_ptr<PhysXTriangleMesh>    mPhysXMesh;
    mutable std::atomic<const PhysXTriangleMesh*> mPublishedPhysX{nullptr};
    std::vector<float>                            mVertices;
    std::shared_ptr<const TriangleMeshTopology>   mTopology;
    bool                                          mDerived = false; // Cooked by refitting the topology's template.
    ovspatialquery_aabb_t                         mLocalBounds{};
};

uint64_t PreparedTriangleMesh::residentByteCount() const noexcept
{
    const uint64_t total = saturatingResidentByteAdd(sizeof(PreparedTriangleMesh), mVertices.capacity(), sizeof(float));
    // The topology is shared with the meshes derived from this one; each counts it in
    // full here, which is conservative, and once where allocations are accumulated by owner.
    // The query structure is not here: the mesh holds its own charge for it, taken once by
    // the admission that added it (admitTriangleMeshStructures).
    return saturatingResidentByteAdd(total, 1u, mTopology ? mTopology->residentByteCount() : 0u);
}

ovspatialquery_api_status_t PreparedTriangleMesh::create(const TriangleMeshSource&                    source,
                                                         std::shared_ptr<const PreparedTriangleMesh>& outMesh)
{
    if (source.key == 0u || source.vertices.empty() || source.vertices.size() % 3u != 0u || source.triangles.empty() ||
        source.triangles.size() % 3u != 0u ||
        source.triangles.size() / 3u > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::shared_ptr<PreparedTriangleMesh> mesh = std::make_shared<PreparedTriangleMesh>();
        mesh->mVertices                            = source.vertices;
        const size_t sourceFaceCount               = source.triangles.size() / 3u;
        if ((!source.sourceFaces.empty() && source.sourceFaces.size() != sourceFaceCount) ||
            (!source.sourceCorners.empty() && source.sourceCorners.size() != source.triangles.size()))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        std::vector<uint64_t> faces;
        if (source.sourceFaces.empty())
        {
            faces.resize(sourceFaceCount);
            std::iota(faces.begin(), faces.end(), 0u);
        }
        else
        {
            faces = source.sourceFaces;
        }
        std::vector<uint32_t> corners;
        if (source.sourceCorners.empty())
        {
            corners.resize(source.triangles.size());
            for (size_t corner = 0u; corner < corners.size(); ++corner)
            {
                corners[corner] = static_cast<uint32_t>(corner % 3u);
            }
        }
        else
        {
            corners = source.sourceCorners;
        }
        ovspatialquery_api_status_t status = finishVertices(*mesh);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        // Of the source's arrays, so a source with dropped triangles still finds this mesh.
        const uint64_t                        contentHash = hashTopologyArrays(source.triangles, faces, corners);
        std::shared_ptr<TriangleMeshTopology> topology    = std::make_shared<TriangleMeshTopology>();
        double                                boxArea     = 0.0;
        status = keepTrianglesWithArea(mesh->mVertices, source.triangles, std::move(faces), std::move(corners),
                                       *topology, boxArea);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        topology->contentHash = contentHash;
        topology->baseBoxArea = boxArea;
        mesh->mTopology       = std::move(topology);
        outMesh               = std::move(mesh);
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

ovspatialquery_api_status_t PreparedTriangleMesh::finishVertices(PreparedTriangleMesh& mesh)
{
    const std::vector<float>& vertices    = mesh.mVertices;
    const size_t              vertexCount = vertices.size() / 3u;
    for (size_t component = 0u; component < vertices.size(); ++component)
    {
        if (!inInputDomain(static_cast<double>(vertices[component])))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        mesh.mLocalBounds.minimum[axis] = vertices[axis];
        mesh.mLocalBounds.maximum[axis] = vertices[axis];
    }
    for (size_t vertex = 1u; vertex < vertexCount; ++vertex)
    {
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            const double value              = vertexComponent(vertices, static_cast<uint32_t>(vertex), axis);
            mesh.mLocalBounds.minimum[axis] = std::min(mesh.mLocalBounds.minimum[axis], value);
            mesh.mLocalBounds.maximum[axis] = std::max(mesh.mLocalBounds.maximum[axis], value);
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PreparedTriangleMesh::createDerived(const std::shared_ptr<const PreparedTriangleMesh>& base,
                                                                const TriangleMeshSource&                    source,
                                                                std::shared_ptr<const PreparedTriangleMesh>& outMesh)
{
    if (base == nullptr || base->mTopology == nullptr || source.key == 0u || source.vertices.empty() ||
        source.vertices.size() != base->mVertices.size() || !source.triangles.empty() || !source.sourceFaces.empty() ||
        !source.sourceCorners.empty())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::shared_ptr<PreparedTriangleMesh> mesh = std::make_shared<PreparedTriangleMesh>();
        mesh->mVertices                            = source.vertices;
        mesh->mTopology                            = base->mTopology;
        mesh->mDerived                             = true;
        ovspatialquery_api_status_t status         = finishVertices(*mesh);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        const TriangleMeshTopology& baseTopology = *base->mTopology;
        double                      boxArea      = 0.0;
        std::vector<uint64_t>       withoutArea;
        status = scanTriangles(mesh->mVertices, baseTopology.triangles.data(), baseTopology.sourceFaces.size(), boxArea,
                               withoutArea);
        bool sameTriangles = status == OVSPATIALQUERY_API_SUCCESS && withoutArea.empty();
        if (sameTriangles && !baseTopology.dropped.empty())
        {
            // The dropped triangles must still have no area for the base's tree to fit.
            std::vector<uint32_t> dropped;
            dropped.reserve(3u * baseTopology.dropped.size());
            for (const DroppedTriangle& entry : baseTopology.dropped)
                dropped.insert(dropped.end(), entry.vertices, entry.vertices + 3);
            double unused = 0.0;
            status = scanTriangles(mesh->mVertices, dropped.data(), baseTopology.dropped.size(), unused, withoutArea);
            sameTriangles = status == OVSPATIALQUERY_API_SUCCESS && withoutArea.size() == baseTopology.dropped.size();
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        if (!sameTriangles || boxArea > kRefitStretchLimit * baseTopology.baseBoxArea)
        {
            // Stretched too far for the base's tree, or a triangle lost or regained its
            // area (ADR-0073): the source's triangles under a topology of its own, cooked
            // afresh, from which later republications refit.
            std::shared_ptr<TriangleMeshTopology> topology = std::make_shared<TriangleMeshTopology>();
            if (sameTriangles)
            {
                topology->triangles     = baseTopology.triangles;
                topology->sourceFaces   = baseTopology.sourceFaces;
                topology->sourceCorners = baseTopology.sourceCorners;
                topology->dropped       = baseTopology.dropped;
            }
            else
            {
                std::vector<uint32_t> triangles;
                std::vector<uint64_t> faces;
                std::vector<uint32_t> corners;
                sourceArrays(baseTopology, triangles, faces, corners);
                status = keepTrianglesWithArea(mesh->mVertices, std::move(triangles), std::move(faces),
                                               std::move(corners), *topology, boxArea);
                if (status != OVSPATIALQUERY_API_SUCCESS)
                {
                    return status;
                }
            }
            topology->contentHash = baseTopology.contentHash;
            topology->baseBoxArea = baseTopology.baseBoxArea;
            mesh->mTopology       = std::move(topology);
            mesh->mDerived        = false;
        }
        outMesh = std::move(mesh);
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

namespace
{

void transformMeshVertex(const std::vector<float>& vertices, uint32_t vertex,
                         const std::array<double, 16u>& worldFromLocal, double world[3])
{
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        const size_t row = static_cast<size_t>(axis) * 4u;
        world[axis]      = std::fma(
            worldFromLocal[row + 2u], vertexComponent(vertices, vertex, 2u),
            std::fma(worldFromLocal[row + 1u], vertexComponent(vertices, vertex, 1u),
                          std::fma(worldFromLocal[row], vertexComponent(vertices, vertex, 0u), worldFromLocal[row + 3u])));
    }
}

bool worldTriangleNormal(const double triangle[3][3], double normal[3])
{
    double first[3]  = {};
    double second[3] = {};
    double edgeScale = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        first[axis]  = triangle[1][axis] - triangle[0][axis];
        second[axis] = triangle[2][axis] - triangle[0][axis];
        edgeScale    = std::max(edgeScale, std::abs(first[axis]));
        edgeScale    = std::max(edgeScale, std::abs(second[axis]));
    }
    if (!std::isfinite(edgeScale) || edgeScale == 0.0)
    {
        return false;
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        first[axis] /= edgeScale;
        second[axis] /= edgeScale;
    }
    normal[0]           = first[1] * second[2] - first[2] * second[1];
    normal[1]           = first[2] * second[0] - first[0] * second[2];
    normal[2]           = first[0] * second[1] - first[1] * second[0];
    const double length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (!std::isfinite(length) || length == 0.0)
    {
        return false;
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        normal[axis] /= length;
    }
    return true;
}

// closestPointOnTriangle(triangle, point, closest, outU, outV) now lives in
// ovspatialquery_float_bounds.h, shared with ovspatialquery_convex_geometry.cpp.

// Distances are path parameters: the center moves to origin + t d with the
// caller's float direction as given, which is unit only to within rounding
// (squared length 1 +- 1e-5 is accepted). The vertex and edge entries below
// therefore keep |d|^2, as the face entry and the contact check already do;
// treating it as one leaves the sphere (|d|^2 - 1) t^2 off touching at the
// root. They also solve around the path's closest approach, as capsule paths
// do: the textbook discriminant subtracts squared distances of the whole path,
// and on sweeps longer than a few metres its rounding alone can leave a root
// outside the contact check's tolerance, which then rejects the first contact.
// The rounded offset from the start already limits the gap to about eps times
// the path length, so only the gap itself keeps a fused multiply-add (a libm
// call in this build), which leaves the closest parameter's error along the path.
double inverseSquaredLength(const float direction[3])
{
    const double x = direction[0], y = direction[1], z = direction[2];
    return 1.0 / (x * x + y * y + z * z);
}

bool raySphereEntry(const double origin[3], const float direction[3], double inverseDirectionSquare,
                    const double point[3], double radius, double& outDistance)
{
    double offset[3]  = {};
    double projection = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        offset[axis] = origin[axis] - point[axis];
        projection += offset[axis] * static_cast<double>(direction[axis]);
    }
    const double closest    = -projection * inverseDirectionSquare;
    double       gapSquared = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        const double gap = std::fma(static_cast<double>(direction[axis]), closest, offset[axis]);
        gapSquared += gap * gap;
    }
    const double reach = radius * radius - gapSquared;
    if (!std::isfinite(reach) || reach < 0.0)
    {
        return false;
    }
    outDistance = closest - std::sqrt(reach * inverseDirectionSquare);
    return std::isfinite(outDistance);
}

bool rayCapsuleEdgeEntry(const double origin[3], const float direction[3], const double first[3],
                         const double second[3], double radius, double& outDistance)
{
    double edge[3]       = {};
    double offset[3]     = {};
    double edgeSquared   = 0.0;
    double edgeOffset    = 0.0;
    double edgeDirection = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        edge[axis]   = second[axis] - first[axis];
        offset[axis] = origin[axis] - first[axis];
        edgeSquared += edge[axis] * edge[axis];
        edgeOffset += edge[axis] * offset[axis];
        edgeDirection += edge[axis] * static_cast<double>(direction[axis]);
    }
    // Offset and motion across the edge's line, both scaled by |e|^2, then their
    // closest approach; the scale cancels from the parameter and the gap is |e|^2 too.
    double across[3]     = {};
    double motion[3]     = {};
    double motionSquared = 0.0;
    double approach      = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        across[axis] = edgeSquared * offset[axis] - edgeOffset * edge[axis];
        motion[axis] = edgeSquared * static_cast<double>(direction[axis]) - edgeDirection * edge[axis];
        motionSquared += motion[axis] * motion[axis];
        approach += across[axis] * motion[axis];
    }
    if (!std::isfinite(motionSquared) || motionSquared <= 0.0)
    {
        return false;
    }
    const double inverseMotion = 1.0 / motionSquared;
    const double closest       = -approach * inverseMotion;
    double       gapSquared    = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        const double gap = std::fma(motion[axis], closest, across[axis]);
        gapSquared += gap * gap;
    }
    const double scaledRadius = radius * edgeSquared;
    const double reach        = scaledRadius * scaledRadius - gapSquared;
    if (!std::isfinite(reach) || reach < 0.0)
    {
        return false;
    }
    const double distance      = closest - std::sqrt(reach * inverseMotion);
    const double edgeParameter = edgeOffset + distance * edgeDirection;
    if (!std::isfinite(distance) || edgeParameter < 0.0 || edgeParameter > edgeSquared)
    {
        return false;
    }
    outDistance = distance;
    return true;
}

// The squared-distance slack a contact solved exactly at the radius can need: its center
// is rebuilt from the query center, the triangle anchor and the path length, and the
// closest point from the triangle's extent, each rounding by ulps of its own magnitude, so
// a contact far from the sweep's start lands outside a slack of the radius alone. Kept out
// of line: inlined into every candidate contact it made CPU sweeps up to 6 % slower.
#if defined(_MSC_VER)
#define OVS_SPHERE_SLACK_COLD __declspec(noinline)
#else
#define OVS_SPHERE_SLACK_COLD __attribute__((noinline, cold))
#endif
OVS_SPHERE_SLACK_COLD double sphereContactRoundingSlack(const double center[3], const double anchor[3],
                                                        const double relative[3][3], double radius, double distance)
{
    double magnitude = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        magnitude = std::max(magnitude, std::abs(center[axis]) + std::abs(anchor[axis]));
        for (uint32_t corner = 1u; corner < 3u; ++corner)
            magnitude = std::max(magnitude, std::abs(relative[corner][axis]));
    }
    const double spread = (magnitude + std::abs(distance)) * std::numeric_limits<double>::epsilon() * 256.0;
    return spread * (2.0 * radius + spread);
}
#undef OVS_SPHERE_SLACK_COLD

void considerSphereSweepContact(const double triangle[3][3], const double faceNormal[3], const double center[3],
                                double radius, const float unitDirection[3], double minimumDistance,
                                double maximumDistance, double distance, uint32_t faceIndex, bool& hasHit,
                                double& closestDistance, PreciseTriangleHit& outHit, bool flipBackFaces)
{
    if (!std::isfinite(distance) || distance < minimumDistance || distance > maximumDistance ||
        (hasHit && distance > closestDistance))
    {
        return;
    }
    // Evaluate contact in a triangle-relative frame: adding displacement to
    // a large world center first can round a valid contact outside the radius
    // tolerance. Anchoring at a vertex also preserves short triangle edges when
    // the query begins far away. Only the final output position returns to world.
    double relativeTriangle[3][3] = {};
    for (uint32_t corner = 0u; corner < 3u; ++corner)
        for (uint32_t axis = 0u; axis < 3u; ++axis)
            relativeTriangle[corner][axis] = triangle[corner][axis] - triangle[0][axis];
    double sphereCenter[3] = {};
    double contact[3]      = {};
    double u               = 0.0;
    double v               = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        sphereCenter[axis] =
            std::fma(static_cast<double>(unitDirection[axis]), distance, center[axis] - triangle[0][axis]);
    }
    closestPointOnTriangle(relativeTriangle, sphereCenter, contact, u, v);
    double delta[3]        = {};
    double distanceSquared = 0.0;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        delta[axis]     = sphereCenter[axis] - contact[axis];
        distanceSquared = std::fma(delta[axis], delta[axis], distanceSquared);
    }
    const double radiusSquared = radius * radius;
    const double tolerance     = std::max(1.0, radiusSquared) * std::numeric_limits<double>::epsilon() * 256.0;
    // Only a solved contact time takes the coordinate slack: the sweep's start is checked
    // for overlap, and widening that would report a sphere that starts just apart and
    // then grazes the face as touching at once.
    if (!std::isfinite(distanceSquared) ||
        (distanceSquared > radiusSquared + tolerance &&
         (distance == minimumDistance ||
          distanceSquared >
              radiusSquared + sphereContactRoundingSlack(center, triangle[0], relativeTriangle, radius, distance))))
    {
        return;
    }
    if (hasHit && distance == closestDistance && faceIndex >= outHit.faceIndex)
    {
        return;
    }
    double       contactNormal[3] = {};
    const double contactDistance  = std::sqrt(std::max(0.0, distanceSquared));
    if (contactDistance > std::sqrt(tolerance))
    {
        double directionDotNormal = 0.0;
        double normalSquared      = 0.0;
        double alongNormal        = 0.0;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            contactNormal[axis] = delta[axis] / contactDistance;
            directionDotNormal =
                std::fma(static_cast<double>(unitDirection[axis]), faceNormal[axis], directionDotNormal);
            normalSquared = std::fma(faceNormal[axis], faceNormal[axis], normalSquared);
            alongNormal   = std::fma(contactNormal[axis], faceNormal[axis], alongNormal);
        }
        // A ray with a contact tolerance that reaches a single-sided triangle only because
        // the query counts it from both sides reports its contact normal mirrored to the
        // front side, as a ray reports the front normal there (ADR-0065).
        if (directionDotNormal > 0.0 && !flipBackFaces && normalSquared > 0.0)
        {
            const double scale = 2.0 * alongNormal / normalSquared;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                contactNormal[axis] = std::fma(-scale, faceNormal[axis], contactNormal[axis]);
            }
        }
    }
    else
    {
        double directionDotNormal = 0.0;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            contactNormal[axis] = faceNormal[axis];
            directionDotNormal =
                std::fma(static_cast<double>(unitDirection[axis]), faceNormal[axis], directionDotNormal);
        }
        // Only a triangle counted from both sides survives with the motion along its front
        // normal. A sweep, and a ray against a mesh stored double-sided, report the normal
        // facing the motion; a ray against a single-sided mesh that the query counts from
        // both sides keeps the front normal, as PhysX's eMESH_BOTH_SIDES does (ADR-0065).
        if (directionDotNormal > 0.0 && flipBackFaces)
        {
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                contactNormal[axis] = -contactNormal[axis];
            }
        }
    }
    hasHit          = true;
    closestDistance = distance;
    PreciseTriangleHit hit{};
    hit.distance  = distance;
    hit.faceIndex = faceIndex;
    hit.uv[0]     = static_cast<float>(u);
    hit.uv[1]     = static_cast<float>(v);
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        hit.position[axis] = contact[axis] + triangle[0][axis];
        hit.normal[axis]   = static_cast<float>(contactNormal[axis]);
    }
    outHit = hit;
}

bool separatesTriangleAndBox(const double axis[3], const double triangle[3][3], const double halfExtents[3])
{
    double axisScale = 0.0;
    for (uint32_t component = 0u; component < 3u; ++component)
    {
        axisScale = std::max(axisScale, std::abs(axis[component]));
    }
    if (!std::isfinite(axisScale) || axisScale == 0.0)
    {
        return false;
    }
    const double normalizedAxis[3] = {
        axis[0] / axisScale,
        axis[1] / axisScale,
        axis[2] / axisScale,
    };
    double minimum = dot3(triangle[0], normalizedAxis);
    double maximum = minimum;
    for (uint32_t corner = 1u; corner < 3u; ++corner)
    {
        const double projection = dot3(triangle[corner], normalizedAxis);
        minimum                 = std::min(minimum, projection);
        maximum                 = std::max(maximum, projection);
    }
    const double radius = halfExtents[0] * std::abs(normalizedAxis[0]) + halfExtents[1] * std::abs(normalizedAxis[1]) +
                          halfExtents[2] * std::abs(normalizedAxis[2]);
    return minimum > radius || maximum < -radius;
}

bool triangleOverlapsCenteredBox(const double triangle[3][3], const double halfExtents[3])
{
    const double boxAxes[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (separatesTriangleAndBox(boxAxes[axis], triangle, halfExtents))
        {
            return false;
        }
    }
    double edges[3][3] = {};
    for (uint32_t edge = 0u; edge < 3u; ++edge)
    {
        const uint32_t next = (edge + 1u) % 3u;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            edges[edge][axis] = triangle[next][axis] - triangle[edge][axis];
        }
        for (uint32_t boxAxis = 0u; boxAxis < 3u; ++boxAxis)
        {
            const double crossAxis[3] = {
                edges[edge][1] * boxAxes[boxAxis][2] - edges[edge][2] * boxAxes[boxAxis][1],
                edges[edge][2] * boxAxes[boxAxis][0] - edges[edge][0] * boxAxes[boxAxis][2],
                edges[edge][0] * boxAxes[boxAxis][1] - edges[edge][1] * boxAxes[boxAxis][0],
            };
            if (separatesTriangleAndBox(crossAxis, triangle, halfExtents))
            {
                return false;
            }
        }
    }
    const double triangleNormal[3] = {
        edges[0][1] * edges[1][2] - edges[0][2] * edges[1][1],
        edges[0][2] * edges[1][0] - edges[0][0] * edges[1][2],
        edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0],
    };
    return !separatesTriangleAndBox(triangleNormal, triangle, halfExtents);
}

// `doubleRotation`, when given, replaces the float32 quaternion (ADR-0026).
void quaternionRotation(const float rotationXyzw[4], const double (*doubleRotation)[3], double rotation[3][3])
{
    if (doubleRotation != nullptr)
    {
        std::copy_n(&doubleRotation[0][0], 9u, &rotation[0][0]);
        return;
    }
    const double quaternion[4] = {
        static_cast<double>(rotationXyzw[0]),
        static_cast<double>(rotationXyzw[1]),
        static_cast<double>(rotationXyzw[2]),
        static_cast<double>(rotationXyzw[3]),
    };
    const double inverseLength = 1.0 / std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                                                 quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
    quaternionRotationMatrix(quaternion, inverseLength, rotation);
}

} // namespace

namespace
{

// The largest center distance at which considerSphereSweepContact accepts a contact:
// its squared-distance tolerance widens a tiny sphere's reach well beyond its radius.
double sphereContactAcceptanceRadius(double radius)
{
    const double radiusSquared = radius * radius;
    return std::sqrt(radiusSquared + std::max(1.0, radiusSquared) * std::numeric_limits<double>::epsilon() * 256.0);
}

// Two unit axes across a nonzero direction: the first crosses the coordinate axis the
// direction leans on least, the second completes the frame.
void crossAxes(const double direction[3], double across[2][3])
{
    uint32_t least = 0;
    for (uint32_t a = 1; a < 3; ++a)
        if (std::abs(direction[a]) < std::abs(direction[least]))
            least = a;
    double coordinate[3]{};
    coordinate[least] = 1.0;
    const double* from[2]{coordinate, across[0]};
    for (uint32_t k = 0; k < 2; ++k)
    {
        const double* v = from[k];
        across[k][0]    = direction[1] * v[2] - direction[2] * v[1];
        across[k][1]    = direction[2] * v[0] - direction[0] * v[2];
        across[k][2]    = direction[0] * v[1] - direction[1] * v[0];
        const double length =
            std::sqrt(across[k][0] * across[k][0] + across[k][1] * across[k][1] + across[k][2] * across[k][2]);
        for (uint32_t a = 0; a < 3; ++a)
            across[k][a] /= length;
    }
}

// Where a sphere swept along its path can first touch a triangle. In the path's
// frame (the direction d and two unit axes across it) a point s along d and h from the
// axis is within R of the center at parameter t only while h <= R and
// t >= (s - |d| sqrt(R^2 - h^2)) / |d|^2. A triangle therefore touches no earlier than
// its least s less the half chord at its least distance from the axis, and not at all
// when its projection across the path stays farther than R from the axis. R is the
// largest center distance considerSphereSweepContact accepts. Local vertices project
// through the frame pulled back by the placement, and the allowance covers rounding of
// the world triangle and of that predicate as well as of these projections.
class SpherePathBound
{
  public:
    SpherePathBound(const std::array<double, 16u>& worldFromLocal, const ovspatialquery_aabb_t& localBounds,
                    const double center[3], double reach, const float unitDirection[3], double minimumDistance,
                    double maximumDistance)
    {
        double axes[3][3]{}, squared = 0.0;
        for (uint32_t a = 0; a < 3; ++a)
        {
            axes[0][a] = static_cast<double>(unitDirection[a]);
            squared += axes[0][a] * axes[0][a];
        }
        crossAxes(axes[0], axes + 1);
        double placement = 0.0, extent = 0.0, linear = 0.0;
        for (uint32_t a = 0; a < 3; ++a)
        {
            placement += std::abs(worldFromLocal[a * 4 + 3]) + std::abs(center[a]);
            extent = std::max(std::max(extent, std::abs(double(localBounds.minimum[a]))),
                              std::abs(double(localBounds.maximum[a])));
            for (uint32_t b = 0; b < 3; ++b)
                linear = std::max(linear, std::abs(worldFromLocal[a * 4 + b]));
        }
        for (uint32_t k = 0; k < 3; ++k)
        {
            mOffset[k] = 0.0;
            for (uint32_t a = 0; a < 3; ++a)
            {
                mOffset[k] += (worldFromLocal[a * 4 + 3] - center[a]) * axes[k][a];
                mRows[k][a] = worldFromLocal[0 * 4 + a] * axes[k][0] + worldFromLocal[1 * 4 + a] * axes[k][1] +
                              worldFromLocal[2 * 4 + a] * axes[k][2];
            }
        }
        const double magnitude =
            placement + 9.0 * linear * extent + std::abs(minimumDistance) + std::abs(maximumDistance) + reach;
        mAllowance        = 256.0 * std::numeric_limits<double>::epsilon() * magnitude;
        mReach            = reach + mAllowance;
        mDirectionSquared = squared;
        mDirectionLength  = std::sqrt(squared);
        mValid            = std::isfinite(mAllowance) && std::isfinite(mReach) && std::isfinite(mOffset[0]) &&
                 std::isfinite(mOffset[1]) && std::isfinite(mOffset[2]);
    }

    // No point of the triangle can touch the sphere before this parameter; false when the
    // triangle stays beyond the reach of the axis. The quick bound measures the axis's
    // distance to the triangle's box across the path, the exact one to the triangle itself.
    bool entry(const float* const corners[3], bool exact, double& outEntry) const
    {
        double least = std::numeric_limits<double>::infinity(), across[3][2], low[2], high[2];
        for (uint32_t corner = 0; corner < 3; ++corner)
        {
            least = std::min(least, project(0, corners[corner]));
            for (uint32_t k = 0; k < 2; ++k)
            {
                across[corner][k] = project(k + 1, corners[corner]);
                low[k]            = corner == 0 ? across[corner][k] : std::min(low[k], across[corner][k]);
                high[k]           = corner == 0 ? across[corner][k] : std::max(high[k], across[corner][k]);
            }
        }
        double axis = 0.0;
        if (exact)
            axis = std::sqrt(axisDistanceSquared(across));
        else
        {
            for (uint32_t k = 0; k < 2; ++k)
            {
                const double gap = std::max(std::max(0.0, low[k]), -high[k]);
                axis += gap * gap;
            }
            axis = std::sqrt(axis);
        }
        axis = std::max(0.0, axis - mAllowance);
        if (!(axis <= mReach))
            return false;
        const double halfChord = std::sqrt((mReach - axis) * (mReach + axis));
        outEntry               = (least - mAllowance - mDirectionLength * halfChord) / mDirectionSquared - mAllowance;
        return true;
    }

    // Without a finite frame every gathered triangle is evaluated.
    bool valid() const { return mValid; }

  private:
    double project(uint32_t axis, const float* corner) const
    {
        return mRows[axis][0] * corner[0] + mRows[axis][1] * corner[1] + mRows[axis][2] * corner[2] + mOffset[axis];
    }
    // Squared distance from the axis (the origin across the path) to the projected triangle.
    static double axisDistanceSquared(const double p[3][2])
    {
        double sides[3]{}, best = std::numeric_limits<double>::infinity();
        for (uint32_t k = 0; k < 3; ++k)
        {
            const double* a = p[k];
            const double* b = p[(k + 1) % 3];
            const double  e[2]{b[0] - a[0], b[1] - a[1]};
            sides[k]              = a[1] * e[0] - a[0] * e[1];
            const double toOrigin = -(a[0] * e[0] + a[1] * e[1]);
            const double length   = e[0] * e[0] + e[1] * e[1];
            const double start    = a[0] * a[0] + a[1] * a[1];
            // The nearest point of this edge: its start, its end or its interior foot.
            const double squared = toOrigin <= 0.0      ? start
                                   : toOrigin >= length ? b[0] * b[0] + b[1] * b[1]
                                                        : start - toOrigin * (toOrigin / length);
            best                 = std::min(best, squared);
        }
        // The origin lies inside when it is on the same side of every edge.
        if ((sides[0] > 0.0 && sides[1] > 0.0 && sides[2] > 0.0) ||
            (sides[0] < 0.0 && sides[1] < 0.0 && sides[2] < 0.0))
            return 0.0;
        return std::max(0.0, best);
    }
    double mRows[3][3]{};
    double mOffset[3]{};
    double mAllowance        = 0.0;
    double mReach            = 0.0;
    double mDirectionSquared = 0.0;
    double mDirectionLength  = 0.0;
    bool   mValid            = false;
};

// Where a sphere's center first passes a surface point it holds: the nearest hit of
// rays along its axis and half its radius to either side, infinite when none hits.
// The sweep's first contact usually lies near it, but correctness never depends on it.
double firstSurfaceNearAxis(const PhysXTriangleMesh& mesh, const double origin[3], const double direction[3],
                            double offset, double range, InstanceHostLedger* ledger)
{
    double across[2][3]{};
    crossAxes(direction, across);
    double first = std::numeric_limits<double>::infinity();
    for (uint32_t ray = 0; ray < 5; ++ray)
    {
        const double side = ray == 0 ? 0.0 : ray % 2 ? offset : -offset;
        double       start[3]{};
        for (uint32_t a = 0; a < 3; ++a)
            start[a] = origin[a] + side * across[ray == 0 ? 0 : (ray - 1) / 2][a];
        bool               surface = false;
        PreciseTriangleHit hit{};
        if (mesh.raycast(start, direction, range, true, false, surface, hit, ledger) == OVSPATIALQUERY_API_SUCCESS &&
            surface)
            first = std::min(first, hit.distance);
    }
    return first;
}

// Meshes up to this many triangles gather a sweep's whole capsule at once; the axis rays
// that split a larger mesh's gather would cost more than they save.
constexpr uint32_t kWholeGatherTriangles = 4096u;

// A gathered triangle keyed by a lower bound on its first contact, `exact` once that
// bound measures the axis's distance to the triangle rather than to its box.
struct OrderedTriangle
{
    double   entry;
    uint32_t triangle;
    bool     exact;
    bool     operator>(const OrderedTriangle& other) const
    {
        return entry > other.entry || (entry == other.entry && triangle > other.triangle);
    }
};

// The gathered triangles a sweep can touch, in a heap ordered by entry bound.
struct TriangleOrder
{
    HostLease                          charge;
    std::unique_ptr<OrderedTriangle[]> large;
    OrderedTriangle                    small[128];
    uint32_t                           count = 0;

    OrderedTriangle* data() { return large ? large.get() : small; }
};

} // namespace

bool PreparedTriangleMesh::sweepSphere(const std::array<double, 16u>& worldFromLocal, const double center[3],
                                       double radius, const float unitDirection[3], double minimumDistance,
                                       double maximumDistance, bool doubleSided, bool flipBackFaces,
                                       PreciseTriangleHit& outHit, const PhysXTriangleMesh& cooked,
                                       const TriangleGatherVolume& volume, bool raycast, InstanceHostLedger* ledger,
                                       ovspatialquery_api_status_t* outStatus, const uint32_t* onlyTriangle) const
{
    bool       hasHit          = false;
    double     closestDistance = maximumDistance;
    const auto fail            = [&](ovspatialquery_api_status_t status)
    {
        if (outStatus)
            *outStatus = status;
        return false;
    };
    const auto worldTriangle = [&](uint32_t faceIndex, double triangle[3][3])
    {
        const uint32_t* indices = mTopology->triangles.data() + static_cast<size_t>(faceIndex) * 3u;
        for (uint32_t corner = 0u; corner < 3u; ++corner)
        {
            transformMeshVertex(mVertices, indices[corner], worldFromLocal, triangle[corner]);
        }
    };
    const double inverseDirectionSquare = inverseSquaredLength(unitDirection);
    const auto   evaluate               = [&](uint32_t faceIndex, const double triangle[3][3])
    {
        double faceNormal[3] = {};
        if (!worldTriangleNormal(triangle, faceNormal))
        {
            return;
        }
        if (raycast)
        {
            const double velocity =
                faceNormal[0] * unitDirection[0] + faceNormal[1] * unitDirection[1] + faceNormal[2] * unitDirection[2];
            if (velocity == 0 || (!doubleSided && velocity > 0))
                return;
        }
        double signedOrigin   = 0.0;
        double normalVelocity = 0.0;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            signedOrigin   = std::fma(center[axis] - triangle[0][axis], faceNormal[axis], signedOrigin);
            normalVelocity = std::fma(static_cast<double>(unitDirection[axis]), faceNormal[axis], normalVelocity);
        }
        if (!raycast)
        {
            // A sphere ignores a single-sided triangle whose front normal points along the motion
            // and meets every other one as a two-sided surface (ADR-0051), so the least contact
            // time found below is the first touch from either side.
            if (!doubleSided && normalVelocity > 0.0)
                return;
            considerSphereSweepContact(triangle, faceNormal, center, radius, unitDirection, minimumDistance,
                                       maximumDistance, minimumDistance, faceIndex, hasHit, closestDistance, outHit,
                                       flipBackFaces);
        }
        // Rays skip the initial contact above, whose tolerance must not manufacture a hit at the
        // near clipping boundary, and meet a single-sided plane only from the front.
        const int32_t firstSide = raycast && !doubleSided ? 1 : -1;
        for (int32_t side = firstSide; side <= 1; side += 2)
        {
            const double orientedVelocity = static_cast<double>(side) * normalVelocity;
            if (orientedVelocity >= 0.0)
            {
                continue;
            }
            const double distance = (static_cast<double>(side) * radius - signedOrigin) / normalVelocity;
            considerSphereSweepContact(triangle, faceNormal, center, radius, unitDirection, minimumDistance,
                                       maximumDistance, distance, faceIndex, hasHit, closestDistance, outHit,
                                       flipBackFaces);
        }
        if (raycast)
            return;
        for (uint32_t corner = 0u; corner < 3u; ++corner)
        {
            double distance = 0.0;
            if (raySphereEntry(center, unitDirection, inverseDirectionSquare, triangle[corner], radius, distance))
            {
                considerSphereSweepContact(triangle, faceNormal, center, radius, unitDirection, minimumDistance,
                                           maximumDistance, distance, faceIndex, hasHit, closestDistance, outHit,
                                           flipBackFaces);
            }
            const uint32_t next = (corner + 1u) % 3u;
            if (rayCapsuleEdgeEntry(center, unitDirection, triangle[corner], triangle[next], radius, distance))
            {
                considerSphereSweepContact(triangle, faceNormal, center, radius, unitDirection, minimumDistance,
                                           maximumDistance, distance, faceIndex, hasHit, closestDistance, outHit,
                                           flipBackFaces);
            }
        }
    };
    const auto evaluateAll = [&](const TriangleCandidates& candidates)
    {
        for (uint32_t candidate = 0; candidate < candidates.count; ++candidate)
        {
            double triangle[3][3] = {};
            worldTriangle(candidates.data()[candidate], triangle);
            evaluate(candidates.data()[candidate], triangle);
        }
    };
    const uint32_t triangleCount = static_cast<uint32_t>(mTopology->triangles.size() / 3);
    if (raycast && onlyTriangle != nullptr)
    {
        if (*onlyTriangle >= triangleCount)
            return fail(OVSPATIALQUERY_API_INTERNAL);
        double triangle[3][3] = {};
        worldTriangle(*onlyTriangle, triangle);
        evaluate(*onlyTriangle, triangle);
    }
    else if (raycast)
    {
        TriangleCandidates candidates;
        const auto         status = gatherTriangleCandidates(cooked, volume, triangleCount, ledger, candidates);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return fail(status);
        evaluateAll(candidates);
    }
    else
    {
        const SpherePathBound bound(worldFromLocal, mLocalBounds, center, sphereContactAcceptanceRadius(radius),
                                    unitDirection, minimumDistance, maximumDistance);
        const auto            corners = [&](uint32_t faceIndex, const float* out[3])
        {
            const uint32_t* indices = mTopology->triangles.data() + static_cast<size_t>(faceIndex) * 3u;
            for (uint32_t corner = 0u; corner < 3u; ++corner)
                out[corner] = mVertices.data() + static_cast<size_t>(indices[corner]) * 3u;
        };
        // Evaluates gathered triangles nearest first and stops once none left can touch the
        // sphere before the closest contact; a triangle entered at that contact can still tie it.
        const auto evaluateGathered = [&](const TriangleCandidates& candidates)
        {
            if (!bound.valid())
            {
                evaluateAll(candidates);
                return OVSPATIALQUERY_API_SUCCESS;
            }
            TriangleOrder order;
            if (candidates.count > 128u)
            {
                if (ledger)
                {
                    HostTransaction transaction(*ledger);
                    if (!transaction.reserve(uint64_t(candidates.count) * sizeof(OrderedTriangle)) ||
                        !transaction.commit(order.charge))
                        return OVSPATIALQUERY_API_HOST_BUDGET;
                }
                order.large.reset(new OrderedTriangle[candidates.count]);
            }
            OrderedTriangle* ordered = order.data();
            for (uint32_t candidate = 0; candidate < candidates.count; ++candidate)
            {
                const float* points[3]{};
                double       entry = 0.0;
                corners(candidates.data()[candidate], points);
                if (bound.entry(points, false, entry) && entry <= closestDistance)
                    ordered[order.count++] = {entry, candidates.data()[candidate], false};
            }
            // A triangle reaching the front with its quick bound takes the exact one and
            // returns to the heap when that no longer puts it first.
            const auto later = std::greater<OrderedTriangle>();
            std::make_heap(ordered, ordered + order.count, later);
            for (uint32_t remaining = order.count; remaining > 0u;)
            {
                std::pop_heap(ordered, ordered + remaining, later);
                OrderedTriangle& next = ordered[remaining - 1u];
                if (next.entry > closestDistance)
                    break;
                if (!next.exact)
                {
                    const float* points[3]{};
                    corners(next.triangle, points);
                    next.exact = true;
                    if (!bound.entry(points, true, next.entry) || next.entry > closestDistance)
                    {
                        --remaining;
                        continue;
                    }
                    if (remaining > 1u && later(next, ordered[0]))
                    {
                        std::push_heap(ordered, ordered + remaining, later);
                        continue;
                    }
                }
                --remaining;
                double triangle[3][3] = {};
                worldTriangle(next.triangle, triangle);
                evaluate(next.triangle, triangle);
            }
            return OVSPATIALQUERY_API_SUCCESS;
        };
        // A capsule through a large mesh is gathered in two parts, the first ending where the
        // sphere first holds a surface point near its axis. The rest is gathered only when the
        // closest contact lies beyond that part: every triangle not gathered stays beyond the
        // evaluator's reach from each center up to the part's end.
        const bool   split = volume.capsule && triangleCount > kWholeGatherTriangles;
        const double firstEnd =
            !split ? maximumDistance
                   : std::min(maximumDistance,
                              minimumDistance + firstSurfaceNearAxis(cooked, volume.start, volume.direction,
                                                                     0.5 * volume.radius,
                                                                     maximumDistance - minimumDistance, ledger));
        TriangleGatherVolume part      = volume;
        double               partStart = minimumDistance;
        double               partEnd   = minimumDistance;
        do
        {
            // The second part ends at a contact the first part found beyond it.
            partEnd = partStart < firstEnd ? firstEnd : closestDistance;
            if (split)
                for (uint32_t a = 0; a < 3; ++a)
                {
                    part.start[a] = std::fma(partStart - minimumDistance, volume.direction[a], volume.start[a]);
                    part.end[a]   = partEnd == maximumDistance
                                        ? volume.end[a]
                                        : std::fma(partEnd - minimumDistance, volume.direction[a], volume.start[a]);
                }
            TriangleCandidates candidates;
            auto status = gatherTriangleCandidates(cooked, part, triangleCount, ledger, candidates, split);
            if (status == OVSPATIALQUERY_API_SUCCESS)
                status = evaluateGathered(candidates);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return fail(status);
        } while ((partStart = partEnd) < maximumDistance && !(hasHit && closestDistance <= partEnd));
    }
    if (hasHit)
    {
        const uint32_t  generatedTriangle = static_cast<uint32_t>(outHit.faceIndex);
        const uint32_t* corners           = sourceCorners(generatedTriangle);
        outHit.generatedTriangle          = generatedTriangle;
        outHit.sourceCorners[0]           = corners[0];
        outHit.sourceCorners[1]           = corners[1];
        outHit.sourceCorners[2]           = corners[2];
        outHit.faceIndex                  = sourceFace(generatedTriangle);
        outHit.hasTriangleSource          = true;
    }
    return hasHit;
}

bool PreparedTriangleMesh::overlap(const std::array<double, 16u>&        worldFromLocal,
                                   const ovspatialquery_overlap_shape_t& geometry, const double (*boxRotation)[3],
                                   const PhysXTriangleMesh& cooked, const double localMinimum[3],
                                   const double localMaximum[3], InstanceHostLedger* ledger,
                                   ovspatialquery_api_status_t* outStatus) const
{
    TriangleCandidates candidates;
    const auto         status = gatherTriangleCandidates(
        cooked, localMinimum, localMaximum, static_cast<uint32_t>(mTopology->triangles.size() / 3), ledger, candidates);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        if (outStatus)
            *outStatus = status;
        return false;
    }
    double rotation[3][3] = {};
    if (geometry.kind == OVSPATIALQUERY_OVERLAP_BOX)
        quaternionRotation(geometry.data.box.rotation_xyzw, boxRotation, rotation);
    for (uint32_t candidate = 0; candidate < candidates.count; ++candidate)
    {
        const uint32_t  faceIndex      = candidates.data()[candidate];
        const uint32_t* indices        = mTopology->triangles.data() + static_cast<size_t>(faceIndex) * 3u;
        double          triangle[3][3] = {};
        for (uint32_t corner = 0u; corner < 3u; ++corner)
        {
            transformMeshVertex(mVertices, indices[corner], worldFromLocal, triangle[corner]);
        }
        if (geometry.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
        {
            const ovspatialquery_overlap_sphere_t& sphere     = geometry.data.sphere;
            double                                 closest[3] = {};
            double                                 u          = 0.0;
            double                                 v          = 0.0;
            closestPointOnTriangle(triangle, sphere.center, closest, u, v);
            double normalizedDistanceSquared = 0.0;
            bool   separated                 = false;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                const double distance = sphere.center[axis] - closest[axis];
                if (std::abs(distance) > sphere.radius)
                {
                    separated = true;
                    break;
                }
                const double normalized   = distance / sphere.radius;
                normalizedDistanceSquared = std::fma(normalized, normalized, normalizedDistanceSquared);
            }
            if (!separated && normalizedDistanceSquared <= 1.0)
            {
                return true;
            }
        }
        else
        {
            double localTriangle[3][3] = {};
            double halfExtents[3]      = {};
            if (geometry.kind == OVSPATIALQUERY_OVERLAP_AABB)
            {
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    const double center =
                        geometry.data.aabb.min_corner[axis] * 0.5 + geometry.data.aabb.max_corner[axis] * 0.5;
                    halfExtents[axis] =
                        geometry.data.aabb.max_corner[axis] * 0.5 - geometry.data.aabb.min_corner[axis] * 0.5;
                    for (uint32_t corner = 0u; corner < 3u; ++corner)
                    {
                        localTriangle[corner][axis] = triangle[corner][axis] - center;
                    }
                }
            }
            else if (geometry.kind == OVSPATIALQUERY_OVERLAP_BOX)
            {
                for (uint32_t localAxis = 0u; localAxis < 3u; ++localAxis)
                {
                    halfExtents[localAxis] = geometry.data.box.half_extents[localAxis];
                    for (uint32_t corner = 0u; corner < 3u; ++corner)
                    {
                        for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
                        {
                            localTriangle[corner][localAxis] =
                                std::fma(rotation[worldAxis][localAxis],
                                         triangle[corner][worldAxis] - geometry.data.box.center[worldAxis],
                                         localTriangle[corner][localAxis]);
                        }
                    }
                }
            }
            else
            {
                return false;
            }
            if (triangleOverlapsCenteredBox(localTriangle, halfExtents))
            {
                return true;
            }
        }
    }
    return false;
}

ovspatialquery_aabb_t preparedTriangleMeshLocalBounds(const PreparedTriangleMesh& prepared) noexcept
{
    return prepared.localBounds();
}

uint64_t preparedTriangleMeshDroppedCount(const PreparedTriangleMesh& prepared) noexcept
{
    return prepared.droppedTriangles().size();
}

std::string describeDroppedTriangles(const PreparedTriangleMesh& prepared, const char* faceNoun)
{
    constexpr size_t                    kListed = 8u;
    const std::vector<DroppedTriangle>& dropped = prepared.droppedTriangles();
    if (dropped.empty())
    {
        return {};
    }
    std::string faces;
    uint64_t    previous = 0u;
    size_t      listed   = 0u;
    for (size_t index = 0u; index < dropped.size() && listed < kListed; ++index)
    {
        // A face triangulated into several triangles without area is named once.
        if (index != 0u && dropped[index].sourceFace == previous)
            continue;
        faces += (listed++ == 0u ? "" : ", ") + std::to_string(dropped[index].sourceFace);
        previous = dropped[index].sourceFace;
    }
    if (listed == kListed && dropped.back().sourceFace != previous)
        faces += ", ...";
    return "dropped " + std::to_string(dropped.size()) + " of its " + std::to_string(prepared.sourceTriangleCount()) +
           " triangles, which have no area (" + faceNoun + (listed == 1u ? " " : "s ") + faces + ")";
}

namespace
{

bool prepareObject(const std::shared_ptr<const PreparedTriangleMesh>& mesh,
                   const std::array<double, 16u>& worldFromLocal, bool doubleSided,
                   std::shared_ptr<const TriangleObjectGeometry>& outObject)
{
    for (double value : worldFromLocal)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    if (worldFromLocal[12] != 0.0 || worldFromLocal[13] != 0.0 || worldFromLocal[14] != 0.0 ||
        worldFromLocal[15] != 1.0)
    {
        return false;
    }
    double       linearScale     = 0.0;
    const size_t linearIndices[] = {
        0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 10u,
    };
    for (size_t index : linearIndices)
    {
        linearScale = std::max(linearScale, std::abs(worldFromLocal[index]));
    }
    if (linearScale == 0.0)
    {
        return false;
    }
    const double a           = worldFromLocal[0] / linearScale;
    const double b           = worldFromLocal[1] / linearScale;
    const double c           = worldFromLocal[2] / linearScale;
    const double d           = worldFromLocal[4] / linearScale;
    const double e           = worldFromLocal[5] / linearScale;
    const double f           = worldFromLocal[6] / linearScale;
    const double g           = worldFromLocal[8] / linearScale;
    const double h           = worldFromLocal[9] / linearScale;
    const double i           = worldFromLocal[10] / linearScale;
    const double determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!std::isfinite(determinant) || determinant == 0.0)
    {
        return false;
    }
    const double           inverseDeterminant = 1.0 / determinant;
    std::array<double, 9u> inverse{
        (e * i - f * h) * inverseDeterminant / linearScale, (c * h - b * i) * inverseDeterminant / linearScale,
        (b * f - c * e) * inverseDeterminant / linearScale, (f * g - d * i) * inverseDeterminant / linearScale,
        (a * i - c * g) * inverseDeterminant / linearScale, (c * d - a * f) * inverseDeterminant / linearScale,
        (d * h - e * g) * inverseDeterminant / linearScale, (b * g - a * h) * inverseDeterminant / linearScale,
        (a * e - b * d) * inverseDeterminant / linearScale,
    };
    for (double value : inverse)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    // Columns of one length and orthogonal, within rounding of the adapters' own checks.
    double squared[3]{};
    for (uint32_t column = 0u; column < 3u; ++column)
        for (uint32_t row = 0u; row < 3u; ++row)
            squared[column] += worldFromLocal[row * 4u + column] * worldFromLocal[row * 4u + column];
    if (determinant < 0.0 || !(squared[0] > 0.0) || !std::isfinite(squared[0]))
        return false;
    for (uint32_t first = 0u; first < 3u; ++first)
    {
        if (std::abs(squared[first] - squared[0]) > squared[0] * 1e-10)
            return false;
        for (uint32_t second = first + 1u; second < 3u; ++second)
        {
            double product = 0.0;
            for (uint32_t row = 0u; row < 3u; ++row)
                product += worldFromLocal[row * 4u + first] * worldFromLocal[row * 4u + second];
            if (std::abs(product) > squared[0] * 1e-10)
                return false;
        }
    }
    std::shared_ptr<TriangleObjectGeometry> object = std::make_shared<TriangleObjectGeometry>();
    object->mesh                                   = mesh;
    object->worldFromLocal                         = worldFromLocal;
    object->localFromWorldLinear                   = inverse;
    object->uniformScale                           = std::sqrt(squared[0]);
    object->doubleSided                            = doubleSided;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        object->worldBounds.minimum[axis] = std::numeric_limits<double>::infinity();
        object->worldBounds.maximum[axis] = -std::numeric_limits<double>::infinity();
    }
    const ovspatialquery_aabb_t& localBounds = mesh->localBounds();
    for (uint32_t corner = 0u; corner < 8u; ++corner)
    {
        double local[3] = {};
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            local[axis] = (corner & (1u << axis)) != 0u ? localBounds.maximum[axis] : localBounds.minimum[axis];
        }
        for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
        {
            const size_t row   = static_cast<size_t>(worldAxis) * 4u;
            const double world = std::fma(worldFromLocal[row + 2u], local[2],
                                          std::fma(worldFromLocal[row + 1u], local[1],
                                                   std::fma(worldFromLocal[row], local[0], worldFromLocal[row + 3u])));
            if (!std::isfinite(world))
            {
                return false;
            }
            object->worldBounds.minimum[worldAxis] = std::min(object->worldBounds.minimum[worldAxis], world);
            object->worldBounds.maximum[worldAxis] = std::max(object->worldBounds.maximum[worldAxis], world);
        }
    }
    outObject = std::move(object);
    return true;
}

} // namespace

struct TriangleGeometryCache::Impl
{
    std::mutex                                                      mutex;
    std::map<uint64_t, std::shared_ptr<const PreparedTriangleMesh>> meshes;
};

TriangleGeometryCache::TriangleGeometryCache() : mImpl(new Impl()) {}

TriangleGeometryCache::~TriangleGeometryCache() = default;

ovspatialquery_api_status_t TriangleGeometryCache::adoptDefinitions(
    const std::vector<PreparedTriangleDefinition>& definitions)
{
    try
    {
        std::lock_guard<std::mutex>                                     lock(mImpl->mutex);
        std::map<uint64_t, std::shared_ptr<const PreparedTriangleMesh>> candidate = mImpl->meshes;
        for (const PreparedTriangleDefinition& definition : definitions)
        {
            if (definition.key == 0u || definition.geometry == nullptr || definition.geometry->mesh == nullptr ||
                !candidate.emplace(definition.key, definition.geometry->mesh).second)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
        }
        mImpl->meshes.swap(candidate);
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

ovspatialquery_api_status_t TriangleGeometryCache::bindDefinition(
    uint64_t key, const std::array<double, 16u>& worldFromLocal, bool doubleSided,
    std::shared_ptr<const TriangleObjectGeometry>& outObject) const
{
    std::shared_ptr<const PreparedTriangleMesh> mesh;
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        const auto                  found = mImpl->meshes.find(key);
        if (found == mImpl->meshes.end())
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        mesh = found->second;
    }
    return bindPreparedTriangleGeometry(mesh, worldFromLocal, doubleSided, outObject);
}

ovspatialquery_api_status_t TriangleGeometryCache::localBounds(uint64_t key, ovspatialquery_aabb_t& outBounds) const
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    const auto                  found = mImpl->meshes.find(key);
    if (found == mImpl->meshes.end())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    outBounds = found->second->localBounds();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t TriangleGeometryCache::droppedTriangleCount(uint64_t key, uint64_t& outCount) const
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    const auto                  found = mImpl->meshes.find(key);
    if (found == mImpl->meshes.end())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    outCount = found->second->droppedTriangles().size();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t admitTriangleMeshStructures(InstanceHostLedger&                             ledger,
                                                        const std::vector<const PreparedTriangleMesh*>& meshes)
{
    try
    {
        std::vector<const PreparedTriangleMesh*>        admitted;
        std::unordered_set<const PreparedTriangleMesh*> seen;
        std::vector<uint64_t>                           resident, scratch;
        std::vector<size_t>                             claimOf;
        HostTransaction                                 transaction(ledger);
        for (const PreparedTriangleMesh* mesh : meshes)
        {
            if (mesh == nullptr || !seen.insert(mesh).second)
                continue;
            uint64_t structureBytes = 0u, scratchBytes = 0u;
            mesh->structureAdmission(structureBytes, scratchBytes);
            if (structureBytes == 0u)
                continue;
            admitted.push_back(mesh);
            resident.push_back(structureBytes);
            claimOf.push_back(scratchBytes == 0u ? SIZE_MAX : scratch.size());
            if (scratchBytes != 0u)
                scratch.push_back(scratchBytes);
            if (!transaction.reserve(structureBytes))
                return OVSPATIALQUERY_API_HOST_BUDGET;
        }
        if (admitted.empty())
            return OVSPATIALQUERY_API_SUCCESS;
        HostLease structures;
        if (!transaction.commit(structures))
            return OVSPATIALQUERY_API_HOST_BUDGET;
        std::vector<BuildScratchClaim> claims;
        const BuildScratchOutcome      outcome = ledger.claimBuildScratch(scratch, claims);
        if (outcome != BuildScratchOutcome::success)
            return outcome == BuildScratchOutcome::budget ? OVSPATIALQUERY_API_HOST_BUDGET
                                                          : OVSPATIALQUERY_API_OUT_OF_MEMORY;
        for (size_t index = 0u; index < admitted.size(); ++index)
        {
            HostLease structure;
            if (!structures.split(resident[index], structure))
                return OVSPATIALQUERY_API_INTERNAL;
            admitted[index]->adoptStructure(std::move(structure), claimOf[index] == SIZE_MAX
                                                                      ? BuildScratchClaim()
                                                                      : std::move(claims[claimOf[index]]));
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

void collectTriangleMesh(const TriangleObjectGeometry* object, std::vector<const PreparedTriangleMesh*>& meshes)
{
    if (object != nullptr && object->mesh != nullptr)
        meshes.push_back(object->mesh.get());
}

ovspatialquery_api_status_t prepareTriangleMesh(const TriangleMeshSource&                    source,
                                                std::shared_ptr<const PreparedTriangleMesh>& outMesh)
{
    return source.topologyBase != nullptr ? PreparedTriangleMesh::createDerived(source.topologyBase, source, outMesh)
                                          : PreparedTriangleMesh::create(source, outMesh);
}

uint64_t triangleMeshContentHash(const TriangleMeshSource& source) noexcept
{
    // The vertices, then the topology's content hash, so a mesh read in full and one
    // derived from an earlier preparation hash alike when their content is the same
    // and identical prims keep sharing one preparation while they deform together.
    ContentHasher hasher;
    hasher.mixArray(source.vertices);
    hasher.mix(source.topologyBase != nullptr
                   ? source.topologyBase->topologyContentHash()
                   : hashTopologyArrays(source.triangles, source.sourceFaces, source.sourceCorners));
    return hasher.hash;
}

bool preparedTriangleMeshMatches(const PreparedTriangleMesh& prepared, const TriangleMeshSource& source) noexcept
{
    return prepared.sameContent(source);
}

ovspatialquery_api_status_t bindPreparedTriangleGeometry(const std::shared_ptr<const PreparedTriangleMesh>& prepared,
                                                         const std::array<double, 16u>&                 worldFromLocal,
                                                         bool                                           doubleSided,
                                                         std::shared_ptr<const TriangleObjectGeometry>& outObject)
{
    if (prepared == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        return prepareObject(prepared, worldFromLocal, doubleSided, outObject) ? OVSPATIALQUERY_API_SUCCESS
                                                                               : OVSPATIALQUERY_API_INVALID_ARGUMENT;
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

ovspatialquery_api_status_t TriangleGeometryCache::destroyDefinition(uint64_t key)
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    const auto                  found = mImpl->meshes.find(key);
    if (found == mImpl->meshes.end())
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (found->second.use_count() != 1)
    {
        return OVSPATIALQUERY_API_BUSY;
    }
    mImpl->meshes.erase(found);
    return OVSPATIALQUERY_API_SUCCESS;
}

namespace
{
void meshLocalQuery(const TriangleObjectGeometry& object, const double origin[3], const float direction[3],
                    double minimum, double localOrigin[3], double localDirection[3])
{
    for (uint32_t a = 0; a < 3; ++a)
        for (uint32_t b = 0; b < 3; ++b)
        {
            const double inverse = object.localFromWorldLinear[a * 3 + b];
            localOrigin[a] = std::fma(inverse, (origin[b] - object.worldFromLocal[b * 4 + 3]) + minimum * direction[b],
                                      localOrigin[a]);
            localDirection[a] = std::fma(inverse, double(direction[b]), localDirection[a]);
        }
}
void meshLocalBounds(const TriangleObjectGeometry& object, const double center[3], const double half[3],
                     double minimum[3], double maximum[3])
{
    const float zero[3]{};
    double      localCenter[3]{}, unused[3]{};
    meshLocalQuery(object, center, zero, 0, localCenter, unused);
    for (uint32_t a = 0; a < 3; ++a)
    {
        double extent = 0;
        for (uint32_t b = 0; b < 3; ++b)
            extent += std::abs(object.localFromWorldLinear[a * 3 + b]) * half[b];
        minimum[a] = localCenter[a] - extent;
        maximum[a] = localCenter[a] + extent;
    }
}
void meshWorldHit(const TriangleObjectGeometry& object, PreciseTriangleHit& hit, double minimum)
{
    hit.distance += minimum;
    hit.faceIndex = object.mesh->sourceFace(hit.generatedTriangle);
    std::copy_n(object.mesh->sourceCorners(hit.generatedTriangle), 3, hit.sourceCorners);
    hit.hasTriangleSource = true;
    double normal[3]{}, position[3]{};
    for (uint32_t a = 0; a < 3; ++a)
    {
        position[a] = object.worldFromLocal[a * 4 + 3];
        for (uint32_t b = 0; b < 3; ++b)
        {
            position[a] = std::fma(object.worldFromLocal[a * 4 + b], hit.position[b], position[a]);
            normal[a] += object.localFromWorldLinear[b * 3 + a] * hit.normal[b];
        }
    }
    const double length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    for (uint32_t a = 0; a < 3; ++a)
    {
        hit.position[a] = position[a];
        hit.normal[a]   = float(normal[a] / length);
    }
}
} // namespace

ovspatialquery_api_status_t raycastTriangleObject(const TriangleObjectGeometry& object, const double origin[3],
                                                  const float unitDirection[3], double minimumDistance,
                                                  double maximumDistance, bool /* anyHit */, InstanceHostLedger* ledger,
                                                  bool& hasHit, PreciseTriangleHit& outHit, bool bothSides)
{
    hasHit = false;
    double localOrigin[3]{}, localDirection[3]{};
    meshLocalQuery(object, origin, unitDirection, minimumDistance, localOrigin, localDirection);
    const PhysXTriangleMesh* mesh              = nullptr;
    const auto               preparationStatus = object.mesh->physxMesh(mesh);
    if (preparationStatus != OVSPATIALQUERY_API_SUCCESS)
        return preparationStatus;
    PreciseTriangleHit hit{};
    auto status = mesh->raycast(localOrigin, localDirection, maximumDistance - minimumDistance, object.doubleSided,
                                bothSides, hasHit, hit, ledger);
    if (status == OVSPATIALQUERY_API_NOT_SUPPORTED)
    {
        // Double query ranges may exceed a float geometry query even though the
        // stored mesh is ordinary. Keep that case on the shared contact path.
        status = OVSPATIALQUERY_API_SUCCESS;
        hasHit = sweepSphereTriangleObject(object, origin, 0, unitDirection, minimumDistance, maximumDistance, outHit,
                                           ledger, &status, true, bothSides);
        return status;
    }
    if (status != OVSPATIALQUERY_API_SUCCESS || !hasHit)
        return status;
    meshWorldHit(object, hit, minimumDistance);
    outHit = hit;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t raycastTriangleObjectAll(const TriangleObjectGeometry& object, const double origin[3],
                                                     const float unitDirection[3], double minimumDistance,
                                                     double maximumDistance, InstanceHostLedger*           ledger,
                                                     bool (*emit)(void*, const PreciseTriangleHit&), void* context,
                                                     bool bothSides)
{
    double localOrigin[3]{}, localDirection[3]{};
    meshLocalQuery(object, origin, unitDirection, minimumDistance, localOrigin, localDirection);
    const PhysXTriangleMesh* mesh   = nullptr;
    const auto               status = object.mesh->physxMesh(mesh);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    struct WorldSink
    {
        const TriangleObjectGeometry& object;
        const PhysXTriangleMesh&      cooked;
        const double*                 origin;
        const float*                  unitDirection;
        double                        minimumDistance;
        double                        maximumDistance;
        InstanceHostLedger*           ledger;
        bool (*emit)(void*, const PreciseTriangleHit&);
        void*                       context;
        bool                        bothSides;
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
    } sink{object, *mesh, origin, unitDirection, minimumDistance, maximumDistance, ledger, emit, context, bothSides};
    const auto crossed = mesh->raycastAll(
        localOrigin, localDirection, maximumDistance - minimumDistance, object.doubleSided, bothSides, ledger,
        [](void* opaque, const PreciseTriangleHit& local, bool recheck)
        {
            auto& world = *static_cast<WorldSink*>(opaque);
            if (!recheck)
            {
                PreciseTriangleHit hit = local;
                meshWorldHit(world.object, hit, world.minimumDistance);
                return world.emit(world.context, hit);
            }
            // Decide the one triangle with the double contact evaluation the
            // nearest path falls back to; its hit is already in world form.
            PreciseTriangleHit exact{};
            const uint32_t     triangle = local.generatedTriangle;
            if (world.object.mesh->sweepSphere(
                    world.object.worldFromLocal, world.origin, 0, world.unitDirection, world.minimumDistance,
                    world.maximumDistance, world.object.doubleSided || world.bothSides, world.object.doubleSided, exact,
                    world.cooked, TriangleGatherVolume{}, true, world.ledger, &world.status, &triangle))
                return world.emit(world.context, exact);
            return world.status == OVSPATIALQUERY_API_SUCCESS;
        },
        &sink);
    return sink.status != OVSPATIALQUERY_API_SUCCESS ? sink.status : crossed;
}

bool sweepSphereTriangleObject(const TriangleObjectGeometry& object, const double center[3], double radius,
                               const float direction[3], double minimum, double maximum, PreciseTriangleHit& outHit,
                               InstanceHostLedger* ledger, ovspatialquery_api_status_t* outStatus, bool raycast,
                               bool bothSides, bool rayNormals)
{
    const PhysXTriangleMesh* mesh   = nullptr;
    auto                     status = object.mesh->physxMesh(mesh);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        if (outStatus)
            *outStatus = status;
        return false;
    }
    // Gather every triangle within the radius the exact evaluator accepts; PhysX's own
    // sphere sweep is not a candidate source, since it misses grazing vertex contacts.
    const double reach = raycast ? radius : sphereContactAcceptanceRadius(radius);
    double       boundsCenter[3]{}, half[3]{}, localMinimum[3]{}, localMaximum[3]{};
    for (uint32_t a = 0; a < 3; ++a)
    {
        boundsCenter[a] = std::fma(minimum * 0.5 + maximum * 0.5, double(direction[a]), center[a]);
        half[a]         = (maximum * 0.5 - minimum * 0.5) * std::abs(double(direction[a])) + reach;
    }
    meshLocalBounds(object, boundsCenter, half, localMinimum, localMaximum);
    TriangleGatherVolume volume{};
    std::copy_n(localMinimum, 3, volume.minimum);
    std::copy_n(localMaximum, 3, volume.maximum);
    if (radius > 0 && !raycast)
    {
        // Under a uniform rigid scale the local sweep is exact, so gather by the swept sphere itself.
        double localOrigin[3]{}, localDirection[3]{};
        meshLocalQuery(object, center, direction, minimum, localOrigin, localDirection);
        volume.capsule = true;
        volume.radius  = reach / object.uniformScale;
        for (uint32_t a = 0; a < 3; ++a)
        {
            volume.start[a]     = localOrigin[a];
            volume.end[a]       = std::fma(maximum - minimum, localDirection[a], localOrigin[a]);
            volume.direction[a] = localDirection[a];
        }
    }
    return object.mesh->sweepSphere(object.worldFromLocal, center, radius, direction, minimum, maximum,
                                    object.doubleSided || bothSides, !(raycast || rayNormals) || object.doubleSided,
                                    outHit, *mesh, volume, raycast, ledger, outStatus);
}

namespace
{
// Whether the object's world bounds lie inside the overlap shape by a margin of 2^-16 of
// the magnitudes PhysX's float test works with: the query's and placement's coordinates
// and the mesh's local coordinates in world units. Float rounding there is about 2^-24 of
// them, and the double test's far less, so both report a touch for every kept triangle,
// each of which has area and lies within the bounds.
bool overlapShapeContainsObject(const TriangleObjectGeometry& object, const ovspatialquery_overlap_shape_t& shape,
                                const double (*boxRotation)[3])
{
    double center[3]{}, half[3]{}, diagonal = 0;
    for (uint32_t a = 0; a < 3; ++a)
    {
        center[a] = object.worldBounds.minimum[a] * 0.5 + object.worldBounds.maximum[a] * 0.5;
        half[a]   = object.worldBounds.maximum[a] * 0.5 - object.worldBounds.minimum[a] * 0.5;
        diagonal  = std::fma(half[a], half[a], diagonal);
    }
    // Most meshes a small query touches are larger than it: a contained one's half diagonal
    // cannot exceed the shape's, so reject those before the margin's work.
    if (shape.kind == OVSPATIALQUERY_OVERLAP_SPHERE &&
        !(diagonal <= shape.data.sphere.radius * shape.data.sphere.radius))
        return false;
    if (shape.kind == OVSPATIALQUERY_OVERLAP_BOX)
    {
        const double* e = shape.data.box.half_extents;
        if (!(diagonal <= e[0] * e[0] + e[1] * e[1] + e[2] * e[2]))
            return false;
    }
    const ovspatialquery_aabb_t& bounds    = object.mesh->localBounds();
    double                       magnitude = 0;
    for (uint32_t a = 0; a < 3; ++a)
        // Nested pairs, not an initializer list: MSVC sends std::max({...}) of doubles to an
        // out-of-line vectorized max_element, a call per object in a giant overlap.
        magnitude = std::max(std::max(magnitude, std::abs(bounds.minimum[a])), std::abs(bounds.maximum[a]));
    magnitude *= object.uniformScale;
    for (uint32_t a = 0; a < 3; ++a)
        magnitude += std::abs(object.worldFromLocal[a * 4 + 3]);
    constexpr double kMargin = 0x1p-16;
    if (shape.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
    {
        const ovspatialquery_overlap_sphere_t& sphere = shape.data.sphere;
        double                                 far    = 0;
        for (uint32_t a = 0; a < 3; ++a)
        {
            magnitude += std::abs(sphere.center[a]);
            const double reach = std::abs(center[a] - sphere.center[a]) + half[a];
            far                = std::fma(reach, reach, far);
        }
        const double inner = sphere.radius - kMargin * (magnitude + sphere.radius);
        return inner > 0 && far <= inner * inner;
    }
    if (shape.kind == OVSPATIALQUERY_OVERLAP_AABB)
    {
        const ovspatialquery_overlap_aabb_t& box = shape.data.aabb;
        for (uint32_t a = 0; a < 3; ++a)
            magnitude += std::max(std::abs(box.min_corner[a]), std::abs(box.max_corner[a]));
        const double margin = kMargin * magnitude;
        for (uint32_t a = 0; a < 3; ++a)
            if (!(object.worldBounds.minimum[a] - box.min_corner[a] >= margin &&
                  box.max_corner[a] - object.worldBounds.maximum[a] >= margin))
                return false;
        return true;
    }
    if (shape.kind != OVSPATIALQUERY_OVERLAP_BOX)
        return false;
    const ovspatialquery_overlap_box_t& box = shape.data.box;
    double                              axes[3][3]{};
    quaternionRotation(box.rotation_xyzw, boxRotation, axes);
    for (uint32_t a = 0; a < 3; ++a)
        magnitude += std::abs(box.center[a]) + box.half_extents[a];
    const double margin = kMargin * magnitude;
    // Along each box axis, the bounds' projection around the box center stays inside its extent.
    for (uint32_t local = 0; local < 3; ++local)
    {
        double offset = 0, reach = 0;
        for (uint32_t world = 0; world < 3; ++world)
        {
            offset = std::fma(axes[world][local], center[world] - box.center[world], offset);
            reach  = std::fma(std::abs(axes[world][local]), half[world], reach);
        }
        if (!(std::abs(offset) + reach <= box.half_extents[local] - margin))
            return false;
    }
    return true;
}

// Whether a sphere misses the mesh's world bounds by the containment margin: 2^-16 of the
// magnitudes PhysX's float test works with, as overlapShapeContainsObject counts them (the
// mesh's local coordinates in world units, the placement's translation and the sphere's
// centre and radius) plus the bounds' own. Every kept triangle lies in the bounds and both
// the float and the double test round by far less than the margin, so neither reports a
// touch for a mesh this rejects. A giant sphere's AABB reaches many meshes the sphere misses.
bool overlapSphereMissesObject(const TriangleObjectGeometry& object, const ovspatialquery_overlap_sphere_t& sphere)
{
    double near = 0.0;
    for (uint32_t a = 0; a < 3; ++a)
    {
        const double gap = std::max(std::max(object.worldBounds.minimum[a] - sphere.center[a],
                                             sphere.center[a] - object.worldBounds.maximum[a]),
                                    0.0);
        // Not std::fma, a libm call in this build: the margin dwarfs a second rounding.
        near += gap * gap;
    }
    // Most candidates of a small sphere touch their bounds: keep them before the margin's work.
    if (!(near > sphere.radius * sphere.radius))
        return false;
    const ovspatialquery_aabb_t& bounds = object.mesh->localBounds();
    double                       local  = 0;
    for (uint32_t a = 0; a < 3; ++a)
        local = std::max(std::max(local, std::abs(bounds.minimum[a])), std::abs(bounds.maximum[a]));
    const double size      = local * object.uniformScale;
    double       magnitude = size + sphere.radius;
    for (uint32_t a = 0; a < 3; ++a)
        magnitude += std::abs(object.worldFromLocal[a * 4 + 3]) + std::abs(sphere.center[a]) +
                     std::max(std::abs(object.worldBounds.minimum[a]), std::abs(object.worldBounds.maximum[a]));
    // PhysX's float test squares these magnitudes and the margin in the mesh's cooked frame,
    // which divides local coordinates by a power of two within a factor of two of the largest
    // (cookedScale): there each is at most about 2 sqrt(3) magnitude / size. Beyond 2^38 or
    // below 2^-38, in either frame, a square can overflow to infinity or vanish, and the test
    // may then report a touch the geometry does not have. Such rows keep every candidate, so
    // it decides as before.
    const double cooked = magnitude / size;
    if (!(magnitude >= 0x1p-38 && magnitude <= 0x1p+38 && cooked >= 0x1p-38 && cooked <= 0x1p+38))
        return false;
    const double outer = sphere.radius + 0x1p-16 * magnitude;
    return near > outer * outer;
}
} // namespace

bool overlapTriangleObject(const TriangleObjectGeometry& object, const ovspatialquery_overlap_shape_t& geometry,
                           const double (*boxRotation)[3], InstanceHostLedger*                         ledger,
                           ovspatialquery_api_status_t* outStatus)
{
    const PhysXTriangleMesh* mesh   = nullptr;
    auto                     status = object.mesh->physxMesh(mesh);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        if (outStatus)
            *outStatus = status;
        return false;
    }
    if (geometry.kind == OVSPATIALQUERY_OVERLAP_SPHERE && overlapSphereMissesObject(object, geometry.data.sphere))
        return false;
    if (overlapShapeContainsObject(object, geometry, boxRotation))
        return true;
    // PhysX decides in float32; a BOX row with a double rotation (ADR-0026)
    // takes the double-precision triangle test below.
    bool hasOverlap = false;
    if (boxRotation == nullptr)
    {
        status = mesh->overlap(object, geometry, hasOverlap);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            return hasOverlap;
    }
    double center[3]{}, half[3]{}, localMinimum[3]{}, localMaximum[3]{};
    if (geometry.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
        for (uint32_t a = 0; a < 3; ++a)
        {
            center[a] = geometry.data.sphere.center[a];
            half[a]   = geometry.data.sphere.radius;
        }
    else if (geometry.kind == OVSPATIALQUERY_OVERLAP_AABB)
        for (uint32_t a = 0; a < 3; ++a)
        {
            center[a] = geometry.data.aabb.min_corner[a] * 0.5 + geometry.data.aabb.max_corner[a] * 0.5;
            half[a]   = geometry.data.aabb.max_corner[a] * 0.5 - geometry.data.aabb.min_corner[a] * 0.5;
        }
    else
    {
        std::copy_n(geometry.data.box.center, 3, center);
        double axes[3][3]{};
        quaternionRotation(geometry.data.box.rotation_xyzw, boxRotation, axes);
        for (uint32_t a = 0; a < 3; ++a)
            for (uint32_t b = 0; b < 3; ++b)
                half[a] += std::abs(axes[a][b]) * geometry.data.box.half_extents[b];
    }
    meshLocalBounds(object, center, half, localMinimum, localMaximum);
    return object.mesh->overlap(object.worldFromLocal, geometry, boxRotation, *mesh, localMinimum, localMaximum, ledger,
                                outStatus);
}

namespace
{
// A mesh-local box around the swept query box for rigid, uniformly scaled
// placements. Its long axis follows the sweep and its cross-section is the
// box's support, much tighter than a circumscribed capsule or a local AABB.
struct SweptBoxVolume
{
    double center[3]{};
    double axes[3][3]{};
    double half[3]{};
};

// The query box's unit axes in the mesh's frame.
void localBoxAxes(const TriangleObjectGeometry& object, const BoxGeometrySource& query, double boxAxes[3][3])
{
    for (unsigned j = 0; j < 3; ++j)
    {
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            boxAxes[j][axis] = 0;
            for (unsigned b = 0; b < 3; ++b)
                boxAxes[j][axis] =
                    std::fma(object.localFromWorldLinear[axis * 3 + b], query.axes[j][b], boxAxes[j][axis]);
        }
        const double norm = std::hypot(boxAxes[j][0], boxAxes[j][1], boxAxes[j][2]);
        for (unsigned axis = 0; axis < 3; ++axis)
            boxAxes[j][axis] /= norm;
    }
}

SweptBoxVolume sweptBoxVolume(const TriangleObjectGeometry& object, const BoxGeometrySource& query,
                              const float direction[3], double minimum, double maximum, double scale)
{
    double start[3]{}, localDirection[3]{}, boxAxes[3][3]{};
    meshLocalQuery(object, query.center.data(), direction, minimum, start, localDirection);
    const double   length = std::hypot(localDirection[0], localDirection[1], localDirection[2]) * (maximum - minimum);
    SweptBoxVolume volume{};
    localBoxAxes(object, query, boxAxes);
    // The sweep direction, then the box axis after the one most aligned with it
    // made perpendicular. That axis keeps a squared length of at least 1/3, so
    // the frame never degenerates, even for an empty sweep along box axis 0.
    const double directionLength = std::hypot(localDirection[0], localDirection[1], localDirection[2]);
    for (unsigned axis = 0; axis < 3; ++axis)
        volume.axes[0][axis] = directionLength > 0 ? localDirection[axis] / directionLength : boxAxes[0][axis];
    unsigned aligned   = 0;
    double   alignment = -1;
    for (unsigned j = 0; j < 3; ++j)
    {
        const double value = std::abs(boxAxes[j][0] * volume.axes[0][0] + boxAxes[j][1] * volume.axes[0][1] +
                                      boxAxes[j][2] * volume.axes[0][2]);
        if (value > alignment)
        {
            alignment = value;
            aligned   = j;
        }
    }
    const double* next       = boxAxes[(aligned + 1) % 3];
    const double  projection = next[0] * volume.axes[0][0] + next[1] * volume.axes[0][1] + next[2] * volume.axes[0][2];
    for (unsigned axis = 0; axis < 3; ++axis)
        volume.axes[1][axis] = next[axis] - projection * volume.axes[0][axis];
    const double norm = std::hypot(volume.axes[1][0], volume.axes[1][1], volume.axes[1][2]);
    for (unsigned axis = 0; axis < 3; ++axis)
        volume.axes[1][axis] /= norm;
    const double* e0  = volume.axes[0];
    const double* e1  = volume.axes[1];
    volume.axes[2][0] = e0[1] * e1[2] - e0[2] * e1[1];
    volume.axes[2][1] = e0[2] * e1[0] - e0[0] * e1[2];
    volume.axes[2][2] = e0[0] * e1[1] - e0[1] * e1[0];
    volume.half[0]    = length * 0.5;
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            volume.half[i] += query.halfExtents[j] / scale *
                              std::abs(boxAxes[j][0] * volume.axes[i][0] + boxAxes[j][1] * volume.axes[i][1] +
                                       boxAxes[j][2] * volume.axes[i][2]);
    for (unsigned axis = 0; axis < 3; ++axis)
        volume.center[axis] = std::fma(localDirection[axis], (maximum - minimum) * 0.5, start[axis]);
    return volume;
}

// A mesh-local box in the query box's own frame around the box from `minimum` to
// `maximum`. Unlike the box along the path, it reaches no further than the box's
// faces ahead at `maximum`.
SweptBoxVolume boxFrameVolume(const TriangleObjectGeometry& object, const BoxGeometrySource& query,
                              const float direction[3], double minimum, double maximum, double scale)
{
    double start[3]{}, localDirection[3]{};
    meshLocalQuery(object, query.center.data(), direction, minimum, start, localDirection);
    SweptBoxVolume volume{};
    localBoxAxes(object, query, volume.axes);
    for (unsigned i = 0; i < 3; ++i)
        volume.half[i] = query.halfExtents[i] / scale +
                         std::abs(volume.axes[i][0] * localDirection[0] + volume.axes[i][1] * localDirection[1] +
                                  volume.axes[i][2] * localDirection[2]) *
                             (maximum - minimum) * 0.5;
    for (unsigned axis = 0; axis < 3; ++axis)
        volume.center[axis] = std::fma(localDirection[axis], (maximum - minimum) * 0.5, start[axis]);
    return volume;
}

// Bounds on boxTriangle's entry and exit times from its first three axes, the
// query's face axes A_i: the box against a triangle's bounds in the box frame.
// They read float mesh-local vertices through the rows A_i W instead of
// transforming each vertex to world space as boxTriangle does.
//
// Both evaluations of a projection A_i (W l - c) lie within 8u S_i of its exact
// value, with u the double unit roundoff, S_i = sum_k |A_ik| (sum_j |W_kj| L_j
// + |c_k|), L the mesh's local vertex bound and c the box center relative to
// the placement. Through the box extent e_i and the axis velocity v_i, the
// times then differ by less than 24u (S_i + e_i) / |v_i|. Widening each
// interval by 1024u (S_i + e_i) keeps every entry at or below boxTriangle's
// and every exit at or above it, so a triangle is dropped only when
// boxTriangle rejects it too. Magnitudes beyond double range make the slack
// infinite, which leaves the interval unbounded.
class BoxFaceBounds
{
  public:
    BoxFaceBounds(const TriangleObjectGeometry& object, const box_sweep::Box& box, const box_sweep::BoxFaceAxes& faces)
    {
        namespace math     = box_sweep;
        const auto& bounds = object.mesh->localBounds();
        double      reach[3]{};
        for (unsigned j = 0; j < 3; ++j)
            reach[j] = std::max(std::abs(bounds.minimum[j]), std::abs(bounds.maximum[j]));
        for (unsigned i = 0; i < 3; ++i)
        {
            // The same axis, extent and velocity values boxTriangle computes.
            const math::Vec3 axis     = faces.axis[i];
            const double     velocity = faces.velocity[i];
            const double     inverse  = velocity != 0 ? 1 / velocity : 0;
            // An axis without length, or whose times overflow, bounds nothing.
            if (math::dot(axis, axis) == 0 || (velocity != 0 && !std::isfinite(inverse)))
                continue;
            double magnitude = 0;
            for (unsigned k = 0; k < 3; ++k)
            {
                double placed = std::abs(box.center[k]);
                for (unsigned j = 0; j < 3; ++j)
                    placed += std::abs(object.worldFromLocal[4 * k + j]) * reach[j];
                magnitude += std::abs(axis[k]) * placed;
            }
            // Axes the box moves along come first; the rest only bound overlap.
            Axis& a = velocity != 0 ? mAxes[mMoving++] : mAxes[2 - mStill++];
            for (unsigned j = 0; j < 3; ++j)
                a.row[j] = axis.x * object.worldFromLocal[j] + axis.y * object.worldFromLocal[4 + j] +
                           axis.z * object.worldFromLocal[8 + j];
            a.offset  = -math::dot(axis, box.center);
            a.extent  = faces.extent[i];
            a.slack   = 512 * std::numeric_limits<double>::epsilon() * (magnitude + a.extent);
            a.inverse = inverse;
        }
    }

    // Narrows [enter, exit]; false when boxTriangle is certain to reject.
    bool narrow(const float* a, const float* b, const float* c, double& enter, double& exit) const
    {
        const double p[3][3] = {{a[0], a[1], a[2]}, {b[0], b[1], b[2]}, {c[0], c[1], c[2]}};
        double       low[3], high[3];
        for (unsigned i = 0; i < 3; ++i)
        {
            const double* row = mAxes[i].row;
            const double  x   = row[0] * p[0][0] + row[1] * p[0][1] + row[2] * p[0][2] + mAxes[i].offset;
            const double  y   = row[0] * p[1][0] + row[1] * p[1][1] + row[2] * p[1][2] + mAxes[i].offset;
            const double  z   = row[0] * p[2][0] + row[1] * p[2][1] + row[2] * p[2][2] + mAxes[i].offset;
            low[i]            = std::min(x, std::min(y, z)) - mAxes[i].extent - mAxes[i].slack;
            high[i]           = std::max(x, std::max(y, z)) + mAxes[i].extent + mAxes[i].slack;
        }
        for (unsigned i = 0; i < mMoving; ++i)
        {
            // The interval's ends in time order: low/v <= high/v when v > 0.
            const double inverse = mAxes[i].inverse;
            const double lower   = inverse > 0 ? low[i] * inverse : high[i] * inverse;
            const double upper   = inverse > 0 ? high[i] * inverse : low[i] * inverse;
            if (lower > enter)
                enter = lower;
            if (upper < exit)
                exit = upper;
        }
        for (unsigned i = 3 - mStill; i < 3; ++i)
            if (low[i] > 0 || high[i] < 0)
                return false;
        return enter <= exit;
    }

  private:
    // An unused axis keeps a zero row and an unbounded extent, so it rejects nothing.
    struct Axis
    {
        double row[3]{}, offset = 0, extent = std::numeric_limits<double>::infinity(), slack = 0, inverse = 0;
    };
    Axis     mAxes[3]{};
    unsigned mMoving = 0, mStill = 0;
};
} // namespace

ovspatialquery_api_status_t sweepBoxTriangleObject(const TriangleObjectGeometry& object, const BoxGeometrySource& query,
                                                   const float direction[3], double minimum, double maximum,
                                                   bool& hasHit, ExactGeometryHit& outHit, InstanceHostLedger* ledger,
                                                   bool doubleSided)
{
    const PhysXTriangleMesh* cooked = nullptr;
    const auto               status = object.mesh->physxMesh(cooked);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    TriangleGeometryDescriptionView description{};
    object.mesh->exportDescription(description);
    const auto triangleCount = static_cast<uint32_t>(description.triangleCount);
    namespace math           = box_sweep;
    math::Box box{};
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        box.center[axis] = query.center[axis] - object.worldFromLocal[4 * axis + 3];
        box.half[axis]   = query.halfExtents[axis];
        for (unsigned i = 0; i < 3; ++i)
            box.axis[i][axis] = query.axes[i][axis];
    }
    const math::Vec3        d     = {direction[0], direction[1], direction[2]};
    const math::BoxFaceAxes faces = math::boxFaceAxes(box, d);
    const BoxFaceBounds     bounds(object, box, faces);
    const auto              vertex = [&](uint32_t triangle, unsigned corner)
    { return description.localVertices + 3ull * description.triangleVertexIndices[3ull * triangle + corner]; };
    // fma rounds r x + s once. When every coefficient r of the placement's linear part
    // is a float, as for an unrotated placement, r x of a float vertex is exact in
    // double, so r x + s rounds the same without the library call fma is here.
    bool floatRows = true;
    for (unsigned axis = 0; axis < 3; ++axis)
        for (unsigned j = 0; j < 3; ++j)
        {
            const double r = object.worldFromLocal[4 * axis + j];
            floatRows      = floatRows && std::abs(r) <= std::numeric_limits<float>::max() && double(float(r)) == r;
        }
    const auto worldTriangle = [&](uint32_t triangle, math::Vec3 points[3])
    {
        for (unsigned corner = 0; corner < 3; ++corner)
        {
            const float* local = vertex(triangle, corner);
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                const double* row = object.worldFromLocal.data() + 4 * axis;
                points[corner][axis] =
                    floatRows ? row[0] * double(local[0]) + (row[1] * double(local[1]) + row[2] * double(local[2]))
                              : std::fma(row[0], double(local[0]),
                                         std::fma(row[1], double(local[1]), row[2] * double(local[2])));
            }
        }
    };
    // Each test keeps the interval only; the contact point of the closest
    // triangle, the costly part of boxTriangle, is computed once at the end.
    hasHit                  = false;
    double         best     = maximum;
    uint32_t       selected = UINT32_MAX;
    math::Interval chosen{};
    unsigned       tieAxis = 0;
    const auto     test    = [&](uint32_t triangle)
    {
        math::Vec3 points[3]{};
        worldTriangle(triangle, points);
        // Of triangles entered at the same time the first wins, so one the box's
        // face axes already place at the best, behind the selected one, cannot win.
        // Most such ties show on one axis: the face axis that enters the selected one last.
        if (hasHit && triangle > selected)
        {
            math::Interval face{minimum, best, d};
            if (!math::boxTriangleProjection(box, faces.axis[tieAxis], faces.extent[tieAxis], faces.velocity[tieAxis],
                                             points[0], points[1], points[2], face) ||
                face.enter >= best)
                return;
        }
        math::Interval interval{};
        if (!math::boxTriangleBoxAxes(box, faces, points[0], points[1], points[2], d, minimum, best, interval) ||
            (hasHit && interval.enter == best && triangle >= selected) ||
            !math::boxTriangleTriangleAxes(box, points[0], points[1], points[2], doubleSided, d, interval) ||
            (hasHit && interval.enter == best && triangle >= selected))
            return;
        hasHit        = true;
        best          = interval.enter;
        selected      = triangle;
        chosen        = interval;
        double latest = -std::numeric_limits<double>::infinity();
        for (unsigned i = 0; i < 3; ++i)
        {
            math::Interval face{-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), d};
            math::boxTriangleProjection(box, faces.axis[i], faces.extent[i], faces.velocity[i], points[0], points[1],
                                        points[2], face);
            if (face.enter > latest)
            {
                latest  = face.enter;
                tieAxis = i;
            }
        }
    };

    // When above zero, the parameter over which the box moves its own depth along
    // the path: each part's last stretch of that length is gathered in the box's frame.
    double frameTime = 0;

    // Tests the triangles the box can touch while it moves from `from` to `to`.
    // Most gathered triangles lie beside the path or beyond the first contact.
    // Visit them by their face-axis entry bound and stop once it passes the
    // closest hit: the exact test decides every remaining triangle as before.
    // Like the candidates, a few entries stay inline and more are staged.
    const auto evaluatePart = [&](double from, double to)
    {
        // The box along the path reaches up to the box's depth beyond its faces ahead
        // at `to`, where a surface the box meets head-on lies. That box stops a box
        // depth earlier and the box's own frame covers the rest; should a triangle in
        // both leave no room, the box along the path alone fits.
        const double split     = frameTime > 0 ? std::max(from, to - frameTime) : to;
        const auto   gatherBox = [&](const SweptBoxVolume& volume, uint32_t* results, uint32_t capacity, bool& overflow)
        { return cooked->overlapTrianglesBox(volume.center, volume.axes, volume.half, results, capacity, overflow); };
        // As in sphere sweeps, a part through a large mesh skips the inline attempt.
        TriangleCandidates          candidates;
        ovspatialquery_api_status_t gathered = gatherTriangleCandidatesWith(
            [&](uint32_t* results, uint32_t capacity, bool& overflow)
            {
                uint32_t count = 0;
                overflow       = false;
                // A part without length is the box at one parameter, gathered along the path.
                if (split > from || split == to)
                    count = gatherBox(sweptBoxVolume(object, query, direction, from, split, object.uniformScale),
                                      results, capacity, overflow);
                if (!overflow && split < to)
                    count += gatherBox(boxFrameVolume(object, query, direction, split, to, object.uniformScale),
                                       results + count, capacity - count, overflow);
                if (overflow && split < to && capacity == triangleCount)
                    count = gatherBox(sweptBoxVolume(object, query, direction, from, to, object.uniformScale), results,
                                      capacity, overflow);
                return count;
            },
            triangleCount, ledger, candidates, triangleCount > kWholeGatherTriangles);
        if (gathered != OVSPATIALQUERY_API_SUCCESS)
            return gathered;
        struct Entry
        {
            double   enter;
            uint32_t triangle;
        };
        Entry                    small[128];
        std::unique_ptr<Entry[]> large;
        HostLease                entryCharge;
        Entry*                   entries = small;
        if (candidates.count > std::size(small))
        {
            if (ledger)
            {
                HostTransaction transaction(*ledger);
                if (!transaction.reserve(uint64_t(candidates.count) * sizeof(Entry)) ||
                    !transaction.commit(entryCharge))
                    return OVSPATIALQUERY_API_HOST_BUDGET;
            }
            large.reset(new Entry[candidates.count]);
            entries = large.get();
        }
        size_t count = 0;
        size_t first = 0;
        for (uint32_t candidate = 0; candidate < candidates.count; ++candidate)
        {
            const uint32_t triangle = candidates.data()[candidate];
            double         enter = minimum, exit = best;
            if (!bounds.narrow(vertex(triangle, 0), vertex(triangle, 1), vertex(triangle, 2), enter, exit))
                continue;
            if (count != 0 && enter < entries[first].enter)
                first = count;
            entries[count++] = {enter, triangle};
        }
        // The lowest bound's triangle usually holds the hit, and the rest that can still
        // beat it mostly tie with it, so they are tested in the order gathered.
        if (count != 0)
            test(entries[first].triangle);
        for (size_t entry = 0; entry < count; ++entry)
            if (entry != first && entries[entry].enter <= best)
                test(entries[entry].triangle);
        return OVSPATIALQUERY_API_SUCCESS;
    };

    // Through a large mesh the box's path is gathered in two parts, the first
    // ending where rays from the box's centre and from the corners of its leading
    // face first meet the surface: the box touches the surface by then. The rest
    // is gathered only when the closest contact lies beyond that part: every
    // triangle the box touches before then intersects the first part's volume.
    // Correctness never depends on where the first part ends.
    double firstEnd = maximum;
    if (triangleCount > kWholeGatherTriangles && maximum > minimum)
    {
        double start[3]{}, localDirection[3]{}, axes[3][3]{}, half[3]{}, velocity[3]{};
        meshLocalQuery(object, query.center.data(), direction, minimum, start, localDirection);
        localBoxAxes(object, query, axes);
        unsigned lead   = 0;
        double   centre = std::numeric_limits<double>::infinity(), depth = 0, speed = 0;
        for (unsigned i = 0; i < 3; ++i)
        {
            half[i] = query.halfExtents[i] / object.uniformScale;
            velocity[i] =
                axes[i][0] * localDirection[0] + axes[i][1] * localDirection[1] + axes[i][2] * localDirection[2];
            if (std::abs(velocity[i]) > std::abs(velocity[lead]))
                lead = i;
            // The box holds a point ahead of its centre once it has moved this much less.
            if (velocity[i] != 0)
                centre = std::min(centre, half[i] / std::abs(velocity[i]));
            depth += 2 * half[i] * std::abs(velocity[i]);
            speed += localDirection[i] * localDirection[i];
        }
        // In the box's frame a part's last stretch gathers no surface met head-on beyond
        // the box, but a surface along a face the box moves nearly parallel to over a
        // wider rectangle than the box along the path does: a box gliding over a floor.
        // So that frame is used only while the box moves at least a fifth of its speed
        // along each of its axes.
        if (std::min({std::abs(velocity[0]), std::abs(velocity[1]), std::abs(velocity[2])}) >= 0.2 * std::sqrt(speed) &&
            std::isfinite(depth / speed))
            frameTime = depth / speed;
        // Rays 1 to 4 leave the corners of the face ahead, which lies on the leading
        // axis's side the box moves to, and ray 5 the centre. A ray matters only if
        // it meets the surface before the earliest end so far, so each stops there.
        double first = std::numeric_limits<double>::infinity();
        for (unsigned ray = 1; ray <= 5 && first > 0; ++ray)
        {
            const bool fromCentre = ray == 5;
            double     corner[3]{};
            if (!fromCentre)
            {
                const unsigned across[2] = {(lead + 1) % 3, (lead + 2) % 3};
                corner[lead]             = velocity[lead] > 0 ? half[lead] : -half[lead];
                corner[across[0]]        = (ray - 1) & 1 ? half[across[0]] : -half[across[0]];
                corner[across[1]]        = (ray - 1) & 2 ? half[across[1]] : -half[across[1]];
            }
            double origin[3]{start[0], start[1], start[2]};
            for (unsigned i = 0; i < 3; ++i)
                for (unsigned a = 0; a < 3; ++a)
                    origin[a] += corner[i] * axes[i][a];
            bool               surface = false;
            PreciseTriangleHit hit{};
            const double       reach = std::min(maximum - minimum, fromCentre ? first + centre : first);
            if (cooked->raycast(origin, localDirection, reach, true, false, surface, hit, ledger) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                surface)
                first = std::min(first, fromCentre ? std::max(0.0, hit.distance - centre) : hit.distance);
        }
        // A contact at a ray's hit can round just past it; a little room keeps it in the first part.
        firstEnd = std::min(maximum, minimum + first + 0x1p-16 * centre);
    }
    double partStart = minimum, partEnd = minimum;
    do
    {
        // The second part ends at a contact the first part found beyond it.
        partEnd                                     = partStart < firstEnd ? firstEnd : best;
        const ovspatialquery_api_status_t evaluated = evaluatePart(partStart, partEnd);
        if (evaluated != OVSPATIALQUERY_API_SUCCESS)
            return evaluated;
    } while ((partStart = partEnd) < maximum && !(hasHit && best <= partEnd));
    if (hasHit)
    {
        math::Vec3 points[3]{};
        worldTriangle(selected, points);
        math::Hit hit{};
        math::boxTriangleContact(box, points[0], points[1], points[2], d, chosen, hit);
        outHit          = {};
        outHit.distance = hit.distance;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            outHit.position[axis]      = hit.position[axis] + object.worldFromLocal[4 * axis + 3];
            outHit.normal[axis]        = float(hit.normal[axis]);
            outHit.sourceCorners[axis] = description.sourceCorners[3ull * selected + axis];
        }
        outHit.uv[0]             = float(hit.u);
        outHit.uv[1]             = float(hit.v);
        outHit.sourceFace        = description.sourceFaces[selected];
        outHit.generatedTriangle = selected;
        outHit.hasTriangleSource = true;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

uint64_t triangleGeometryResidentByteCount(const TriangleObjectGeometry& object) noexcept
{
    return object.mesh == nullptr ? 0u : object.mesh->residentByteCount();
}

bool exportTriangleGeometryDescription(const TriangleObjectGeometry&    object,
                                       TriangleGeometryDescriptionView& outDescription) noexcept
{
    if (object.mesh == nullptr)
    {
        return false;
    }
    TriangleGeometryDescriptionView description{};
    object.mesh->exportDescription(description);
    description.owner          = &object;
    description.worldFromLocal = object.worldFromLocal;
    description.doubleSided    = object.doubleSided;
    outDescription             = description;
    return true;
}

} // namespace detail
} // namespace ovspatialquery
