/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TYPED_GEOMETRY_H
#define OVSPATIALQUERY_TYPED_GEOMETRY_H

#include "ovspatialquery_geometry.h"
#include "ovspatialquery_physx_backend.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace physx
{
class PxConvexMesh;
}

struct ovspatialquery_backend_runtime_t;

namespace ovspatialquery
{
namespace detail
{

class ExactGeometry;

// Public tags are contiguous; keep their range and the zero-based private
// family conversion independent of the endpoint family names at call sites.
constexpr auto kGeometryTypeFirst        = OVSPATIALQUERY_GEOMETRY_USER;
constexpr auto kGeometryTypeFirstBuiltin = OVSPATIALQUERY_GEOMETRY_SPHERE;
constexpr auto kGeometryTypeLast         = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;

enum class ExactGeometryFamily : uint8_t
{
    sphere,
    box,
    capsule,
    convex,
    triangleMesh,
    user,
};

struct SphereGeometrySource
{
    std::array<double, 3u> center{};
    double                 radius = 0.0;
};

struct BoxGeometrySource
{
    std::array<double, 3u>                 center{};
    std::array<std::array<double, 3u>, 3u> axes{};
    std::array<double, 3u>                 halfExtents{};
};

/* One shared conversion keeps all exact-family query paths numerically equal.
   `rotation`, when given, is the box's normalized rotation[world][local] in
   double precision and replaces its float32 quaternion (ADR-0026). */
BoxGeometrySource makeQueryBoxGeometry(const ovspatialquery_overlap_box_t& source,
                                       const double (*rotation)[3] = nullptr);
BoxGeometrySource makeQueryBoxGeometry(const ovspatialquery_overlap_aabb_t& source);
/* A BOX or AABB overlap shape as a box (see above); false for any other kind. */
bool queryBoxGeometry(const ovspatialquery_overlap_shape_t& shape, const double (*rotation)[3],
                      BoxGeometrySource&                    outBox);

struct CapsuleGeometrySource
{
    std::array<double, 3u> endpointA{};
    std::array<double, 3u> endpointB{};
    double                 radius = 0.0;
};

/* Prepared world-space query instances, not authored definitions; each has its source's fields. */
using SphereExactGeometryDescription  = SphereGeometrySource;
using BoxExactGeometryDescription     = BoxGeometrySource;
using CapsuleExactGeometryDescription = CapsuleGeometrySource;

struct ConvexPlaneDescription
{
    std::array<double, 3u> normal{};
    double                 offset = 0.0;
};

struct ConvexEdgeDescription
{
    uint32_t first  = 0u;
    uint32_t second = 0u;
};

/*
 * Convex arrays use local coordinates when worldFromLocal is non-null, otherwise
 * world coordinates. A local placement is rigid with positive uniform scale. Planes
 * and true polytope edges are included because a triangulated surface alone
 * cannot reproduce the current convex ray/box-overlap semantics exactly.
 */
struct ConvexExactGeometryDescription
{
    const double*                 vertices              = nullptr;
    const double*                 worldFromLocal        = nullptr;
    double                        uniformScale          = 1.0;
    uint64_t                      vertexCount           = 0u;
    const uint32_t*               triangleVertexIndices = nullptr;
    uint64_t                      triangleCount         = 0u;
    const ConvexPlaneDescription* planes                = nullptr;
    uint64_t                      planeCount            = 0u;
    const ConvexEdgeDescription*  edges                 = nullptr;
    uint64_t                      edgeCount             = 0u;
};

using TriangleMeshExactGeometryDescription = TriangleGeometryDescriptionView;

/*
 * Description array pointers remain valid while the exporting ExactGeometry
 * is alive. Population generations own that geometry for their full retained
 * lifetime. GeometryReference identity intentionally remains separate from
 * this per-object query-instance description.
 */
using ExactGeometryDescription =
    std::variant<SphereExactGeometryDescription, BoxExactGeometryDescription, CapsuleExactGeometryDescription,
                 ConvexExactGeometryDescription, TriangleMeshExactGeometryDescription>;

struct ConvexGeometrySource
{
    std::vector<float>      vertices;
    std::vector<uint32_t>   faceVertexCounts;
    std::vector<uint32_t>   faceVertexIndices;
    std::array<double, 16u> worldFromLocal{};
};

/*
 * Immutable backend-prepared hull data. Authored geometry identity stays
 * with GeometryIdentityRegistry.
 */
class PreparedConvexHull
{
  public:
    PreparedConvexHull(std::vector<double> vertices, std::vector<uint32_t> faceVertexCounts,
                       std::vector<uint32_t> faceVertexIndices, std::shared_ptr<void> residentCharge = {},
                       std::shared_ptr<physx::PxConvexMesh>          cookedMesh      = {},
                       std::shared_ptr<const TriangleObjectGeometry> preparedSurface = {},
                       std::shared_ptr<const ExactGeometry> localGeometry = {}, CookedFrame cookedFrame = {});

    const std::vector<double>&                  vertices() const noexcept;
    const std::vector<uint32_t>&                faceVertexCounts() const noexcept;
    const std::vector<uint32_t>&                faceVertexIndices() const noexcept;
    uint64_t                                    vertexCount() const noexcept;
    uint64_t                                    residentByteCount() const noexcept;
    const std::shared_ptr<physx::PxConvexMesh>& cookedMesh() const noexcept { return mCookedMesh; }
    // Where the cooked mesh sits in the hull's coordinates.
    const CookedFrame&                                   cookedFrame() const noexcept { return mCookedFrame; }
    const std::shared_ptr<const TriangleObjectGeometry>& preparedSurface() const noexcept { return mPreparedSurface; }

    const std::shared_ptr<const ExactGeometry>& localGeometry() const noexcept { return mLocalGeometry; }

  private:
    /* Declared first so retained storage dies before its ledger charge. */
    const std::shared_ptr<void>                         mResidentCharge;
    const std::vector<double>                           mVertices;
    const std::vector<uint32_t>                         mFaceVertexCounts;
    const std::vector<uint32_t>                         mFaceVertexIndices;
    const std::shared_ptr<physx::PxConvexMesh>          mCookedMesh;
    const std::shared_ptr<const TriangleObjectGeometry> mPreparedSurface;
    const std::shared_ptr<const ExactGeometry>          mLocalGeometry;
    const CookedFrame                                   mCookedFrame;
};

enum class ConvexHullPreparationOutcome : uint8_t
{
    success,
    invalidArgument,
    resourceExhausted,
    outOfMemory,
    internalFailure,
    /* Only from createPreparedConvexHull: hull validation rejected the supplied topology. */
    topologyRejected,
    /* The points' hull has more vertices, or more faces, than kMaximumConvexHullVertices or
     * kMaximumConvexHullFaces (ADR-0074); public status INVALID_ARGUMENT. */
    beyondVertexLimit,
    beyondFaceLimit,
    /* The backend has no hull of the points that has only points as vertices and encloses
     * every point, as for nearly coplanar points (ADR-0076); public status INVALID_ARGUMENT. */
    noEnclosingHull,
};

/* The most vertices and faces a convex hull may have; a definition beyond either is refused
 * (ADR-0074). PhysX's hulls stop at 254 polygons under its largest polygon limit, 255. */
constexpr uint32_t kMaximumConvexHullVertices = 255u;
constexpr uint32_t kMaximumConvexHullFaces    = 254u;

/* How a convex registration asks for its hull (ADR-0078). With `vertexLimit` zero the hull is
 * the points' own: its vertices are points and no point lies further outside a face plane than
 * `enclosureTolerance` times the points' largest half extent. With a `vertexLimit` from
 * kMinimumHullVertexLimit to kMaximumHullVertexLimit it is the hull PhysX cooks for a
 * convexHull collider under that limit, as Omni Physics does: at most that many vertices, which
 * need not be points, and points may lie outside it, so no enclosure test applies. */
constexpr double   kDefaultEnclosureTolerance = 0x1.0p-20;
constexpr double   kMaximumEnclosureTolerance = 0x1.0p-16;
constexpr uint32_t kMinimumHullVertexLimit    = 8u;
constexpr uint32_t kMaximumHullVertexLimit    = 64u;
/* PhysxSchema's default for physxConvexHullCollision:hullVertexLimit. */
constexpr uint32_t kDefaultHullVertexLimit = 64u;

struct ConvexHullCookingOptions
{
    uint32_t vertexLimit        = 0u;
    double   enclosureTolerance = kDefaultEnclosureTolerance;
};

/* SUCCESS, INVALID_ARGUMENT (also beyond a hull limit), HOST_BUDGET (resourceExhausted),
 * OUT_OF_MEMORY, else INTERNAL. */
ovspatialquery_api_status_t convexHullPreparationStatus(ConvexHullPreparationOutcome outcome) noexcept;

using ConvexHullResidentChargeAcquirer = ConvexHullPreparationOutcome (*)(void* state, uint64_t bytes,
                                                                          std::shared_ptr<void>& outOwner) noexcept;

struct ConvexHullPreparationBudget
{
    void*                            state   = nullptr;
    ConvexHullResidentChargeAcquirer acquire = nullptr;
};

/*
 * Private backend preparation. The output is committed only on success and
 * contains no backend-native type.
 */
ConvexHullPreparationOutcome prepareConvexHull(ovspatialquery_backend_runtime_t* backendRuntime,
                                               const std::vector<float>& authoredPoints, uint64_t maximumResidentBytes,
                                               std::shared_ptr<const PreparedConvexHull>& outHull,
                                               const ConvexHullPreparationBudget*         budget             = nullptr,
                                               const ConvexHullCookingOptions&            options            = {},
                                               double*                                    outFarthestOutside = nullptr);

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
void failNextConvexHullPreparationForTest(ovspatialquery_backend_runtime_t* backendRuntime,
                                          ConvexHullPreparationOutcome      failure);
void failNextPlacedConvexForTest();
#endif

/* Policy-neutral validation result; source/API adapters choose public status. */
enum class ConvexConstructionOutcome : uint8_t
{
    success,
    malformed,
    nonConvex,
    outOfMemory,
    internalFailure,
};

enum class PrimitiveScaleRule : uint8_t
{
    uniform,
    independentOrthogonal,
};

struct QualifiedPrimitiveTransform
{
    std::array<double, 3u>                 center{};
    std::array<std::array<double, 3u>, 3u> axes{};
    std::array<double, 3u>                 scales{};
};

struct ExactGeometryHit
{
    double   distance          = 0.0;
    double   position[3]       = {};
    float    normal[3]         = {};
    uint32_t generatedTriangle = 0u;
    uint64_t sourceFace        = 0u;
    /* Face-local authored corner ordinals, valid with hasTriangleSource. */
    uint32_t sourceCorners[3]  = {};
    float    uv[2]             = {};
    bool     hasTriangleSource = false;
};

class ExactGeometryResidentAccumulator
{
  public:
    ExactGeometryResidentAccumulator();
    ~ExactGeometryResidentAccumulator();

    ExactGeometryResidentAccumulator(const ExactGeometryResidentAccumulator&)            = delete;
    ExactGeometryResidentAccumulator& operator=(const ExactGeometryResidentAccumulator&) = delete;

    bool     add(const void* owner, uint64_t bytes) noexcept;
    uint64_t residentByteCount() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

class InstanceHostLedger;

class ExactGeometry
{
  public:
    virtual ~ExactGeometry() = default;

    virtual ExactGeometryFamily          family() const noexcept                                                    = 0;
    virtual bool                         exportDescription(ExactGeometryDescription& outDescription) const noexcept = 0;
    virtual const ovspatialquery_aabb_t& worldBounds() const noexcept                                               = 0;
    // Initialize optional outStatus to SUCCESS before each query.
    // Allocation-free geometry leaves it unchanged; resource paths report failures.
    // `bothSides` counts every triangle of a mesh from both sides for this one query; a
    // ray's back-face normal stays the front normal unless the mesh is stored double-sided
    // (ADR-0065). Solid geometry has no sides and ignores it. `rayNormals` marks a sweep
    // that is a ray with a contact tolerance, whose back-face normals follow that ray rule.
    virtual bool raycast(const double origin[3], const float unitDirection[3], double minimumDistance,
                         double maximumDistance, bool anyHit, ExactGeometryHit& outHit,
                         InstanceHostLedger* ledger = nullptr, ovspatialquery_api_status_t* outStatus = nullptr,
                         bool bothSides = false) const = 0;
    // Every surface crossing of the ray, unordered: a triangle mesh reports each
    // triangle it crosses, any other geometry its one raycast hit. `emit` returns
    // false when it cannot store a hit. Returns the evaluation status.
    using HitSink = bool (*)(void* context, const ExactGeometryHit& hit);
    virtual ovspatialquery_api_status_t raycastAll(const double origin[3], const float unitDirection[3],
                                                   double minimumDistance, double maximumDistance, HitSink emit,
                                                   void* context, InstanceHostLedger* ledger,
                                                   bool bothSides = false) const
    {
        ExactGeometryHit            hit{};
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        if (!raycast(origin, unitDirection, minimumDistance, maximumDistance, false, hit, ledger, &status, bothSides))
            return status;
        return emit(context, hit) ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    virtual bool sweepSphere(const double center[3], double radius, const float unitDirection[3],
                             double minimumDistance, double maximumDistance, ExactGeometryHit& outHit,
                             InstanceHostLedger* ledger = nullptr, ovspatialquery_api_status_t* outStatus = nullptr,
                             bool bothSides = false, bool rayNormals = false) const = 0;
    // `boxRotation`, when given, orients a BOX query in double precision in
    // place of its float32 quaternion; see makeQueryBoxGeometry.
    virtual bool     overlaps(const ovspatialquery_overlap_shape_t& geometry, const double (*boxRotation)[3] = nullptr,
                              InstanceHostLedger*                   ledger    = nullptr,
                              ovspatialquery_api_status_t*          outStatus = nullptr) const             = 0;
    virtual uint64_t residentByteCount() const noexcept                                           = 0;
    virtual void     accumulateResidentAllocations(ExactGeometryResidentAccumulator& accumulator,
                                                   bool includeSharedTriangleMesh) const noexcept = 0;
    // The triangle surface whose query structure the first exact reader builds, if any.
    virtual const TriangleObjectGeometry* triangleSurface() const noexcept { return nullptr; }
};

ovspatialquery_api_status_t exportExactGeometryDescription(const std::shared_ptr<const ExactGeometry>& geometry,
                                                           ExactGeometryDescription& outDescription) noexcept;

ovspatialquery_api_status_t createSphereExactGeometry(const SphereGeometrySource&           source,
                                                      std::shared_ptr<const ExactGeometry>& outGeometry);

ovspatialquery_api_status_t createBoxExactGeometry(const BoxGeometrySource&              source,
                                                   std::shared_ptr<const ExactGeometry>& outGeometry);

ovspatialquery_api_status_t createCapsuleExactGeometry(const CapsuleGeometrySource&          source,
                                                       std::shared_ptr<const ExactGeometry>& outGeometry);

ConvexConstructionOutcome createConvexExactGeometry(const ConvexGeometrySource&           source,
                                                    std::shared_ptr<const ExactGeometry>& outGeometry);

ConvexConstructionOutcome createConvexExactGeometry(const std::shared_ptr<const PreparedConvexHull>& preparedHull,
                                                    const std::array<double, 16u>&                   worldFromLocal,
                                                    std::shared_ptr<const ExactGeometry>&            outGeometry);

/* Authored convex points once validated: finite, duplicates removed, at least four not
 * coplanar; `normalized` holds them centred on their bounds, the largest offset scaled to one. */
struct ConvexPointSet
{
    std::vector<std::array<double, 3u>> authored;
    std::vector<std::array<double, 3u>> normalized;
    std::array<double, 3u>              minimum{};
    std::array<double, 3u>              maximum{};
};

/* Builds a backend's hull of the points in authored coordinates, with the cooked mesh it
 * came from when there is one and the power of two its coordinates are the hull's times.
 * Returns success, beyondVertexLimit or beyondFaceLimit when the points' hull exceeds a
 * limit, or topologyRejected when the backend has no hull for them. */
using ConvexHullBuilder = std::function<ConvexHullPreparationOutcome(
    const ConvexPointSet& points, uint32_t vertexLimit, std::vector<double>& vertices,
    std::vector<uint32_t>& faceVertexCounts, std::vector<uint32_t>& faceVertexIndices,
    std::shared_ptr<physx::PxConvexMesh>& cookedMesh, CookedFrame& cookedFrame)>;

/*
 * Validates the points (invalidArgument unless finite with at least four not coplanar),
 * returns the builder's limit outcome when their hull exceeds a limit, and keeps the
 * builder's hull when it validates and, without a vertex limit, encloses every point within
 * the options' tolerance. A hull that leaves a point outside returns beyondVertexLimit when
 * the builder stopped at kMaximumConvexHullVertices vertices (ADR-0074), otherwise
 * noEnclosingHull (ADR-0076), and `outFarthestOutside` then holds how far outside the farthest
 * point lies, as a fraction of the points' largest half extent (ADR-0078). With a vertex limit
 * the builder's hull is kept as PhysX cooked it; one the hull check rejects even split returns
 * noEnclosingHull.
 */
ConvexHullPreparationOutcome prepareConvexHullFrom(
    const std::vector<float>& authoredPoints, uint64_t maximumResidentBytes,
    std::shared_ptr<const PreparedConvexHull>& outHull, const ConvexHullPreparationBudget* budget,
    const ConvexHullBuilder& build, const ConvexHullCookingOptions& options = {}, double* outFarthestOutside = nullptr);

/*
 * Common validated commit for backend-native hull builders. Returns
 * topologyRejected when the supplied vertices and faces fail hull validation.
 */
ConvexHullPreparationOutcome createPreparedConvexHull(
    std::vector<double> vertices, std::vector<uint32_t> faceVertexCounts, std::vector<uint32_t> faceVertexIndices,
    std::shared_ptr<const PreparedConvexHull>& outHull, const ConvexHullPreparationBudget* budget = nullptr,
    std::shared_ptr<physx::PxConvexMesh> cookedMesh = {}, CookedFrame cookedFrame = {});

uint64_t saturatingResidentByteAdd(uint64_t total, uint64_t elementCount, uint64_t elementSize) noexcept;

/* Shared fail-closed transform gate for authored analytic primitives. */
ovspatialquery_api_status_t qualifyPrimitiveTransform(const std::array<double, 16u>& worldFromLocal,
                                                      PrimitiveScaleRule             scaleRule,
                                                      QualifiedPrimitiveTransform&   outTransform);

ovspatialquery_api_status_t createTriangleMeshExactGeometry(
    std::shared_ptr<const TriangleObjectGeometry> triangleGeometry, std::shared_ptr<const ExactGeometry>& outGeometry);

} // namespace detail
} // namespace ovspatialquery

#endif
