/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_GEOMETRY_H
#define OVSPATIALQUERY_GEOMETRY_H

#include "ovspatialquery_private_api.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

class PreparedTriangleMesh;

struct TriangleMeshSource
{
    uint64_t              key = 0u;
    std::array<double, 9> ovstageMetric{1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool                  ovstageMirrored = false;
    std::vector<float>    vertices;
    std::vector<uint32_t> triangles;
    std::vector<uint64_t> sourceFaces;
    /* Three face-local authored corner ordinals per generated triangle. */
    std::vector<uint32_t> sourceCorners;
    /* A mesh that keeps this prepared mesh's triangles, source faces and corners under
     * new vertices (a deforming sheet, a flexing part): the index arrays above stay
     * empty and the preparation shares the base's topology and, for the cooked backend
     * mesh, refits the base's tree instead of cooking one. */
    std::shared_ptr<const PreparedTriangleMesh> topologyBase;
};

struct TriangleObjectSource
{
    uint64_t                meshIndex = 0u;
    std::array<double, 16u> worldFromLocal{};
    bool                    doubleSided = false;
};

struct TriangleGeometrySource
{
    std::vector<TriangleMeshSource>   meshes;
    std::vector<TriangleObjectSource> objects;
};

struct TriangleObjectGeometry
{
    std::shared_ptr<const PreparedTriangleMesh> mesh;
    std::array<double, 16u>                     worldFromLocal{};
    std::array<double, 9u>                      localFromWorldLinear{};
    ovspatialquery_aabb_t                       worldBounds{};
    double                                      uniformScale = 1.0;
    bool                                        doubleSided  = false;
};

struct PreparedTriangleDefinition
{
    uint64_t                                      key = 0u;
    std::shared_ptr<const TriangleObjectGeometry> geometry;
};

class InstanceHostLedger;

/* Reserves the query structure (ADR-0075) of every mesh among `meshes` that no admission
 * has reserved, at the admission that adds them, and claims the instance's one build
 * scratch for those not yet built; HOST_BUDGET, reserving nothing, when they do not fit.
 * A mesh keeps the structure's charge for its lifetime and the claim until it is built. */
ovspatialquery_api_status_t admitTriangleMeshStructures(InstanceHostLedger&                             ledger,
                                                        const std::vector<const PreparedTriangleMesh*>& meshes);

/* Appends the object's prepared mesh, if any, for admitTriangleMeshStructures. */
void collectTriangleMesh(const TriangleObjectGeometry* object, std::vector<const PreparedTriangleMesh*>& meshes);

/* Prepares one local triangle surface outside a cache; the caller owns its sharing. A
 * source with a topology base shares that base's index arrays and cooked tree. */
ovspatialquery_api_status_t prepareTriangleMesh(const TriangleMeshSource&                    source,
                                                std::shared_ptr<const PreparedTriangleMesh>& outMesh);

/* Hash of the source's vertices, triangles, source faces and corners, so identical
 * authored meshes can find one preparation; confirm a match with
 * preparedTriangleMeshMatches before sharing. A source with a topology base hashes
 * its vertices and the base's topology identity instead of the index arrays. */
uint64_t triangleMeshContentHash(const TriangleMeshSource& source) noexcept;

/* Whether the prepared mesh holds exactly the source's vertices, triangles, source
 * faces and corners (bitwise; a source without explicit faces or corners never matches).
 * A source with a topology base matches a mesh sharing that base's topology. */
bool preparedTriangleMeshMatches(const PreparedTriangleMesh& prepared, const TriangleMeshSource& source) noexcept;

/* The prepared surface's bounds in its local frame. */
ovspatialquery_aabb_t preparedTriangleMeshLocalBounds(const PreparedTriangleMesh& prepared) noexcept;

/* How many of the source's triangles the preparation dropped for having no area (ADR-0073). */
uint64_t preparedTriangleMeshDroppedCount(const PreparedTriangleMesh& prepared) noexcept;

/* What a successful call that prepared the mesh reports: "dropped 2 of its 10 triangles,
 * which have no area (faces 1, 4)", naming the source faces of the first few dropped
 * triangles as `faceNoun` ("triangle" or "face"); empty when nothing was dropped. */
std::string describeDroppedTriangles(const PreparedTriangleMesh& prepared, const char* faceNoun);

/* Rebinds one already prepared local triangle surface to a new pose: a finite rigid pose with
 * positive uniform scale, as both public adapters admit; any other pose is INVALID_ARGUMENT. */
ovspatialquery_api_status_t bindPreparedTriangleGeometry(const std::shared_ptr<const PreparedTriangleMesh>& prepared,
                                                         const std::array<double, 16u>&                 worldFromLocal,
                                                         bool                                           doubleSided,
                                                         std::shared_ptr<const TriangleObjectGeometry>& outObject);

/*
 * Immutable backend-neutral view of one prepared triangle-mesh query
 * instance. Array storage remains owned by TriangleObjectGeometry::mesh;
 * callers must retain the owning TriangleObjectGeometry (normally through
 * its ExactGeometry and population generation) while using this view.
 */
struct TriangleGeometryDescriptionView
{
    // Retains no ownership; the exporting exact geometry owns this placement.
    const TriangleObjectGeometry* owner                 = nullptr;
    const float*                  localVertices         = nullptr;
    uint64_t                      vertexCount           = 0u;
    const uint32_t*               triangleVertexIndices = nullptr;
    uint64_t                      triangleCount         = 0u;
    const uint64_t*               sourceFaces           = nullptr;
    const uint32_t*               sourceCorners         = nullptr;
    std::array<double, 16u>       worldFromLocal{};
    bool                          doubleSided = false;
    /* The mesh took its index arrays from an earlier preparation under new vertices
     * within the refit limit: a backend may refit that preparation's tree instead
     * of building one. */
    bool refitTree = false;
};

class TriangleGeometryCache
{
  public:
    TriangleGeometryCache();
    ~TriangleGeometryCache();

    TriangleGeometryCache(const TriangleGeometryCache&)            = delete;
    TriangleGeometryCache& operator=(const TriangleGeometryCache&) = delete;

    ovspatialquery_api_status_t adoptDefinitions(const std::vector<PreparedTriangleDefinition>& definitions);
    ovspatialquery_api_status_t bindDefinition(uint64_t key, const std::array<double, 16u>& worldFromLocal,
                                               bool                                           doubleSided,
                                               std::shared_ptr<const TriangleObjectGeometry>& outObject) const;
    // The registered definition's local bounds, from which a bound object's
    // world bounds are derived; INVALID_ARGUMENT for an unknown definition.
    ovspatialquery_api_status_t localBounds(uint64_t key, ovspatialquery_aabb_t& outBounds) const;
    // How many triangles the registered definition dropped for having no area (ADR-0073);
    // INVALID_ARGUMENT for an unknown definition.
    ovspatialquery_api_status_t droppedTriangleCount(uint64_t key, uint64_t& outCount) const;
    ovspatialquery_api_status_t destroyDefinition(uint64_t key);

  private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

struct PreciseTriangleHit
{
    double   distance          = 0.0;
    double   position[3]       = {};
    float    normal[3]         = {};
    uint64_t faceIndex         = 0u;
    uint32_t generatedTriangle = 0u;
    uint32_t sourceCorners[3]  = {};
    float    uv[2]             = {};
    bool     hasTriangleSource = false;
};

class InstanceHostLedger;
// `bothSides` counts every triangle from both sides whatever the object stores; a ray's
// back-face normal then stays the front normal unless the object is double-sided (ADR-0065).
ovspatialquery_api_status_t raycastTriangleObject(const TriangleObjectGeometry& object, const double origin[3],
                                                  const float unitDirection[3], double minimumDistance,
                                                  double maximumDistance, bool anyHit, InstanceHostLedger* ledger,
                                                  bool& hasHit, PreciseTriangleHit& outHit, bool bothSides = false);
// Every triangle crossing of the ray in [minimumDistance, maximumDistance], in
// world form and unordered; see PhysXTriangleMesh::raycastAll.
ovspatialquery_api_status_t raycastTriangleObjectAll(const TriangleObjectGeometry& object, const double origin[3],
                                                     const float unitDirection[3], double minimumDistance,
                                                     double maximumDistance, InstanceHostLedger*           ledger,
                                                     bool (*emit)(void*, const PreciseTriangleHit&), void* context,
                                                     bool bothSides = false);

// `rayNormals` evaluates a ray with a contact tolerance: a back-face contact that only
// `bothSides` admits reports its normal mirrored to the triangle's front side (ADR-0065).
bool sweepSphereTriangleObject(const TriangleObjectGeometry& object, const double center[3], double radius,
                               const float unitDirection[3], double minimumDistance, double maximumDistance,
                               PreciseTriangleHit& outHit, InstanceHostLedger* ledger = nullptr,
                               ovspatialquery_api_status_t* status = nullptr, bool raycast = false,
                               bool bothSides = false, bool rayNormals = false);

// `boxRotation`, when given, orients a BOX query in double precision (ADR-0026).
bool overlapTriangleObject(const TriangleObjectGeometry& object, const ovspatialquery_overlap_shape_t& geometry,
                           const double (*boxRotation)[3] = nullptr, InstanceHostLedger*               ledger = nullptr,
                           ovspatialquery_api_status_t* outStatus = nullptr);

uint64_t triangleGeometryResidentByteCount(const TriangleObjectGeometry& object) noexcept;

bool exportTriangleGeometryDescription(const TriangleObjectGeometry&    object,
                                       TriangleGeometryDescriptionView& outDescription) noexcept;

} // namespace detail
} // namespace ovspatialquery

#endif
