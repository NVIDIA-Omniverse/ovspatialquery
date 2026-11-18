/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_GEOMETRY_IDENTITY_H
#define OVSPATIALQUERY_GEOMETRY_IDENTITY_H

#include "ovspatialquery_typed_geometry.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct PopulationObject;

/* The first reference value the identity registry assigns. Explicit registration
 * accepts values below it and the registry assigns values from it up, so a caller's
 * explicit value and an ovstage-assigned one never contend for a reference
 * (`ovspatialquery_geometry_ref_t`). */
constexpr uint64_t kFirstAdapterGeometryValue = uint64_t{1} << 63u;

struct GeometryReference
{
    ExactGeometryFamily family = ExactGeometryFamily::sphere;
    uint64_t            value  = 0u;
};

bool operator==(const GeometryReference& first, const GeometryReference& second) noexcept;
bool operator!=(const GeometryReference& first, const GeometryReference& second) noexcept;

// Orders references by family, then value, for ordered maps and sets.
struct GeometryReferenceLess
{
    bool operator()(GeometryReference first, GeometryReference second) const noexcept
    {
        return first.family != second.family ? first.family < second.family : first.value < second.value;
    }
};

struct SphereDefinitionValue
{
    double                radius = 0.0;
    std::array<double, 9> shapeMetric{1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool                  mirrored = false;
};

struct BoxDefinitionValue
{
    double                size = 0.0;
    std::array<double, 9> shapeMetric{1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool                  mirrored = false;
};

enum class CapsuleAxis : uint8_t
{
    x,
    y,
    z,
};

struct CapsuleDefinitionValue
{
    double                radius = 0.0;
    double                height = 0.0;
    CapsuleAxis           axis   = CapsuleAxis::x;
    std::array<double, 9> shapeMetric{1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool                  mirrored = false;
};

struct ConvexDefinitionValue
{
    std::array<double, 9> shapeMetric{1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool                  mirrored = false;
    /* The hull vertex limit PhysX cooks the prim's hull under, from its authored
     * physxConvexHullCollision:hullVertexLimit (ADR-0078). */
    uint32_t              hullVertexLimit = 0u;
    std::vector<float>    points;
    std::vector<uint32_t> faceVertexCounts;
    std::vector<uint32_t> faceVertexIndices;
};

struct MeshDefinitionValue
{
    std::array<double, 9> shapeMetric{1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool                  mirrored   = false;
    bool                  leftHanded = false;
    std::vector<uint32_t> holeIndices;
    std::vector<float>    points;
    std::vector<uint32_t> faceVertexCounts;
    std::vector<uint32_t> faceVertexIndices;
    std::vector<uint64_t> sourceFaces;
    std::vector<uint32_t> sourceCorners;
    bool                  doubleSided = false;
};

using GeometryDefinitionValue = std::variant<SphereDefinitionValue, BoxDefinitionValue, CapsuleDefinitionValue,
                                             ConvexDefinitionValue, MeshDefinitionValue>;

enum class GeometryIdentityOutcome : uint8_t
{
    success,
    invalidArgument,
    resourceExhausted,
    outOfMemory,
    internalFailure,
};

class GeometryReferenceClaimDirectory
{
  public:
    class Batch
    {
      public:
        Batch();
        ~Batch();
        Batch(Batch&&) noexcept;
        Batch& operator=(Batch&&) noexcept;
        Batch(const Batch&)            = delete;
        Batch& operator=(const Batch&) = delete;

        uint64_t additionalResidentByteCount() const noexcept;
        uint64_t stagingResidentByteCount() const noexcept;

      private:
        friend class GeometryReferenceClaimDirectory;
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };

    GeometryReferenceClaimDirectory();
    ~GeometryReferenceClaimDirectory();
    GeometryReferenceClaimDirectory(const GeometryReferenceClaimDirectory&)            = delete;
    GeometryReferenceClaimDirectory& operator=(const GeometryReferenceClaimDirectory&) = delete;

    static uint64_t         fixedResidentByteCount() noexcept;
    GeometryIdentityOutcome stage(const std::vector<GeometryReference>& references, Batch& outBatch) noexcept;
    void                    commit(Batch& batch, std::shared_ptr<void> memoryOwner = {}) noexcept;
    /* Drops a committed claim; returns the resident bytes it held (zero when absent). */
    uint64_t release(GeometryReference reference) noexcept;
    size_t   claimCount() const noexcept;

  private:
    struct Impl;
    std::shared_ptr<Impl> mImpl;
};

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
struct GeometryIdentityRegistryInspection
{
    uint64_t definitionCount   = 0u;
    uint64_t nextValue         = 0u;
    uint64_t residentByteCount = 0u;
};
#endif

class GeometryIdentityRegistry
{
  public:
    class Batch
    {
      public:
        Batch();
        ~Batch();
        Batch(Batch&& other) noexcept;
        Batch& operator=(Batch&& other) noexcept;

        Batch(const Batch&)            = delete;
        Batch& operator=(const Batch&) = delete;

        const std::vector<GeometryReference>& references() const noexcept;
        uint64_t                              additionalResidentByteCount() const noexcept;
        uint64_t                              stagingResidentByteCount() const noexcept;
        /* Resident bytes of the definitions this staging dropped because every prepared
         * geometry that carried them has been released; the caller returns them to its
         * ledger whether or not the batch is committed. */
        uint64_t releasedResidentByteCount() const noexcept;
        uint64_t releaseStaging() noexcept;

      private:
        friend class GeometryIdentityRegistry;
        struct Impl;
        std::unique_ptr<Impl> mImpl;
        uint64_t              mReleasedResidentBytes = 0u;
    };

    GeometryIdentityRegistry();
    explicit GeometryIdentityRegistry(std::shared_ptr<GeometryReferenceClaimDirectory> claimDirectory);
    ~GeometryIdentityRegistry();

    static uint64_t fixedResidentByteCount() noexcept;

    GeometryIdentityRegistry(const GeometryIdentityRegistry&)            = delete;
    GeometryIdentityRegistry& operator=(const GeometryIdentityRegistry&) = delete;

    /* `owners`, when given, is parallel to `definitions`: a non-null owner is the prepared
     * geometry that carries the definition (a shared prepared mesh or convex hull), and a
     * definition whose every recorded owner has been released leaves the registry at a
     * later staging, so neither a source that keeps publishing new content (a deforming
     * mesh) nor populations created and destroyed over new content grow it without
     * bound. Definitions staged without an owner (the fixed-size primitives) stay for the
     * registry's lifetime. Reference values are never reused either way. */
    /* Definitions the registry keeps are moved out of `definitions`; a definition it
     * already holds is left in place. Callers do not read `definitions` afterwards. */
    GeometryIdentityOutcome stage(std::vector<GeometryDefinitionValue>& definitions, Batch& outBatch,
                                  const std::vector<std::shared_ptr<const void>>* owners = nullptr) noexcept;
    GeometryIdentityOutcome stage(std::vector<GeometryDefinitionValue>&& definitions, Batch& outBatch,
                                  const std::vector<std::shared_ptr<const void>>* owners = nullptr) noexcept
    {
        return stage(definitions, outBatch, owners);
    }
    void commit(Batch& batch) noexcept;
    /* Points each entry of `inOutByReference`, keyed by triangle-mesh reference value, at
     * the definition the registry holds under that reference, and leaves entries it does
     * not hold null. A definition stays while a prepared mesh carrying it is alive, so a
     * caller that holds such a mesh may read the definition after the call returns. */
    void findMeshDefinitions(std::unordered_map<uint64_t, const MeshDefinitionValue*>& inOutByReference) const noexcept;

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    bool inspect(GeometryIdentityRegistryInspection& outInspection) const noexcept;
#endif

  private:
    struct Impl;
    std::shared_ptr<Impl> mImpl;
};

ExactGeometryFamily geometryDefinitionFamily(const GeometryDefinitionValue& definition) noexcept;
uint64_t            geometryDefinitionRetainedByteCount(const GeometryDefinitionValue& definition) noexcept;
uint64_t            typedGeometryStagingResidentByteCount(
               const std::vector<PopulationObject>& objects, const std::vector<GeometryDefinitionValue>& definitions,
               const std::vector<std::shared_ptr<const PreparedConvexHull>>& preparedConvexHulls) noexcept;
// The triangle meshes and convex surfaces of an Ovstage publication's objects and hulls,
// for its admission to reserve their query structures (admitTriangleMeshStructures).
std::vector<const PreparedTriangleMesh*> typedGeometryTriangleMeshes(
    const std::vector<PopulationObject>&                          objects,
    const std::vector<std::shared_ptr<const PreparedConvexHull>>& preparedConvexHulls);

} // namespace detail
} // namespace ovspatialquery

#endif
