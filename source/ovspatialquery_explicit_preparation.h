/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_EXPLICIT_PREPARATION_H
#define OVSPATIALQUERY_EXPLICIT_PREPARATION_H

#include "ovspatialquery_geometry_identity.h"
#include "ovspatialquery_population_source.h"
#include "ovspatialquery_population_table.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct ExplicitSphereDefinition
{
    double radius = 0.0;
};

struct ExplicitBoxDefinition
{
    std::array<double, 3u> halfExtents{};
};

struct ExplicitCapsuleDefinition
{
    double radius      = 0.0;
    double halfSegment = 0.0;
};

struct ExplicitConvexDefinition
{
    std::vector<float>    vertices;
    std::vector<uint32_t> faceVertexCounts;
    std::vector<uint32_t> faceVertexIndices;
    /* The registration call's hull vertex limit and enclosure tolerance (ADR-0078). */
    ConvexHullCookingOptions options;
};

struct ExplicitTriangleMeshDefinition
{
    std::vector<float>    vertices;
    std::vector<uint32_t> triangleVertexIndices;
    bool                  doubleSided = false;
    // The caller's source face and its three corners per submitted triangle, both empty
    // when the caller supplies none (the identity). Preparation moves them into the
    // prepared mesh's topology, so a prepared definition holds them once.
    std::vector<uint64_t> sourceFaces;
    std::vector<uint32_t> sourceCorners;
    bool                  hasSourceFaces = false;
    // Per source face, null when the definition assigns no nonzero material. Marshalling
    // fills it per submitted triangle; with source faces, preparation turns it into one
    // entry per face value up to the largest.
    std::shared_ptr<const std::vector<uint64_t>> sourceFaceMaterialIds;
};

using ExplicitDefinitionValue = std::variant<ExplicitSphereDefinition, ExplicitBoxDefinition, ExplicitCapsuleDefinition,
                                             ExplicitConvexDefinition, ExplicitTriangleMeshDefinition>;

struct ExplicitDefinitionInput
{
    GeometryReference       reference{};
    ExplicitDefinitionValue definition{};
};

class PreparedConvexHull;

struct DeviceShapeDescription
{
    int32_t                                   type  = 0;  // ovspatialquery_geometry_type_t
    uint64_t                                  value = 0u; // the definition's value within its type
    std::array<double, 3u>                    parameters{};
    std::shared_ptr<const PreparedConvexHull> hull;
    std::shared_ptr<void>                     memoryOwner; // the hull definition's host charge, when it has one
    ovspatialquery_aabb_t                     localBounds{};
};

struct ExplicitInstanceInput
{
    uint64_t               objectValue = 0u;
    GeometryReference      geometry{};
    std::array<double, 3u> translation{};
    std::array<double, 4u> rotationXyzw{0.0, 0.0, 0.0, 1.0};
    double                 uniformScale      = 1.0;
    ParticipationBits      participationBits = kAllParticipationBits;
    uint64_t               materialId        = 0u;
};

// Fills the binding and, unless the rows carry them, the scale and the
// participation bits of poses-only rows from the object the base holds under
// each ID; INVALID_ARGUMENT when a row names an object the base does not have,
// NOT_SUPPORTED for a device-prepared base, like prepareUpdate.
ovspatialquery_api_status_t completePoseRows(const GenerationLease& base, bool rowsCarryScale, bool rowsCarryBits,
                                             std::vector<ExplicitInstanceInput>& rows) noexcept;

// Builds the host objects of a readable device version (ON_DEVICE_HOST_READABLE)
// from each live slot's identity, binding and pose: the explicit registry for an
// explicit population, the adapter's own definitions for an Ovstage one.
class DeviceObjectPreparer
{
  public:
    virtual ~DeviceObjectPreparer()                                                                     = default;
    virtual ovspatialquery_api_status_t prepareObjects(const std::vector<ExplicitInstanceInput>& instances,
                                                       std::vector<PopulationObject>&            outObjects) const = 0;
};

class ExplicitPreparedRegistration
{
  public:
    ExplicitPreparedRegistration();
    ~ExplicitPreparedRegistration();
    ExplicitPreparedRegistration(ExplicitPreparedRegistration&&) noexcept;
    ExplicitPreparedRegistration& operator=(ExplicitPreparedRegistration&&) noexcept;

    ExplicitPreparedRegistration(const ExplicitPreparedRegistration&)            = delete;
    ExplicitPreparedRegistration& operator=(const ExplicitPreparedRegistration&) = delete;

    bool                         initialized() const noexcept;
    size_t                       definitionCount() const noexcept;
    const std::vector<uint64_t>& definitionResidentByteCounts() const noexcept;
    const std::vector<uint64_t>& definitionRetainedByteCounts() const noexcept;
    uint64_t                     claimResidentByteCount() const noexcept;
    uint64_t                     transientResidentByteCount() const noexcept;
    // The triangle meshes and convex surfaces the definitions prepared, for their
    // admission to reserve their query structures (admitTriangleMeshStructures).
    std::vector<const PreparedTriangleMesh*> triangleMeshes() const;

  private:
    friend class ExplicitPreparationRegistry;
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

class ExplicitPreparedUpdate
{
  public:
    ExplicitPreparedUpdate();
    ~ExplicitPreparedUpdate();
    ExplicitPreparedUpdate(ExplicitPreparedUpdate&&) noexcept;
    ExplicitPreparedUpdate& operator=(ExplicitPreparedUpdate&&) noexcept;

    ExplicitPreparedUpdate(const ExplicitPreparedUpdate&)            = delete;
    ExplicitPreparedUpdate& operator=(const ExplicitPreparedUpdate&) = delete;

    const ObjectTable& objects() const noexcept;
    // exactGeometryResidentByteCount(objects()) when the update derived it from its base's charge.
    std::optional<uint64_t> exactGeometryCharge() const noexcept;
    ObjectTable             releaseObjects() noexcept;
    // What the update changed against its base, by position (ADR-0019).
    std::optional<VersionChangeRecord> releaseChanges() noexcept;

  private:
    friend class ExplicitPreparationRegistry;
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

class ExplicitPreparationRegistry final : public DeviceObjectPreparer
{
  public:
    // backendRuntime is non-owning, may be used concurrently, and must outlive
    // the registry. InstanceState member order enforces that lifetime.
    ExplicitPreparationRegistry(ovspatialquery_backend_runtime_t*                backendRuntime,
                                uint64_t                                         maximumPreparedConvexResidentBytes,
                                std::shared_ptr<GeometryReferenceClaimDirectory> claimDirectory = {});
    ~ExplicitPreparationRegistry();

    ExplicitPreparationRegistry(const ExplicitPreparationRegistry&)            = delete;
    ExplicitPreparationRegistry& operator=(const ExplicitPreparationRegistry&) = delete;

    static uint64_t fixedResidentByteCount() noexcept;

    // A refused triangle mesh whose triangle names a missing vertex, or that has no
    // triangle with area, also gets `outDiagnostic` naming its definition and why. A
    // successful one that dropped triangles without area gets `outDropReport` naming
    // the first such definitions and their triangles (ADR-0073). `hostBytesInUse`, the
    // instance's current host charge, bounds the face-material tables built before
    // admission to what is left of the budget (ADR-0080).
    ovspatialquery_api_status_t prepareRegistration(const std::vector<ExplicitDefinitionInput>& definitions,
                                                    ExplicitPreparedRegistration&               outRegistration,
                                                    std::string*                                outDiagnostic = nullptr,
                                                    std::string*                                outDropReport = nullptr,
                                                    uint64_t hostBytesInUse = 0u) const noexcept;
    ovspatialquery_api_status_t commitRegistration(const ExplicitPreparedRegistration& registration,
                                                   std::vector<std::shared_ptr<void>>  definitionMemoryOwners = {},
                                                   std::shared_ptr<void>               claimMemoryOwner = {}) noexcept;
    ovspatialquery_api_status_t retainDefinitions(const std::vector<GeometryReference>&     references,
                                                  std::vector<std::shared_ptr<const void>>& outOwners) const noexcept;
    // What a device-prepared population keeps per referenced definition: the
    // public geometry type, the authored sphere radius, box half extents or
    // capsule radius and half segment, the prepared convex hull, or a mesh's
    // local bounds; before a publication supplies its scale and world pose.
    // NOT_SUPPORTED for a hull without a prepared surface.
    // `outOwners` keeps each definition alive, as retainDefinitions does;
    // INVALID_ARGUMENT for a reference to no definition comes before either.
    ovspatialquery_api_status_t describeDeviceShapes(
        const std::vector<GeometryReference>& references, std::vector<DeviceShapeDescription>& outShapes,
        std::vector<std::shared_ptr<const void>>& outOwners) const noexcept;
    ovspatialquery_api_status_t destroyDefinition(GeometryReference reference) noexcept;
    // How many triangles a committed triangle-mesh definition dropped for having no
    // area (ADR-0073); INVALID_ARGUMENT for a value naming no such definition.
    ovspatialquery_api_status_t droppedTriangleCount(uint64_t value, uint64_t& outCount) const noexcept;

    // The objects of `instances` in ascending ID order, as prepareSet makes them.
    ovspatialquery_api_status_t prepareObjects(const std::vector<ExplicitInstanceInput>& instances,
                                               std::vector<PopulationObject>&            outObjects) const override;
    ovspatialquery_api_status_t prepareSet(const std::vector<ExplicitInstanceInput>& instances,
                                           ExplicitPreparedUpdate&                   outUpdate) const noexcept;
    // NOT_SUPPORTED for a device-prepared base: a host-prepared update cannot
    // build on it, even when it became the head after the caller checked.
    ovspatialquery_api_status_t prepareUpdate(const GenerationLease&                    base,
                                              ovspatialquery_ordinal_t                  requestedBaseOrdinal,
                                              const std::vector<ExplicitInstanceInput>& changedInstances,
                                              const std::vector<uint64_t>&              removedObjectValues,
                                              ExplicitPreparedUpdate&                   outUpdate) const noexcept;

    // A lifetime-only binding: explicit populations have no snapshot acquirer.
    PopulationSourceBinding sourceBinding() const noexcept;
    size_t                  definitionCount() const noexcept;

  private:
    struct State;
    std::shared_ptr<State> mState;
};

} // namespace detail
} // namespace ovspatialquery

#endif
