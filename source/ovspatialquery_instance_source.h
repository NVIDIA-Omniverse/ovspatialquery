/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_INSTANCE_SOURCE_H
#define OVSPATIALQUERY_INSTANCE_SOURCE_H

#include "ovspatialquery_private_api.h"

#include "ovspatialquery_internal.h"
#include "ovspatialquery_participation.h"
#include "ovspatialquery_population_source.h"
#include "ovspatialquery_shared_worker_loan.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct SourceDeviceBudget
{
    DLDevice device{kDLCPU, 0};
    uint64_t byteBudget = 0u;
};

struct SourceInstanceConfig
{
    uint32_t                  operationCapacity    = 0u;
    uint32_t                  deviceResultCapacity = 0u;
    uint32_t                  maxCpuConcurrency    = 0u;
    uint64_t                  hostByteBudget       = 0u;
    const SourceDeviceBudget* deviceBudgets        = nullptr;
    uint32_t                  deviceBudgetCount    = 0u;
};

ovspatialquery_api_status_t createInstance(const SourceInstanceConfig&       config,
                                           ovspatialquery_instance_handle_t* outInstance);

struct SourceExclusions
{
    std::vector<uint64_t> objects;
    std::vector<int32_t>  reasons;
};

/* Mesh objects that dropped triangles without area (ADR-0073), sorted by object ID,
 * with how many each dropped. */
struct SourceDroppedTriangles
{
    std::vector<uint64_t> objects;
    std::vector<uint64_t> counts;
};

struct CopiedSourceSnapshot
{
    SourceExclusions              exclusions;
    std::vector<PopulationObject> objects;
    /* Typed acquisition: one definition per object that still needs a geometry
     * reference, in object order, skipping the objects below that reuse one and those
     * that share an earlier object's definition through definitionSources. */
    std::vector<GeometryDefinitionValue> geometryDefinitions;
    /* Typed acquisition: parallel to objects when non-empty, and then the index into
     * geometryDefinitions of each object's definition, so the instance proxies of one
     * prototype prim stage one definition (ADR-0050); SIZE_MAX for an object that reuses
     * a reference. Empty means geometryDefinitions is in object order. */
    std::vector<size_t> definitionSources;
    /* Typed acquisition: the distinct prototype prims whose definitions instance proxies
     * among the objects take (ADR-0050), as ovx_primpath_t values. */
    std::vector<uint64_t> prototypePaths;
    /* Typed acquisition: how many matched prims were PointInstancer prototype prims or their
     * copies, dropped before every rule, so neither objects nor exclusions. */
    uint64_t droppedPointPrototypes = 0u;
    /* Typed acquisition: parallel to geometryDefinitions when non-empty; a non-null
     * entry is the prepared geometry carrying that definition, whose release lets the
     * identity registry drop the definition (see GeometryIdentityRegistry::stage). */
    std::vector<std::shared_ptr<const void>> definitionOwners;
    /* Typed acquisition: parallel to objects when non-empty; a non-zero value is
     * the reference of geometry the source reported unchanged since the base
     * generation, carried over instead of staged again. */
    std::vector<GeometryReference> reusedReferences;
    /* Temporary private preparation owners, charged only while submitting. */
    std::vector<std::shared_ptr<const PreparedConvexHull>> preparedConvexHulls;
    struct ViewMembership
    {
        ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
        std::vector<uint64_t>        selectedIds;
    };
    std::vector<ViewMembership> viewMemberships;
    uint64_t                    indexedCount  = 0u;
    uint64_t                    excludedCount = 0u;
    /* Set by an acquirer that refuses the source: what it refused and why. */
    std::string refusal;
    /* Set by an acquirer that dropped triangles without area from meshes it read
     * (ADR-0073): the diagnostic naming the first few, and every one with its count. */
    std::string            droppedTriangles;
    SourceDroppedTriangles droppedTriangleCounts;
    /* Input, set before acquiring: whether the acquirer resolves hit materials
     * (ADR-0066). A device-prepared publication reports none, so it skips them. */
    bool resolveMaterials = true;
    /* Host charges for data the acquirer left in `objects` (an Ovstage mesh's face material
     * table), held until the publication's typed staging reservation, which counts that
     * data, replaces them; so the data is never resident uncharged in between. */
    std::vector<std::shared_ptr<void>> retainedCharges;
};

struct SourceGeometryPreparationContext
{
    ovspatialquery_backend_runtime_t* backendRuntime       = nullptr;
    uint64_t                          maximumResidentBytes = 0u;
    ConvexHullPreparationBudget       convexHullBudget{};
    SharedWorkerLoan                  workers{};
};

ovspatialquery_api_status_t getSourceGeometryPreparationContext(ovspatialquery_instance_handle_t  instance,
                                                                SourceGeometryPreparationContext& outPreparation);

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
ovspatialquery_api_status_t failNextTypedGeometryAggregateStagingForTest(ovspatialquery_instance_handle_t instance);
#endif

struct SourceViewRequest
{
    ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
    std::shared_ptr<void>        sourceState;
};

struct PreparedSourceViewMembership
{
    ovspatialquery_view_handle_t                 view = OVSPATIALQUERY_INVALID_VIEW;
    std::shared_ptr<const std::vector<uint64_t>> selectedIds;
};

ParticipationOutcome installParticipationTable(ovspatialquery_instance_handle_t   instance,
                                               ovspatialquery_population_handle_t population,
                                               const ParticipationTableDesc*      table);

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
ovspatialquery_api_status_t inspectGeometryIdentityRegistry(ovspatialquery_instance_handle_t    instance,
                                                            GeometryIdentityRegistryInspection& outInspection);
#endif

// `pendingPublicationCapacity` below one means one (ADR-0021).
ovspatialquery_api_status_t createEmptyPopulationForSource(ovspatialquery_instance_handle_t instance,
                                                           PopulationSourceBinding          source,
                                                           uint32_t retainedOrdinalCapacity, uint64_t sourceStateCharge,
                                                           ovspatialquery_population_handle_t& outPopulation,
                                                           uint32_t pendingPublicationCapacity = 1u);

ovspatialquery_api_status_t createViewForSource(ovspatialquery_instance_handle_t   instance,
                                                ovspatialquery_source_kind_t       sourceKind,
                                                ovspatialquery_population_handle_t population,
                                                std::shared_ptr<void> sourceState, uint64_t sourceStateCharge,
                                                ovspatialquery_view_handle_t& outView);

ovspatialquery_api_status_t submitPopulationSnapshotForSource(
    ovspatialquery_instance_handle_t instance, ovspatialquery_source_kind_t sourceKind,
    ovspatialquery_population_handle_t population, const ovspatialquery_ordinal_t* exactBaseOrdinal,
    ovspatialquery_ordinal_t targetOrdinal, ovspatialquery_op_id_t& outOperation, std::string* outDiagnostic = nullptr);

/* Acquires a source population's complete snapshot at `targetOrdinal` as a host
 * publication does, with every object's geometry reference committed to the instance's
 * identity registry and the installed participation table applied, for a publication
 * that keeps identities and definitions but reads its poses on the device (ADR-0047).
 * `inspect` sees the snapshot before staging moves its definitions into the registry;
 * every object has its definition, in order. NOT_SUPPORTED while the population has a view. */
ovspatialquery_api_status_t acquireSourceObjects(
    ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
    ovspatialquery_ordinal_t targetOrdinal, CopiedSourceSnapshot& outSnapshot,
    std::shared_ptr<const ParticipationTable>&                                     outParticipation,
    const std::function<ovspatialquery_api_status_t(const CopiedSourceSnapshot&)>& inspect, std::string* outDiagnostic,
    const std::shared_ptr<void>& sourceStateOverride = {});

ovspatialquery_api_status_t getPopulationSource(ovspatialquery_instance_handle_t   instance,
                                                ovspatialquery_population_handle_t population,
                                                ovspatialquery_source_kind_t       sourceKind,
                                                PopulationSourceBinding&           outSource);

ovspatialquery_result_t apiResult(ovspatialquery_api_status_t status, const char* message = nullptr) noexcept;

/* The dropped-triangle list a successful publication operation keeps (ADR-0073): NOT_READY
 * while pending, OPERATION_FAILED after failure, INVALID_HANDLE for another operation. */
ovspatialquery_api_status_t fetchDroppedTriangles(ovspatialquery_instance_handle_t instance,
                                                  ovspatialquery_op_id_t operation, DLTensor& outObjects,
                                                  DLTensor& outCounts) noexcept;

ovspatialquery_enqueue_result_t sourcePublicEnqueueResult(ovspatialquery_api_status_t status,
                                                          ovspatialquery_op_id_t operation, const char* message);

} // namespace detail
} // namespace ovspatialquery

#endif
