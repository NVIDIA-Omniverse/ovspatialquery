/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_device_box_population.h"
#include "ovspatialquery_instance_state.h"

#include <algorithm>
#include <limits>
#include <new>

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t createEmptyPopulationForSource(ovspatialquery_instance_handle_t instanceHandle,
                                                           PopulationSourceBinding          source,
                                                           uint32_t retainedOrdinalCapacity, uint64_t sourceStateCharge,
                                                           ovspatialquery_population_handle_t& outPopulation,
                                                           uint32_t pendingPublicationCapacity)
{
    outPopulation              = OVSPATIALQUERY_INVALID_POPULATION;
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (source.sourceState == nullptr || sourceStateCharge > std::numeric_limits<uint64_t>::max() - 1024u)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const uint64_t                                     populationCharge = 1024u + sourceStateCharge;
    std::shared_ptr<ovspatialquery::detail::HostLease> populationLease;
    const ovspatialquery_api_status_t leaseStatus = reserveSharedHostLease(instance, populationCharge, populationLease);
    if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return leaseStatus;
    }
    ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
    try
    {
        std::lock_guard<std::mutex> lifecycleLock(instance->lifecycleMutex);
        if (instance->released)
        {
            status = OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        else
        {
            PopulationKey internal{};
            status = instance->populations.create(retainedOrdinalCapacity, internal, pendingPublicationCapacity);
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                status = instance->handles.createPopulation(internal, outPopulation, std::move(source));
                if (status != OVSPATIALQUERY_API_SUCCESS)
                {
                    std::shared_ptr<void> deferredDestruction;
                    (void)instance->populations.release(internal, deferredDestruction);
                    deferredDestruction.reset();
                }
            }
        }
    }
    catch (const std::bad_alloc&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        status = OVSPATIALQUERY_API_INTERNAL;
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    try
    {
        std::lock_guard<std::mutex> lock(instance->budgetMutex);
        instance->populationCharges.emplace(outPopulation, std::move(populationLease));
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (...)
    {
        (void)ovspatialquery_release_population(instanceHandle, outPopulation);
        outPopulation = OVSPATIALQUERY_INVALID_POPULATION;
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

ovspatialquery_api_status_t createViewForSource(ovspatialquery_instance_handle_t   instanceHandle,
                                                ovspatialquery_source_kind_t       sourceKind,
                                                ovspatialquery_population_handle_t population,
                                                std::shared_ptr<void> sourceState, uint64_t sourceStateCharge,
                                                ovspatialquery_view_handle_t& outView)
{
    outView                    = OVSPATIALQUERY_INVALID_VIEW;
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (sourceState == nullptr || sourceStateCharge > std::numeric_limits<uint64_t>::max() - 512u)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const uint64_t                                     viewCharge = 512u + sourceStateCharge;
    std::shared_ptr<ovspatialquery::detail::HostLease> viewLease;
    const ovspatialquery_api_status_t leaseStatus = reserveSharedHostLease(instance, viewCharge, viewLease);
    if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return leaseStatus;
    }
    const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
    PopulationKey                      parent{};
    PopulationSourceBinding            parentSource;
    ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, parent, parentSource);
    // A device-prepared head has no host objects to evaluate a view against (ADR-0024).
    if (status == OVSPATIALQUERY_API_SUCCESS &&
        (parentSource.kind != sourceKind ||
         (parentSource.deviceState != nullptr && parentSource.deviceState->current() != nullptr)))
    {
        status = OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        bool updatePending = false;
        status             = instance->populations.getUpdatePending(parent, updatePending);
        if (status == OVSPATIALQUERY_API_SUCCESS && updatePending)
        {
            status = OVSPATIALQUERY_API_BUSY;
        }
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = instance->handles.createSourceView(parent, std::move(sourceState), outView);
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    try
    {
        instance->viewCharges.emplace(outView, std::move(viewLease));
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (...)
    {
        (void)instance->handles.invalidateView(outView);
        outView = OVSPATIALQUERY_INVALID_VIEW;
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

namespace
{

/* Assigns each acquired object its geometry reference: a reused one, or one staged in
 * the instance's identity registry, which the caller commits. */
ovspatialquery_api_status_t stageSourceIdentities(InstanceState& instance, bool stageDefinitions,
                                                  CopiedSourceSnapshot&            snapshot,
                                                  GeometryIdentityRegistry::Batch& identityBatch)
{
    GeometryIdentityOutcome outcome = GeometryIdentityOutcome::success;
    if (stageDefinitions)
    {
        outcome = instance.geometryIdentityRegistry.stage(
            snapshot.geometryDefinitions, identityBatch,
            snapshot.definitionOwners.empty() ? nullptr : &snapshot.definitionOwners);
        /* Definitions the registry dropped while staging leave the ledger now,
         * committed or not: the entries are gone either way. */
        instance.hostLedger.release(identityBatch.releasedResidentByteCount());
    }
    if (outcome == GeometryIdentityOutcome::success)
    {
        /* Staged references belong to the objects without a reused one, in order, or as
         * definitionSources names them. */
        const std::vector<GeometryReference>& references = identityBatch.references();
        size_t                                staged     = 0u;
        for (size_t index = 0u; index < snapshot.objects.size(); ++index)
        {
            const bool   reused = !snapshot.reusedReferences.empty() && snapshot.reusedReferences[index].value != 0u;
            const size_t source = reused                               ? 0u
                                  : snapshot.definitionSources.empty() ? staged++
                                                                       : snapshot.definitionSources[index];
            if (!reused && source >= references.size())
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
            snapshot.objects[index].geometryReference = reused ? snapshot.reusedReferences[index] : references[source];
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }
    if (outcome == GeometryIdentityOutcome::invalidArgument)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (outcome == GeometryIdentityOutcome::outOfMemory || outcome == GeometryIdentityOutcome::resourceExhausted)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    return OVSPATIALQUERY_API_INTERNAL;
}

// The publication reserves the query structures of the meshes it adds and the build
// scratch of the unbuilt ones (ADR-0075).
ovspatialquery_api_status_t admitSnapshotMeshStructures(InstanceState&              instance,
                                                        const CopiedSourceSnapshot& snapshot) noexcept
{
    try
    {
        return admitTriangleMeshStructures(instance.hostLedger,
                                           typedGeometryTriangleMeshes(snapshot.objects, snapshot.preparedConvexHulls));
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

} // namespace

ovspatialquery_api_status_t submitPopulationSnapshotForSource(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_source_kind_t sourceKind,
    ovspatialquery_population_handle_t population, const ovspatialquery_ordinal_t* exactBaseOrdinal,
    ovspatialquery_ordinal_t targetOrdinal, ovspatialquery_op_id_t& outOperation, std::string* outDiagnostic)
{
    outOperation               = OVSPATIALQUERY_INVALID_OP;
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    // The aggregate-staging failure hook reports whether this publication holds host charges by then.
    const uint64_t hostAtEntry = instance->hostLedger.total();
#endif
    if (targetOrdinal < 0 || (exactBaseOrdinal != nullptr && *exactBaseOrdinal < 0))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    PopulationKey               key{};
    PopulationSourceBinding     source;
    ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, key, source);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    if (source.kind != sourceKind)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    // A version prepared on the device stays there (ADR-0047).
    if (source.deviceState != nullptr && source.deviceState->current() != nullptr)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    status = instance->populations.validate(key);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    std::shared_ptr<PopulationUpdateTask> update;
    try
    {
        update = std::make_shared<PopulationUpdateTask>(*instance);
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    GenerationLease                           current;
    std::vector<SourceViewRequest>            viewRequests;
    std::shared_ptr<const ParticipationTable> participation;
    {
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        status = instance->populations.beginUpdate(key, targetOrdinal, exactBaseOrdinal, current);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = instance->handles.beginSourcePublication(key, viewRequests, participation);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                instance->populations.finishUpdate(key);
            }
        }
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    uint32_t               slotIndex         = 0u;
    ovspatialquery_op_id_t reservedOperation = OVSPATIALQUERY_INVALID_OP;
    status = reserveOperation(instance, OperationKind::populationUpdate, slotIndex, reservedOperation);
    /* Ends the update without publishing: the operation slot (when reserved), the base
     * lease, the views' pending publication and the population's update all let go. */
    const auto rollBack = [&](ovspatialquery_api_status_t failure, bool reserved)
    {
        update.reset();
        if (reserved)
        {
            abandonReservedOperation(instance, slotIndex);
        }
        current.reset();
        const std::unique_lock<std::mutex> completionLock    = instance->acquirePopulationCompletion();
        const ovspatialquery_api_status_t  viewCleanupStatus = instance->handles.finishSourceViewUpdate(viewRequests);
        instance->populations.finishUpdate(key);
        return viewCleanupStatus == OVSPATIALQUERY_API_SUCCESS ? failure : OVSPATIALQUERY_API_INTERNAL;
    };
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return rollBack(status, false);
    }
    const bool            typedPreparation = source.preparation == PopulationPreparationStrategy::typedFiveFamily;
    HostLease             typedStagingCharge;
    CopiedSourceSnapshot  snapshot;
    SourceAcquisitionBase acquisitionBase{};
    const SourceAcquisitionBase* base = nullptr;
    if (current.initialized())
    {
        acquisitionBase = {current.ordinal(), &current.objects(), &instance->geometryIdentityRegistry};
        base            = &acquisitionBase;
    }
    try
    {
        status = source.acquireSnapshot(source.instance, source.pathDictionary, source.sourceState, viewRequests,
                                        targetOrdinal, base, snapshot);
    }
    catch (const std::bad_alloc&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        status = OVSPATIALQUERY_API_INTERNAL;
    }
    if (status != OVSPATIALQUERY_API_SUCCESS && outDiagnostic != nullptr)
    {
        outDiagnostic->swap(snapshot.refusal);
    }
    const bool typedIdentityTransaction = typedPreparation && !snapshot.geometryDefinitions.empty();
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    if (status == OVSPATIALQUERY_API_SUCCESS && typedPreparation &&
        instance->failNextTypedGeometryAggregateStaging.exchange(false))
    {
        status =
            instance->hostLedger.total() > hostAtEntry ? OVSPATIALQUERY_API_HOST_BUDGET : OVSPATIALQUERY_API_INTERNAL;
    }
#endif
    if (status == OVSPATIALQUERY_API_SUCCESS && typedPreparation)
    {
        // The staging reservation counts what the acquirer's retained charges covered. They
        // end first, so the ledger never holds both: a concurrent operation taking the
        // released bytes makes this reservation refuse with HOST_BUDGET, never exceed it.
        snapshot.retainedCharges.clear();
        HostTransaction transaction(instance->hostLedger);
        if (!transaction.reserve(typedGeometryStagingResidentByteCount(snapshot.objects, snapshot.geometryDefinitions,
                                                                       snapshot.preparedConvexHulls)) ||
            !transaction.commit(typedStagingCharge))
        {
            status = OVSPATIALQUERY_API_HOST_BUDGET;
        }
        else
        {
            status = admitSnapshotMeshStructures(*instance, snapshot);
        }
    }
    GeometryIdentityRegistry::Batch identityBatch;
    if (status == OVSPATIALQUERY_API_SUCCESS && typedPreparation)
    {
        status = stageSourceIdentities(*instance, typedIdentityTransaction, snapshot, identityBatch);
    }
    std::vector<PreparedSourceViewMembership> preparedViewMemberships;
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        try
        {
            preparedViewMemberships.reserve(viewRequests.size());
            for (CopiedSourceSnapshot::ViewMembership& membership : snapshot.viewMemberships)
            {
                std::sort(membership.selectedIds.begin(), membership.selectedIds.end());
                preparedViewMemberships.push_back({membership.view, std::make_shared<const std::vector<uint64_t>>(
                                                                        std::move(membership.selectedIds))});
            }
        }
        catch (const std::bad_alloc&)
        {
            status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (...)
        {
            status = OVSPATIALQUERY_API_INTERNAL;
        }
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        applyParticipationTable(participation.get(), snapshot.objects);
    }
    PreparedPopulationInput input;
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = PreparedPopulationInput::prepareCopiedSet(targetOrdinal, std::move(snapshot.objects), input);
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return rollBack(status, true);
    }
    input.setTargetOrdinal(targetOrdinal);
    uint64_t membershipBytes = 0u;
    uint64_t changedIdCount  = 0u;
    try
    {
        std::vector<GenerationViewMembership> memberships;
        memberships.reserve(preparedViewMemberships.size());
        for (const PreparedSourceViewMembership& membership : preparedViewMemberships)
        {
            memberships.push_back({membership.view, membership.selectedIds});
            membershipBytes = saturatingResidentByteAdd(membershipBytes, 1u, memberships.back().residentByteCount());
        }
        std::sort(memberships.begin(), memberships.end(),
                  [](const GenerationViewMembership& first, const GenerationViewMembership& second)
                  { return first.view < second.view; });
        PopulationTable::countMembershipChanges(current, input.objectCount(), memberships, changedIdCount);
    }
    catch (const std::bad_alloc&)
    {
        return rollBack(OVSPATIALQUERY_API_OUT_OF_MEMORY, true);
    }
    update->exclusions      = std::move(snapshot.exclusions);
    uint64_t exclusionBytes = 0u;
    if (!addHostCharge(update->exclusions.objects.capacity(), sizeof(uint64_t), exclusionBytes) ||
        !addHostCharge(update->exclusions.reasons.capacity(), sizeof(int32_t), exclusionBytes) ||
        !reserveHostLease(*instance, exclusionBytes, update->exclusionLease))
    {
        return rollBack(OVSPATIALQUERY_API_HOST_BUDGET, true);
    }
    const uint64_t identityStagingCharge = identityBatch.stagingResidentByteCount();
    // The worker compares the memberships with the base's beside the input.
    uint64_t retainedInputCharge = saturatingResidentByteAdd(
        publicationAdmissionCharge(input, current, membershipBytes), 1u, identityStagingCharge);
    retainedInputCharge = saturatingResidentByteAdd(retainedInputCharge, changedIdCount, sizeof(uint64_t));
    if (!reserveHostLease(*instance, retainedInputCharge, update->retainedInputLease) ||
        !instance->hostLedger.reserve(identityBatch.additionalResidentByteCount()))
    {
        return rollBack(OVSPATIALQUERY_API_HOST_BUDGET, true);
    }
    if (typedIdentityTransaction)
    {
        instance->geometryIdentityRegistry.commit(identityBatch);
        // Releases the identity staging from the retained input charge.
        update->retainedInputLease.shrink(retainedInputCharge - identityBatch.releaseStaging());
    }
    if (!adoptDroppedTriangles(*instance, *update, std::move(snapshot.droppedTriangleCounts)))
    {
        return rollBack(OVSPATIALQUERY_API_HOST_BUDGET, true);
    }
    update->population                        = key;
    update->ordinal                           = targetOrdinal;
    update->input                             = std::move(input);
    update->source                            = std::move(current);
    update->indexedCount                      = snapshot.indexedCount;
    update->excludedCount                     = snapshot.excludedCount;
    update->hasCounts                         = true;
    update->viewRequests                      = std::move(viewRequests);
    update->viewMemberships                   = std::move(preparedViewMemberships);
    instance->operationSlots[slotIndex]->task = std::move(update);
    enqueueOperation(instance, slotIndex);
    outOperation = reservedOperation;
    if (outDiagnostic != nullptr)
    {
        outDiagnostic->swap(snapshot.droppedTriangles);
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t acquireSourceObjects(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_population_handle_t population,
    ovspatialquery_ordinal_t targetOrdinal, CopiedSourceSnapshot& outSnapshot,
    std::shared_ptr<const ParticipationTable>&                                     outParticipation,
    const std::function<ovspatialquery_api_status_t(const CopiedSourceSnapshot&)>& inspect, std::string* outDiagnostic,
    const std::shared_ptr<void>& sourceStateOverride)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    PopulationKey               key{};
    PopulationSourceBinding     source;
    ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, key, source);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    std::vector<SourceViewRequest> viewRequests;
    {
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        status = instance->handles.beginSourcePublication(key, viewRequests, outParticipation);
        const ovspatialquery_api_status_t finished = instance->handles.finishSourceViewUpdate(viewRequests);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = finished;
        }
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    if (!viewRequests.empty())
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED; // no view selects a device-prepared version
    }
    CopiedSourceSnapshot snapshot;
    snapshot.resolveMaterials = false; // device-prepared versions report no material (ADR-0066)
    try
    {
        status = source.acquireSnapshot(source.instance, source.pathDictionary,
                                        sourceStateOverride ? sourceStateOverride : source.sourceState, viewRequests,
                                        targetOrdinal, nullptr, snapshot);
    }
    catch (const std::bad_alloc&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        status = OVSPATIALQUERY_API_INTERNAL;
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        if (outDiagnostic != nullptr)
        {
            outDiagnostic->swap(snapshot.refusal);
        }
        return status;
    }
    if (snapshot.definitionSources.empty() ? snapshot.geometryDefinitions.size() != snapshot.objects.size()
                                           : snapshot.definitionSources.size() != snapshot.objects.size())
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    status = inspect(snapshot);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    HostLease       typedStagingCharge;
    HostTransaction transaction(instance->hostLedger);
    if (!transaction.reserve(typedGeometryStagingResidentByteCount(snapshot.objects, snapshot.geometryDefinitions,
                                                                   snapshot.preparedConvexHulls)) ||
        !transaction.commit(typedStagingCharge))
    {
        return OVSPATIALQUERY_API_HOST_BUDGET;
    }
    status = admitSnapshotMeshStructures(*instance, snapshot);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    GeometryIdentityRegistry::Batch identityBatch;
    status = stageSourceIdentities(*instance, !snapshot.geometryDefinitions.empty(), snapshot, identityBatch);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    if (!snapshot.geometryDefinitions.empty())
    {
        if (!instance->hostLedger.reserve(identityBatch.additionalResidentByteCount()))
        {
            return OVSPATIALQUERY_API_HOST_BUDGET;
        }
        instance->geometryIdentityRegistry.commit(identityBatch);
    }
    applyParticipationTable(outParticipation.get(), snapshot.objects);
    outSnapshot = std::move(snapshot);
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t getSourceGeometryPreparationContext(ovspatialquery_instance_handle_t  instanceHandle,
                                                                SourceGeometryPreparationContext& outPreparation)
{
    outPreparation             = {};
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    outPreparation.backendRuntime       = instance->runtimeLease.runtime;
    outPreparation.maximumResidentBytes = instance->hostByteBudget;
    outPreparation.convexHullBudget     = {instance,
                                           [](void* state, uint64_t bytes, std::shared_ptr<void>& outOwner) noexcept
                                           {
                                           std::shared_ptr<ovspatialquery::detail::HostLease> lease;
                                           const ovspatialquery_api_status_t                  outcome =
                                               reserveSharedHostLease(static_cast<InstanceState*>(state), bytes, lease);
                                           if (outcome == OVSPATIALQUERY_API_SUCCESS)
                                           {
                                               outOwner = std::move(lease);
                                               return ConvexHullPreparationOutcome::success;
                                           }
                                           if (outcome == OVSPATIALQUERY_API_HOST_BUDGET)
                                           {
                                               return ConvexHullPreparationOutcome::resourceExhausted;
                                           }
                                           if (outcome == OVSPATIALQUERY_API_OUT_OF_MEMORY)
                                           {
                                               return ConvexHullPreparationOutcome::outOfMemory;
                                           }
                                           return ConvexHullPreparationOutcome::internalFailure;
                                       }};
    outPreparation.workers              = instance->sharedWorkerLoan();
    return outPreparation.backendRuntime != nullptr ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INTERNAL;
}

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
ovspatialquery_api_status_t failNextTypedGeometryAggregateStagingForTest(
    ovspatialquery_instance_handle_t instanceHandle)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    instance->failNextTypedGeometryAggregateStaging.store(true);
    return OVSPATIALQUERY_API_SUCCESS;
}
#endif

ovspatialquery_api_status_t getPopulationSource(ovspatialquery_instance_handle_t   instanceHandle,
                                                ovspatialquery_population_handle_t population,
                                                ovspatialquery_source_kind_t       sourceKind,
                                                PopulationSourceBinding&           outSource)
{
    outSource                  = {};
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    PopulationKey                     key{};
    const ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, key, outSource);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    return outSource.kind == sourceKind ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_NOT_SUPPORTED;
}

ParticipationOutcome installParticipationTable(ovspatialquery_instance_handle_t   instanceHandle,
                                               ovspatialquery_population_handle_t population,
                                               const ParticipationTableDesc*      table)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return ParticipationOutcome::invalidHandle;
    }
    std::shared_ptr<const ParticipationTable> candidate;
    uint64_t                                  candidateReservation = 0u;
    if (table != nullptr)
    {
        candidateReservation = participationTableReservationByteCount(table->count);
        if (candidateReservation == std::numeric_limits<uint64_t>::max())
        {
            return ParticipationOutcome::outOfMemory;
        }
        if (!instance->hostLedger.reserve(candidateReservation))
        {
            return ParticipationOutcome::hostBudget;
        }
        const ParticipationOutcome copyOutcome = copyParticipationTable(*table, candidate);
        if (copyOutcome != ParticipationOutcome::success)
        {
            instance->hostLedger.release(candidateReservation);
            return copyOutcome;
        }
    }

    std::shared_ptr<const ParticipationTable> retired;
    ParticipationOutcome                      outcome = ParticipationOutcome::internalFailure;
    {
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        PopulationKey                      key{};
        PopulationSourceBinding            source;
        const ovspatialquery_api_status_t  resolveStatus =
            instance->handles.resolvePopulationSource(population, key, source);
        if (resolveStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            outcome = ParticipationOutcome::invalidHandle;
        }
        else if (source.kind != OVSPATIALQUERY_SOURCE_OVSTAGE)
        {
            outcome = ParticipationOutcome::invalidTargetSource;
        }
        else
        {
            bool                              updatePending = false;
            const ovspatialquery_api_status_t pendingStatus =
                instance->populations.getUpdatePending(key, updatePending);
            if (pendingStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                outcome = ParticipationOutcome::internalFailure;
            }
            else if (updatePending)
            {
                outcome = ParticipationOutcome::busy;
            }
            else
            {
                outcome =
                    instance->handles.replaceParticipationTable(population, candidate, candidateReservation, retired);
            }
        }
    }
    if (outcome != ParticipationOutcome::success)
    {
        candidate.reset();
        instance->hostLedger.release(candidateReservation);
        return outcome;
    }
    const uint64_t installedBytes = candidate == nullptr ? 0u : candidate->residentByteCount();
    if (installedBytes < candidateReservation)
    {
        instance->hostLedger.release(candidateReservation - installedBytes);
    }
    const uint64_t retiredBytes = retired == nullptr ? 0u : retired->residentByteCount();
    retired.reset();
    instance->hostLedger.release(retiredBytes);
    return ParticipationOutcome::success;
}

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
ovspatialquery_api_status_t inspectGeometryIdentityRegistry(ovspatialquery_instance_handle_t    instanceHandle,
                                                            GeometryIdentityRegistryInspection& outInspection)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    return instance->geometryIdentityRegistry.inspect(outInspection) ? OVSPATIALQUERY_API_SUCCESS
                                                                     : OVSPATIALQUERY_API_INTERNAL;
}
#endif

} // namespace detail
} // namespace ovspatialquery
