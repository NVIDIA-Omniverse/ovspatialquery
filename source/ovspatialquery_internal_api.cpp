/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_instance_state.h"

#include "ovspatialquery_target_resolver.h"

#include <ovspatialquery_query_evaluation.h>

#include <algorithm>
#include <stdexcept>

using namespace ovspatialquery::detail;

namespace
{

} // namespace

extern "C" ovspatialquery_api_status_t ovspatialquery_release_instance(ovspatialquery_instance_handle_t instance_handle)
{
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
    {
        std::lock_guard<std::mutex> testLock(instance->instanceOperationTestMutex);
        instance->instanceReleaseBeforeLifecycle = true;
    }
    instance->instanceOperationTestCondition.notify_all();
#endif
    std::lock_guard<std::mutex> lifecycleLock(instance->lifecycleMutex);
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
    {
        std::lock_guard<std::mutex> testLock(instance->instanceOperationTestMutex);
        instance->instanceReleaseBeforeLifecycle = false;
    }
    instance->instanceOperationTestCondition.notify_all();
#endif
    if (instance->released)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    size_t liveOperationCount                 = 0u;
    size_t liveDeviceResultCount              = 0u;
    size_t liveApiOwnerCount                  = 0u;
    size_t retiredPrivateDeviceOperationCount = 0u;
    {
        std::lock_guard<std::mutex> lock(instance->operationMutex);
        liveOperationCount                 = instance->liveOperationCount;
        liveDeviceResultCount              = instance->liveDeviceResultCount.load();
        retiredPrivateDeviceOperationCount = instance->retiredPrivateDeviceOperations.size();
    }
    // An operation's API owner lives in its slot, which liveOperationCount counts.
    liveApiOwnerCount = instance->handles.adapterCount();
    if (instance->populations.populationCount() != 0u ||
        instance->explicitPreparationRegistry->definitionCount() != 0u || liveApiOwnerCount != 0u ||
        liveOperationCount != 0u || liveDeviceResultCount != 0u || retiredPrivateDeviceOperationCount != 0u)
    {
        return OVSPATIALQUERY_API_BUSY;
    }
    instance->released = true;
    instanceDirectory().erase(instance_handle);
    return OVSPATIALQUERY_API_SUCCESS;
}

extern "C" ovspatialquery_api_status_t ovspatialquery_release_population(
    ovspatialquery_instance_handle_t instance_handle, ovspatialquery_population_handle_t population)
{
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    ovspatialquery::detail::PopulationKey           key{};
    ovspatialquery::detail::PopulationSourceBinding resolvedSource;
    ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, key, resolvedSource);
    resolvedSource                     = {}; // only the key is needed; retiredSource below stays the last owner
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    std::shared_ptr<void>                                             deferredDestruction;
    ovspatialquery::detail::PopulationSourceBinding                   retiredSource;
    std::shared_ptr<const ovspatialquery::detail::ParticipationTable> retiredParticipation;
    {
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        status                                            = instance->handles.hasLiveView(key) ? OVSPATIALQUERY_API_BUSY
                                                                                               : instance->populations.release(key, deferredDestruction);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = instance->handles.invalidatePopulation(population, retiredSource, &retiredParticipation);
        }
    }
    deferredDestruction.reset();
    retiredSource = {};
    const uint64_t retiredParticipationBytes =
        retiredParticipation == nullptr ? 0u : retiredParticipation->residentByteCount();
    retiredParticipation.reset();
    instance->hostLedger.release(retiredParticipationBytes);
    return status;
}

extern "C" ovspatialquery_api_status_t ovspatialquery_operation_release(
    ovspatialquery_instance_handle_t instance_handle, ovspatialquery_op_id_t* operation)
{
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (operation == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    uint32_t slotIndex  = 0u;
    uint32_t generation = 0u;
    if (!decodeOperationId(instance, *operation, slotIndex, generation))
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    // Released after the operation table is unlocked.
    std::shared_ptr<ovspatialquery::detail::OperationTask> task;
    std::vector<std::shared_ptr<const void>>               definitionOwners;
    {
        std::lock_guard<std::mutex> operationLock(instance->operationMutex);
        if (slotIndex >= instance->operationSlots.size())
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        OperationSlot& slot = *instance->operationSlots[slotIndex];
        if (!slot.inUse || slot.generation != generation)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        std::lock_guard<std::mutex> slotLock(slot.mutex);
        if (!slot.terminal)
        {
            return OVSPATIALQUERY_API_BUSY;
        }
        task             = std::move(slot.task);
        definitionOwners = std::move(slot.updateDefinitionOwners);
        slot.inUse       = false;
        ++slot.generation;
        --instance->liveOperationCount;
        *operation = OVSPATIALQUERY_INVALID_OP;
    }
    instance->operationCondition.notify_all();
    return OVSPATIALQUERY_API_SUCCESS;
}
