/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_instance_state.h"

#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_target_resolver.h"

using namespace ovspatialquery::detail;

namespace
{

thread_local std::string gLastOperationError;

#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
std::mutex              gApiFetchProjectionTestMutex;
std::condition_variable gApiFetchProjectionTestCondition;
std::array<bool, 2u>    gApiFetchProjectionPaused{};
std::array<bool, 2u>    gApiFetchProjectionReached{};
#endif

const char* operationFailureMessage(ovspatialquery_api_status_t status)
{
    switch (status)
    {
    case OVSPATIALQUERY_API_OUT_OF_MEMORY: return "operation ran out of memory";
    case OVSPATIALQUERY_API_INTERNAL: return "operation failed internally";
    default: return "operation failed";
    }
}

// Whether a slot's task records its own failure status and diagnostic: the
// family-plan and pair-reporter operations.
bool apiReportsTaskFailure(const OperationSlot& slot)
{
    return slot.kind == OperationKind::familyBatchPlanPrivate || slot.kind == OperationKind::pairReporterPrivate;
}

// Materializes a completed operation's projection; a CUDA plan's results stay
// on its owner, so only CPU plans and reporters materialize one.
ovspatialquery_api_status_t apiMaterializeLocked(const ApiOperationOwner& owner, ovspatialquery_op_id_t operationId)
{
    if (owner.plan != nullptr)
    {
        return owner.plan->cudaPlan ? OVSPATIALQUERY_API_SUCCESS : apiMaterializeFamilyLocked(owner, operationId);
    }
    return apiMaterializeReporterLocked(owner, operationId);
}

// Calls visit with the plan or the pair reporter an API owner names.
template <typename Visit>
void apiVisitAdapter(const ApiOperationOwner& owner, Visit visit)
{
    if (owner.plan != nullptr)
    {
        visit(*owner.plan);
    }
    else if (owner.reporter != nullptr)
    {
        visit(*owner.reporter);
    }
}

} // namespace

namespace ovspatialquery
{
namespace detail
{

#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
void apiFetchProjectionPausePointForTest(uint32_t kind)
{
    if (kind >= gApiFetchProjectionPaused.size())
    {
        return;
    }
    std::unique_lock<std::mutex> lock(gApiFetchProjectionTestMutex);
    gApiFetchProjectionReached[kind] = true;
    gApiFetchProjectionTestCondition.notify_all();
    gApiFetchProjectionTestCondition.wait(lock, [kind]() { return !gApiFetchProjectionPaused[kind]; });
}
#endif

ovspatialquery::detail::PopulationTarget apiPlanTarget(const InstanceReference&           instance,
                                                       const ovspatialquery_target_ref_t& source, bool pairRetention,
                                                       std::shared_ptr<void>&       outViewRetention,
                                                       ovspatialquery_api_status_t& outStatus)
{
    ovspatialquery::detail::PopulationTarget target{};
    outStatus = OVSPATIALQUERY_API_SUCCESS;
    if (source.kind == 0u)
    {
        outStatus = instance->handles.resolvePopulation(source.handle.population, target.population);
    }
    else if (source.kind == 1u)
    {
        const ovspatialquery_api_status_t retainStatus =
            instance->handles.retainView(source.handle.view, !pairRetention, target.view);
        if (retainStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            outStatus = retainStatus;
        }
        else
        {
            target.population = target.view->parent;
            try
            {
                const std::weak_ptr<InstanceState>                       weakInstance = instance.shared();
                const std::shared_ptr<ovspatialquery::detail::ViewState> view         = target.view;
                outViewRetention = std::shared_ptr<void>(reinterpret_cast<void*>(UINTPTR_MAX),
                                                         [weakInstance, view](void*)
                                                         {
                                                             const std::shared_ptr<InstanceState> locked =
                                                                 weakInstance.lock();
                                                             if (locked != nullptr)
                                                             {
                                                                 locked->handles.finishViewReference(view);
                                                             }
                                                         });
            }
            catch (...)
            {
                // Only the handle's allocation can throw, and the shared_ptr
                // constructor then runs the deleter: the reference is finished.
                target    = {};
                outStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
        }
    }
    else
    {
        outStatus = OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    return target;
}

ovspatialquery::detail::TargetSelection apiSelection(const ovspatialquery_ordinal_selection_t& source, bool& valid)
{
    valid = true;
    if (source.kind == 0u)
    {
        return {ovspatialquery::detail::TargetSelectionKind::head, 0};
    }
    if (source.kind == 1u && source.source_ordinal >= 0)
    {
        return {ovspatialquery::detail::TargetSelectionKind::retainedExact, source.source_ordinal};
    }
    if (source.kind == 2u && source.source_ordinal >= 0)
    {
        return {ovspatialquery::detail::TargetSelectionKind::admittedExact, source.source_ordinal};
    }
    valid = false;
    return {};
}

ovspatialquery_op_state_t apiOpState(InstanceOperationState state)
{
    switch (state)
    {
    case InstanceOperationState::pending: return OVSPATIALQUERY_OP_PENDING;
    case InstanceOperationState::completed: return OVSPATIALQUERY_OP_COMPLETED;
    case InstanceOperationState::failed: return OVSPATIALQUERY_OP_FAILURE;
    }
    return OVSPATIALQUERY_OP_FAILURE;
}

} // namespace detail
} // namespace ovspatialquery

#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
extern "C" void ovspatialquery_test_current_api_pause_fetch_projection(uint32_t kind, bool paused)
{
    if (kind >= gApiFetchProjectionPaused.size())
    {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(gApiFetchProjectionTestMutex);
        gApiFetchProjectionPaused[kind] = paused;
        if (paused)
        {
            gApiFetchProjectionReached[kind] = false;
        }
    }
    gApiFetchProjectionTestCondition.notify_all();
}

extern "C" void ovspatialquery_test_current_api_wait_fetch_projection(uint32_t kind)
{
    if (kind >= gApiFetchProjectionReached.size())
    {
        return;
    }
    std::unique_lock<std::mutex> lock(gApiFetchProjectionTestMutex);
    gApiFetchProjectionTestCondition.wait(lock, [kind]() { return gApiFetchProjectionReached[kind]; });
}
#endif

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_wait_op(ovspatialquery_instance_handle_t instance_handle, ovspatialquery_op_id_t op_id,
                       ovspatialquery_timeout_t timeout, ovspatialquery_op_wait_result_t* out_wait)
{
    if (out_wait == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT, "wait output must not be null");
    }
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE, "instance handle is invalid");
    }
    // A plan or pair reporter is BUSY to destroy while one of its operations
    // is waited on; its completed projection is materialized afterwards.
    const ApiOperationOwner owner      = apiOperationOwner(*instance, op_id);
    const bool              hasAdapter = owner.plan != nullptr || owner.reporter != nullptr;
    bool                    live       = true;
    apiVisitAdapter(owner,
                    [&](auto& adapter)
                    {
                        std::lock_guard<std::mutex> adapterLock(adapter.adapterMutex);
                        live = adapter.owner != nullptr;
                        adapter.activeWaitCount += live ? 1u : 0u;
                    });
    if (!live)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    InstanceOperationState            state           = InstanceOperationState::pending;
    ovspatialquery_api_status_t       executionStatus = OVSPATIALQUERY_API_SUCCESS;
    const ovspatialquery_api_status_t status =
        waitInstanceOperation(instance_handle, op_id, timeout, state, &executionStatus);
    ovspatialquery_api_status_t materializeStatus = OVSPATIALQUERY_API_SUCCESS;
    apiVisitAdapter(owner,
                    [&](auto& adapter)
                    {
                        std::lock_guard<std::mutex> adapterLock(adapter.adapterMutex);
                        if (status == OVSPATIALQUERY_API_SUCCESS && state == InstanceOperationState::completed)
                        {
                            materializeStatus = apiMaterializeLocked(owner, op_id);
                        }
                        --adapter.activeWaitCount;
                    });
    if (status == OVSPATIALQUERY_API_TIMEOUT)
    {
        return apiResult(OVSPATIALQUERY_API_TIMEOUT);
    }
    if (status != OVSPATIALQUERY_API_SUCCESS && status != OVSPATIALQUERY_API_OPERATION_FAILED)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE,
                         hasAdapter ? nullptr : "operation ID is invalid for the instance");
    }
    if (materializeStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(materializeStatus);
    }
    ovspatialquery_op_wait_result_t result{};
    result.state = apiOpState(state);
    *out_wait    = result;
    if (!hasAdapter && state == InstanceOperationState::failed)
    {
        try
        {
            gLastOperationError = operationFailureMessage(executionStatus);
        }
        catch (...)
        {
            gLastOperationError.clear();
        }
    }
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_destroy_op(ovspatialquery_instance_handle_t instance_handle, ovspatialquery_op_id_t* inout_op_id)
{
    if (inout_op_id == nullptr || *inout_op_id == OVSPATIALQUERY_INVALID_OP)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT, "operation ID pointer must name a nonzero operation");
    }
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE, "instance handle is invalid");
    }
    ApiOperationOwner apiOwner = apiOperationOwner(*instance, *inout_op_id);
    if (apiOwner.plan != nullptr || apiOwner.reporter != nullptr)
    {
        ovspatialquery_api_status_t outcome = OVSPATIALQUERY_API_INVALID_HANDLE;
        if (apiOwner.plan != nullptr)
        {
            std::lock_guard<std::mutex> adapterLock(apiOwner.plan->adapterMutex);
            if (apiOwner.plan->owner != nullptr)
            {
                // A CUDA operation's status query also marks a terminal one
                // observed. CUDA views, including recoverable-failure
                // reservations, must transfer consumer_done before their slot
                // and retained generation can be released; the private owner
                // alone can distinguish a completed usable view from device loss.
                InstanceOperationState state = InstanceOperationState::completed;
                if (apiOwner.plan->cudaPlan &&
                    (queryInstanceOperation(instance_handle, *inout_op_id, state) != OVSPATIALQUERY_API_SUCCESS ||
                     state == InstanceOperationState::pending))
                {
                    return apiResult(OVSPATIALQUERY_API_BUSY);
                }
                ovspatialquery::detail::FamilyBatchPlanOperationId planOperation{*inout_op_id};
                outcome = apiOwner.plan->owner->destroyOperation(planOperation);
            }
        }
        else if (apiOwner.reporter != nullptr)
        {
            std::lock_guard<std::mutex> adapterLock(apiOwner.reporter->adapterMutex);
            if (apiOwner.reporter->owner != nullptr)
            {
                ovspatialquery::detail::PairReporterOperationId reporterOperation{*inout_op_id};
                outcome = apiOwner.reporter->owner->destroyOperation(reporterOperation);
            }
        }
        if (outcome == OVSPATIALQUERY_API_BUSY)
        {
            return apiResult(OVSPATIALQUERY_API_BUSY);
        }
        if (outcome != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
        }
        *inout_op_id = OVSPATIALQUERY_INVALID_OP;
        return apiResult(OVSPATIALQUERY_API_SUCCESS);
    }
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(instance, *inout_op_id, slotIndex, generation);
    if (slot == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE, "operation ID is invalid for the instance");
    }
    {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (!slot->terminal || !slot->terminalObserved)
        {
            return apiResult(OVSPATIALQUERY_API_BUSY, "operation must be terminal and observed before destruction");
        }
    }
    if (slot->kind == OperationKind::explicitRegistrationPrivate)
    {
        std::lock_guard<std::mutex> explicitLock(instance->explicitRuntimeMutex);
        if (instance->explicitRegistrationOwner == nullptr)
        {
            return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
        }
        ovspatialquery::detail::ExplicitRegistrationOperationId explicitId{*inout_op_id};
        const ovspatialquery_api_status_t outcome = instance->explicitRegistrationOwner->destroyOperation(explicitId);
        if (outcome != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(outcome == OVSPATIALQUERY_API_BUSY ? OVSPATIALQUERY_API_BUSY
                                                                : OVSPATIALQUERY_API_INVALID_HANDLE);
        }
        *inout_op_id = OVSPATIALQUERY_INVALID_OP;
        return apiResult(OVSPATIALQUERY_API_SUCCESS);
    }
    return apiResult(ovspatialquery_operation_release(instance_handle, inout_op_id));
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_op_after_cuda(
    ovspatialquery_instance_handle_t instance_handle, ovspatialquery_op_id_t* inout_op_id,
    ovspatialquery_cuda_event_t* inout_consumer_done)
{
    if (inout_op_id == nullptr || inout_consumer_done == nullptr || *inout_op_id == OVSPATIALQUERY_INVALID_OP ||
        inout_consumer_done->value == 0u)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    ApiOperationOwner apiOwner         = apiOperationOwner(*instance, *inout_op_id);
    const auto        destroyAfterCuda = [&](auto& adapter, auto operation)
    {
        std::lock_guard<std::mutex> adapterLock(adapter.adapterMutex);
        if (adapter.owner == nullptr)
        {
            return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
        }
        ovspatialquery_cuda_event_t       consumerDone = *inout_consumer_done;
        const ovspatialquery_api_status_t status = adapter.owner->destroyOperationAfterCuda(operation, consumerDone);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(status == OVSPATIALQUERY_API_BUSY ? OVSPATIALQUERY_API_BUSY
                                                               : OVSPATIALQUERY_API_INVALID_STATE);
        }
        *inout_consumer_done = consumerDone;
        *inout_op_id         = OVSPATIALQUERY_INVALID_OP;
        return apiResult(OVSPATIALQUERY_API_SUCCESS);
    };
    if (apiOwner.plan != nullptr && apiOwner.plan->cudaPlan)
    {
        return destroyAfterCuda(*apiOwner.plan, ovspatialquery::detail::FamilyBatchPlanOperationId{*inout_op_id});
    }
    if (apiOwner.reporter != nullptr && apiOwner.reporter->resultDevice.device_type == kDLCUDA)
    {
        return destroyAfterCuda(*apiOwner.reporter, ovspatialquery::detail::PairReporterOperationId{*inout_op_id});
    }
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(instance, *inout_op_id, slotIndex, generation);
    if (slot == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    {
        std::lock_guard<std::mutex> operationLock(instance->operationMutex);
        std::lock_guard<std::mutex> slotLock(slot->mutex);
        if (!slot->terminal || !slot->terminalObserved)
        {
            return apiResult(OVSPATIALQUERY_API_BUSY);
        }
        return apiResult(OVSPATIALQUERY_API_INVALID_STATE);
    }
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_query_op_status(ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_op_id_t operationId,
                               ovspatialquery_op_status_t* outStatus)
{
    if (outStatus == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    // A plan or pair reporter operation is read and its completed projection
    // materialized under the adapter lock, which destroy_op and the plan or
    // reporter destruction take before they release either.
    const ApiOperationOwner      owner = apiOperationOwner(*instance, operationId);
    std::unique_lock<std::mutex> adapterLock;
    bool                         live = true;
    apiVisitAdapter(owner,
                    [&](auto& adapter)
                    {
                        adapterLock = std::unique_lock<std::mutex>(adapter.adapterMutex);
                        live        = adapter.owner != nullptr;
                    });
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = live ? resolveOperation(instance, operationId, slotIndex, generation) : nullptr;
    if (slot == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    ovspatialquery_op_status_t result{};
    {
        std::lock_guard<std::mutex> lock(slot->mutex);
        if (!slot->inUse || slot->generation != generation)
        {
            return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
        }
        const ovspatialquery_api_status_t executed = slot->executionStatus;
        result.state                               = !slot->terminal ? OVSPATIALQUERY_OP_PENDING
                                                     : executed == OVSPATIALQUERY_API_SUCCESS ? OVSPATIALQUERY_OP_COMPLETED
                                                                                              : OVSPATIALQUERY_OP_FAILURE;
        // A plan or reporter task reports its own failure; any other
        // operation's execution status is folded into the terminal causes.
        result.terminal_status = !slot->terminal                          ? OVSPATIALQUERY_API_NOT_READY
                                 : executed == OVSPATIALQUERY_API_SUCCESS ? OVSPATIALQUERY_API_SUCCESS
                                 : apiReportsTaskFailure(*slot)           ? slot->privateFailure
                                 : executed == OVSPATIALQUERY_API_OUT_OF_MEMORY ||
                                         executed == OVSPATIALQUERY_API_HOST_BUDGET ||
                                         executed == OVSPATIALQUERY_API_DEVICE_BUDGET
                                     ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                 : executed == OVSPATIALQUERY_API_INTERNAL || executed == OVSPATIALQUERY_API_DEVICE_LOST
                                     ? executed
                                     : OVSPATIALQUERY_API_OPERATION_FAILED;
        result.progress        = slot->terminal ? 1.0 : 0.0;
        slot->terminalObserved = slot->terminalObserved || slot->terminal;
    }
    if (adapterLock.owns_lock() && result.state == OVSPATIALQUERY_OP_COMPLETED)
    {
        const ovspatialquery_api_status_t materializeStatus = apiMaterializeLocked(owner, operationId);
        if (materializeStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(materializeStatus);
        }
    }
    *outStatus = result;
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_string_t
ovspatialquery_get_last_op_error(ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_op_id_t operationId)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr || operationId == OVSPATIALQUERY_INVALID_OP)
    {
        return {"", 0u};
    }
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(instance, operationId, slotIndex, generation);
    if (slot == nullptr)
    {
        return {"", 0u};
    }
    std::lock_guard<std::mutex> lock(slot->mutex);
    if (!slot->inUse || slot->generation != generation || !slot->terminal ||
        slot->executionStatus == OVSPATIALQUERY_API_SUCCESS)
    {
        return {"", 0u};
    }
    // A plan or reporter task names its failure; any other operation's text
    // follows its execution status.
    const char* diagnostic = !apiReportsTaskFailure(*slot)        ? operationFailureMessage(slot->executionStatus)
                             : slot->privateDiagnostic != nullptr ? slot->privateDiagnostic
                                                                  : "";
    if (*diagnostic == '\0')
    {
        return {"", 0u};
    }
    try
    {
        gLastOperationError = diagnostic;
    }
    catch (...)
    {
        return {"", 0u};
    }
    return {gLastOperationError.c_str(), gLastOperationError.size()};
}
