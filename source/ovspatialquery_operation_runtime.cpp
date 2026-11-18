/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_instance_state.h"

#include "ovspatialquery_instance_operation_runtime.h"

#include <algorithm>
#include <type_traits>

namespace ovspatialquery
{
namespace detail
{

namespace
{

OperationKind privateOperationKind(InstanceOperationKind kind) noexcept
{
    switch (kind)
    {
    case InstanceOperationKind::familyBatchPlan: return OperationKind::familyBatchPlanPrivate;
    case InstanceOperationKind::pairReporter: return OperationKind::pairReporterPrivate;
    case InstanceOperationKind::explicitRegistration: return OperationKind::explicitRegistrationPrivate;
    }
    return OperationKind::populationUpdate;
}

bool privateOperationMatches(const OperationSlot& slot, InstanceOperationKind kind, const void* owner) noexcept
{
    return slot.inUse && slot.kind == privateOperationKind(kind) && slot.privateOwner == owner;
}

InstanceOperationState privateOperationState(const OperationSlot& slot) noexcept
{
    if (!slot.terminal)
    {
        return InstanceOperationState::pending;
    }
    return slot.executionStatus == OVSPATIALQUERY_API_SUCCESS ? InstanceOperationState::completed
                                                              : InstanceOperationState::failed;
}

} // namespace

ovspatialquery_api_status_t reserveInstanceOperation(ovspatialquery_instance_handle_t instanceHandle,
                                                     InstanceOperationKind kind, const void* owner,
                                                     InstanceOperationReservation& outReservation) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    try
    {
        std::lock_guard<std::mutex> lifecycleLock(instance->lifecycleMutex);
        if (instance->released)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        {
            std::unique_lock<std::mutex> testLock(instance->instanceOperationTestMutex);
            instance->instanceOperationReservationAfterLifecycle = true;
            instance->instanceOperationTestCondition.notify_all();
            instance->instanceOperationTestCondition.wait(
                testLock, [instance]() { return !instance->pauseInstanceOperationReservationAfterLifecycle; });
            instance->instanceOperationReservationAfterLifecycle = false;
        }
#endif
        uint32_t                          slotIndex = UINT32_MAX;
        ovspatialquery_op_id_t            operation = OVSPATIALQUERY_INVALID_OP;
        const ovspatialquery_api_status_t status =
            reserveOperation(instance, privateOperationKind(kind), slotIndex, operation);
        // reserveOperation fails only when every slot is taken.
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return OVSPATIALQUERY_API_OPERATION_CAPACITY;
        }
        OperationSlot& slot = *instance->operationSlots[slotIndex];
        {
            std::lock_guard<std::mutex> slotLock(slot.mutex);
            slot.privateOwner = owner;
        }
        InstanceOperationReservation result{};
        result.operation  = operation;
        result.slotIndex  = slotIndex;
        result.generation = slot.generation;
        result.kind       = kind;
        result.owner      = owner;
        outReservation    = result;
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

bool reserveInstanceOperationHostTask(ovspatialquery_instance_handle_t instanceHandle, uint64_t bytes,
                                      HostLease& outLease) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return false;
    }
    HostTransaction transaction(instance->hostLedger);
    return transaction.reserve(bytes) && transaction.commit(outLease);
}

ovspatialquery_api_status_t bindInstanceOperation(ovspatialquery_instance_handle_t    instanceHandle,
                                                  const InstanceOperationReservation& reservation,
                                                  std::shared_ptr<OperationTask> task, HostLease&& taskLease,
                                                  bool activate) noexcept
{
    if (task == nullptr || reservation.operation == OVSPATIALQUERY_INVALID_OP || reservation.owner == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    try
    {
        {
            std::lock_guard<std::mutex> operationLock(instance->operationMutex);
            if (reservation.slotIndex >= instance->operationSlots.size())
            {
                return OVSPATIALQUERY_API_INVALID_HANDLE;
            }
            OperationSlot&              slot = *instance->operationSlots[reservation.slotIndex];
            std::lock_guard<std::mutex> slotLock(slot.mutex);
            if (slot.generation != reservation.generation ||
                !privateOperationMatches(slot, reservation.kind, reservation.owner) || slot.task != nullptr)
            {
                return OVSPATIALQUERY_API_INVALID_HANDLE;
            }
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
            if (instance->failNextInstanceOperationScheduling)
            {
                instance->failNextInstanceOperationScheduling = false;
                return OVSPATIALQUERY_API_INTERNAL;
            }
#endif
            // Queue allocation is the final fallible admission step. Commit the
            // typed task only after it succeeds so caller rollback can always
            // abandon an unscheduled reservation atomically.
            if (activate)
            {
                instance->pendingOperations.push_back(reservation.slotIndex);
            }
            slot.task      = std::move(task);
            slot.taskLease = std::move(taskLease);
        }
        if (activate)
        {
            // Only workers wait on this condition (the CUDA reaper has its own),
            // so one wake reaches a worker that can run the operation. Notified
            // after the locks are released (see activateInstanceOperation).
            instance->operationCondition.notify_one();
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t activateInstanceOperation(ovspatialquery_instance_handle_t    instanceHandle,
                                                      const InstanceOperationReservation& reservation) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    {
        std::lock_guard<std::mutex> operationLock(instance->operationMutex);
        if (reservation.slotIndex >= instance->operationSlots.size())
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        OperationSlot&              slot = *instance->operationSlots[reservation.slotIndex];
        std::lock_guard<std::mutex> slotLock(slot.mutex);
        if (slot.generation != reservation.generation ||
            !privateOperationMatches(slot, reservation.kind, reservation.owner) || slot.task == nullptr)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        instance->pendingOperations.push_back(reservation.slotIndex);
    }
    // Wake one worker after releasing both locks, as enqueueOperation does: a
    // worker woken while they are held can preempt the caller on its CPU (Windows
    // boosts the thread it wakes), block on them, and switch back at each release.
    instance->operationCondition.notify_one();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t reserveInstanceDeviceResult(ovspatialquery_instance_handle_t instanceHandle,
                                                        std::shared_ptr<void>&           outReservation) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex> operationLock(instance->operationMutex);
    if (instance->liveDeviceResultCount.load() >= instance->deviceResultCapacity)
    {
        return OVSPATIALQUERY_API_RESULT_CAPACITY;
    }
    try
    {
        std::shared_ptr<DeviceResultReservation> reservation = std::make_shared<DeviceResultReservation>();
        reservation->liveCount                               = &instance->liveDeviceResultCount;
        reservation->condition                               = &instance->operationCondition;
        instance->liveDeviceResultCount.fetch_add(1u);
        outReservation = std::move(reservation);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_DEVICE_BUDGET;
    }
}

ovspatialquery_api_status_t reserveInstanceDeviceMemory(ovspatialquery_instance_handle_t instanceHandle,
                                                        DLDevice device, uint64_t bytes,
                                                        std::shared_ptr<void>& outReservation) noexcept
{
    outReservation.reset();
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (!instance->reserveDevice(device, bytes))
    {
        return OVSPATIALQUERY_API_DEVICE_BUDGET;
    }
    try
    {
        std::weak_ptr<InstanceState> weakInstance = instance.shared();
        outReservation                            = std::shared_ptr<void>(reinterpret_cast<void*>(UINTPTR_MAX),
                                               [weakInstance, device, bytes](void*)
                                               {
                                                   const std::shared_ptr<InstanceState> locked = weakInstance.lock();
                                                   if (locked != nullptr)
                                                   {
                                                       locked->releaseDevice(device, bytes);
                                                   }
                                               });
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (...)
    {
        // Only the handle's allocation can throw, and the shared_ptr
        // constructor then runs the deleter: the charge is released.
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

// Abandons a reservation with no task bound, or (`bound`) one whose task is
// bound but has not started; the task is released after the slot.
static void abandonPrivateOperation(ovspatialquery_instance_handle_t instanceHandle,
                                    InstanceOperationReservation& reservation, bool bound) noexcept
{
    InstanceReference              instance = instanceState(instanceHandle);
    std::shared_ptr<OperationTask> task;
    if (instance != nullptr && reservation.slotIndex < instance->operationSlots.size())
    {
        OperationSlot& slot    = *instance->operationSlots[reservation.slotIndex];
        bool           matches = false;
        {
            std::lock_guard<std::mutex> operationLock(instance->operationMutex);
            std::lock_guard<std::mutex> slotLock(slot.mutex);
            matches = slot.generation == reservation.generation &&
                      privateOperationMatches(slot, reservation.kind, reservation.owner) && !slot.workerActive &&
                      (slot.task != nullptr) == bound;
            if (matches)
            {
                task = std::move(slot.task);
            }
        }
        if (matches)
        {
            abandonReservedOperation(instance, reservation.slotIndex);
        }
    }
    task.reset();
    reservation = {};
}

void abandonInstanceOperation(ovspatialquery_instance_handle_t instanceHandle,
                              InstanceOperationReservation&    reservation) noexcept
{
    abandonPrivateOperation(instanceHandle, reservation, false);
}

void abandonBoundInstanceOperation(ovspatialquery_instance_handle_t instanceHandle,
                                   InstanceOperationReservation&    reservation) noexcept
{
    abandonPrivateOperation(instanceHandle, reservation, true);
}

ovspatialquery_api_status_t waitInstanceOperation(ovspatialquery_instance_handle_t instanceHandle,
                                                  ovspatialquery_op_id_t operation, uint64_t timeoutNanoseconds,
                                                  InstanceOperationState&      outState,
                                                  ovspatialquery_api_status_t* outExecutionStatus) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(instance, operation, slotIndex, generation);
    if (slot == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::unique_lock<std::mutex> lock(slot->mutex);
    if (!slot->inUse || slot->generation != generation)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    bool waited = false;
    if (!slot->terminal)
    {
        if (timeoutNanoseconds == 0u)
        {
            return OVSPATIALQUERY_API_TIMEOUT;
        }
        // An unbounded wait on a still-queued host operation runs it here instead
        // of handing it to a worker and sleeping until the worker finishes.
        if (timeoutNanoseconds == UINT64_MAX)
        {
            lock.unlock();
            (void)instance->runQueuedOperationOnWaiter(slotIndex, generation);
            lock.lock();
        }
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        {
            std::lock_guard<std::mutex> testLock(instance->instanceOperationTestMutex);
            ++instance->instanceOperationWaiterEntryCount;
        }
#endif
        waited = true;
        if (!waitUntil(slot->condition, lock, timeoutNanoseconds,
                       [slot, generation]()
                       { return slot->terminal || !slot->inUse || slot->generation != generation; }))
        {
            return OVSPATIALQUERY_API_TIMEOUT;
        }
    }
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
    if (waited)
    {
        lock.unlock();
        {
            std::unique_lock<std::mutex> testLock(instance->instanceOperationTestMutex);
            if (instance->pauseInstanceOperationWaiterBeforeRevalidation)
            {
                instance->instanceOperationWaiterAtRevalidation = true;
                instance->instanceOperationTestCondition.notify_all();
                instance->instanceOperationTestCondition.wait(
                    testLock, [instance]() { return !instance->pauseInstanceOperationWaiterBeforeRevalidation; });
                instance->instanceOperationWaiterAtRevalidation = false;
            }
        }
        lock.lock();
    }
#else
    (void)waited;
#endif
    if (!slot->inUse || slot->generation != generation)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    slot->terminalObserved             = true;
    const InstanceOperationState state = privateOperationState(*slot);
    outState                           = state;
    if (outExecutionStatus != nullptr)
    {
        *outExecutionStatus = slot->executionStatus;
    }
    return state == InstanceOperationState::failed ? OVSPATIALQUERY_API_OPERATION_FAILED : OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t queryInstanceOperation(ovspatialquery_instance_handle_t instanceHandle,
                                                   ovspatialquery_op_id_t           operation,
                                                   InstanceOperationState&          outState) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(instance, operation, slotIndex, generation);
    if (slot == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex> lock(slot->mutex);
    if (!slot->inUse || slot->generation != generation)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    const InstanceOperationState state = privateOperationState(*slot);
    if (state != InstanceOperationState::pending)
    {
        slot->terminalObserved = true;
    }
    outState = state;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t destroyInstanceOperation(
    ovspatialquery_instance_handle_t instanceHandle, InstanceOperationKind kind, const void* owner,
    ovspatialquery_op_id_t& operation, ovspatialquery_cuda_event_t* consumerDone,
    std::shared_ptr<InstanceOperationDeviceRetirement> retirement) noexcept
{
    const bool        retiring = retirement != nullptr;
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    uint32_t slotIndex  = 0u;
    uint32_t generation = 0u;
    if (!decodeOperationId(instance, operation, slotIndex, generation))
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::shared_ptr<OperationTask> task;
    HostLease                      taskLease;
    ApiOperationOwner              apiOwner;
    {
        std::lock_guard<std::mutex> operationLock(instance->operationMutex);
        if (slotIndex >= instance->operationSlots.size())
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        OperationSlot&              slot = *instance->operationSlots[slotIndex];
        std::lock_guard<std::mutex> slotLock(slot.mutex);
        if (slot.generation != generation || !privateOperationMatches(slot, kind, owner))
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        if (slot.workerActive || !slot.terminal || !slot.terminalObserved)
        {
            return OVSPATIALQUERY_API_BUSY;
        }
        if (retiring)
        {
            if (instance->retiredPrivateDeviceOperations.size() >= instance->retiredPrivateDeviceOperations.capacity())
            {
                return OVSPATIALQUERY_API_BUSY;
            }
            if (retirement->transferConsumerDone(*consumerDone) != OVSPATIALQUERY_API_SUCCESS)
            {
                return OVSPATIALQUERY_API_OPERATION_FAILED;
            }
            retirement->operationOwner = owner;
            // The reaper wakes when the consumer finishes instead of polling for it.
            retirement->ringWhenDeviceSafe(instance->operationCondition.signal);
            instance->retiredPrivateDeviceOperations.push_back(std::move(retirement));
        }
        task = std::move(slot.task);
        if (retiring && task != nullptr)
        {
            task->disarmForDeviceRetirement();
        }
        taskLease              = std::move(slot.taskLease);
        apiOwner               = std::move(slot.apiOwner);
        slot.privateOwner      = nullptr;
        slot.privateFailure    = OVSPATIALQUERY_API_SUCCESS;
        slot.privateDiagnostic = nullptr;
        slot.inUse             = false;
        ++slot.generation;
        --instance->liveOperationCount;
        operation = OVSPATIALQUERY_INVALID_OP;
        if (retiring)
        {
            *consumerDone = {};
        }
    }
    task.reset();
    taskLease.reset();
    apiOwner = {};
    if (retiring)
    {
        instance->operationCondition.notify_all();
        instance->cudaReaperCondition.notify_all();
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

void reclaimInstanceDeviceRetirements(ovspatialquery_instance_handle_t instanceHandle, const void* owner) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance != nullptr && owner != nullptr && instance->reclaimDeviceRetirements(owner))
    {
        instance->operationCondition.notify_all();
    }
}

#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
ovspatialquery_api_status_t setInstanceOperationWorkersPausedForTest(ovspatialquery_instance_handle_t instanceHandle,
                                                                     bool                             paused) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    {
        std::lock_guard<std::mutex> lock(instance->operationMutex);
        instance->instanceOperationWorkersPaused = paused;
    }
    instance->operationCondition.notify_all();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t setInstanceCudaReaperPausedForTest(ovspatialquery_instance_handle_t instanceHandle,
                                                               bool                             paused) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    instance->cudaReaperPaused.store(paused);
    // A reaper pass already under way finishes before this returns; later
    // passes see the flag.
    {
        std::lock_guard<std::mutex> reclaimLock(instance->retirementReclaimMutex);
    }
    instance->operationCondition.notify_all();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t setInstanceOperationExecutionsPausedForTest(ovspatialquery_instance_handle_t instanceHandle,
                                                                        bool paused) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    {
        std::lock_guard<std::mutex> lock(instance->operationMutex);
        instance->instanceOperationExecutionsPaused = paused;
    }
    instance->instanceOperationExecutionCondition.notify_all();
    instance->operationCondition.notify_all();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t failNextInstanceOperationSchedulingForTest(
    ovspatialquery_instance_handle_t instanceHandle) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex> lock(instance->operationMutex);
    instance->failNextInstanceOperationScheduling = true;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t setInstanceOperationReservationPausedAfterLifecycleForTest(
    ovspatialquery_instance_handle_t instanceHandle, bool paused) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    {
        std::lock_guard<std::mutex> lock(instance->instanceOperationTestMutex);
        instance->pauseInstanceOperationReservationAfterLifecycle = paused;
    }
    instance->instanceOperationTestCondition.notify_all();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t waitForInstanceOperationReservationAfterLifecycleForTest(
    ovspatialquery_instance_handle_t instanceHandle) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::unique_lock<std::mutex> lock(instance->instanceOperationTestMutex);
    instance->instanceOperationTestCondition.wait(lock, [instance]()
                                                  { return instance->instanceOperationReservationAfterLifecycle; });
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t setInstanceOperationBeforeTerminalPausedForTest(
    ovspatialquery_instance_handle_t instanceHandle, bool paused) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    {
        std::lock_guard<std::mutex> lock(instance->instanceOperationTestMutex);
        instance->pauseInstanceOperationBeforeTerminal = paused;
    }
    instance->instanceOperationTestCondition.notify_all();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t waitForInstanceOperationBeforeTerminalForTest(
    ovspatialquery_instance_handle_t instanceHandle) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::unique_lock<std::mutex> lock(instance->instanceOperationTestMutex);
    instance->instanceOperationTestCondition.wait(lock,
                                                  [instance]() { return instance->instanceOperationBeforeTerminal; });
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t waitForInstanceReleaseBeforeLifecycleForTest(
    ovspatialquery_instance_handle_t instanceHandle) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::unique_lock<std::mutex> lock(instance->instanceOperationTestMutex);
    instance->instanceOperationTestCondition.wait(lock,
                                                  [instance]() { return instance->instanceReleaseBeforeLifecycle; });
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t setInstanceOperationWaiterPausedBeforeRevalidationForTest(
    ovspatialquery_instance_handle_t instanceHandle, bool paused) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    {
        std::lock_guard<std::mutex> lock(instance->instanceOperationTestMutex);
        instance->pauseInstanceOperationWaiterBeforeRevalidation = paused;
    }
    instance->instanceOperationTestCondition.notify_all();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t waitForInstanceOperationWaiterAtRevalidationForTest(
    ovspatialquery_instance_handle_t instanceHandle) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::unique_lock<std::mutex> lock(instance->instanceOperationTestMutex);
    instance->instanceOperationTestCondition.wait(lock, [instance]()
                                                  { return instance->instanceOperationWaiterAtRevalidation; });
    return OVSPATIALQUERY_API_SUCCESS;
}

uint64_t instanceOperationWaiterEntryCountForTest(ovspatialquery_instance_handle_t instanceHandle) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return 0u;
    }
    std::lock_guard<std::mutex> lock(instance->instanceOperationTestMutex);
    return instance->instanceOperationWaiterEntryCount;
}

ovspatialquery_api_status_t inspectInstanceOperationRuntimeForTest(
    ovspatialquery_instance_handle_t instanceHandle, InstanceOperationRuntimeInspection& outInspection) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex>        lock(instance->operationMutex);
    InstanceOperationRuntimeInspection result{};
    result.pendingOperationCount     = static_cast<uint32_t>(instance->pendingOperations.size());
    result.runningOperationCount     = instance->runningInstanceOperationCount;
    result.peakRunningOperationCount = instance->peakRunningInstanceOperationCount;
    outInspection                    = result;
    return OVSPATIALQUERY_API_SUCCESS;
}
#endif

} // namespace detail
} // namespace ovspatialquery
