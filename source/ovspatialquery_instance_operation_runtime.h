/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_INSTANCE_OPERATION_RUNTIME_H
#define OVSPATIALQUERY_INSTANCE_OPERATION_RUNTIME_H

#include "ovspatialquery_device_signal.h"
#include "ovspatialquery_private_api.h"

#include <cstdint>
#include <memory>
#include <string>

namespace ovspatialquery
{
namespace detail
{

enum class InstanceOperationKind : uint8_t
{
    familyBatchPlan,
    pairReporter,
    explicitRegistration,
};

/**
 * Private post-terminal CUDA ownership retained by the InstanceState reaper.
 * A family operation transfers the caller's aggregate consumer event before
 * its operation slot is released; the reaper then destroys this continuation
 * only after both producer and consumer dependencies are device-safe.
 */
class InstanceOperationDeviceRetirement
{
  public:
    virtual ~InstanceOperationDeviceRetirement()                                                         = default;
    virtual ovspatialquery_api_status_t transferConsumerDone(ovspatialquery_cuda_event_t event) noexcept = 0;
    virtual bool                        deviceSafe() noexcept                                            = 0;
    virtual void                        releaseReadyDependencies() noexcept                              = 0;

    // Asks for `signal` to be rung when the transferred consumer event has
    // completed, so the reaper need not poll deviceSafe(). Best effort.
    virtual void ringWhenDeviceSafe(const std::shared_ptr<DeviceSignal>& signal) noexcept = 0;

    // The plan or reporter whose result slot this retirement holds; set by
    // destroyInstanceOperation so that owner can reclaim it itself.
    const void* operationOwner = nullptr;
};

enum class InstanceOperationState : uint8_t
{
    pending,
    completed,
    failed,
};

// A task execution's failure status: OUT_OF_MEMORY, OPERATION_FAILED,
// INTERNAL or DEVICE_LOST, SUCCESS when it succeeded, or NOT_READY when it
// cannot complete yet and consumed nothing (the worker requeues it behind the
// other pending work).
struct InstanceOperationExecution
{
    ovspatialquery_api_status_t status     = OVSPATIALQUERY_API_SUCCESS;
    const char*                 diagnostic = nullptr;
};

struct InstanceOperationReservation
{
    ovspatialquery_op_id_t operation  = OVSPATIALQUERY_INVALID_OP;
    uint32_t               slotIndex  = UINT32_MAX;
    uint32_t               generation = 0u;
    InstanceOperationKind  kind       = InstanceOperationKind::familyBatchPlan;
    const void*            owner      = nullptr;
};

/**
 * Private instance operation control plane.
 *
 * Reserving a free InstanceState slot under its operation mutex is the shared
 * capacity and ID-admission linearization point. Scheduling makes that already
 * admitted slot visible to the instance's fixed worker pool. Abandonment must
 * complete before a rejected owner admission returns. Terminal publication
 * retires worker ownership in one generation-checked operation-table/slot
 * critical section, so observation cannot race stale worker bookkeeping.
 * Destruction linearizes under the same operation mutex when the slot is
 * cleared and its generation advances; task destruction happens
 * afterward so owner locks and generation leases are never released while the
 * operation table is locked.
 */

class HostLease;

// The work a private operation's slot holds: the worker executes it, and the
// slot keeps it until destroy_op.
class OperationTask
{
  public:
    virtual ~OperationTask()                              = default;
    virtual InstanceOperationExecution execute() noexcept = 0;
    // A CUDA result retired after its consumers keeps its result slot until
    // the device retirement releases it.
    virtual void disarmForDeviceRetirement() noexcept {}
    // Publication with the terminal state: begin runs before the operation
    // table is locked (false fails the operation), end inside it.
    virtual bool beginTerminalPublication(bool executionSucceeded) noexcept
    {
        (void)executionSucceeded;
        return true;
    }
    virtual void endTerminalPublication() noexcept {}
    // A requeued task asks for `signal` to be rung when it can make progress.
    virtual void ringWhenReady(const std::shared_ptr<DeviceSignal>& signal) noexcept { (void)signal; }
    // True when a caller waiting on the operation may run it in a worker's place:
    // work with no thread affinity, such as host work or a wait for device work.
    virtual bool waiterMayRun() const noexcept { return false; }
};

ovspatialquery_api_status_t reserveInstanceOperation(ovspatialquery_instance_handle_t instance,
                                                     InstanceOperationKind kind, const void* owner,
                                                     InstanceOperationReservation& outReservation) noexcept;
// Charges a task's retained input to the host budget; false when it does not
// fit. Called with a live reservation, which keeps the instance from release.
bool reserveInstanceOperationHostTask(ovspatialquery_instance_handle_t instance, uint64_t bytes,
                                      HostLease& outLease) noexcept;

/* Binds a task to its admitted reservation; `activate` also queues it for the
 * instance's fixed worker pool. The bind/activate split prevents
 * already-admitted CUDA work from racing a fallible worker-queue allocation:
 * the owner binds its terminal waiter, submits device work, then calls
 * activateInstanceOperation. */
ovspatialquery_api_status_t bindInstanceOperation(ovspatialquery_instance_handle_t    instance,
                                                  const InstanceOperationReservation& reservation,
                                                  std::shared_ptr<OperationTask> task, HostLease&& taskLease,
                                                  bool activate) noexcept;
ovspatialquery_api_status_t activateInstanceOperation(ovspatialquery_instance_handle_t    instance,
                                                      const InstanceOperationReservation& reservation) noexcept;
ovspatialquery_api_status_t reserveInstanceDeviceResult(ovspatialquery_instance_handle_t instance,
                                                        std::shared_ptr<void>&           outReservation) noexcept;
ovspatialquery_api_status_t reserveInstanceDeviceMemory(ovspatialquery_instance_handle_t instance, DLDevice device,
                                                        uint64_t bytes, std::shared_ptr<void>& outReservation) noexcept;
void                        abandonBoundInstanceOperation(ovspatialquery_instance_handle_t instance,
                                                          InstanceOperationReservation&    reservation) noexcept;

void abandonInstanceOperation(ovspatialquery_instance_handle_t instance,
                              InstanceOperationReservation&    reservation) noexcept;

// Waits for any live operation to become terminal and marks it observed:
// SUCCESS or OPERATION_FAILED with its state (and the slot's execution
// status), TIMEOUT, or INVALID_HANDLE once the operation is gone.
ovspatialquery_api_status_t waitInstanceOperation(ovspatialquery_instance_handle_t instance,
                                                  ovspatialquery_op_id_t operation, uint64_t timeoutNanoseconds,
                                                  InstanceOperationState&      outState,
                                                  ovspatialquery_api_status_t* outExecutionStatus = nullptr) noexcept;
// The state of any live operation; a terminal one is marked observed.
ovspatialquery_api_status_t queryInstanceOperation(ovspatialquery_instance_handle_t instance,
                                                   ovspatialquery_op_id_t           operation,
                                                   InstanceOperationState&          outState) noexcept;
// Frees a terminal, observed private operation's slot. With a device
// retirement it first takes the consumer's event (zeroed on success), disarms
// the task and hands the retirement to the reaper, which ends the device work
// after the consumer.
ovspatialquery_api_status_t destroyInstanceOperation(
    ovspatialquery_instance_handle_t instance, InstanceOperationKind kind, const void* owner,
    ovspatialquery_op_id_t& operation, ovspatialquery_cuda_event_t* consumerDone = nullptr,
    std::shared_ptr<InstanceOperationDeviceRetirement> retirement = nullptr) noexcept;
// Ends, on the calling thread, every retirement of `owner` that is already
// device-safe, so its result slot is free when this returns. A plan or
// reporter calls it before refusing for want of a free slot: the reaper
// otherwise frees such a slot only after a host callback and a thread hop.
// Call it holding none of the owner's locks.
void reclaimInstanceDeviceRetirements(ovspatialquery_instance_handle_t instance, const void* owner) noexcept;

#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
ovspatialquery_api_status_t setInstanceOperationWorkersPausedForTest(ovspatialquery_instance_handle_t instance,
                                                                     bool                             paused) noexcept;
// Holds the CUDA reaper back, so only an owner's own reclaim can end a
// retirement: the reaper's slowest interleaving, made deterministic.
ovspatialquery_api_status_t setInstanceCudaReaperPausedForTest(ovspatialquery_instance_handle_t instance,
                                                               bool                             paused) noexcept;
ovspatialquery_api_status_t setInstanceOperationExecutionsPausedForTest(ovspatialquery_instance_handle_t instance,
                                                                        bool paused) noexcept;

ovspatialquery_api_status_t failNextInstanceOperationSchedulingForTest(
    ovspatialquery_instance_handle_t instance) noexcept;

ovspatialquery_api_status_t setInstanceOperationReservationPausedAfterLifecycleForTest(
    ovspatialquery_instance_handle_t instance, bool paused) noexcept;
ovspatialquery_api_status_t waitForInstanceOperationReservationAfterLifecycleForTest(
    ovspatialquery_instance_handle_t instance) noexcept;

ovspatialquery_api_status_t setInstanceOperationBeforeTerminalPausedForTest(ovspatialquery_instance_handle_t instance,
                                                                            bool paused) noexcept;
ovspatialquery_api_status_t waitForInstanceOperationBeforeTerminalForTest(
    ovspatialquery_instance_handle_t instance) noexcept;
ovspatialquery_api_status_t waitForInstanceReleaseBeforeLifecycleForTest(
    ovspatialquery_instance_handle_t instance) noexcept;

ovspatialquery_api_status_t setInstanceOperationWaiterPausedBeforeRevalidationForTest(
    ovspatialquery_instance_handle_t instance, bool paused) noexcept;
ovspatialquery_api_status_t waitForInstanceOperationWaiterAtRevalidationForTest(
    ovspatialquery_instance_handle_t instance) noexcept;
uint64_t instanceOperationWaiterEntryCountForTest(ovspatialquery_instance_handle_t instance) noexcept;

struct InstanceOperationRuntimeInspection
{
    uint32_t pendingOperationCount     = 0u;
    uint32_t runningOperationCount     = 0u;
    uint32_t peakRunningOperationCount = 0u;
};

ovspatialquery_api_status_t inspectInstanceOperationRuntimeForTest(
    ovspatialquery_instance_handle_t instance, InstanceOperationRuntimeInspection& outInspection) noexcept;
#endif

} // namespace detail
} // namespace ovspatialquery

#endif
