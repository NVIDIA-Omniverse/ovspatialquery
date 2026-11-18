/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_INSTANCE_STATE_H
#define OVSPATIALQUERY_INSTANCE_STATE_H

#include "ovspatialquery_backend.h"
#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_device_signal.h"
#include "ovspatialquery_explicit_preparation.h"
#include "ovspatialquery_explicit_registration_operation.h"
#include "ovspatialquery_family_batch_plan_operation.h"
#include "ovspatialquery_handle_directory.h"
#include "ovspatialquery_input.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_instance_source.h"
#include "ovspatialquery_pair_reporter_operation.h"
#include "ovspatialquery_population_table.h"
#include "ovspatialquery_prepared_array.h"
#include "ovspatialquery_target_resolver.h"
#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

// Declared ahead of InstanceState because its constructor seeds the salt.
uint32_t nextInstanceOperationIdSalt() noexcept;

} // namespace detail
} // namespace ovspatialquery

// Shared by the family-plan and pair-reporter storage-charge helpers in
// ovspatialquery_current_api_family.cpp and ovspatialquery_current_api_pair.cpp.
constexpr uint64_t kApiSharedControlAllowance = 3u * sizeof(void*);
constexpr uint64_t kApiDirectoryNodeAllowance = sizeof(uint64_t) + sizeof(std::shared_ptr<void>) + 16u * sizeof(void*);

enum class ApiFamily : uint8_t
{
    raycast,
    sphereSweep,
    boxSweep,
    overlap,
};

// A CPU plan's result slot holds its family's batch result; box sweeps share
// the sphere-sweep result type.
using ApiFamilyProjectionPayload =
    std::variant<std::monostate, ovspatialquery_raycast_batch_result_t, ovspatialquery_sphere_sweep_batch_result_t,
                 ovspatialquery_overlap_batch_result_t>;

struct ApiFamilyProjection
{
    std::mutex                 mutex;
    ApiFamilyProjectionPayload payload;
    bool                       ready = false;
};

struct ApiReporterProjection
{
    std::mutex                         mutex;
    uint32_t                           resultSlot              = UINT32_MAX;
    uint32_t                           outcome                 = 0u;
    ovspatialquery_ordinal_t           firstSourceOrdinal      = 0;
    ovspatialquery_ordinal_t           secondSourceOrdinal     = 0;
    uint64_t                           acceptedActivePairCount = 0u;
    std::vector<ovspatialquery_pair_t> entered;
    std::vector<ovspatialquery_pair_t> exited;
    std::array<int64_t, 2u>            enteredShape{0, 2};
    std::array<int64_t, 2u>            enteredStrides{2, 1};
    std::array<int64_t, 2u>            exitedShape{0, 2};
    std::array<int64_t, 2u>            exitedStrides{2, 1};
    // A CUDA reporter's rows stay on its device; the reservation describes
    // the whole capacity-shaped slot and its counts (ADR-0022 decision 2).
    DLDevice                    device{kDLCPU, 0};
    const void*                 deviceEntered = nullptr;
    const void*                 deviceExited  = nullptr;
    const void*                 deviceCounts  = nullptr;
    std::array<int64_t, 2u>     reservedShape{0, 2};
    std::array<int64_t, 1u>     countsShape{4};
    std::array<int64_t, 1u>     countsStrides{1};
    ovspatialquery_cuda_event_t resultReady{};
    bool                        ready = false;
};

struct ApiFamilyPlanState
{
    // The top byte of every batch-plan handle.
    static constexpr uint64_t handleKind = UINT64_C(0xb100000000000000);
    // Declared first so the owner and its fixed storage are destroyed before
    // the ledger lease is released.
    std::shared_ptr<ovspatialquery::detail::HostLease> hostLease;
    std::shared_ptr<void>                              viewRetention;
    ApiFamily                                          family            = ApiFamily::raycast;
    uint64_t                                           maximumQueryCount = 0u;
    // CPU plans only: the plan-fixed values a row takes unless its batch
    // carries its own. overlapShapeKind is the public
    // ovspatialquery_overlap_shape_kind_t value (0 sphere, 1 box, 2 AABB); a
    // batch supplies rotations only for box. radius serves a sphere sweep or
    // sphere overlap, halfExtents a box sweep or box/AABB overlap.
    uint32_t                                          overlapShapeKind = 0u;
    double                                            radius           = 0.0;
    std::array<double, 3u>                            halfExtents{};
    std::mutex                                        adapterMutex;
    uint32_t                                          activeWaitCount = 0u;
    ovspatialquery::detail::RaycastBatchLaunch        raycastScratch;
    ovspatialquery::detail::SphereSweepBatchLaunch    sweepScratch;
    ovspatialquery::detail::OverlapBatchLaunch        overlapScratch;
    std::vector<std::shared_ptr<ApiFamilyProjection>> projections;
    // CUDA plans also retain the device reservation they take, so the public
    // plan lifetime cannot release device storage under the owner.
    bool                                                                   cudaPlan     = false;
    int32_t                                                                cudaDeviceId = -1;
    std::shared_ptr<void>                                                  cudaTargetRetention;
    std::unique_ptr<ovspatialquery::detail::FamilyBatchPlanOperationOwner> owner;
};

struct ApiPairReporterState
{
    // The top byte of every pair-reporter handle.
    static constexpr uint64_t handleKind = UINT64_C(0xc100000000000000);
    // Declared first so the owner, result slots, and baseline are destroyed
    // before the ledger lease is released.
    std::shared_ptr<ovspatialquery::detail::HostLease> hostLease;
    std::shared_ptr<ovspatialquery::detail::HostLease> capacityHostLease;
    std::array<std::shared_ptr<void>, 2u>              viewRetentions;
    uint64_t                                           activePairCapacity = 0u;
    uint64_t                                           deltaPairCapacity  = 0u;
    uint32_t                                           resultSlotCount    = 0u;
    // A CUDA result device and the device bytes the reporter keeps there.
    DLDevice                                                            resultDevice{kDLCPU, 0};
    std::shared_ptr<void>                                               deviceLease;
    std::mutex                                                          adapterMutex;
    uint32_t                                                            activeWaitCount = 0u;
    std::vector<std::shared_ptr<ApiReporterProjection>>                 projections;
    std::unique_ptr<ovspatialquery::detail::PairReporterOperationOwner> owner;
};

struct ApiOperationOwner
{
    std::shared_ptr<ApiFamilyPlanState>   plan;
    std::shared_ptr<ApiPairReporterState> reporter;
    std::shared_ptr<void>                 projection;
};

// Both set by reserveInstanceDeviceResult, its only producer, before it is shared.
struct DeviceResultReservation
{
    std::atomic<size_t>*                        liveCount = nullptr;
    ovspatialquery::detail::OperationCondition* condition = nullptr;

    ~DeviceResultReservation()
    {
        liveCount->fetch_sub(1u);
        condition->notify_all();
    }
};

enum class OperationKind : uint32_t
{
    populationUpdate            = 1u,
    familyBatchPlanPrivate      = 7u,
    pairReporterPrivate         = 8u,
    explicitRegistrationPrivate = 9u,
};

/* Independent tasks a synchronous caller or a running CPU job runs with the help of idle
 * workers (InstanceState::runSharedWork): the caller and every helping worker claim indices
 * from `next` below `count`; `helping` counts workers inside a task and is guarded
 * by operationMutex. */
struct SharedWork
{
    void (*task)(void* state, size_t index) = nullptr;
    void*               state               = nullptr;
    size_t              count               = 0u;
    std::atomic<size_t> next{0u};
    uint32_t            helping = 0u;
};

struct InstanceState;

// A population update: the worker materializes its input against its base,
// or observes the device outcome of a version admitted at enqueue, and
// publishes the generation. Its retained input and staging charges last until
// it has run; the generation takes its device publication charge and, from a
// host-prepared input, its host charge out of the retained input charge.
struct PopulationUpdateTask final : ovspatialquery::detail::OperationTask
{
    explicit PopulationUpdateTask(InstanceState& owner) noexcept : instance(owner) {}
    ovspatialquery::detail::InstanceOperationExecution execute() noexcept override;
    void ringWhenReady(const std::shared_ptr<ovspatialquery::detail::DeviceSignal>& signal) noexcept override;
    // A host-prepared publication has no device work to observe.
    bool waiterMayRun() const noexcept override
    {
        return input.cudaCapture() == nullptr && !admittedGeneration.initialized();
    }

    InstanceState&                    instance;
    ovspatialquery::detail::HostLease retainedInputLease;
    ovspatialquery::detail::HostLease stagingHostLease;
    // Operation-owned result storage and its charge survive completion until destroy_op.
    ovspatialquery::detail::HostLease        exclusionLease;
    ovspatialquery::detail::SourceExclusions exclusions;
    int64_t                                  exclusionShape = 0;
    // The Mesh objects this publication read that dropped triangles (ADR-0073),
    // operation-owned like the exclusions and charged to droppedTriangleLease.
    ovspatialquery::detail::HostLease               droppedTriangleLease;
    ovspatialquery::detail::SourceDroppedTriangles  droppedTriangles;
    int64_t                                         droppedTriangleShape = 0;
    std::shared_ptr<void>                           stagingDeviceLease;
    std::shared_ptr<void>                           publishedDeviceLease;
    ovspatialquery::detail::PopulationKey           population;
    ovspatialquery_ordinal_t                        ordinal = 0u;
    ovspatialquery::detail::PreparedPopulationInput input;
    ovspatialquery::detail::GenerationLease         source;
    // A device-prepared version's generation, built and made selectable at
    // admission (ADR-0021); the worker publishes it once the device work ends.
    ovspatialquery::detail::GenerationLease admittedGeneration;
    // Kept past the input's release so a publication that did not become the
    // head can settle the outcome the next device publication waits on.
    std::shared_ptr<ovspatialquery::detail::CudaPopulationCapture>    endedCapture;
    uint64_t                                                          indexedCount  = 0u;
    uint64_t                                                          excludedCount = 0u;
    bool                                                              hasCounts     = false;
    ovspatialquery_population_update_result_t                         result{};
    std::vector<ovspatialquery::detail::SourceViewRequest>            viewRequests;
    std::vector<ovspatialquery::detail::PreparedSourceViewMembership> viewMemberships;
};

struct OperationSlot
{
    std::mutex                  mutex;
    std::condition_variable     condition;
    uint32_t                    generation       = 1u;
    bool                        inUse            = false;
    bool                        terminal         = false;
    bool                        terminalObserved = false;
    bool                        workerActive     = false;
    OperationKind               kind             = OperationKind::populationUpdate;
    ovspatialquery_api_status_t executionStatus  = OVSPATIALQUERY_API_SUCCESS;
    const void*                 privateOwner     = nullptr;
    ovspatialquery_api_status_t privateFailure   = OVSPATIALQUERY_API_SUCCESS;
    // Every failing execution names a string literal.
    const char* privateDiagnostic = nullptr;
    // A batch-plan or pair-reporter operation's API owner, set once its
    // enqueue is admitted and released when the slot is destroyed.
    ApiOperationOwner apiOwner;
    // Declared before the charged tasks so fallback destruction also retires
    // task/control-block and generation owners before releasing ledger charge.
    ovspatialquery::detail::HostLease                      taskLease;
    std::shared_ptr<ovspatialquery::detail::OperationTask> task;
    // A device population update's definition references; destroy_op
    // releases them with the slot.
    std::vector<std::shared_ptr<const void>> updateDefinitionOwners;
};

struct BackendRuntimeLease
{
    ovspatialquery_backend_runtime_t* runtime             = nullptr;
    bool                              releaseWithInstance = false;

    ~BackendRuntimeLease()
    {
        if (releaseWithInstance)
        {
            ovspatialquery::detail::releaseDefaultBackendRuntime(runtime);
        }
    }
};

class InstanceHostBudgetExceeded final : public std::exception
{
  public:
    const char* what() const noexcept override { return "instance fixed host footprint exceeds the host budget"; }
};

// Waits until `ready` holds or `timeoutNs` passes (OVSPATIALQUERY_TIMEOUT_INFINITE
// never passes), in waits of at most a day so no clock overflows. False on
// timeout.
template <typename Ready>
bool waitUntil(std::condition_variable& condition, std::unique_lock<std::mutex>& lock, uint64_t timeoutNs, Ready ready)
{
    if (timeoutNs == OVSPATIALQUERY_TIMEOUT_INFINITE)
    {
        condition.wait(lock, ready);
        return true;
    }
    const uint64_t maximumChunk =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(24)).count());
    uint64_t remaining = timeoutNs;
    while (!ready())
    {
        const uint64_t chunk = std::min(remaining, maximumChunk);
        if (condition.wait_for(lock, std::chrono::nanoseconds(static_cast<int64_t>(chunk)), ready))
        {
            return true;
        }
        remaining -= chunk;
        if (remaining == 0u)
        {
            return false;
        }
    }
    return true;
}

struct InstanceState
{
    // Lock order is populationCompletionMutex -> PopulationTable internal
    // mutex, then OperationSlot::mutex after the table call returns; the table
    // mutex and slot mutex are never held together. operationMutex precedes
    // OperationSlot::mutex on the separate operation-table path. Backend
    // construction, traversal, generation destruction, and result allocation
    // happen outside these locks.
    InstanceState(ovspatialquery_backend_runtime_t* runtime, uint32_t operationCapacity,
                  uint32_t configuredDeviceResultCapacity, uint32_t configuredMaxCpuConcurrency,
                  uint64_t                                          configuredHostByteBudget,
                  const ovspatialquery::detail::SourceDeviceBudget* configuredDeviceBudgets,
                  uint32_t                                          configuredDeviceBudgetCount)
        : runtimeLease{runtime, false}, hostByteBudget(configuredHostByteBudget), hostLedger(configuredHostByteBudget),
          explicitPreparationRegistry(std::make_shared<ovspatialquery::detail::ExplicitPreparationRegistry>(
              runtime, configuredHostByteBudget, geometryReferenceClaims)),
          populations(runtime), deviceResultCapacity(configuredDeviceResultCapacity),
          maxCpuConcurrency(configuredMaxCpuConcurrency),
          operationIdSalt(ovspatialquery::detail::nextInstanceOperationIdSalt())
    {
        deviceBudgets.reserve(configuredDeviceBudgetCount);
        deviceUsage.reserve(configuredDeviceBudgetCount);
        for (uint32_t index = 0u; index < configuredDeviceBudgetCount; ++index)
        {
            deviceBudgets.emplace_back(configuredDeviceBudgets[index].device,
                                       configuredDeviceBudgets[index].byteBudget);
            deviceUsage.push_back(0u);
        }
        operationSlots.reserve(operationCapacity);
        pendingOperations.reserve(operationCapacity);
        retiredPrivateDeviceOperations.reserve(configuredDeviceResultCapacity);
        privateDeviceReaperOperations.reserve(configuredDeviceResultCapacity);
        privateDeviceReaperCompleted.reserve(configuredDeviceResultCapacity);
        for (uint32_t index = 0u; index < operationCapacity; ++index)
        {
            operationSlots.push_back(std::make_unique<OperationSlot>());
        }
        workers.reserve(maxCpuConcurrency);
        uint64_t fixedHostCharge = 0u;
        if (!retainedFixedHostCharge(fixedHostCharge) || !hostLedger.reserve(fixedHostCharge))
        {
            throw InstanceHostBudgetExceeded{};
        }
        try
        {
            for (uint32_t workerIndex = 0u; workerIndex < maxCpuConcurrency; ++workerIndex)
            {
                workers.emplace_back([this]() { runWorker(); });
            }
            if (deviceResultCapacity != 0u)
            {
                cudaReaper = std::thread([this]() { runCudaReaper(); });
            }
        }
        catch (...)
        {
            stopThreads();
            throw;
        }
    }

    ~InstanceState() { stopThreads(); }

    // Stops the workers and the CUDA reaper and joins them.
    void stopThreads() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(operationMutex);
            stopping = true;
        }
        operationCondition.notify_all();
        cudaReaperCondition.notify_all();
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        instanceOperationExecutionCondition.notify_all();
#endif
        if (cudaReaper.joinable())
        {
            cudaReaper.join();
        }
        for (std::thread& worker : workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    std::unique_lock<std::mutex> acquirePopulationCompletion() const
    {
        return std::unique_lock<std::mutex>(populationCompletionMutex);
    }

    // How a dequeued operation ended: terminal, or requeued with or without other work queued.
    enum class QueuedRun
    {
        completed,
        deferredAlone,
        deferredBehindWork,
    };
    void      runWorker();
    bool      canStartQueuedOperation() const noexcept;
    QueuedRun runQueuedOperation(uint32_t slotIndex, uint32_t operationGeneration) noexcept;
    bool      runQueuedOperationOnWaiter(uint32_t slotIndex, uint32_t generation) noexcept;
    void      helpSharedWork(std::unique_lock<std::mutex>& lock);
    void      runSharedWork(size_t count, void (*task)(void* state, size_t index), void* taskState) noexcept;
    // This instance's workers lent through runSharedWork, for a caller that holds the instance.
    ovspatialquery::detail::SharedWorkerLoan sharedWorkerLoan() noexcept
    {
        return {this,
                [](void* state, size_t count, void (*task)(void* taskState, size_t index), void* taskState) noexcept
                { static_cast<InstanceState*>(state)->runSharedWork(count, task, taskState); },
                [](void* state) noexcept
                {
                    const InstanceState& instance = *static_cast<const InstanceState*>(state);
                    const uint32_t       busy     = instance.workerJobs.load();
                    return busy < instance.workers.size() ? static_cast<uint32_t>(instance.workers.size()) - busy : 0u;
                }};
    }
    void runCudaReaper();
    // Ends the device-safe retirements of `owner`, or of every owner when it is
    // null; true when it ended one. The reaper and an owner short of result
    // slots both call it, one at a time.
    bool reclaimDeviceRetirements(const void* owner) noexcept;

    static bool minimumFixedHostCharge(uint32_t operationCapacity, uint32_t deviceResultCapacity,
                                       uint32_t maxCpuConcurrency, uint32_t deviceBudgetCount,
                                       uint64_t& outCharge) noexcept;
    bool        retainedFixedHostCharge(uint64_t& outCharge) const noexcept;

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    std::atomic<bool> failNextTypedGeometryAggregateStaging{false};
#endif
    // Declared before populations so the private backend remains alive until
    // every population generation has been destroyed.
    BackendRuntimeLease runtimeLease;
    // Declared before every instance-owned object that may retain a HostLease;
    // reverse destruction therefore releases owners/storage before charges.
    const uint64_t                             hostByteBudget;
    ovspatialquery::detail::InstanceHostLedger hostLedger;
    // Declared immediately after the ledger so instance-owned population/view
    // resources are destroyed before their persistent leases on teardown.
    std::unordered_map<ovspatialquery_population_handle_t, std::shared_ptr<ovspatialquery::detail::HostLease>>
        populationCharges;
    std::unordered_map<ovspatialquery_view_handle_t, std::shared_ptr<ovspatialquery::detail::HostLease>> viewCharges;
    std::shared_ptr<ovspatialquery::detail::GeometryReferenceClaimDirectory> geometryReferenceClaims =
        std::make_shared<ovspatialquery::detail::GeometryReferenceClaimDirectory>();
    ovspatialquery::detail::GeometryIdentityRegistry geometryIdentityRegistry{geometryReferenceClaims};
    std::shared_ptr<ovspatialquery::detail::ExplicitPreparationRegistry> explicitPreparationRegistry;
    ovspatialquery::detail::PopulationTable                              populations;
    ovspatialquery::detail::HandleDirectory                              handles;
    const uint32_t                                                       deviceResultCapacity;
    const uint32_t                                                       maxCpuConcurrency;
    const uint32_t                                                       operationIdSalt;
    std::mutex                                                           lifecycleMutex;
    mutable std::mutex                                                   populationCompletionMutex;
    std::mutex                                                           operationMutex;
    ovspatialquery::detail::OperationCondition                           operationCondition;
    std::vector<std::unique_ptr<OperationSlot>>                          operationSlots;
    std::vector<uint32_t>                                                pendingOperations;
    // CPU jobs executing on workers or waiting callers; at most maxCpuConcurrency.
    uint32_t runningCpuJobs = 0u;
    // Operations running on workers, not on waiting callers; read without the mutex.
    std::atomic<uint32_t>    workerJobs{0u};
    std::vector<SharedWork*> pendingSharedWork;
    std::condition_variable  sharedWorkCondition;
    std::vector<std::thread> workers;
    std::thread              cudaReaper;
    // Wakes a reaper that has nothing retired: a retirement or stopping, not every
    // host enqueue and completion that operationCondition announces.
    std::condition_variable cudaReaperCondition;
    // Held across a whole reclaim pass, so a retirement taken out of
    // retiredPrivateDeviceOperations has no other reference left and its
    // destructor, which frees the result slot, runs before the pass returns.
    std::mutex retirementReclaimMutex;
    std::vector<std::shared_ptr<ovspatialquery::detail::InstanceOperationDeviceRetirement>>
        retiredPrivateDeviceOperations;
    std::vector<std::shared_ptr<ovspatialquery::detail::InstanceOperationDeviceRetirement>>
        privateDeviceReaperOperations;
    std::vector<std::shared_ptr<ovspatialquery::detail::InstanceOperationDeviceRetirement>>
           privateDeviceReaperCompleted;
    size_t liveOperationCount = 0u;
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
    // While set, a reaper pass ends no retirement (until the instance stops).
    std::atomic<bool> cudaReaperPaused{false};

    bool instanceOperationWorkersPaused    = false;
    bool instanceOperationExecutionsPaused = false;
    // A thread holding an operation while executions are paused waits here, with
    // operationMutex, so a wake meant for one idle worker cannot land on it.
    std::condition_variable instanceOperationExecutionCondition;
    bool                    failNextInstanceOperationScheduling = false;
    uint32_t                runningInstanceOperationCount       = 0u;
    uint32_t                peakRunningInstanceOperationCount   = 0u;
    std::mutex              instanceOperationTestMutex;
    std::condition_variable instanceOperationTestCondition;
    bool                    pauseInstanceOperationWaiterBeforeRevalidation  = false;
    bool                    instanceOperationWaiterAtRevalidation           = false;
    uint64_t                instanceOperationWaiterEntryCount               = 0u;
    bool                    pauseInstanceOperationReservationAfterLifecycle = false;
    bool                    instanceOperationReservationAfterLifecycle      = false;
    bool                    pauseInstanceOperationBeforeTerminal            = false;
    bool                    instanceOperationBeforeTerminal                 = false;
    bool                    instanceReleaseBeforeLifecycle                  = false;
#endif
    std::atomic<size_t>                                                         liveDeviceResultCount{0u};
    mutable std::mutex                                                          budgetMutex;
    std::mutex                                                                  explicitRuntimeMutex;
    std::unique_ptr<ovspatialquery::detail::ExplicitRegistrationOperationOwner> explicitRegistrationOwner;
    std::vector<std::pair<DLDevice, uint64_t>>                                  deviceBudgets;
    std::vector<uint64_t>                                                       deviceUsage;
    // Finds the device copies of mesh definitions this instance's CUDA plans share.
    ovspatialquery::detail::CudaMeshResidencyRegistry cudaMeshResidency;
    // Keeps the CUDA module of every budgeted device a request named loaded for
    // the instance's life, so its CUDA users load the PTX once per device.
    ovspatialquery::detail::CudaModulePins cudaModules;
    ovspatialquery_instance_handle_t       publicHandle = OVSPATIALQUERY_INVALID_INSTANCE;
    bool                                   released     = false;
    bool                                   stopping     = false;

    bool reserveDevice(DLDevice device, uint64_t bytes)
    {
        std::lock_guard<std::mutex> lock(budgetMutex);
        for (size_t index = 0u; index < deviceBudgets.size(); ++index)
        {
            if (deviceBudgets[index].first.device_type != device.device_type ||
                deviceBudgets[index].first.device_id != device.device_id)
            {
                continue;
            }
            if (bytes > deviceBudgets[index].second - std::min(deviceBudgets[index].second, deviceUsage[index]))
            {
                return false;
            }
            deviceUsage[index] += bytes;
            return true;
        }
        return false;
    }

    // Where the instance keeps CUDA device `deviceId`'s module: only on a device it
    // has a budget for, since the module and the primary context it retains take
    // device memory outside every budget. Null for any other device.
    ovspatialquery::detail::CudaModulePins* cudaModulePins(int32_t deviceId)
    {
        std::lock_guard<std::mutex> lock(budgetMutex);
        for (const std::pair<DLDevice, uint64_t>& budget : deviceBudgets)
        {
            if (budget.first.device_type == kDLCUDA && budget.first.device_id == deviceId)
            {
                return &cudaModules;
            }
        }
        return nullptr;
    }

    void releaseDevice(DLDevice device, uint64_t bytes)
    {
        std::lock_guard<std::mutex> lock(budgetMutex);
        for (size_t index = 0u; index < deviceBudgets.size(); ++index)
        {
            if (deviceBudgets[index].first.device_type == device.device_type &&
                deviceBudgets[index].first.device_id == device.device_id)
            {
                deviceUsage[index] -= std::min(bytes, deviceUsage[index]);
                return;
            }
        }
    }
};

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t reserveSharedHostLease(InstanceState* instance, uint64_t bytes,
                                                   std::shared_ptr<ovspatialquery::detail::HostLease>& outLease);

class InstanceDirectory
{
  public:
    ovspatialquery_api_status_t publish(const std::shared_ptr<InstanceState>& instance,
                                        ovspatialquery_instance_handle_t&     outHandle)
    {
        outHandle = OVSPATIALQUERY_INVALID_INSTANCE;
        try
        {
            std::lock_guard<std::mutex> lock(mMutex);
            // Handles count up from one and are never reused.
            if (mNextHandle >= ovspatialquery::detail::kHandlePayloadLimit)
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
            const ovspatialquery_instance_handle_t handle{mNextHandle++};
            mInstances.emplace(handle, instance);
            instance->publicHandle = handle;
            outHandle              = handle;
            return OVSPATIALQUERY_API_SUCCESS;
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (...)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
    }

    // The instance a handle names, or null; no instance holds the invalid handle.
    std::shared_ptr<InstanceState> resolve(ovspatialquery_instance_handle_t handle) const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        const auto                  found = mInstances.find(handle);
        return found != mInstances.end() ? found->second : nullptr;
    }

    // Called once per instance, by its release under the lifecycle mutex.
    void erase(ovspatialquery_instance_handle_t handle)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mInstances.erase(handle);
    }

  private:
    mutable std::mutex                                                                   mMutex;
    std::unordered_map<ovspatialquery_instance_handle_t, std::shared_ptr<InstanceState>> mInstances;
    uint64_t                                                                             mNextHandle = 1u;
};

InstanceDirectory& instanceDirectory();

class InstanceReference
{
  public:
    InstanceReference() = default;
    explicit InstanceReference(std::shared_ptr<InstanceState> instance) : mInstance(std::move(instance)) {}

    InstanceState* operator->() const { return mInstance.get(); }

    operator InstanceState*() const { return mInstance.get(); }

    bool operator==(std::nullptr_t) const { return mInstance == nullptr; }

    const std::shared_ptr<InstanceState>& shared() const { return mInstance; }

  private:
    std::shared_ptr<InstanceState> mInstance;
};

InstanceReference instanceState(ovspatialquery_instance_handle_t handle);

// An operation ID is its slot index plus one in the low half and the slot
// generation XOR the instance's salt in the high half. Salts of distinct
// instances differ in a high bit, so an ID from another instance decodes to a
// generation its slot reaches only after millions of reuses; the check is
// best effort, since a 64-bit ID cannot name the instance as well.
ovspatialquery_op_id_t makeOperationId(const InstanceState* instance, uint32_t slotIndex, uint32_t generation);

bool decodeOperationId(const InstanceState* instance, ovspatialquery_op_id_t operation, uint32_t& outSlotIndex,
                       uint32_t& outGeneration);

OperationSlot* resolveOperation(InstanceState* instance, ovspatialquery_op_id_t operation, uint32_t& outSlotIndex,
                                uint32_t& outGeneration);

// The API owner of a live operation, or none.
ApiOperationOwner apiOperationOwner(InstanceState& instance, ovspatialquery_op_id_t operation);
// Records the API owner of an operation its enqueue just admitted.
void apiSetOperationOwner(InstanceState& instance, ovspatialquery_op_id_t operation, ApiOperationOwner owner);

ovspatialquery_api_status_t reserveOperation(InstanceState* instance, OperationKind kind, uint32_t& outSlotIndex,
                                             ovspatialquery_op_id_t& outOperation);

// Charges `bytes` to the host budget; false when they do not fit.
bool reserveHostLease(InstanceState& instance, uint64_t bytes, ovspatialquery::detail::HostLease& outLease) noexcept;

// The host bytes a publication of `input` on `base` charges at admission. A
// host-prepared input's charge also covers the generation built from it, with
// `membershipBytes` of view memberships, which the worker takes out of it.
uint64_t publicationAdmissionCharge(ovspatialquery::detail::PreparedPopulationInput& input,
                                    const ovspatialquery::detail::GenerationLease& base, uint64_t membershipBytes);

void abandonReservedOperation(InstanceState* instance, uint32_t slotIndex);

void enqueueOperation(InstanceState* instance, uint32_t slotIndex);

// Shared by ovspatialquery_current_api_query.cpp,
// ovspatialquery_current_api_family.cpp, ovspatialquery_current_api_pair.cpp,
// and ovspatialquery_current_api_operation.cpp: state-mapping and
// target-resolution glue used by more than one of the four service-ABI
// bodies files. Each is defined exactly once, in
// ovspatialquery_current_api_operation.cpp, except the two "materialize"
// functions below, whose bodies stay next to the owners they read
// (ovspatialquery_current_api_family.cpp and
// ovspatialquery_current_api_pair.cpp respectively) because wait_op and
// query_op_status (in the operation file) need to call into both.
ovspatialquery::detail::PopulationTarget apiPlanTarget(const InstanceReference&           instance,
                                                       const ovspatialquery_target_ref_t& source, bool pairRetention,
                                                       std::shared_ptr<void>&       outViewRetention,
                                                       ovspatialquery_api_status_t& outStatus);

ovspatialquery::detail::TargetSelection apiSelection(const ovspatialquery_ordinal_selection_t& source, bool& valid);

#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
void apiFetchProjectionPausePointForTest(uint32_t kind);
#endif

ovspatialquery_api_status_t apiMaterializeFamilyLocked(const ApiOperationOwner& owner,
                                                       ovspatialquery_op_id_t   operationId);

ovspatialquery_api_status_t apiMaterializeReporterLocked(const ApiOperationOwner& owner,
                                                         ovspatialquery_op_id_t   operationId);

// Batch plans and pair reporters share their handle allocation, lookup and
// teardown; these templates are defined here for both files.
// Publishes an adapter in the instance's handle directory under a
// process-unique handle of its kind: the kind in the top byte and a counter
// below it; INTERNAL once the counter is spent.
template <typename Adapter>
ovspatialquery_api_status_t apiPublishAdapter(InstanceState& instance, std::shared_ptr<Adapter> adapter,
                                              uint64_t& outHandle)
{
    uint64_t handle = 0u;
    {
        static std::mutex           mutex;
        static uint64_t             next = 1u;
        std::lock_guard<std::mutex> lock(mutex);
        if (next > UINT64_C(0x00ffffffffffffff))
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        handle = Adapter::handleKind | next++;
    }
    instance.handles.insertAdapter(handle, std::move(adapter));
    outHandle = handle;
    return OVSPATIALQUERY_API_SUCCESS;
}

// The plan or pair reporter a handle names in the instance, or null.
template <typename Adapter>
std::shared_ptr<Adapter> apiFindAdapter(InstanceState& instance, uint64_t handle)
{
    return (handle & UINT64_C(0xff00000000000000)) == Adapter::handleKind
               ? std::static_pointer_cast<Adapter>(instance.handles.findAdapter(handle))
               : nullptr;
}

// Serves destroy_batch_plan and destroy_pair_reporter: BUSY while a wait
// holds the adapter, else the owner's destroy status; on success the handle
// leaves the instance.
template <typename Adapter>
ovspatialquery_result_t apiDestroyAdapter(ovspatialquery_instance_handle_t instanceHandle, uint64_t handle)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    const std::shared_ptr<Adapter> adapter = apiFindAdapter<Adapter>(*instance, handle);
    if (adapter == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    ovspatialquery_api_status_t status = OVSPATIALQUERY_API_INVALID_HANDLE;
    {
        std::lock_guard<std::mutex> adapterLock(adapter->adapterMutex);
        if (adapter->activeWaitCount != 0u)
        {
            status = OVSPATIALQUERY_API_BUSY;
        }
        else if (adapter->owner != nullptr)
        {
            status = std::decay_t<decltype(*adapter->owner)>::destroy(adapter->owner);
        }
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(status);
    }
    instance->handles.eraseAdapter(handle);
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

ovspatialquery_api_status_t submitPopulationInput(InstanceState*                                  instance,
                                                  ovspatialquery::detail::PopulationKey           population,
                                                  ovspatialquery::detail::PreparedPopulationInput input,
                                                  ovspatialquery_op_id_t&                         outOperation,
                                                  ovspatialquery::detail::SourceDroppedTriangles droppedTriangles = {});

// Moves a publication's dropped-triangle list into its task under a host charge.
bool adoptDroppedTriangles(InstanceState& instance, PopulationUpdateTask& update,
                           ovspatialquery::detail::SourceDroppedTriangles&& droppedTriangles);

} // namespace detail
} // namespace ovspatialquery

// Release paths shared by the exported destroy functions and the source
// binding error paths (ovspatialquery_internal_api.cpp).
extern "C" ovspatialquery_api_status_t ovspatialquery_release_instance(ovspatialquery_instance_handle_t instance);
extern "C" ovspatialquery_api_status_t ovspatialquery_release_population(ovspatialquery_instance_handle_t   instance,
                                                                         ovspatialquery_population_handle_t population);
extern "C" ovspatialquery_api_status_t ovspatialquery_operation_release(ovspatialquery_instance_handle_t instance,
                                                                        ovspatialquery_op_id_t*          operation);

#endif
