/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_family_batch_plan_operation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ovspatialquery
{
namespace detail
{
namespace
{

bool validPlan(const RaycastBatchPlan& plan)
{
    return (plan.precision == QueryPrecision::boundsCandidate || plan.precision == QueryPrecision::exact) &&
           (plan.hitMode == HitMode::closest || plan.hitMode == HitMode::any || plan.hitMode == HitMode::multiple) &&
           std::isfinite(plan.contactTolerance) && plan.contactTolerance >= 0.0;
}

bool validPlan(const SphereSweepBatchPlan& plan)
{
    return (plan.precision == QueryPrecision::boundsCandidate || plan.precision == QueryPrecision::exact) &&
           (plan.hitMode == HitMode::closest || plan.hitMode == HitMode::any || plan.hitMode == HitMode::multiple) &&
           std::isfinite(plan.contactTolerance) && plan.contactTolerance >= 0.0;
}

bool validPlan(const OverlapBatchPlan& plan)
{
    return (plan.precision == QueryPrecision::boundsCandidate || plan.precision == QueryPrecision::exact) &&
           (plan.hitMode == HitMode::any || plan.hitMode == HitMode::multiple) &&
           std::isfinite(plan.contactTolerance) && plan.contactTolerance >= 0.0;
}

struct RaycastPlanPolicy
{
    using Plan              = RaycastBatchPlan;
    using Launch            = RaycastBatchLaunch;
    using Row               = RaycastBatchRow;
    using Result            = RaycastBatchResult;
    using Hit               = RaycastSemanticHit;
    using ColumnStorage     = RaycastBatchColumnStorage;
    using PublicReservation = ovspatialquery_raycast_batch_reservation_t;
    using PublicResult      = ovspatialquery_raycast_batch_result_t;
};

struct SphereSweepPlanPolicy
{
    using Plan              = SphereSweepBatchPlan;
    using Launch            = SphereSweepBatchLaunch;
    using Row               = SphereSweepBatchRow;
    using Result            = SphereSweepBatchResult;
    using Hit               = SphereSweepSemanticHit;
    using ColumnStorage     = SphereSweepBatchColumnStorage;
    using PublicReservation = ovspatialquery_sphere_sweep_batch_reservation_t;
    using PublicResult      = ovspatialquery_sphere_sweep_batch_result_t;
};

struct OverlapPlanPolicy
{
    using Plan              = OverlapBatchPlan;
    using Launch            = OverlapBatchLaunch;
    using Row               = OverlapBatchRow;
    using Result            = OverlapBatchResult;
    using Hit               = OverlapSemanticHit;
    using ColumnStorage     = OverlapBatchColumnStorage;
    using PublicReservation = ovspatialquery_overlap_batch_reservation_t;
    using PublicResult      = ovspatialquery_overlap_batch_result_t;
};

template <typename Policy>
bool validResultFields(uint32_t resultFields, QueryPrecision precision) noexcept
{
    uint32_t known = kBatchResultGeometryIdentity | kBatchResultMaterial;
    if constexpr (std::is_same_v<Policy, RaycastPlanPolicy>)
    {
        known |= kBatchResultPosition | kBatchResultNormal | kBatchResultDistance | kBatchResultSourceFeature;
    }
    else if constexpr (std::is_same_v<Policy, SphereSweepPlanPolicy>)
    {
        known |= kBatchResultPosition | kBatchResultNormal | kBatchResultDistance | kBatchResultSourceFeature |
                 kBatchResultQueryPosition;
    }
    return (resultFields & ~known) == 0u &&
           (precision == QueryPrecision::exact || (resultFields & kBatchResultSourceFeature) == 0u);
}

template <typename Policy>
struct TypedPlanState
{
    using PolicyType = Policy;
    typename Policy::Plan plan;
};

using PlanState   = std::variant<TypedPlanState<RaycastPlanPolicy>, TypedPlanState<SphereSweepPlanPolicy>,
                               TypedPlanState<OverlapPlanPolicy>>;
using LaunchState = std::variant<RaycastBatchLaunch, SphereSweepBatchLaunch, OverlapBatchLaunch>;
using ResultState = std::variant<RaycastBatchResult, SphereSweepBatchResult, OverlapBatchResult>;
using ColumnState =
    std::variant<std::unique_ptr<RaycastBatchColumnStorage>, std::unique_ptr<SphereSweepBatchColumnStorage>,
                 std::unique_ptr<OverlapBatchColumnStorage>>;

InstanceOperationExecution executionResult(FamilyBatchOutcome outcome) noexcept
{
    switch (outcome)
    {
    case FamilyBatchOutcome::success: return {};
    case FamilyBatchOutcome::hostBudget:
        return {
            OVSPATIALQUERY_API_OPERATION_FAILED,
            "family batch execution exceeded the host memory budget while preparing float32 geometry",
        };
    case FamilyBatchOutcome::outOfMemory:
        return {
            OVSPATIALQUERY_API_OUT_OF_MEMORY,
            "family batch execution exhausted resources",
        };
    case FamilyBatchOutcome::invalidArgument:
    case FamilyBatchOutcome::internalFailure: break;
    }
    return {
        OVSPATIALQUERY_API_INTERNAL,
        "family batch execution failed internally",
    };
}

} // namespace

struct FamilyBatchPlanOperationOwner::Impl : std::enable_shared_from_this<FamilyBatchPlanOperationOwner::Impl>
{
    struct ResultSlot
    {
        bool                                 occupied  = false;
        bool                                 ready     = false;
        ovspatialquery_op_id_t               operation = OVSPATIALQUERY_INVALID_OP;
        ResolvedTarget                       target;
        LaunchState                          launch;
        ResultState                          result;
        std::shared_ptr<CudaFamilyOperation> cudaOperation;
        std::shared_ptr<void>                deviceResultReservation;
        ColumnState                          columns;
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
        ovspatialquery_api_status_t forcedFailure = OVSPATIALQUERY_API_SUCCESS;
#endif
    };

    template <typename Config, typename Policy>
    Impl(PopulationTable& value, ovspatialquery_instance_handle_t operationInstance, const Config& source, Policy)
        : populations(value), resolver(value), instance(operationInstance), common(source.common),
          planState(TypedPlanState<Policy>{source.plan})
    {}

    ~Impl() { releaseTarget(); }

    uint64_t maximumQueryCount() const noexcept
    {
        return std::visit([](const auto& state) { return state.plan.maximumQueryCount; }, planState);
    }

    uint64_t hitCapacity() const noexcept
    {
        return std::visit([](const auto& state) { return state.plan.hitCapacity; }, planState);
    }

    bool estimateResidentBytes(uint64_t& outBytes) const noexcept
    {
        return std::visit(
            [this, &outBytes](const auto& state)
            {
                using State = std::decay_t<decltype(state)>;
                return estimateResidentBytes<typename State::PolicyType>(common, state.plan, outBytes);
            },
            planState);
    }

    CudaFamilyPlanConfig cudaConfig() const noexcept
    {
        CudaFamilyPlanConfig value{};
        value.device          = common.cudaDevice;
        value.meshResidency   = common.cudaMeshResidency;
        value.resultFields    = common.cudaResultFields;
        value.hostMapped      = common.cudaHostMapped;
        value.resultSlotCount = common.resultSlotCount;
        value.overlapKind     = static_cast<CudaOverlapKind>(common.cudaOverlapKind);
        value.planRadius      = common.cudaPlanRadius;
        for (size_t axis = 0u; axis < 3u; ++axis)
        {
            value.planHalfExtents[axis] = common.cudaPlanHalfExtents[axis];
        }
        std::visit(
            [&value](const auto& state)
            {
                using State              = std::decay_t<decltype(state)>;
                using Policy             = typename State::PolicyType;
                value.maximumQueryCount  = state.plan.maximumQueryCount;
                value.hitCapacityPerSlot = state.plan.hitCapacity;
                value.contactTolerance   = state.plan.contactTolerance;
                value.workers            = state.plan.workers;
                value.precision          = state.plan.precision == QueryPrecision::exact ? CudaFamilyPrecision::exact
                                                                                         : CudaFamilyPrecision::boundsCandidate;
                if constexpr (std::is_same_v<Policy, RaycastPlanPolicy>)
                {
                    value.family              = CudaFamilyKind::raycast;
                    value.maximumHitsPerQuery = state.plan.maximumHitsPerQuery;
                    value.hitMode             = static_cast<CudaFamilyHitMode>(state.plan.hitMode);
                    value.allMeshHits         = state.plan.allMeshHits;
                }
                else if constexpr (std::is_same_v<Policy, SphereSweepPlanPolicy>)
                {
                    value.family              = state.plan.box ? CudaFamilyKind::boxSweep : CudaFamilyKind::sphereSweep;
                    value.maximumHitsPerQuery = state.plan.maximumHitsPerQuery;
                    value.hitMode             = static_cast<CudaFamilyHitMode>(state.plan.hitMode);
                }
                else
                {
                    value.family              = CudaFamilyKind::overlap;
                    value.maximumHitsPerQuery = state.plan.maximumHitsPerQuery;
                    value.hitMode             = static_cast<CudaFamilyHitMode>(state.plan.hitMode);
                }
            },
            planState);
        if (common.cudaAllowSinglePrecision)
        {
            value.precision = CudaFamilyPrecision::exactSingle;
        }
        return value;
    }

    template <typename Policy>
    static bool estimateResidentBytes(const FamilyBatchPlanCommonConfig& common, const typename Policy::Plan& plan,
                                      uint64_t& outBytes) noexcept
    {
        uint64_t total = sizeof(Impl) + sizeof(FamilyBatchPlanOperationOwner);
        if (common.backend == FamilyBatchPlanBackend::cuda)
        {
            uint64_t backendPlanBytes = 0u;
            if (!addHostCharge(common.resultSlotCount, sizeof(ResultSlot), total) ||
                !estimateCudaFamilyPlanFixedHostByteCharge(common.resultSlotCount, backendPlanBytes) ||
                !addHostCharge(1u, backendPlanBytes, total))
            {
                return false;
            }
            // Mapped result columns are page-locked host memory, so they are host bytes (ADR-0077).
            CudaFamilyMappedLayout mapped{};
            if (common.cudaHostMapped && (!layoutCudaFamilyMappedColumns(plan.maximumQueryCount, plan.hitCapacity,
                                                                         common.cudaResultFields, mapped) ||
                                          !addHostCharge(common.resultSlotCount, mapped.bytes, total)))
            {
                return false;
            }
            outBytes = total;
            return true;
        }
        if (plan.maximumQueryCount > std::numeric_limits<size_t>::max() ||
            plan.hitCapacity > std::numeric_limits<size_t>::max())
        {
            return false;
        }
        uint64_t rowsPerSlot       = 0u;
        uint64_t partitionsPerSlot = 0u;
        uint64_t hitsPerSlot       = 0u;
        uint64_t columnsPerSlot    = 0u;
        if (!addHostCharge(plan.maximumQueryCount, sizeof(typename Policy::Row), rowsPerSlot) ||
            !addHostCharge(plan.maximumQueryCount, sizeof(RowOutcome) + 2u * sizeof(uint64_t) + sizeof(uint8_t),
                           partitionsPerSlot) ||
            !addHostCharge(plan.hitCapacity, sizeof(typename Policy::Hit), hitsPerSlot))
        {
            return false;
        }
        const FamilyBatchColumnConfig columnConfig{
            plan.maximumQueryCount,
            plan.hitCapacity,
            common.resultFields,
            plan.precision == QueryPrecision::exact ? HitTruth::exactSurface : HitTruth::boundsCandidate,
        };
        if (!Policy::ColumnStorage::estimateResidentBytes(columnConfig, columnsPerSlot))
        {
            return false;
        }
        if (!addHostCharge(common.resultSlotCount, sizeof(ResultSlot), total) ||
            !addHostCharge(common.resultSlotCount, rowsPerSlot, total) ||
            !addHostCharge(common.resultSlotCount, partitionsPerSlot, total) ||
            !addHostCharge(common.resultSlotCount, hitsPerSlot, total) ||
            !addHostCharge(common.resultSlotCount, columnsPerSlot, total))
        {
            return false;
        }
        outBytes = total;
        return true;
    }

    void prepareStorage()
    {
        results.resize(common.resultSlotCount);
        const size_t rowCapacity       = static_cast<size_t>(maximumQueryCount());
        const size_t storedHitCapacity = static_cast<size_t>(hitCapacity());
        for (ResultSlot& slot : results)
        {
            std::visit(
                [this, &slot, rowCapacity, storedHitCapacity](const auto& state)
                {
                    using State  = std::decay_t<decltype(state)>;
                    using Policy = typename State::PolicyType;
                    slot.launch  = typename Policy::Launch{};
                    std::get<typename Policy::Launch>(slot.launch).rows.reserve(rowCapacity);
                    slot.result                     = typename Policy::Result{};
                    typename Policy::Result& result = std::get<typename Policy::Result>(slot.result);
                    result.hits.reserve(storedHitCapacity);
                    result.common.rowOutcomes.reserve(rowCapacity);
                    result.common.hitOffsets.reserve(rowCapacity);
                    result.common.hitCounts.reserve(rowCapacity);
                    result.common.truncated.reserve(rowCapacity);
                    const FamilyBatchColumnConfig columnConfig{
                        static_cast<uint64_t>(rowCapacity),
                        static_cast<uint64_t>(storedHitCapacity),
                        common.resultFields,
                        state.plan.precision == QueryPrecision::exact ? HitTruth::exactSurface
                                                                      : HitTruth::boundsCandidate,
                    };
                    slot.columns = Policy::ColumnStorage::create(columnConfig);
                },
                planState);
        }
    }

    ovspatialquery_api_status_t initialize() noexcept
    {
        if (instance == OVSPATIALQUERY_INVALID_INSTANCE)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        const ovspatialquery_api_status_t validationStatus = populations.validate(common.target.population);
        if (validationStatus == OVSPATIALQUERY_API_INVALID_ARGUMENT ||
            validationStatus == OVSPATIALQUERY_API_INVALID_HANDLE)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        if (validationStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        uint64_t deviceBytes = 0u;
        if (common.backend == FamilyBatchPlanBackend::cuda &&
            validateCudaFamilyPlanConfig(cudaConfig(), deviceBytes) != OVSPATIALQUERY_API_SUCCESS)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        uint64_t residentBytes = 0u;
        if (!estimateResidentBytes(residentBytes))
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        if (common.backend == FamilyBatchPlanBackend::cuda)
        {
            const ovspatialquery_api_status_t cudaStatus = createCudaFamilyPlan(cudaConfig(), cudaPlan);
            if (cudaStatus == OVSPATIALQUERY_API_NOT_SUPPORTED)
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            if (cudaStatus == OVSPATIALQUERY_API_OUT_OF_MEMORY)
            {
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            if (cudaStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
        }
        try
        {
            if (common.backend == FamilyBatchPlanBackend::cuda)
            {
                results.resize(common.resultSlotCount);
                if (results.capacity() != common.resultSlotCount)
                {
                    throw std::length_error("CUDA family result owner storage exceeded its charge");
                }
            }
            else
            {
                prepareStorage();
            }
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (const std::length_error&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (...)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        const ovspatialquery_api_status_t status = populations.retainPersistentReferences({common.target.population});
        if (status == OVSPATIALQUERY_API_INVALID_HANDLE)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        if (status == OVSPATIALQUERY_API_OUT_OF_MEMORY)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        targetRetained = true;
        return OVSPATIALQUERY_API_SUCCESS;
    }

    void clearResult(ResultSlot& slot)
    {
        slot.ready = false;
        std::visit(
            [](auto& result)
            {
                FamilyBatchResultCommon& value = result.common;
                value.sourcePopulation         = {};
                value.sourceOrdinal            = 0;
                value.truth                    = HitTruth::boundsCandidate;
                value.hitCapacity              = 0u;
                value.storedHitCount           = 0u;
                value.rowOutcomes.clear();
                value.hitOffsets.clear();
                value.hitCounts.clear();
                value.truncated.clear();
                result.hits.clear();
            },
            slot.result);
    }

    void clearSlot(ResultSlot& slot)
    {
        slot.cudaOperation.reset();
        slot.deviceResultReservation.reset();
        slot.target = {};
        std::visit([](auto& launch) { launch.rows.clear(); }, slot.launch);
        clearResult(slot);
        slot.occupied  = false;
        slot.operation = OVSPATIALQUERY_INVALID_OP;
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
        slot.forcedFailure = OVSPATIALQUERY_API_SUCCESS;
#endif
    }

    void releaseSlot(uint32_t index, ovspatialquery_op_id_t operation) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (index < results.size() && results[index].occupied && results[index].operation == operation)
        {
            clearSlot(results[index]);
        }
    }

    // A CUDA slot retired after its consumers returns on the instance's reaper
    // thread, which can trail a launch made right after the retirement. When
    // every slot is taken, end this owner's device-safe retirements here
    // rather than refuse for a slot that is already free.
    void reclaimRetiredSlotsWhenFull() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (std::any_of(results.begin(), results.end(), [](const ResultSlot& slot) { return !slot.occupied; }))
            {
                return;
            }
        }
        reclaimInstanceDeviceRetirements(instance, this);
    }

    template <typename Policy>
    InstanceOperationExecution executeTyped(uint32_t resultIndex, ovspatialquery_op_id_t operation,
                                            const typename Policy::Plan& plan) noexcept
    {
        if (resultIndex >= results.size())
        {
            return {
                OVSPATIALQUERY_API_INTERNAL,
                "family batch result slot is invalid",
            };
        }
        ResultSlot& slot = results[resultIndex];
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
        switch (slot.forcedFailure)
        {
        case OVSPATIALQUERY_API_OUT_OF_MEMORY:
            return {
                OVSPATIALQUERY_API_OUT_OF_MEMORY,
                "family batch forced resource failure",
            };
        case OVSPATIALQUERY_API_OPERATION_FAILED:
            return {
                OVSPATIALQUERY_API_OPERATION_FAILED,
                "family batch forced backend failure",
            };
        case OVSPATIALQUERY_API_INTERNAL:
            return {
                OVSPATIALQUERY_API_INTERNAL,
                "family batch forced internal failure",
            };
        default: break;
        }
#endif
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!slot.occupied || slot.operation != operation)
            {
                return {
                    OVSPATIALQUERY_API_INTERNAL,
                    "family batch result ownership changed before execution",
                };
            }
            clearResult(slot);
        }
        typename Policy::Result& result = std::get<typename Policy::Result>(slot.result);
        const FamilyBatchOutcome outcome =
            executeFamilyBatchCpu(slot.target, plan, std::get<typename Policy::Launch>(slot.launch), result);
        if (outcome != FamilyBatchOutcome::success)
        {
            return executionResult(outcome);
        }
        std::unique_ptr<typename Policy::ColumnStorage>& columns =
            std::get<std::unique_ptr<typename Policy::ColumnStorage>>(slot.columns);
        if (columns == nullptr || columns->publish(result) != FamilyBatchColumnsOutcome::success)
        {
            return {
                OVSPATIALQUERY_API_INTERNAL,
                "family batch column publication failed",
            };
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (!slot.occupied || slot.operation != operation)
        {
            return {
                OVSPATIALQUERY_API_INTERNAL,
                "family batch result ownership changed during execution",
            };
        }
        slot.ready = true;
        return {};
    }

    InstanceOperationExecution execute(uint32_t resultIndex, ovspatialquery_op_id_t operation) noexcept
    {
        if (common.backend == FamilyBatchPlanBackend::cuda)
        {
            std::shared_ptr<CudaFamilyOperation> cudaOperation;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (resultIndex >= results.size() || !results[resultIndex].occupied ||
                    results[resultIndex].operation != operation || results[resultIndex].cudaOperation == nullptr)
                {
                    return {OVSPATIALQUERY_API_INTERNAL, "CUDA family result ownership is invalid"};
                }
                cudaOperation = results[resultIndex].cudaOperation;
            }
            // Spin briefly for launches that finish within a scheduling quantum, then block
            // on the launch's blocking-sync event so a long kernel neither burns a core nor
            // waits out a sleep: on Windows a sleep lasts a whole timer tick (15.6 ms by
            // default), which would quantize every launch. Poll with sleeps only when the
            // event cannot be waited on.
            const std::chrono::steady_clock::time_point spinUntil =
                std::chrono::steady_clock::now() + std::chrono::microseconds(200);
            bool blocked = false;
            for (;;)
            {
                const CudaCompletionState state = cudaOperation->resultState();
                if (state == CudaCompletionState::pending)
                {
                    if (std::chrono::steady_clock::now() < spinUntil)
                        std::this_thread::yield();
                    else if (!blocked)
                    {
                        blocked = true;
                        cudaOperation->waitForResult();
                    }
                    else
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    continue;
                }
                cudaOperation->releaseReadyDependencies();
                if (state == CudaCompletionState::deviceLost)
                {
                    return {OVSPATIALQUERY_API_DEVICE_LOST, "CUDA family device was lost"};
                }
                if (state != CudaCompletionState::complete)
                {
                    return {OVSPATIALQUERY_API_OPERATION_FAILED, "CUDA family execution failed"};
                }
                std::lock_guard<std::mutex> lock(mutex);
                if (resultIndex >= results.size() || !results[resultIndex].occupied ||
                    results[resultIndex].operation != operation)
                {
                    return {OVSPATIALQUERY_API_INTERNAL, "CUDA family result ownership changed"};
                }
                results[resultIndex].ready = true;
                return {};
            }
        }
        return std::visit(
            [this, resultIndex, operation](const auto& state)
            {
                using State  = std::decay_t<decltype(state)>;
                using Policy = typename State::PolicyType;
                return executeTyped<Policy>(resultIndex, operation, state.plan);
            },
            planState);
    }

    // A const launch is copied into the result slot; an rvalue one is exchanged with the
    // slot's previous launch, so a caller's scratch launch keeps reusable storage.
    template <typename Policy, typename LaunchArg>
    FamilyBatchPlanAdmission enqueueTyped(TargetSelection selection, LaunchArg&& launch,
                                          typename Policy::PublicReservation& outColumnReservation) noexcept;
    FamilyBatchPlanAdmission enqueueCuda(TargetSelection selection, const CudaFamilyLaunch& launch,
                                         CudaFamilyResultColumns& outReservation) noexcept;

    // Success once the operation completed; otherwise why its result cannot be read.
    ovspatialquery_api_status_t completedAccess(FamilyBatchPlanOperationId operation) const noexcept
    {
        InstanceOperationState            state  = InstanceOperationState::pending;
        const ovspatialquery_api_status_t access = queryInstanceOperation(instance, operation.value, state);
        if (access != OVSPATIALQUERY_API_SUCCESS)
        {
            return access;
        }
        if (state == InstanceOperationState::pending)
        {
            return OVSPATIALQUERY_API_NOT_READY;
        }
        if (state == InstanceOperationState::failed)
        {
            return OVSPATIALQUERY_API_OPERATION_FAILED;
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }

    template <typename Policy>
    ovspatialquery_api_status_t fetchColumnsTyped(FamilyBatchPlanOperationId     operation,
                                                  typename Policy::PublicResult& outResult) const noexcept
    {
        if (!std::holds_alternative<TypedPlanState<Policy>>(planState))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        const ovspatialquery_api_status_t completed = completedAccess(operation);
        if (completed != OVSPATIALQUERY_API_SUCCESS)
        {
            return completed;
        }
        std::lock_guard<std::mutex> lock(mutex);
        for (const ResultSlot& slot : results)
        {
            if (slot.occupied && slot.operation == operation.value && slot.ready &&
                std::holds_alternative<std::unique_ptr<typename Policy::ColumnStorage>>(slot.columns))
            {
                const std::unique_ptr<typename Policy::ColumnStorage>& columns =
                    std::get<std::unique_ptr<typename Policy::ColumnStorage>>(slot.columns);
                if (columns == nullptr)
                {
                    return OVSPATIALQUERY_API_INVALID_HANDLE;
                }
                outResult = columns->result();
                return OVSPATIALQUERY_API_SUCCESS;
            }
        }
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }

    void releaseTarget() noexcept
    {
        if (targetRetained)
        {
            populations.finishPersistentReferences({common.target.population});
            targetRetained = false;
        }
    }

    PopulationTable&                       populations;
    TargetResolver                         resolver;
    const ovspatialquery_instance_handle_t instance;
    const FamilyBatchPlanCommonConfig      common;
    const PlanState                        planState;
    std::shared_ptr<CudaFamilyPlan>        cudaPlan;
    bool                                   targetRetained = false;
    bool                                   closed         = false;
    mutable std::mutex                     mutex;
    std::vector<ResultSlot>                results;
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
    bool failNextResource = false;
    bool failNextBackend  = false;
    bool failNextInternal = false;
#endif
};

bool FamilyBatchPlanOperationOwner::estimateResidentBytes(const RaycastFamilyBatchPlanConfig& config,
                                                          uint64_t&                           outBytes) noexcept
{
    return Impl::estimateResidentBytes<RaycastPlanPolicy>(config.common, config.plan, outBytes);
}

bool FamilyBatchPlanOperationOwner::estimateResidentBytes(const SphereSweepFamilyBatchPlanConfig& config,
                                                          uint64_t&                               outBytes) noexcept
{
    return Impl::estimateResidentBytes<SphereSweepPlanPolicy>(config.common, config.plan, outBytes);
}

bool FamilyBatchPlanOperationOwner::estimateResidentBytes(const OverlapFamilyBatchPlanConfig& config,
                                                          uint64_t&                           outBytes) noexcept
{
    return Impl::estimateResidentBytes<OverlapPlanPolicy>(config.common, config.plan, outBytes);
}

#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
namespace
{
std::atomic<FamilyBatchPlanTaskPayloadDestroyHook> gFamilyBatchPlanTaskPayloadDestroyHook{nullptr};
std::atomic<FamilyBatchPlanRetiredSlotHook>        gFamilyBatchPlanRetiredSlotHook{nullptr};

struct FamilyBatchPlanTaskPayloadDestroyProbe
{
    ~FamilyBatchPlanTaskPayloadDestroyProbe()
    {
        const FamilyBatchPlanTaskPayloadDestroyHook hook =
            gFamilyBatchPlanTaskPayloadDestroyHook.load(std::memory_order_acquire);
        if (hook != nullptr)
        {
            hook();
        }
    }
};
} // namespace

void setFamilyBatchPlanRetiredSlotHookForTest(FamilyBatchPlanRetiredSlotHook hook) noexcept
{
    gFamilyBatchPlanRetiredSlotHook.store(hook, std::memory_order_release);
}

void setFamilyBatchPlanTaskPayloadDestroyHookForTest(FamilyBatchPlanTaskPayloadDestroyHook hook) noexcept
{
    gFamilyBatchPlanTaskPayloadDestroyHook.store(hook, std::memory_order_release);
}
#endif

struct FamilyBatchPlanRuntimeTask final : OperationTask
{
    InstanceOperationExecution execute() noexcept override { return owner->execute(resultIndex, operation); }
    void                       disarmForDeviceRetirement() noexcept override { releaseResultSlot = false; }
    // A waiter runs a CPU launch, or a CUDA launch's wait for its device completion,
    // without handing the result back from a worker.
    bool waiterMayRun() const noexcept override { return true; }

    std::shared_ptr<FamilyBatchPlanOperationOwner::Impl> owner;
    uint32_t                                             resultIndex       = UINT32_MAX;
    ovspatialquery_op_id_t                               operation         = OVSPATIALQUERY_INVALID_OP;
    bool                                                 releaseResultSlot = true;
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
    FamilyBatchPlanTaskPayloadDestroyProbe payloadDestroyProbe;
#endif

    ~FamilyBatchPlanRuntimeTask()
    {
        if (releaseResultSlot && owner != nullptr)
        {
            owner->releaseSlot(resultIndex, operation);
        }
    }
};

template <typename Policy, typename LaunchArg>
FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::Impl::enqueueTyped(
    TargetSelection selection, LaunchArg&& launch, typename Policy::PublicReservation& outColumnReservation) noexcept
{
    FamilyBatchPlanAdmission admission;
    using State = TypedPlanState<Policy>;
    if (!std::holds_alternative<State>(planState))
    {
        admission.outcome = OVSPATIALQUERY_API_NOT_SUPPORTED;
        return admission;
    }
    if ((selection.kind != TargetSelectionKind::head && selection.kind != TargetSelectionKind::retainedExact) ||
        (selection.kind == TargetSelectionKind::retainedExact && selection.ordinal < 0) ||
        launch.rows.size() > maximumQueryCount())
    {
        admission.outcome = OVSPATIALQUERY_API_INVALID_ARGUMENT;
        return admission;
    }

    InstanceOperationReservation       reservation{};
    uint32_t                           resultIndex = UINT32_MAX;
    typename Policy::PublicReservation columnReservation{};
    try
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed)
        {
            admission.outcome = OVSPATIALQUERY_API_INVALID_HANDLE;
            return admission;
        }
        ResolvedTarget             resolved;
        const TargetResolveOutcome resolveOutcome = resolver.resolve({common.target, selection}, resolved);
        if (resolveOutcome != TargetResolveOutcome::success)
        {
            admission.outcome = admissionStatus(resolveOutcome);
            return admission;
        }
        if (resolved.generation.preparedOnDevice())
        {
            // A readable device version is copied to the host on first read,
            // charged here; plain ON_DEVICE has no host objects (ADR-0024).
            switch (resolver.materializeHostCopy(resolved.generation, common.hostLedger))
            {
            case OVSPATIALQUERY_API_SUCCESS: break;
            case OVSPATIALQUERY_API_NOT_SUPPORTED:
                admission.outcome = OVSPATIALQUERY_API_NOT_SUPPORTED;
                return admission;
            case OVSPATIALQUERY_API_HOST_BUDGET: admission.outcome = OVSPATIALQUERY_API_HOST_BUDGET; return admission;
            case OVSPATIALQUERY_API_OUT_OF_MEMORY:
                admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
                return admission;
            default: admission.outcome = OVSPATIALQUERY_API_INVALID_HANDLE; return admission;
            }
        }
        const ovspatialquery_api_status_t reserveOutcome =
            reserveInstanceOperation(instance, InstanceOperationKind::familyBatchPlan, this, reservation);
        if (reserveOutcome == OVSPATIALQUERY_API_OPERATION_CAPACITY)
        {
            admission.outcome = OVSPATIALQUERY_API_OPERATION_CAPACITY;
            return admission;
        }
        if (reserveOutcome != OVSPATIALQUERY_API_SUCCESS)
        {
            admission.outcome = OVSPATIALQUERY_API_INTERNAL;
            return admission;
        }
        const auto freeSlot =
            std::find_if(results.begin(), results.end(), [](const ResultSlot& slot) { return !slot.occupied; });
        if (freeSlot == results.end())
        {
            abandonInstanceOperation(instance, reservation);
            admission.outcome = OVSPATIALQUERY_API_RESULT_CAPACITY;
            return admission;
        }
        resultIndex = static_cast<uint32_t>(freeSlot - results.begin());

        HostLease executionLease;
        if (common.hostLedger != nullptr)
        {
            uint64_t executionPeak = 0u;
            if (!estimateFamilyBatchExecutionPeakBytes(resolved, std::get<State>(planState).plan, executionPeak))
            {
                abandonInstanceOperation(instance, reservation);
                admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
                return admission;
            }
            HostTransaction    transaction(*common.hostLedger);
            constexpr uint64_t taskPayload = sizeof(FamilyBatchPlanRuntimeTask) + 3u * sizeof(void*);
            if (!transaction.reserve(executionPeak) || !transaction.reserve(taskPayload) ||
                !transaction.commit(executionLease))
            {
                abandonInstanceOperation(instance, reservation);
                admission.outcome = OVSPATIALQUERY_API_HOST_BUDGET;
                return admission;
            }
        }

        ResultSlot& slot                    = results[resultIndex];
        slot.occupied                       = true;
        slot.operation                      = reservation.operation;
        slot.target                         = std::move(resolved);
        typename Policy::Launch& slotLaunch = std::get<typename Policy::Launch>(slot.launch);
        if constexpr (std::is_const_v<std::remove_reference_t<LaunchArg>>)
        {
            slotLaunch = launch;
        }
        else
        {
            std::swap(slotLaunch, launch);
        }
        std::unique_ptr<typename Policy::ColumnStorage>& columns =
            std::get<std::unique_ptr<typename Policy::ColumnStorage>>(slot.columns);
        if (columns == nullptr)
        {
            clearSlot(slot);
            abandonInstanceOperation(instance, reservation);
            admission.outcome = OVSPATIALQUERY_API_INTERNAL;
            return admission;
        }
        columns->begin(resultIndex, static_cast<uint64_t>(slotLaunch.rows.size()), slot.target.ordinal,
                       columnReservation);
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
        slot.forcedFailure = OVSPATIALQUERY_API_SUCCESS;
        if (failNextResource)
        {
            failNextResource   = false;
            slot.forcedFailure = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        else if (failNextBackend)
        {
            failNextBackend    = false;
            slot.forcedFailure = OVSPATIALQUERY_API_OPERATION_FAILED;
        }
        else if (failNextInternal)
        {
            failNextInternal   = false;
            slot.forcedFailure = OVSPATIALQUERY_API_INTERNAL;
        }
#endif
        std::shared_ptr<FamilyBatchPlanRuntimeTask> task = std::make_shared<FamilyBatchPlanRuntimeTask>();
        task->owner                                      = shared_from_this();
        task->resultIndex                                = resultIndex;
        task->operation                                  = reservation.operation;
        const ovspatialquery_api_status_t scheduleOutcome =
            bindInstanceOperation(instance, reservation, task, std::move(executionLease), true);
        if (scheduleOutcome != OVSPATIALQUERY_API_SUCCESS)
        {
            task->owner.reset();
            clearSlot(slot);
            task.reset();
            abandonInstanceOperation(instance, reservation);
            admission.outcome = OVSPATIALQUERY_API_INTERNAL;
            return admission;
        }
        admission.operation  = {reservation.operation};
        admission.resultSlot = resultIndex;
        admission.outcome    = OVSPATIALQUERY_API_SUCCESS;
        outColumnReservation = columnReservation;
        return admission;
    }
    catch (const std::bad_alloc&)
    {
        admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        admission.outcome = OVSPATIALQUERY_API_INTERNAL;
    }
    if (resultIndex != UINT32_MAX && resultIndex < results.size())
    {
        releaseSlot(resultIndex, reservation.operation);
    }
    abandonInstanceOperation(instance, reservation);
    return admission;
}

FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::Impl::enqueueCuda(
    TargetSelection selection, const CudaFamilyLaunch& launch, CudaFamilyResultColumns& outReservation) noexcept
{
    FamilyBatchPlanAdmission admission{};
    // Preserve the caller's reservation on every failed admission.  CUDA
    // measurement and capacity checks are deliberately side-effect free from
    // the caller's perspective; only a successfully activated operation
    // publishes its reservation below.
    // CUDA launches may also select an admitted, device-prepared publication
    // (ADR-0021); their kernels wait for it on the stream.
    if (common.backend != FamilyBatchPlanBackend::cuda || cudaPlan == nullptr ||
        (selection.kind != TargetSelectionKind::head && selection.kind != TargetSelectionKind::retainedExact &&
         selection.kind != TargetSelectionKind::admittedExact) ||
        (selection.kind != TargetSelectionKind::head && selection.ordinal < 0))
    {
        admission.outcome = OVSPATIALQUERY_API_INVALID_ARGUMENT;
        return admission;
    }
    reclaimRetiredSlotsWhenFull();
    InstanceOperationReservation reservation{};
    uint32_t                     resultIndex = UINT32_MAX;
    bool                         bound       = false;
    try
    {
        std::shared_ptr<CudaFamilyOperation> prepared;
        std::shared_ptr<void>                deviceResultReservation;
        std::shared_ptr<void>                stagingReservation;
        std::shared_ptr<void>                publishedCatalogueReservation;
        std::vector<std::shared_ptr<void>>   publishedDefinitionReservations;
        std::shared_ptr<HostLease>           hostStagingReservation;
        ResolvedTarget                       resolved;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (closed)
            {
                admission.outcome = OVSPATIALQUERY_API_INVALID_HANDLE;
                return admission;
            }
            const TargetResolveOutcome resolvedStatus = resolver.resolve({common.target, selection}, resolved);
            if (resolvedStatus != TargetResolveOutcome::success)
            {
                admission.outcome = admissionStatus(resolvedStatus);
                return admission;
            }
            // An exact plan over a device-prepared version evaluates the host
            // copy of a readable version, uploaded as a catalogue like a
            // host-prepared version's objects; bounds plans borrow the device
            // tables and need no copy (ADR-0024 decision 4).
            if (cudaPlan->needsHostObjects(resolved.generation))
            {
                switch (resolver.materializeHostCopy(resolved.generation, common.hostLedger))
                {
                case OVSPATIALQUERY_API_SUCCESS: break;
                case OVSPATIALQUERY_API_NOT_SUPPORTED:
                    admission.outcome = OVSPATIALQUERY_API_NOT_SUPPORTED;
                    return admission;
                case OVSPATIALQUERY_API_HOST_BUDGET:
                    admission.outcome = OVSPATIALQUERY_API_HOST_BUDGET;
                    return admission;
                case OVSPATIALQUERY_API_OUT_OF_MEMORY:
                    admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
                    return admission;
                default: admission.outcome = OVSPATIALQUERY_API_INVALID_HANDLE; return admission;
                }
            }
            CudaFamilyStagingCharge           staging{};
            const ovspatialquery_api_status_t measuredStatus = cudaPlan->measure(launch, resolved.generation, staging);
            if (measuredStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                admission.outcome = measuredStatus == OVSPATIALQUERY_API_NOT_SUPPORTED
                                        ? OVSPATIALQUERY_API_NOT_SUPPORTED
                                        : OVSPATIALQUERY_API_INVALID_ARGUMENT;
                return admission;
            }
            constexpr uint64_t sharedControlAllowance = 3u * sizeof(void*);
            constexpr uint64_t ownerRuntimeHostBytes =
                sizeof(FamilyBatchPlanRuntimeTask) + sizeof(HostLease) + 2u * sharedControlAllowance;
            if (!addHostCharge(1u, ownerRuntimeHostBytes, staging.hostBytes))
            {
                admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
                return admission;
            }
            const ovspatialquery_api_status_t stagingReserve =
                reserveInstanceDeviceMemory(instance, common.cudaDevice, staging.deviceBytes, stagingReservation);
            if (stagingReserve != OVSPATIALQUERY_API_SUCCESS)
            {
                admission.outcome = stagingReserve == OVSPATIALQUERY_API_DEVICE_BUDGET
                                        ? OVSPATIALQUERY_API_DEVICE_BUDGET
                                        : OVSPATIALQUERY_API_OUT_OF_MEMORY;
                return admission;
            }
            if (staging.publishedDeviceBytes != 0u)
            {
                const ovspatialquery_api_status_t publishedReserve = reserveInstanceDeviceMemory(
                    instance, common.cudaDevice, staging.publishedDeviceBytes, publishedCatalogueReservation);
                if (publishedReserve != OVSPATIALQUERY_API_SUCCESS)
                {
                    admission.outcome = publishedReserve == OVSPATIALQUERY_API_DEVICE_BUDGET
                                            ? OVSPATIALQUERY_API_DEVICE_BUDGET
                                            : OVSPATIALQUERY_API_OUT_OF_MEMORY;
                    return admission;
                }
            }
            publishedDefinitionReservations.reserve(staging.publishedDefinitionBytes.size());
            for (const uint64_t definitionBytes : staging.publishedDefinitionBytes)
            {
                std::shared_ptr<void>             definitionReservation;
                const ovspatialquery_api_status_t definitionReserve =
                    reserveInstanceDeviceMemory(instance, common.cudaDevice, definitionBytes, definitionReservation);
                if (definitionReserve != OVSPATIALQUERY_API_SUCCESS)
                {
                    admission.outcome = definitionReserve == OVSPATIALQUERY_API_DEVICE_BUDGET
                                            ? OVSPATIALQUERY_API_DEVICE_BUDGET
                                            : OVSPATIALQUERY_API_OUT_OF_MEMORY;
                    return admission;
                }
                publishedDefinitionReservations.push_back(std::move(definitionReservation));
            }
            if (staging.hostBytes != 0u && common.hostLedger != nullptr)
            {
                HostLease       hostLease;
                HostTransaction hostTransaction(*common.hostLedger);
                if (!hostTransaction.reserve(staging.hostBytes) || !hostTransaction.commit(hostLease))
                {
                    admission.outcome = OVSPATIALQUERY_API_HOST_BUDGET;
                    return admission;
                }
                hostStagingReservation = std::make_shared<HostLease>(std::move(hostLease));
            }
            const ovspatialquery_api_status_t preparedStatus =
                cudaPlan->prepare(launch, std::move(resolved.generation), std::move(staging.staging), prepared);
            if (preparedStatus != OVSPATIALQUERY_API_SUCCESS || prepared == nullptr)
            {
                admission.outcome =
                    preparedStatus == OVSPATIALQUERY_API_OUT_OF_MEMORY
                        ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                        : (preparedStatus == OVSPATIALQUERY_API_NOT_SUPPORTED ? OVSPATIALQUERY_API_NOT_SUPPORTED
                                                                              : OVSPATIALQUERY_API_INVALID_ARGUMENT);
                return admission;
            }
            prepared->attachPublishedReservations(std::move(publishedCatalogueReservation),
                                                  std::move(publishedDefinitionReservations));
            const ovspatialquery_api_status_t operationReserve =
                reserveInstanceOperation(instance, InstanceOperationKind::familyBatchPlan, this, reservation);
            if (operationReserve != OVSPATIALQUERY_API_SUCCESS)
            {
                admission.outcome = operationReserve == OVSPATIALQUERY_API_OPERATION_CAPACITY
                                        ? OVSPATIALQUERY_API_OPERATION_CAPACITY
                                        : OVSPATIALQUERY_API_INTERNAL;
                return admission;
            }
            const auto freeSlot =
                std::find_if(results.begin(), results.end(), [](const ResultSlot& slot) { return !slot.occupied; });
            if (freeSlot == results.end())
            {
                abandonInstanceOperation(instance, reservation);
                admission.outcome = OVSPATIALQUERY_API_RESULT_CAPACITY;
                return admission;
            }
            resultIndex = static_cast<uint32_t>(freeSlot - results.begin());
            const ovspatialquery_api_status_t deviceReserve =
                reserveInstanceDeviceResult(instance, deviceResultReservation);
            if (deviceReserve != OVSPATIALQUERY_API_SUCCESS)
            {
                abandonInstanceOperation(instance, reservation);
                admission.outcome = deviceReserve == OVSPATIALQUERY_API_RESULT_CAPACITY
                                        ? OVSPATIALQUERY_API_RESULT_CAPACITY
                                        : OVSPATIALQUERY_API_OUT_OF_MEMORY;
                return admission;
            }
            ResultSlot& slot                                 = results[resultIndex];
            slot.occupied                                    = true;
            slot.operation                                   = reservation.operation;
            slot.deviceResultReservation                     = deviceResultReservation;
            std::shared_ptr<FamilyBatchPlanRuntimeTask> task = std::make_shared<FamilyBatchPlanRuntimeTask>();
            task->owner                                      = shared_from_this();
            task->resultIndex                                = resultIndex;
            task->operation                                  = reservation.operation;
            const ovspatialquery_api_status_t boundStatus =
                bindInstanceOperation(instance, reservation, task, HostLease{}, false);
            if (boundStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                task->owner.reset();
                clearSlot(slot);
                task.reset();
                abandonInstanceOperation(instance, reservation);
                admission.outcome = OVSPATIALQUERY_API_INTERNAL;
                return admission;
            }
            bound = true;
        }
        std::shared_ptr<CudaFamilyOperation> cudaOperation;
        const ovspatialquery_api_status_t    launchStatus =
            cudaPlan->launch(resultIndex, std::move(prepared), cudaOperation);
        if (launchStatus != OVSPATIALQUERY_API_SUCCESS || cudaOperation == nullptr)
        {
            const ovspatialquery_op_id_t admittedOperation = reservation.operation;
            abandonBoundInstanceOperation(instance, reservation);
            releaseSlot(resultIndex, admittedOperation);
            admission.outcome =
                launchStatus == OVSPATIALQUERY_API_NOT_SUPPORTED
                    ? OVSPATIALQUERY_API_NOT_SUPPORTED
                    : (launchStatus == OVSPATIALQUERY_API_OUT_OF_MEMORY ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                                                        : OVSPATIALQUERY_API_INTERNAL);
            return admission;
        }
        cudaOperation->attachDeviceResultReservation(deviceResultReservation);
        cudaOperation->attachStagingReservation(std::move(stagingReservation));
        cudaOperation->attachHostStagingReservation(std::move(hostStagingReservation));
        const CudaFamilyResultColumns cudaReservation = cudaOperation->reservation();
        {
            std::lock_guard<std::mutex> lock(mutex);
            ResultSlot&                 slot = results[resultIndex];
            if (!slot.occupied || slot.operation != reservation.operation)
            {
                return {OVSPATIALQUERY_API_INTERNAL, {}, UINT32_MAX};
            }
            slot.cudaOperation = cudaOperation;
        }
        if (activateInstanceOperation(instance, reservation) != OVSPATIALQUERY_API_SUCCESS)
        {
            std::terminate();
        }
        outReservation       = cudaReservation;
        admission.outcome    = OVSPATIALQUERY_API_SUCCESS;
        admission.operation  = {reservation.operation};
        admission.resultSlot = resultIndex;
        return admission;
    }
    catch (const std::bad_alloc&)
    {
        admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        admission.outcome = OVSPATIALQUERY_API_INTERNAL;
    }
    if (bound)
    {
        abandonBoundInstanceOperation(instance, reservation);
    }
    else
    {
        abandonInstanceOperation(instance, reservation);
    }
    if (resultIndex != UINT32_MAX)
    {
        releaseSlot(resultIndex, reservation.operation);
    }
    return admission;
}

FamilyBatchPlanOperationOwner::FamilyBatchPlanOperationOwner(std::shared_ptr<Impl> impl) noexcept
    : mImpl(std::move(impl))
{}

FamilyBatchPlanOperationOwner::~FamilyBatchPlanOperationOwner() = default;

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::destroy(
    std::unique_ptr<FamilyBatchPlanOperationOwner>& owner) noexcept
{
    if (owner == nullptr)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    const ovspatialquery_api_status_t outcome = owner->close();
    if (outcome == OVSPATIALQUERY_API_SUCCESS)
    {
        owner.reset();
    }
    return outcome;
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::create(
    PopulationTable& populations, ovspatialquery_instance_handle_t instance, const RaycastFamilyBatchPlanConfig& config,
    std::unique_ptr<FamilyBatchPlanOperationOwner>& outOwner) noexcept
{
    return createTyped<RaycastPlanPolicy>(populations, instance, config, outOwner);
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::create(
    PopulationTable& populations, ovspatialquery_instance_handle_t instance,
    const SphereSweepFamilyBatchPlanConfig& config, std::unique_ptr<FamilyBatchPlanOperationOwner>& outOwner) noexcept
{
    return createTyped<SphereSweepPlanPolicy>(populations, instance, config, outOwner);
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::create(
    PopulationTable& populations, ovspatialquery_instance_handle_t instance, const OverlapFamilyBatchPlanConfig& config,
    std::unique_ptr<FamilyBatchPlanOperationOwner>& outOwner) noexcept
{
    return createTyped<OverlapPlanPolicy>(populations, instance, config, outOwner);
}

template <typename Policy, typename Config>
ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::createTyped(
    PopulationTable& populations, ovspatialquery_instance_handle_t instance, const Config& config,
    std::unique_ptr<FamilyBatchPlanOperationOwner>& outOwner) noexcept
{
    if (instance == OVSPATIALQUERY_INVALID_INSTANCE || config.common.resultSlotCount == 0u || !validPlan(config.plan) ||
        !validResultFields<Policy>(config.common.resultFields, config.plan.precision))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::shared_ptr<Impl>             impl    = std::make_shared<Impl>(populations, instance, config, Policy{});
        const ovspatialquery_api_status_t outcome = impl->initialize();
        if (outcome != OVSPATIALQUERY_API_SUCCESS)
        {
            return outcome;
        }
        std::unique_ptr<FamilyBatchPlanOperationOwner> candidate(new FamilyBatchPlanOperationOwner(std::move(impl)));
        outOwner = std::move(candidate);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::enqueueColumns(
    TargetSelection selection, const RaycastBatchLaunch& launch,
    ovspatialquery_raycast_batch_reservation_t& outReservation) noexcept
{
    return mImpl->enqueueTyped<RaycastPlanPolicy>(selection, launch, outReservation);
}

FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::enqueueColumns(
    TargetSelection selection, RaycastBatchLaunch&& launch,
    ovspatialquery_raycast_batch_reservation_t& outReservation) noexcept
{
    return mImpl->enqueueTyped<RaycastPlanPolicy>(selection, std::move(launch), outReservation);
}

FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::enqueueColumns(
    TargetSelection selection, const SphereSweepBatchLaunch& launch,
    ovspatialquery_sphere_sweep_batch_reservation_t& outReservation) noexcept
{
    return mImpl->enqueueTyped<SphereSweepPlanPolicy>(selection, launch, outReservation);
}

FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::enqueueColumns(
    TargetSelection selection, SphereSweepBatchLaunch&& launch,
    ovspatialquery_sphere_sweep_batch_reservation_t& outReservation) noexcept
{
    return mImpl->enqueueTyped<SphereSweepPlanPolicy>(selection, std::move(launch), outReservation);
}

FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::enqueueColumns(
    TargetSelection selection, const OverlapBatchLaunch& launch,
    ovspatialquery_overlap_batch_reservation_t& outReservation) noexcept
{
    return mImpl->enqueueTyped<OverlapPlanPolicy>(selection, launch, outReservation);
}

FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::enqueueColumns(
    TargetSelection selection, OverlapBatchLaunch&& launch,
    ovspatialquery_overlap_batch_reservation_t& outReservation) noexcept
{
    return mImpl->enqueueTyped<OverlapPlanPolicy>(selection, std::move(launch), outReservation);
}

FamilyBatchPlanAdmission FamilyBatchPlanOperationOwner::enqueueCuda(TargetSelection          selection,
                                                                    const CudaFamilyLaunch&  launch,
                                                                    CudaFamilyResultColumns& outReservation) noexcept
{
    return mImpl->enqueueCuda(selection, launch, outReservation);
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::fetchColumns(
    FamilyBatchPlanOperationId operation, ovspatialquery_raycast_batch_result_t& outResult) const noexcept
{
    return mImpl->fetchColumnsTyped<RaycastPlanPolicy>(operation, outResult);
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::fetchColumns(
    FamilyBatchPlanOperationId operation, ovspatialquery_sphere_sweep_batch_result_t& outResult) const noexcept
{
    return mImpl->fetchColumnsTyped<SphereSweepPlanPolicy>(operation, outResult);
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::fetchColumns(
    FamilyBatchPlanOperationId operation, ovspatialquery_overlap_batch_result_t& outResult) const noexcept
{
    return mImpl->fetchColumnsTyped<OverlapPlanPolicy>(operation, outResult);
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::fetchCuda(FamilyBatchPlanOperationId operation,
                                                                     CudaFamilyResultColumns& outResult) const noexcept
{
    if (mImpl->common.backend != FamilyBatchPlanBackend::cuda)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const ovspatialquery_api_status_t completed = mImpl->completedAccess(operation);
    if (completed != OVSPATIALQUERY_API_SUCCESS)
    {
        return completed;
    }
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    for (const Impl::ResultSlot& slot : mImpl->results)
    {
        if (slot.occupied && slot.operation == operation.value && slot.ready && slot.cudaOperation != nullptr)
        {
            const CudaFamilyCompletedResultOutcome outcome = slot.cudaOperation->completedResult(outResult);
            switch (outcome)
            {
            case CudaFamilyCompletedResultOutcome::success: return OVSPATIALQUERY_API_SUCCESS;
            case CudaFamilyCompletedResultOutcome::deviceLost: return OVSPATIALQUERY_API_DEVICE_LOST;
            case CudaFamilyCompletedResultOutcome::unavailable:
            case CudaFamilyCompletedResultOutcome::transferFailure:
            case CudaFamilyCompletedResultOutcome::invariantFailure: return OVSPATIALQUERY_API_INTERNAL;
            }
            return OVSPATIALQUERY_API_INTERNAL;
        }
    }
    return OVSPATIALQUERY_API_INVALID_HANDLE;
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::destroyOperation(
    FamilyBatchPlanOperationId& operation) noexcept
{
    if (mImpl->common.backend == FamilyBatchPlanBackend::cuda)
    {
        // CUDA result columns remain device-owned until the caller supplies
        // consumer_done through destroyOperationAfterCuda().  Recoverable
        // failure still signals result_ready and does not waive aggregate
        // consumer retirement. Only device loss has no usable event path, and
        // a HOST_MAPPED plan's terminal launch has no library work left on the
        // device, because its plan confirms the device is done before it reports
        // one (ADR-0077), so its host reader retires it here. A launch whose
        // completion could not be confirmed reports device loss, and its plan
        // then keeps every storage the device may still use.
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        for (const Impl::ResultSlot& slot : mImpl->results)
        {
            if (slot.occupied && slot.operation == operation.value && slot.cudaOperation != nullptr)
            {
                const CudaCompletionState resultState = slot.cudaOperation->resultState();
                const bool                terminal =
                    resultState == CudaCompletionState::complete || resultState == CudaCompletionState::error;
                if (resultState != CudaCompletionState::deviceLost && !(mImpl->common.cudaHostMapped && terminal))
                {
                    return OVSPATIALQUERY_API_BUSY;
                }
            }
        }
    }
    ovspatialquery_op_id_t            value = operation.value;
    const ovspatialquery_api_status_t outcome =
        destroyInstanceOperation(mImpl->instance, InstanceOperationKind::familyBatchPlan, mImpl.get(), value);
    if (outcome == OVSPATIALQUERY_API_SUCCESS)
    {
        operation = {};
    }
    return outcome;
}

class FamilyBatchPlanCudaRetirement final : public InstanceOperationDeviceRetirement
{
  public:
    std::shared_ptr<FamilyBatchPlanOperationOwner::Impl> owner;
    // Set non-null before the instance receives the retirement; reset only by the destructor.
    std::shared_ptr<CudaFamilyOperation> operation;
    uint32_t                             resultIndex = UINT32_MAX;
    ovspatialquery_op_id_t               operationId = OVSPATIALQUERY_INVALID_OP;
    bool                                 transferred = false;

    ~FamilyBatchPlanCudaRetirement() override
    {
        // Drop the reaper's operation reference before making the owning result
        // slot observable as free.  The slot holds the other operation reference,
        // so releaseSlot() then destroys the CUDA operation and its instance device
        // reservation before close() can succeed.  Releasing the slot first left
        // a short window in which plan destruction succeeded while instance
        // destruction still observed a live device result.
        operation.reset();
        if (transferred && owner != nullptr)
        {
            owner->releaseSlot(resultIndex, operationId);
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
            if (const auto hook = gFamilyBatchPlanRetiredSlotHook.load(std::memory_order_acquire))
            {
                hook();
            }
#endif
        }
    }

    ovspatialquery_api_status_t transferConsumerDone(ovspatialquery_cuda_event_t event) noexcept override
    {
        const auto status = operation->transferConsumerDone(event);
        transferred       = status == OVSPATIALQUERY_API_SUCCESS;
        return status;
    }
    bool deviceSafe() noexcept override
    {
        return operation->resultState() != CudaCompletionState::pending &&
               operation->consumerState() != CudaCompletionState::pending && operation->consumerDoneOwnershipReleased();
    }
    void releaseReadyDependencies() noexcept override { operation->releaseReadyDependencies(); }
    void ringWhenDeviceSafe(const std::shared_ptr<DeviceSignal>& signal) noexcept override
    {
        operation->ringWhenConsumed(signal);
    }
};

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::destroyOperationAfterCuda(
    FamilyBatchPlanOperationId& operation, ovspatialquery_cuda_event_t& consumerDone) noexcept
{
    if (mImpl->common.backend != FamilyBatchPlanBackend::cuda || consumerDone.value == 0u)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::shared_ptr<CudaFamilyOperation> cudaOperation;
        uint32_t                             resultIndex = UINT32_MAX;
        {
            std::lock_guard<std::mutex> lock(mImpl->mutex);
            for (uint32_t index = 0u; index < mImpl->results.size(); ++index)
            {
                const Impl::ResultSlot& slot = mImpl->results[index];
                if (slot.occupied && slot.operation == operation.value)
                {
                    cudaOperation = slot.cudaOperation;
                    resultIndex   = index;
                    break;
                }
            }
        }
        if (cudaOperation == nullptr || resultIndex == UINT32_MAX)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        std::shared_ptr<FamilyBatchPlanCudaRetirement> retirement = std::make_shared<FamilyBatchPlanCudaRetirement>();
        retirement->owner                                         = mImpl;
        retirement->operation                                     = std::move(cudaOperation);
        retirement->resultIndex                                   = resultIndex;
        retirement->operationId                                   = operation.value;
        ovspatialquery_op_id_t            value                   = operation.value;
        const ovspatialquery_api_status_t outcome =
            destroyInstanceOperation(mImpl->instance, InstanceOperationKind::familyBatchPlan, mImpl.get(), value,
                                     &consumerDone, std::move(retirement));
        if (outcome == OVSPATIALQUERY_API_SUCCESS)
        {
            operation = {};
        }
        return outcome;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_OPERATION_FAILED;
    }
}

ovspatialquery_api_status_t FamilyBatchPlanOperationOwner::close() noexcept
{
    {
        std::unique_lock<std::mutex> lock(mImpl->mutex, std::try_to_lock);
        if (!lock.owns_lock())
        {
            return OVSPATIALQUERY_API_BUSY;
        }
        if (std::any_of(mImpl->results.begin(), mImpl->results.end(),
                        [](const Impl::ResultSlot& slot) { return slot.occupied; }))
        {
            return OVSPATIALQUERY_API_BUSY;
        }
        if (mImpl->closed)
        {
            return OVSPATIALQUERY_API_SUCCESS;
        }
        // A retired result may still hold Impl after releasing its slot. Drop
        // cached catalogue generations before close succeeds, rather than waiting
        // for that last owner to disappear and keeping geometry spuriously busy.
        mImpl->cudaPlan.reset();
        mImpl->closed = true;
    }
    mImpl->releaseTarget();
    return OVSPATIALQUERY_API_SUCCESS;
}

#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
void FamilyBatchPlanOperationOwner::failNextResourceForTest() noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->failNextResource = true;
}

void FamilyBatchPlanOperationOwner::failNextBackendForTest() noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->failNextBackend = true;
}

void FamilyBatchPlanOperationOwner::failNextInternalForTest() noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->failNextInternal = true;
}
#endif

} // namespace detail
} // namespace ovspatialquery
