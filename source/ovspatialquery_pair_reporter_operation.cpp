/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_pair_reporter_operation.h"

#include "ovspatialquery_cuda_pair_reporter.h"
#include "ovspatialquery_cuda_pair_types.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
namespace
{

// Gathers the objects of one endpoint that changed between the accepted
// version and the resolved one from the change records of the resolved
// version and the versions in between (ADR-0019), with their sorted
// identities; the accepted version itself need not be retained. False when
// the advance recomputes everything instead: a version in between is no
// longer retained, a record says everything changed, or the versions together
// changed more than a quarter of the objects, where recomputation is about as
// cheap. Allocates one flag per object and, for at most a quarter of them, a
// position and an identity.
bool gatherChangedObjects(const PopulationTable& populations, const ResolvedTarget& target,
                          ovspatialquery_ordinal_t acceptedOrdinal, PairChangedObjects& outChanged,
                          std::vector<uint64_t>& outIds)
{
    const ObjectTable& objects = target.generation.objects();
    const size_t       limit   = VersionChangeRecord::positionLimit(objects.size());
    outChanged.positions.clear();
    outChanged.positions.reserve(limit);
    outChanged.flags.assign(objects.size(), 0u);
    GenerationLease version = target.generation;
    while (version.ordinal() != acceptedOrdinal)
    {
        const VersionChangeRecord& record = version.changeRecord();
        if (record.everything || record.baseOrdinal < acceptedOrdinal)
        {
            return false;
        }
        for (const uint32_t position : record.positions)
        {
            if (position >= objects.size())
            {
                return false;
            }
            if (outChanged.flags[position] == 0u)
            {
                if (outChanged.positions.size() == limit)
                {
                    return false;
                }
                outChanged.flags[position] = 1u;
                outChanged.positions.push_back(position);
            }
        }
        if (record.baseOrdinal == acceptedOrdinal)
        {
            break;
        }
        GenerationLease base;
        if (populations.pinExact(target.population, record.baseOrdinal, base) != OVSPATIALQUERY_API_SUCCESS)
        {
            return false;
        }
        version = std::move(base);
    }
    std::sort(outChanged.positions.begin(), outChanged.positions.end());
    outIds.clear();
    outIds.reserve(outChanged.positions.size());
    for (const uint32_t position : outChanged.positions)
    {
        outIds.push_back(objects[position].id.value);
    }
    std::sort(outIds.begin(), outIds.end());
    return true;
}

#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
std::atomic<uint64_t> gPatchedAdvanceCount{0u};
std::atomic<uint64_t> gPatchedCudaAdvanceCount{0u};

bool sameTransition(const PairReporterTransition& first, const PairReporterTransition& second)
{
    if (first.outcome != second.outcome)
    {
        return false;
    }
    if (first.outcome != PairReporterResultOutcome::advanced)
    {
        return true;
    }
    return first.entered == second.entered && first.exited == second.exited && first.replacement.has_value() &&
           second.replacement.has_value() && first.replacement->firstOrdinal == second.replacement->firstOrdinal &&
           first.replacement->secondOrdinal == second.replacement->secondOrdinal &&
           first.replacement->activePairs == second.replacement->activePairs;
}
#endif

bool validConfig(const PairReporterOperationConfig& config)
{
    if (!std::isfinite(config.contactTolerance) || config.contactTolerance < 0.0 || config.resultSlotCount == 0u)
    {
        return false;
    }
    const bool samePopulation = config.first.target.population == config.second.target.population;
    const bool identical      = samePopulation && config.first.target.view == config.second.target.view &&
                           participationEqual(config.first.queryBits, config.second.queryBits);
    switch (config.relation)
    {
    case PairEndpointRelation::distinctPopulations: return !samePopulation;
    case PairEndpointRelation::samePopulationDistinctSelections: return samePopulation && !identical;
    case PairEndpointRelation::identicalSelection: return samePopulation && identical;
    }
    return false;
}

// A launch failure's admission status: the budget, memory, support and
// device-loss statuses pass through; any other is INTERNAL.
ovspatialquery_api_status_t launchAdmissionStatus(ovspatialquery_api_status_t status)
{
    return status == OVSPATIALQUERY_API_DEVICE_BUDGET || status == OVSPATIALQUERY_API_HOST_BUDGET ||
                   status == OVSPATIALQUERY_API_OUT_OF_MEMORY || status == OVSPATIALQUERY_API_NOT_SUPPORTED ||
                   status == OVSPATIALQUERY_API_DEVICE_LOST
               ? status
               : OVSPATIALQUERY_API_INTERNAL;
}

bool cudaResultDevice(const PairReporterOperationConfig& config) { return config.resultDevice.device_type == kDLCUDA; }

} // namespace

struct PairReporterOperationOwner::Impl : std::enable_shared_from_this<PairReporterOperationOwner::Impl>
{
    struct ResultSlot
    {
        bool                                     occupied  = false;
        bool                                     ready     = false;
        ovspatialquery_op_id_t                   operation = OVSPATIALQUERY_INVALID_OP;
        PairReporterResult                       result;
        std::optional<PairReporterAcceptedState> pendingReplacement;
        // A CUDA reporter's advance: its device work, the live device result
        // it counts against, its delta counts, and whether a fetch returned
        // its device views, which then retire only after their consumers.
        std::shared_ptr<CudaPairAdvance> cudaAdvance;
        std::shared_ptr<void>            deviceResultReservation;
        uint64_t                         cudaEnteredCount = 0u;
        uint64_t                         cudaExitedCount  = 0u;
        bool                             viewsReturned    = false;
        // The advance's place in the reporter's admission order, from 1; its
        // outcome is published after the one before it (ADR-0022 decision 5).
        uint64_t sequence = 0u;
    };

    Impl(PopulationTable& populationTable, ovspatialquery_instance_handle_t operationInstance,
         PairReporterOperationConfig value)
        : populations(populationTable), resolver(populationTable), instance(operationInstance), config(std::move(value))
    {
        results.resize(config.resultSlotCount);
        for (ResultSlot& slot : results)
        {
            if (!cudaResultDevice(config))
            {
                slot.result.entered.reserve(static_cast<size_t>(config.deltaPairCapacity));
                slot.result.exited.reserve(static_cast<size_t>(config.deltaPairCapacity));
            }
        }
        retainedPopulations.reserve(2u);
        retainedPopulations.push_back(config.first.target.population);
        retainedPopulations.push_back(config.second.target.population);
    }

    ~Impl() { releaseReferences(); }

    ovspatialquery_api_status_t initialize() noexcept
    {
        const ovspatialquery_api_status_t retainStatus = populations.retainPersistentReferences(retainedPopulations);
        if (retainStatus == OVSPATIALQUERY_API_INVALID_HANDLE)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        if (retainStatus == OVSPATIALQUERY_API_OUT_OF_MEMORY)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        if (retainStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        referencesRetained = true;
        if (cudaResultDevice(config))
        {
            CudaPairReporterConfig device{};
            device.device                            = config.resultDevice;
            device.relation                          = config.relation;
            device.contactTolerance                  = config.contactTolerance;
            device.firstQueryBits                    = config.first.queryBits;
            device.secondQueryBits                   = config.second.queryBits;
            device.firstSelection                    = std::shared_ptr<const void>(config.first.target.view);
            device.secondSelection                   = std::shared_ptr<const void>(config.second.target.view);
            device.activePairCapacity                = config.activePairCapacity;
            device.deltaPairCapacity                 = config.deltaPairCapacity;
            device.resultSlotCount                   = config.resultSlotCount;
            const ovspatialquery_api_status_t status = CudaPairReporterDevice::create(device, cuda);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status == OVSPATIALQUERY_API_NOT_SUPPORTED
                           ? OVSPATIALQUERY_API_NOT_SUPPORTED
                           : (status == OVSPATIALQUERY_API_OUT_OF_MEMORY ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                                                         : OVSPATIALQUERY_API_INTERNAL);
            }
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }

    void releaseReferences() noexcept
    {
        if (referencesRetained)
        {
            populations.finishPersistentReferences(retainedPopulations);
            referencesRetained = false;
        }
    }

    void clearSlot(ResultSlot& slot) noexcept
    {
        slot.result.resultSlot              = UINT32_MAX;
        slot.result.outcome                 = PairReporterResultOutcome::advanced;
        slot.result.firstSourceOrdinal      = 0;
        slot.result.secondSourceOrdinal     = 0;
        slot.result.acceptedActivePairCount = 0u;
        slot.result.entered.clear();
        slot.result.exited.clear();
        slot.pendingReplacement.reset();
        slot.cudaAdvance.reset();
        slot.deviceResultReservation.reset();
        slot.cudaEnteredCount = 0u;
        slot.cudaExitedCount  = 0u;
        slot.viewsReturned    = false;
        slot.sequence         = 0u;
        slot.ready            = false;
        slot.occupied         = false;
        slot.operation        = OVSPATIALQUERY_INVALID_OP;
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

    // The occupied result slot of `operation`, or null; the caller holds `mutex`.
    ResultSlot* occupiedSlot(ovspatialquery_op_id_t operation) noexcept
    {
        const auto found =
            std::find_if(results.begin(), results.end(),
                         [operation](const ResultSlot& slot) { return slot.occupied && slot.operation == operation; });
        return found == results.end() ? nullptr : &*found;
    }

    // A result slot's storage. A CUDA reporter's rows are on the device: a
    // completed advance counts its deltas, a reserved one names the
    // capacity-shaped storage the advance writes (ADR-0022 decision 2).
    PairReporterResultStorageView storageView(const ResultSlot& slot, bool completed) const noexcept
    {
        PairReporterResultStorageView view{};
        view.resultSlot              = slot.result.resultSlot;
        view.outcome                 = slot.result.outcome;
        view.firstSourceOrdinal      = slot.result.firstSourceOrdinal;
        view.secondSourceOrdinal     = slot.result.secondSourceOrdinal;
        view.acceptedActivePairCount = slot.result.acceptedActivePairCount;
        view.entered                 = slot.result.entered.data();
        view.enteredCount            = static_cast<uint64_t>(slot.result.entered.size());
        view.exited                  = slot.result.exited.data();
        view.exitedCount             = static_cast<uint64_t>(slot.result.exited.size());
        if (cuda != nullptr)
        {
            const bool advanced = completed && slot.result.outcome == PairReporterResultOutcome::advanced;
            view.device         = config.resultDevice;
            view.deviceEntered  = cuda->enteredRows(slot.result.resultSlot);
            view.deviceExited   = cuda->exitedRows(slot.result.resultSlot);
            view.deviceCounts   = cuda->countsRows(slot.result.resultSlot);
            view.enteredCount   = advanced ? slot.cudaEnteredCount : 0u;
            view.exitedCount    = advanced ? slot.cudaExitedCount : 0u;
            view.resultReady    = cuda->readyEvent(slot.result.resultSlot);
        }
        return view;
    }

    void releaseSlot(uint32_t index, ovspatialquery_op_id_t operation) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (index < results.size() && results[index].occupied && results[index].operation == operation)
        {
            clearSlot(results[index]);
        }
    }

    InstanceOperationExecution execute(uint32_t resultIndex, ovspatialquery_op_id_t operation,
                                       const ResolvedTarget& first, const ResolvedTarget& second) noexcept
    {
        if (cuda != nullptr)
        {
            return executeCuda(resultIndex, operation, first, second);
        }
#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (std::exchange(failNextInternal, false))
            {
                return {OVSPATIALQUERY_API_INTERNAL, "pair reporter forced internal failure"};
            }
        }
#endif
        PairMaterializationRequest request{};
        request.first              = {first, config.first.queryBits};
        request.second             = {second, config.second.queryBits};
        request.relation           = config.relation;
        request.contactTolerance   = config.contactTolerance;
        request.activePairCapacity = config.activePairCapacity;

        PairAdvance                      advance         = advanceAcceptedSet(request);
        const PairMaterializationOutcome materialization = advance.materialization;
        PairReporterTransition&          transition      = advance.transition;
        if (materialization == PairMaterializationOutcome::resourceExhausted)
        {
            return {
                OVSPATIALQUERY_API_OUT_OF_MEMORY,
                "pair reporter materialization exhausted resources",
            };
        }
        if (materialization == PairMaterializationOutcome::internalFailure)
        {
            return {
                OVSPATIALQUERY_API_INTERNAL,
                "pair reporter materialization failed internally",
            };
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (resultIndex >= results.size() || !results[resultIndex].occupied ||
            results[resultIndex].operation != operation)
        {
            return {
                OVSPATIALQUERY_API_INTERNAL,
                "pair reporter result ownership changed during execution",
            };
        }
        // Admission cleared the slot. Its rows are assigned, not moved, so they
        // keep the capacity reserved for the deltas and the commit allocates nothing.
        PairReporterResult& stored = results[resultIndex].result;
        stored.outcome             = transition.outcome;
        stored.firstSourceOrdinal  = first.ordinal;
        stored.secondSourceOrdinal = second.ordinal;
        stored.acceptedActivePairCount =
            accepted.has_value() ? static_cast<uint64_t>(accepted->activePairs.size()) : 0u;
        if (transition.outcome == PairReporterResultOutcome::advanced)
        {
            stored.acceptedActivePairCount = static_cast<uint64_t>(transition.replacement->activePairs.size());
            stored.entered.assign(transition.entered.begin(), transition.entered.end());
            stored.exited.assign(transition.exited.begin(), transition.exited.end());
        }
        results[resultIndex].pendingReplacement = std::move(transition.replacement);
        results[resultIndex].ready              = true;
        return {};
    }

    // Waits for an admitted CUDA advance without synchronizing the host on the
    // device, then reads the outcome the device reported. Test builds also
    // recompute the advance on the CPU and exit on any difference.
    InstanceOperationExecution executeCuda(uint32_t resultIndex, ovspatialquery_op_id_t operation,
                                           const ResolvedTarget& first, const ResolvedTarget& second) noexcept
    {
        std::shared_ptr<CudaPairAdvance> advance;
        uint64_t                         sequence = 0u;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (resultIndex >= results.size() || !results[resultIndex].occupied ||
                results[resultIndex].operation != operation || results[resultIndex].cudaAdvance == nullptr)
            {
                return {OVSPATIALQUERY_API_INTERNAL, "CUDA pair reporter result ownership is invalid"};
            }
            advance  = results[resultIndex].cudaAdvance;
            sequence = results[resultIndex].sequence;
            // Outcomes are published in admission order, which the device
            // computed them in, so the accepted ordinals and the baseline buffer
            // move in step. A worker that reaches a later advance first hands it
            // back to the queue instead of waiting on another worker, or on
            // itself when it is alone.
            if (publishedSequence + 1u < sequence)
            {
                return {OVSPATIALQUERY_API_NOT_READY, nullptr};
            }
        }
        // Spin briefly for work that finishes within a scheduling quantum, then
        // block on the slot's blocking-sync event so a long advance neither burns
        // a core nor waits out a sleep (a whole 15.6 ms timer tick on Windows by
        // default). Poll with sleeps only when the event cannot be waited on.
        const std::chrono::steady_clock::time_point spinUntil =
            std::chrono::steady_clock::now() + std::chrono::microseconds(200);
        bool                blocked = false;
        CudaCompletionState state   = CudaCompletionState::pending;
        while ((state = CudaPairReporterDevice::resultState(*advance)) == CudaCompletionState::pending)
        {
            if (std::chrono::steady_clock::now() < spinUntil)
            {
                std::this_thread::yield();
            }
            else if (!blocked)
            {
                blocked = true;
                CudaPairReporterDevice::waitForResult(*advance);
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        CudaPairReporterDevice::releaseReadyDependencies(*advance);
        if (state == CudaCompletionState::deviceLost)
        {
            return {OVSPATIALQUERY_API_DEVICE_LOST, "CUDA pair reporter device was lost"};
        }
        if (state != CudaCompletionState::complete)
        {
            return {OVSPATIALQUERY_API_OPERATION_FAILED, "CUDA pair reporter advance failed"};
        }
        CudaPairAdvanceResult device{};
        if (!CudaPairReporterDevice::result(*advance, device))
        {
            return {OVSPATIALQUERY_API_INTERNAL, "CUDA pair reporter outcome is invalid"};
        }
        if (device.outcome == cuda_pairs::kOutcomeFailed)
        {
            // A selected admitted publication failed (ADR-0021); the device
            // wrote FAILED to the slot's counts and the baseline stays.
            return {OVSPATIALQUERY_API_OPERATION_FAILED,
                    "CUDA pair reporter advance failed: a selected publication failed"};
        }
        try
        {
            std::optional<PairReporterAcceptedState> replacement;
            if (device.outcome == 0u)
            {
                replacement = PairReporterAcceptedState{first.ordinal, second.ordinal, {}};
            }
#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
            {
                PairMaterializationRequest request{};
                request.first              = {first, config.first.queryBits};
                request.second             = {second, config.second.queryBits};
                request.relation           = config.relation;
                request.contactTolerance   = config.contactTolerance;
                request.activePairCapacity = config.activePairCapacity;
                const PairAdvance complete = recompute(request);
                const uint32_t    expected = static_cast<uint32_t>(complete.transition.outcome);
                bool              same     = (complete.materialization == PairMaterializationOutcome::success ||
                             complete.materialization == PairMaterializationOutcome::activeCapacity) &&
                            expected == device.outcome;
                if (same && expected == 0u)
                {
                    std::vector<ovspatialquery_pair_t> entered;
                    std::vector<ovspatialquery_pair_t> exited;
                    same = device.acceptedCount == complete.transition.replacement->activePairs.size() &&
                           cuda->downloadForTest(resultIndex, device.enteredCount, device.exitedCount, entered, exited);
                    std::sort(entered.begin(), entered.end(), PairLess{});
                    std::sort(exited.begin(), exited.end(), PairLess{});
                    same = same && entered == complete.transition.entered && exited == complete.transition.exited;
                }
                else if (same)
                {
                    same = device.acceptedCount ==
                           (accepted.has_value() ? static_cast<uint64_t>(accepted->activePairs.size()) : 0u);
                }
                if (!same)
                {
                    std::fprintf(stderr, "CUDA pair reporter advance differs from the CPU recomputation\n");
                    std::fflush(stderr);
                    std::_Exit(EXIT_FAILURE);
                }
                if (device.patched)
                {
                    gPatchedCudaAdvanceCount.fetch_add(1u, std::memory_order_relaxed);
                }
                // The shadow baseline the next check compares against.
                replacement = complete.transition.replacement;
            }
#endif
            std::lock_guard<std::mutex> lock(mutex);
            if (resultIndex >= results.size() || !results[resultIndex].occupied ||
                results[resultIndex].operation != operation)
            {
                return {OVSPATIALQUERY_API_INTERNAL, "CUDA pair reporter result ownership changed during execution"};
            }
            ResultSlot& slot = results[resultIndex];
            // The device writes the public outcome values.
            static_assert(static_cast<uint32_t>(PairReporterResultOutcome::deltaCapacity) == 2u, "outcome numbering");
            slot.result.outcome                 = static_cast<PairReporterResultOutcome>(device.outcome);
            slot.result.firstSourceOrdinal      = first.ordinal;
            slot.result.secondSourceOrdinal     = second.ordinal;
            slot.result.acceptedActivePairCount = device.acceptedCount;
            slot.cudaEnteredCount               = device.enteredCount;
            slot.cudaExitedCount                = device.exitedCount;
            slot.pendingReplacement             = std::move(replacement);
            slot.ready                          = true;
            return {};
        }
        catch (...)
        {
            return {OVSPATIALQUERY_API_OUT_OF_MEMORY, "CUDA pair reporter result commit exhausted resources"};
        }
    }

    // Both ways to advance read `accepted`, and patching writes
    // `spareActivePairs`, without the lock: both change only at this
    // operation's terminal publication or while the reporter is idle.
    struct PairAdvance
    {
        bool                       patched         = false;
        PairMaterializationOutcome materialization = PairMaterializationOutcome::internalFailure;
        PairReporterTransition     transition;
    };

    // An active overflow is the one unsuccessful materialization whose
    // transition is read; callers stop on the others first.
    static PairReporterTransition outcomeOnly(PairMaterializationOutcome materialization)
    {
        PairReporterTransition transition;
        if (materialization == PairMaterializationOutcome::activeCapacity)
        {
            transition.outcome = PairReporterResultOutcome::activeCapacity;
        }
        return transition;
    }

    // Recomputes every current pair and compares the set with the accepted one.
    PairAdvance recompute(const PairMaterializationRequest& request) const
    {
        std::vector<ovspatialquery_pair_t> activePairs;
        const PairMaterializationOutcome   outcome = materializePairSet(request, activePairs);
        return {false, outcome,
                outcome == PairMaterializationOutcome::success
                    ? evaluatePairReporterTransition(accepted, request.first.target.ordinal,
                                                     request.second.target.ordinal, std::move(activePairs),
                                                     config.deltaPairCapacity)
                    : outcomeOnly(outcome)};
    }

    // Patches the accepted set from the change records of the versions in
    // between (ADR-0019), evaluating only the pairs of changed objects, and
    // recomputes when the records cannot say what changed.
    PairAdvance patchOrRecompute(const PairMaterializationRequest& request)
    {
        if (!accepted.has_value())
        {
            return recompute(request);
        }
        PairChangedObjects    firstChanged;
        PairChangedObjects    secondChanged;
        std::vector<uint64_t> firstIds;
        std::vector<uint64_t> secondIds;
        const bool            samePopulation = request.first.target.population == request.second.target.population;
        if (!gatherChangedObjects(populations, request.first.target, accepted->firstOrdinal, firstChanged, firstIds) ||
            (!samePopulation && !gatherChangedObjects(populations, request.second.target, accepted->secondOrdinal,
                                                      secondChanged, secondIds)))
        {
            // Release the gathered sets first: admission charged the larger
            // path, not both.
            firstChanged  = {};
            secondChanged = {};
            firstIds      = {};
            secondIds     = {};
            return recompute(request);
        }
        std::vector<ovspatialquery_pair_t> changedPairs;
        const PairMaterializationOutcome   outcome =
            materializeChangedPairs(request, firstChanged, samePopulation ? firstChanged : secondChanged, changedPairs);
        return {true, outcome,
                outcome == PairMaterializationOutcome::success
                    ? evaluatePatchedPairReporterTransition(
                          *accepted, request.first.target.ordinal, request.second.target.ordinal, changedPairs,
                          firstIds, samePopulation ? firstIds : secondIds, config.activePairCapacity,
                          config.deltaPairCapacity, spareActivePairs)
                    : outcomeOnly(outcome)};
    }

    PairAdvance advanceAcceptedSet(const PairMaterializationRequest& request) noexcept
    {
        try
        {
            PairAdvance advance = patchOrRecompute(request);
#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
            // Test builds recompute every patched advance and stop on any difference.
            if (advance.patched && (advance.materialization == PairMaterializationOutcome::success ||
                                    advance.materialization == PairMaterializationOutcome::activeCapacity))
            {
                const PairAdvance complete = recompute(request);
                if ((complete.materialization != PairMaterializationOutcome::success &&
                     complete.materialization != PairMaterializationOutcome::activeCapacity) ||
                    !sameTransition(advance.transition, complete.transition))
                {
                    std::fprintf(stderr, "patched pair reporter advance differs from the complete recomputation\n");
                    std::fflush(stderr);
                    std::_Exit(EXIT_FAILURE);
                }
                gPatchedAdvanceCount.fetch_add(1u, std::memory_order_relaxed);
            }
#endif
            return advance;
        }
        catch (const std::bad_alloc&)
        {
            return {false, PairMaterializationOutcome::resourceExhausted, {}};
        }
        catch (const std::length_error&)
        {
            return {false, PairMaterializationOutcome::resourceExhausted, {}};
        }
        catch (...)
        {
            return {false, PairMaterializationOutcome::internalFailure, {}};
        }
    }

    PopulationTable&                         populations;
    TargetResolver                           resolver;
    const ovspatialquery_instance_handle_t   instance;
    PairReporterOperationConfig              config;
    std::vector<PopulationKey>               retainedPopulations;
    bool                                     referencesRetained = false;
    bool                                     closed             = false;
    mutable std::mutex                       mutex;
    std::vector<ResultSlot>                  results;
    std::optional<PairReporterAcceptedState> accepted;
    // The previous accepted set's buffer, which a patched advance writes its
    // replacement into: a fresh buffer each advance cost its page faults
    // again. It is charged with the reporter.
    std::vector<ovspatialquery_pair_t> spareActivePairs;
    bool                               operationPending = false;
    // A CUDA result device's reporter state; `accepted` then holds only its
    // ordinals (and, in test builds, the pairs a CPU recomputation checks).
    // Such a reporter admits up to its result slots of advances at once
    // (ADR-0022 decision 5): `frontier` is the endpoints of the newest, which
    // stale and backward checks compare with while any is pending, and the
    // sequence numbers order their terminal publication.
    std::shared_ptr<CudaPairReporterDevice> cuda;
    uint32_t                                pendingCudaAdvances = 0u;
    std::optional<CudaPairBaseline>         frontier;
    uint64_t                                admittedSequence  = 0u;
    uint64_t                                publishedSequence = 0u;
#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
    bool failNextInternal = false;
#endif
};

#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
namespace
{
std::atomic<PairReporterTaskPayloadDestroyHook> gPairReporterTaskPayloadDestroyHook{nullptr};

struct PairReporterTaskPayloadDestroyProbe
{
    ~PairReporterTaskPayloadDestroyProbe()
    {
        const PairReporterTaskPayloadDestroyHook hook =
            gPairReporterTaskPayloadDestroyHook.load(std::memory_order_acquire);
        if (hook != nullptr)
        {
            hook();
        }
    }
};
} // namespace

void setPairReporterTaskPayloadDestroyHookForTest(PairReporterTaskPayloadDestroyHook hook) noexcept
{
    gPairReporterTaskPayloadDestroyHook.store(hook, std::memory_order_release);
}

uint64_t patchedPairReporterAdvanceCountForTest() noexcept
{
    return gPatchedAdvanceCount.load(std::memory_order_relaxed);
}

uint64_t patchedCudaPairReporterAdvanceCountForTest() noexcept
{
    return gPatchedCudaAdvanceCount.load(std::memory_order_relaxed);
}
#endif

struct PairReporterRuntimeTask final : OperationTask
{
    InstanceOperationExecution execute() noexcept override
    {
        return owner->execute(resultIndex, operation, first, second);
    }
    void disarmForDeviceRetirement() noexcept override { releaseResultSlot = false; }

    std::shared_ptr<PairReporterOperationOwner::Impl> owner;
    uint32_t                                          resultIndex = UINT32_MAX;
    ovspatialquery_op_id_t                            operation   = OVSPATIALQUERY_INVALID_OP;
    ResolvedTarget                                    first;
    ResolvedTarget                                    second;
    // False once a device retirement owns the result slot.
    bool releaseResultSlot = true;
    // True once terminal publication released this advance's hold on the owner.
    bool published = false;
    // A CUDA advance's admission sequence, zero for a CPU advance.
    uint64_t sequence = 0u;
#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
    PairReporterTaskPayloadDestroyProbe payloadDestroyProbe;
#endif
    std::unique_lock<std::mutex> completionLock;

    // The worker publishes a task it ran, whose owner only an unactivated
    // admission's rollback clears.
    bool beginTerminalPublication(bool executionSucceeded) noexcept override
    {
        try
        {
            completionLock = std::unique_lock<std::mutex>(owner->mutex);
            if (resultIndex >= owner->results.size())
            {
                return false;
            }
            PairReporterOperationOwner::Impl::ResultSlot& result = owner->results[resultIndex];
            if (!result.occupied || result.operation != operation)
            {
                return false;
            }
            if (executionSucceeded && result.pendingReplacement.has_value())
            {
                if (owner->accepted.has_value() && owner->cuda == nullptr)
                {
                    owner->spareActivePairs.swap(owner->accepted->activePairs);
                }
                owner->accepted = std::move(result.pendingReplacement);
                if (owner->cuda != nullptr && result.cudaAdvance != nullptr)
                {
                    // The device wrote the new set into the advance's ring buffer.
                    owner->cuda->accept(*result.cudaAdvance);
                }
            }
            if (owner->cuda != nullptr && result.cudaAdvance != nullptr)
            {
                owner->cuda->release(*result.cudaAdvance);
                if (owner->pendingCudaAdvances != 0u)
                {
                    --owner->pendingCudaAdvances;
                }
            }
            result.pendingReplacement.reset();
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    void endTerminalPublication() noexcept override
    {
        if (!completionLock.owns_lock())
        {
            // Publication was refused; the sequence still moves on so the
            // advance behind this one does not wait forever.
            completionLock = std::unique_lock<std::mutex>(owner->mutex);
        }
        owner->operationPending  = false;
        owner->publishedSequence = std::max(owner->publishedSequence, sequence);
        completionLock.unlock();
        published = true;
        // OperationSlot retains this task until observed operation destroy.
        // The combined task/staging lease deliberately follows that lifetime.
    }

    ~PairReporterRuntimeTask()
    {
        if (owner != nullptr)
        {
            if (releaseResultSlot)
            {
                owner->releaseSlot(resultIndex, operation);
            }
            // An advance that never published still frees the ones behind it;
            // one that did must not free an advance admitted after it.
            if (!published)
            {
                std::lock_guard<std::mutex> lock(owner->mutex);
                owner->operationPending  = false;
                owner->publishedSequence = std::max(owner->publishedSequence, sequence);
            }
        }
    }
};

bool PairReporterOperationOwner::estimateResidentBytes(const PairReporterOperationConfig& config,
                                                       uint64_t&                          outBytes) noexcept
{
    uint64_t total = sizeof(Impl) + sizeof(PairReporterOperationOwner);
    if (cudaResultDevice(config))
    {
        // The accepted set and the deltas live on the device.
        if (!addHostCharge(config.resultSlotCount, sizeof(Impl::ResultSlot), total) ||
            !addHostCharge(2u, sizeof(PopulationKey), total) ||
            !addHostCharge(1u, CudaPairReporterDevice::residentHostBytes(config.resultSlotCount), total))
        {
            return false;
        }
        outBytes = total;
        return true;
    }
    uint64_t deltaEntries = 0u;
    if (!addHostCharge(config.resultSlotCount, config.deltaPairCapacity, deltaEntries) ||
        !addHostCharge(config.resultSlotCount, sizeof(Impl::ResultSlot), total) ||
        !addHostCharge(deltaEntries, 2u * sizeof(ovspatialquery_pair_t), total) ||
        !addHostCharge(config.activePairCapacity, 2u * sizeof(ovspatialquery_pair_t), total) ||
        !addHostCharge(2u, sizeof(PopulationKey), total))
    {
        return false;
    }
    outBytes = total;
    return true;
}

PairReporterOperationOwner::PairReporterOperationOwner(std::shared_ptr<Impl> impl) noexcept : mImpl(std::move(impl)) {}

PairReporterOperationOwner::~PairReporterOperationOwner() = default;

ovspatialquery_api_status_t PairReporterOperationOwner::destroy(
    std::unique_ptr<PairReporterOperationOwner>& owner) noexcept
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

ovspatialquery_api_status_t PairReporterOperationOwner::create(
    PopulationTable& populations, ovspatialquery_instance_handle_t instance, const PairReporterOperationConfig& config,
    std::unique_ptr<PairReporterOperationOwner>& outOwner) noexcept
{
    if (instance == OVSPATIALQUERY_INVALID_INSTANCE || !validConfig(config))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::shared_ptr<Impl>             impl    = std::make_shared<Impl>(populations, instance, config);
        const ovspatialquery_api_status_t outcome = impl->initialize();
        if (outcome != OVSPATIALQUERY_API_SUCCESS)
        {
            return outcome;
        }
        std::unique_ptr<PairReporterOperationOwner> candidate(new PairReporterOperationOwner(std::move(impl)));
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

PairReporterAdmission PairReporterOperationOwner::enqueue(const PairReporterAdvance& advance) noexcept
{
    PairReporterAdmission        admission;
    InstanceOperationReservation reservation{};
    // Nothing throws once the result slot is taken, so a rejection here only
    // abandons the reservation.
    const auto reject = [&](ovspatialquery_api_status_t outcome) noexcept
    {
        abandonInstanceOperation(mImpl->instance, reservation);
        admission.outcome = outcome;
        return admission;
    };
    if (mImpl->cuda != nullptr)
    {
        mImpl->reclaimRetiredSlotsWhenFull();
    }
    try
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (mImpl->closed)
        {
            admission.outcome = OVSPATIALQUERY_API_INVALID_HANDLE;
            return admission;
        }
        const std::array<TargetRequest, 2u> requests{TargetRequest{mImpl->config.first.target, advance.first},
                                                     TargetRequest{mImpl->config.second.target, advance.second}};
        std::array<ResolvedTarget, 2u>      resolved{};
        const TargetResolveOutcome          resolveOutcome = mImpl->resolver.resolvePair(requests, resolved);
        if (resolveOutcome != TargetResolveOutcome::success)
        {
            admission.outcome = admissionStatus(resolveOutcome);
            return admission;
        }
        // A CUDA reporter on the version's device borrows a device-prepared
        // version's tables over the whole population and recomputes its pairs
        // (ADR-0024 decision 4); CPU reporters and view targets need host objects.
        for (const ResolvedTarget& endpoint : resolved)
        {
            if (!endpoint.generation.preparedOnDevice())
            {
                continue;
            }
            const std::shared_ptr<const CudaPopulationVersion> version = endpoint.generation.cudaVersion();
            if (!cudaResultDevice(mImpl->config) || version == nullptr || endpoint.generation.hasSelection() ||
                version->description().device.device_id != mImpl->config.resultDevice.device_id)
            {
                admission.outcome = OVSPATIALQUERY_API_NOT_SUPPORTED;
                return admission;
            }
        }
        // Stale and backward are judged against the newest admitted endpoints
        // while CUDA advances are pending, else against the accepted ones.
        std::optional<CudaPairBaseline> reference;
        if (mImpl->cuda != nullptr && mImpl->pendingCudaAdvances != 0u && mImpl->frontier.has_value())
        {
            reference = mImpl->frontier;
        }
        else if (mImpl->accepted.has_value())
        {
            reference = CudaPairBaseline{mImpl->accepted->firstOrdinal, mImpl->accepted->secondOrdinal};
        }
        if (reference.has_value())
        {
            if (resolved[0].ordinal < reference->firstOrdinal || resolved[1].ordinal < reference->secondOrdinal)
            {
                admission.outcome = OVSPATIALQUERY_API_BACKWARD_ORDINAL;
                return admission;
            }
            if (resolved[0].ordinal == reference->firstOrdinal && resolved[1].ordinal == reference->secondOrdinal)
            {
                admission.outcome = OVSPATIALQUERY_API_STALE_ORDINAL;
                return admission;
            }
        }
        if (mImpl->config.first.target.population == mImpl->config.second.target.population &&
            resolved[0].ordinal != resolved[1].ordinal)
        {
            admission.outcome = OVSPATIALQUERY_API_ORDINAL_MISMATCH;
            return admission;
        }
        // A CPU reporter has one advance pending; a CUDA reporter as many as it
        // has result slots, which the slot search below bounds.
        if (mImpl->operationPending && mImpl->cuda == nullptr)
        {
            admission.outcome = OVSPATIALQUERY_API_BUSY;
            return admission;
        }

        const ovspatialquery_api_status_t reserveOutcome =
            reserveInstanceOperation(mImpl->instance, InstanceOperationKind::pairReporter, mImpl.get(), reservation);
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
        uint32_t resultIndex = UINT32_MAX;
        for (uint32_t index = 0u; index < mImpl->results.size(); ++index)
        {
            if (!mImpl->results[index].occupied)
            {
                resultIndex = index;
                break;
            }
        }
        if (resultIndex == UINT32_MAX)
        {
            abandonInstanceOperation(mImpl->instance, reservation);
            admission.outcome = OVSPATIALQUERY_API_RESULT_CAPACITY;
            return admission;
        }
        // A CUDA advance's results take a device result slot of the instance.
        const bool            cuda = mImpl->cuda != nullptr;
        std::shared_ptr<void> deviceResult;
        if (cuda)
        {
            const ovspatialquery_api_status_t reservedResult =
                reserveInstanceDeviceResult(mImpl->instance, deviceResult);
            if (reservedResult != OVSPATIALQUERY_API_SUCCESS)
            {
                abandonInstanceOperation(mImpl->instance, reservation);
                admission.outcome = reservedResult == OVSPATIALQUERY_API_RESULT_CAPACITY
                                        ? OVSPATIALQUERY_API_RESULT_CAPACITY
                                        : OVSPATIALQUERY_API_OUT_OF_MEMORY;
                return admission;
            }
        }
        HostLease executionLease;
        if (mImpl->config.hostLedger != nullptr)
        {
            uint64_t           executionPeak = 0u;
            constexpr uint64_t taskPayload   = sizeof(PairReporterRuntimeTask) + 3u * sizeof(void*);
            if (!cuda)
            {
                // An advance from an accepted baseline may patch it instead of
                // recomputing (ADR-0019), so it charges the larger scratch.
                uint64_t scratch = pairTraversalScratchBytes(resolved[0], mImpl->config.first.queryBits, resolved[1],
                                                             mImpl->config.second.queryBits, mImpl->config.relation,
                                                             mImpl->config.contactTolerance);
                if (mImpl->accepted.has_value())
                {
                    scratch = std::max(scratch, pairIncrementalScratchBytes(resolved[0], mImpl->config.first.queryBits,
                                                                            resolved[1], mImpl->config.second.queryBits,
                                                                            mImpl->config.relation));
                }
                // A recomputation's set becomes the replacement; a patch writes
                // it into the spare buffer charged with the reporter.
                if (!addHostCharge(scratch, 1u, executionPeak) ||
                    !addHostCharge(mImpl->config.activePairCapacity, sizeof(ovspatialquery_pair_t), executionPeak) ||
                    !addHostCharge(mImpl->config.deltaPairCapacity, sizeof(ovspatialquery_pair_t), executionPeak) ||
                    !addHostCharge(mImpl->config.deltaPairCapacity, sizeof(ovspatialquery_pair_t), executionPeak))
                {
                    abandonInstanceOperation(mImpl->instance, reservation);
                    admission.outcome = OVSPATIALQUERY_API_OUT_OF_MEMORY;
                    return admission;
                }
            }
            HostTransaction transaction(*mImpl->config.hostLedger);
            if (!transaction.reserve(executionPeak) || !transaction.reserve(taskPayload) ||
                !transaction.commit(executionLease))
            {
                abandonInstanceOperation(mImpl->instance, reservation);
                admission.outcome = OVSPATIALQUERY_API_HOST_BUDGET;
                return admission;
            }
        }

        std::shared_ptr<PairReporterRuntimeTask> task   = std::make_shared<PairReporterRuntimeTask>();
        Impl::ResultSlot&                        result = mImpl->results[resultIndex];
        mImpl->clearSlot(result);
        mImpl->operationPending  = true;
        result.occupied          = true;
        result.operation         = reservation.operation;
        result.result.resultSlot = resultIndex;
        task->owner              = mImpl;
        task->resultIndex        = resultIndex;
        task->operation          = reservation.operation;
        task->first              = std::move(resolved[0]);
        task->second             = std::move(resolved[1]);
        const auto rollback      = [&]()
        {
            // The task must not release the slot under this owner's lock.
            task->owner.reset();
            mImpl->operationPending = false;
            mImpl->clearSlot(result);
        };
        if (cuda)
        {
            // Bind the operation, enqueue the device work, then activate the
            // worker that waits for it.
            if (bindInstanceOperation(mImpl->instance, reservation, task, std::move(executionLease), false) !=
                OVSPATIALQUERY_API_SUCCESS)
            {
                rollback();
                abandonInstanceOperation(mImpl->instance, reservation);
                admission.outcome = OVSPATIALQUERY_API_INTERNAL;
                return admission;
            }
            // Patching needs the baseline the device will compare with; behind a
            // pending advance that baseline is not known yet, so recompute.
            std::optional<CudaPairBaseline> baseline;
            if (mImpl->accepted.has_value() && mImpl->pendingCudaAdvances == 0u)
            {
                baseline = CudaPairBaseline{mImpl->accepted->firstOrdinal, mImpl->accepted->secondOrdinal};
            }
            std::shared_ptr<CudaPairAdvance> work;
            CudaPairAdvanceSync              sync{};
            sync.hasStream = advance.hasStream;
            sync.stream    = advance.stream;
            sync.waitEvent = advance.waitEvent;
            const ovspatialquery_api_status_t launched =
                mImpl->cuda->launch(mImpl->instance, mImpl->populations, mImpl->config.hostLedger, resultIndex,
                                    task->first, task->second, baseline, sync, work);
            if (launched != OVSPATIALQUERY_API_SUCCESS || work == nullptr)
            {
                rollback();
                abandonBoundInstanceOperation(mImpl->instance, reservation);
                admission.outcome = launchAdmissionStatus(
                    launched == OVSPATIALQUERY_API_SUCCESS ? OVSPATIALQUERY_API_INTERNAL : launched);
                return admission;
            }
            result.cudaAdvance             = std::move(work);
            result.deviceResultReservation = std::move(deviceResult);
            result.sequence                = ++mImpl->admittedSequence;
            task->sequence                 = result.sequence;
            mImpl->frontier                = CudaPairBaseline{task->first.ordinal, task->second.ordinal};
            ++mImpl->pendingCudaAdvances;
            if (activateInstanceOperation(mImpl->instance, reservation) != OVSPATIALQUERY_API_SUCCESS)
            {
                // The bound operation owns device work that nothing else could observe.
                std::terminate();
            }
        }
        else
        {
            if (bindInstanceOperation(mImpl->instance, reservation, task, std::move(executionLease), true) !=
                OVSPATIALQUERY_API_SUCCESS)
            {
                rollback();
                abandonInstanceOperation(mImpl->instance, reservation);
                admission.outcome = OVSPATIALQUERY_API_INTERNAL;
                return admission;
            }
        }
        admission.operation  = {reservation.operation};
        admission.resultSlot = resultIndex;
        admission.outcome    = OVSPATIALQUERY_API_SUCCESS;
        return admission;
    }
    catch (const std::bad_alloc&)
    {
        return reject(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    catch (const std::length_error&)
    {
        return reject(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    catch (...)
    {
        return reject(OVSPATIALQUERY_API_INTERNAL);
    }
}

ovspatialquery_api_status_t PairReporterOperationOwner::growCapacity(uint64_t activePairCapacity,
                                                                     uint64_t deltaPairCapacity) noexcept
{
    if (mImpl->cuda != nullptr)
    {
        // Growing needs every result slot free, retired ones included.
        reclaimInstanceDeviceRetirements(mImpl->instance, mImpl.get());
    }
    try
    {
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (activePairCapacity < mImpl->config.activePairCapacity ||
            deltaPairCapacity < mImpl->config.deltaPairCapacity ||
            (activePairCapacity == mImpl->config.activePairCapacity &&
             deltaPairCapacity == mImpl->config.deltaPairCapacity))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        if (mImpl->closed || mImpl->operationPending ||
            std::any_of(mImpl->results.begin(), mImpl->results.end(),
                        [](const Impl::ResultSlot& slot) { return slot.occupied; }))
        {
            return OVSPATIALQUERY_API_BUSY;
        }
        if (mImpl->cuda != nullptr)
        {
            const ovspatialquery_api_status_t status = mImpl->cuda->grow(activePairCapacity, deltaPairCapacity);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status == OVSPATIALQUERY_API_OUT_OF_MEMORY ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                                                  : OVSPATIALQUERY_API_INTERNAL;
            }
            mImpl->config.activePairCapacity = activePairCapacity;
            mImpl->config.deltaPairCapacity  = deltaPairCapacity;
            return OVSPATIALQUERY_API_SUCCESS;
        }

        std::optional<PairReporterAcceptedState> accepted;
        if (mImpl->accepted.has_value())
        {
            PairReporterAcceptedState replacement{};
            replacement.firstOrdinal  = mImpl->accepted->firstOrdinal;
            replacement.secondOrdinal = mImpl->accepted->secondOrdinal;
            replacement.activePairs.reserve(static_cast<size_t>(activePairCapacity));
            replacement.activePairs.assign(mImpl->accepted->activePairs.begin(), mImpl->accepted->activePairs.end());
            accepted = std::move(replacement);
        }

        std::vector<ovspatialquery_pair_t> spareActivePairs;
        spareActivePairs.reserve(static_cast<size_t>(activePairCapacity));

        std::vector<Impl::ResultSlot> results;
        results.resize(mImpl->config.resultSlotCount);
        for (Impl::ResultSlot& slot : results)
        {
            slot.result.entered.reserve(static_cast<size_t>(deltaPairCapacity));
            slot.result.exited.reserve(static_cast<size_t>(deltaPairCapacity));
        }

        mImpl->accepted = std::move(accepted);
        mImpl->spareActivePairs.swap(spareActivePairs);
        mImpl->results.swap(results);
        mImpl->config.activePairCapacity = activePairCapacity;
        mImpl->config.deltaPairCapacity  = deltaPairCapacity;
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

ovspatialquery_api_status_t PairReporterOperationOwner::fetchStorageView(
    PairReporterOperationId operation, PairReporterResultStorageView& outView) const noexcept
{
    InstanceOperationState            state  = InstanceOperationState::pending;
    const ovspatialquery_api_status_t access = queryInstanceOperation(mImpl->instance, operation.value, state);
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
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    const Impl::ResultSlot*     slot = mImpl->occupiedSlot(operation.value);
    if (slot == nullptr || !slot->ready)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    outView = mImpl->storageView(*slot, true);
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PairReporterOperationOwner::reservedStorageView(
    PairReporterOperationId operation, PairReporterResultStorageView& outView) const noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    const Impl::ResultSlot*     slot = mImpl->occupiedSlot(operation.value);
    if (slot == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    outView = mImpl->storageView(*slot, false);
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PairReporterOperationOwner::noteViewsReturned(PairReporterOperationId operation) noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    Impl::ResultSlot*           slot = mImpl->occupiedSlot(operation.value);
    if (slot == nullptr || !slot->ready)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    slot->viewsReturned = mImpl->cuda != nullptr;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PairReporterOperationOwner::destroyOperation(PairReporterOperationId& operation) noexcept
{
    if (mImpl->cuda != nullptr)
    {
        // Returned device views stay readable until destroyOperationAfterCuda
        // retires them; only a lost device waives that.
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        const Impl::ResultSlot*     slot = mImpl->occupiedSlot(operation.value);
        if (slot != nullptr && slot->viewsReturned && slot->cudaAdvance != nullptr &&
            !CudaPairReporterDevice::deviceLost(*slot->cudaAdvance))
        {
            return OVSPATIALQUERY_API_BUSY;
        }
    }
    ovspatialquery_op_id_t            value = operation.value;
    const ovspatialquery_api_status_t outcome =
        destroyInstanceOperation(mImpl->instance, InstanceOperationKind::pairReporter, mImpl.get(), value);
    if (outcome == OVSPATIALQUERY_API_SUCCESS)
    {
        operation = {};
    }
    return outcome;
}

// Keeps a CUDA reporter advance's result slot, and the device storage its
// views read, until the consumers the caller handed over have finished.
class PairReporterCudaRetirement final : public InstanceOperationDeviceRetirement
{
  public:
    std::shared_ptr<PairReporterOperationOwner::Impl> owner;
    // Set non-null before the instance receives the retirement; reset only by the destructor.
    std::shared_ptr<CudaPairAdvance> advance;
    uint32_t                         resultIndex = UINT32_MAX;
    ovspatialquery_op_id_t           operationId = OVSPATIALQUERY_INVALID_OP;

    ~PairReporterCudaRetirement() override
    {
        advance.reset();
        if (owner != nullptr)
        {
            owner->releaseSlot(resultIndex, operationId);
        }
    }

    ovspatialquery_api_status_t transferConsumerDone(ovspatialquery_cuda_event_t event) noexcept override
    {
        return CudaPairReporterDevice::transferConsumerDone(*advance, event);
    }
    bool deviceSafe() noexcept override { return CudaPairReporterDevice::deviceSafe(*advance); }
    void releaseReadyDependencies() noexcept override { CudaPairReporterDevice::releaseReadyDependencies(*advance); }
    void ringWhenDeviceSafe(const std::shared_ptr<DeviceSignal>& signal) noexcept override
    {
        CudaPairReporterDevice::ringWhenConsumed(*advance, signal);
    }
};

ovspatialquery_api_status_t PairReporterOperationOwner::destroyOperationAfterCuda(
    PairReporterOperationId& operation, ovspatialquery_cuda_event_t& consumerDone) noexcept
{
    if (mImpl->cuda == nullptr || consumerDone.value == 0u)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    try
    {
        std::shared_ptr<CudaPairAdvance> advance;
        uint32_t                         resultIndex = UINT32_MAX;
        {
            std::lock_guard<std::mutex> lock(mImpl->mutex);
            if (const Impl::ResultSlot* slot = mImpl->occupiedSlot(operation.value))
            {
                advance     = slot->cudaAdvance;
                resultIndex = static_cast<uint32_t>(slot - mImpl->results.data());
            }
        }
        if (advance == nullptr)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        std::shared_ptr<PairReporterCudaRetirement> retirement = std::make_shared<PairReporterCudaRetirement>();
        retirement->owner                                      = mImpl;
        retirement->advance                                    = std::move(advance);
        retirement->resultIndex                                = resultIndex;
        retirement->operationId                                = operation.value;
        ovspatialquery_op_id_t            value                = operation.value;
        const ovspatialquery_api_status_t outcome              = destroyInstanceOperation(
            mImpl->instance, InstanceOperationKind::pairReporter, mImpl.get(), value, &consumerDone, retirement);
        if (outcome == OVSPATIALQUERY_API_SUCCESS)
        {
            operation = {};
        }
        else
        {
            retirement->owner.reset();
        }
        return outcome;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_OPERATION_FAILED;
    }
}

ovspatialquery_api_status_t PairReporterOperationOwner::close() noexcept
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
        mImpl->closed = true;
    }
    mImpl->releaseReferences();
    return OVSPATIALQUERY_API_SUCCESS;
}

#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
void PairReporterOperationOwner::failNextInternalForTest() noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    mImpl->failNextInternal = true;
}
#endif

} // namespace detail
} // namespace ovspatialquery
