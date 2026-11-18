/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_family_batch_cpu.h"
#include "ovspatialquery_instance_host_ledger.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
namespace
{

bool validTarget(const ResolvedTarget& target)
{
    return target.population.slot != UINT32_MAX && target.generation.initialized() &&
           target.ordinal == target.generation.ordinal();
}

bool validRaycastMode(HitMode mode)
{
    return mode == HitMode::closest || mode == HitMode::any || mode == HitMode::multiple;
}

bool validOverlapMode(HitMode mode) { return mode == HitMode::any || mode == HitMode::multiple; }

bool validPrecision(QueryPrecision precision)
{
    return precision == QueryPrecision::boundsCandidate || precision == QueryPrecision::exact;
}

RowOutcome rowOutcome(SemanticEvaluationOutcome outcome)
{
    switch (outcome)
    {
    case SemanticEvaluationOutcome::success: return RowOutcome::success;
    case SemanticEvaluationOutcome::invalidArgument: return RowOutcome::invalidArgument;
    case SemanticEvaluationOutcome::notSupported: return RowOutcome::notSupported;
    case SemanticEvaluationOutcome::hostBudget:
    case SemanticEvaluationOutcome::outOfMemory:
    case SemanticEvaluationOutcome::internalFailure: return RowOutcome::failed;
    }
    return RowOutcome::failed;
}

void initializeCommon(const ResolvedTarget& target, QueryPrecision precision, uint64_t hitCapacity, size_t rowCount,
                      FamilyBatchResultCommon& result)
{
    result.sourcePopulation = target.population;
    result.sourceOrdinal    = target.ordinal;
    result.truth            = precision == QueryPrecision::exact ? HitTruth::exactSurface : HitTruth::boundsCandidate;
    result.hitCapacity      = hitCapacity;
    result.rowOutcomes.resize(rowCount, RowOutcome::failed);
    result.hitOffsets.resize(rowCount, 0u);
    result.hitCounts.resize(rowCount, 0u);
    result.truncated.resize(rowCount, 0u);
}

// Grants a successful row up to `kept` of its hits from the slot's remaining capacity, in
// row order; `omitted` says the row's semantic limit already left a match out. Returns the
// number of hits stored.
uint64_t grantSuccessfulRow(FamilyBatchResultCommon& common, size_t row, uint64_t kept, bool omitted)
{
    const uint64_t stored   = std::min(kept, common.hitCapacity - common.storedHitCount);
    common.rowOutcomes[row] = RowOutcome::success;
    common.hitOffsets[row]  = common.storedHitCount;
    common.hitCounts[row]   = stored;
    common.truncated[row]   = omitted || stored < kept ? 1u : 0u;
    common.storedHitCount += stored;
    return stored;
}

template <typename Hit>
void storeSuccessfulRow(FamilyBatchResultCommon& common, std::vector<Hit>& destination, size_t row,
                        const std::vector<Hit>& evaluated, uint64_t semanticCapacity)
{
    const uint64_t kept   = std::min(static_cast<uint64_t>(evaluated.size()), semanticCapacity);
    const uint64_t stored = grantSuccessfulRow(common, row, kept, static_cast<uint64_t>(evaluated.size()) > kept);
    destination.insert(destination.end(), evaluated.begin(), evaluated.begin() + static_cast<size_t>(stored));
}

void storeFailedRow(FamilyBatchResultCommon& common, size_t row, SemanticEvaluationOutcome outcome)
{
    common.rowOutcomes[row] = rowOutcome(outcome);
    common.hitOffsets[row]  = common.storedHitCount;
}

struct RaycastBatchPolicy
{
    using Plan   = RaycastBatchPlan;
    using Launch = RaycastBatchLaunch;
    using Row    = RaycastBatchRow;
    using Result = RaycastBatchResult;
    using Hit    = RaycastSemanticHit;

    static bool validPlan(const Plan& plan) noexcept
    {
        return validPrecision(plan.precision) && validRaycastMode(plan.hitMode) &&
               std::isfinite(plan.contactTolerance) && plan.contactTolerance >= 0.0;
    }

    static SemanticEvaluationOutcome evaluate(const ResolvedTarget& target, const Plan& plan, const Row& row,
                                              std::vector<Hit>& outHits) noexcept
    {
        // One crossing beyond the limit tells the store the row was truncated.
        const uint64_t limit = semanticCapacity(plan, row);
        return evaluateSemanticRaycast(target.generation, row.query, plan.precision, plan.hitMode, row.queryBits,
                                       outHits, plan.contactTolerance, plan.hostLedger, plan.allMeshHits,
                                       limit == UINT64_MAX ? limit : limit + 1u);
    }

    static uint64_t semanticCapacity(const Plan& plan, const Row& row) noexcept
    {
        return plan.hitMode == HitMode::multiple ? std::min(plan.maximumHitsPerQuery, row.maximumHits) : 1u;
    }

    // The hits a synchronous MULTIPLE query keeps beyond the row's execution peak.
    // Every-crossing rows charge theirs while they are found.
    static uint64_t retainedHits(const Plan& plan, uint64_t objectCount) noexcept
    {
        return plan.hitMode == HitMode::multiple && !plan.allMeshHits ? std::min(objectCount, plan.maximumHitsPerQuery)
                                                                      : 0u;
    }
};

struct SphereSweepBatchPolicy
{
    using Plan   = SphereSweepBatchPlan;
    using Launch = SphereSweepBatchLaunch;
    using Row    = SphereSweepBatchRow;
    using Result = SphereSweepBatchResult;
    using Hit    = SphereSweepSemanticHit;

    static bool validPlan(const Plan& plan) noexcept
    {
        return validPrecision(plan.precision) && validRaycastMode(plan.hitMode) &&
               std::isfinite(plan.contactTolerance) && plan.contactTolerance >= 0.0;
    }

    static SemanticEvaluationOutcome evaluate(const ResolvedTarget& target, const Plan& plan, const Row& row,
                                              std::vector<Hit>& outHits) noexcept
    {
        // One hit beyond the limit tells the store the row was truncated.
        const uint64_t limit = semanticCapacity(plan, row);
        return evaluateSemanticSphereSweep(target.generation, row.query, plan.precision, plan.hitMode, row.queryBits,
                                           outHits, plan.contactTolerance, plan.hostLedger,
                                           limit == UINT64_MAX ? limit : limit + 1u);
    }

    static uint64_t semanticCapacity(const Plan& plan, const Row&) noexcept
    {
        return plan.hitMode == HitMode::multiple ? plan.maximumHitsPerQuery : 1u;
    }

    static uint64_t retainedHits(const Plan& plan, uint64_t objectCount) noexcept
    {
        return plan.hitMode == HitMode::multiple ? std::min(objectCount, plan.maximumHitsPerQuery) : 0u;
    }
};

struct OverlapBatchPolicy
{
    using Plan   = OverlapBatchPlan;
    using Launch = OverlapBatchLaunch;
    using Row    = OverlapBatchRow;
    using Result = OverlapBatchResult;
    using Hit    = OverlapSemanticHit;

    static bool validPlan(const Plan& plan) noexcept
    {
        return validPrecision(plan.precision) && validOverlapMode(plan.hitMode) &&
               std::isfinite(plan.contactTolerance) && plan.contactTolerance >= 0.0;
    }

    static SemanticEvaluationOutcome evaluate(const ResolvedTarget& target, const Plan& plan, const Row& row,
                                              std::vector<Hit>& outHits) noexcept
    {
        return evaluateSemanticOverlap(target.generation, row.query, plan.precision, plan.hitMode, row.queryBits,
                                       outHits, plan.contactTolerance, plan.hostLedger);
    }

    static uint64_t semanticCapacity(const Plan& plan, const Row& row) noexcept
    {
        return plan.hitMode == HitMode::multiple ? std::min(plan.maximumHitsPerQuery, row.maximumHits) : 1u;
    }

    static uint64_t retainedHits(const Plan& plan, uint64_t objectCount) noexcept
    {
        return plan.hitMode == HitMode::multiple ? std::min(objectCount, plan.maximumHitsPerQuery) : 0u;
    }
};

template <typename Policy>
ovspatialquery_api_status_t executeFamilyRowCpuSync(const TargetResolver& resolver, const TargetRequest& request,
                                                    const typename Policy::Plan& plan, const typename Policy::Row& row,
                                                    ResolvedTarget&                    outTarget,
                                                    std::vector<typename Policy::Hit>& outHits) noexcept
{
    try
    {
        switch (resolver.resolve(request, outTarget))
        {
        case TargetResolveOutcome::success: break;
        case TargetResolveOutcome::invalid: return OVSPATIALQUERY_API_INVALID_HANDLE;
        case TargetResolveOutcome::notReady: return OVSPATIALQUERY_API_NOT_READY;
        case TargetResolveOutcome::ordinalNotRetained: return OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED;
        case TargetResolveOutcome::ordinalMismatch: return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
        case TargetResolveOutcome::resourceExhausted: return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        default: return OVSPATIALQUERY_API_INTERNAL;
        }
        if (outTarget.generation.preparedOnDevice())
        {
            // A readable device version is copied to the host on first read,
            // charged here; plain ON_DEVICE has no host objects (ADR-0024).
            const ovspatialquery_api_status_t status =
                resolver.materializeHostCopy(outTarget.generation, plan.hostLedger);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status == OVSPATIALQUERY_API_NOT_SUPPORTED || status == OVSPATIALQUERY_API_HOST_BUDGET ||
                               status == OVSPATIALQUERY_API_OUT_OF_MEMORY
                           ? status
                           : OVSPATIALQUERY_API_INTERNAL;
            }
        }
        HostLease executionLease;
        if (plan.hostLedger != nullptr)
        {
            const uint64_t objectCount   = static_cast<uint64_t>(outTarget.generation.objects().size());
            uint64_t       executionPeak = 0u;
            if (!estimateFamilyBatchExecutionPeakBytes(outTarget, plan, executionPeak) ||
                !addHostCharge(Policy::retainedHits(plan, objectCount), sizeof(typename Policy::Hit), executionPeak))
            {
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            HostTransaction transaction(*plan.hostLedger);
            if (!transaction.reserve(executionPeak) || !transaction.commit(executionLease))
            {
                return OVSPATIALQUERY_API_HOST_BUDGET;
            }
        }
        switch (Policy::evaluate(outTarget, plan, row, outHits))
        {
        case SemanticEvaluationOutcome::success: return OVSPATIALQUERY_API_SUCCESS;
        case SemanticEvaluationOutcome::invalidArgument: return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        case SemanticEvaluationOutcome::notSupported: return OVSPATIALQUERY_API_NOT_SUPPORTED;
        case SemanticEvaluationOutcome::hostBudget: return OVSPATIALQUERY_API_HOST_BUDGET;
        case SemanticEvaluationOutcome::outOfMemory: return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        default: return OVSPATIALQUERY_API_INTERNAL;
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
}

// A launch splits into chunks of at least this many rows, and only into two or more: a
// primitive row costs about 0.3 µs, so a smaller chunk costs more to hand to a worker than
// it saves.
constexpr size_t kMinimumRowsPerChunk = 64u;
// Chunks per thread that may take part, so threads that finish early share what remains.
constexpr size_t kChunksPerThread = 4u;

/* A launch split into chunks of consecutive rows. Each chunk evaluates its rows into its
 * own staged hits, keeping per row the hits its semantic limit and the slot capacity allow
 * within the chunk; the merge then grants the slot capacity in row order with
 * grantSuccessfulRow, as the row loop does. Every row before the first one that fails the
 * launch is evaluated, so the launch fails with the row loop's status. Staged hits and a
 * helping worker's row execution peak are charged to the plan's ledger while held; when it
 * refuses either, the split is abandoned and the launch runs row by row.
 *
 * A single-hit launch whose rows all fit the slot's capacity stages in place instead: each
 * row's hit goes to its own index of the result's hits, which the plan reserved and charged
 * up front, and the merge moves the hits down to their offsets. Staging allocated per launch
 * was fresh memory every time: a 16,384-ray launch took about 700 page faults and 84 growths,
 * a fifth of its time, where the row loop writes the warm result storage. */
template <typename Policy>
struct ChunkedLaunch
{
    using Hit = typename Policy::Hit;

    const ResolvedTarget*          target        = nullptr;
    const typename Policy::Plan*   plan          = nullptr;
    const typename Policy::Launch* launch        = nullptr;
    typename Policy::Result*       result        = nullptr;
    size_t                         chunkRows     = 0u;
    uint64_t                       executionPeak = 0u;
    // Staging in place: row r's hit, if any, is staged at result->hits[r].
    bool                                   inPlace = false;
    std::thread::id                        caller  = std::this_thread::get_id();
    std::vector<SemanticEvaluationOutcome> outcomes;
    // Declared before the staged hits, so their storage is freed before its charge.
    std::vector<std::vector<HostLease>> stagedLeases;
    std::vector<std::vector<Hit>>       staged;
    std::atomic<size_t>                 firstFailure{SIZE_MAX};
    std::atomic<bool>                   abandoned{false};

    void fail(size_t row, SemanticEvaluationOutcome outcome) noexcept
    {
        outcomes[row]   = outcome;
        size_t previous = firstFailure.load();
        while (row < previous && !firstFailure.compare_exchange_weak(previous, row)) {}
    }

    // Grows a chunk's staged hits to hold `count` more, charging each capacity growth.
    bool reserveStaged(size_t chunk, uint64_t count)
    {
        std::vector<Hit>& hits   = staged[chunk];
        const size_t      needed = hits.size() + static_cast<size_t>(count);
        if (needed <= hits.capacity())
        {
            return true;
        }
        const size_t grown = std::max(needed, 2u * hits.capacity());
        if (plan->hostLedger != nullptr)
        {
            uint64_t        bytes = 0u;
            HostLease       lease;
            HostTransaction transaction(*plan->hostLedger);
            if (!addHostCharge(grown - hits.capacity(), sizeof(Hit), bytes) || !transaction.reserve(bytes) ||
                !transaction.commit(lease))
            {
                return false;
            }
            stagedLeases[chunk].push_back(std::move(lease));
        }
        hits.reserve(grown);
        return true;
    }

    static void run(void* state, size_t chunk) noexcept
    {
        ChunkedLaunch&                           self = *static_cast<ChunkedLaunch*>(state);
        const std::vector<typename Policy::Row>& rows = self.launch->rows;
        const size_t                             end  = std::min(rows.size(), (chunk + 1u) * self.chunkRows);
        size_t                                   row  = chunk * self.chunkRows;
        try
        {
            // The caller runs under the launch's own execution peak; a helper charges its own.
            HostLease helperPeak;
            if (self.plan->hostLedger != nullptr && std::this_thread::get_id() != self.caller)
            {
                HostTransaction transaction(*self.plan->hostLedger);
                if (!transaction.reserve(self.executionPeak) || !transaction.commit(helperPeak))
                {
                    self.abandoned.store(true);
                    return;
                }
            }
            std::vector<Hit> evaluated;
            for (; row < end && row <= self.firstFailure.load(std::memory_order_relaxed) && !self.abandoned.load();
                 ++row)
            {
                const SemanticEvaluationOutcome outcome =
                    Policy::evaluate(*self.target, *self.plan, rows[row], evaluated);
                if (outcome == SemanticEvaluationOutcome::hostBudget)
                {
                    self.abandoned.store(true);
                    return;
                }
                if (outcome == SemanticEvaluationOutcome::outOfMemory)
                {
                    self.fail(row, outcome);
                    return;
                }
                self.outcomes[row] = outcome;
                if (outcome != SemanticEvaluationOutcome::success)
                {
                    continue;
                }
                uint64_t kept =
                    std::min(static_cast<uint64_t>(evaluated.size()), Policy::semanticCapacity(*self.plan, rows[row]));
                if (self.inPlace)
                {
                    if (kept != 0u)
                    {
                        self.result->hits[row] = evaluated.front();
                    }
                }
                else
                {
                    const uint64_t room = self.plan->hitCapacity - static_cast<uint64_t>(self.staged[chunk].size());
                    if (kept > room)
                    {
                        kept = room;
                    }
                    if (!self.reserveStaged(chunk, kept))
                    {
                        self.abandoned.store(true);
                        return;
                    }
                    self.staged[chunk].insert(self.staged[chunk].end(), evaluated.begin(),
                                              evaluated.begin() + static_cast<size_t>(kept));
                }
                self.result->common.hitCounts[row] = kept;
                self.result->common.truncated[row] = static_cast<uint64_t>(evaluated.size()) > kept ? 1u : 0u;
            }
        }
        catch (...)
        {
            // Only allocations throw.
            self.fail(row, SemanticEvaluationOutcome::outOfMemory);
        }
    }

    FamilyBatchOutcome merge() const
    {
        typename Policy::Result& out     = *result;
        const size_t             failure = firstFailure.load();
        if (failure != SIZE_MAX)
        {
            out.hits.clear();
            return outcomes[failure] == SemanticEvaluationOutcome::hostBudget ? FamilyBatchOutcome::hostBudget
                                                                              : FamilyBatchOutcome::outOfMemory;
        }
        const size_t rows = launch->rows.size();
        for (size_t chunk = 0u; chunk < staged.size(); ++chunk)
        {
            size_t next = 0u;
            for (size_t row = chunk * chunkRows; row < std::min(rows, (chunk + 1u) * chunkRows); ++row)
            {
                if (outcomes[row] != SemanticEvaluationOutcome::success)
                {
                    storeFailedRow(out.common, row, outcomes[row]);
                    continue;
                }
                const uint64_t kept   = out.common.hitCounts[row];
                const uint64_t stored = grantSuccessfulRow(out.common, row, kept, out.common.truncated[row] != 0u);
                if (inPlace)
                {
                    // A row's offset is at most its index, and every row before it has moved.
                    if (stored != 0u)
                    {
                        out.hits[static_cast<size_t>(out.common.hitOffsets[row])] = out.hits[row];
                    }
                }
                else
                {
                    out.hits.insert(out.hits.end(), staged[chunk].begin() + static_cast<std::ptrdiff_t>(next),
                                    staged[chunk].begin() + static_cast<std::ptrdiff_t>(next + stored));
                }
                next += static_cast<size_t>(kept);
            }
        }
        if (inPlace)
        {
            out.hits.resize(static_cast<size_t>(out.common.storedHitCount));
        }
        return FamilyBatchOutcome::success;
    }
};

/* Splits a launch across plan.workers when it has rows for two chunks and the instance has
 * an idle worker to lend. Returns false, with the result as the caller initialized it, when
 * the launch should run row by row instead. */
template <typename Policy>
bool executeChunkedLaunch(const ResolvedTarget& target, const typename Policy::Plan& plan,
                          const typename Policy::Launch& launch, typename Policy::Result& outResult,
                          FamilyBatchOutcome& outOutcome)
{
    const size_t   rows = launch.rows.size();
    const uint32_t idle = plan.workers.idleWorkers != nullptr ? plan.workers.idleWorkers(plan.workers.state) : 0u;
    if (plan.workers.run == nullptr || idle == 0u || rows < 2u * kMinimumRowsPerChunk)
    {
        return false;
    }
    const size_t          threads = static_cast<size_t>(idle) + 1u;
    const size_t          chunks  = std::min(rows / kMinimumRowsPerChunk, threads * kChunksPerThread);
    ChunkedLaunch<Policy> chunked;
    chunked.target    = &target;
    chunked.plan      = &plan;
    chunked.launch    = &launch;
    chunked.result    = &outResult;
    chunked.chunkRows = (rows + chunks - 1u) / chunks;
    if (plan.hostLedger != nullptr && !estimateFamilyBatchExecutionPeakBytes(target, plan, chunked.executionPeak))
    {
        return false;
    }
    chunked.outcomes.resize(rows, SemanticEvaluationOutcome::internalFailure);
    chunked.stagedLeases.resize((rows + chunked.chunkRows - 1u) / chunked.chunkRows);
    chunked.staged.resize(chunked.stagedLeases.size());
    // Every Policy::semanticCapacity is 1 outside MULTIPLE, and the result reserved hitCapacity.
    if (plan.hitMode != HitMode::multiple && rows <= plan.hitCapacity)
    {
        chunked.inPlace = true;
        outResult.hits.resize(rows);
    }
    plan.workers.run(plan.workers.state, chunked.staged.size(), &ChunkedLaunch<Policy>::run, &chunked);
    if (chunked.abandoned.load())
    {
        outResult.hits.clear();
        std::fill(outResult.common.hitCounts.begin(), outResult.common.hitCounts.end(), 0u);
        std::fill(outResult.common.truncated.begin(), outResult.common.truncated.end(), 0u);
        return false;
    }
    outOutcome = chunked.merge();
    return true;
}

template <typename Policy>
FamilyBatchOutcome executeTypedFamilyBatchCpuDirect(const ResolvedTarget& target, const typename Policy::Plan& plan,
                                                    const typename Policy::Launch& launch,
                                                    typename Policy::Result&       outResult) noexcept
{
    if (!validTarget(target) || !Policy::validPlan(plan) || launch.rows.size() > plan.maximumQueryCount)
    {
        return FamilyBatchOutcome::invalidArgument;
    }
    try
    {
        outResult.hits.reserve(static_cast<size_t>(plan.hitCapacity));
        outResult.common.rowOutcomes.reserve(launch.rows.size());
        outResult.common.hitOffsets.reserve(launch.rows.size());
        outResult.common.hitCounts.reserve(launch.rows.size());
        outResult.common.truncated.reserve(launch.rows.size());
        initializeCommon(target, plan.precision, plan.hitCapacity, launch.rows.size(), outResult.common);
        outResult.hits.clear();
        FamilyBatchOutcome chunkedOutcome = FamilyBatchOutcome::success;
        if (executeChunkedLaunch<Policy>(target, plan, launch, outResult, chunkedOutcome))
        {
            return chunkedOutcome;
        }
        // Each evaluation clears this first; reusing it keeps single-hit rows allocation-free.
        // A larger row's storage is charged only while that row evaluates, so it is
        // released before the next row, as a fresh vector per row did.
        std::vector<typename Policy::Hit> evaluated;
        for (size_t rowIndex = 0u; rowIndex < launch.rows.size(); ++rowIndex)
        {
            if (evaluated.capacity() > 1u)
            {
                std::vector<typename Policy::Hit>().swap(evaluated);
            }
            const typename Policy::Row&     row     = launch.rows[rowIndex];
            const SemanticEvaluationOutcome outcome = Policy::evaluate(target, plan, row, evaluated);
            if (outcome == SemanticEvaluationOutcome::outOfMemory)
            {
                return FamilyBatchOutcome::outOfMemory;
            }
            if (outcome == SemanticEvaluationOutcome::hostBudget)
            {
                return FamilyBatchOutcome::hostBudget;
            }
            if (outcome != SemanticEvaluationOutcome::success)
            {
                storeFailedRow(outResult.common, rowIndex, outcome);
                continue;
            }
            storeSuccessfulRow(outResult.common, outResult.hits, rowIndex, evaluated,
                               Policy::semanticCapacity(plan, row));
        }
        return FamilyBatchOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return FamilyBatchOutcome::outOfMemory;
    }
    catch (const std::length_error&)
    {
        return FamilyBatchOutcome::outOfMemory;
    }
    catch (...)
    {
        return FamilyBatchOutcome::internalFailure;
    }
}

template <typename Policy>
bool pristineResult(const typename Policy::Result& result) noexcept
{
    return result.common.sourcePopulation.slot == UINT32_MAX && result.common.sourcePopulation.generation == 0u &&
           result.common.sourceOrdinal == 0 && result.common.storedHitCount == 0u &&
           result.common.rowOutcomes.empty() && result.common.hitOffsets.empty() && result.common.hitCounts.empty() &&
           result.common.truncated.empty() && result.hits.empty();
}

template <typename Policy>
FamilyBatchOutcome executeTypedFamilyBatchCpu(const ResolvedTarget& target, const typename Policy::Plan& plan,
                                              const typename Policy::Launch& launch,
                                              typename Policy::Result&       outResult) noexcept
{
    if (pristineResult<Policy>(outResult))
    {
        return executeTypedFamilyBatchCpuDirect<Policy>(target, plan, launch, outResult);
    }
    typename Policy::Result  candidate;
    const FamilyBatchOutcome outcome = executeTypedFamilyBatchCpuDirect<Policy>(target, plan, launch, candidate);
    if (outcome == FamilyBatchOutcome::success)
    {
        outResult = std::move(candidate);
    }
    return outcome;
}

} // namespace

FamilyBatchOutcome executeFamilyBatchCpu(const ResolvedTarget& target, const RaycastBatchPlan& plan,
                                         const RaycastBatchLaunch& launch, RaycastBatchResult& outResult) noexcept
{
    return executeTypedFamilyBatchCpu<RaycastBatchPolicy>(target, plan, launch, outResult);
}

bool estimateFamilyBatchExecutionPeakBytes(const ResolvedTarget& target, const RaycastBatchPlan&,
                                           uint64_t&             outBytes) noexcept
{
    return estimateSemanticRaycastPeakBytes(static_cast<uint64_t>(target.generation.objects().size()), outBytes);
}

bool estimateFamilyBatchExecutionPeakBytes(const ResolvedTarget& target, const SphereSweepBatchPlan& plan,
                                           uint64_t& outBytes) noexcept
{
    return estimateSemanticSphereSweepPeakBytes(static_cast<uint64_t>(target.generation.objects().size()), plan.hitMode,
                                                outBytes);
}

bool estimateFamilyBatchExecutionPeakBytes(const ResolvedTarget& target, const OverlapBatchPlan&,
                                           uint64_t&             outBytes) noexcept
{
    return estimateSemanticOverlapPeakBytes(static_cast<uint64_t>(target.generation.objects().size()), outBytes);
}

FamilyBatchOutcome executeFamilyBatchCpu(const ResolvedTarget& target, const SphereSweepBatchPlan& plan,
                                         const SphereSweepBatchLaunch& launch,
                                         SphereSweepBatchResult&       outResult) noexcept
{
    return executeTypedFamilyBatchCpu<SphereSweepBatchPolicy>(target, plan, launch, outResult);
}

FamilyBatchOutcome executeFamilyBatchCpu(const ResolvedTarget& target, const OverlapBatchPlan& plan,
                                         const OverlapBatchLaunch& launch, OverlapBatchResult& outResult) noexcept
{
    return executeTypedFamilyBatchCpu<OverlapBatchPolicy>(target, plan, launch, outResult);
}

ovspatialquery_api_status_t executeFamilyRowCpu(const TargetResolver& resolver, const TargetRequest& request,
                                                const RaycastBatchPlan& plan, const RaycastBatchRow& row,
                                                ResolvedTarget&                  outTarget,
                                                std::vector<RaycastSemanticHit>& outHits) noexcept
{
    return executeFamilyRowCpuSync<RaycastBatchPolicy>(resolver, request, plan, row, outTarget, outHits);
}

ovspatialquery_api_status_t executeFamilyRowCpu(const TargetResolver& resolver, const TargetRequest& request,
                                                const SphereSweepBatchPlan& plan, const SphereSweepBatchRow& row,
                                                ResolvedTarget&                      outTarget,
                                                std::vector<SphereSweepSemanticHit>& outHits) noexcept
{
    return executeFamilyRowCpuSync<SphereSweepBatchPolicy>(resolver, request, plan, row, outTarget, outHits);
}

ovspatialquery_api_status_t executeFamilyRowCpu(const TargetResolver& resolver, const TargetRequest& request,
                                                const OverlapBatchPlan& plan, const OverlapBatchRow& row,
                                                ResolvedTarget&                  outTarget,
                                                std::vector<OverlapSemanticHit>& outHits) noexcept
{
    return executeFamilyRowCpuSync<OverlapBatchPolicy>(resolver, request, plan, row, outTarget, outHits);
}

} // namespace detail
} // namespace ovspatialquery
