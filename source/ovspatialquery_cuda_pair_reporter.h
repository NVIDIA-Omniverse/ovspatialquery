/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_PAIR_REPORTER_H
#define OVSPATIALQUERY_CUDA_PAIR_REPORTER_H

#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_device_signal.h"
#include "ovspatialquery_pair_materialization.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

class InstanceHostLedger;

// A pair reporter whose results live on a CUDA device (ADR-0019 stage 3):
// its accepted set and results stay on the device, and each advance reads
// device copies of the endpoint versions that belong to the versions
// (decisions 6 and 7), patching the accepted set from their change records
// when it can and recomputing it otherwise, with the CPU reporter's pairs and
// outcomes. Each result slot's header and delta rows are the capacity-shaped
// tensors the reservation describes (ADR-0022), written before the slot's
// ready event; a failed advance writes the FAILED outcome there.
struct CudaPairReporterConfig
{
    DLDevice             device{kDLCUDA, 0};
    PairEndpointRelation relation         = PairEndpointRelation::distinctPopulations;
    double               contactTolerance = 0.0;
    ParticipationBits    firstQueryBits{};
    ParticipationBits    secondQueryBits{};
    // The view each endpoint targets, the key of its membership on a version's
    // device copy, compared by owner so that a destroyed view never matches a
    // later one; empty for a whole population.
    std::weak_ptr<const void> firstSelection;
    std::weak_ptr<const void> secondSelection;
    uint64_t                  activePairCapacity = 0u;
    uint64_t                  deltaPairCapacity  = 0u;
    uint32_t                  resultSlotCount    = 0u;
};

// Where the reporter's accepted baseline stands.
struct CudaPairBaseline
{
    ovspatialquery_ordinal_t firstOrdinal  = 0;
    ovspatialquery_ordinal_t secondOrdinal = 0;
};

// Where a CUDA advance runs (ADR-0022 decision 1): the caller's stream, which
// may be stream 0, after its wait event, or the reporter's own stream when the
// caller supplies none (ADR-0060).
struct CudaPairAdvanceSync
{
    bool      hasStream = false;
    uintptr_t stream    = 0u;
    uintptr_t waitEvent = 0u;
};

// The typed result of one completed advance.
struct CudaPairAdvanceResult
{
    // ovspatialquery_pair_reporter_outcome_t
    uint32_t outcome       = 0u;
    uint64_t enteredCount  = 0u;
    uint64_t exitedCount   = 0u;
    uint64_t acceptedCount = 0u;
    // Whether the advance patched the baseline instead of recomputing it.
    bool patched = false;
};

// One admitted advance: its device work, the device copies it reads until
// that work completes, and the consumer event that retires its result slot.
class CudaPairAdvance;

class CudaPairReporterDevice : public std::enable_shared_from_this<CudaPairReporterDevice>
{
  public:
    // Device bytes the reporter keeps for its lifetime: the ring of
    // result_slot_count + 1 pair sets, and every result slot.
    static bool residentDeviceBytes(uint64_t activePairCapacity, uint64_t deltaPairCapacity, uint32_t resultSlotCount,
                                    uint64_t& outBytes) noexcept;
    // Host bytes the reporter keeps for its lifetime beside its owner's.
    static uint64_t                    residentHostBytes(uint32_t resultSlotCount) noexcept;
    static ovspatialquery_api_status_t create(const CudaPairReporterConfig&            config,
                                              std::shared_ptr<CudaPairReporterDevice>& outReporter) noexcept;
    ~CudaPairReporterDevice();

    CudaPairReporterDevice(const CudaPairReporterDevice&)            = delete;
    CudaPairReporterDevice& operator=(const CudaPairReporterDevice&) = delete;

    // Admits one advance into `resultSlot`: makes or finds the endpoint
    // versions' device copies, charging a new copy to the device as published
    // data (DEVICE_BUDGET when it does not fit), charges the advance's scratch,
    // and enqueues its work on `sync`'s stream (the reporter's own when zero),
    // behind the advance admitted before it, up to the slot's ready event. The
    // new set goes to a free buffer of the ring of result_slot_count + 1; the
    // device decides which buffer is the baseline. Patches from the change
    // records between `baseline` and the selected versions when they allow it.
    // Nothing is enqueued on failure.
    ovspatialquery_api_status_t launch(ovspatialquery_instance_handle_t instance, const PopulationTable& populations,
                                       InstanceHostLedger* hostLedger, uint32_t resultSlot, const ResolvedTarget& first,
                                       const ResolvedTarget& second, const std::optional<CudaPairBaseline>& baseline,
                                       const CudaPairAdvanceSync&        sync,
                                       std::shared_ptr<CudaPairAdvance>& outAdvance) noexcept;
    // Makes the set an accepted advance wrote the baseline the host knows,
    // and frees an advance's ring buffer once its outcome is published
    // (ADR-0022 decision 5); both in publication order.
    void accept(const CudaPairAdvance& advance) noexcept;
    void release(const CudaPairAdvance& advance) noexcept;
    // Replaces the accepted set and every result slot with larger ones while
    // no advance is live, keeping the baseline.
    ovspatialquery_api_status_t grow(uint64_t activePairCapacity, uint64_t deltaPairCapacity) noexcept;

    ovspatialquery_cuda_event_t readyEvent(uint32_t resultSlot) const noexcept;
    // Device rows (uint64[D,2]) of a slot's entered and exited pairs, and its
    // counts (uint64[4]: outcome, entered, exited, accepted).
    const void* enteredRows(uint32_t resultSlot) const noexcept;
    const void* exitedRows(uint32_t resultSlot) const noexcept;
    const void* countsRows(uint32_t resultSlot) const noexcept;

    static CudaCompletionState resultState(const CudaPairAdvance& advance) noexcept;
    // Blocks the calling thread until the advance's device work completes, or
    // returns at once when it cannot; either way resultState() decides.
    static void waitForResult(const CudaPairAdvance& advance) noexcept;
    static bool result(const CudaPairAdvance& advance, CudaPairAdvanceResult& outResult) noexcept;
    // Rings `signal` once the consumer event transferConsumerDone() recorded
    // has completed. Best effort; a no-op before the transfer.
    static void ringWhenConsumed(const CudaPairAdvance& advance, const std::shared_ptr<DeviceSignal>& signal) noexcept;
    // Drops the device copies an advance read once its work has completed.
    static void                        releaseReadyDependencies(CudaPairAdvance& advance) noexcept;
    static ovspatialquery_api_status_t transferConsumerDone(CudaPairAdvance&            advance,
                                                            ovspatialquery_cuda_event_t event) noexcept;
    // Whether the advance's work and the consumer the caller handed over have
    // completed, so its result slot can be reused.
    static bool deviceSafe(CudaPairAdvance& advance) noexcept;
    static bool deviceLost(const CudaPairAdvance& advance) noexcept;

#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
    // Device copies an advance found or made, by how: already attached,
    // borrowed from a CUDA publication, rewritten in place, derived by copying
    // another, or uploaded whole.
    static std::array<uint64_t, 5u> copyCountsForTest() noexcept;
    // Copies a completed slot's entered and exited rows to the host.
    bool downloadForTest(uint32_t resultSlot, uint64_t enteredCount, uint64_t exitedCount,
                         std::vector<ovspatialquery_pair_t>& outEntered,
                         std::vector<ovspatialquery_pair_t>& outExited) noexcept;
#endif

  private:
    friend class CudaPairAdvance;
    struct Impl;
    explicit CudaPairReporterDevice(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> mImpl;
};

} // namespace detail
} // namespace ovspatialquery

#endif
