// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef OVSPATIALQUERY_DEVICE_BOX_POPULATION_H
#define OVSPATIALQUERY_DEVICE_BOX_POPULATION_H
#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_explicit_preparation.h"
#include <array>
#include <memory>
#include <mutex>
namespace ovspatialquery
{
namespace detail
{
struct DeviceColumn
{
    uintptr_t address = 0;
    uint32_t  bits    = 0;
};
// The host-side end of one admitted device publication (ADR-0021): whether it
// became the head or not. A publication built on one that is already known to
// have been abandoned fails with it; the device status word carries validation
// failures between them without the host. Nothing waits here: workers take
// operations in any order, so a wait could deadlock a one-worker instance.
class DevicePublicationOutcome
{
  public:
    enum class State : uint8_t
    {
        pending,
        published,
        abandoned,
    };
    void finish(State state) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mState == State::pending)
            mState = state;
    }
    State state() const noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mState;
    }

  private:
    mutable std::mutex mMutex;
    State              mState = State::pending;
};
// What a membership-changing publication does to the slots of a device-prepared
// population: rows that add objects (assigned the next free slots, in row
// order) and slots whose objects it removes. Resolved on the host at admission
// from the identity columns, which such a publication reads back once.
struct DeviceMembershipChange
{
    std::vector<ExplicitInstanceInput>  added;
    std::vector<DeviceShapeDescription> addedShapes;
    std::vector<uint32_t>               removedSlots;
    bool                                empty() const noexcept { return added.empty() && removedSlots.empty(); }
};
// The membership of one explicit population whose versions are prepared on a
// CUDA device (ADR-0024): spheres, boxes, capsules, convex hulls and triangle
// meshes. Slots are assigned in order up to a capacity fixed by the first
// complete set and never reused; each slot's identity, binding, authored shape
// parameters and definition record are written once when it is assigned, and
// the hulls and meshes it references keep their local vertices or bounds
// resident on the device in a per-population pool. Kernels resolve IDs
// through a sorted index of the live objects, so a removed object's ID is
// retired for the population's life. Every version is one immutable pose,
// bounds and participation-bits table over all slots (ADR-0003: a retained
// version keeps the bits it was published with), where an unassigned or
// removed slot holds inverted bounds that no query hits, followed by one status
// word that is nonzero once the version, or a version it was built on, failed.
class DeviceBoxPopulation
{
  public:
    DeviceBoxPopulation();
    struct Impl;
    std::shared_ptr<Impl> impl;
    // Slots the tables are sized for.
    uint64_t        capacity() const;
    DLDevice        device() const;
    static uint64_t versionBytes(uint64_t capacity, bool ovstage = false);
    static uint64_t membershipBytes(uint64_t capacity);
    // Retained asynchronous parameter, definition-index and shape-record uploads.
    static uint64_t forkStagingBytes(uint64_t rows);
    // Device bytes the definition pool needs for `shapes` beyond what the
    // population already holds (records and hull vertices); the first set asks
    // the static form. Reserved by the caller and handed in as `poolLease`.
    static uint64_t definitionBytes(const std::vector<DeviceShapeDescription>& shapes);
    // A fork also charges a copy of the existing record table when new records
    // are needed, even when its current capacity already covers them.
    uint64_t additionalDefinitionBytes(const std::vector<DeviceShapeDescription>& shapes, bool fork = false) const;
    ovspatialquery_api_status_t initialize(DLDevice, uint64_t capacity, const std::vector<ExplicitInstanceInput>&,
                                           const std::vector<DeviceShapeDescription>& shapes,
                                           std::vector<std::shared_ptr<const void>>, std::shared_ptr<void> hostLease,
                                           std::shared_ptr<void>                       deviceLease,
                                           std::shared_ptr<const DeviceObjectPreparer> objects,
                                           std::shared_ptr<void>                       poolLease);
    // Replaces only existing Ovstage slots in a new immutable membership owner.
    // Retained versions keep the old bindings and definitions; the new owner
    // shares compiled kernels, stream and immutable definition allocations.
    ovspatialquery_api_status_t forkOvstage(const std::vector<ExplicitInstanceInput>&   replacements,
                                            const std::vector<DeviceShapeDescription>&  shapes,
                                            std::shared_ptr<const DeviceObjectPreparer> preparer,
                                            std::vector<std::shared_ptr<const void>>    definitions,
                                            std::shared_ptr<void> hostLease, std::shared_ptr<void> deviceLease,
                                            std::shared_ptr<void> poolLease, std::shared_ptr<void> stagingLease,
                                            std::shared_ptr<DeviceBoxPopulation>& outPopulation) const;
    // Classifies a complete-rows update's identities on the host: a known live
    // ID keeps its slot, an unknown ID is an addition, a retired or removed ID
    // or a removal of an unknown ID is INVALID_ARGUMENT, and more additions
    // than free slots is INVALID_ARGUMENT. Nothing is changed yet.
    ovspatialquery_api_status_t planMembershipChange(const std::vector<ExplicitInstanceInput>& rows,
                                                     const std::vector<uint64_t>&              removedIds,
                                                     DeviceMembershipChange&                   outChange) const;
    // Applies a planned change: assigns slots, writes their membership rows and
    // the sorted index on the population's stream (an event-rate upload,
    // ordered before the launch that follows), retires removed IDs.
    ovspatialquery_api_status_t applyMembershipChange(const DeviceMembershipChange&,
                                                      std::vector<std::shared_ptr<const void>> addedDefinitions,
                                                      std::shared_ptr<void>                    poolLease);
    // Columns: 0 IDs, 1 geometry types, 2 geometry values, 3 translations,
    // 4 rotations, 5 scales, 6 participation bits; an absent column has
    // address zero. Poses-only rows carry columns 0, 3, 4 and optionally 5,
    // and 6 when they set bits. A row with column 6 writes its bits into the
    // version; a complete row without it writes all bits set; a pose row
    // without it keeps the base's.
    // Enqueues the version's validation and preparation on the population's
    // stream, after `afterEvent` when nonzero, and returns the version at once
    // (ADR-0021): its status word and readiness event are set on the device,
    // and `complete` observes them later. The base version's table is copied
    // first, so the version is as valid as its base; the removed slots of the
    // change applied before this launch are tombstoned in the new table.
    // A poses-only row that leaves its slot's pose and bits bit for bit as the
    // base had them is listed as ~0u in the version's changed slots, which
    // readers skip. `snapshotRows` says the rows are every slot's pose, not the
    // moved ones, so the list's length bounds the moved slots rather than
    // counting them. `hostReadable` marks the version for CPU readers
    // (ON_DEVICE_HOST_READABLE). Ovstage's boxExtents carries three world-space
    // primitive parameters per input row: BOX half extents, SPHERE radius, or
    // CAPSULE radius and half height. The version keeps its own immutable copy.
    ovspatialquery_api_status_t launch(ovspatialquery_ordinal_t, const GenerationLease& base, uint64_t rows,
                                       const std::array<DeviceColumn, 7>&, bool posesOnly, bool snapshotRows,
                                       bool hostReadable, uintptr_t afterEvent,
                                       const std::vector<uint32_t>& removedSlots,
                                       std::shared_ptr<CudaPopulationVersion>&, bool ovstage = false,
                                       uintptr_t boxExtents = 0u, uintptr_t exclusionCount = 0u,
                                       uintptr_t excludedIds = 0u, uintptr_t exclusionReasons = 0u);
    // Waits for the version's device work. SUCCESS, OPERATION_FAILED when a row
    // was invalid or a publication it was built on failed validation or is
    // already known not to have become the head, DEVICE_LOST when the device
    // was lost, INTERNAL otherwise.
    ovspatialquery_api_status_t complete(const std::shared_ptr<CudaPopulationVersion>&);
    // Rings `signal` once the version's device work is done; once per version,
    // best effort.
    void ringWhenReady(const std::shared_ptr<CudaPopulationVersion>&, const std::shared_ptr<DeviceSignal>& signal);
    // The host-side outcome of a version this population launched; null otherwise.
    static std::shared_ptr<DevicePublicationOutcome> outcome(const std::shared_ptr<const CudaPopulationVersion>&);
    static void setOvstageLiveCount(const std::shared_ptr<CudaPopulationVersion>&, uint64_t);
    bool        compatibleContext(void* context) const;
};
// What an explicit population knows about device preparation: nothing until
// its first complete ON_DEVICE set publishes, then the membership every later
// device publication validates against.
class CudaStagingPool;

class DevicePreparedState
{
  public:
    // Device buffers this population's CUDA captures copy columns into, kept
    // for the next capture (ADR-0053); `create` makes the pool on first use.
    template <typename Create>
    std::shared_ptr<CudaStagingPool> staging(Create create)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mStaging == nullptr)
        {
            mStaging = create();
        }
        return mStaging;
    }

    std::shared_ptr<DeviceBoxPopulation> current() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mBoxes;
    }
    void commit(std::shared_ptr<DeviceBoxPopulation> boxes)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mBoxes = std::move(boxes);
    }

  private:
    mutable std::mutex                   mMutex;
    std::shared_ptr<DeviceBoxPopulation> mBoxes;
    std::shared_ptr<CudaStagingPool>     mStaging;
};
} // namespace detail
} // namespace ovspatialquery
#endif
