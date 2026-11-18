/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_POPULATION_TABLE_H
#define OVSPATIALQUERY_POPULATION_TABLE_H

#include "ovspatialquery_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct PopulationKey
{
    uint32_t slot       = UINT32_MAX;
    uint32_t generation = 0u;

    bool operator==(const PopulationKey& other) const;
};

struct PopulationGeneration;
class CudaPopulationVersion;
class TargetResolver;
struct GenerationSelectionRequest;
enum class TargetResolveOutcome : uint8_t;

struct GenerationViewMembership
{
    ovspatialquery_view_handle_t                 view = OVSPATIALQUERY_INVALID_VIEW;
    std::shared_ptr<const std::vector<uint64_t>> selectedIds;

    uint64_t residentByteCount() const;
};

// What changed in a version since the version it was built from (ADR-0019),
// by object position: an index into the version's object table. Positions are
// stable while a population only moves, recategorizes or appends objects. A
// first publication, one that renumbers positions (a removal, an insertion
// before existing objects, a source set whose identities no longer line up
// with its base) or one that changed more than a quarter of its objects, where
// a reporter recomputes anyway, records that everything changed. The record is
// exact or over-inclusive, never missing a change: a pair reporter trusts it
// to skip unchanged objects.
struct VersionChangeRecord
{
    // More changed objects than this, of `objectCount`, record everything.
    static size_t positionLimit(size_t objectCount) noexcept { return objectCount / 4u; }

    bool                     everything  = true;
    ovspatialquery_ordinal_t baseOrdinal = -1;
    /** Sorted unique positions of the objects inserted beyond the base's table
     * or whose bounds, participation bits, view membership or stored exact
     * geometry differ from the base. */
    std::vector<uint32_t> positions;

    uint64_t residentByteCount() const noexcept;
};

// Device state that belongs to one version (ADR-0019 decisions 6 and 7), at
// most one entry per CUDA device, read and changed under the population
// table's device copy mutex. It is released with the version at the latest;
// its owner may detach it earlier.
struct GenerationDeviceAttachments
{
    std::vector<std::pair<int32_t, std::shared_ptr<void>>> entries;
};

// Whether an object carries stored exact geometry for its exact queries.
bool hasStoredExactGeometry(const PopulationObject& object) noexcept;

// Whether an object at the same position in consecutive versions changed
// its bounds, its participation bits or whether it has stored exact geometry.
// View membership is compared per view.
bool objectChanged(const PopulationObject& previous, const PopulationObject& current) noexcept;

class InstanceHostLedger;

class GenerationLease
{
  public:
    GenerationLease();

    bool initialized() const;
    // A sufficient per-generation guarantee; mixed generations use query-time checks.
    bool                     hasOnlyStoredExactGeometry() const noexcept;
    ovspatialquery_ordinal_t ordinal() const;
    const ObjectTable&       objects() const;
    uint64_t                 objectCount() const;
    // Prepared by device kernels (ADR-0024): no host objects of its own, so
    // only CUDA plans that read the device tables can serve it, unless the
    // publication was ON_DEVICE_HOST_READABLE and a CPU reader materialized the
    // host copy that `objects()` and `backend()` then answer from.
    bool                         preparedOnDevice() const noexcept;
    bool                         hasHostCopy() const noexcept;
    const std::vector<uint64_t>& usedObjectValues() const;
    // The same values, shared by versions that publish the same identities.
    std::shared_ptr<const std::vector<uint64_t>> sharedUsedObjectValues() const;
    // Deduplicated resident bytes of the objects' exact geometry charged when this head was
    // published, if recorded. Only explicit updates adjust it for their successor: explicit
    // objects never share per-object allocations, so replaced objects can be subtracted.
    std::optional<uint64_t>                             exactGeometryCharge() const noexcept;
    const BackendPopulation&                            backend() const;
    bool                                                hasBackend() const;
    std::shared_ptr<const CudaPopulationVersion>        cudaVersion() const;
    bool                                                includesObject(size_t objectIndex) const;
    const VersionChangeRecord&                          changeRecord() const;
    const std::shared_ptr<GenerationDeviceAttachments>& deviceAttachments() const;
    // Whether a view or ID selection narrows this generation's objects.
    bool hasSelection() const noexcept;
    // One flag per object: included and, unless `queryBits` is zero,
    // intersecting its participation bits. Objects in ascending ID order are
    // matched against the selection in one walk instead of a search each.
    void admittedObjects(const ParticipationBits& queryBits, std::vector<uint8_t>& outFlags) const;
    bool sameSelection(const GenerationLease& other) const noexcept;
    bool selectViewMembership(ovspatialquery_view_handle_t view);
    void selectIds(std::shared_ptr<const std::vector<uint64_t>> selectedIds);
    void reset();

  private:
    friend class PopulationTable;
    explicit GenerationLease(std::shared_ptr<const PopulationGeneration> generation);

    std::shared_ptr<const PopulationGeneration>  mGeneration;
    std::shared_ptr<const std::vector<uint64_t>> mSelectedIds;
};

class PopulationTable
{
  public:
    explicit PopulationTable(ovspatialquery_backend_runtime_t* backendRuntime);
    ~PopulationTable();

    static uint64_t fixedResidentByteCount() noexcept;

    PopulationTable(const PopulationTable&)            = delete;
    PopulationTable& operator=(const PopulationTable&) = delete;

    // `pendingPublicationCapacity` below one means one (ADR-0021).
    ovspatialquery_api_status_t create(uint32_t retainedOrdinalCapacity, PopulationKey& outHandle,
                                       uint32_t pendingPublicationCapacity = 1u);
    ovspatialquery_api_status_t create(uint32_t retainedOrdinalCapacity, ovspatialquery_ordinal_t initialOrdinal,
                                       std::vector<PopulationObject> objects, PopulationKey& outHandle);
    ovspatialquery_api_status_t release(PopulationKey handle, std::shared_ptr<void>& outDeferredDestruction);
    // All generation pins, including atomic multi-target admission,
    // publishUpdate, and release linearize while holding the table mutex. A
    // successful publish replaces head/history as one step, so concurrent
    // current-head readers never observe a transient no-head state.
    ovspatialquery_api_status_t pin(PopulationKey handle, GenerationLease& outLease) const;
    ovspatialquery_api_status_t pinExact(PopulationKey handle, ovspatialquery_ordinal_t ordinal,
                                         GenerationLease& outLease) const;
    ovspatialquery_api_status_t getInfoState(PopulationKey handle, bool& outReady, ovspatialquery_ordinal_t& outOrdinal,
                                             uint64_t& outObjectCount, uint32_t& outRetainedOrdinalCount,
                                             bool& outUpdatePending) const;
    ovspatialquery_api_status_t getUpdatePending(PopulationKey handle, bool& outUpdatePending) const;
    // Resolves the handle without pinning: INVALID_ARGUMENT when unknown.
    ovspatialquery_api_status_t validate(PopulationKey handle) const;

    // With `chainPendingPublication`, a publication whose exact base is the
    // newest admitted publication is accepted while that one is pending, up to
    // the slot's pending capacity, and `outCurrentGeneration` is the admitted
    // base (ADR-0021). Otherwise one publication may be pending (BUSY).
    ovspatialquery_api_status_t beginUpdate(PopulationKey handle, ovspatialquery_ordinal_t targetOrdinal,
                                            const ovspatialquery_ordinal_t* exactBaseOrdinal,
                                            GenerationLease&                outCurrentGeneration,
                                            bool                            requireUnreadyPopulation = false,
                                            bool                            chainPendingPublication  = false);
    // Makes a pending publication's generation selectable by admitted-exact
    // work and usable as the base of the next chained publication.
    ovspatialquery_api_status_t admitPending(PopulationKey handle, const GenerationLease& admitted);
    ovspatialquery_api_status_t buildGeneration(ovspatialquery_ordinal_t ordinal, ObjectTable objects,
                                                std::vector<GenerationViewMembership> viewMemberships,
                                                GenerationLease& outGeneration, std::shared_ptr<void> memoryLease = {},
                                                std::shared_ptr<const CudaPopulationVersion> cudaVersion       = {},
                                                bool                                         requireCpuBackend = false,
                                                std::shared_ptr<const std::vector<uint64_t>> usedObjectValues  = {},
                                                std::optional<uint64_t>                      exactGeometryCharge = {},
                                                VersionChangeRecord                          changeRecord        = {},
                                                const GenerationLease& base = GenerationLease()) const;
    // Compares `objects`, about to be published over `base`, with it by
    // position (IDs, bounds, participation bits) and each automatic view's
    // membership by content. Records everything when `base` is absent or the
    // positions were renumbered.
    static ovspatialquery_api_status_t describeChanges(const GenerationLease& base, const ObjectTable& objects,
                                                       const std::vector<GenerationViewMembership>& viewMemberships,
                                                       VersionChangeRecord&                         outRecord) noexcept;
    // Whether describeChanges compares `memberships`, sorted by view, with
    // `base`'s for a version of `objectCount` objects, and how many IDs it
    // then gathers from the views' differences before deduplicating them.
    static bool                 countMembershipChanges(const GenerationLease& base, size_t objectCount,
                                                       const std::vector<GenerationViewMembership>& memberships,
                                                       uint64_t&                                    outIdCount) noexcept;
    ovspatialquery_api_status_t publishUpdate(PopulationKey handle, const GenerationLease& generation);
    // Ends the newest pending publication (admission failure paths) or the one
    // publishing `targetOrdinal` (completion), published or not.
    void finishUpdate(PopulationKey handle);
    void finishUpdate(PopulationKey handle, ovspatialquery_ordinal_t targetOrdinal);

    ovspatialquery_api_status_t retainPersistentReferences(const std::vector<PopulationKey>& handles);
    void                        finishPersistentReferences(const std::vector<PopulationKey>& handles);

    // Builds the host copy of a readable device-prepared version the first
    // time a CPU reader selects it: the version's poses come down, the objects
    // and the backend tree are built as the host path builds them, the bytes
    // are charged to the ledger's published data (HOST_BUDGET when they do not
    // fit) and released with the version. SUCCESS at once for a host-prepared
    // version or a copy already built; NOT_SUPPORTED for plain ON_DEVICE.
    ovspatialquery_api_status_t materializeHostCopy(const GenerationLease& generation,
                                                    InstanceHostLedger*    hostLedger) const;

    // Per population and CUDA device, the attachments of the version whose
    // device copy a CUDA pair reporter selected last (ADR-0019 decision 7).
    // Callers serialize copies through `deviceCopyMutex`.
    std::weak_ptr<GenerationDeviceAttachments> lastDeviceAttachments(PopulationKey handle, int32_t deviceId) const;
    void                                       setLastDeviceAttachments(PopulationKey handle, int32_t deviceId,
                                                                        std::weak_ptr<GenerationDeviceAttachments> attachments) const;
    std::mutex&                                deviceCopyMutex() const;

    size_t populationCount() const;

  private:
    friend class TargetResolver;

    TargetResolveOutcome pinResolvedTarget(const GenerationSelectionRequest& request, GenerationLease& outLease) const;
    TargetResolveOutcome pinResolvedPair(const std::array<GenerationSelectionRequest, 2u>& requests,
                                         std::array<GenerationLease, 2u>&                  outLeases) const;
    ovspatialquery_api_status_t createSlot(uint32_t                                    retainedOrdinalCapacity,
                                           std::shared_ptr<const PopulationGeneration> initialGeneration,
                                           PopulationKey& outHandle, uint32_t pendingPublicationCapacity = 1u);

    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace detail
} // namespace ovspatialquery

#endif
