/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_population_table.h"
#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_target_resolver.h"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
#if defined(OVSPATIALQUERY_ENABLE_TARGET_RESOLVER_TEST_HOOK)
namespace
{
std::atomic<TargetResolverAfterFirstPinHook> gAfterFirstPinHook{nullptr};
}

void setTargetResolverAfterFirstPinHook(TargetResolverAfterFirstPinHook hook)
{
    gAfterFirstPinHook.store(hook, std::memory_order_release);
}
#endif

// The host side of a readable device-prepared version, built once on first
// CPU read and owned by the generation (ADR-0024 decision 4).
struct GenerationHostCopy
{
    ObjectTable                        objects;
    std::unique_ptr<BackendPopulation> backend;
    size_t                             unstoredGeometryCount = 0u;
    HostLease                          lease;
};

struct PopulationGeneration
{
    ObjectTable objects;
    // Objects without stored exact geometry; follows the change record from the base (ADR-0019).
    size_t                                       unstoredGeometryCount = 0u;
    std::shared_ptr<const std::vector<uint64_t>> usedObjectValues;
    std::optional<uint64_t>                      exactGeometryCharge;
    std::unique_ptr<BackendPopulation>           backend;
    ovspatialquery_ordinal_t                     ordinal = 0u;
    std::shared_ptr<const CudaPopulationVersion> cudaVersion;
    std::vector<GenerationViewMembership>        viewMemberships;
    VersionChangeRecord                          changes;
    std::shared_ptr<GenerationDeviceAttachments> deviceAttachments = std::make_shared<GenerationDeviceAttachments>();
    // Written once under the mutex by the reader that materializes it; every
    // CPU reader materializes before it reads, so later reads need no lock.
    mutable std::mutex                          hostCopyMutex;
    mutable std::shared_ptr<GenerationHostCopy> hostCopy;
};

// Each slot owns the head plus up to N prior full immutable snapshots. Evicted
// snapshots remain alive while leased. Exact lookup and full-window rotation
// are O(N); publication performs no allocation after slot creation.
class PopulationVersionStore
{
  public:
    explicit PopulationVersionStore(uint32_t retainedOrdinalCapacity)
        : mRetainedOrdinalCapacity(retainedOrdinalCapacity)
    {
        mRetained.reserve(retainedOrdinalCapacity);
    }

    bool ready() const { return mHead != nullptr; }

    const std::shared_ptr<const PopulationGeneration>& head() const { return mHead; }

    uint32_t retainedOrdinalCount() const noexcept { return static_cast<uint32_t>(mRetained.size()); }

    std::shared_ptr<const PopulationGeneration> exact(ovspatialquery_ordinal_t ordinal) const
    {
        if (mHead != nullptr && mHead->ordinal == ordinal)
        {
            return mHead;
        }
        for (std::vector<std::shared_ptr<const PopulationGeneration>>::const_reverse_iterator generation =
                 mRetained.rbegin();
             generation != mRetained.rend(); ++generation)
        {
            if ((*generation)->ordinal == ordinal)
            {
                return *generation;
            }
        }
        return {};
    }

    void publish(std::shared_ptr<const PopulationGeneration>  generation,
                 std::shared_ptr<const PopulationGeneration>& outEvicted) noexcept
    {
        if (mHead != nullptr)
        {
            if (mRetainedOrdinalCapacity == 0u)
            {
                outEvicted = std::move(mHead);
            }
            else if (mRetained.size() < mRetainedOrdinalCapacity)
            {
                mRetained.push_back(std::move(mHead));
            }
            else
            {
                outEvicted = std::move(mRetained.front());
                for (size_t index = 1u; index < mRetained.size(); ++index)
                {
                    mRetained[index - 1u] = std::move(mRetained[index]);
                }
                mRetained.back() = std::move(mHead);
            }
        }
        mHead = std::move(generation);
    }

    // A version published after a newer one (ADR-0021: publications complete
    // in any worker order) enters the retained window behind the head, where
    // it is the newest prior head; with no window it is evicted at once.
    void insertBehindHead(std::shared_ptr<const PopulationGeneration>  generation,
                          std::shared_ptr<const PopulationGeneration>& outEvicted) noexcept
    {
        if (mRetainedOrdinalCapacity == 0u)
        {
            outEvicted = std::move(generation);
            return;
        }
        if (mRetained.size() == mRetainedOrdinalCapacity)
        {
            outEvicted = std::move(mRetained.front());
            for (size_t index = 1u; index < mRetained.size(); ++index)
            {
                mRetained[index - 1u] = std::move(mRetained[index]);
            }
            mRetained.pop_back();
        }
        mRetained.push_back(std::move(generation));
    }

  private:
    uint32_t                                                 mRetainedOrdinalCapacity;
    std::shared_ptr<const PopulationGeneration>              mHead;
    std::vector<std::shared_ptr<const PopulationGeneration>> mRetained;
};

// One admitted publication that is not yet terminal, in admission order. A
// device-prepared publication gets its generation at admission (ADR-0021), so
// admitted-exact work can select it and the next publication can build on it.
struct PendingPublication
{
    ovspatialquery_ordinal_t                    ordinal         = 0;
    bool                                        stagingReserved = false;
    std::shared_ptr<const PopulationGeneration> admitted;
};

struct PopulationSlot
{
    PopulationSlot(uint32_t retainedOrdinalCapacity, uint32_t pendingPublicationCapacity)
        : versions(retainedOrdinalCapacity),
          pendingCapacity(pendingPublicationCapacity < 1u ? 1u : pendingPublicationCapacity)
    {}

    bool mutationPending() const noexcept { return !pending.empty(); }

    PopulationVersionStore                                                      versions;
    size_t                                                                      pendingOperationCount = 0u;
    std::vector<PendingPublication>                                             pending;
    uint32_t                                                                    pendingCapacity = 1u;
    std::vector<std::pair<int32_t, std::weak_ptr<GenerationDeviceAttachments>>> lastDeviceAttachments;
};

namespace
{
// The generation a selection names in one slot, or why none is available.
TargetResolveOutcome selectGeneration(const PopulationSlot& slot, const TargetSelection& selection,
                                      std::shared_ptr<const PopulationGeneration>& outGeneration)
{
    if (selection.kind == TargetSelectionKind::head)
    {
        if (!slot.versions.ready())
        {
            return TargetResolveOutcome::notReady;
        }
        outGeneration = slot.versions.head();
        return TargetResolveOutcome::success;
    }
    if (selection.kind != TargetSelectionKind::retainedExact && selection.kind != TargetSelectionKind::admittedExact)
    {
        return TargetResolveOutcome::invalid;
    }
    if (selection.ordinal < 0)
    {
        return TargetResolveOutcome::invalid;
    }
    if (slot.versions.ready())
    {
        outGeneration = slot.versions.exact(selection.ordinal);
    }
    if (outGeneration == nullptr && selection.kind == TargetSelectionKind::admittedExact)
    {
        for (const PendingPublication& publication : slot.pending)
        {
            if (publication.ordinal == selection.ordinal && publication.admitted != nullptr)
            {
                outGeneration = publication.admitted;
                break;
            }
        }
    }
    return outGeneration != nullptr ? TargetResolveOutcome::success : TargetResolveOutcome::ordinalNotRetained;
}
} // namespace

struct PopulationEntry
{
    std::shared_ptr<PopulationSlot> population;
    uint32_t                        generation = 1u;
    uint32_t                        nextFree   = std::numeric_limits<uint32_t>::max();
};

struct PopulationTable::Impl
{
    explicit Impl(ovspatialquery_backend_runtime_t* runtime) : backendRuntime(runtime) {}

    ovspatialquery_backend_runtime_t*   backendRuntime;
    mutable std::mutex                  mutex;
    mutable std::mutex                  deviceCopyMutex;
    std::map<uint32_t, PopulationEntry> entries;
    uint64_t                            nextSlot        = 0u;
    uint32_t                            freeHead        = std::numeric_limits<uint32_t>::max();
    size_t                              populationCount = 0u;

    std::shared_ptr<PopulationSlot> resolve(PopulationKey handle) const
    {
        const std::map<uint32_t, PopulationEntry>::const_iterator iterator = entries.find(handle.slot);
        if (iterator == entries.end())
        {
            return {};
        }
        const PopulationEntry& entry = iterator->second;
        return entry.population != nullptr && entry.generation == handle.generation ? entry.population
                                                                                    : std::shared_ptr<PopulationSlot>();
    }
};

bool PopulationKey::operator==(const PopulationKey& other) const
{
    return slot == other.slot && generation == other.generation;
}

uint64_t GenerationViewMembership::residentByteCount() const
{
    constexpr uint64_t base = static_cast<uint64_t>(sizeof(GenerationViewMembership)) + 128u;
    if (selectedIds == nullptr ||
        selectedIds->capacity() > static_cast<size_t>((std::numeric_limits<uint64_t>::max() - base) / sizeof(uint64_t)))
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return base + static_cast<uint64_t>(selectedIds->capacity()) * sizeof(uint64_t);
}

uint64_t VersionChangeRecord::residentByteCount() const noexcept
{
    return static_cast<uint64_t>(positions.capacity()) * sizeof(uint32_t);
}

GenerationLease::GenerationLease() = default;

GenerationLease::GenerationLease(std::shared_ptr<const PopulationGeneration> generation)
    : mGeneration(std::move(generation))
{}

bool GenerationLease::initialized() const { return mGeneration != nullptr; }

bool GenerationLease::hasOnlyStoredExactGeometry() const noexcept
{
    if (hasHostCopy())
        return mGeneration->hostCopy->unstoredGeometryCount == 0u;
    return mGeneration != nullptr && mGeneration->unstoredGeometryCount == 0u;
}

bool GenerationLease::hasHostCopy() const noexcept { return preparedOnDevice() && mGeneration->hostCopy != nullptr; }

ovspatialquery_ordinal_t GenerationLease::ordinal() const { return mGeneration == nullptr ? 0u : mGeneration->ordinal; }

const ObjectTable& GenerationLease::objects() const
{
    return hasHostCopy() ? mGeneration->hostCopy->objects : mGeneration->objects;
}
uint64_t GenerationLease::objectCount() const
{
    if (preparedOnDevice())
        return mGeneration->cudaVersion->description().liveObjectCount;
    return static_cast<uint64_t>(mGeneration->objects.size());
}

bool GenerationLease::preparedOnDevice() const noexcept
{
    return mGeneration != nullptr && mGeneration->cudaVersion != nullptr &&
           mGeneration->cudaVersion->description().devicePrepared;
}

const std::vector<uint64_t>& GenerationLease::usedObjectValues() const
{
    static const std::vector<uint64_t> empty;
    return mGeneration == nullptr || mGeneration->usedObjectValues == nullptr ? empty : *mGeneration->usedObjectValues;
}

std::shared_ptr<const std::vector<uint64_t>> GenerationLease::sharedUsedObjectValues() const
{
    return mGeneration == nullptr ? nullptr : mGeneration->usedObjectValues;
}

std::optional<uint64_t> GenerationLease::exactGeometryCharge() const noexcept
{
    return mGeneration == nullptr ? std::nullopt : mGeneration->exactGeometryCharge;
}

const BackendPopulation& GenerationLease::backend() const
{
    return hasHostCopy() ? *mGeneration->hostCopy->backend : *mGeneration->backend;
}

bool GenerationLease::hasBackend() const
{
    if (hasHostCopy())
        return mGeneration->hostCopy->backend != nullptr;
    return mGeneration != nullptr && mGeneration->backend != nullptr;
}

ovspatialquery_api_status_t PopulationTable::materializeHostCopy(const GenerationLease& generation,
                                                                 InstanceHostLedger*    hostLedger) const
{
    const std::shared_ptr<const PopulationGeneration>& version = generation.mGeneration;
    if (version == nullptr || !generation.preparedOnDevice())
        return OVSPATIALQUERY_API_SUCCESS;
    if (!version->cudaVersion->hostReadable())
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    try
    {
        std::lock_guard<std::mutex> lock(version->hostCopyMutex);
        if (version->hostCopy != nullptr)
            return OVSPATIALQUERY_API_SUCCESS;
        std::vector<PopulationObject> objects;
        ovspatialquery_api_status_t   status = version->cudaVersion->hostObjects(objects);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        auto copy     = std::make_shared<GenerationHostCopy>();
        copy->objects = ObjectTable(std::move(objects));
        // The objects' own batch plus what a published generation is charged per
        // object for its backend structures (see estimateGenerationCharge).
        const uint64_t perObject = 256u;
        const uint64_t count     = static_cast<uint64_t>(copy->objects.size());
        if (count > (UINT64_MAX - 1024u) / perObject)
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        uint64_t charge = 1024u + count * perObject;
        for (const uint64_t part : {copy->objects.ownBatchBytes(), copy->objects.chunkEntryBytes()})
        {
            if (part > UINT64_MAX - charge)
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            charge += part;
        }
        if (hostLedger != nullptr)
        {
            HostTransaction transaction(*hostLedger);
            if (!transaction.reserve(charge) || !transaction.commit(copy->lease))
                return OVSPATIALQUERY_API_HOST_BUDGET;
        }
        status = createBackendPopulation(mImpl->backendRuntime, copy->backend);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = copy->backend->update(copy->objects);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        for (const PopulationObject& object : copy->objects)
            copy->unstoredGeometryCount += hasStoredExactGeometry(object) ? size_t{0u} : size_t{1u};
        version->hostCopy = std::move(copy);
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

std::shared_ptr<const CudaPopulationVersion> GenerationLease::cudaVersion() const
{
    return mGeneration == nullptr ? std::shared_ptr<const CudaPopulationVersion>() : mGeneration->cudaVersion;
}

bool GenerationLease::includesObject(size_t objectIndex) const
{
    if (mSelectedIds == nullptr)
    {
        return true;
    }
    return std::binary_search(mSelectedIds->begin(), mSelectedIds->end(), objects()[objectIndex].id.value);
}

const VersionChangeRecord& GenerationLease::changeRecord() const
{
    static const VersionChangeRecord everything;
    return mGeneration == nullptr ? everything : mGeneration->changes;
}

const std::shared_ptr<GenerationDeviceAttachments>& GenerationLease::deviceAttachments() const
{
    return mGeneration->deviceAttachments;
}

bool GenerationLease::hasSelection() const noexcept { return mSelectedIds != nullptr; }

void GenerationLease::admittedObjects(const ParticipationBits& queryBits, std::vector<uint8_t>& outFlags) const
{
    const ObjectTable& table = objects();
    outFlags.assign(table.size(), 0u);
    size_t cursor  = 0u;
    bool   ordered = true;
    for (size_t index = 0u; index < table.size(); ++index)
    {
        const uint64_t id       = table[index].id.value;
        bool           included = true;
        if (mSelectedIds != nullptr)
        {
            const std::vector<uint64_t>& selected = *mSelectedIds;
            ordered                               = ordered && (index == 0u || table[index - 1u].id.value < id);
            if (ordered)
            {
                while (cursor < selected.size() && selected[cursor] < id)
                {
                    ++cursor;
                }
                included = cursor < selected.size() && selected[cursor] == id;
            }
            else
            {
                included = std::binary_search(selected.begin(), selected.end(), id);
            }
        }
        outFlags[index] = included && participationAccepts(queryBits, table[index].participationBits);
    }
}

bool GenerationLease::sameSelection(const GenerationLease& other) const noexcept
{
    return mGeneration == other.mGeneration && mSelectedIds == other.mSelectedIds;
}

bool GenerationLease::selectViewMembership(ovspatialquery_view_handle_t view)
{
    if (mGeneration == nullptr)
    {
        return false;
    }
    const std::vector<GenerationViewMembership>::const_iterator membership =
        std::lower_bound(mGeneration->viewMemberships.begin(), mGeneration->viewMemberships.end(), view,
                         [](const GenerationViewMembership& candidate, ovspatialquery_view_handle_t requested)
                         { return candidate.view < requested; });
    if (membership == mGeneration->viewMemberships.end() || membership->view != view)
    {
        return false;
    }
    mSelectedIds = membership->selectedIds;
    return mSelectedIds != nullptr;
}

void GenerationLease::selectIds(std::shared_ptr<const std::vector<uint64_t>> selectedIds)
{
    mSelectedIds = std::move(selectedIds);
}

void GenerationLease::reset()
{
    mGeneration.reset();
    mSelectedIds.reset();
}

PopulationTable::PopulationTable(ovspatialquery_backend_runtime_t* backendRuntime) : mImpl(new Impl(backendRuntime)) {}

PopulationTable::~PopulationTable() = default;

uint64_t PopulationTable::fixedResidentByteCount() noexcept { return static_cast<uint64_t>(sizeof(Impl)); }

ovspatialquery_api_status_t PopulationTable::buildGeneration(
    ovspatialquery_ordinal_t ordinal, ObjectTable objects, std::vector<GenerationViewMembership> viewMemberships,
    GenerationLease& outGeneration, std::shared_ptr<void> memoryLease,
    std::shared_ptr<const CudaPopulationVersion> cudaVersion, bool requireCpuBackend,
    std::shared_ptr<const std::vector<uint64_t>> usedObjectValues, std::optional<uint64_t> exactGeometryCharge,
    VersionChangeRecord changeRecord, const GenerationLease& base) const
{
    if (ordinal < 0 || (!changeRecord.everything && changeRecord.baseOrdinal >= ordinal))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::sort(viewMemberships.begin(), viewMemberships.end(),
                  [](const GenerationViewMembership& first, const GenerationViewMembership& second)
                  { return first.view < second.view; });
        for (size_t index = 0u; index < viewMemberships.size(); ++index)
        {
            if (viewMemberships[index].view == OVSPATIALQUERY_INVALID_VIEW ||
                viewMemberships[index].selectedIds == nullptr ||
                !std::is_sorted(viewMemberships[index].selectedIds->begin(),
                                viewMemberships[index].selectedIds->end()) ||
                std::adjacent_find(viewMemberships[index].selectedIds->begin(),
                                   viewMemberships[index].selectedIds->end()) !=
                    viewMemberships[index].selectedIds->end() ||
                (index != 0u && viewMemberships[index - 1u].view == viewMemberships[index].view))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
        }
        std::unique_ptr<BackendPopulation> backend;
        ovspatialquery_api_status_t        status = OVSPATIALQUERY_API_SUCCESS;
        if (cudaVersion == nullptr || requireCpuBackend)
        {
            status = createBackendPopulation(mImpl->backendRuntime, backend);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
            // A version that lists its changes against `base` may reuse the
            // base's structures (ADR-0019).
            status = !changeRecord.everything && base.hasBackend() && base.ordinal() == changeRecord.baseOrdinal
                         ? backend->derive(base.backend(), objects, changeRecord.positions)
                         : backend->update(objects);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
        }
        PopulationGeneration*                 rawGeneration = new PopulationGeneration();
        std::shared_ptr<PopulationGeneration> generation(
            rawGeneration,
            [memoryLease = std::move(memoryLease)](PopulationGeneration* value) mutable
            {
                // Destroy every generation-owned allocation and backend object
                // before releasing its host/device budget reservation.
                delete value;
                memoryLease.reset();
            });
        const auto unstored = [](const PopulationObject& object)
        { return hasStoredExactGeometry(object) ? size_t{0u} : size_t{1u}; };
        if (!changeRecord.everything && base.initialized() && base.ordinal() == changeRecord.baseOrdinal)
        {
            // Only listed positions differ from the base; appended ones follow its last object.
            const ObjectTable& baseObjects    = base.mGeneration->objects;
            generation->unstoredGeometryCount = base.mGeneration->unstoredGeometryCount;
            for (const uint32_t position : changeRecord.positions)
            {
                if (position >= objects.size())
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                if (position < baseObjects.size())
                {
                    generation->unstoredGeometryCount -= unstored(baseObjects[position]);
                }
                generation->unstoredGeometryCount += unstored(objects[position]);
            }
        }
        else
        {
            for (const PopulationObject& object : objects)
            {
                generation->unstoredGeometryCount += unstored(object);
            }
        }
        generation->objects             = std::move(objects);
        generation->usedObjectValues    = std::move(usedObjectValues);
        generation->exactGeometryCharge = exactGeometryCharge;
        generation->backend             = std::move(backend);
        generation->ordinal             = ordinal;
        generation->cudaVersion         = std::move(cudaVersion);
        generation->viewMemberships     = std::move(viewMemberships);
        generation->changes             = std::move(changeRecord);
        outGeneration                   = GenerationLease(std::move(generation));
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

bool hasStoredExactGeometry(const PopulationObject& object) noexcept { return object.exactGeometry != nullptr; }

bool objectChanged(const PopulationObject& previous, const PopulationObject& current) noexcept
{
    if (!participationEqual(current.participationBits, previous.participationBits))
    {
        return true;
    }
    for (size_t axis = 0u; axis < 3u; ++axis)
    {
        if (!(current.bounds.minimum[axis] == previous.bounds.minimum[axis]) ||
            !(current.bounds.maximum[axis] == previous.bounds.maximum[axis]))
        {
            return true;
        }
    }
    // A version counts the objects without stored exact geometry from its
    // record, so a change of kind is listed even with equal bounds.
    return hasStoredExactGeometry(current) != hasStoredExactGeometry(previous);
}

ovspatialquery_api_status_t PopulationTable::describeChanges(
    const GenerationLease& base, const ObjectTable& objects,
    const std::vector<GenerationViewMembership>& viewMemberships, VersionChangeRecord& outRecord) noexcept
{
    outRecord = VersionChangeRecord{};
    if (base.mGeneration == nullptr)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    const ObjectTable&                           baseObjects     = base.mGeneration->objects;
    const std::vector<GenerationViewMembership>& baseMemberships = base.mGeneration->viewMemberships;
    try
    {
        // A view membership is present for every automatic view at every
        // publication; a different view set is rare enough to record all.
        std::vector<GenerationViewMembership> memberships = viewMemberships;
        std::sort(memberships.begin(), memberships.end(),
                  [](const GenerationViewMembership& first, const GenerationViewMembership& second)
                  { return first.view < second.view; });
        // Reserved exactly, the storage publication admission charged.
        uint64_t changedIdCount = 0u;
        if (!countMembershipChanges(base, objects.size(), memberships, changedIdCount))
        {
            return OVSPATIALQUERY_API_SUCCESS;
        }
        std::vector<uint64_t> membershipChangedIds;
        membershipChangedIds.reserve(static_cast<size_t>(changedIdCount));
        for (size_t index = 0u; index < memberships.size(); ++index)
        {
            const GenerationViewMembership& current  = memberships[index];
            const GenerationViewMembership& previous = baseMemberships[index];
            if (current.selectedIds != previous.selectedIds)
            {
                std::set_symmetric_difference(previous.selectedIds->begin(), previous.selectedIds->end(),
                                              current.selectedIds->begin(), current.selectedIds->end(),
                                              std::back_inserter(membershipChangedIds));
            }
        }
        std::sort(membershipChangedIds.begin(), membershipChangedIds.end());
        membershipChangedIds.erase(std::unique(membershipChangedIds.begin(), membershipChangedIds.end()),
                                   membershipChangedIds.end());

        VersionChangeRecord record;
        record.everything  = false;
        record.baseOrdinal = base.mGeneration->ordinal;
        for (size_t index = 0u; index < objects.size(); ++index)
        {
            const PopulationObject& current = objects[index];
            if (index < baseObjects.size())
            {
                const PopulationObject& previous = baseObjects[index];
                if (current.id.value != previous.id.value)
                {
                    return OVSPATIALQUERY_API_SUCCESS;
                }
                if (!objectChanged(previous, current) &&
                    !std::binary_search(membershipChangedIds.begin(), membershipChangedIds.end(), current.id.value))
                {
                    continue;
                }
            }
            if (record.positions.size() == VersionChangeRecord::positionLimit(objects.size()))
            {
                return OVSPATIALQUERY_API_SUCCESS;
            }
            record.positions.push_back(static_cast<uint32_t>(index));
        }
        outRecord = std::move(record);
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

bool PopulationTable::countMembershipChanges(const GenerationLease& base, size_t objectCount,
                                             const std::vector<GenerationViewMembership>& memberships,
                                             uint64_t&                                    outIdCount) noexcept
{
    outIdCount = 0u;
    if (base.mGeneration == nullptr || objectCount < base.mGeneration->objects.size() || objectCount > UINT32_MAX ||
        memberships.size() != base.mGeneration->viewMemberships.size())
    {
        return false;
    }
    for (size_t index = 0u; index < memberships.size(); ++index)
    {
        const GenerationViewMembership& current  = memberships[index];
        const GenerationViewMembership& previous = base.mGeneration->viewMemberships[index];
        if (current.view != previous.view || current.selectedIds == nullptr || previous.selectedIds == nullptr)
        {
            outIdCount = 0u;
            return false;
        }
        if (current.selectedIds == previous.selectedIds)
        {
            continue;
        }
        // The size of the symmetric difference of two ascending ID lists.
        auto first  = previous.selectedIds->begin();
        auto second = current.selectedIds->begin();
        while (first != previous.selectedIds->end() && second != current.selectedIds->end())
        {
            if (*first == *second)
            {
                ++first;
                ++second;
                continue;
            }
            ++outIdCount;
            ++(*first < *second ? first : second);
        }
        outIdCount += static_cast<uint64_t>(previous.selectedIds->end() - first) +
                      static_cast<uint64_t>(current.selectedIds->end() - second);
    }
    return true;
}

ovspatialquery_api_status_t PopulationTable::create(uint32_t retainedOrdinalCapacity, PopulationKey& outHandle,
                                                    uint32_t pendingPublicationCapacity)
{
    if (mImpl->backendRuntime == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    return createSlot(retainedOrdinalCapacity, {}, outHandle, pendingPublicationCapacity);
}

ovspatialquery_api_status_t PopulationTable::create(uint32_t                      retainedOrdinalCapacity,
                                                    ovspatialquery_ordinal_t      initialOrdinal,
                                                    std::vector<PopulationObject> objects, PopulationKey& outHandle)
{
    if (mImpl->backendRuntime == nullptr || retainedOrdinalCapacity < 2u || initialOrdinal < 0)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    GenerationLease             initialGeneration;
    ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
    try
    {
        status = buildGeneration(initialOrdinal, ObjectTable(std::move(objects)), {}, initialGeneration);
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    return createSlot(retainedOrdinalCapacity, initialGeneration.mGeneration, outHandle);
}

ovspatialquery_api_status_t PopulationTable::createSlot(uint32_t retainedOrdinalCapacity,
                                                        std::shared_ptr<const PopulationGeneration> initialGeneration,
                                                        PopulationKey& outHandle, uint32_t pendingPublicationCapacity)
{
    try
    {
        std::shared_ptr<PopulationSlot> slot =
            std::make_shared<PopulationSlot>(retainedOrdinalCapacity, pendingPublicationCapacity);
        if (initialGeneration != nullptr)
        {
            std::shared_ptr<const PopulationGeneration> evicted;
            slot->versions.publish(std::move(initialGeneration), evicted);
        }
        // Entries are never erased and keys only rise, so the insert always succeeds.
        std::map<uint32_t, PopulationEntry> prepared;
        prepared.emplace(0u, PopulationEntry{std::move(slot), 1u});
        std::map<uint32_t, PopulationEntry>::node_type node = prepared.extract(prepared.begin());
        std::lock_guard<std::mutex>                    lock(mImpl->mutex);
        if (mImpl->freeHead != std::numeric_limits<uint32_t>::max())
        {
            const uint32_t   slotIndex = mImpl->freeHead;
            PopulationEntry& entry     = mImpl->entries.find(slotIndex)->second;
            mImpl->freeHead            = entry.nextFree;
            entry.nextFree             = std::numeric_limits<uint32_t>::max();
            entry.population           = std::move(node.mapped().population);
            ++mImpl->populationCount;
            outHandle = {slotIndex, entry.generation};
            return OVSPATIALQUERY_API_SUCCESS;
        }
        if (mImpl->nextSlot > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        const uint32_t slotIndex = static_cast<uint32_t>(mImpl->nextSlot++);
        node.key()               = slotIndex;
        mImpl->entries.insert(std::move(node));
        ++mImpl->populationCount;
        outHandle = {slotIndex, 1u};
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

ovspatialquery_api_status_t PopulationTable::release(PopulationKey          handle,
                                                     std::shared_ptr<void>& outDeferredDestruction)
{
    if (outDeferredDestruction != nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    std::shared_ptr<PopulationSlot> releasedSlot;
    {
        std::lock_guard<std::mutex>           lock(mImpl->mutex);
        const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
        if (slot == nullptr)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        if (slot->pendingOperationCount != 0u || slot->mutationPending())
        {
            return OVSPATIALQUERY_API_BUSY;
        }
        PopulationEntry& entry = mImpl->entries.find(handle.slot)->second;
        if (entry.generation != std::numeric_limits<uint32_t>::max())
        {
            entry.nextFree  = mImpl->freeHead;
            mImpl->freeHead = handle.slot;
            ++entry.generation;
        }
        releasedSlot = std::move(entry.population);
        --mImpl->populationCount;
    }
    outDeferredDestruction = std::move(releasedSlot);
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PopulationTable::pin(PopulationKey handle, GenerationLease& outLease) const
{
    GenerationLease lease;
    {
        std::lock_guard<std::mutex>           lock(mImpl->mutex);
        const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
        if (slot == nullptr)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        if (!slot->versions.ready())
        {
            return OVSPATIALQUERY_API_NOT_READY;
        }
        lease = GenerationLease(slot->versions.head());
    }
    outLease = std::move(lease);
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PopulationTable::pinExact(PopulationKey handle, ovspatialquery_ordinal_t ordinal,
                                                      GenerationLease& outLease) const
{
    if (ordinal < 0)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    GenerationLease lease;
    {
        std::lock_guard<std::mutex>           lock(mImpl->mutex);
        const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
        if (slot == nullptr)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        if (!slot->versions.ready())
        {
            return OVSPATIALQUERY_API_NOT_READY;
        }
        std::shared_ptr<const PopulationGeneration> generation = slot->versions.exact(ordinal);
        if (generation == nullptr)
        {
            return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
        }
        lease = GenerationLease(std::move(generation));
    }
    outLease = std::move(lease);
    return OVSPATIALQUERY_API_SUCCESS;
}

TargetResolveOutcome PopulationTable::pinResolvedTarget(const GenerationSelectionRequest& request,
                                                        GenerationLease&                  outLease) const
{
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(request.population);
    if (slot == nullptr)
    {
        return TargetResolveOutcome::invalid;
    }
    std::shared_ptr<const PopulationGeneration> generation;
    const TargetResolveOutcome                  selected = selectGeneration(*slot, request.selection, generation);
    if (selected != TargetResolveOutcome::success)
    {
        return selected;
    }
    outLease = GenerationLease(std::move(generation));
    return TargetResolveOutcome::success;
}

TargetResolveOutcome PopulationTable::pinResolvedPair(const std::array<GenerationSelectionRequest, 2u>& requests,
                                                      std::array<GenerationLease, 2u>&                  outLeases) const
{
    std::array<GenerationLease, 2u> leases{};
    std::lock_guard<std::mutex>     lock(mImpl->mutex);
    for (size_t index = 0u; index < requests.size(); ++index)
    {
        const GenerationSelectionRequest&     request = requests[index];
        const std::shared_ptr<PopulationSlot> slot    = mImpl->resolve(request.population);
        if (slot == nullptr)
        {
            return TargetResolveOutcome::invalid;
        }
        std::shared_ptr<const PopulationGeneration> generation;
        const TargetResolveOutcome                  selected = selectGeneration(*slot, request.selection, generation);
        if (selected != TargetResolveOutcome::success)
        {
            return selected;
        }
        if (index != 0u && requests[0].population == request.population && leases[0].ordinal() != generation->ordinal)
        {
            return TargetResolveOutcome::ordinalMismatch;
        }
        leases[index] = GenerationLease(std::move(generation));
#if defined(OVSPATIALQUERY_ENABLE_TARGET_RESOLVER_TEST_HOOK)
        if (index == 0u)
        {
            const TargetResolverAfterFirstPinHook hook = gAfterFirstPinHook.load(std::memory_order_acquire);
            if (hook != nullptr)
            {
                hook();
            }
        }
#endif
    }
    outLeases = std::move(leases);
    return TargetResolveOutcome::success;
}

ovspatialquery_api_status_t PopulationTable::getUpdatePending(PopulationKey handle, bool& outUpdatePending) const
{
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
    if (slot == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    outUpdatePending = slot->mutationPending();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PopulationTable::getInfoState(PopulationKey handle, bool& outReady,
                                                          ovspatialquery_ordinal_t& outOrdinal,
                                                          uint64_t& outObjectCount, uint32_t& outRetainedOrdinalCount,
                                                          bool& outUpdatePending) const
{
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
    if (slot == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const bool ready        = slot->versions.ready();
    outReady                = ready;
    outOrdinal              = ready ? slot->versions.head()->ordinal : 0;
    outObjectCount          = ready ? GenerationLease(slot->versions.head()).objectCount() : 0u;
    outRetainedOrdinalCount = slot->versions.retainedOrdinalCount();
    outUpdatePending        = slot->mutationPending();
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PopulationTable::validate(PopulationKey handle) const
{
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
    if (slot == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PopulationTable::beginUpdate(PopulationKey handle, ovspatialquery_ordinal_t targetOrdinal,
                                                         const ovspatialquery_ordinal_t* exactBaseOrdinal,
                                                         GenerationLease&                outCurrentGeneration,
                                                         bool requireUnreadyPopulation, bool chainPendingPublication)
{
    if (targetOrdinal < 0 || (exactBaseOrdinal != nullptr && *exactBaseOrdinal < 0))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    GenerationLease currentGeneration;
    {
        std::lock_guard<std::mutex>           lock(mImpl->mutex);
        const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
        if (slot == nullptr)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        if (slot->mutationPending())
        {
            // ADR-0021: a device-prepared update may build on the newest admitted
            // publication while others are pending; the first set must complete
            // first. Publications complete in any order, so the newest admitted
            // ordinal is either the newest pending one or an already published
            // head that overtook it.
            const PendingPublication&      newest       = slot->pending.back();
            const bool                     ready        = slot->versions.ready();
            const bool                     headIsNewest = ready && slot->versions.head()->ordinal > newest.ordinal;
            const ovspatialquery_ordinal_t newestOrdinal =
                headIsNewest ? slot->versions.head()->ordinal : newest.ordinal;
            if (!chainPendingPublication || exactBaseOrdinal == nullptr || !ready ||
                (!headIsNewest && newest.admitted == nullptr) || slot->pending.size() >= slot->pendingCapacity)
            {
                return OVSPATIALQUERY_API_BUSY;
            }
            if (*exactBaseOrdinal != newestOrdinal)
            {
                return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
            }
            if (targetOrdinal <= newestOrdinal)
            {
                return OVSPATIALQUERY_API_STALE_ORDINAL;
            }
            currentGeneration =
                headIsNewest ? GenerationLease(slot->versions.head()) : GenerationLease(newest.admitted);
        }
        else
        {
            if (requireUnreadyPopulation && slot->versions.ready())
            {
                return OVSPATIALQUERY_API_INVALID_STATE;
            }
            if (!slot->versions.ready())
            {
                if (exactBaseOrdinal != nullptr)
                {
                    return OVSPATIALQUERY_API_NOT_READY;
                }
            }
            else if (exactBaseOrdinal != nullptr && *exactBaseOrdinal != slot->versions.head()->ordinal)
            {
                return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
            }
            if (slot->versions.ready() && targetOrdinal <= slot->versions.head()->ordinal)
            {
                return OVSPATIALQUERY_API_STALE_ORDINAL;
            }
            if (slot->versions.ready())
            {
                currentGeneration = GenerationLease(slot->versions.head());
            }
        }
        PendingPublication publication;
        publication.ordinal         = targetOrdinal;
        publication.stagingReserved = true;
        slot->pending.push_back(std::move(publication));
        ++slot->pendingOperationCount;
    }
    outCurrentGeneration = std::move(currentGeneration);
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PopulationTable::admitPending(PopulationKey handle, const GenerationLease& admitted)
{
    if (!admitted.initialized())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
    if (slot == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    for (PendingPublication& publication : slot->pending)
    {
        if (publication.ordinal == admitted.ordinal())
        {
            publication.admitted = admitted.mGeneration;
            return OVSPATIALQUERY_API_SUCCESS;
        }
    }
    return OVSPATIALQUERY_API_INTERNAL;
}

ovspatialquery_api_status_t PopulationTable::publishUpdate(PopulationKey handle, const GenerationLease& generation)
{
    if (!generation.initialized())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    std::shared_ptr<const PopulationGeneration> evicted;
    {
        std::lock_guard<std::mutex>           lock(mImpl->mutex);
        const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
        if (slot == nullptr || !slot->mutationPending())
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        std::vector<PendingPublication>::iterator publication = slot->pending.begin();
        for (; publication != slot->pending.end(); ++publication)
        {
            if (publication->ordinal == generation.ordinal())
            {
                break;
            }
        }
        if (publication == slot->pending.end())
        {
            return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
        }
        if (!publication->stagingReserved)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        // Workers complete publications in any order. One whose successor has
        // already become the head joins the retained window behind it, so the
        // head never moves backwards (ADR-0021). Ordinals ascend along a chain.
        if (slot->versions.ready() && slot->versions.head()->ordinal > generation.ordinal())
        {
            slot->versions.insertBehindHead(generation.mGeneration, evicted);
        }
        else
        {
            slot->versions.publish(generation.mGeneration, evicted);
        }
        publication->stagingReserved = false;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

void PopulationTable::finishUpdate(PopulationKey handle)
{
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
    if (slot != nullptr && slot->mutationPending())
    {
        slot->pending.pop_back();
        if (slot->pendingOperationCount != 0u)
        {
            --slot->pendingOperationCount;
        }
    }
}

void PopulationTable::finishUpdate(PopulationKey handle, ovspatialquery_ordinal_t targetOrdinal)
{
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
    if (slot == nullptr)
    {
        return;
    }
    for (std::vector<PendingPublication>::iterator publication = slot->pending.begin();
         publication != slot->pending.end(); ++publication)
    {
        if (publication->ordinal == targetOrdinal)
        {
            slot->pending.erase(publication);
            if (slot->pendingOperationCount != 0u)
            {
                --slot->pendingOperationCount;
            }
            return;
        }
    }
}

void PopulationTable::finishPersistentReferences(const std::vector<PopulationKey>& handles)
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    for (size_t handleIndex = 0u; handleIndex < handles.size(); ++handleIndex)
    {
        bool alreadyReleased = false;
        for (size_t earlierIndex = 0u; earlierIndex < handleIndex; ++earlierIndex)
        {
            if (handles[earlierIndex] == handles[handleIndex])
            {
                alreadyReleased = true;
                break;
            }
        }
        const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handles[handleIndex]);
        if (!alreadyReleased && slot != nullptr && slot->pendingOperationCount != 0u)
        {
            --slot->pendingOperationCount;
        }
    }
}

ovspatialquery_api_status_t PopulationTable::retainPersistentReferences(const std::vector<PopulationKey>& handles)
{
    try
    {
        std::vector<std::shared_ptr<PopulationSlot>> slots;
        slots.reserve(handles.size());
        {
            std::lock_guard<std::mutex> lock(mImpl->mutex);
            for (const PopulationKey handle : handles)
            {
                const std::shared_ptr<PopulationSlot> slot = mImpl->resolve(handle);
                if (slot == nullptr)
                {
                    return OVSPATIALQUERY_API_INVALID_HANDLE;
                }
                slots.push_back(slot);
            }
            for (size_t index = 0u; index < slots.size(); ++index)
            {
                bool alreadyRetained = false;
                for (size_t earlier = 0u; earlier < index; ++earlier)
                {
                    if (slots[earlier].get() == slots[index].get())
                    {
                        alreadyRetained = true;
                        break;
                    }
                }
                if (!alreadyRetained)
                {
                    ++slots[index]->pendingOperationCount;
                }
            }
        }
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

std::weak_ptr<GenerationDeviceAttachments> PopulationTable::lastDeviceAttachments(PopulationKey handle,
                                                                                  int32_t       deviceId) const
{
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> population = mImpl->resolve(handle);
    if (population != nullptr)
    {
        for (const std::pair<int32_t, std::weak_ptr<GenerationDeviceAttachments>>& entry :
             population->lastDeviceAttachments)
        {
            if (entry.first == deviceId)
            {
                return entry.second;
            }
        }
    }
    return {};
}

void PopulationTable::setLastDeviceAttachments(PopulationKey handle, int32_t deviceId,
                                               std::weak_ptr<GenerationDeviceAttachments> attachments) const
{
    std::lock_guard<std::mutex>           lock(mImpl->mutex);
    const std::shared_ptr<PopulationSlot> population = mImpl->resolve(handle);
    if (population == nullptr)
    {
        return;
    }
    for (std::pair<int32_t, std::weak_ptr<GenerationDeviceAttachments>>& entry : population->lastDeviceAttachments)
    {
        if (entry.first == deviceId)
        {
            entry.second = std::move(attachments);
            return;
        }
    }
    try
    {
        population->lastDeviceAttachments.emplace_back(deviceId, std::move(attachments));
    }
    catch (...)
    {
        // Without the entry the next copy is uploaded whole instead of derived.
    }
}

std::mutex& PopulationTable::deviceCopyMutex() const { return mImpl->deviceCopyMutex; }

size_t PopulationTable::populationCount() const
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->populationCount;
}

} // namespace detail
} // namespace ovspatialquery
