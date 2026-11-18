/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_instance_state.h"

#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_input_layout.h"
#include "ovspatialquery_instance_operation_runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <thread>
#include <type_traits>

using namespace ovspatialquery::detail;

namespace
{

// Heap-sort the parallel columns without allocating temporary result storage.
void sortSourceExclusions(SourceExclusions& exclusions) noexcept
{
    const auto swapRows = [&](size_t first, size_t second)
    {
        std::swap(exclusions.objects[first], exclusions.objects[second]);
        std::swap(exclusions.reasons[first], exclusions.reasons[second]);
    };
    const auto sift = [&](size_t root, size_t count)
    {
        while (root < count / 2u)
        {
            size_t child = root * 2u + 1u;
            if (child + 1u < count && exclusions.objects[child] < exclusions.objects[child + 1u])
            {
                ++child;
            }
            if (exclusions.objects[root] >= exclusions.objects[child])
            {
                break;
            }
            swapRows(root, child);
            root = child;
        }
    };
    const size_t count = exclusions.objects.size();
    for (size_t root = count / 2u; root != 0u;)
    {
        sift(--root, count);
    }
    for (size_t end = count; end > 1u;)
    {
        swapRows(0u, --end);
        sift(0u, end);
    }
}

std::atomic<uint32_t> gNextInstanceOperationIdSalt{1u};

thread_local std::string gLastApiError;

constexpr uint64_t kInstanceControlAndDirectoryNodeAllowance = 256u;
constexpr uint64_t kWorkerLaunchStateAllowance               = 256u;
// A generation's fixed bytes, and per object the backend's structures; the
// objects themselves are charged with the batches that hold them, which
// versions share (ADR-0019).
constexpr uint64_t kGenerationBaseCharge   = 1024u;
constexpr uint64_t kGenerationObjectCharge = 256u;

bool estimateGenerationCharge(const ovspatialquery::detail::ObjectTable&                           objects,
                              const std::vector<ovspatialquery::detail::GenerationViewMembership>& memberships,
                              const std::shared_ptr<const std::vector<uint64_t>>&                  usedObjectValues,
                              const ovspatialquery::detail::VersionChangeRecord& changes, uint64_t& outPublishedCharge,
                              uint64_t& outGeometryDetailCharge, std::optional<uint64_t> measuredGeometryCharge)
{
    constexpr uint64_t perObject   = kGenerationObjectCharge;
    uint64_t           objectBytes = 0u;
    if (objects.size() > static_cast<size_t>(std::numeric_limits<uint64_t>::max() / perObject))
    {
        return false;
    }
    objectBytes         = static_cast<uint64_t>(objects.size()) * perObject + objects.chunkEntryBytes();
    const uint64_t base = kGenerationBaseCharge;
    if (objectBytes > std::numeric_limits<uint64_t>::max() - base)
    {
        return false;
    }
    outPublishedCharge = base + objectBytes;
    for (const ovspatialquery::detail::GenerationViewMembership& membership : memberships)
    {
        const uint64_t membershipBytes = membership.residentByteCount();
        if (membershipBytes == std::numeric_limits<uint64_t>::max() ||
            membershipBytes > std::numeric_limits<uint64_t>::max() - outPublishedCharge)
        {
            return false;
        }
        outPublishedCharge += membershipBytes;
    }
    const size_t usedObjectValueCapacity = usedObjectValues == nullptr ? 0u : usedObjectValues->capacity();
    if (usedObjectValueCapacity > static_cast<size_t>(std::numeric_limits<uint64_t>::max() / sizeof(uint64_t)))
    {
        return false;
    }
    const uint64_t usedObjectValueBytes = static_cast<uint64_t>(usedObjectValueCapacity) * sizeof(uint64_t);
    if (usedObjectValueBytes > std::numeric_limits<uint64_t>::max() - outPublishedCharge)
    {
        return false;
    }
    outPublishedCharge += usedObjectValueBytes;
    if (changes.residentByteCount() > std::numeric_limits<uint64_t>::max() - outPublishedCharge)
    {
        return false;
    }
    outPublishedCharge += changes.residentByteCount();
    outGeometryDetailCharge           = 0u;
    const uint64_t exactGeometryBytes = measuredGeometryCharge
                                            ? *measuredGeometryCharge
                                            : ovspatialquery::detail::exactGeometryResidentByteCount(objects);
    if (exactGeometryBytes == std::numeric_limits<uint64_t>::max())
    {
        return false;
    }
    outGeometryDetailCharge = exactGeometryBytes;
    return true;
}

struct GenerationMemoryLease
{
    // Declared first so the device charge is released before the atomic host
    // lease is.
    ovspatialquery::detail::HostLease hostLease;
    std::shared_ptr<void>             deviceLease;
};

std::shared_ptr<void> makeGenerationMemoryLease(ovspatialquery::detail::HostLease hostLease,
                                                std::shared_ptr<void>             deviceLease = {})
{
    std::shared_ptr<GenerationMemoryLease> lease = std::make_shared<GenerationMemoryLease>();
    lease->hostLease                             = std::move(hostLease);
    lease->deviceLease                           = std::move(deviceLease);
    return lease;
}

// Charges a version's batch and its own bytes, taking them first from what
// its publication reserved at admission and then from the ledger, all or none.
bool chargeGenerationHostBytes(InstanceState& instance, ovspatialquery::detail::HostLease* admitted,
                               uint64_t batchCharge, uint64_t generationCharge,
                               ovspatialquery::detail::HostLease& outBatch,
                               ovspatialquery::detail::HostLease& outGeneration) noexcept
{
    if (batchCharge > std::numeric_limits<uint64_t>::max() - generationCharge)
    {
        return false;
    }
    const uint64_t                          needed = batchCharge + generationCharge;
    const uint64_t                          carved = admitted == nullptr ? 0u : std::min(needed, admitted->bytes());
    ovspatialquery::detail::HostLease       charge;
    ovspatialquery::detail::HostTransaction transaction(instance.hostLedger);
    if (!transaction.reserve(needed - carved) || !transaction.commit(charge))
    {
        return false;
    }
    ovspatialquery::detail::HostLease carvedLease;
    if (carved != 0u && (!admitted->split(carved, carvedLease) || !charge.absorb(carvedLease)))
    {
        return false;
    }
    if (!charge.split(batchCharge, outBatch))
    {
        return false;
    }
    outGeneration = std::move(charge);
    return true;
}

// Charges a version's objects and structures, moving the device charge its
// update holds into the generation's lease, and builds its generation. The
// batch of objects the version adds is its own charge: later versions may
// share it after this one. A host-prepared update passes its admission charge,
// which covers the version (publicationAdmissionCharge).
ovspatialquery_api_status_t buildChargedGeneration(
    InstanceState& instance, PopulationUpdateTask& update, ovspatialquery_ordinal_t targetOrdinal,
    ovspatialquery::detail::ObjectTable                           objects,
    std::vector<ovspatialquery::detail::GenerationViewMembership> memberships,
    std::shared_ptr<const std::vector<uint64_t>> usedObjectValues, ovspatialquery::detail::VersionChangeRecord changes,
    std::optional<uint64_t>                                              measuredGeometryCharge,
    std::shared_ptr<const ovspatialquery::detail::CudaPopulationVersion> version, bool requiresCpuQueryBackend,
    const ovspatialquery::detail::GenerationLease& base, ovspatialquery::detail::HostLease* admittedCharge,
    ovspatialquery::detail::GenerationLease& outStaged)
{
    uint64_t publishedCharge      = 0u;
    uint64_t geometryDetailCharge = 0u;
    if (!estimateGenerationCharge(objects, memberships, usedObjectValues, changes, publishedCharge,
                                  geometryDetailCharge, measuredGeometryCharge))
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    ovspatialquery::detail::HostLease batchCharge;
    ovspatialquery::detail::HostLease generationCharge;
    if (publishedCharge > std::numeric_limits<uint64_t>::max() - geometryDetailCharge ||
        !chargeGenerationHostBytes(instance, admittedCharge, objects.ownBatchBytes(),
                                   publishedCharge + geometryDetailCharge, batchCharge, generationCharge))
    {
        return OVSPATIALQUERY_API_HOST_BUDGET;
    }
    std::shared_ptr<void> generationLease =
        makeGenerationMemoryLease(std::move(generationCharge), update.publishedDeviceLease);
    objects.attachBatchLease(makeGenerationMemoryLease(std::move(batchCharge)));
    update.publishedDeviceLease.reset();
    return instance.populations.buildGeneration(targetOrdinal, std::move(objects), std::move(memberships), outStaged,
                                                std::move(generationLease), std::move(version), requiresCpuQueryBackend,
                                                std::move(usedObjectValues), geometryDetailCharge, std::move(changes),
                                                base);
}

ovspatialquery_api_status_t prepareUsedObjectValues(
    const ovspatialquery::detail::GenerationLease& base, const ovspatialquery::detail::ObjectTable& objects,
    const std::optional<ovspatialquery::detail::VersionChangeRecord>& knownChanges,
    std::shared_ptr<const std::vector<uint64_t>>&                     outUsedObjectValues)
{
    try
    {
        // A record that lists no appended object keeps every identity at its
        // position, so the history is the base's (ADR-0019).
        if (base.initialized() && knownChanges && !knownChanges->everything &&
            knownChanges->baseOrdinal == base.ordinal() && objects.size() == base.objects().size())
        {
            outUsedObjectValues = base.sharedUsedObjectValues();
            return OVSPATIALQUERY_API_SUCCESS;
        }
        if (base.initialized() && base.objects().size() == objects.size() &&
            base.objects().size() == base.usedObjectValues().size())
        {
            bool sameObjectValues = true;
            for (size_t index = 0u; index < objects.size(); ++index)
            {
                const uint64_t value = objects[index].id.value;
                if (value == 0u || base.objects()[index].id.value != value || base.usedObjectValues()[index] != value ||
                    (index != 0u && base.usedObjectValues()[index - 1u] >= value))
                {
                    sameObjectValues = false;
                    break;
                }
            }
            if (sameObjectValues)
            {
                outUsedObjectValues = base.sharedUsedObjectValues();
                return OVSPATIALQUERY_API_SUCCESS;
            }
        }
        // Validate the new head's values against the sorted history in one ordered
        // walk: a value the history holds, or a repeated value, may appear only
        // while the base still holds it. Explicit heads arrive in ID order, so
        // sorting is a fallback.
        const auto sortedValues = [](const auto& source)
        {
            std::vector<uint64_t> values;
            values.reserve(source.size());
            for (const ovspatialquery::detail::PopulationObject& object : source)
            {
                values.push_back(object.id.value);
            }
            if (!std::is_sorted(values.begin(), values.end()))
            {
                std::sort(values.begin(), values.end());
            }
            return values;
        };
        const std::vector<uint64_t> values = sortedValues(objects);
        const std::vector<uint64_t> activeObjectValues =
            base.initialized() ? sortedValues(base.objects()) : std::vector<uint64_t>{};
        static const std::vector<uint64_t> noHistory;
        const std::vector<uint64_t>&       previous = base.initialized() ? base.usedObjectValues() : noHistory;
        std::vector<uint64_t>              newValues;
        size_t                             used   = 0u;
        size_t                             active = 0u;
        for (size_t index = 0u; index < values.size(); ++index)
        {
            const uint64_t value = values[index];
            while (used < previous.size() && previous[used] < value)
            {
                ++used;
            }
            if ((used == previous.size() || previous[used] != value) && (index == 0u || values[index - 1u] != value))
            {
                newValues.push_back(value);
                continue;
            }
            while (active < activeObjectValues.size() && activeObjectValues[active] < value)
            {
                ++active;
            }
            if (active == activeObjectValues.size() || activeObjectValues[active] != value)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
        }
        std::vector<uint64_t> candidate;
        candidate.reserve(previous.size() + newValues.size());
        std::merge(previous.begin(), previous.end(), newValues.begin(), newValues.end(), std::back_inserter(candidate));
        if ((!candidate.empty() && candidate.front() == 0u) ||
            std::adjacent_find(candidate.begin(), candidate.end()) != candidate.end())
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        outUsedObjectValues = std::make_shared<const std::vector<uint64_t>>(std::move(candidate));
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

const char* apiStatusMessage(ovspatialquery_api_status_t status)
{
    switch (status)
    {
    case OVSPATIALQUERY_API_ERROR: return "backend or runtime error";
    case OVSPATIALQUERY_API_TIMEOUT: return "operation is still pending";
    case OVSPATIALQUERY_API_INVALID_ARGUMENT: return "invalid argument";
    case OVSPATIALQUERY_API_INVALID_HANDLE: return "invalid handle";
    case OVSPATIALQUERY_API_INVALID_STATE: return "invalid state";
    case OVSPATIALQUERY_API_NOT_SUPPORTED: return "operation is not supported";
    case OVSPATIALQUERY_API_NOT_READY: return "resource is not ready";
    case OVSPATIALQUERY_API_BUSY: return "resource is busy";
    case OVSPATIALQUERY_API_STALE_ORDINAL: return "target ordinal does not advance";
    case OVSPATIALQUERY_API_ORDINAL_MISMATCH: return "base ordinal does not match";
    case OVSPATIALQUERY_API_OPERATION_CAPACITY: return "operation capacity is full";
    case OVSPATIALQUERY_API_RESULT_CAPACITY: return "result capacity is full";
    case OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED: return "requested ordinal is not retained";
    case OVSPATIALQUERY_API_BACKWARD_ORDINAL: return "reporter ordinal moves backward";
    case OVSPATIALQUERY_API_HOST_BUDGET: return "host byte budget is full";
    case OVSPATIALQUERY_API_DEVICE_BUDGET: return "device byte budget is full";
    case OVSPATIALQUERY_API_OPERATION_FAILED: return "operation failed";
    case OVSPATIALQUERY_API_OUT_OF_MEMORY: return "allocation failed";
    case OVSPATIALQUERY_API_INTERNAL: return "internal invariant failed";
    case OVSPATIALQUERY_API_SUCCESS:
    default: return "";
    }
}

} // namespace

namespace ovspatialquery
{
namespace detail
{

// The salt is the instance counter with its bits reversed: two counters first
// differ in a low bit, so their salts differ in a high bit, and an ID from an
// instance created k instances apart decodes to a generation of at least
// 2^31 / k, which a slot reaches only after that many reuses.
uint32_t nextInstanceOperationIdSalt() noexcept
{
    uint32_t counter = gNextInstanceOperationIdSalt.fetch_add(1u, std::memory_order_relaxed);
    uint32_t salt    = 0u;
    for (uint32_t bit = 0u; bit < 32u; ++bit, counter >>= 1u)
    {
        salt = (salt << 1u) | (counter & 1u);
    }
    return salt;
}

// Never destroyed: an instance its host never released stays to process exit, and destroying it
// from static destruction would call Ovstage, PhysX and CUDA after their owners (a stage the
// host already destroyed, a runtime already shut down) are gone. Constructed in place, so first
// use cannot fail to allocate.
InstanceDirectory& instanceDirectory()
{
    alignas(InstanceDirectory) static unsigned char storage[sizeof(InstanceDirectory)];
    static InstanceDirectory* const                 directory = new (storage) InstanceDirectory();
    return *directory;
}

InstanceReference instanceState(ovspatialquery_instance_handle_t handle)
{
    return InstanceReference(instanceDirectory().resolve(handle));
}

ovspatialquery_result_t apiResult(ovspatialquery_api_status_t status, const char* message) noexcept
{
    try
    {
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            gLastApiError.clear();
        }
        else
        {
            gLastApiError = message == nullptr ? apiStatusMessage(status) : message;
        }
    }
    catch (...)
    {
        gLastApiError.clear();
    }
    return ovspatialquery_result_t{status};
}

ovspatialquery_api_status_t reserveSharedHostLease(InstanceState* instance, uint64_t bytes,
                                                   std::shared_ptr<ovspatialquery::detail::HostLease>& outLease)
{
    outLease.reset();
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    ovspatialquery::detail::HostTransaction transaction(instance->hostLedger);
    if (!transaction.reserve(bytes))
    {
        return OVSPATIALQUERY_API_HOST_BUDGET;
    }
    ovspatialquery::detail::HostLease lease;
    if (!transaction.commit(lease))
    {
        return OVSPATIALQUERY_API_HOST_BUDGET;
    }
    try
    {
        outLease = std::make_shared<ovspatialquery::detail::HostLease>(std::move(lease));
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

ovspatialquery_op_id_t makeOperationId(const InstanceState* instance, uint32_t slotIndex, uint32_t generation)
{
    const uint32_t encodedGeneration = generation ^ instance->operationIdSalt;
    return ovspatialquery_op_id_t{(static_cast<uint64_t>(encodedGeneration) << 32u) |
                                  static_cast<uint64_t>(slotIndex + 1u)};
}

bool decodeOperationId(const InstanceState* instance, ovspatialquery_op_id_t operation, uint32_t& outSlotIndex,
                       uint32_t& outGeneration)
{
    if (instance == nullptr)
    {
        return false;
    }
    const uint32_t encodedSlot = static_cast<uint32_t>(operation.raw_id);
    const uint32_t generation  = static_cast<uint32_t>(operation.raw_id >> 32u) ^ instance->operationIdSalt;
    if (encodedSlot == 0u || generation == 0u)
    {
        return false;
    }
    outSlotIndex  = encodedSlot - 1u;
    outGeneration = generation;
    return true;
}

OperationSlot* resolveOperation(InstanceState* instance, ovspatialquery_op_id_t operation, uint32_t& outSlotIndex,
                                uint32_t& outGeneration)
{
    if (!decodeOperationId(instance, operation, outSlotIndex, outGeneration))
    {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(instance->operationMutex);
    if (outSlotIndex >= instance->operationSlots.size())
    {
        return nullptr;
    }
    OperationSlot* slot = instance->operationSlots[outSlotIndex].get();
    return slot->inUse && slot->generation == outGeneration ? slot : nullptr;
}

ApiOperationOwner apiOperationOwner(InstanceState& instance, ovspatialquery_op_id_t operation)
{
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(&instance, operation, slotIndex, generation);
    if (slot == nullptr)
    {
        return {};
    }
    std::lock_guard<std::mutex> slotLock(slot->mutex);
    return slot->inUse && slot->generation == generation ? slot->apiOwner : ApiOperationOwner{};
}

void apiSetOperationOwner(InstanceState& instance, ovspatialquery_op_id_t operation, ApiOperationOwner owner)
{
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(&instance, operation, slotIndex, generation);
    if (slot != nullptr)
    {
        std::lock_guard<std::mutex> slotLock(slot->mutex);
        if (slot->inUse && slot->generation == generation)
        {
            slot->apiOwner = std::move(owner);
        }
    }
}

ovspatialquery_api_status_t reserveOperation(InstanceState* instance, OperationKind kind, uint32_t& outSlotIndex,
                                             ovspatialquery_op_id_t& outOperation)
{
    std::lock_guard<std::mutex> operationLock(instance->operationMutex);
    for (uint32_t slotIndex = 0u; slotIndex < instance->operationSlots.size(); ++slotIndex)
    {
        OperationSlot& slot = *instance->operationSlots[slotIndex];
        if (!slot.inUse && !slot.workerActive && slot.generation != std::numeric_limits<uint32_t>::max())
        {
            std::lock_guard<std::mutex> slotLock(slot.mutex);
            slot.inUse             = true;
            slot.terminal          = false;
            slot.terminalObserved  = false;
            slot.kind              = kind;
            slot.executionStatus   = OVSPATIALQUERY_API_SUCCESS;
            slot.privateOwner      = nullptr;
            slot.privateFailure    = OVSPATIALQUERY_API_SUCCESS;
            slot.privateDiagnostic = nullptr;
            slot.task.reset();
            slot.taskLease.reset();
            ++instance->liveOperationCount;
            outSlotIndex = slotIndex;
            outOperation = makeOperationId(instance, slotIndex, slot.generation);
            return OVSPATIALQUERY_API_SUCCESS;
        }
    }
    return OVSPATIALQUERY_API_OPERATION_CAPACITY;
}

bool reserveHostLease(InstanceState& instance, uint64_t bytes, ovspatialquery::detail::HostLease& outLease) noexcept
{
    ovspatialquery::detail::HostTransaction transaction(instance.hostLedger);
    return transaction.reserve(bytes) && transaction.commit(outLease);
}

uint64_t publicationAdmissionCharge(PreparedPopulationInput& input, const GenerationLease& base,
                                    uint64_t membershipBytes)
{
    constexpr uint64_t saturated = std::numeric_limits<uint64_t>::max();
    const uint64_t     retained  = input.retainedByteCount();
    if (retained == saturated)
    {
        return saturated;
    }
    // The materialization peak includes the retained input.
    const uint64_t peak = input.materializationPeakByteCount(
        base.initialized() ? static_cast<uint64_t>(base.objects().size()) : 0u, retained);
    if (input.cudaCapture() != nullptr)
    {
        return peak > saturated - retained ? saturated : retained + peak;
    }
    // The version holds the input's objects, their chunk entries and exact
    // geometry, and its own record of changes, which moves from the input
    // when it has one. Beyond the input it adds the bytes that
    // estimateGenerationCharge counts per object, the view memberships, the
    // object values used so far, shared with the base or grown from it, and a
    // described record of changes, which holds at most one position per object.
    const uint64_t objectCount    = input.objectCount();
    const uint64_t usedValueCount = base.initialized() && base.sharedUsedObjectValues() != nullptr
                                        ? static_cast<uint64_t>(base.sharedUsedObjectValues()->capacity())
                                        : 0u;
    uint64_t       charge         = retained;
    if (membershipBytes > saturated - kGenerationBaseCharge ||
        !addHostCharge(1u, kGenerationBaseCharge + membershipBytes, charge) ||
        !addHostCharge(objectCount, kGenerationObjectCharge + sizeof(uint64_t) + sizeof(uint32_t), charge) ||
        !addHostCharge(usedValueCount, sizeof(uint64_t), charge))
    {
        return saturated;
    }
    return std::max(peak, charge);
}

void abandonReservedOperation(InstanceState* instance, uint32_t slotIndex)
{
    std::lock_guard<std::mutex> operationLock(instance->operationMutex);
    OperationSlot&              slot = *instance->operationSlots[slotIndex];
    std::lock_guard<std::mutex> slotLock(slot.mutex);
    slot.privateOwner      = nullptr;
    slot.privateFailure    = OVSPATIALQUERY_API_SUCCESS;
    slot.privateDiagnostic = nullptr;
    slot.task.reset();
    slot.taskLease.reset();
    slot.inUse = false;
    ++slot.generation;
    --instance->liveOperationCount;
}

void enqueueOperation(InstanceState* instance, uint32_t slotIndex)
{
    {
        std::lock_guard<std::mutex> lock(instance->operationMutex);
        instance->pendingOperations.push_back(slotIndex);
    }
    instance->operationCondition.notify_one();
}

} // namespace detail
} // namespace ovspatialquery

bool InstanceState::minimumFixedHostCharge(uint32_t operationCapacity, uint32_t deviceResultCapacity,
                                           uint32_t maxCpuConcurrency, uint32_t deviceBudgetCount,
                                           uint64_t& outCharge) noexcept
{
    uint64_t       total = static_cast<uint64_t>(sizeof(InstanceState)) + kInstanceControlAndDirectoryNodeAllowance;
    const uint64_t operations         = operationCapacity;
    const uint64_t deviceResults      = deviceResultCapacity;
    const uint64_t deviceBudgetsCount = deviceBudgetCount;
    const bool     valid =
        addHostCharge(operations, sizeof(OperationSlot), total) &&
        addHostCharge(operations, sizeof(std::vector<std::unique_ptr<OperationSlot>>::value_type), total) &&
        addHostCharge(operations, sizeof(std::vector<uint32_t>::value_type), total) &&
        // The three device-retirement vectors reserve one entry per device
        // result slot.
        addHostCharge(
            deviceResults,
            3u * sizeof(std::vector<
                        std::shared_ptr<ovspatialquery::detail::InstanceOperationDeviceRetirement>>::value_type),
            total) &&
        addHostCharge(maxCpuConcurrency, sizeof(std::vector<std::thread>::value_type), total) &&
        addHostCharge(maxCpuConcurrency, kWorkerLaunchStateAllowance, total) &&
        addHostCharge(deviceBudgetsCount, sizeof(std::vector<std::pair<DLDevice, uint64_t>>::value_type), total) &&
        addHostCharge(deviceBudgetsCount, sizeof(std::vector<std::array<uint64_t, 5u>>::value_type), total) &&
        addHostCharge(1u, ovspatialquery::detail::GeometryReferenceClaimDirectory::fixedResidentByteCount(), total) &&
        addHostCharge(1u, ovspatialquery::detail::GeometryIdentityRegistry::fixedResidentByteCount(), total) &&
        addHostCharge(1u, ovspatialquery::detail::ExplicitPreparationRegistry::fixedResidentByteCount(), total) &&
        addHostCharge(1u, ovspatialquery::detail::ExplicitRegistrationOperationOwner::fixedResidentByteCount(),
                      total) &&
        addHostCharge(1u, ovspatialquery::detail::PopulationTable::fixedResidentByteCount(), total);
    if (valid)
    {
        outCharge = total;
    }
    return valid;
}

bool InstanceState::retainedFixedHostCharge(uint64_t& outCharge) const noexcept
{
    // InstanceState is created with make_shared and published through one
    // instance-directory node. These conservative allowances cover the two
    // implementation-defined control/node allocations without relying on a
    // particular standard-library layout.
    uint64_t   total     = static_cast<uint64_t>(sizeof(InstanceState)) + kInstanceControlAndDirectoryNodeAllowance;
    const auto addVector = [&total](const auto& values)
    {
        using Vector = std::decay_t<decltype(values)>;
        return addHostCharge(static_cast<uint64_t>(values.capacity()),
                             static_cast<uint64_t>(sizeof(typename Vector::value_type)), total);
    };
    const bool valid =
        addHostCharge(static_cast<uint64_t>(operationSlots.size()), sizeof(OperationSlot), total) &&
        addVector(operationSlots) && addVector(pendingOperations) && addVector(retiredPrivateDeviceOperations) &&
        addVector(privateDeviceReaperOperations) && addVector(privateDeviceReaperCompleted) && addVector(workers) &&
        addVector(deviceBudgets) && addVector(deviceUsage) &&
        addHostCharge(static_cast<uint64_t>(workers.capacity()), kWorkerLaunchStateAllowance, total) &&
        addHostCharge(1u, geometryReferenceClaims->fixedResidentByteCount(), total) &&
        addHostCharge(1u, geometryIdentityRegistry.fixedResidentByteCount(), total) &&
        addHostCharge(1u, explicitPreparationRegistry->fixedResidentByteCount(), total) &&
        addHostCharge(1u, ovspatialquery::detail::ExplicitRegistrationOperationOwner::fixedResidentByteCount(),
                      total) &&
        addHostCharge(1u, populations.fixedResidentByteCount(), total);
    if (valid)
    {
        outCharge = total;
    }
    return valid;
}

void InstanceState::runCudaReaper()
{
    // While a retirement is outstanding, every notification of operationCondition
    // rings the signal, and a host function rings it when a retired result's
    // consumer has finished, so the reaper wakes on progress rather than on a
    // 1 ms poll, which Windows rounds up to a whole timer tick (15.6 ms by
    // default). The timeout only covers a ring that never comes. With nothing
    // retired it sleeps on cudaReaperCondition instead: woken by every host
    // enqueue and completion, it took operationMutex from the launching thread
    // each time, which on one CPU made a one-row CPU launch several times slower.
    ovspatialquery::detail::DeviceSignal& signal = *operationCondition.signal;
    for (;;)
    {
        const uint64_t seen = signal.count();
        {
            std::unique_lock<std::mutex> operationLock(operationMutex);
            if (stopping && liveDeviceResultCount.load() == 0u && retiredPrivateDeviceOperations.empty())
            {
                return;
            }
            if (!stopping && retiredPrivateDeviceOperations.empty())
            {
                cudaReaperCondition.wait(operationLock,
                                         [this]() { return stopping || !retiredPrivateDeviceOperations.empty(); });
                continue;
            }
        }
        if (reclaimDeviceRetirements(nullptr))
        {
            operationCondition.notify_all();
        }
        else
        {
            signal.waitPast(seen, std::chrono::milliseconds(1));
        }
    }
}

bool InstanceState::reclaimDeviceRetirements(const void* owner) noexcept
{
    using Retirement = ovspatialquery::detail::InstanceOperationDeviceRetirement;
    std::lock_guard<std::mutex> reclaimLock(retirementReclaimMutex);
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
    if (owner == nullptr && cudaReaperPaused.load())
    {
        std::lock_guard<std::mutex> operationLock(operationMutex);
        if (!stopping)
        {
            return false;
        }
    }
#endif
    {
        // Both lists are reserved to the retired list's capacity, so these
        // copies never allocate.
        std::lock_guard<std::mutex> operationLock(operationMutex);
        for (const std::shared_ptr<Retirement>& retirement : retiredPrivateDeviceOperations)
        {
            if (owner == nullptr || retirement->operationOwner == owner)
            {
                privateDeviceReaperOperations.push_back(retirement);
            }
        }
    }
    for (const std::shared_ptr<Retirement>& retirement : privateDeviceReaperOperations)
    {
        retirement->releaseReadyDependencies();
    }
    privateDeviceReaperOperations.clear();

    {
        std::lock_guard<std::mutex> operationLock(operationMutex);
        for (std::vector<std::shared_ptr<Retirement>>::iterator iterator = retiredPrivateDeviceOperations.begin();
             iterator != retiredPrivateDeviceOperations.end();)
        {
            if ((owner == nullptr || (*iterator)->operationOwner == owner) && (*iterator)->deviceSafe())
            {
                privateDeviceReaperCompleted.push_back(std::move(*iterator));
                iterator = retiredPrivateDeviceOperations.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
    }
    const bool reclaimed = !privateDeviceReaperCompleted.empty();
    for (const std::shared_ptr<Retirement>& retirement : privateDeviceReaperCompleted)
    {
        retirement->releaseReadyDependencies();
    }
    // The last references: each destructor frees its owner's result slot.
    privateDeviceReaperCompleted.clear();
    return reclaimed;
}

/* Runs the oldest posted shared work's tasks until none is left or an operation is
 * queued, and drops exhausted work from the list. Called with operationMutex held;
 * returns with it held. */
void InstanceState::helpSharedWork(std::unique_lock<std::mutex>& lock)
{
    SharedWork* work = pendingSharedWork.front();
    if (work->next.load() >= work->count)
    {
        pendingSharedWork.erase(pendingSharedWork.begin());
        return;
    }
    ++work->helping;
    for (;;)
    {
        /* A queued operation takes precedence: the caller and the other helpers finish. */
        if (canStartQueuedOperation())
        {
            break;
        }
        const size_t index = work->next.fetch_add(1u);
        if (index >= work->count)
        {
            break;
        }
        lock.unlock();
        work->task(work->state, index);
        lock.lock();
    }
    --work->helping;
    if (work->next.load() >= work->count)
    {
        pendingSharedWork.erase(std::remove(pendingSharedWork.begin(), pendingSharedWork.end(), work),
                                pendingSharedWork.end());
    }
    sharedWorkCondition.notify_all();
}

/* The caller runs the tasks itself and posts them for idle workers to share; it
 * returns once every task has run and no worker is still inside one. Posting is
 * best effort: without it, or when every worker is busy, the caller runs them all. */
void InstanceState::runSharedWork(size_t count, void (*task)(void* state, size_t index), void* taskState) noexcept
{
    SharedWork work;
    work.task   = task;
    work.state  = taskState;
    work.count  = count;
    bool posted = false;
    if (count > 1u && !workers.empty())
    {
        try
        {
            std::lock_guard<std::mutex> lock(operationMutex);
            pendingSharedWork.push_back(&work);
            posted = true;
        }
        catch (...)
        {
            posted = false;
        }
        if (posted)
        {
            operationCondition.notify_all();
        }
    }
    for (size_t index = work.next.fetch_add(1u); index < count; index = work.next.fetch_add(1u))
    {
        task(taskState, index);
    }
    if (posted)
    {
        std::unique_lock<std::mutex> lock(operationMutex);
        pendingSharedWork.erase(std::remove(pendingSharedWork.begin(), pendingSharedWork.end(), &work),
                                pendingSharedWork.end());
        sharedWorkCondition.wait(lock, [&work]() { return work.helping == 0u; });
    }
}

// A worker or a waiting caller may take a queued operation only while fewer
// than max_cpu_concurrency CPU jobs run. Called with operationMutex held.
bool InstanceState::canStartQueuedOperation() const noexcept
{
    return !pendingOperations.empty() && runningCpuJobs < maxCpuConcurrency;
}

void InstanceState::runWorker()
{
    // How many times in a row this worker found only deferred operations: it
    // yields, then waits for progress, so a slow producer costs no core.
    unsigned idlePolls = 0u;
    for (;;)
    {
        uint32_t slotIndex           = 0u;
        uint32_t operationGeneration = 0u;
        // Read before the operation runs, so progress made while it runs is
        // not waited for when it is deferred.
        uint64_t progressSeen = 0u;
        {
            std::unique_lock<std::mutex> lock(operationMutex);
            operationCondition.wait(lock,
                                    [this]()
                                    {
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
                                        return stopping || (!instanceOperationWorkersPaused &&
                                                            (canStartQueuedOperation() || !pendingSharedWork.empty()));
#else
                return stopping || canStartQueuedOperation() || !pendingSharedWork.empty();
#endif
                                    });
            if (stopping && pendingOperations.empty())
            {
                return;
            }
            if (!canStartQueuedOperation())
            {
                if (!pendingSharedWork.empty())
                {
                    helpSharedWork(lock);
                }
                continue;
            }
            slotIndex = pendingOperations.back();
            pendingOperations.pop_back();
            ++runningCpuJobs;
            ++workerJobs;
            operationSlots[slotIndex]->workerActive = true;
            operationGeneration                     = operationSlots[slotIndex]->generation;
            progressSeen                            = operationCondition.signal->count();
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
            ++runningInstanceOperationCount;
            peakRunningInstanceOperationCount =
                std::max(peakRunningInstanceOperationCount, runningInstanceOperationCount);
#endif
        }

#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        {
            std::unique_lock<std::mutex> lock(operationMutex);
            instanceOperationExecutionCondition.wait(lock, [this]()
                                                     { return stopping || !instanceOperationExecutionsPaused; });
        }
#endif

        const QueuedRun run = runQueuedOperation(slotIndex, operationGeneration);
        --workerJobs;
        if (run == QueuedRun::completed)
        {
            idlePolls = 0u;
            continue;
        }
        if (run == QueuedRun::deferredBehindWork)
        {
            idlePolls = 0u;
        }
        else if (++idlePolls < 64u)
        {
            std::this_thread::yield();
        }
        else
        {
            // Nothing else to run: wait for the progress this operation
            // needs, not a sleep, which on Windows lasts a whole timer tick
            // (15.6 ms by default). An enqueue or a completion (a reporter
            // advance waits for the one before it to publish) notifies
            // operationCondition, which rings the signal; a host function
            // rings it when a device publication is done.
            const std::shared_ptr<ovspatialquery::detail::OperationTask> task = operationSlots[slotIndex]->task;
            if (task != nullptr)
            {
                task->ringWhenReady(operationCondition.signal);
            }
            operationCondition.signal->waitPast(progressSeen, std::chrono::milliseconds(1));
        }
    }
}

// Runs a waiter's own operation on the waiting thread when it is still queued,
// its task permits it and a CPU job may start; the operation then needs no
// worker handoff. Returns false, leaving the operation queued, otherwise.
bool InstanceState::runQueuedOperationOnWaiter(uint32_t slotIndex, uint32_t generation) noexcept
{
    {
        std::lock_guard<std::mutex> operationLock(operationMutex);
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        if (instanceOperationWorkersPaused)
        {
            return false;
        }
#endif
        if (stopping || !canStartQueuedOperation() || slotIndex >= operationSlots.size())
        {
            return false;
        }
        const std::vector<uint32_t>::iterator queued =
            std::find(pendingOperations.begin(), pendingOperations.end(), slotIndex);
        if (queued == pendingOperations.end())
        {
            return false;
        }
        OperationSlot&              slot = *operationSlots[slotIndex];
        std::lock_guard<std::mutex> slotLock(slot.mutex);
        if (!slot.inUse || slot.generation != generation || slot.workerActive || slot.terminal ||
            slot.task == nullptr || !slot.task->waiterMayRun())
        {
            return false;
        }
        pendingOperations.erase(queued);
        ++runningCpuJobs;
        slot.workerActive = true;
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        ++runningInstanceOperationCount;
        peakRunningInstanceOperationCount = std::max(peakRunningInstanceOperationCount, runningInstanceOperationCount);
#endif
    }
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
    // A waiter that took the operation holds it like a worker while executions are paused.
    {
        std::unique_lock<std::mutex> lock(operationMutex);
        instanceOperationExecutionCondition.wait(lock,
                                                 [this]() { return stopping || !instanceOperationExecutionsPaused; });
    }
#endif
    if (runQueuedOperation(slotIndex, generation) != QueuedRun::completed)
    {
        // Requeued: wake a worker to poll it again.
        operationCondition.notify_one();
    }
    return true;
}

// Executes one dequeued operation and publishes its terminal state, or requeues
// it when the task is not ready. The caller counted it in runningCpuJobs.
InstanceState::QueuedRun InstanceState::runQueuedOperation(uint32_t slotIndex, uint32_t operationGeneration) noexcept
{
    OperationSlot&              slot     = *operationSlots[slotIndex];
    ovspatialquery_api_status_t status   = OVSPATIALQUERY_API_SUCCESS;
    bool                        deferred = false;
    // A queued operation holds its task: binding and activation require one,
    // and only abandonment before activation or destroy_op after the
    // terminal state removes it.
    const ovspatialquery::detail::InstanceOperationExecution execution = slot.task->execute();
    // A device publication, or a CUDA advance publishing behind the one
    // admitted before it (ADR-0022 decision 5), is requeued below.
    deferred = execution.status == OVSPATIALQUERY_API_NOT_READY;
    status   = execution.status;
    if (slot.kind != OperationKind::populationUpdate)
    {
        slot.privateFailure    = deferred ? OVSPATIALQUERY_API_SUCCESS : execution.status;
        slot.privateDiagnostic = deferred ? nullptr : execution.diagnostic;
        status                 = execution.status == OVSPATIALQUERY_API_SUCCESS || deferred
                                     ? OVSPATIALQUERY_API_SUCCESS
                                     : (execution.status == OVSPATIALQUERY_API_OUT_OF_MEMORY ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                                                                             : OVSPATIALQUERY_API_INTERNAL);
    }
    if (deferred)
    {
        bool alone = false;
        {
            std::lock_guard<std::mutex> operationLock(operationMutex);
            std::lock_guard<std::mutex> slotLock(slot.mutex);
            alone = pendingOperations.empty() && pendingSharedWork.empty();
            --runningCpuJobs;
            if (slot.inUse && slot.generation == operationGeneration && slot.workerActive)
            {
                // The oldest end of the queue: every operation enqueued
                // meanwhile runs before this one is polled again.
                slot.workerActive = false;
                pendingOperations.insert(pendingOperations.begin(), slotIndex);
            }
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
            if (runningInstanceOperationCount != 0u)
            {
                --runningInstanceOperationCount;
            }
#endif
        }
        return alone ? QueuedRun::deferredAlone : QueuedRun::deferredBehindWork;
    }

    std::vector<ovspatialquery::detail::SourceViewRequest>            completedViewRequests;
    std::vector<ovspatialquery::detail::PreparedSourceViewMembership> completedViewMemberships;
    // The completion lock makes target-reference cleanup and terminal
    // publication one externally atomic transition for population
    // mutation admission and population release.
    std::unique_lock<std::mutex> completionLock(populationCompletionMutex);
    {
        if (slot.kind == OperationKind::populationUpdate)
        {
            PopulationUpdateTask& update = static_cast<PopulationUpdateTask&>(*slot.task);
            if (handles.finishSourceViewUpdate(update.viewRequests) != OVSPATIALQUERY_API_SUCCESS)
            {
                status = OVSPATIALQUERY_API_INTERNAL;
            }
            populations.finishUpdate(update.population, update.ordinal);
            // A device publication that did not become the head releases the
            // chain built on it (ADR-0021); after a commit this is a no-op.
            if (update.endedCapture != nullptr)
            {
                update.endedCapture->publicationAbandoned();
                // Retire this worker's capture before terminal publication.
                // Slot-owned definition references remain until destroy_op;
                // the population may also still own those definitions.
                update.endedCapture.reset();
            }
            completedViewRequests    = std::move(update.viewRequests);
            completedViewMemberships = std::move(update.viewMemberships);
        }
    }
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
    if (slot.kind == OperationKind::familyBatchPlanPrivate || slot.kind == OperationKind::pairReporterPrivate)
    {
        std::unique_lock<std::mutex> testLock(instanceOperationTestMutex);
        instanceOperationBeforeTerminal = true;
        instanceOperationTestCondition.notify_all();
        instanceOperationTestCondition.wait(testLock, [this]() { return !pauseInstanceOperationBeforeTerminal; });
        instanceOperationBeforeTerminal = false;
    }
#endif
    if (slot.task != nullptr && !slot.task->beginTerminalPublication(status == OVSPATIALQUERY_API_SUCCESS))
    {
        status                 = OVSPATIALQUERY_API_INTERNAL;
        slot.privateFailure    = OVSPATIALQUERY_API_INTERNAL;
        slot.privateDiagnostic = "pair reporter completion publication failed internally";
    }
    bool retiredWorkerOwnership = false;
    bool queuedWork             = false;
    {
        std::lock_guard<std::mutex> operationLock(operationMutex);
        std::lock_guard<std::mutex> slotLock(slot.mutex);
        if (slot.inUse && slot.generation == operationGeneration && slot.workerActive)
        {
            slot.executionStatus   = status;
            slot.workerActive      = false;
            slot.terminal          = true;
            retiredWorkerOwnership = true;
        }
        --runningCpuJobs;
        queuedWork = !pendingOperations.empty();
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        if (runningInstanceOperationCount != 0u)
        {
            --runningInstanceOperationCount;
        }
#endif
        if (slot.task != nullptr)
        {
            slot.task->endTerminalPublication();
        }
    }
    completionLock.unlock();
    if (retiredWorkerOwnership)
        slot.condition.notify_all();
    // Admission and device retirement wake their own progress paths.
    // Completion need not wake idle workers when there is no queued work, and
    // one finished job lets at most one more start.
    if (queuedWork)
        operationCondition.notify_one();
    return QueuedRun::completed;
}

// Runs on a worker: materializes the update against its base, or observes the
// device outcome of one admitted at enqueue, and publishes the generation.
InstanceOperationExecution PopulationUpdateTask::execute() noexcept
{
    ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
    try
    {
        ovspatialquery::detail::GenerationLease                              staged;
        const std::shared_ptr<ovspatialquery::detail::CudaPopulationCapture> cudaCapture = input.cudaCapture();
        if (admittedGeneration.initialized())
        {
            // Built and admitted at enqueue (ADR-0021): the worker observes
            // the device outcome and publishes the same generation.
            ovspatialquery::detail::ObjectTable                            objects;
            std::shared_ptr<ovspatialquery::detail::CudaPopulationVersion> cudaVersion;
            status = cudaCapture != nullptr ? cudaCapture->execute(input.targetOrdinal(), source, objects, cudaVersion)
                                            : OVSPATIALQUERY_API_INTERNAL;
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                staged = admittedGeneration;
            }
        }
        else
        {
            ovspatialquery::detail::ObjectTable objects;
            std::optional<uint64_t>             measuredGeometryCharge;
            status = input.materialize(source, objects, measuredGeometryCharge);
            // An explicit update knows what it changed against its base;
            // any other input is compared with the base below.
            std::optional<ovspatialquery::detail::VersionChangeRecord> changes = input.releaseChanges();
            std::shared_ptr<const std::vector<uint64_t>>               usedObjectValues;
            if (status == OVSPATIALQUERY_API_SUCCESS && input.enforcesObjectValueNoReuse())
            {
                status = prepareUsedObjectValues(source, objects, changes, usedObjectValues);
            }
            std::vector<ovspatialquery::detail::GenerationViewMembership>  generationViewMemberships;
            std::shared_ptr<ovspatialquery::detail::CudaPopulationVersion> cudaVersion;
            if (status == OVSPATIALQUERY_API_SUCCESS && cudaCapture != nullptr)
            {
                status              = cudaCapture->execute(input.targetOrdinal(), source, objects, cudaVersion);
                const bool executed = status == OVSPATIALQUERY_API_SUCCESS;
                // A capture may publish no device version of its own.
                while (status == OVSPATIALQUERY_API_SUCCESS && cudaVersion != nullptr)
                {
                    const ovspatialquery::detail::CudaCompletionState state = cudaVersion->resultState();
                    if (state == ovspatialquery::detail::CudaCompletionState::pending)
                    {
                        std::this_thread::yield();
                        continue;
                    }
                    if (state != ovspatialquery::detail::CudaCompletionState::complete)
                    {
                        status = OVSPATIALQUERY_API_INTERNAL;
                    }
                    cudaVersion->releaseReadyDependencies();
                    break;
                }
                if (executed && cudaCapture->materializesHostObjects())
                {
                    stagingHostLease.reset();
                    stagingDeviceLease.reset();
                }
            }
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                generationViewMemberships.reserve(viewMemberships.size());
                for (const ovspatialquery::detail::PreparedSourceViewMembership& membership : viewMemberships)
                {
                    generationViewMemberships.push_back({membership.view, membership.selectedIds});
                }
                if (!changes || !generationViewMemberships.empty() ||
                    (!changes->everything && (!source.initialized() || changes->baseOrdinal != source.ordinal())))
                {
                    changes.emplace();
                    status = ovspatialquery::detail::PopulationTable::describeChanges(
                        source, objects, generationViewMemberships, *changes);
                }
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
                // Test builds check a known record against the comparison.
                else if (!changes->everything)
                {
                    ovspatialquery::detail::VersionChangeRecord compared;
                    if (ovspatialquery::detail::PopulationTable::describeChanges(
                            source, objects, generationViewMemberships, compared) != OVSPATIALQUERY_API_SUCCESS ||
                        compared.everything || compared.baseOrdinal != changes->baseOrdinal ||
                        compared.positions != changes->positions)
                    {
                        std::fprintf(stderr, "a publication's known changes differ from its comparison\n");
                        std::fflush(stderr);
                        std::_Exit(EXIT_FAILURE);
                    }
                }
#endif
                if (status == OVSPATIALQUERY_API_SUCCESS)
                {
                    status =
                        buildChargedGeneration(instance, *this, input.targetOrdinal(), std::move(objects),
                                               std::move(generationViewMemberships), std::move(usedObjectValues),
                                               std::move(*changes), measuredGeometryCharge, std::move(cudaVersion),
                                               cudaCapture != nullptr && cudaCapture->requiresCpuQueryBackend(), source,
                                               cudaCapture == nullptr ? &retainedInputLease : nullptr, staged);
                }
            }
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && cudaCapture != nullptr)
        {
            // Admission counts describe capacity; completed rows determine the result.
            hasCounts = cudaCapture->objectCounts(indexedCount, excludedCount);
            if (cudaCapture->exclusions(exclusions.objects, exclusions.reasons))
            {
                excludedCount = exclusions.objects.size();
            }
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            if (exclusions.objects.size() != exclusions.reasons.size() ||
                exclusions.objects.size() > static_cast<uint64_t>(INT64_MAX))
            {
                status = OVSPATIALQUERY_API_INTERNAL;
            }
            else
            {
                sortSourceExclusions(exclusions);
                exclusionShape           = static_cast<int64_t>(exclusions.objects.size());
                result.excluded_objects  = {exclusions.objects.data(), {kDLCPU, 0}, 1, {kDLUInt, 64, 1},
                                            &exclusionShape,           nullptr,     0u};
                result.exclusion_reasons = {exclusions.reasons.data(), {kDLCPU, 0}, 1, {kDLInt, 32, 1},
                                            &exclusionShape,           nullptr,     0u};
            }
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
            ovspatialquery::detail::reachPopulationBeforePublish(instance.runtimeLease.runtime, ordinal);
#endif
            {
                std::lock_guard<std::mutex> publicationLock(instance.populationCompletionMutex);
                status = instance.handles.validateSourceViewUpdate(viewRequests, viewMemberships);
                if (status == OVSPATIALQUERY_API_SUCCESS)
                {
                    status = instance.populations.publishUpdate(population, staged);
                }
                if (status == OVSPATIALQUERY_API_SUCCESS)
                {
                    if (cudaCapture != nullptr)
                    {
                        cudaCapture->publicationCommitted();
                    }
                    instance.handles.publishSourceViewUpdate(viewRequests, viewMemberships);
                }
            }
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                ovspatialquery::detail::reachPopulationAfterPublish(instance.runtimeLease.runtime, ordinal);
            }
#endif
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                result.published_ordinal = ordinal;
                if (hasCounts)
                {
                    result.object_count = indexedCount;
                }
                else
                {
                    result.object_count = staged.objectCount();
                }
                result.excluded_object_count = exclusions.objects.size();
            }
        }
    }
    catch (const std::bad_alloc&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        status = OVSPATIALQUERY_API_INTERNAL;
    }
    // The device has not finished it: the worker requeues it behind the
    // other pending work, with nothing consumed.
    if (status == OVSPATIALQUERY_API_NOT_READY)
    {
        return {status, nullptr};
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        // A rejected result keeps no uncharged output allocation in the operation.
        exclusions = {};
        exclusionLease.reset();
    }
    // Kept past the input's release so a publication that did not become the
    // head can settle the outcome the next device publication waits on.
    endedCapture = input.cudaCapture();
    input.reset();
    source.reset();
    admittedGeneration.reset();
    retainedInputLease.reset();
    publishedDeviceLease.reset();
    stagingHostLease.reset();
    stagingDeviceLease.reset();
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    ovspatialquery::detail::reachOperationBeforeCompletion(instance.runtimeLease.runtime, ordinal);
#endif
    return {status, nullptr};
}

void PopulationUpdateTask::ringWhenReady(const std::shared_ptr<DeviceSignal>& signal) noexcept
{
    if (const std::shared_ptr<CudaPopulationCapture> capture = input.cudaCapture())
    {
        capture->ringWhenReady(signal);
    }
}

namespace ovspatialquery
{
namespace detail
{

// Builds the generation of a device-prepared version at admission and makes
// it selectable while its publication is pending (ADR-0021). The version has
// no host objects; its device charge moves into the generation's lease.
ovspatialquery_api_status_t stageAdmittedGeneration(
    InstanceState* instance, PopulationUpdateTask& update, ovspatialquery::detail::PopulationKey population,
    const ovspatialquery::detail::GenerationLease& base, ovspatialquery_ordinal_t targetOrdinal,
    std::shared_ptr<ovspatialquery::detail::CudaPopulationVersion> version)
{
    ovspatialquery::detail::ObjectTable                        objects;
    std::shared_ptr<const std::vector<uint64_t>>               usedObjectValues;
    std::optional<ovspatialquery::detail::VersionChangeRecord> knownChanges;
    ovspatialquery_api_status_t status = prepareUsedObjectValues(base, objects, knownChanges, usedObjectValues);
    // A device-prepared version has no host objects to compare, and the host
    // never learns which slots a pose-rate update moved, so its record says
    // that everything changed: exact or over-inclusive, never missing
    // (ADR-0019 decision 3). A device-written changed-slot list may refine it.
    ovspatialquery::detail::VersionChangeRecord changes;
    changes.everything  = true;
    changes.baseOrdinal = base.initialized() ? base.ordinal() : -1;
    ovspatialquery::detail::GenerationLease staged;
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = buildChargedGeneration(*instance, update, targetOrdinal, std::move(objects), {},
                                        std::move(usedObjectValues), std::move(changes), {}, std::move(version), false,
                                        base, nullptr, staged);
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = instance->populations.admitPending(population, staged);
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        update.admittedGeneration = std::move(staged);
    }
    return status;
}

bool adoptDroppedTriangles(InstanceState& instance, PopulationUpdateTask& update,
                           ovspatialquery::detail::SourceDroppedTriangles&& droppedTriangles)
{
    uint64_t bytes = 0u;
    if (droppedTriangles.objects.empty())
    {
        return true;
    }
    if (!addHostCharge(droppedTriangles.objects.capacity(), sizeof(uint64_t), bytes) ||
        !addHostCharge(droppedTriangles.counts.capacity(), sizeof(uint64_t), bytes) ||
        !reserveHostLease(instance, bytes, update.droppedTriangleLease))
    {
        return false;
    }
    update.droppedTriangles = std::move(droppedTriangles);
    return true;
}

ovspatialquery_api_status_t submitPopulationInput(InstanceState*                                  instance,
                                                  ovspatialquery::detail::PopulationKey           population,
                                                  ovspatialquery::detail::PreparedPopulationInput input,
                                                  ovspatialquery_op_id_t&                         outOperation,
                                                  ovspatialquery::detail::SourceDroppedTriangles  droppedTriangles)
{
    ovspatialquery_api_status_t status = instance->populations.validate(population);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    std::shared_ptr<PopulationUpdateTask> update;
    try
    {
        update = std::make_shared<PopulationUpdateTask>(*instance);
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    ovspatialquery::detail::GenerationLease source;
    const ovspatialquery_ordinal_t          baseOrdinal = input.baseOrdinal();
    const ovspatialquery_ordinal_t*         exactBase   = input.isSet() ? nullptr : &baseOrdinal;
    status                                              = OVSPATIALQUERY_API_SUCCESS;
    {
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        status =
            instance->populations.beginUpdate(population, input.targetOrdinal(), exactBase, source,
                                              input.requiresUnreadyPopulation(), input.chainsPendingPublications());
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    uint32_t slotIndex = 0u;
    // Ends a rejected submission; a reserved operation frees its slot and id.
    const auto reject = [&](ovspatialquery_api_status_t rejected, bool reserved)
    {
        update.reset();
        if (reserved)
        {
            abandonReservedOperation(instance, slotIndex);
        }
        source.reset();
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        instance->populations.finishUpdate(population);
        if (reserved)
        {
            outOperation = OVSPATIALQUERY_INVALID_OP;
        }
        return rejected;
    };
    const std::shared_ptr<ovspatialquery::detail::CudaPopulationCapture> preparedCudaCapture = input.cudaCapture();
    if (preparedCudaCapture != nullptr)
    {
        status = preparedCudaCapture->prepareAdmission(source);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return reject(status, false);
        }
    }
    status = reserveOperation(instance, OperationKind::populationUpdate, slotIndex, outOperation);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return reject(status, false);
    }
    if (!reserveHostLease(*instance, publicationAdmissionCharge(input, source, 0u), update->retainedInputLease))
    {
        return reject(OVSPATIALQUERY_API_HOST_BUDGET, true);
    }
    OperationSlot&                                                       slot = *instance->operationSlots[slotIndex];
    const std::shared_ptr<ovspatialquery::detail::CudaPopulationCapture> cudaCapture = preparedCudaCapture;
    if (cudaCapture != nullptr)
    {
        if (!reserveHostLease(*instance, cudaCapture->hostStagingByteCharge(), update->stagingHostLease))
        {
            return reject(OVSPATIALQUERY_API_HOST_BUDGET, true);
        }
        status = reserveInstanceDeviceMemory(instance->publicHandle, cudaCapture->device(),
                                             cudaCapture->deviceStagingByteCharge(), update->stagingDeviceLease);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = reserveInstanceDeviceMemory(instance->publicHandle, cudaCapture->device(),
                                                 cudaCapture->deviceByteCharge(), update->publishedDeviceLease);
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return reject(status, true);
        }
        const uint64_t outputCapacity = cudaCapture->exclusionCapacity();
        uint64_t       outputBytes    = 0;
        if (!addHostCharge(outputCapacity, sizeof(uint64_t) + sizeof(int32_t), outputBytes) ||
            !reserveHostLease(*instance, outputBytes, update->exclusionLease))
            return reject(OVSPATIALQUERY_API_HOST_BUDGET, true);
        try
        {
            update->exclusions.objects.reserve(static_cast<size_t>(outputCapacity));
            update->exclusions.reasons.reserve(static_cast<size_t>(outputCapacity));
        }
        catch (const std::bad_alloc&)
        {
            return reject(OVSPATIALQUERY_API_OUT_OF_MEMORY, true);
        }
        status = cudaCapture->captureAdmissionInputs();
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return reject(status, true);
        }
        slot.updateDefinitionOwners = cudaCapture->releaseDefinitionOwners();
        update->hasCounts           = cudaCapture->objectCounts(update->indexedCount, update->excludedCount);
        if (std::shared_ptr<ovspatialquery::detail::CudaPopulationVersion> admitted = cudaCapture->admittedVersion())
        {
            status = stageAdmittedGeneration(instance, *update, population, source, input.targetOrdinal(),
                                             std::move(admitted));
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                cudaCapture->publicationAbandoned();
                return reject(status, true);
            }
        }
    }
    if (!adoptDroppedTriangles(*instance, *update, std::move(droppedTriangles)))
    {
        return reject(OVSPATIALQUERY_API_HOST_BUDGET, true);
    }
    update->population = population;
    update->ordinal    = input.targetOrdinal();
    update->input      = std::move(input);
    update->source     = std::move(source);
    slot.task          = std::move(update);
    enqueueOperation(instance, slotIndex);
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace detail
} // namespace ovspatialquery

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t createInstance(const SourceInstanceConfig&       config,
                                           ovspatialquery_instance_handle_t* outInstance)
{
    if (outInstance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    *outInstance = OVSPATIALQUERY_INVALID_INSTANCE;
    if (config.operationCapacity == 0u || config.maxCpuConcurrency == 0u || config.hostByteBudget == 0u ||
        ((config.deviceBudgetCount == 0u) != (config.deviceBudgets == nullptr)))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0u; index < config.deviceBudgetCount; ++index)
    {
        const SourceDeviceBudget& budget = config.deviceBudgets[index];
        if (budget.byteBudget == 0u || budget.device.device_type != kDLCUDA || budget.device.device_id < 0)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        for (uint32_t previous = 0u; previous < index; ++previous)
        {
            const SourceDeviceBudget& prior = config.deviceBudgets[previous];
            if (prior.device.device_type == budget.device.device_type &&
                prior.device.device_id == budget.device.device_id)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
        }
    }
    uint64_t minimumFixedHostCharge = 0u;
    if (!InstanceState::minimumFixedHostCharge(config.operationCapacity, config.deviceResultCapacity,
                                               config.maxCpuConcurrency, config.deviceBudgetCount,
                                               minimumFixedHostCharge) ||
        minimumFixedHostCharge > config.hostByteBudget)
    {
        return OVSPATIALQUERY_API_HOST_BUDGET;
    }

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    ovspatialquery_api_status_t       status  = acquireDefaultBackendRuntime(&runtime);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    bool runtimeTransferred = false;
    try
    {
        std::shared_ptr<InstanceState> instance = std::make_shared<InstanceState>(
            runtime, config.operationCapacity, config.deviceResultCapacity, config.maxCpuConcurrency,
            config.hostByteBudget, config.deviceBudgets, config.deviceBudgetCount);
        instance->runtimeLease.releaseWithInstance = true;
        runtimeTransferred                         = true;
        status                                     = instanceDirectory().publish(instance, *outInstance);
        return status;
    }
    catch (const std::bad_alloc&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const InstanceHostBudgetExceeded&)
    {
        status = OVSPATIALQUERY_API_HOST_BUDGET;
    }
    catch (...)
    {
        status = OVSPATIALQUERY_API_INTERNAL;
    }
    if (!runtimeTransferred)
    {
        releaseDefaultBackendRuntime(runtime);
    }
    return status;
}

ovspatialquery_enqueue_result_t sourcePublicEnqueueResult(ovspatialquery_api_status_t status,
                                                          ovspatialquery_op_id_t operation, const char* message)
{
    // A refused column layout names its column unless the caller already has a
    // more precise message (ADR-0053).
    const ovspatialquery_result_t result =
        apiResult(status, message != nullptr ? message : inputLayoutDiagnostic(status));
    clearInputLayoutDiagnostic();
    // A successful source call that dropped triangles without area names them (ADR-0073).
    if (status == OVSPATIALQUERY_API_SUCCESS && message != nullptr)
    {
        try
        {
            gLastApiError = message;
        }
        catch (...)
        {
            gLastApiError.clear();
        }
    }
    return ovspatialquery_enqueue_result_t{
        result.status, status == OVSPATIALQUERY_API_SUCCESS ? operation : OVSPATIALQUERY_INVALID_OP};
}

} // namespace detail
} // namespace ovspatialquery

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_destroy_instance(ovspatialquery_instance_handle_t instance_handle)
{
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE, "instance handle is invalid");
    }
    return apiResult(ovspatialquery_release_instance(instance_handle));
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_get_version(uint32_t* out_major,
                                                                                 uint32_t* out_minor,
                                                                                 uint32_t* out_patch)
{
    if (out_major == nullptr || out_minor == nullptr || out_patch == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT, "version outputs must not be null");
    }
    *out_major = OVSPATIALQUERY_VERSION_MAJOR;
    *out_minor = OVSPATIALQUERY_VERSION_MINOR;
    *out_patch = OVSPATIALQUERY_VERSION_PATCH;
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_string_t ovspatialquery_get_last_error(void)
{
    return ovspatialquery_string_t{gLastApiError.c_str(), gLastApiError.size()};
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_instance(
    const ovspatialquery_instance_desc_t* desc, ovspatialquery_instance_handle_t* outInstance)
{
    if (outInstance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    *outInstance = OVSPATIALQUERY_INVALID_INSTANCE;
    if (desc == nullptr || desc->operation_capacity == 0u || desc->max_cpu_concurrency == 0u ||
        desc->host_byte_budget == 0u || ((desc->device_budget_count == 0u) != (desc->device_budgets == nullptr)))
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    try
    {
        std::vector<ovspatialquery::detail::SourceDeviceBudget> budgets;
        budgets.reserve(desc->device_budget_count);
        for (uint32_t index = 0u; index < desc->device_budget_count; ++index)
        {
            budgets.push_back({desc->device_budgets[index].device, desc->device_budgets[index].byte_budget});
        }
        const ovspatialquery::detail::SourceInstanceConfig config{desc->operation_capacity,
                                                                  desc->operation_capacity,
                                                                  desc->max_cpu_concurrency,
                                                                  desc->host_byte_budget,
                                                                  budgets.empty() ? nullptr : budgets.data(),
                                                                  static_cast<uint32_t>(budgets.size())};
        return apiResult(ovspatialquery::detail::createInstance(config, outInstance));
    }
    catch (const std::bad_alloc&)
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    catch (...)
    {
        return apiResult(OVSPATIALQUERY_API_INTERNAL);
    }
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_get_instance_memory_usage(
    ovspatialquery_instance_handle_t instanceHandle, DLDevice device, uint64_t* outResidentBytes)
{
    if (outResidentBytes == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    if (device.device_type == kDLCPU && device.device_id == 0)
    {
        *outResidentBytes = instance->hostLedger.total();
        return apiResult(OVSPATIALQUERY_API_SUCCESS);
    }
    std::lock_guard<std::mutex> lock(instance->budgetMutex);
    size_t                      found = instance->deviceBudgets.size();
    for (size_t index = 0u; index < instance->deviceBudgets.size(); ++index)
    {
        if (instance->deviceBudgets[index].first.device_type == device.device_type &&
            instance->deviceBudgets[index].first.device_id == device.device_id)
        {
            found = index;
            break;
        }
    }
    if (found == instance->deviceBudgets.size())
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED);
    }
    *outResidentBytes = instance->deviceUsage[found];
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_query_capability(
    ovspatialquery_instance_handle_t instanceHandle, DLDevice device, ovspatialquery_capability_t capability)
{
    const bool cuda = device.device_type == kDLCUDA;
    if (cuda ? device.device_id < 0 : (device.device_type != kDLCPU || device.device_id != 0))
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    // Compared as an integer: a newer header may pass a key this library does not know.
    const uint32_t key = static_cast<uint32_t>(capability);
    if (key > static_cast<uint32_t>(OVSPATIALQUERY_CAPABILITY_PAIR_REPORTER))
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED, "this ovspatialquery library does not know the capability");
    }
    if (!cuda)
    {
        // Every plan family, precision and pair reporter is mandatory on the CPU.
        return key == static_cast<uint32_t>(OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE)
                   ? apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED,
                               "PREPARE_ON_DEVICE needs a CUDA device; the CPU prepares versions on the host")
                   : apiResult(OVSPATIALQUERY_API_SUCCESS);
    }
    // This implementation's CUDA backend serves every key once it can run on the device.
    if (const char* unavailable =
            cudaBackendUnavailableReason(device.device_id, instance->cudaModulePins(device.device_id)))
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED, unavailable);
    }
    std::lock_guard<std::mutex> lock(instance->budgetMutex);
    for (const std::pair<DLDevice, uint64_t>& budget : instance->deviceBudgets)
    {
        if (budget.first.device_type == device.device_type && budget.first.device_id == device.device_id)
        {
            return apiResult(OVSPATIALQUERY_API_SUCCESS);
        }
    }
    return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED,
                     "the instance has no device budget for the CUDA device, so it cannot use that device");
}
