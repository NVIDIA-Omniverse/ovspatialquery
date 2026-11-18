/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_cuda_pair_reporter.h"

#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_instance_operation_runtime.h"

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
#include "ovspatialquery_cuda_driver.h"
#include "ovspatialquery_cuda_family_plan_internal.h"
#include "ovspatialquery_cuda_pair_types.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

namespace ovspatialquery
{
namespace detail
{

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
namespace
{

namespace cudaPairs = cuda_pairs;

#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
std::array<std::atomic<uint64_t>, 5u> gCopyCounts{};
#endif

constexpr uint64_t kAlignment        = 256u;
constexpr uint32_t kMaximumBlocks    = 1024u;
constexpr size_t   kFunctionCount    = 11u;
constexpr uint32_t kMaximumHashSlots = 0x80000000u;

enum Function : size_t
{
    fnMark,
    fnGridStats,
    fnGridLink,
    fnCarry,
    fnDiscover,
    fnClear,
    fnIndex,
    fnDifference,
    fnFinish,
    fnScatterRows,
    fnScatterFlags,
};

constexpr std::array<const char*, kFunctionCount> kFunctionNames{
    "ovspatialquery_cuda_pairs_mark",          "ovspatialquery_cuda_pairs_grid_stats",
    "ovspatialquery_cuda_pairs_grid_link",     "ovspatialquery_cuda_pairs_carry",
    "ovspatialquery_cuda_pairs_discover",      "ovspatialquery_cuda_pairs_clear",
    "ovspatialquery_cuda_pairs_index",         "ovspatialquery_cuda_pairs_difference",
    "ovspatialquery_cuda_pairs_finish",        "ovspatialquery_cuda_pairs_scatter_rows",
    "ovspatialquery_cuda_pairs_scatter_flags",
};

bool addBytes(uint64_t count, uint64_t size, uint64_t& total) noexcept
{
    if (count != 0u && size > std::numeric_limits<uint64_t>::max() / count)
    {
        return false;
    }
    const uint64_t bytes = count * size;
    if (bytes > std::numeric_limits<uint64_t>::max() - total - kAlignment)
    {
        return false;
    }
    total += (bytes + kAlignment - 1u) & ~(kAlignment - 1u);
    return true;
}

// Slots of an identity index for `capacity` pairs, at most half of
// kMaximumHashSlots (residentDeviceBytes): a power of two at least twice the
// capacity, so probing stays short and always finds an empty slot.
uint32_t hashSlots(uint64_t capacity) noexcept
{
    uint64_t slots = 2u;
    while (slots < 2u * capacity)
    {
        slots <<= 1u;
    }
    return static_cast<uint32_t>(slots);
}

uint32_t bucketSlots(uint32_t objectCount) noexcept
{
    uint32_t slots = 64u;
    while (slots < 2u * static_cast<uint64_t>(objectCount) && slots < kMaximumHashSlots)
    {
        slots <<= 1u;
    }
    return slots;
}

template <typename Value>
Value* devicePointer(CuDevicePtr pointer) noexcept
{
    return reinterpret_cast<Value*>(static_cast<uintptr_t>(pointer));
}

// The driver, the device's primary context and the kernels one reporter and
// the device copies it makes use; shared so a copy outlives its reporter. The
// kernels come from the device's shared module.
struct DeviceContext
{
    Driver                                 driver;
    CuDevice                               device  = 0;
    CuContext                              context = nullptr;
    std::shared_ptr<CudaContextModule>     module;
    bool                                   retained = false;
    std::array<CuFunction, kFunctionCount> functions{};
    std::atomic<bool>                      lost{false};

    ~DeviceContext()
    {
        if (retained && !lost.load(std::memory_order_relaxed))
        {
            (void)driver.primaryRelease(device);
        }
    }

    ovspatialquery_api_status_t open(int32_t id) noexcept
    {
        if (!driver.open())
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        CuContext restore = nullptr;
        if (driver.deviceGet(&device, id) != kCuSuccess || driver.contextGetCurrent(&restore) != kCuSuccess ||
            driver.primaryRetain(&context, device) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        retained                                 = true;
        const ovspatialquery_api_status_t loaded = acquireCudaContextModule(id, module);
        if (loaded != OVSPATIALQUERY_API_SUCCESS)
        {
            return loaded;
        }
        if (driver.contextSetCurrent(context) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        CuResult status = kCuSuccess;
        for (size_t index = 0u; status == kCuSuccess && index < kFunctionCount; ++index)
        {
            status = driver.moduleGetFunction(&functions[index], module->module, kFunctionNames[index]);
        }
        (void)driver.contextSetCurrent(restore);
        return status == kCuSuccess ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_NOT_SUPPORTED;
    }

    CuResult observe(CuResult status) noexcept
    {
        if (status == kCuDeviceLost)
        {
            lost.store(true, std::memory_order_relaxed);
        }
        return status;
    }

    CuResult launch(Function function, uint64_t count, void** parameters, CuStream stream, bool gridStride = true)
    {
        // Grid-stride kernels take at most kMaximumBlocks blocks, the others one thread per item.
        const uint64_t     blocks = std::max<uint64_t>(1u, (count + cudaPairs::kThreads - 1u) / cudaPairs::kThreads);
        const unsigned int grid =
            static_cast<unsigned int>(std::min<uint64_t>(blocks, gridStride ? kMaximumBlocks : 0x7fffffffu));
        return observe(driver.launchKernel(functions[function], grid, 1u, 1u, cudaPairs::kThreads, 1u, 1u, 0u, stream,
                                           parameters, nullptr));
    }
};

// Makes the device's context current for one scope and restores the caller's.
struct CurrentContext : CudaCurrentContext
{
    explicit CurrentContext(DeviceContext& device) noexcept
        : CudaCurrentContext(device.driver, device.context, device.lost)
    {}
};

// The view a membership belongs to, compared by owner.
using SelectionKey = std::weak_ptr<const void>;

bool sameSelection(const SelectionKey& first, const SelectionKey& second) noexcept
{
    return !first.owner_before(second) && !second.owner_before(first);
}

bool hasSelection(const SelectionKey& key) noexcept { return !sameSelection(key, SelectionKey{}); }

// One view's membership on a device copy: a flag per object.
struct Membership
{
    SelectionKey key;
    CuDevicePtr  flags = 0u;
};

// The device copy of one version (ADR-0019 decisions 6 and 7): its objects'
// identities, participation bits and bounds by slot, the membership of the
// views CUDA reporters target, and its change record. The arrays are borrowed
// from a complete set published through CUDA on the same device, otherwise
// owned. The copy's ready event follows the work that wrote it.
class VersionTable
{
  public:
    explicit VersionTable(std::shared_ptr<DeviceContext> context) noexcept : device(std::move(context)) {}

    ~VersionTable()
    {
        if (device == nullptr || device->lost.load(std::memory_order_relaxed))
        {
            return;
        }
        CurrentContext scope(*device);
        if (scope.status() != kCuSuccess)
        {
            return;
        }
        // Readers hold the copy until their work completes; this waits only for
        // the copy's own writes, when it is dropped right after them.
        if (ready != nullptr)
        {
            (void)device->observe(device->driver.eventSynchronize(ready));
            (void)device->driver.eventDestroy(ready);
        }
        for (const Membership& membership : memberships)
        {
            (void)device->driver.memFree(membership.flags);
        }
        if (objects != 0u)
        {
            (void)device->driver.memFree(objects);
        }
        if (recordPositions != 0u)
        {
            (void)device->driver.memFree(recordPositions);
        }
    }

    VersionTable(const VersionTable&)            = delete;
    VersionTable& operator=(const VersionTable&) = delete;

    CuDevicePtr membership(const SelectionKey& key) const noexcept
    {
        for (const Membership& entry : memberships)
        {
            if (sameSelection(entry.key, key))
            {
                return entry.flags;
            }
        }
        return 0u;
    }

    std::shared_ptr<DeviceContext>               device;
    ovspatialquery_ordinal_t                     ordinal     = 0;
    uint32_t                                     objectCount = 0u;
    uint32_t                                     capacity    = 0u;
    std::shared_ptr<const CudaPopulationVersion> borrowed;
    CuDevicePtr                                  objects  = 0u; // one owned allocation for the four arrays
    CuDevicePtr                                  ids      = 0u;
    CuDevicePtr                                  bits     = 0u;
    CuDevicePtr                                  minimums = 0u;
    CuDevicePtr                                  maximums = 0u;
    std::vector<Membership>                      memberships;
    CuDevicePtr                                  recordPositions = 0u;
    uint32_t                                     recordCount     = 0u;
    CuEvent                                      ready           = nullptr;
    CuEvent                                      borrowedReady   = nullptr;
    // A borrowed version's preparation status word (ADR-0021), or zero.
    CuDevicePtr borrowedFailed = 0u;
    // The copy's charge as published data on the device, in the leases that hold it.
    std::vector<std::shared_ptr<void>> budgets;
    uint64_t                           charged = 0u;
};

std::shared_ptr<VersionTable> attachedTable(GenerationDeviceAttachments& attachments, int32_t deviceId)
{
    for (const std::pair<int32_t, std::shared_ptr<void>>& entry : attachments.entries)
    {
        if (entry.first == deviceId)
        {
            return std::static_pointer_cast<VersionTable>(entry.second);
        }
    }
    return {};
}

// Frees the copy when no admitted advance holds it.
void detachTable(GenerationDeviceAttachments& attachments, int32_t deviceId) noexcept
{
    for (size_t index = 0u; index < attachments.entries.size(); ++index)
    {
        if (attachments.entries[index].first == deviceId)
        {
            attachments.entries.erase(attachments.entries.begin() + static_cast<std::ptrdiff_t>(index));
            return;
        }
    }
}

// The changed positions of the records that lead from a selected version back
// to the reporter's baseline: host positions from host records, and for a
// device-prepared version the slot list its prepare kernel wrote, read on the
// device while the version is held.
struct ChangeChain
{
    std::vector<uint32_t>                                     host;
    std::vector<std::pair<CuDevicePtr, uint32_t>>             device;
    std::vector<std::shared_ptr<const CudaPopulationVersion>> held;
    size_t                                                    records = 0u;

    size_t total() const noexcept
    {
        size_t count = host.size();
        for (const std::pair<CuDevicePtr, uint32_t>& segment : device)
        {
            count += segment.second;
        }
        return count;
    }
    bool empty() const noexcept { return total() == 0u; }
};

// Positions of the records that lead from `version` back to `baseOrdinal`,
// or false when a record says everything without a device list, a version in
// between is gone, or together they exceed `limit` positions; a padded device
// list is held to the object count instead, and leaves no room after it.
bool chainPositions(const PopulationTable& populations, PopulationKey population, GenerationLease version,
                    ovspatialquery_ordinal_t baseOrdinal, size_t limit, ChangeChain& out)
{
    const size_t objectCount = version.objects().size();
    out                      = ChangeChain{};
    while (version.ordinal() != baseOrdinal)
    {
        const VersionChangeRecord& record = version.changeRecord();
        if (record.baseOrdinal < baseOrdinal)
        {
            return false;
        }
        const size_t room = limit - std::min(limit, out.total());
        if (record.everything)
        {
            // A device-prepared POSES version lists the slots it rewrote on the device.
            const std::shared_ptr<const CudaPopulationVersion> cuda = version.cudaVersion();
            const CudaPopulationDescription                    description =
                cuda != nullptr ? cuda->description() : CudaPopulationDescription{};
            // A padded list's length bounds its moved slots: the unmoved hold ~0u,
            // which the mark kernel skips, so only the device knows how many moved.
            // It is admitted while the whole chain stays within the object count, so
            // the workspace stays bounded by one snapshot whatever the chain's length.
            const size_t padRoom = description.objectCount - std::min<size_t>(description.objectCount, out.total());
            if (description.changedSlots == nullptr ||
                description.changedSlotCount > (description.changedSlotsPadded ? padRoom : room))
            {
                return false;
            }
            out.device.emplace_back(static_cast<CuDevicePtr>(reinterpret_cast<uintptr_t>(description.changedSlots)),
                                    static_cast<uint32_t>(description.changedSlotCount));
            out.held.push_back(cuda);
        }
        else
        {
            if (record.positions.size() > room)
            {
                return false;
            }
            for (const uint32_t position : record.positions)
            {
                if (position >= objectCount)
                {
                    return false;
                }
            }
            out.host.insert(out.host.end(), record.positions.begin(), record.positions.end());
        }
        ++out.records;
        if (record.baseOrdinal == baseOrdinal)
        {
            break;
        }
        GenerationLease base;
        if (populations.pinExact(population, record.baseOrdinal, base) != OVSPATIALQUERY_API_SUCCESS)
        {
            return false;
        }
        version = std::move(base);
    }
    return true;
}

// The rows a version's device table has: its slots for a device-prepared
// version, whose tombstoned slots hold inverted bounds and pair with nothing,
// else its host objects.
uint32_t tableRowCount(const GenerationLease& generation)
{
    if (generation.preparedOnDevice())
    {
        return static_cast<uint32_t>(generation.cudaVersion()->description().objectCount);
    }
    return static_cast<uint32_t>(generation.objects().size());
}

cudaPairs::ObjectRow objectRow(const PopulationObject& object) noexcept
{
    cudaPairs::ObjectRow row{};
    row.id      = object.id.value;
    row.bits[0] = object.participationBits.words[0];
    row.bits[1] = object.participationBits.words[1];
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        row.minimum[axis] = object.bounds.minimum[axis];
        row.maximum[axis] = object.bounds.maximum[axis];
    }
    return row;
}

// Host bytes staged for the device and where they land.
struct Upload
{
    std::vector<unsigned char> bytes;
    uint64_t                   scratchOffset = UINT64_MAX; // into the scratch arena, or
    CuDevicePtr                destination   = 0u;         // straight into a device copy
};

// A kernel or copy that writes a device copy from staged bytes, in order.
struct TableWrite
{
    enum class Kind : uint8_t
    {
        rows,
        flags,
        copy,
    };
    Kind                          kind        = Kind::rows;
    uint32_t                      count       = 0u;
    size_t                        positions   = SIZE_MAX; // upload index of the positions, or none for every row
    size_t                        values      = SIZE_MAX; // upload index of the rows or flags
    CuDevicePtr                   source      = 0u;       // copy: device source
    uint64_t                      copyBytes   = 0u;
    CuDevicePtr                   destination = 0u; // flags array or copy destination
    std::shared_ptr<VersionTable> table;
};

} // namespace

// Device scratch of a reporter: shared by its advances, which run in order on
// the device, and replaced by a larger one when an advance needs more; the
// advances still reading the old one keep it alive.
struct DeviceArena
{
    std::shared_ptr<DeviceContext> device;
    CuDevicePtr                    pointer = 0u;
    uint64_t                       bytes   = 0u;
    std::shared_ptr<void>          budget;

    ~DeviceArena()
    {
        if (pointer == 0u || device == nullptr || device->lost.load(std::memory_order_relaxed))
        {
            return;
        }
        CurrentContext scope(*device);
        if (scope.status() == kCuSuccess)
        {
            (void)device->driver.memFree(pointer);
        }
    }
};

// Pinned staging of one advance's uploads, held until its copies complete so
// a pending advance never overwrites bytes another has not uploaded yet; a
// reporter pools them.
struct HostArena
{
    std::shared_ptr<DeviceContext> device;
    void*                          pointer = nullptr;
    uint64_t                       bytes   = 0u;
    HostLease                      lease;

    ~HostArena()
    {
        if (pointer == nullptr || device == nullptr || device->lost.load(std::memory_order_relaxed))
        {
            return;
        }
        CurrentContext scope(*device);
        if (scope.status() == kCuSuccess)
        {
            (void)device->driver.memFreeHost(pointer);
        }
    }
};

class CudaPairAdvance
{
  public:
    std::shared_ptr<CudaPairReporterDevice> reporter;
    std::shared_ptr<DeviceContext>          device;
    uint32_t                                slot = UINT32_MAX;
    // The device copies the advance reads and the copies it derived them
    // from, the versions whose change lists it reads, and the arenas it reads,
    // released once its work has completed.
    std::vector<std::shared_ptr<VersionTable>>                tables;
    std::vector<std::shared_ptr<const CudaPopulationVersion>> heldVersions;
    std::shared_ptr<DeviceArena>                              scratch;
    std::shared_ptr<HostArena>                                staging;
    bool                                                      patched = false;
    // The stream the advance runs on: the caller's or the reporter's.
    CuStream stream = nullptr;
    // The ring buffer the advance writes its new set into.
    uint32_t                    writeBuffer = UINT32_MAX;
    mutable std::mutex          mutex;
    bool                        workEnqueued      = false;
    bool                        readyRecorded     = false;
    bool                        wakeRecorded      = false;
    bool                        failed            = false;
    mutable bool                lost              = false;
    mutable CudaCompletionState readyState        = CudaCompletionState::pending;
    bool                        consumerRecorded  = false;
    CuEvent                     ownedConsumerDone = nullptr;
    mutable CudaCompletionState consumerState     = CudaCompletionState::pending;
};

struct CudaPairReporterDevice::Impl
{
    struct ResultSlot : CudaResultSlotEvents
    {
        CuDevicePtr header  = 0u;
        CuDevicePtr entered = 0u;
        CuDevicePtr exited  = 0u;
    };

    // Everything sized by the capacities, replaced as a whole when they grow:
    // the ring of pair sets (one more than the result slots, so every pending
    // advance has a buffer beside the baseline), and every slot's header and
    // delta rows.
    struct Storage
    {
        uint64_t                 activePairCapacity = 0u;
        uint64_t                 deltaPairCapacity  = 0u;
        CuDevicePtr              block              = 0u;
        uint64_t                 entryStride        = 0u; // entries between ring buffers
        std::vector<CuDevicePtr> slotBlocks;

        CuDevicePtr buffer(uint64_t index) const noexcept
        {
            return block + index * entryStride * sizeof(cudaPairs::Entry);
        }
    };

    CudaPairReporterConfig         config;
    std::shared_ptr<DeviceContext> device;
    CuStream                       stream = nullptr;
    // Where a retirement queues the ring that wakes the reaper after the slot's
    // consumer; up to two per slot may be outstanding.
    CudaSignalRings         signalRings;
    CuDevicePtr             control     = 0u;
    CuDevicePtr             ringCounts  = 0u;      // one accepted count per ring buffer
    void*                   hostHeaders = nullptr; // pinned, one Header per slot
    Storage                 storage;
    std::vector<ResultSlot> slots;
    // The ring buffer holding the baseline as far as the host has published,
    // the advance each buffer is reserved for, and the slot of the advance
    // admitted last, which the next one is ordered behind on the device.
    uint32_t                                    current = 0u;
    std::vector<std::weak_ptr<CudaPairAdvance>> ringOwners;
    uint32_t                                    lastSlot = UINT32_MAX;
    // Scratch on the device and pinned staging on the host, kept between
    // advances and charged while held; see DeviceArena and HostArena.
    std::shared_ptr<DeviceArena>            scratch;
    std::vector<std::shared_ptr<HostArena>> stagingPool;
    std::mutex                              mutex;

    ~Impl()
    {
        if (device == nullptr || device->lost.load(std::memory_order_relaxed))
        {
            return;
        }
        CurrentContext scope(*device);
        if (scope.status() != kCuSuccess)
        {
            return;
        }
        if (stream != nullptr)
        {
            (void)device->observe(device->driver.streamSynchronize(stream));
        }
        if (signalRings.stream != nullptr)
        {
            (void)device->driver.streamSynchronize(signalRings.stream);
            (void)device->driver.streamDestroy(signalRings.stream);
        }
        for (ResultSlot& slot : slots)
        {
            if (slot.cleanupStream != nullptr)
            {
                (void)device->driver.streamSynchronize(slot.cleanupStream);
            }
            (void)slot.destroy(device->driver, [](CuResult) { return true; });
        }
        freeStorage(storage);
        if (control != 0u)
        {
            (void)device->driver.memFree(control);
        }
        if (ringCounts != 0u)
        {
            (void)device->driver.memFree(ringCounts);
        }
        if (hostHeaders != nullptr)
        {
            (void)device->driver.memFreeHost(hostHeaders);
        }
        if (stream != nullptr)
        {
            (void)device->driver.streamDestroy(stream);
        }
    }

    void freeStorage(Storage& value) noexcept
    {
        if (value.block != 0u)
        {
            (void)device->driver.memFree(value.block);
        }
        for (const CuDevicePtr block : value.slotBlocks)
        {
            (void)device->driver.memFree(block);
        }
        value = Storage{};
    }

    // Allocates the capacity-sized arrays.
    CuResult allocateStorage(uint64_t activePairCapacity, uint64_t deltaPairCapacity, Storage& out) noexcept
    {
        out                    = Storage{};
        out.activePairCapacity = activePairCapacity;
        out.deltaPairCapacity  = deltaPairCapacity;
        // Buffers follow each other at exactly the capacity, so the kernels
        // step through the ring by entries; the charge rounds up, this does not.
        const uint64_t entryStride = std::max<uint64_t>(activePairCapacity, 1u);
        const uint64_t ring        = slots.size() + 1u;
        if (entryStride > std::numeric_limits<uint64_t>::max() / sizeof(cudaPairs::Entry) / ring)
        {
            return kCuOutOfMemory;
        }
        const uint64_t entryBytes = entryStride * sizeof(cudaPairs::Entry);
        CuResult       status     = device->observe(device->driver.memAlloc(&out.block, ring * entryBytes));
        if (status != kCuSuccess)
        {
            out.block = 0u;
            return status;
        }
        out.entryStride   = entryStride;
        uint64_t rowBytes = 0u;
        (void)addBytes(deltaPairCapacity, 2u * sizeof(uint64_t), rowBytes);
        try
        {
            out.slotBlocks.reserve(slots.size());
        }
        catch (...)
        {
            return kCuOutOfMemory;
        }
        for (size_t index = 0u; status == kCuSuccess && index < slots.size(); ++index)
        {
            CuDevicePtr block = 0u;
            status            = device->observe(device->driver.memAlloc(&block, kAlignment + 2u * rowBytes));
            if (status == kCuSuccess)
            {
                out.slotBlocks.push_back(block);
            }
        }
        return status;
    }

    void useStorage(Storage& value) noexcept
    {
        std::swap(storage, value);
        uint64_t rowBytes = 0u;
        (void)addBytes(storage.deltaPairCapacity, 2u * sizeof(uint64_t), rowBytes);
        for (size_t index = 0u; index < slots.size(); ++index)
        {
            slots[index].header  = storage.slotBlocks[index];
            slots[index].entered = storage.slotBlocks[index] + kAlignment;
            slots[index].exited  = storage.slotBlocks[index] + kAlignment + rowBytes;
        }
    }

    // The scratch arena an advance runs in, grown when it needs more and
    // charged to the device as staging; a replaced arena lives on in the
    // pending advances that read it.
    ovspatialquery_api_status_t reserveScratch(ovspatialquery_instance_handle_t instance, uint64_t bytes,
                                               std::shared_ptr<DeviceArena>& outArena) noexcept
    {
        if (scratch != nullptr && bytes <= scratch->bytes)
        {
            outArena = scratch;
            return OVSPATIALQUERY_API_SUCCESS;
        }
        const uint64_t held  = scratch != nullptr ? scratch->bytes : 0u;
        const uint64_t grown = std::max<uint64_t>(bytes, held + held / 2u);
        try
        {
            std::shared_ptr<DeviceArena>      arena = std::make_shared<DeviceArena>();
            const ovspatialquery_api_status_t reserved =
                reserveInstanceDeviceMemory(instance, config.device, grown, arena->budget);
            if (reserved != OVSPATIALQUERY_API_SUCCESS)
            {
                return reserved == OVSPATIALQUERY_API_DEVICE_BUDGET ? OVSPATIALQUERY_API_DEVICE_BUDGET
                                                                    : OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            arena->device = device;
            if (device->observe(device->driver.memAlloc(&arena->pointer, grown)) != kCuSuccess)
            {
                arena->pointer = 0u;
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            arena->bytes = grown;
            scratch      = arena;
            outArena     = std::move(arena);
            return OVSPATIALQUERY_API_SUCCESS;
        }
        catch (...)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }

    // A pinned staging arena no pending advance holds, grown when too small
    // and charged to the host as staging; a new one when every arena is held.
    ovspatialquery_api_status_t acquireStaging(InstanceHostLedger* ledger, uint64_t bytes,
                                               std::shared_ptr<HostArena>& outArena) noexcept
    {
        std::shared_ptr<HostArena>* slot = nullptr;
        for (std::shared_ptr<HostArena>& arena : stagingPool)
        {
            if (arena.use_count() == 1)
            {
                if (bytes <= arena->bytes)
                {
                    outArena = arena;
                    return OVSPATIALQUERY_API_SUCCESS;
                }
                slot = &arena;
            }
        }
        const uint64_t held  = slot != nullptr ? (*slot)->bytes : 0u;
        const uint64_t grown = std::max<uint64_t>(bytes, held + held / 2u);
        try
        {
            std::shared_ptr<HostArena> arena = std::make_shared<HostArena>();
            if (ledger != nullptr)
            {
                HostTransaction transaction(*ledger);
                if (!transaction.reserve(grown) || !transaction.commit(arena->lease))
                {
                    return OVSPATIALQUERY_API_HOST_BUDGET;
                }
            }
            arena->device = device;
            if (device->observe(device->driver.memHostAlloc(&arena->pointer, static_cast<size_t>(grown),
                                                            kCuMemHostAllocPortable)) != kCuSuccess)
            {
                arena->pointer = nullptr;
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            arena->bytes = grown;
            if (slot != nullptr)
            {
                *slot = arena;
            }
            else
            {
                stagingPool.push_back(arena);
            }
            outArena = std::move(arena);
            return OVSPATIALQUERY_API_SUCCESS;
        }
        catch (...)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }

    // Completes an advance whose enqueue failed on the host after some of its
    // work was enqueued: writes the FAILED outcome, zero counts and the unchanged
    // accepted count to the slot's counts behind that work (ADR-0022 decision
    // 3), then records the slot's ready and wake events. Returns whether the
    // ready event was recorded. The device's context is current.
    bool failOnDevice(CudaPairAdvance& advance) noexcept
    {
        Driver&           driver = device->driver;
        const ResultSlot& slot   = slots[advance.slot];
        const CuStream    on     = advance.stream;
        // `finish` with `failed` set: no write buffer, capacities or endpoint status words.
        cudaPairs::Control* controlArgument = devicePointer<cudaPairs::Control>(control);
        unsigned long long* countsArgument  = devicePointer<unsigned long long>(ringCounts);
        unsigned int        writeBuffer     = 0u;
        unsigned long long  capacity        = 0u;
        cudaPairs::Header*  headerArgument  = devicePointer<cudaPairs::Header>(slot.header);
        const unsigned int* noStatus        = nullptr;
        unsigned int        failed          = 1u;
        void*               parameters[]    = {&controlArgument, &countsArgument, &writeBuffer, &capacity, &capacity,
                                               &headerArgument,  &noStatus,       &noStatus,    &failed};
        if (device->launch(fnFinish, 1u, parameters, on, true) != kCuSuccess)
        {
            // The outcome word alone, when even a launch fails.
            constexpr size_t kWords = sizeof(cudaPairs::Header) / sizeof(unsigned int);
            (void)driver.memsetD32Async(slot.header, 0u, kWords, on);
            (void)driver.memsetD32Async(slot.header, cudaPairs::kOutcomeFailed, 1u, on);
        }
        if (driver.eventRecord(slot.ready, on) != kCuSuccess)
        {
            return false;
        }
        advance.wakeRecorded = driver.eventRecord(slot.wake, on) == kCuSuccess;
        return true;
    }
};

namespace
{

// The device copy one endpoint population's version needs for an advance,
// and how to make it.
struct CopyPlan
{
    enum class Build : uint8_t
    {
        existing,
        borrowed,
        stolen,
        derived,
        full,
    };

    PopulationKey         population;
    const ResolvedTarget* version = nullptr;
    // The views of this population's endpoints, each with a target that
    // selects it; a whole-population endpoint has no key.
    std::array<SelectionKey, 2u>                 keys{};
    std::array<const ResolvedTarget*, 2u>        selections{};
    size_t                                       selectionCount = 0u;
    Build                                        build          = Build::existing;
    std::shared_ptr<GenerationDeviceAttachments> attachments;
    // The copy of the version a CUDA advance selected last, when another.
    std::shared_ptr<GenerationDeviceAttachments> previousAttachments;
    std::shared_ptr<VersionTable>                previous;
    std::shared_ptr<VersionTable>                table;
    // Rows a derived or rewritten copy takes from the version.
    std::vector<uint32_t>              positions;
    std::vector<std::shared_ptr<void>> leases;
    uint64_t                           chargedBytes = 0u;
    // Memberships this plan added to the copy, removed again if the advance
    // is not admitted.
    std::vector<SelectionKey> addedKeys;
};

// Frees and removes the memberships `drop` selects; the device's context is current.
template <typename Predicate>
void dropMemberships(VersionTable& table, Predicate drop) noexcept
{
    for (size_t index = 0u; index < table.memberships.size();)
    {
        if (!drop(table.memberships[index].key))
        {
            ++index;
            continue;
        }
        if (table.memberships[index].flags != 0u)
        {
            (void)table.device->driver.memFree(table.memberships[index].flags);
        }
        table.memberships.erase(table.memberships.begin() + static_cast<std::ptrdiff_t>(index));
    }
}

// Removes the memberships a plan added to an existing copy before anything
// reads them.
void removeAddedMemberships(CopyPlan& plan) noexcept
{
    if (plan.build == CopyPlan::Build::existing && plan.table != nullptr)
    {
        dropMemberships(*plan.table,
                        [&](const SelectionKey& key)
                        {
                            return std::any_of(plan.addedKeys.begin(), plan.addedKeys.end(),
                                               [&](const SelectionKey& added) { return sameSelection(key, added); });
                        });
    }
    plan.addedKeys.clear();
}

// What a launch uploads, and the writes that build device copies from it.
// Every transient host buffer is charged as staging before it is built.
struct Staging
{
    InstanceHostLedger*     ledger = nullptr;
    std::vector<HostLease>  leases;
    std::vector<Upload>     uploads;
    std::vector<TableWrite> writes;

    bool charge(uint64_t bytes)
    {
        if (ledger == nullptr || bytes == 0u)
        {
            return true;
        }
        HostLease       lease;
        HostTransaction transaction(*ledger);
        if (!transaction.reserve(bytes) || !transaction.commit(lease))
        {
            return false;
        }
        leases.push_back(std::move(lease));
        return true;
    }

    // An upload of `bytes` the caller fills in place.
    size_t allocate(size_t bytes)
    {
        Upload upload;
        upload.bytes.resize(bytes);
        uploads.push_back(std::move(upload));
        return uploads.size() - 1u;
    }

    size_t stage(const void* data, size_t bytes)
    {
        const size_t index = allocate(bytes);
        if (bytes != 0u)
        {
            std::memcpy(uploads[index].bytes.data(), data, bytes);
        }
        return index;
    }

    // Writes the selection's membership at the rows a copy takes from its version.
    void stageFlags(const CopyPlan& plan, size_t selection, CuDevicePtr flags)
    {
        if (plan.positions.empty())
        {
            return;
        }
        TableWrite write;
        write.kind      = TableWrite::Kind::flags;
        write.count     = static_cast<uint32_t>(plan.positions.size());
        write.positions = stage(plan.positions.data(), plan.positions.size() * sizeof(uint32_t));
        write.values    = allocate(plan.positions.size());
        for (size_t index = 0u; index < plan.positions.size(); ++index)
        {
            uploads[write.values].bytes[index] =
                plan.selections[selection]->generation.includesObject(plan.positions[index]) ? 1u : 0u;
        }
        write.destination = flags;
        writes.push_back(std::move(write));
    }
};

// What an event or stream query says about the work before it.
CudaCompletionState completionState(CuResult status, bool failed = false) noexcept
{
    return status == kCuSuccess      ? (failed ? CudaCompletionState::error : CudaCompletionState::complete)
           : status == kCuNotReady   ? CudaCompletionState::pending
           : status == kCuDeviceLost ? CudaCompletionState::deviceLost
                                     : CudaCompletionState::error;
}

ovspatialquery_api_status_t deviceStatus(CuResult status) noexcept
{
    return status == kCuDeviceLost
               ? OVSPATIALQUERY_API_DEVICE_LOST
               : (status == kCuOutOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY : OVSPATIALQUERY_API_INTERNAL);
}

// Finds, or plans, charges and allocates, the device copy of one endpoint
// population's version (ADR-0019 decisions 6 and 7), staging the writes that
// fill it. A complete set published through CUDA on the device lends its arrays;
// otherwise the copy derives from the copy selected last when the records
// between them allow it, rewriting that copy in place when nothing reads it,
// and is uploaded whole when they do not. The caller holds the population
// table's device copy mutex.
ovspatialquery_api_status_t prepareCopy(ovspatialquery_instance_handle_t instance, const PopulationTable& populations,
                                        const std::shared_ptr<DeviceContext>& device, DLDevice deviceHandle,
                                        CopyPlan& plan, Staging& staging)
{
    Driver&                driver   = device->driver;
    const int32_t          deviceId = deviceHandle.device_id;
    const GenerationLease& version  = plan.version->generation;
    const ObjectTable&     objects  = version.objects();
    const uint32_t         count    = tableRowCount(version);
    plan.attachments                = version.deviceAttachments();
    plan.table                      = attachedTable(*plan.attachments, deviceId);
    plan.previousAttachments        = populations.lastDeviceAttachments(plan.population, deviceId).lock();
    if (plan.previousAttachments == plan.attachments)
    {
        plan.previousAttachments.reset();
    }
    std::shared_ptr<const CudaPopulationVersion> cudaVersion;
    CudaPopulationDescription                    description{};
    if (plan.table != nullptr)
    {
        plan.build = CopyPlan::Build::existing;
        // A view no reporter holds any more cannot be named again: its
        // membership is freed, and no admitted advance reads it.
        dropMemberships(*plan.table, [](const SelectionKey& key) { return key.expired(); });
    }
    else
    {
        if (plan.previousAttachments != nullptr)
        {
            plan.previous = attachedTable(*plan.previousAttachments, deviceId);
        }
        cudaVersion = version.cudaVersion();
        if (cudaVersion != nullptr)
        {
            description = cudaVersion->description();
        }
        ChangeChain  derivation;
        const size_t limit = std::max<size_t>(count / 2u, 1u);
        if (plan.previous != nullptr && !staging.charge(uint64_t{limit} * sizeof(uint32_t)))
        {
            return OVSPATIALQUERY_API_HOST_BUDGET;
        }
        if (cudaVersion != nullptr && description.device.device_type == kDLCUDA &&
            description.device.device_id == deviceId && description.objectCount == count)
        {
            plan.build = CopyPlan::Build::borrowed;
        }
        else if (plan.previous != nullptr && plan.previous->borrowed == nullptr &&
                 plan.previous->ordinal < version.ordinal() &&
                 chainPositions(populations, plan.population, version, plan.previous->ordinal, limit, derivation) &&
                 derivation.device.empty())
        {
            // A copy derives from host positions only; device lists belong to
            // versions whose tables are borrowed instead.
            std::sort(derivation.host.begin(), derivation.host.end());
            derivation.host.erase(std::unique(derivation.host.begin(), derivation.host.end()), derivation.host.end());
            plan.positions = std::move(derivation.host);
            // Only the previous version's attachment and this plan hold the
            // copy: no advance reads it, so it is rewritten in place.
            plan.build = plan.previous.use_count() == 2 && plan.previous->capacity >= count ? CopyPlan::Build::stolen
                                                                                            : CopyPlan::Build::derived;
        }
        else
        {
            plan.build = CopyPlan::Build::full;
        }
    }
    const bool           reused   = plan.build == CopyPlan::Build::existing || plan.build == CopyPlan::Build::stolen;
    const VersionTable*  original = plan.build == CopyPlan::Build::existing
                                        ? plan.table.get()
                                        : (plan.build == CopyPlan::Build::stolen ? plan.previous.get() : nullptr);
    std::array<bool, 2u> added{};
    size_t               membershipCount = 0u;
    for (size_t selection = 0u; selection < plan.selectionCount; ++selection)
    {
        const SelectionKey& key = plan.keys[selection];
        added[selection]        = hasSelection(key) && (original == nullptr || original->membership(key) == 0u);
        membershipCount += hasSelection(key) ? 1u : 0u;
    }
    if (plan.build == CopyPlan::Build::existing)
    {
        // An existing copy keeps the memberships of other reporters' views.
        membershipCount = original->memberships.size();
        for (const bool value : added)
        {
            membershipCount += value ? 1u : 0u;
        }
    }
    const VersionChangeRecord& record   = version.changeRecord();
    const uint32_t             capacity = original != nullptr ? original->capacity : count;
    const bool                 ownsRows =
        plan.build != CopyPlan::Build::borrowed && (original == nullptr || original->borrowed == nullptr);
    const uint64_t recordCount = plan.build == CopyPlan::Build::existing
                                     ? original->recordCount
                                     : (record.everything ? 0u : static_cast<uint64_t>(record.positions.size()));
    uint64_t       bytes       = 0u;
    if (!addBytes(ownsRows ? capacity : 0u, sizeof(cudaPairs::ObjectRow), bytes) ||
        !addBytes(membershipCount, capacity, bytes) || !addBytes(recordCount, sizeof(uint32_t), bytes))
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    const uint64_t charged = original != nullptr ? original->charged : 0u;
    if (bytes > charged)
    {
        std::shared_ptr<void>             lease;
        const ovspatialquery_api_status_t reserved =
            reserveInstanceDeviceMemory(instance, deviceHandle, bytes - charged, lease);
        if (reserved != OVSPATIALQUERY_API_SUCCESS)
        {
            return reserved == OVSPATIALQUERY_API_DEVICE_BUDGET ? OVSPATIALQUERY_API_DEVICE_BUDGET
                                                                : OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        plan.leases.push_back(std::move(lease));
    }
    plan.chargedBytes = std::max(bytes, charged);
    if (plan.build == CopyPlan::Build::existing && !added[0] && !added[1])
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    // The host buffers this copy stages, charged before any is built.
    const uint64_t rewritten = plan.positions.size();
    uint64_t       transient = 0u;
    if (plan.build == CopyPlan::Build::full)
    {
        transient += uint64_t{count} * sizeof(cudaPairs::ObjectRow);
    }
    else if (plan.build == CopyPlan::Build::stolen || plan.build == CopyPlan::Build::derived)
    {
        transient += rewritten * (sizeof(cudaPairs::ObjectRow) + sizeof(uint32_t));
    }
    if (plan.build != CopyPlan::Build::existing && !record.everything)
    {
        transient += uint64_t{record.positions.size()} * sizeof(uint32_t);
    }
    for (size_t selection = 0u; selection < plan.selectionCount; ++selection)
    {
        const bool kept = plan.build == CopyPlan::Build::stolen && hasSelection(plan.keys[selection]) &&
                          plan.previous->membership(plan.keys[selection]) != 0u;
        const bool patched = kept || (added[selection] && plan.build == CopyPlan::Build::derived &&
                                      plan.previous->membership(plan.keys[selection]) != 0u);
        transient += patched ? rewritten * (1u + sizeof(uint32_t)) : (added[selection] ? uint64_t{count} : 0u);
    }
    if (!staging.charge(transient))
    {
        return OVSPATIALQUERY_API_HOST_BUDGET;
    }
    if (plan.build == CopyPlan::Build::stolen)
    {
        plan.table = plan.previous;
    }
    else if (!reused)
    {
        plan.table = std::make_shared<VersionTable>(device);
        if (device->observe(driver.eventCreate(&plan.table->ready, kCuEventDisableTiming)) != kCuSuccess)
        {
            plan.table->ready = nullptr;
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }
    VersionTable& table = *plan.table;
    if (plan.build == CopyPlan::Build::borrowed)
    {
        table.borrowed       = cudaVersion;
        table.capacity       = count;
        table.ids            = static_cast<CuDevicePtr>(reinterpret_cast<uintptr_t>(description.objectIds));
        table.bits           = static_cast<CuDevicePtr>(reinterpret_cast<uintptr_t>(description.objectBits));
        table.minimums       = static_cast<CuDevicePtr>(reinterpret_cast<uintptr_t>(description.objectMinimums));
        table.maximums       = static_cast<CuDevicePtr>(reinterpret_cast<uintptr_t>(description.objectMaximums));
        table.borrowedReady  = reinterpret_cast<CuEvent>(description.ready.value);
        table.borrowedFailed = static_cast<CuDevicePtr>(reinterpret_cast<uintptr_t>(description.preparationFailed));
    }
    else if (!reused)
    {
        table.capacity = count;
        if (count != 0u && device->observe(driver.memAlloc(
                               &table.objects, uint64_t{count} * sizeof(cudaPairs::ObjectRow))) != kCuSuccess)
        {
            table.objects = 0u;
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        // Per row one ID word, two participation words, three minimums and three maximums.
        const uint64_t column = uint64_t{count} * sizeof(uint64_t);
        table.ids             = table.objects;
        table.bits            = table.objects + column;
        table.minimums        = table.objects + 3u * column;
        table.maximums        = table.objects + 6u * column;
    }
    if (plan.build != CopyPlan::Build::existing)
    {
        if (plan.build == CopyPlan::Build::derived && plan.previous->objectCount != 0u)
        {
            // The source's rows, column by column, before the changed rows.
            const VersionTable&                                       source = *plan.previous;
            const uint64_t                                            rows   = source.objectCount;
            const std::array<std::pair<CuDevicePtr, CuDevicePtr>, 4u> columns{{
                {source.ids, table.ids},
                {source.bits, table.bits},
                {source.minimums, table.minimums},
                {source.maximums, table.maximums},
            }};
            for (size_t column = 0u; column < columns.size(); ++column)
            {
                TableWrite write;
                write.kind        = TableWrite::Kind::copy;
                write.source      = columns[column].first;
                write.destination = columns[column].second;
                write.copyBytes   = rows * (column == 0u   ? sizeof(uint64_t)
                                            : column == 1u ? 2u * sizeof(uint64_t)
                                                           : 3u * sizeof(double));
                staging.writes.push_back(std::move(write));
            }
        }
        if (plan.build != CopyPlan::Build::borrowed)
        {
            // The rows are written straight into their upload.
            TableWrite write;
            write.kind         = TableWrite::Kind::rows;
            write.table        = plan.table;
            const bool   whole = plan.build == CopyPlan::Build::full;
            const size_t rows  = whole ? size_t{count} : plan.positions.size();
            if (!whole)
            {
                write.positions = staging.stage(plan.positions.data(), plan.positions.size() * sizeof(uint32_t));
            }
            write.count  = static_cast<uint32_t>(rows);
            write.values = staging.allocate(rows * sizeof(cudaPairs::ObjectRow));
            cudaPairs::ObjectRow* staged =
                reinterpret_cast<cudaPairs::ObjectRow*>(staging.uploads[write.values].bytes.data());
            for (size_t index = 0u; index < rows; ++index)
            {
                staged[index] = objectRow(objects[whole ? index : plan.positions[index]]);
            }
            if (write.count != 0u)
            {
                staging.writes.push_back(std::move(write));
            }
        }
        table.ordinal     = version.ordinal();
        table.objectCount = count;
        // The version's own record, which a later advance marks on the device.
        if (table.recordPositions != 0u)
        {
            (void)driver.memFree(table.recordPositions);
            table.recordPositions = 0u;
        }
        table.recordCount = 0u;
        if (!record.everything && !record.positions.empty())
        {
            if (device->observe(driver.memAlloc(&table.recordPositions, record.positions.size() * sizeof(uint32_t))) !=
                kCuSuccess)
            {
                table.recordPositions = 0u;
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            table.recordCount   = static_cast<uint32_t>(record.positions.size());
            const size_t upload = staging.stage(record.positions.data(), record.positions.size() * sizeof(uint32_t));
            staging.uploads[upload].destination = table.recordPositions;
        }
        if (plan.build == CopyPlan::Build::stolen)
        {
            // A rewritten copy keeps the memberships this advance names, patched
            // at the changed rows, and drops the others.
            dropMemberships(table,
                            [&](const SelectionKey& key) {
                                return !hasSelection(key) ||
                                       (!sameSelection(key, plan.keys[0]) && !sameSelection(key, plan.keys[1]));
                            });
            for (size_t selection = 0u; selection < plan.selectionCount; ++selection)
            {
                const CuDevicePtr flags =
                    hasSelection(plan.keys[selection]) ? table.membership(plan.keys[selection]) : 0u;
                if (flags != 0u)
                {
                    staging.stageFlags(plan, selection, flags);
                }
            }
        }
    }
    for (size_t selection = 0u; selection < plan.selectionCount; ++selection)
    {
        if (!added[selection])
        {
            continue;
        }
        Membership membership;
        membership.key = plan.keys[selection];
        // Both lists grow before the flags exist, so neither push can fail after.
        table.memberships.reserve(table.memberships.size() + 1u);
        plan.addedKeys.reserve(plan.addedKeys.size() + 1u);
        if (table.capacity != 0u && device->observe(driver.memAlloc(&membership.flags, table.capacity)) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        table.memberships.push_back(membership);
        plan.addedKeys.push_back(membership.key);
        const CuDevicePtr source =
            plan.build == CopyPlan::Build::derived ? plan.previous->membership(membership.key) : 0u;
        if (source != 0u)
        {
            TableWrite copy;
            copy.kind        = TableWrite::Kind::copy;
            copy.source      = source;
            copy.destination = membership.flags;
            copy.copyBytes   = plan.previous->objectCount;
            staging.writes.push_back(std::move(copy));
            staging.stageFlags(plan, selection, membership.flags);
        }
        else if (table.objectCount != 0u)
        {
            const size_t upload = staging.allocate(0u);
            plan.selections[selection]->generation.admittedObjects(ParticipationBits{}, staging.uploads[upload].bytes);
            staging.uploads[upload].destination = membership.flags;
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace

CudaPairReporterDevice::CudaPairReporterDevice(std::unique_ptr<Impl> impl) noexcept : mImpl(std::move(impl)) {}

CudaPairReporterDevice::~CudaPairReporterDevice() = default;

uint64_t CudaPairReporterDevice::residentHostBytes(uint32_t resultSlotCount) noexcept
{
    // The reporter, its device context and their ownership blocks, and per
    // slot its handles, its pinned outcome header and its storage block, held
    // twice while the capacities grow.
    constexpr uint64_t sharedControl = 4u * sizeof(void*);
    return sizeof(CudaPairReporterDevice) + sizeof(Impl) + sizeof(DeviceContext) + 3u * sharedControl +
           uint64_t{resultSlotCount} *
               (sizeof(Impl::ResultSlot) + sizeof(cudaPairs::Header) + 2u * sizeof(CuDevicePtr));
}

bool CudaPairReporterDevice::residentDeviceBytes(uint64_t activePairCapacity, uint64_t deltaPairCapacity,
                                                 uint32_t resultSlotCount, uint64_t& outBytes) noexcept
{
    if (activePairCapacity > kMaximumHashSlots / 2u)
    {
        return false;
    }
    uint64_t total = 0u;
    uint64_t rows  = 0u;
    uint64_t set   = 0u;
    // Charged as allocated: the control record and the ring counts, then the
    // ring of result_slot_count + 1 sets, then the slots.
    const uint64_t ring = uint64_t{resultSlotCount} + 1u;
    if (!addBytes(1u, sizeof(cudaPairs::Control), total) || !addBytes(ring, sizeof(unsigned long long), total) ||
        !addBytes(std::max<uint64_t>(activePairCapacity, 1u), sizeof(cudaPairs::Entry), set) ||
        !addBytes(ring, set, total) || !addBytes(deltaPairCapacity, 2u * sizeof(uint64_t), rows) ||
        rows > std::numeric_limits<uint64_t>::max() / 2u || !addBytes(resultSlotCount, kAlignment + 2u * rows, total))
    {
        return false;
    }
    outBytes = total;
    return true;
}

ovspatialquery_api_status_t CudaPairReporterDevice::create(
    const CudaPairReporterConfig& config, std::shared_ptr<CudaPairReporterDevice>& outReporter) noexcept
{
    uint64_t resident = 0u;
    if (config.device.device_type != kDLCUDA || config.device.device_id < 0 || config.resultSlotCount == 0u ||
        !residentDeviceBytes(config.activePairCapacity, config.deltaPairCapacity, config.resultSlotCount, resident))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::unique_ptr<Impl> impl(new Impl());
        impl->config                       = config;
        impl->device                       = std::make_shared<DeviceContext>();
        ovspatialquery_api_status_t status = impl->device->open(config.device.device_id);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        CurrentContext scope(*impl->device);
        if (scope.status() != kCuSuccess)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        Driver& driver = impl->device->driver;
        impl->slots.resize(config.resultSlotCount);
        CuResult result = driver.streamCreate(&impl->stream, kCuStreamNonBlocking);
        if (result == kCuSuccess)
        {
            result = impl->signalRings.create(driver, 2u * config.resultSlotCount);
        }
        for (size_t index = 0u; result == kCuSuccess && index < impl->slots.size(); ++index)
        {
            result = impl->slots[index].create(driver);
        }
        if (result == kCuSuccess)
        {
            result = driver.memAlloc(&impl->control, sizeof(cudaPairs::Control));
        }
        if (result == kCuSuccess)
        {
            result = driver.memsetD8Async(impl->control, 0u, sizeof(cudaPairs::Control), impl->stream);
        }
        const uint64_t ring = uint64_t{config.resultSlotCount} + 1u;
        if (result == kCuSuccess)
        {
            result = driver.memAlloc(&impl->ringCounts, ring * sizeof(unsigned long long));
        }
        if (result == kCuSuccess)
        {
            result = driver.memsetD8Async(impl->ringCounts, 0u, ring * sizeof(unsigned long long), impl->stream);
        }
        impl->ringOwners.resize(ring);
        if (result == kCuSuccess)
        {
            result = driver.memHostAlloc(&impl->hostHeaders, sizeof(cudaPairs::Header) * config.resultSlotCount,
                                         kCuMemHostAllocPortable);
        }
        Impl::Storage storage;
        if (result == kCuSuccess)
        {
            result = impl->allocateStorage(config.activePairCapacity, config.deltaPairCapacity, storage);
        }
        if (result == kCuSuccess)
        {
            impl->useStorage(storage);
            result = impl->device->observe(driver.streamSynchronize(impl->stream));
        }
        else
        {
            impl->freeStorage(storage);
        }
        if (result != kCuSuccess)
        {
            return result == kCuOutOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY : OVSPATIALQUERY_API_INTERNAL;
        }
        outReporter = std::shared_ptr<CudaPairReporterDevice>(new CudaPairReporterDevice(std::move(impl)));
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

ovspatialquery_cuda_event_t CudaPairReporterDevice::readyEvent(uint32_t resultSlot) const noexcept
{
    return {reinterpret_cast<uintptr_t>(mImpl->slots[resultSlot].ready)};
}

const void* CudaPairReporterDevice::enteredRows(uint32_t resultSlot) const noexcept
{
    return devicePointer<const void>(mImpl->slots[resultSlot].entered);
}

const void* CudaPairReporterDevice::exitedRows(uint32_t resultSlot) const noexcept
{
    return devicePointer<const void>(mImpl->slots[resultSlot].exited);
}

const void* CudaPairReporterDevice::countsRows(uint32_t resultSlot) const noexcept
{
    return devicePointer<const void>(mImpl->slots[resultSlot].header);
}

void CudaPairReporterDevice::accept(const CudaPairAdvance& advance) noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (advance.writeBuffer < mImpl->ringOwners.size())
    {
        mImpl->current = advance.writeBuffer;
    }
}

void CudaPairReporterDevice::release(const CudaPairAdvance& advance) noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    if (advance.writeBuffer < mImpl->ringOwners.size() &&
        mImpl->ringOwners[advance.writeBuffer].lock().get() == &advance)
    {
        mImpl->ringOwners[advance.writeBuffer].reset();
    }
}

ovspatialquery_api_status_t CudaPairReporterDevice::grow(uint64_t activePairCapacity,
                                                         uint64_t deltaPairCapacity) noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    uint64_t                    resident = 0u;
    if (!residentDeviceBytes(activePairCapacity, deltaPairCapacity, mImpl->config.resultSlotCount, resident))
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    CurrentContext scope(*mImpl->device);
    if (scope.status() != kCuSuccess)
    {
        return deviceStatus(scope.status());
    }
    Driver&       driver = mImpl->device->driver;
    Impl::Storage storage;
    CuResult      status = mImpl->allocateStorage(activePairCapacity, deltaPairCapacity, storage);
    // The baseline moves into the new buffer of the same index.
    const uint32_t     current_      = mImpl->current;
    const CuDevicePtr  count         = mImpl->ringCounts + current_ * sizeof(unsigned long long);
    unsigned long long baselineCount = 0u;
    if (status == kCuSuccess)
    {
        status = mImpl->device->observe(driver.streamSynchronize(mImpl->stream));
    }
    if (status == kCuSuccess)
    {
        status = mImpl->device->observe(driver.copyDtoH(&baselineCount, count, sizeof(baselineCount)));
    }
    if (status == kCuSuccess && baselineCount > activePairCapacity)
    {
        status = kCuInvalidHandle;
    }
    if (status == kCuSuccess && baselineCount != 0u)
    {
        status = mImpl->device->observe(driver.copyDtoDAsync(storage.buffer(current_), mImpl->storage.buffer(current_),
                                                             baselineCount * sizeof(cudaPairs::Entry), mImpl->stream));
    }
    if (status == kCuSuccess)
    {
        status = mImpl->device->observe(driver.streamSynchronize(mImpl->stream));
    }
    if (status != kCuSuccess)
    {
        (void)driver.streamSynchronize(mImpl->stream);
        mImpl->freeStorage(storage);
        return deviceStatus(status);
    }
    mImpl->useStorage(storage);
    mImpl->freeStorage(storage);
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t CudaPairReporterDevice::launch(ovspatialquery_instance_handle_t instance,
                                                           const PopulationTable&           populations,
                                                           InstanceHostLedger* hostLedger, uint32_t resultSlot,
                                                           const ResolvedTarget& first, const ResolvedTarget& second,
                                                           const std::optional<CudaPairBaseline>& baseline,
                                                           const CudaPairAdvanceSync&             sync,
                                                           std::shared_ptr<CudaPairAdvance>&      outAdvance) noexcept
{
    Impl&                       impl = *mImpl;
    std::lock_guard<std::mutex> lock(impl.mutex);
    if (resultSlot >= impl.slots.size())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (impl.device->lost.load(std::memory_order_relaxed))
    {
        return OVSPATIALQUERY_API_DEVICE_LOST;
    }
    const CudaPairReporterConfig& config         = impl.config;
    const bool                    samePopulation = first.population == second.population;
    const bool                    identical      = config.relation == PairEndpointRelation::identicalSelection;
    const int32_t                 deviceId       = config.device.device_id;
    const uint64_t                firstCount     = tableRowCount(first.generation);
    const uint64_t                secondCount    = tableRowCount(second.generation);
    if (firstCount >= kMaximumHashSlots || secondCount >= kMaximumHashSlots)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    // A free ring buffer for the new set: not the published baseline and not
    // reserved by an advance whose outcome is not published yet.
    uint32_t writeBuffer = UINT32_MAX;
    for (uint32_t index = 0u; index < impl.ringOwners.size() && writeBuffer == UINT32_MAX; ++index)
    {
        if (index != impl.current && impl.ringOwners[index].expired())
        {
            writeBuffer = index;
        }
    }
    if (writeBuffer == UINT32_MAX)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    std::shared_ptr<CudaPairAdvance> advance;
    try
    {
        advance              = std::make_shared<CudaPairAdvance>();
        advance->reporter    = shared_from_this();
        advance->device      = impl.device;
        advance->slot        = resultSlot;
        advance->writeBuffer = writeBuffer;
        advance->stream      = sync.hasStream ? reinterpret_cast<CuStream>(sync.stream) : impl.stream;
        Staging staging;
        staging.ledger = hostLedger;

        // Patch from the records between the baseline and the selected
        // versions when every link is retained and together they list at most
        // a quarter of the objects; otherwise recompute (ADR-0019 decision 4).
        // The positions are charged, with their staged copy, before they are gathered.
        std::array<ChangeChain, 2u> chains;
        bool                        patched = baseline.has_value();
        if (patched &&
            !staging.charge(2u * sizeof(uint32_t) *
                            (uint64_t{VersionChangeRecord::positionLimit(firstCount)} +
                             (samePopulation ? 0u : uint64_t{VersionChangeRecord::positionLimit(secondCount)}))))
        {
            return OVSPATIALQUERY_API_HOST_BUDGET;
        }
        if (patched)
        {
            patched = chainPositions(populations, first.population, first.generation, baseline->firstOrdinal,
                                     VersionChangeRecord::positionLimit(firstCount), chains[0]) &&
                      (samePopulation ||
                       chainPositions(populations, second.population, second.generation, baseline->secondOrdinal,
                                      VersionChangeRecord::positionLimit(secondCount), chains[1]));
        }
        if (!patched)
        {
            chains = {};
        }
        advance->patched = patched;
        // The versions whose device change lists the kernels read stay alive
        // until the advance completes.
        advance->heldVersions = std::move(chains[0].held);
        advance->heldVersions.insert(advance->heldVersions.end(), chains[1].held.begin(), chains[1].held.end());

        CurrentContext scope(*impl.device);
        if (scope.status() != kCuSuccess)
        {
            return deviceStatus(scope.status());
        }
        Driver& driver = impl.device->driver;

        // The device copy of each endpoint population's version.
        std::array<CopyPlan, 2u> plans;
        const size_t             planCount = samePopulation ? 1u : 2u;
        plans[0].population                = first.population;
        plans[0].version                   = &first;
        plans[0].keys[0]                   = config.firstSelection;
        plans[0].selections[0]             = &first;
        plans[0].selectionCount            = 1u;
        if (!samePopulation)
        {
            plans[1].population     = second.population;
            plans[1].version        = &second;
            plans[1].keys[0]        = config.secondSelection;
            plans[1].selections[0]  = &second;
            plans[1].selectionCount = 1u;
        }
        else if (!sameSelection(config.secondSelection, config.firstSelection))
        {
            plans[0].keys[1]        = config.secondSelection;
            plans[0].selections[1]  = &second;
            plans[0].selectionCount = 2u;
        }
        std::unique_lock<std::mutex> copies(populations.deviceCopyMutex());
        // Undoes what planning changed on attached copies on every return
        // before the advance is enqueued: a copy chosen for rewriting in place
        // is detached, since its description no longer matches its contents,
        // and memberships added to an existing copy are removed.
        struct Rollback
        {
            std::array<CopyPlan, 2u>& plans;
            size_t                    count;
            int32_t                   deviceId;
            bool                      armed = true;
            ~Rollback()
            {
                for (size_t index = 0u; armed && index < count; ++index)
                {
                    CopyPlan& plan = plans[index];
                    if (plan.build == CopyPlan::Build::stolen && plan.previousAttachments != nullptr)
                    {
                        detachTable(*plan.previousAttachments, deviceId);
                    }
                    removeAddedMemberships(plan);
                }
            }
        } rollback{plans, planCount, deviceId};
        for (size_t index = 0u; index < planCount; ++index)
        {
            const ovspatialquery_api_status_t prepared =
                prepareCopy(instance, populations, impl.device, config.device, plans[index], staging);
            if (prepared != OVSPATIALQUERY_API_SUCCESS)
            {
                return prepared;
            }
        }
        const std::array<const VersionTable*, 2u> tables{plans[0].table.get(),
                                                         plans[samePopulation ? 0u : 1u].table.get()};

        // Scratch: the uploads that are not written straight into a copy, each
        // population's changed bitset, list and staged chain, the grids, the
        // baseline pairs with a changed object, and the identity indexes of the
        // two sets an advance compares, each sized for the capacity.
        uint64_t   scratchBytes = 0u;
        bool       fits         = true;
        const auto take         = [&](uint64_t count, uint64_t size)
        {
            const uint64_t offset = scratchBytes;
            fits                  = fits && addBytes(count, size, scratchBytes);
            return offset;
        };
        for (Upload& upload : staging.uploads)
        {
            if (upload.destination == 0u)
            {
                upload.scratchOffset = take(upload.bytes.size(), 1u);
            }
        }
        const std::array<uint64_t, 2u> populationCounts{firstCount, samePopulation ? firstCount : secondCount};
        std::array<uint64_t, 2u>       bitsOffset{};
        std::array<uint64_t, 2u>       listOffset{};
        std::array<CuDevicePtr, 2u>    chainSource{};
        std::array<size_t, 2u>         chainUpload{SIZE_MAX, SIZE_MAX};
        for (size_t population = 0u; patched && population < planCount; ++population)
        {
            bitsOffset[population] = take((populationCounts[population] + 31u) / 32u, sizeof(uint32_t));
            listOffset[population] = take(chains[population].total(), sizeof(uint32_t));
            if (chains[population].records == 1u && chains[population].device.empty())
            {
                // The selected version's own record already sits on its copy.
                chainSource[population] = tables[population]->recordPositions;
            }
            else if (!chains[population].host.empty())
            {
                // Records of versions in between travel with the advance; device
                // lists are read where the versions keep them.
                chainUpload[population] =
                    staging.stage(chains[population].host.data(), chains[population].host.size() * sizeof(uint32_t));
                staging.uploads[chainUpload[population]].scratchOffset =
                    take(chains[population].host.size(), sizeof(uint32_t));
            }
        }
        // Grids over the build side of every discovery pass.
        const bool           firstChanges  = !chains[0].empty();
        const bool           secondChanges = samePopulation ? firstChanges : !chains[1].empty();
        std::array<bool, 2u> grids{};
        if (identical)
        {
            grids[0] = !patched || firstChanges;
        }
        else
        {
            grids[1] = !patched || firstChanges;
            grids[0] = patched && secondChanges;
        }
        const std::array<uint32_t, 2u> endpointCounts{static_cast<uint32_t>(firstCount),
                                                      static_cast<uint32_t>(secondCount)};
        std::array<uint64_t, 2u>       headsOffset{};
        std::array<uint64_t, 2u>       nextOffset{};
        std::array<uint64_t, 2u>       keysOffset{};
        std::array<uint64_t, 2u>       oversizeOffset{};
        std::array<uint32_t, 2u>       buckets{};
        for (size_t side = 0u; side < 2u; ++side)
        {
            if (grids[side])
            {
                buckets[side]        = bucketSlots(endpointCounts[side]);
                headsOffset[side]    = take(buckets[side], sizeof(uint32_t));
                nextOffset[side]     = take(endpointCounts[side], sizeof(uint32_t));
                keysOffset[side]     = take(endpointCounts[side], 3u * sizeof(int32_t));
                oversizeOffset[side] = take(endpointCounts[side], sizeof(uint32_t));
            }
        }
        const uint64_t                 capacityBound = std::max<uint64_t>(impl.storage.activePairCapacity, 1u);
        const uint32_t                 slotsPerIndex = hashSlots(capacityBound);
        const uint64_t                 candidates    = patched ? take(capacityBound, sizeof(uint32_t)) : 0u;
        const std::array<uint64_t, 2u> hashOffset{take(slotsPerIndex, sizeof(uint32_t)),
                                                  take(slotsPerIndex, sizeof(uint32_t))};
        uint64_t                       stagingBytes = 0u;
        for (const Upload& upload : staging.uploads)
        {
            fits = fits && addBytes(upload.bytes.size(), 1u, stagingBytes);
        }
        if (!fits)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        ovspatialquery_api_status_t reserved =
            impl.reserveScratch(instance, std::max<uint64_t>(scratchBytes, 1u), advance->scratch);
        if (reserved == OVSPATIALQUERY_API_SUCCESS)
        {
            reserved = impl.acquireStaging(hostLedger, std::max<uint64_t>(stagingBytes, 1u), advance->staging);
        }
        if (reserved != OVSPATIALQUERY_API_SUCCESS)
        {
            return reserved;
        }
        const CuDevicePtr scratchBase = advance->scratch->pointer;
        unsigned char*    stagingBase = static_cast<unsigned char*>(advance->staging->pointer);
        for (size_t population = 0u; population < planCount; ++population)
        {
            if (chainUpload[population] != SIZE_MAX)
            {
                chainSource[population] = scratchBase + staging.uploads[chainUpload[population]].scratchOffset;
            }
        }
        // The endpoints as the kernels read them, fixed while the copies are locked.
        const auto side = [&](size_t endpoint)
        {
            const VersionTable& table = *tables[endpoint];
            cudaPairs::Side     value{};
            value.objectCount       = endpointCounts[endpoint];
            value.ids               = devicePointer<const unsigned long long>(table.ids);
            value.bits              = devicePointer<const unsigned long long>(table.bits);
            value.minimums          = devicePointer<const double>(table.minimums);
            value.maximums          = devicePointer<const double>(table.maximums);
            const SelectionKey& key = endpoint == 0u ? config.firstSelection : config.secondSelection;
            value.membership = hasSelection(key) ? devicePointer<const unsigned char>(table.membership(key)) : nullptr;
            const ParticipationBits& queryBits = endpoint == 0u ? config.firstQueryBits : config.secondQueryBits;
            value.queryBits[0]                 = queryBits.words[0];
            value.queryBits[1]                 = queryBits.words[1];
            return value;
        };
        const std::array<cudaPairs::Side, 2u> sides{side(0u), side(1u)};

        if (takeCudaFailurePoint(kCudaFailurePairReporterAdmission))
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        // Every container the bookkeeping after enqueue grows is sized now,
        // so nothing after the first enqueue can throw.
        advance->tables.reserve(2u * planCount);
        for (size_t index = 0u; index < planCount; ++index)
        {
            CopyPlan& plan = plans[index];
            plan.table->budgets.reserve(plan.table->budgets.size() + plan.leases.size());
            if (plan.build != CopyPlan::Build::existing)
            {
                plan.attachments->entries.reserve(plan.attachments->entries.size() + 1u);
            }
        }
        // Enqueue. From here a failure completes the advance as failed and
        // leaves the baseline; nothing enqueued is abandoned.
        rollback.armed        = false;
        const CuStream stream = advance->stream;
        CuResult       status = kCuSuccess;
        // Each call runs only while every earlier one succeeded; uploads,
        // copies and kernels count as enqueued work, waits and records do not.
        const auto issue = [&](auto&& call, bool work = true)
        {
            if (status == kCuSuccess)
            {
                status                = impl.device->observe(call());
                advance->workEnqueued = advance->workEnqueued || (work && status == kCuSuccess);
            }
        };
        // Arguments pass by value with the kernel's exact parameter types.
        const auto run = [&](Function function, uint64_t count, bool gridStride, auto... arguments)
        {
            issue(
                [&]()
                {
                    void* parameters[] = {&arguments...};
                    return impl.device->launch(function, count, parameters, stream, gridStride);
                });
        };
        // The caller's input ordering (ADR-0022 decision 1) comes before anything
        // else, then the advance admitted before this one: pending advances
        // share the counters and the scratch, and compare in admission order.
        if (sync.waitEvent != 0u)
        {
            issue([&]() { return driver.streamWaitEvent(stream, reinterpret_cast<CuEvent>(sync.waitEvent), 0u); },
                  false);
        }
        if (impl.lastSlot != UINT32_MAX)
        {
            issue([&]() { return driver.streamWaitEvent(stream, impl.slots[impl.lastSlot].ready, 0u); }, false);
        }
        uint64_t stagingOffset = 0u;
        for (const Upload& upload : staging.uploads)
        {
            if (status != kCuSuccess || upload.bytes.empty())
            {
                continue;
            }
            unsigned char* host = stagingBase + stagingOffset;
            std::memcpy(host, upload.bytes.data(), upload.bytes.size());
            const CuDevicePtr target =
                upload.destination != 0u ? upload.destination : scratchBase + upload.scratchOffset;
            issue([&]() { return driver.copyHtoDAsync(target, host, upload.bytes.size(), stream); });
            stagingOffset += upload.bytes.size();
        }
        // Copies written elsewhere, or read as a source, are waited for.
        for (size_t index = 0u; index < planCount; ++index)
        {
            const VersionTable&   table = *plans[index].table;
            const CopyPlan::Build build = plans[index].build;
            const auto            wait  = [&](CuEvent event)
            {
                if (event != nullptr)
                {
                    issue([&]() { return driver.streamWaitEvent(stream, event, 0u); }, false);
                }
            };
            wait(table.borrowedReady);
            wait((build == CopyPlan::Build::existing || build == CopyPlan::Build::stolen) ? table.ready : nullptr);
            wait(build == CopyPlan::Build::derived ? plans[index].previous->ready : nullptr);
        }
        for (const TableWrite& write : staging.writes)
        {
            if (write.kind == TableWrite::Kind::copy)
            {
                issue([&]() { return driver.copyDtoDAsync(write.destination, write.source, write.copyBytes, stream); });
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
                noteCudaDeviceToDevice(write.copyBytes);
#endif
                continue;
            }
            const unsigned int count = write.count;
            const CuDevicePtr  positions =
                write.positions == SIZE_MAX ? 0u : scratchBase + staging.uploads[write.positions].scratchOffset;
            const CuDevicePtr values = scratchBase + staging.uploads[write.values].scratchOffset;
            if (write.kind == TableWrite::Kind::rows)
            {
                run(fnScatterRows, count, true, count, positions, values, write.table->ids, write.table->bits,
                    write.table->minimums, write.table->maximums);
            }
            else
            {
                run(fnScatterFlags, count, true, count, positions, values, write.destination);
            }
        }
        for (size_t index = 0u; index < planCount; ++index)
        {
            const CopyPlan& plan = plans[index];
            if (plan.table->ready != nullptr &&
                (plan.build != CopyPlan::Build::existing || !staging.writes.empty() || !staging.uploads.empty()))
            {
                issue([&]() { return driver.eventRecord(plan.table->ready, stream); }, false);
            }
        }
        // Each population now keeps only the copy this advance selected, and
        // the copies admitted advances hold (ADR-0019 decision 7).
        for (size_t index = 0u; index < planCount; ++index)
        {
            CopyPlan& plan = plans[index];
#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
            gCopyCounts[static_cast<size_t>(plan.build)].fetch_add(1u, std::memory_order_relaxed);
#endif
            if (plan.previousAttachments != nullptr)
            {
                detachTable(*plan.previousAttachments, deviceId);
            }
            if (status != kCuSuccess)
            {
                // Memberships whose upload may not have been enqueued.
                removeAddedMemberships(plan);
            }
            if (status == kCuSuccess)
            {
                for (std::shared_ptr<void>& lease : plan.leases)
                {
                    plan.table->budgets.push_back(std::move(lease));
                }
                plan.table->charged = plan.chargedBytes;
                if (plan.build != CopyPlan::Build::existing)
                {
                    plan.attachments->entries.emplace_back(deviceId, plan.table);
                }
                populations.setLastDeviceAttachments(plan.population, deviceId, plan.attachments);
            }
            advance->tables.push_back(plan.table);
            if (plan.build == CopyPlan::Build::derived)
            {
                advance->tables.push_back(plan.previous);
            }
        }
        copies.unlock();

        // The advance proper, on the same stream.
        Impl::ResultSlot&         slot          = impl.slots[resultSlot];
        const Impl::Storage&      storage       = impl.storage;
        const CuDevicePtr         control       = impl.control;
        const unsigned long long  capacity      = storage.activePairCapacity;
        const unsigned long long  delta         = storage.deltaPairCapacity;
        const CuDevicePtr         next          = storage.buffer(writeBuffer);
        const unsigned long long  ringStride    = storage.entryStride;
        const cudaPairs::Entry*   ring          = devicePointer<const cudaPairs::Entry>(storage.block);
        const unsigned long long* ringCounts    = devicePointer<const unsigned long long>(impl.ringCounts);
        const CuDevicePtr         candidateList = patched ? scratchBase + candidates : 0u;
        const CuDevicePtr         acceptedHash  = scratchBase + hashOffset[0];
        const CuDevicePtr         nextHash      = scratchBase + hashOffset[1];
        const auto                field         = [control](size_t offset) { return control + offset; };
        // The counters after the accepted counts start from zero.
        constexpr size_t kCleared = offsetof(cudaPairs::Control, nextCount);
        issue([&]()
              { return driver.memsetD8Async(control + kCleared, 0u, sizeof(cudaPairs::Control) - kCleared, stream); });
        std::array<CuDevicePtr, 2u> changedBits{};
        std::array<CuDevicePtr, 2u> changedLists{};
        for (size_t population = 0u; patched && population < planCount; ++population)
        {
            const unsigned int objects = static_cast<unsigned int>(populationCounts[population]);
            changedBits[population]    = scratchBase + bitsOffset[population];
            changedLists[population]   = scratchBase + listOffset[population];
            issue([&]() { return driver.memsetD32Async(changedBits[population], 0u, (objects + 31u) / 32u, stream); });
            const CuDevicePtr changedCount =
                field(offsetof(cudaPairs::Control, changedCount) + population * sizeof(unsigned int));
            if (!chains[population].host.empty())
            {
                const unsigned int count = static_cast<unsigned int>(chains[population].host.size());
                run(fnMark, count, true, count, chainSource[population], objects, changedBits[population],
                    changedLists[population], changedCount);
            }
            for (const std::pair<CuDevicePtr, uint32_t>& segment : chains[population].device)
            {
                const unsigned int count = segment.second;
                run(fnMark, count, true, count, segment.first, objects, changedBits[population],
                    changedLists[population], changedCount);
            }
        }
        if (samePopulation)
        {
            changedBits[1]  = changedBits[0];
            changedLists[1] = changedLists[0];
        }
        const double tolerance = config.contactTolerance;
        for (size_t endpoint = 0u; endpoint < 2u; ++endpoint)
        {
            if (!grids[endpoint])
            {
                continue;
            }
            const cudaPairs::Side value = sides[endpoint];
            const CuDevicePtr     grid = field(offsetof(cudaPairs::Control, grid) + endpoint * sizeof(cudaPairs::Grid));
            const CuDevicePtr     heads = scratchBase + headsOffset[endpoint];
            issue([&]() { return driver.memsetD32Async(heads, cudaPairs::kEmpty, buckets[endpoint], stream); });
            run(fnGridStats, value.objectCount, true, value, grid);
            if (value.objectCount != 0u)
            {
                run(fnGridLink, value.objectCount, false, value, grid, tolerance, buckets[endpoint] - 1u, heads,
                    scratchBase + nextOffset[endpoint], scratchBase + keysOffset[endpoint],
                    scratchBase + oversizeOffset[endpoint]);
            }
        }
        if (patched)
        {
            run(fnCarry, capacity, true, ring, ringStride, ringCounts, changedBits[0], changedBits[1], next, capacity,
                candidateList, control);
        }
        // Discovery: every first object against the second endpoint when
        // recomputing, the changed objects of each role when patching.
        struct Pass
        {
            size_t query;
            size_t build;
            bool   queryIsFirst;
        };
        std::array<Pass, 2u> passes{};
        size_t               passCount = 0u;
        if (identical)
        {
            if (grids[0])
            {
                passes[passCount++] = {0u, 0u, true};
            }
        }
        else
        {
            if (grids[1])
            {
                passes[passCount++] = {0u, 1u, true};
            }
            if (grids[0])
            {
                passes[passCount++] = {1u, 0u, false};
            }
        }
        for (size_t index = 0u; index < passCount; ++index)
        {
            const Pass&          pass            = passes[index];
            const size_t         queryPopulation = samePopulation ? 0u : pass.query;
            cudaPairs::Discovery discovery{};
            discovery.query     = sides[pass.query];
            discovery.build     = sides[pass.build];
            discovery.queryList = patched ? devicePointer<const unsigned int>(changedLists[queryPopulation]) : nullptr;
            discovery.queryListCount =
                patched ? devicePointer<const unsigned int>(field(offsetof(cudaPairs::Control, changedCount) +
                                                                  queryPopulation * sizeof(unsigned int)))
                        : nullptr;
            discovery.queryCount =
                patched ? static_cast<unsigned int>(chains[queryPopulation].total()) : endpointCounts[pass.query];
            discovery.relation =
                identical ? cudaPairs::kIdenticalSelection
                          : (samePopulation ? cudaPairs::kDirectedSamePopulation : cudaPairs::kDistinctPopulations);
            discovery.queryIsFirst = pass.queryIsFirst ? 1u : 0u;
            discovery.firstChanged = patched ? devicePointer<const unsigned int>(changedBits[0]) : nullptr;
            discovery.tolerance    = config.contactTolerance;
            discovery.grid         = devicePointer<const cudaPairs::Grid>(
                field(offsetof(cudaPairs::Control, grid) + pass.build * sizeof(cudaPairs::Grid)));
            discovery.heads      = devicePointer<const unsigned int>(scratchBase + headsOffset[pass.build]);
            discovery.next       = devicePointer<const unsigned int>(scratchBase + nextOffset[pass.build]);
            discovery.cellKeys   = devicePointer<const int>(scratchBase + keysOffset[pass.build]);
            discovery.oversize   = devicePointer<const unsigned int>(scratchBase + oversizeOffset[pass.build]);
            discovery.bucketMask = buckets[pass.build] - 1u;
            discovery.entries    = devicePointer<cudaPairs::Entry>(next);
            discovery.capacity   = capacity;
            discovery.control    = devicePointer<cudaPairs::Control>(control);
            if (discovery.queryCount != 0u)
            {
                // One warp per query object.
                run(fnDiscover, uint64_t{discovery.queryCount} * cudaPairs::kWarpSize, false, discovery);
            }
        }
        // Compare the baseline pairs with a changed object and the pairs
        // discovered for changed objects, or both whole sets when recomputing.
        cudaPairs::Sets sets{};
        sets.next       = devicePointer<const cudaPairs::Entry>(next);
        sets.ring       = ring;
        sets.ringStride = ringStride;
        sets.indices    = devicePointer<const unsigned int>(candidateList);
        sets.hash[0]    = devicePointer<unsigned int>(acceptedHash);
        sets.hash[1]    = devicePointer<unsigned int>(nextHash);
        run(fnClear, slotsPerIndex, true, sets, control, ringCounts, capacity, patched ? 1u : 0u, slotsPerIndex - 1u);
        run(fnIndex, capacity, true, sets, control);
        run(fnDifference, capacity, true, sets, slot.entered, slot.exited, delta, control);
        // An endpoint admitted before its publication completed may have failed
        // on the device; its status word decides the outcome (ADR-0022 decision 3).
        run(fnFinish, 1u, true, control, devicePointer<unsigned long long>(impl.ringCounts), writeBuffer, capacity,
            delta, slot.header, devicePointer<const unsigned int>(tables[0]->borrowedFailed),
            devicePointer<const unsigned int>(tables[1]->borrowedFailed), 0u);
        if (status == kCuSuccess && takeCudaFailurePoint(kCudaFailurePairReporterLaunch))
        {
            status = impl.device->observe(kCuDeviceLost);
        }
        void* hostHeader = static_cast<cudaPairs::Header*>(impl.hostHeaders) + resultSlot;
        issue([&]() { return driver.copyDtoHAsync(hostHeader, slot.header, sizeof(cudaPairs::Header), stream); });
        issue([&]() { return driver.eventRecord(slot.ready, stream); }, false);
        advance->readyRecorded = status == kCuSuccess;
        // Without the wake event the worker polls `ready` instead.
        advance->wakeRecorded = advance->readyRecorded && driver.eventRecord(slot.wake, stream) == kCuSuccess;
        if (status != kCuSuccess)
        {
            if (!advance->workEnqueued)
            {
                return deviceStatus(status);
            }
            // Enqueued work completes as a failed advance: the baseline stays.
            advance->failed        = true;
            advance->lost          = status == kCuDeviceLost;
            advance->readyRecorded = !advance->lost && impl.failOnDevice(*advance);
        }
        impl.ringOwners[writeBuffer] = advance;
        impl.lastSlot                = resultSlot;
        outAdvance                   = std::move(advance);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (...)
    {
        // Work already enqueued is never abandoned: it completes as a failed
        // advance, so the arenas it reads are not reused before it finishes.
        if (advance != nullptr && advance->workEnqueued)
        {
            advance->failed = true;
            CurrentContext scope(*impl.device);
            advance->readyRecorded       = scope.status() == kCuSuccess && impl.failOnDevice(*advance);
            impl.ringOwners[writeBuffer] = advance;
            impl.lastSlot                = resultSlot;
            outAdvance                   = std::move(advance);
            return OVSPATIALQUERY_API_SUCCESS;
        }
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

CudaCompletionState CudaPairReporterDevice::resultState(const CudaPairAdvance& advance) noexcept
{
    std::lock_guard<std::mutex> lock(advance.mutex);
    if (advance.lost || advance.device->lost.load(std::memory_order_relaxed))
    {
        advance.lost = true;
        return CudaCompletionState::deviceLost;
    }
    if (!advance.workEnqueued)
    {
        return advance.failed ? CudaCompletionState::error : CudaCompletionState::complete;
    }
    if (advance.readyState != CudaCompletionState::pending)
    {
        return advance.readyState;
    }
    CurrentContext scope(*advance.device);
    if (scope.status() != kCuSuccess)
    {
        return completionState(scope.status());
    }
    const Impl& impl    = *advance.reporter->mImpl;
    CuResult    queried = advance.readyRecorded ? advance.device->driver.eventQuery(impl.slots[advance.slot].ready)
                                                : advance.device->driver.streamQuery(advance.stream);
    if (queried == kCuSuccess && takeCudaFailurePoint(kCudaFailurePairReporterComplete))
    {
        queried = kCuDeviceLost;
    }
    const CudaCompletionState state = completionState(advance.device->observe(queried), advance.failed);
    if (state == CudaCompletionState::deviceLost)
    {
        advance.lost = true;
    }
    if (state != CudaCompletionState::pending)
    {
        advance.readyState = state;
    }
    return state;
}

void CudaPairReporterDevice::waitForResult(const CudaPairAdvance& advance) noexcept
{
    {
        std::lock_guard<std::mutex> lock(advance.mutex);
        if (!advance.wakeRecorded || advance.lost || advance.readyState != CudaCompletionState::pending)
        {
            return;
        }
    }
    // Outside the mutex; a failure here is left to resultState() to observe,
    // and device loss from the context switch is also recorded for the device.
    CurrentContext scope(*advance.device);
    if (scope.status() == kCuSuccess)
    {
        (void)advance.device->driver.eventSynchronize(advance.reporter->mImpl->slots[advance.slot].wake);
    }
}

bool CudaPairReporterDevice::result(const CudaPairAdvance& advance, CudaPairAdvanceResult& outResult) noexcept
{
    const Impl&             impl   = *advance.reporter->mImpl;
    const cudaPairs::Header header = static_cast<const cudaPairs::Header*>(impl.hostHeaders)[advance.slot];
    if (header.outcome > cudaPairs::kOutcomeFailed)
    {
        return false;
    }
    outResult.outcome       = static_cast<uint32_t>(header.outcome);
    outResult.enteredCount  = header.enteredCount;
    outResult.exitedCount   = header.exitedCount;
    outResult.acceptedCount = header.acceptedCount;
    outResult.patched       = advance.patched;
    return true;
}

void CudaPairReporterDevice::releaseReadyDependencies(CudaPairAdvance& advance) noexcept
{
    const CudaCompletionState state = resultState(advance);
    if (state == CudaCompletionState::pending)
    {
        return;
    }
    std::vector<std::shared_ptr<VersionTable>>                tables;
    std::vector<std::shared_ptr<const CudaPopulationVersion>> heldVersions;
    std::shared_ptr<DeviceArena>                              scratch;
    std::shared_ptr<HostArena>                                staging;
    {
        std::lock_guard<std::mutex> lock(advance.mutex);
        tables.swap(advance.tables);
        heldVersions.swap(advance.heldVersions);
        scratch.swap(advance.scratch);
        staging.swap(advance.staging);
        if (advance.consumerRecorded && advance.consumerState == CudaCompletionState::pending &&
            state != CudaCompletionState::deviceLost)
        {
            CurrentContext scope(*advance.device);
            if (scope.status() == kCuSuccess)
            {
                const Impl& impl      = *advance.reporter->mImpl;
                advance.consumerState = completionState(
                    advance.device->observe(advance.device->driver.eventQuery(impl.slots[advance.slot].consumer)));
            }
        }
        if (advance.ownedConsumerDone != nullptr &&
            (advance.consumerState != CudaCompletionState::pending || state == CudaCompletionState::deviceLost))
        {
            (void)releaseTransferredConsumerDone(
                advance.device->driver, advance.device->context, advance.ownedConsumerDone,
                state == CudaCompletionState::deviceLost || advance.consumerState == CudaCompletionState::deviceLost);
        }
    }
    tables.clear();
    heldVersions.clear();
    scratch.reset();
    staging.reset();
}

void CudaPairReporterDevice::ringWhenConsumed(const CudaPairAdvance&               advance,
                                              const std::shared_ptr<DeviceSignal>& signal) noexcept
{
    {
        std::lock_guard<std::mutex> lock(advance.mutex);
        if (!advance.consumerRecorded || advance.lost)
        {
            return;
        }
    }
    CurrentContext scope(*advance.device);
    if (scope.status() == kCuSuccess)
    {
        const Impl& impl = *advance.reporter->mImpl;
        (void)impl.slots[advance.slot].ringWhenConsumed(advance.device->driver, impl.signalRings, signal);
    }
}

ovspatialquery_api_status_t CudaPairReporterDevice::transferConsumerDone(CudaPairAdvance&            advance,
                                                                         ovspatialquery_cuda_event_t event) noexcept
{
    std::lock_guard<std::mutex>       lock(advance.mutex);
    const ovspatialquery_api_status_t status = advance.reporter->mImpl->slots[advance.slot].transferConsumerDone(
        advance.device->driver, advance.device->context, advance.device->lost, event, advance.lost,
        advance.consumerRecorded);
    if (status == OVSPATIALQUERY_API_NOT_SUPPORTED)
    {
        advance.lost = true;
    }
    else if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        advance.ownedConsumerDone = reinterpret_cast<CuEvent>(event.value);
        advance.consumerRecorded  = true;
        advance.consumerState     = CudaCompletionState::pending;
    }
    return status;
}

bool CudaPairReporterDevice::deviceSafe(CudaPairAdvance& advance) noexcept
{
    if (resultState(advance) == CudaCompletionState::pending)
    {
        return false;
    }
    releaseReadyDependencies(advance);
    std::lock_guard<std::mutex> lock(advance.mutex);
    return advance.lost || ((!advance.consumerRecorded || advance.consumerState != CudaCompletionState::pending) &&
                            advance.ownedConsumerDone == nullptr);
}

bool CudaPairReporterDevice::deviceLost(const CudaPairAdvance& advance) noexcept
{
    return resultState(advance) == CudaCompletionState::deviceLost;
}

#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
std::array<uint64_t, 5u> CudaPairReporterDevice::copyCountsForTest() noexcept
{
    std::array<uint64_t, 5u> counts{};
    for (size_t index = 0u; index < counts.size(); ++index)
    {
        counts[index] = gCopyCounts[index].load(std::memory_order_relaxed);
    }
    return counts;
}

bool CudaPairReporterDevice::downloadForTest(uint32_t resultSlot, uint64_t enteredCount, uint64_t exitedCount,
                                             std::vector<ovspatialquery_pair_t>& outEntered,
                                             std::vector<ovspatialquery_pair_t>& outExited) noexcept
{
    try
    {
        if (resultSlot >= mImpl->slots.size() || enteredCount > mImpl->storage.deltaPairCapacity ||
            exitedCount > mImpl->storage.deltaPairCapacity)
        {
            return false;
        }
        CurrentContext scope(*mImpl->device);
        if (scope.status() != kCuSuccess)
        {
            return false;
        }
        std::vector<uint64_t> entered(2u * enteredCount);
        std::vector<uint64_t> exited(2u * exitedCount);
        if ((enteredCount != 0u && mImpl->device->driver.copyDtoH(entered.data(), mImpl->slots[resultSlot].entered,
                                                                  entered.size() * sizeof(uint64_t)) != kCuSuccess) ||
            (exitedCount != 0u && mImpl->device->driver.copyDtoH(exited.data(), mImpl->slots[resultSlot].exited,
                                                                 exited.size() * sizeof(uint64_t)) != kCuSuccess))
        {
            return false;
        }
        outEntered.clear();
        outExited.clear();
        for (uint64_t row = 0u; row < enteredCount; ++row)
        {
            outEntered.push_back({{entered[2u * row]}, {entered[2u * row + 1u]}});
        }
        for (uint64_t row = 0u; row < exitedCount; ++row)
        {
            outExited.push_back({{exited[2u * row]}, {exited[2u * row + 1u]}});
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}
#endif

#else

class CudaPairAdvance
{
};

struct CudaPairReporterDevice::Impl
{
};

CudaPairReporterDevice::CudaPairReporterDevice(std::unique_ptr<Impl> impl) noexcept : mImpl(std::move(impl)) {}
CudaPairReporterDevice::~CudaPairReporterDevice() = default;

bool     CudaPairReporterDevice::residentDeviceBytes(uint64_t, uint64_t, uint32_t, uint64_t&) noexcept { return false; }
uint64_t CudaPairReporterDevice::residentHostBytes(uint32_t) noexcept { return 0u; }

ovspatialquery_api_status_t CudaPairReporterDevice::create(const CudaPairReporterConfig&,
                                                           std::shared_ptr<CudaPairReporterDevice>&) noexcept
{
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
}

ovspatialquery_api_status_t CudaPairReporterDevice::launch(ovspatialquery_instance_handle_t, const PopulationTable&,
                                                           InstanceHostLedger*, uint32_t, const ResolvedTarget&,
                                                           const ResolvedTarget&,
                                                           const std::optional<CudaPairBaseline>&,
                                                           const CudaPairAdvanceSync&,
                                                           std::shared_ptr<CudaPairAdvance>&) noexcept
{
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
}

void                        CudaPairReporterDevice::accept(const CudaPairAdvance&) noexcept {}
void                        CudaPairReporterDevice::release(const CudaPairAdvance&) noexcept {}
ovspatialquery_api_status_t CudaPairReporterDevice::grow(uint64_t, uint64_t) noexcept
{
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
}
ovspatialquery_cuda_event_t CudaPairReporterDevice::readyEvent(uint32_t) const noexcept { return {}; }
const void*                 CudaPairReporterDevice::enteredRows(uint32_t) const noexcept { return nullptr; }
const void*                 CudaPairReporterDevice::exitedRows(uint32_t) const noexcept { return nullptr; }
const void*                 CudaPairReporterDevice::countsRows(uint32_t) const noexcept { return nullptr; }
CudaCompletionState         CudaPairReporterDevice::resultState(const CudaPairAdvance&) noexcept
{
    return CudaCompletionState::error;
}
void CudaPairReporterDevice::waitForResult(const CudaPairAdvance&) noexcept {}
void CudaPairReporterDevice::ringWhenConsumed(const CudaPairAdvance&, const std::shared_ptr<DeviceSignal>&) noexcept {}
bool CudaPairReporterDevice::result(const CudaPairAdvance&, CudaPairAdvanceResult&) noexcept { return false; }
void CudaPairReporterDevice::releaseReadyDependencies(CudaPairAdvance&) noexcept {}
ovspatialquery_api_status_t CudaPairReporterDevice::transferConsumerDone(CudaPairAdvance&,
                                                                         ovspatialquery_cuda_event_t) noexcept
{
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
}
bool                     CudaPairReporterDevice::deviceSafe(CudaPairAdvance&) noexcept { return true; }
bool                     CudaPairReporterDevice::deviceLost(const CudaPairAdvance&) noexcept { return false; }
#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
std::array<uint64_t, 5u> CudaPairReporterDevice::copyCountsForTest() noexcept { return {}; }

bool CudaPairReporterDevice::downloadForTest(uint32_t, uint64_t, uint64_t, std::vector<ovspatialquery_pair_t>&,
                                             std::vector<ovspatialquery_pair_t>&) noexcept
{
    return false;
}
#endif

#endif

} // namespace detail
} // namespace ovspatialquery
