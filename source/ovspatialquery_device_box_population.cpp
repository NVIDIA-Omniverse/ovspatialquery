// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ovspatialquery_device_box_population.h"
#include "ovspatialquery_cuda_driver.h"
#include "ovspatialquery_cuda_family_plan_types.h"
#include "ovspatialquery_ovstage_transform.h"
#include "ovspatialquery_typed_geometry.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <map>

namespace ovspatialquery
{
namespace detail
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
namespace
{
// Each version's table: 8 pose doubles, 3 minimums and 3 maximums per slot,
// then one aligned status word.
constexpr uint64_t kDoublesPerObject = 14u; // 8 pose, 3 minimum, 3 maximum
constexpr uint64_t kStatusBytes      = 256u;
// Per slot in a version: the doubles above, then two participation-bits words.
constexpr uint64_t kVersionBytesPerObject = kDoublesPerObject * sizeof(double) + 2u * sizeof(uint64_t);
constexpr uint32_t kNoDefinition          = 0xffffffffu;
// Pinned words the device copies each version's status into once it is prepared.
constexpr size_t kStatusPoolWords = 1024u;
// The most objects of a set left out of its grid cell size (they are listed as oversize).
constexpr size_t kGridOversizeObjects = 64u;
// Buckets: a power of two of at least twice the slots, so chains stay short.
uint64_t gridBuckets(uint64_t capacity)
{
    uint64_t buckets = 16u;
    while (buckets < 2u * capacity && buckets < (uint64_t{1} << 31))
        buckets *= 2u;
    return buckets;
}
// The grid header, the bucket heads, and per slot a chain link, three cell
// keys and an oversize entry.
uint64_t gridBytes(uint64_t capacity) { return 32u + gridBuckets(capacity) * 4u + capacity * 20u; }
struct SharedBoxRuntime
{
    Driver                             driver;
    DLDevice                           device{kDLCUDA, 0};
    CuContext                          context = nullptr;
    std::shared_ptr<CudaContextModule> contextModule;
    CuStream                           stream = nullptr;

    ~SharedBoxRuntime()
    {
        if (context == nullptr)
            return;
        CuContext previous = nullptr;
        driver.contextGetCurrent(&previous);
        driver.contextSetCurrent(context);
        if (stream != nullptr)
            driver.streamSynchronize(stream);
        if (stream != nullptr)
            driver.streamDestroy(stream);
        driver.contextSetCurrent(previous);
        driver.primaryRelease(device.device_id);
    }
};
struct BoxUploadStaging
{
    std::shared_ptr<void> lease;
    struct Rows
    {
        std::vector<double>   params;
        std::vector<uint32_t> definitions;
    };
    std::deque<Rows>                           rows;
    std::deque<cuda_family::DeviceShapeRecord> records;
};
struct SharedBoxStorage
{
    std::shared_ptr<SharedBoxRuntime>  runtime;
    std::vector<std::shared_ptr<void>> leases;
    std::shared_ptr<BoxUploadStaging>  staging;
    std::vector<CuDevicePtr>           allocations;
    bool                               ownsAllocations = false;

    ~SharedBoxStorage()
    {
        if (!ownsAllocations)
            return;
        CuContext previous = nullptr;
        runtime->driver.contextGetCurrent(&previous);
        runtime->driver.contextSetCurrent(runtime->context);
        for (CuDevicePtr allocation : allocations)
            if (allocation != 0u)
                runtime->driver.memFree(allocation);
        runtime->driver.contextSetCurrent(previous);
    }
};
} // namespace
namespace
{
// The distinct definitions a version's objects bind, each with the number of
// its slots that bind it. A POSES update shares its base's; a membership
// change counts from its base, so a definition leaves a version only when the
// version's own removals unbind its last slot, whatever failed in between.
using DefinitionUses = std::map<const void*, std::pair<std::shared_ptr<const void>, uint64_t>>;
} // namespace
struct DeviceBoxPopulation::Impl
{
    std::shared_ptr<SharedBoxRuntime> runtime;
    std::shared_ptr<SharedBoxStorage> bindingOwner;
    std::shared_ptr<SharedBoxStorage> sourceBindings;
    std::shared_ptr<BoxUploadStaging> uploadStaging;
    // Reservations outlive the allocations and registered definitions they charge.
    std::shared_ptr<void> hostLease, deviceLease;
    // The registered definition of every assigned slot, null once its object is
    // removed, and its address, kept after removal to find it in a version's
    // uses. A version owns the definitions its objects bind (DefinitionUses), so
    // a definition stays referenced exactly while a live object or a retained
    // version binds it.
    std::vector<std::shared_ptr<const void>> definitions;
    std::vector<const void*>                 definitionKeys;
    // Whether each slot's definition still exists: once it is destroyed and
    // freed, another definition may take its address (definitionKeys).
    std::vector<std::weak_ptr<const void>> definitionWatches;
    Driver                                 driver;
    DLDevice                               device{kDLCUDA, 0};
    CuContext                              context = nullptr;
    // The device's shared module (acquireCudaContextModule) the kernels below
    // belong to; the population, a fork and the shared runtime each hold it.
    std::shared_ptr<CudaContextModule> contextModule;
    CuModule                           module    = nullptr;
    CuFunction                         kernel    = nullptr;
    CuFunction                         tombstone = nullptr;
    CuFunction                         gridder   = nullptr;
    // The grid every version builds (W4): one cell width for the population's
    // life, from the half extents of the first set, and a bucket count from the
    // capacity.
    double      inverseCell = 1.0;
    uint32_t    bucketMask  = 0u;
    CuStream    stream      = nullptr;
    CuDevicePtr bindings    = 0;
    uint64_t    capacity    = 0;
    uint64_t    assigned    = 0;
    uint64_t    live        = 0;
    bool        lost        = false;
    // Capacity-sized arrays in one immutable allocation; a slot's row is
    // written once when the slot is assigned.
    CuDevicePtr ids = 0, refs = 0, params = 0, types = 0, defs = 0, sortedIds = 0, sortedSlots = 0;
    // The definitions slots reference (convex hulls and meshes): one record
    // each on the device, grown by doubling, hull vertices in their own
    // allocations, and the hulls kept alive. Retired record buffers wait for
    // the population's end, since a launch may still read them.
    struct DefinitionPool
    {
        CuDevicePtr                                            records      = 0;
        uint64_t                                               capacity     = 0;
        bool                                                   recordsOwned = true;
        std::map<std::pair<int32_t, uint64_t>, uint32_t>       index;
        std::vector<CuDevicePtr>                               allocations;
        std::vector<std::shared_ptr<const PreparedConvexHull>> hulls;
        std::vector<std::shared_ptr<void>>                     leases;
        std::vector<std::shared_ptr<SharedBoxStorage>>         backing;
    } pool;
    // Host identity: the ID, geometry value and type of every assigned slot,
    // the sorted index of live objects the kernels resolve through, and retired
    // IDs. The preparer builds a readable version's host objects.
    std::vector<uint64_t>                       slotIds;
    std::vector<uint64_t>                       slotRefs;
    std::vector<int32_t>                        slotTypes;
    std::shared_ptr<const DeviceObjectPreparer> preparer;
    std::vector<std::pair<uint64_t, uint32_t>>  index;
    std::vector<uint64_t>                       retired;
    std::vector<uint64_t>                       indexIds;   // upload staging
    std::vector<uint32_t>                       indexSlots; // upload staging
    std::mutex                                  membershipMutex;
    // Host status words, pinned when the driver allows it.
    uint32_t*             statusPool       = nullptr;
    bool                  statusPoolPinned = false;
    std::vector<uint16_t> freeStatus;
    // Version tables and scratch of released versions, reused by the next
    // launch: allocating or freeing device memory while the GPU is busy
    // synchronizes with it, which a publishing call must not do (ADR-0021).
    std::vector<CuDevicePtr> freeTables;
    std::vector<CuDevicePtr> freeScratch;
    std::mutex               poolMutex;
    // Waits for a version's ready event and rings a waiting worker
    // (ringWhenReady); created on first use.
    CuStream   signalStream = nullptr;
    std::mutex signalMutex;
    ~Impl()
    {
        if (!context)
            return;
        CuContext previous = nullptr;
        driver.contextGetCurrent(&previous);
        driver.contextSetCurrent(context);
        if (stream && runtime == nullptr)
            driver.streamSynchronize(stream);
        for (CuDevicePtr table : freeTables)
            driver.memFree(table);
        for (CuDevicePtr scratch : freeScratch)
            driver.memFree(scratch);
        for (CuDevicePtr allocation : pool.allocations)
            driver.memFree(allocation);
        if (pool.records && pool.recordsOwned)
            driver.memFree(pool.records);
        if (bindings && bindingOwner == nullptr)
            driver.memFree(bindings);
        bindingOwner.reset();
        sourceBindings.reset();
        if (stream && runtime == nullptr)
            driver.streamDestroy(stream);
        if (signalStream)
            driver.streamDestroy(signalStream);
        if (statusPool)
        {
            if (statusPoolPinned)
                driver.memFreeHost(statusPool);
            else
                delete[] statusPool;
        }
        driver.contextSetCurrent(previous);
        if (runtime == nullptr)
            driver.primaryRelease(device.device_id);
    }
    bool findLive(uint64_t id, uint32_t& outSlot) const noexcept
    {
        const auto at =
            std::lower_bound(index.begin(), index.end(), std::make_pair(id, 0u),
                             [](const std::pair<uint64_t, uint32_t>& entry, const std::pair<uint64_t, uint32_t>& key)
                             { return entry.first < key.first; });
        if (at == index.end() || at->first != id)
            return false;
        outSlot = at->second;
        return true;
    }
    bool isRetired(uint64_t id) const noexcept { return std::binary_search(retired.begin(), retired.end(), id); }
};
namespace
{
// Only the generation lease owns this version; it has no host object mirror.
// Its device work is enqueued at admission and observed later (ADR-0021).
class BoxDeviceVersion final : public CudaPopulationVersion
{
  public:
    std::shared_ptr<DeviceBoxPopulation::Impl> owner;
    CuDevicePtr                                values  = 0;
    CuDevicePtr                                scratch = 0;
    CuEvent                                    ready   = nullptr;
    ovspatialquery_ordinal_t                   ordinal = 0;
    // The shape-record buffer this version's preparation read. A later membership
    // change may replace the pool's buffer with a copy enqueued on the population's
    // stream; the replaced one stays allocated for the population's life.
    CuDevicePtr           records = 0;
    std::atomic<uint64_t> liveCount{0};
    bool                  ovstage = false;
    // The slots this version's rows rewrote, listed by the prepare kernel; known
    // complete only for a POSES or POSES_AND_BITS update of an existing version.
    // A row that left its slot as the base had it is listed as ~0u.
    uint64_t                                  changedCount  = 0;
    bool                                      changedKnown  = false;
    bool                                      changedPadded = false;   // rows were a whole snapshot
    bool                                      readable      = false;   // ON_DEVICE_HOST_READABLE
    uint32_t*                                 hostStatus    = nullptr; // pool word or own allocation
    int                                       statusSlot    = -1;
    std::shared_ptr<DevicePublicationOutcome> outcome       = std::make_shared<DevicePublicationOutcome>();
    std::shared_ptr<DevicePublicationOutcome> predecessor;
    // The definitions this version's objects bind, kept while it is retained,
    // and the slots assigned when it was launched.
    std::shared_ptr<const DefinitionUses> definitions;
    uint64_t                              assigned = 0u;
    mutable std::atomic<uint32_t>         observed{0u};       // 0 unknown, else CudaCompletionState + 1
    bool                                  ringQueued = false; // under the owner's signalMutex
    explicit BoxDeviceVersion(std::shared_ptr<DeviceBoxPopulation::Impl> state) : owner(std::move(state)) {}
    ~BoxDeviceVersion() override
    {
        // A version abandoned before its publication ended releases the chain
        // waiting on it rather than wedging the worker.
        outcome->finish(DevicePublicationOutcome::State::abandoned);
        auto&     d        = owner->driver;
        CuContext previous = nullptr;
        d.contextGetCurrent(&previous);
        d.contextSetCurrent(owner->context);
        if (ready)
        {
            d.eventSynchronize(ready);
            d.eventDestroy(ready);
        }
        d.contextSetCurrent(previous);
        std::lock_guard<std::mutex> lock(owner->poolMutex);
        if (scratch)
            owner->freeScratch.push_back(scratch);
        if (values)
            owner->freeTables.push_back(values);
        if (statusSlot >= 0)
            owner->freeStatus.push_back(static_cast<uint16_t>(statusSlot));
        else
            delete hostStatus;
    }
    CuDevicePtr bitsTable() const { return values + owner->capacity * kDoublesPerObject * sizeof(double); }
    CuDevicePtr statusWord() const { return values + owner->capacity * kVersionBytesPerObject; }
    // The grid follows the status word; the base copy stops before it. The
    // changed-slot list follows the grid.
    CuDevicePtr               gridHeader() const { return statusWord() + kStatusBytes; }
    CuDevicePtr               changedSlots() const { return gridHeader() + gridBytes(owner->capacity); }
    CuDevicePtr               boxExtents() const { return changedSlots() + owner->capacity * sizeof(uint32_t); }
    CudaPopulationDescription description() const override
    {
        CudaPopulationDescription d{};
        d.device             = owner->device;
        d.sourceOrdinal      = ordinal;
        d.objectCount        = owner->capacity;
        d.liveObjectCount    = liveCount;
        d.ready.value        = reinterpret_cast<uintptr_t>(ready);
        d.devicePrepared     = true;
        d.objectIds          = reinterpret_cast<void*>(owner->ids);
        d.objectBits         = reinterpret_cast<void*>(bitsTable());
        d.geometryTypes      = reinterpret_cast<void*>(owner->types);
        d.geometryValues     = reinterpret_cast<void*>(owner->refs);
        d.objectPoses        = reinterpret_cast<void*>(values);
        d.objectMinimums     = reinterpret_cast<void*>(values + owner->capacity * 8u * sizeof(double));
        d.objectMaximums     = reinterpret_cast<void*>(values + owner->capacity * 11u * sizeof(double));
        d.preparationFailed  = reinterpret_cast<void*>(statusWord());
        d.grid               = reinterpret_cast<void*>(gridHeader());
        d.changedSlots       = changedKnown ? reinterpret_cast<void*>(changedSlots()) : nullptr;
        d.changedSlotCount   = changedKnown ? changedCount : 0u;
        d.changedSlotsPadded = changedKnown && changedPadded;
        return d;
    }
    // Pending until the device recorded `ready`; then the status word says
    // whether validation failed. Terminal states are remembered.
    CudaCompletionState resultState() const override
    {
        const uint32_t known = observed.load(std::memory_order_acquire);
        if (known != 0u)
            return static_cast<CudaCompletionState>(known - 1u);
        auto&     d        = owner->driver;
        CuContext previous = nullptr;
        if (d.contextGetCurrent(&previous) != kCuSuccess || d.contextSetCurrent(owner->context) != kCuSuccess)
            return CudaCompletionState::invalid;
        const CuResult status = d.eventQuery(ready);
        d.contextSetCurrent(previous);
        if (status == kCuNotReady)
            return CudaCompletionState::pending;
        CudaCompletionState state = CudaCompletionState::error;
        if (status == kCuSuccess)
            state = *hostStatus != 0u ? CudaCompletionState::error : CudaCompletionState::complete;
        else if (status == kCuDeviceLost)
            state = CudaCompletionState::deviceLost;
        observed.store(static_cast<uint32_t>(state) + 1u, std::memory_order_release);
        return state;
    }
    void releaseReadyDependencies() override
    {
        std::lock_guard<std::mutex> lock(owner->membershipMutex);
        if (owner->bindingOwner != nullptr)
            owner->bindingOwner->staging.reset();
        owner->uploadStaging.reset();
        owner->sourceBindings.reset();
    }
    uint64_t deviceByteCharge() const override { return DeviceBoxPopulation::versionBytes(owner->capacity, ovstage); }
    bool     hostReadable() const noexcept override { return readable; }
    // An Ovstage version's primitives take their extents from the version, so
    // only an explicit one derives its exact objects from the slot history.
    bool exactMembership(CudaExactMembership& outMembership) const noexcept override
    {
        if (!readable || ovstage)
            return false;
        outMembership.membership  = definitions;
        outMembership.slots       = assigned;
        outMembership.params      = reinterpret_cast<const void*>(owner->params);
        outMembership.definitions = reinterpret_cast<const void*>(owner->defs);
        outMembership.records     = reinterpret_cast<const void*>(records);
        return true;
    }
    // Only the definitions this version's membership binds are prepared: they
    // stay registered while it is retained, and a slot whose definition left the
    // membership was removed, so it is absent from every version of it. The
    // census allocates nothing per slot; the per-slot form prepares nothing.
    ovspatialquery_api_status_t exactSlots(uint64_t count, bool perSlot, CudaExactSlots& outSlots) const override
    {
        if (!readable || ovstage || definitions == nullptr)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        auto&                              s = *owner;
        CudaExactSlots                     result;
        std::vector<ExplicitInstanceInput> instances;
        {
            std::lock_guard<std::mutex> lock(s.membershipMutex);
            if (count > s.slotTypes.size())
                return OVSPATIALQUERY_API_INTERNAL;
            if (perSlot)
            {
                result.types.assign(s.slotTypes.begin(), s.slotTypes.begin() + static_cast<std::ptrdiff_t>(count));
                result.definitionIndices.assign(static_cast<size_t>(count), UINT32_MAX);
            }
            std::map<std::pair<int32_t, uint64_t>, uint32_t> seen;
            for (size_t slot = 0; slot < count; ++slot)
            {
                const int32_t type = s.slotTypes[slot];
                if (type == OVSPATIALQUERY_GEOMETRY_SPHERE || type == OVSPATIALQUERY_GEOMETRY_BOX ||
                    type == OVSPATIALQUERY_GEOMETRY_CAPSULE)
                {
                    ++(type == OVSPATIALQUERY_GEOMETRY_SPHERE ? result.spheres
                       : type == OVSPATIALQUERY_GEOMETRY_BOX  ? result.boxes
                                                              : result.capsules);
                    continue;
                }
                if (type != OVSPATIALQUERY_GEOMETRY_CONVEX && type != OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH)
                    continue;
                // The map holds its definitions alive, so a slot whose own definition
                // was destroyed is absent even when another now has its address.
                if (s.definitionWatches[slot].expired() || definitions->count(s.definitionKeys[slot]) == 0u)
                {
                    if (perSlot)
                        result.definitionIndices[slot] = CudaExactSlots::kAbsent;
                    continue;
                }
                const auto placed =
                    seen.emplace(std::make_pair(type, s.slotRefs[slot]), static_cast<uint32_t>(seen.size()));
                if (placed.second && !perSlot)
                {
                    // Each definition once, at the identity, under its index as the ID.
                    ExplicitInstanceInput instance;
                    instance.objectValue = instances.size() + 1u;
                    instance.geometry    = {static_cast<ExactGeometryFamily>(type - kGeometryTypeFirstBuiltin),
                                            s.slotRefs[slot]};
                    instances.push_back(instance);
                    result.definitionSlots.push_back(0u);
                }
                if (perSlot)
                    result.definitionIndices[slot] = placed.first->second;
                else
                    ++result.definitionSlots[placed.first->second];
            }
        }
        if (!perSlot)
        {
            std::vector<PopulationObject> objects;
            const auto                    prepared = s.preparer->prepareObjects(instances, objects);
            if (prepared != OVSPATIALQUERY_API_SUCCESS)
                return prepared;
            if (objects.size() != instances.size())
                return OVSPATIALQUERY_API_INTERNAL;
            result.definitions.reserve(objects.size());
            for (PopulationObject& object : objects)
                result.definitions.push_back({std::move(object.exactGeometry), std::move(object.faceMaterialIds)});
        }
        outSlots = std::move(result);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    // The completed version's live slots as host objects: poses, bits and the
    // tombstone test come from this version's own table, so a retained version
    // answers with the objects it was published with; identities and bindings
    // from the slot history, which is append-only.
    ovspatialquery_api_status_t hostObjects(std::vector<PopulationObject>& outObjects) const override
    {
        if (!readable)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        auto&                 s        = *owner;
        auto&                 d        = s.driver;
        const uint64_t        capacity = s.capacity;
        std::vector<double>   poses(static_cast<size_t>(capacity * 8u));
        std::vector<double>   minimums(static_cast<size_t>(capacity * 3u));
        std::vector<uint64_t> bits(static_cast<size_t>(capacity * 2u));
        std::vector<double>   extents(ovstage ? static_cast<size_t>(capacity * 3u) : 0u);
        CuContext             previous = nullptr;
        d.contextGetCurrent(&previous);
        if (d.contextSetCurrent(s.context) != kCuSuccess)
            return OVSPATIALQUERY_API_INTERNAL;
        CuResult status = d.eventSynchronize(ready);
        if (status == kCuSuccess && capacity != 0u)
            status = d.copyDtoH(poses.data(), values, static_cast<size_t>(capacity * 8u * sizeof(double)));
        if (status == kCuSuccess && capacity != 0u)
            status = d.copyDtoH(minimums.data(), values + capacity * 8u * sizeof(double),
                                static_cast<size_t>(capacity * 3u * sizeof(double)));
        if (status == kCuSuccess && capacity != 0u)
            status = d.copyDtoH(bits.data(), bitsTable(), static_cast<size_t>(capacity * 2u * sizeof(uint64_t)));
        if (status == kCuSuccess && !extents.empty())
            status = d.copyDtoH(extents.data(), boxExtents(), extents.size() * sizeof(double));
        d.contextSetCurrent(previous);
        if (status == kCuDeviceLost)
            return OVSPATIALQUERY_API_DEVICE_LOST;
        if (status != kCuSuccess)
            return OVSPATIALQUERY_API_INTERNAL;
        if (*hostStatus != 0u)
            return OVSPATIALQUERY_API_OPERATION_FAILED;
        std::vector<uint64_t> ids, refs;
        std::vector<int32_t>  types;
        {
            std::lock_guard<std::mutex> lock(s.membershipMutex);
            ids   = s.slotIds;
            refs  = s.slotRefs;
            types = s.slotTypes;
        }
        std::vector<ExplicitInstanceInput> instances;
        std::vector<PopulationObject>      primitives;
        instances.reserve(static_cast<size_t>(liveCount));
        for (size_t slot = 0; slot < ids.size() && slot < capacity; ++slot)
        {
            if (!std::isfinite(minimums[slot * 3u]))
                continue; // tombstoned in this version, or never written
            ExplicitInstanceInput instance;
            instance.objectValue = ids[slot];
            instance.geometry = {static_cast<ExactGeometryFamily>(types[slot] - kGeometryTypeFirstBuiltin), refs[slot]};
            for (size_t axis = 0; axis < 3; ++axis)
                instance.translation[axis] = poses[slot * 8u + axis];
            for (size_t component = 0; component < 4; ++component)
                instance.rotationXyzw[component] = poses[slot * 8u + 3u + component];
            instance.uniformScale      = poses[slot * 8u + 7u];
            instance.participationBits = {{bits[slot * 2u], bits[slot * 2u + 1u]}};
            if (ovstage &&
                (types[slot] == OVSPATIALQUERY_GEOMETRY_BOX || types[slot] == OVSPATIALQUERY_GEOMETRY_SPHERE ||
                 types[slot] == OVSPATIALQUERY_GEOMETRY_CAPSULE))
            {
                PopulationObject object{};
                object.id.value          = instance.objectValue;
                object.geometryReference = instance.geometry;
                object.participationBits = instance.participationBits;
                object.uniformScale      = instance.uniformScale;
                ovspatialquery_api_status_t prepared;
                double                      rotation[9];
                ovstage_transform::rotation(instance.rotationXyzw.data(), rotation);
                if (types[slot] == OVSPATIALQUERY_GEOMETRY_BOX)
                {
                    BoxGeometrySource source;
                    source.center = instance.translation;
                    for (size_t local = 0; local < 3; ++local)
                    {
                        source.halfExtents[local] = extents[slot * 3u + local];
                        for (size_t axis = 0; axis < 3; ++axis)
                            source.axes[local][axis] = rotation[axis * 3u + local];
                    }
                    prepared = createBoxExactGeometry(source, object.exactGeometry);
                }
                else if (types[slot] == OVSPATIALQUERY_GEOMETRY_SPHERE)
                {
                    SphereGeometrySource source;
                    source.center = instance.translation;
                    source.radius = extents[slot * 3u];
                    prepared      = createSphereExactGeometry(source, object.exactGeometry);
                }
                else
                {
                    CapsuleGeometrySource source;
                    source.radius = extents[slot * 3u];
                    for (size_t axis = 0; axis < 3; ++axis)
                    {
                        const double offset    = rotation[axis * 3u + 1u] * extents[slot * 3u + 1u];
                        source.endpointA[axis] = instance.translation[axis] - offset;
                        source.endpointB[axis] = instance.translation[axis] + offset;
                    }
                    prepared = createCapsuleExactGeometry(source, object.exactGeometry);
                }
                if (prepared != OVSPATIALQUERY_API_SUCCESS)
                    return prepared;
                object.bounds = object.exactGeometry->worldBounds();
                primitives.push_back(std::move(object));
            }
            else
                instances.push_back(instance);
        }
        const auto prepared = s.preparer->prepareObjects(instances, outObjects);
        if (prepared != OVSPATIALQUERY_API_SUCCESS)
            return prepared;
        for (auto& primitive : primitives)
            outObjects.push_back(std::move(primitive));
        if (!primitives.empty())
            std::sort(outObjects.begin(), outObjects.end(),
                      [](const PopulationObject& a, const PopulationObject& b) { return a.id.value < b.id.value; });
        return OVSPATIALQUERY_API_SUCCESS;
    }
};
} // namespace
DeviceBoxPopulation::DeviceBoxPopulation() : impl(std::make_shared<Impl>()) {}
uint64_t DeviceBoxPopulation::capacity() const { return impl->capacity; }
DLDevice DeviceBoxPopulation::device() const { return impl->device; }
// Per slot 8 pose, 3 minimum and 3 maximum doubles and two bits words, then the
// status word, then the grid, then the list of slots the version's rows rewrote.
uint64_t DeviceBoxPopulation::versionBytes(uint64_t capacity, bool ovstage)
{
    return capacity * kVersionBytesPerObject + kStatusBytes + gridBytes(capacity) + capacity * 4u +
           (ovstage ? capacity * 3u * sizeof(double) : 0u);
}
// 60 bytes per slot plus the pad that keeps the sorted index 8-byte aligned.
uint64_t DeviceBoxPopulation::membershipBytes(uint64_t slots) { return slots * 60u + 8u; }
uint64_t DeviceBoxPopulation::forkStagingBytes(uint64_t rows) { return 4096u + rows * 256u; }
// Records are kept for at least 16 definitions and doubled when they run out;
// a hull's vertices are its own allocation. Only convex hulls and meshes have
// records: the other families carry their parameters in the membership row.
namespace
{
constexpr uint64_t kShapeRecordBytes = sizeof(cuda_family::DeviceShapeRecord);
static_assert(kShapeRecordBytes == 64u, "the kernel indexes shape records by 64 bytes");
bool hasRecord(const DeviceShapeDescription& shape)
{
    return shape.type == OVSPATIALQUERY_GEOMETRY_CONVEX || shape.type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
}
uint64_t hullVertexBytes(const DeviceShapeDescription& shape)
{
    return shape.hull != nullptr ? shape.hull->vertexCount() * 3u * sizeof(double) : 0u;
}
// Records needed once `shapes` join `known` definitions, and the bytes that
// costs: hull vertices for each new definition, and a doubled record buffer
// when the new count exceeds `recordCapacity`.
uint64_t definitionBytesBeyond(const std::vector<DeviceShapeDescription>&              shapes,
                               const std::map<std::pair<int32_t, uint64_t>, uint32_t>& known, uint64_t recordCapacity,
                               bool copyRecords = false)
{
    std::map<std::pair<int32_t, uint64_t>, uint32_t> added;
    uint64_t                                         bytes = 0u;
    for (const DeviceShapeDescription& shape : shapes)
    {
        const std::pair<int32_t, uint64_t> key{shape.type, shape.value};
        if (!hasRecord(shape) || known.count(key) != 0u || !added.emplace(key, 0u).second)
            continue;
        bytes += hullVertexBytes(shape);
    }
    const uint64_t needed = known.size() + added.size();
    if (needed > recordCapacity || (copyRecords && needed > known.size()))
    {
        uint64_t grown = recordCapacity < 16u ? 16u : recordCapacity;
        while (grown < needed)
            grown *= 2u;
        bytes += grown * kShapeRecordBytes;
    }
    return bytes;
}
} // namespace
uint64_t DeviceBoxPopulation::definitionBytes(const std::vector<DeviceShapeDescription>& shapes)
{
    return definitionBytesBeyond(shapes, {}, 0u);
}
uint64_t DeviceBoxPopulation::additionalDefinitionBytes(const std::vector<DeviceShapeDescription>& shapes,
                                                        bool                                       fork) const
{
    std::lock_guard<std::mutex> lock(impl->membershipMutex);
    return definitionBytesBeyond(shapes, impl->pool.index, impl->pool.capacity, fork || !impl->pool.recordsOwned);
}
void DeviceBoxPopulation::setOvstageLiveCount(const std::shared_ptr<CudaPopulationVersion>& version, uint64_t count)
{
    if (const auto box = std::dynamic_pointer_cast<BoxDeviceVersion>(version))
        box->liveCount = count;
}
bool DeviceBoxPopulation::compatibleContext(void* context) const { return impl->context == context; }
std::shared_ptr<DevicePublicationOutcome> DeviceBoxPopulation::outcome(
    const std::shared_ptr<const CudaPopulationVersion>& version)
{
    const std::shared_ptr<const BoxDeviceVersion> box = std::dynamic_pointer_cast<const BoxDeviceVersion>(version);
    return box != nullptr ? box->outcome : nullptr;
}
namespace
{
// Registers the convex hulls and meshes `shapes` reference that the pool does
// not hold yet (vertices uploaded, one record each, the record buffer grown by
// doubling) and returns each shape's record index, kNoDefinition for the
// families whose parameters live in the membership row. Event-rate work:
// the uploads are synchronous with the host.
CuResult registerDefinitions(DeviceBoxPopulation::Impl& s, const std::vector<DeviceShapeDescription>& shapes,
                             std::vector<uint32_t>& outDefs)
{
    auto& d = s.driver;
    outDefs.assign(shapes.size(), kNoDefinition);
    for (size_t i = 0; i < shapes.size(); ++i)
    {
        const DeviceShapeDescription& shape = shapes[i];
        if (!hasRecord(shape))
            continue;
        const std::pair<int32_t, uint64_t> key{shape.type, shape.value};
        const auto                         known = s.pool.index.find(key);
        if (known != s.pool.index.end())
        {
            outDefs[i] = known->second;
            continue;
        }
        cuda_family::DeviceShapeRecord record{};
        if (shape.hull != nullptr)
        {
            const std::vector<double>& vertices = shape.hull->vertices();
            CuDevicePtr                storage  = 0u;
            CuResult                   status   = d.memAlloc(&storage, vertices.size() * sizeof(double));
            if (status != kCuSuccess)
                return status;
            s.pool.allocations.push_back(storage);
            status = d.copyHtoDAsync(storage, vertices.data(), vertices.size() * sizeof(double), s.stream);
            if (status != kCuSuccess)
                return status;
            record.vertices    = storage;
            record.vertexCount = static_cast<unsigned int>(vertices.size() / 3u);
            s.pool.hulls.push_back(shape.hull);
            // The hull's bytes stay charged while the pool keeps it, even after
            // its definition is destroyed.
            if (shape.memoryOwner != nullptr)
                s.pool.leases.push_back(shape.memoryOwner);
        }
        else
        {
            std::memcpy(record.localBounds, shape.localBounds.minimum, 3u * sizeof(double));
            std::memcpy(record.localBounds + 3, shape.localBounds.maximum, 3u * sizeof(double));
        }
        const uint32_t index = static_cast<uint32_t>(s.pool.index.size()); // one record per index entry
        if (index + 1u > s.pool.capacity || !s.pool.recordsOwned)
        {
            uint64_t grown = s.pool.capacity < 16u ? 16u : s.pool.capacity;
            while (grown < index + 1u)
                grown *= 2u;
            CuDevicePtr    records = 0u;
            const CuResult status  = d.memAlloc(&records, static_cast<size_t>(grown * kShapeRecordBytes));
            if (status != kCuSuccess)
                return status;
            if (s.pool.records != 0u)
            {
                const CuResult copied =
                    d.copyDtoDAsync(records, s.pool.records, static_cast<size_t>(index * kShapeRecordBytes), s.stream);
                if (copied != kCuSuccess)
                {
                    d.memFree(records);
                    return copied;
                }
                if (s.pool.recordsOwned)
                    s.pool.allocations.push_back(s.pool.records); // a launch may still read it
            }
            s.pool.records      = records;
            s.pool.capacity     = grown;
            s.pool.recordsOwned = true;
        }
        const cuda_family::DeviceShapeRecord* upload = &record;
        if (s.uploadStaging != nullptr)
        {
            s.uploadStaging->records.push_back(record);
            upload = &s.uploadStaging->records.back();
        }
        const CuResult written =
            d.copyHtoDAsync(s.pool.records + index * kShapeRecordBytes, upload, sizeof(record), s.stream);
        if (written != kCuSuccess)
            return written;
        s.pool.index.emplace(key, index);
        outDefs[i] = index;
    }
    return kCuSuccess;
}
// Writes the membership rows of slots [first, first + count) and the sorted
// index of live objects to the device, on the population's stream. The
// caller has pushed those slots' IDs, bindings and types onto the slot history.
CuResult uploadMembership(DeviceBoxPopulation::Impl& s, uint64_t first, uint64_t count,
                          const std::vector<DeviceShapeDescription>& shapes, bool uploadIndex = true)
{
    auto&    d      = s.driver;
    CuResult status = kCuSuccess;
    if (count != 0u)
    {
        std::vector<uint32_t>   localDefs;
        std::vector<double>     localParams;
        BoxUploadStaging::Rows* staging = nullptr;
        if (s.uploadStaging != nullptr)
        {
            s.uploadStaging->rows.emplace_back();
            staging = &s.uploadStaging->rows.back();
        }
        auto& defs   = staging != nullptr ? staging->definitions : localDefs;
        auto& params = staging != nullptr ? staging->params : localParams;
        status       = registerDefinitions(s, shapes, defs);
        if (status != kCuSuccess)
            return status;
        params.resize(count * 3u);
        for (uint64_t i = 0; i < count; ++i)
            std::memcpy(params.data() + i * 3u, shapes[i].parameters.data(), 24u);
        status = d.copyHtoDAsync(s.ids + first * 8u, s.slotIds.data() + first, count * 8u, s.stream);
        if (status == kCuSuccess)
            status = d.copyHtoDAsync(s.refs + first * 8u, s.slotRefs.data() + first, count * 8u, s.stream);
        if (status == kCuSuccess)
            status = d.copyHtoDAsync(s.params + first * 24u, params.data(), count * 24u, s.stream);
        if (status == kCuSuccess)
            status = d.copyHtoDAsync(s.types + first * 4u, s.slotTypes.data() + first, count * 4u, s.stream);
        if (status == kCuSuccess)
            status = d.copyHtoDAsync(s.defs + first * 4u, defs.data(), count * 4u, s.stream);
    }
    if (!uploadIndex)
        return status;
    s.indexIds.resize(s.index.size());
    s.indexSlots.resize(s.index.size());
    for (size_t i = 0; i < s.index.size(); ++i)
    {
        s.indexIds[i]   = s.index[i].first;
        s.indexSlots[i] = s.index[i].second;
    }
    if (status == kCuSuccess && !s.index.empty())
        status = d.copyHtoDAsync(s.sortedIds, s.indexIds.data(), s.index.size() * 8u, s.stream);
    if (status == kCuSuccess && !s.index.empty())
        status = d.copyHtoDAsync(s.sortedSlots, s.indexSlots.data(), s.index.size() * 4u, s.stream);
    return status;
}
ovspatialquery_api_status_t apiStatus(CuResult status)
{
    return status == kCuSuccess       ? OVSPATIALQUERY_API_SUCCESS
           : status == kCuOutOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY
           : status == kCuDeviceLost  ? OVSPATIALQUERY_API_DEVICE_LOST
                                      : OVSPATIALQUERY_API_INTERNAL;
}
} // namespace
ovspatialquery_api_status_t DeviceBoxPopulation::initialize(
    DLDevice device, uint64_t capacity, const std::vector<ExplicitInstanceInput>& instances,
    const std::vector<DeviceShapeDescription>& shapes, std::vector<std::shared_ptr<const void>> definitions,
    std::shared_ptr<void> hostLease, std::shared_ptr<void> deviceLease,
    std::shared_ptr<const DeviceObjectPreparer> preparer, std::shared_ptr<void> poolLease)
{
    auto& s = *impl;
    if (capacity < instances.size() || capacity > UINT32_MAX || preparer == nullptr ||
        shapes.size() != instances.size())
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    s.preparer = std::move(preparer);
    if (poolLease != nullptr)
        s.pool.leases.push_back(std::move(poolLease));
    s.device      = device;
    s.capacity    = capacity;
    s.assigned    = instances.size();
    s.live        = instances.size();
    s.hostLease   = std::move(hostLease);
    s.deviceLease = std::move(deviceLease);
    s.definitions = std::move(definitions);
    s.definitionKeys.reserve(s.definitions.size());
    s.definitionWatches.reserve(s.definitions.size());
    for (const std::shared_ptr<const void>& owner : s.definitions)
    {
        s.definitionKeys.push_back(owner.get());
        s.definitionWatches.push_back(owner);
    }
    s.slotIds.reserve(static_cast<size_t>(capacity));
    s.index.reserve(static_cast<size_t>(capacity));
    for (size_t i = 0; i < instances.size(); ++i)
    {
        s.slotIds.push_back(instances[i].objectValue);
        s.slotRefs.push_back(instances[i].geometry.value);
        s.slotTypes.push_back(shapes[i].type);
        s.index.emplace_back(instances[i].objectValue, static_cast<uint32_t>(i));
    }
    std::sort(s.index.begin(), s.index.end());
    s.freeStatus.reserve(kStatusPoolWords);
    for (size_t slot = kStatusPoolWords; slot-- > 0u;)
        s.freeStatus.push_back(static_cast<uint16_t>(slot));
    // A cell twice the widest axis-aligned extent of the set's shapes at unit
    // scale holds any rotation of them within two cells; wider objects go to
    // the oversize list, which every query walks. Each shape's extent is its
    // bounding radius about its origin: the half extents, the radius, the
    // radius plus the half segment, the farthest hull vertex, the farthest
    // mesh corner. The cell is sized for all but the widest sixteenth of the
    // objects, at most kGridOversizeObjects of them, so one ground mesh among
    // many boxes is listed as oversize instead of widening every cell until
    // each query walks the whole population.
    std::vector<double> radii;
    radii.reserve(shapes.size());
    for (const DeviceShapeDescription& shape : shapes)
    {
        double widest = 0.0;
        switch (shape.type)
        {
        case OVSPATIALQUERY_GEOMETRY_SPHERE: widest = shape.parameters[0]; break;
        case OVSPATIALQUERY_GEOMETRY_CAPSULE: widest = shape.parameters[0] + shape.parameters[1]; break;
        case OVSPATIALQUERY_GEOMETRY_CONVEX:
            if (shape.hull != nullptr)
                for (const double coordinate : shape.hull->vertices())
                    widest = std::max(widest, std::abs(coordinate));
            break;
        case OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH:
            for (size_t axis = 0; axis < 3; ++axis)
                widest = std::max(
                    {widest, std::abs(shape.localBounds.minimum[axis]), std::abs(shape.localBounds.maximum[axis])});
            break;
        default: widest = std::max({shape.parameters[0], shape.parameters[1], shape.parameters[2]}); break;
        }
        radii.push_back(widest);
    }
    double widest = 0.0;
    if (!radii.empty())
    {
        const size_t oversize = (std::min)(radii.size() / 16u, kGridOversizeObjects);
        auto         kept     = radii.begin() + static_cast<ptrdiff_t>(oversize);
        std::nth_element(radii.begin(), kept, radii.end(), std::greater<>());
        widest = *kept;
    }
    double target = 4.0 * widest;
    if (!(target > 0.0) || !std::isfinite(target))
        target = 1.0;
    int exponent = 0;
    (void)std::frexp(target, &exponent);
    exponent      = std::clamp(exponent, -1000, 1000);
    s.inverseCell = std::ldexp(1.0, -exponent);
    if (!s.driver.open() || s.driver.primaryRetain(&s.context, device.device_id) != kCuSuccess)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    const ovspatialquery_api_status_t loaded = acquireCudaContextModule(device.device_id, s.contextModule);
    if (loaded != OVSPATIALQUERY_API_SUCCESS)
        return loaded;
    s.module           = s.contextModule->module;
    auto&     d        = s.driver;
    CuContext previous = nullptr;
    if (d.contextGetCurrent(&previous) != kCuSuccess || d.contextSetCurrent(s.context) != kCuSuccess)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    CuResult status = d.moduleGetFunction(&s.kernel, s.module, "ovspatialquery_cuda_prepare_boxes");
    if (status == kCuSuccess)
        status = d.moduleGetFunction(&s.tombstone, s.module, "ovspatialquery_cuda_tombstone_boxes");
    if (status == kCuSuccess)
        status = d.moduleGetFunction(&s.gridder, s.module, "ovspatialquery_cuda_grid_boxes");
    s.bucketMask = static_cast<uint32_t>(gridBuckets(capacity) - 1u);
    if (status == kCuSuccess)
        status = d.streamCreate(&s.stream, kCuStreamNonBlocking);
    const uint64_t bytes = membershipBytes(capacity);
    if (status == kCuSuccess && bytes)
        status = d.memAlloc(&s.bindings, static_cast<size_t>(bytes));
    s.ids         = s.bindings;
    s.refs        = s.ids + capacity * 8u;
    s.params      = s.refs + capacity * 8u;
    s.types       = s.params + capacity * 24u;
    s.defs        = s.types + capacity * 4u;
    s.sortedIds   = s.defs + capacity * 4u; // 48 bytes per slot so far: 8-byte aligned
    s.sortedSlots = s.sortedIds + capacity * 8u;
    if (status == kCuSuccess && bytes)
        status = d.memsetD8Async(s.bindings, 0u, static_cast<size_t>(bytes), s.stream);
    if (status == kCuSuccess)
        status = uploadMembership(s, 0u, s.assigned, shapes);
    if (status == kCuSuccess)
    {
        void* pool = nullptr;
        if (d.memHostAlloc(&pool, kStatusPoolWords * sizeof(uint32_t), 0u) == kCuSuccess)
        {
            s.statusPool       = static_cast<uint32_t*>(pool);
            s.statusPoolPinned = true;
        }
        else
        {
            s.statusPool = new (std::nothrow) uint32_t[kStatusPoolWords];
            if (s.statusPool == nullptr)
                status = kCuOutOfMemory;
        }
    }
    if (status == kCuSuccess)
        status = d.streamSynchronize(s.stream);
    d.contextSetCurrent(previous);
    return status == kCuSuccess       ? OVSPATIALQUERY_API_SUCCESS
           : status == kCuOutOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                      : OVSPATIALQUERY_API_NOT_SUPPORTED;
}
ovspatialquery_api_status_t DeviceBoxPopulation::forkOvstage(
    const std::vector<ExplicitInstanceInput>& replacements, const std::vector<DeviceShapeDescription>& shapes,
    std::shared_ptr<const DeviceObjectPreparer> preparer, std::vector<std::shared_ptr<const void>> definitions,
    std::shared_ptr<void> hostLease, std::shared_ptr<void> deviceLease, std::shared_ptr<void> poolLease,
    std::shared_ptr<void> stagingLease, std::shared_ptr<DeviceBoxPopulation>& outPopulation) const
{
    if (preparer == nullptr || replacements.size() != shapes.size() || definitions.size() != replacements.size())
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(impl->membershipMutex);
    auto&                       old = *impl;
    if (old.lost)
        return OVSPATIALQUERY_API_DEVICE_LOST;
    std::vector<uint32_t> slots;
    slots.reserve(replacements.size());
    for (const auto& replacement : replacements)
    {
        uint32_t slot = 0u;
        if (!old.findLive(replacement.objectValue, slot) || std::find(slots.begin(), slots.end(), slot) != slots.end())
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        slots.push_back(slot);
    }
    if (old.runtime == nullptr)
    {
        auto runtime = std::make_shared<SharedBoxRuntime>();
        if (!runtime->driver.open())
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        runtime->device        = old.device;
        runtime->context       = old.context;
        runtime->contextModule = old.contextModule;
        runtime->stream        = old.stream;
        old.runtime            = std::move(runtime);
    }
    if (old.bindingOwner == nullptr)
    {
        auto owner     = std::make_shared<SharedBoxStorage>();
        owner->runtime = old.runtime;
        owner->leases.push_back(old.deviceLease);
        owner->staging = old.uploadStaging;
        owner->allocations.push_back(old.bindings);
        old.bindingOwner                  = std::move(owner);
        old.bindingOwner->ownsAllocations = true;
    }
    if (!old.pool.allocations.empty() || old.pool.recordsOwned)
    {
        auto backing         = std::make_shared<SharedBoxStorage>();
        backing->runtime     = old.runtime;
        backing->leases      = old.pool.leases;
        backing->allocations = old.pool.allocations;
        if (old.pool.recordsOwned && old.pool.records != 0u)
            backing->allocations.push_back(old.pool.records);
        old.pool.backing.push_back(std::move(backing));
        old.pool.backing.back()->ownsAllocations = true;
        old.pool.allocations.clear();
        old.pool.leases.clear();
        old.pool.recordsOwned = false;
    }
    auto  population       = std::make_shared<DeviceBoxPopulation>();
    auto& s                = *population->impl;
    s.runtime              = old.runtime;
    s.sourceBindings       = old.bindingOwner;
    s.uploadStaging        = std::make_shared<BoxUploadStaging>();
    s.uploadStaging->lease = std::move(stagingLease);
    s.device               = old.device;
    s.capacity             = old.capacity;
    s.assigned             = old.assigned;
    s.live                 = old.live;
    s.hostLease            = std::move(hostLease);
    s.deviceLease          = std::move(deviceLease);
    s.preparer             = std::move(preparer);
    s.slotIds              = old.slotIds;
    s.slotRefs             = old.slotRefs;
    s.slotTypes            = old.slotTypes;
    s.index                = old.index;
    s.retired              = old.retired;
    s.pool.records         = old.pool.records;
    s.pool.capacity        = old.pool.capacity;
    s.pool.recordsOwned    = false;
    s.pool.index           = old.pool.index;
    s.pool.hulls           = old.pool.hulls;
    s.pool.backing         = old.pool.backing;
    if (poolLease != nullptr)
        s.pool.leases.push_back(std::move(poolLease));
    s.inverseCell       = old.inverseCell;
    s.bucketMask        = old.bucketMask;
    s.definitions       = old.definitions;
    s.definitionKeys    = old.definitionKeys;
    s.definitionWatches = old.definitionWatches;
    for (size_t i = 0; i < slots.size(); ++i)
    {
        s.slotRefs[slots[i]]          = replacements[i].geometry.value;
        s.slotTypes[slots[i]]         = shapes[i].type;
        s.definitionKeys[slots[i]]    = definitions[i].get();
        s.definitionWatches[slots[i]] = definitions[i];
        s.definitions[slots[i]]       = std::move(definitions[i]);
    }
    // Driver owns its dynamic-library handle, so borrowing its function table
    // by copying it would close the parent's handle on destruction.
    if (!s.driver.open())
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    s.context       = old.context;
    s.contextModule = old.contextModule;
    s.module        = old.module;
    s.kernel        = old.kernel;
    s.tombstone     = old.tombstone;
    s.gridder       = old.gridder;
    s.stream        = old.stream;
    s.freeStatus.reserve(kStatusPoolWords);
    for (size_t slot = kStatusPoolWords; slot-- > 0u;)
        s.freeStatus.push_back(static_cast<uint16_t>(slot));
    auto&     d        = s.driver;
    CuContext previous = nullptr;
    if (d.contextGetCurrent(&previous) != kCuSuccess || d.contextSetCurrent(s.context) != kCuSuccess)
        return OVSPATIALQUERY_API_INTERNAL;
    struct RestoreContext
    {
        Driver&   driver;
        CuContext previous;
        ~RestoreContext() { driver.contextSetCurrent(previous); }
    } restore{d, previous};
    const uint64_t bytes        = membershipBytes(s.capacity);
    auto           bindingOwner = std::make_shared<SharedBoxStorage>();
    bindingOwner->runtime       = s.runtime;
    bindingOwner->leases.push_back(s.deviceLease);
    bindingOwner->staging = s.uploadStaging;
    bindingOwner->allocations.resize(1u);
    s.bindingOwner                  = std::move(bindingOwner);
    CuResult status                 = d.memAlloc(&s.bindings, static_cast<size_t>(bytes));
    s.bindingOwner->allocations[0]  = s.bindings;
    s.bindingOwner->ownsAllocations = true;
    s.ids                           = s.bindings;
    s.refs                          = s.bindings + (old.refs - old.bindings);
    s.params                        = s.bindings + (old.params - old.bindings);
    s.types                         = s.bindings + (old.types - old.bindings);
    s.defs                          = s.bindings + (old.defs - old.bindings);
    s.sortedIds                     = s.bindings + (old.sortedIds - old.bindings);
    s.sortedSlots                   = s.bindings + (old.sortedSlots - old.bindings);
    if (status == kCuSuccess)
        status = d.copyDtoDAsync(s.bindings, old.bindings, static_cast<size_t>(bytes), s.stream);
    for (size_t i = 0; status == kCuSuccess && i < slots.size(); ++i)
        status = uploadMembership(s, slots[i], 1u, {shapes[i]}, false);
    if (status == kCuSuccess)
    {
        void* pool = nullptr;
        if (d.memHostAlloc(&pool, kStatusPoolWords * sizeof(uint32_t), 0u) == kCuSuccess)
        {
            s.statusPool       = static_cast<uint32_t*>(pool);
            s.statusPoolPinned = true;
        }
        else
        {
            s.statusPool = new (std::nothrow) uint32_t[kStatusPoolWords];
            if (s.statusPool == nullptr)
                status = kCuOutOfMemory;
        }
    }
    if (status != kCuSuccess)
        return apiStatus(status);
    outPopulation = std::move(population);
    return OVSPATIALQUERY_API_SUCCESS;
}
ovspatialquery_api_status_t DeviceBoxPopulation::planMembershipChange(const std::vector<ExplicitInstanceInput>& rows,
                                                                      const std::vector<uint64_t>& removedIds,
                                                                      DeviceMembershipChange&      outChange) const
{
    std::lock_guard<std::mutex> lock(impl->membershipMutex);
    const Impl&                 s = *impl;
    DeviceMembershipChange      change;
    std::vector<uint64_t>       removed(removedIds);
    std::sort(removed.begin(), removed.end());
    if (std::adjacent_find(removed.begin(), removed.end()) != removed.end())
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    for (uint64_t id : removed)
    {
        uint32_t slot = 0u;
        if (!s.findLive(id, slot))
            return OVSPATIALQUERY_API_INVALID_ARGUMENT; // unknown or already retired
        change.removedSlots.push_back(slot);
    }
    std::vector<uint64_t> addedIds;
    for (const ExplicitInstanceInput& row : rows)
    {
        if (row.objectValue == 0u || std::binary_search(removed.begin(), removed.end(), row.objectValue))
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        uint32_t slot = 0u;
        if (s.findLive(row.objectValue, slot))
            continue; // an existing object; the kernel validates its row
        if (s.isRetired(row.objectValue))
            return OVSPATIALQUERY_API_INVALID_ARGUMENT; // an ID is never reused
        change.added.push_back(row);
        addedIds.push_back(row.objectValue);
    }
    std::sort(addedIds.begin(), addedIds.end());
    if (std::adjacent_find(addedIds.begin(), addedIds.end()) != addedIds.end())
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    if (change.added.size() > s.capacity - s.assigned)
        return OVSPATIALQUERY_API_INVALID_ARGUMENT; // beyond the slot capacity
    outChange = std::move(change);
    return OVSPATIALQUERY_API_SUCCESS;
}
ovspatialquery_api_status_t DeviceBoxPopulation::applyMembershipChange(
    const DeviceMembershipChange& change, std::vector<std::shared_ptr<const void>> addedDefinitions,
    std::shared_ptr<void> poolLease)
{
    if (change.empty())
        return OVSPATIALQUERY_API_SUCCESS;
    std::lock_guard<std::mutex> lock(impl->membershipMutex);
    auto&                       s = *impl;
    if (s.lost)
        return OVSPATIALQUERY_API_DEVICE_LOST;
    if (change.added.size() != change.addedShapes.size() || change.added.size() != addedDefinitions.size() ||
        change.added.size() > s.capacity - s.assigned)
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    if (poolLease != nullptr)
        s.pool.leases.push_back(std::move(poolLease));
    const uint64_t first = s.assigned;
    for (size_t i = 0; i < change.added.size(); ++i)
    {
        const uint32_t slot = static_cast<uint32_t>(first + i);
        s.slotIds.push_back(change.added[i].objectValue);
        s.slotRefs.push_back(change.added[i].geometry.value);
        s.slotTypes.push_back(change.addedShapes[i].type);
        s.definitionKeys.push_back(addedDefinitions[i].get());
        s.definitionWatches.push_back(addedDefinitions[i]);
        s.definitions.push_back(std::move(addedDefinitions[i]));
        s.index.insert(
            std::upper_bound(s.index.begin(), s.index.end(), std::make_pair(change.added[i].objectValue, 0u),
                             [](const std::pair<uint64_t, uint32_t>& key, const std::pair<uint64_t, uint32_t>& entry)
                             { return key.first < entry.first; }),
            std::make_pair(change.added[i].objectValue, slot));
    }
    s.assigned += change.added.size();
    s.live += change.added.size();
    // A removed object no longer binds its definition, which the caller may
    // destroy once no version binds it either. Its device record stays: a value
    // is never registered again, and retained versions may still read it.
    // The versions that hold the object hold the definition (DefinitionUses).
    for (uint32_t slot : change.removedSlots)
    {
        const uint64_t id = s.slotIds[slot];
        s.index.erase(std::find(s.index.begin(), s.index.end(), std::make_pair(id, slot)));
        s.retired.insert(std::upper_bound(s.retired.begin(), s.retired.end(), id), id);
        --s.live;
        s.definitions[slot].reset();
    }
    auto&     d        = s.driver;
    CuContext previous = nullptr;
    d.contextGetCurrent(&previous);
    if (d.contextSetCurrent(s.context) != kCuSuccess)
        return OVSPATIALQUERY_API_INTERNAL;
    const CuResult status = uploadMembership(s, first, change.added.size(), change.addedShapes);
    d.contextSetCurrent(previous);
    return apiStatus(status);
}
ovspatialquery_api_status_t DeviceBoxPopulation::launch(ovspatialquery_ordinal_t ordinal, const GenerationLease& base,
                                                        uint64_t rows, const std::array<DeviceColumn, 7>& columns,
                                                        bool posesOnly, bool snapshotRows, bool hostReadable,
                                                        uintptr_t afterEvent, const std::vector<uint32_t>& removedSlots,
                                                        std::shared_ptr<CudaPopulationVersion>& output, bool ovstage,
                                                        uintptr_t boxExtents, uintptr_t exclusionCount,
                                                        uintptr_t excludedIds, uintptr_t exclusionReasons)
{
    auto& s = *impl;
    auto& d = s.driver;
    if (rows > s.capacity)
        return OVSPATIALQUERY_API_OPERATION_FAILED;
    uint64_t                              liveNow = 0u, assignedNow = 0u, indexCount = 0u;
    CuDevicePtr                           recordsNow = 0u;
    std::shared_ptr<const DefinitionUses> definitions;
    const auto baseVersion = std::dynamic_pointer_cast<const BoxDeviceVersion>(base.cudaVersion());
    {
        std::lock_guard<std::mutex> lock(s.membershipMutex);
        if (s.lost)
            return OVSPATIALQUERY_API_DEVICE_LOST;
        liveNow     = s.live;
        assignedNow = s.assigned;
        recordsNow  = s.pool.records;
        indexCount  = s.index.size();
        if (baseVersion != nullptr && baseVersion->owner == impl && baseVersion->definitions != nullptr &&
            removedSlots.empty() && baseVersion->assigned == assignedNow)
            definitions = baseVersion->definitions;
        else
        {
            auto counted = std::make_shared<DefinitionUses>();
            if (baseVersion != nullptr && baseVersion->owner == impl && baseVersion->definitions != nullptr)
            {
                *counted = *baseVersion->definitions;
                // A slot a failed publication assigned after the base is not
                // counted there, and one removed since has no owner to add.
                for (const uint32_t slot : removedSlots)
                {
                    if (slot >= baseVersion->assigned)
                        continue;
                    const auto found = counted->find(s.definitionKeys[slot]);
                    if (found != counted->end() && found->second.second != UINT64_MAX && --found->second.second == 0u)
                        counted->erase(found);
                }
                for (uint64_t slot = baseVersion->assigned; slot < assignedNow; ++slot)
                {
                    if (s.definitions[slot] == nullptr)
                        continue;
                    auto& use = (*counted)[s.definitionKeys[slot]];
                    use.first = s.definitions[slot];
                    use.second += use.second != UINT64_MAX ? 1u : 0u;
                }
            }
            else
            {
                for (const auto& live : s.index)
                {
                    auto& use = (*counted)[s.definitionKeys[live.second]];
                    use.first = s.definitions[live.second];
                    ++use.second;
                }
                // A fork's base belongs to the population it replaced: keep every
                // definition the base held, never released through this lineage.
                if (baseVersion != nullptr && baseVersion->definitions != nullptr)
                    for (const auto& [key, use] : *baseVersion->definitions)
                        counted->emplace(key, std::make_pair(use.first, UINT64_MAX));
            }
            definitions = std::move(counted);
        }
    }
    if (!base.initialized() && rows != assignedNow)
        return OVSPATIALQUERY_API_OPERATION_FAILED;
    auto version           = std::make_shared<BoxDeviceVersion>(impl);
    version->ordinal       = ordinal;
    version->liveCount     = liveNow;
    version->records       = recordsNow;
    version->readable      = hostReadable;
    version->ovstage       = ovstage;
    version->changedCount  = rows;
    version->changedKnown  = posesOnly && base.initialized();
    version->changedPadded = snapshotRows;
    version->definitions   = std::move(definitions);
    version->assigned      = assignedNow;
    {
        std::lock_guard<std::mutex> lock(s.poolMutex);
        if (!s.freeStatus.empty())
        {
            version->statusSlot = s.freeStatus.back();
            s.freeStatus.pop_back();
            version->hostStatus = s.statusPool + version->statusSlot;
        }
        if (!s.freeTables.empty())
        {
            version->values = s.freeTables.back();
            s.freeTables.pop_back();
        }
        if (!s.freeScratch.empty())
        {
            version->scratch = s.freeScratch.back();
            s.freeScratch.pop_back();
        }
    }
    if (version->hostStatus == nullptr)
    {
        version->hostStatus = new (std::nothrow) uint32_t(0u);
        if (version->hostStatus == nullptr)
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    *version->hostStatus = 0u;
    CuContext previous   = nullptr;
    d.contextGetCurrent(&previous);
    if (d.contextSetCurrent(s.context) != kCuSuccess)
        return OVSPATIALQUERY_API_INTERNAL;
    const uint64_t tableBytes = s.capacity * kVersionBytesPerObject; // poses, bounds and bits
    CuResult       status     = d.eventCreate(&version->ready, kCuEventDisableTiming);
    if (status == kCuSuccess && !version->values)
        status = d.memAlloc(&version->values, static_cast<size_t>(versionBytes(s.capacity, ovstage)));
    // Per-slot seen flags plus the error word, then the removed-slot list.
    if (status == kCuSuccess && !version->scratch)
        status = d.memAlloc(&version->scratch, static_cast<size_t>((2u * s.capacity + 1u) * sizeof(uint32_t)));
    if (status == kCuSuccess && afterEvent != 0u)
        status = d.streamWaitEvent(s.stream, reinterpret_cast<CuEvent>(afterEvent), 0u);
    if (status == kCuSuccess)
        status = d.memsetD32Async(version->scratch, 0, static_cast<size_t>(s.capacity + 1), s.stream);
    const CuDevicePtr flag            = version->statusWord();
    CuDevicePtr       removedList     = version->scratch + (s.capacity + 1u) * sizeof(uint32_t);
    CuDevicePtr       predecessorFlag = 0u;
    CuDevicePtr       minimums        = version->values + s.capacity * 8u * sizeof(double);
    CuDevicePtr       maximums        = version->values + s.capacity * 11u * sizeof(double);
    CuDevicePtr       versionBits     = version->bitsTable();
    uint64_t          tombstoneBegin = 0u, tombstoneEnd = 0u, removedCount = removedSlots.size();
    if (status == kCuSuccess && base.initialized())
    {
        const auto old = base.cudaVersion();
        if (!old || !old->description().devicePrepared || old->description().objectCount != s.capacity)
            status = kCuInvalidHandle;
        else
        {
            if (tableBytes)
                status = d.copyDtoDAsync(version->values, reinterpret_cast<CuDevicePtr>(old->description().objectPoses),
                                         static_cast<size_t>(tableBytes), s.stream);
            if (status == kCuSuccess && ovstage)
            {
                const auto oldBoxes = std::dynamic_pointer_cast<const BoxDeviceVersion>(old);
                if (oldBoxes == nullptr || !oldBoxes->ovstage)
                    status = kCuInvalidHandle;
                else if (s.capacity != 0u)
                    status = d.copyDtoDAsync(version->boxExtents(), oldBoxes->boxExtents(),
                                             static_cast<size_t>(s.capacity * 3u * sizeof(double)), s.stream);
            }
            predecessorFlag      = reinterpret_cast<CuDevicePtr>(old->description().preparationFailed);
            version->predecessor = outcome(old);
        }
    }
    else
    {
        // The first version: every slot the set does not fill holds inverted bounds.
        tombstoneBegin = rows;
        tombstoneEnd   = s.capacity;
    }
    // The status starts as the base's: a version built on a failed one fails
    // with it, on the device and without a host round trip (ADR-0021).
    if (status == kCuSuccess)
        status = predecessorFlag ? d.copyDtoDAsync(flag, predecessorFlag, sizeof(uint32_t), s.stream)
                                 : d.memsetD32Async(flag, 0u, 1u, s.stream);
    if (status == kCuSuccess && removedCount != 0u)
        status = d.copyHtoDAsync(removedList, removedSlots.data(), removedCount * sizeof(uint32_t), s.stream);
    if (status == kCuSuccess && (tombstoneEnd > tombstoneBegin || removedCount != 0u))
    {
        void* tombstoneArgs[] = {&tombstoneBegin, &tombstoneEnd, &removedList, &removedCount, &minimums, &maximums};
        const uint64_t work   = (tombstoneEnd - tombstoneBegin) + removedCount;
        status = d.launchKernel(s.tombstone, static_cast<unsigned>((work + 127) / 128), 1, 1, 128, 1, 1, 0, s.stream,
                                tombstoneArgs, nullptr);
    }
    CuDevicePtr addresses[7];
    for (size_t i = 0; i < 7; ++i)
        addresses[i] = static_cast<CuDevicePtr>(columns[i].address);
    uint32_t    t = columns[3].bits, r = columns[4].bits, scale = columns[5].bits, poses = posesOnly ? 1u : 0u;
    CuDevicePtr errorWord      = flag;
    CuDevicePtr changedList    = version->changedSlots();
    uint32_t    ovstageRows    = ovstage ? 1u : 0u;
    CuDevicePtr currentExtents = boxExtents;
    CuDevicePtr versionExtents = ovstage ? version->boxExtents() : 0u;
    CuDevicePtr excludedCount = exclusionCount, excludedObjects = excludedIds, reasons = exclusionReasons;
    void*       args[] = {&indexCount,       &s.sortedIds,
                          &s.sortedSlots,    &rows,
                          &s.refs,           &s.params,
                          &s.types,          &s.defs,
                          &version->records, &addresses[0],
                          &addresses[1],     &addresses[2],
                          &addresses[3],     &t,
                          &addresses[4],     &r,
                          &addresses[5],     &scale,
                          &addresses[6],     &version->values,
                          &versionBits,      &minimums,
                          &maximums,         &version->scratch,
                          &errorWord,        &poses,
                          &changedList,      &ovstageRows,
                          &currentExtents,   &versionExtents,
                          &excludedCount,    &excludedObjects,
                          &reasons};
    if (status == kCuSuccess && rows)
        status = d.launchKernel(s.kernel, static_cast<unsigned>((rows + 127) / 128), 1, 1, 128, 1, 1, 0, s.stream, args,
                                nullptr);
    // The grid over the final bounds: one fill empties the header and the heads, then the slots are linked.
    CuDevicePtr gridHeader = version->gridHeader();
    CuDevicePtr gridHeads  = gridHeader + 32u;
    CuDevicePtr gridNext   = gridHeads + gridBuckets(s.capacity) * 4u;
    CuDevicePtr gridKeys   = gridNext + s.capacity * 4u;
    CuDevicePtr gridOver   = gridKeys + s.capacity * 12u;
    if (status == kCuSuccess)
        status = d.memsetD32Async(gridHeader, 0xffffffffu, static_cast<size_t>(8u + gridBuckets(s.capacity)), s.stream);
    if (status == kCuSuccess && s.capacity != 0u)
    {
        unsigned long long slots       = s.capacity;
        unsigned int       bucketMask  = s.bucketMask;
        double             inverseCell = s.inverseCell;
        void*              gridArgs[]  = {&slots,      &minimums,  &maximums, &bucketMask, &inverseCell,
                                          &gridHeader, &gridHeads, &gridNext, &gridKeys,   &gridOver};
        status = d.launchKernel(s.gridder, static_cast<unsigned>((slots + 127) / 128), 1, 1, 128, 1, 1, 0, s.stream,
                                gridArgs, nullptr);
    }
    if (status == kCuSuccess && takeCudaFailurePoint(kCudaFailureDevicePublicationLaunch))
        status = kCuDeviceLost;
    // Only the status word crosses to the host, after the kernel, before readiness.
    if (status == kCuSuccess)
        status = d.copyDtoHAsync(version->hostStatus, flag, sizeof(uint32_t), s.stream);
    if (status == kCuSuccess)
        status = d.eventRecord(version->ready, s.stream);
    d.contextSetCurrent(previous);
    if (status == kCuDeviceLost)
    {
        std::lock_guard<std::mutex> lock(s.membershipMutex);
        s.lost = true;
    }
    if (status != kCuSuccess)
        return apiStatus(status);
    output = std::move(version);
    return OVSPATIALQUERY_API_SUCCESS;
}
void DeviceBoxPopulation::ringWhenReady(const std::shared_ptr<CudaPopulationVersion>& output,
                                        const std::shared_ptr<DeviceSignal>&          signal)
{
    const std::shared_ptr<BoxDeviceVersion> version = std::dynamic_pointer_cast<BoxDeviceVersion>(output);
    if (version == nullptr || version->owner != impl || version->ready == nullptr)
        return;
    std::lock_guard<std::mutex> lock(impl->signalMutex);
    if (version->ringQueued)
        return;
    auto&     d        = impl->driver;
    CuContext previous = nullptr;
    d.contextGetCurrent(&previous);
    if (d.contextSetCurrent(impl->context) != kCuSuccess)
        return;
    // Not on `stream`: queued there now, the host function would also wait
    // for publications launched after this one, and hold back the next.
    version->ringQueued =
        (impl->signalStream != nullptr || d.streamCreate(&impl->signalStream, kCuStreamNonBlocking) == kCuSuccess) &&
        d.streamWaitEvent(impl->signalStream, version->ready, 0u) == kCuSuccess &&
        ringAfter(d, impl->signalStream, signal);
    d.contextSetCurrent(previous);
}
ovspatialquery_api_status_t DeviceBoxPopulation::complete(const std::shared_ptr<CudaPopulationVersion>& output)
{
    const std::shared_ptr<BoxDeviceVersion> version = std::dynamic_pointer_cast<BoxDeviceVersion>(output);
    if (version == nullptr || version->owner != impl)
        return OVSPATIALQUERY_API_INTERNAL;
    auto&     d        = impl->driver;
    CuContext previous = nullptr;
    d.contextGetCurrent(&previous);
    if (d.contextSetCurrent(impl->context) != kCuSuccess)
        return OVSPATIALQUERY_API_INTERNAL;
    // A poll, not a wait: the worker that asks serves other work until the
    // producer's event and the kernels behind it are done.
    CuResult waited = d.eventQuery(version->ready);
    d.contextSetCurrent(previous);
    if (waited == kCuNotReady)
        return OVSPATIALQUERY_API_NOT_READY;
    if (waited == kCuSuccess && takeCudaFailurePoint(kCudaFailureDevicePublicationComplete))
        waited = kCuDeviceLost;
    if (version->scratch && waited == kCuSuccess)
    {
        // The kernel is done with it; the next launch reuses it.
        std::lock_guard<std::mutex> lock(impl->poolMutex);
        impl->freeScratch.push_back(version->scratch);
        version->scratch = 0u;
    }
    if (waited == kCuDeviceLost)
    {
        // The header promises DEVICE_LOST, not a validation failure, for a lost
        // device: later launches of this membership fail fast with it.
        version->observed.store(static_cast<uint32_t>(CudaCompletionState::deviceLost) + 1u, std::memory_order_release);
        std::lock_guard<std::mutex> lock(impl->membershipMutex);
        impl->lost = true;
        return OVSPATIALQUERY_API_DEVICE_LOST;
    }
    if (waited != kCuSuccess)
        return OVSPATIALQUERY_API_INTERNAL;
    const CudaCompletionState state = version->resultState();
    // A chain built on a publication that is already known not to have become
    // the head fails with it; validation failures arrive through the status word.
    if (version->predecessor != nullptr && version->predecessor->state() == DevicePublicationOutcome::State::abandoned)
        return OVSPATIALQUERY_API_OPERATION_FAILED;
    if (state == CudaCompletionState::complete)
    {
        version->releaseReadyDependencies();
        return OVSPATIALQUERY_API_SUCCESS;
    }
    if (state == CudaCompletionState::deviceLost)
        return OVSPATIALQUERY_API_DEVICE_LOST;
    return state == CudaCompletionState::error ? OVSPATIALQUERY_API_OPERATION_FAILED : OVSPATIALQUERY_API_INTERNAL;
}
#endif
} // namespace detail
} // namespace ovspatialquery
