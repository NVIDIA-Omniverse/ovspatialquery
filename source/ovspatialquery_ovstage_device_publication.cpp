/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_ovstage_device_publication.h"

#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_cuda_driver.h"
#include "ovspatialquery_cuda_family_plan_types.h"
#include "ovspatialquery_device_box_population.h"
#include "ovspatialquery_geometry.h"
#include "ovspatialquery_input.h"
#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_instance_state.h"
#include "ovspatialquery_ovstage_filter.h"
#include "ovspatialquery_ovstage_mesh_geometry.h"
#include "ovspatialquery_typed_geometry.h"

#include <ovstage/ovstage_api/experimental/ovstage_placement_extension.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

const ovstage_placement_extension_v1_t* ovstagePlacementExtension(ovstage_instance_t* stage, std::string& outDiagnostic)
{
    const ovstage_placement_extension_v1_t* extension = nullptr;
    if (ovstage_query_placement_extension(stage, &extension) != OVSTAGE_OK || extension == nullptr)
    {
        outDiagnostic = "the stage's Ovstage runtime does not provide the ovstage.placement extension, version " +
                        std::to_string(OVSTAGE_PLACEMENT_EXTENSION_VERSION) +
                        ", that device-prepared publications read poses through";
        return nullptr;
    }
    return extension;
}

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

namespace
{

namespace family = cuda_family;

// Per output row of a gather: the pose columns the prepare kernel reads, the
// (data row, slot) pair it was gathered from, and for a membership change the
// binding and bits columns of a complete row.
constexpr uint64_t kScratchBytesPerSlot = 8u + 24u + 32u + 8u + 16u + 8u + 8u + 8u + 24u;

// What a device publication keeps for one slot beyond its pose: the definition it
// prepared at the slot's last membership change, scaled by the prim's axis scales
// then, so a readable version can build the slot's host object from a pose alone.
struct OvstageSlotShape
{
    ExactGeometryFamily                         family = ExactGeometryFamily::sphere;
    std::array<double, 3u>                      parameters{};
    std::shared_ptr<const PreparedConvexHull>   hull;
    std::shared_ptr<const PreparedTriangleMesh> mesh;
    bool                                        doubleSided = false;
};

using OvstageSlotMap = std::unordered_map<uint64_t, uint32_t>;

// Builds a readable device version's host objects the way an explicit registry
// builds them from rows (createInstanceGeometry), from the slots' own shapes.
class OvstageObjectPreparer final : public DeviceObjectPreparer
{
  public:
    std::shared_ptr<const OvstageSlotMap> slots;
    std::vector<OvstageSlotShape>         shapes;

    ovspatialquery_api_status_t prepareObjects(const std::vector<ExplicitInstanceInput>& instances,
                                               std::vector<PopulationObject>&            outObjects) const override
    {
        std::vector<PopulationObject> objects;
        objects.reserve(instances.size());
        for (const ExplicitInstanceInput& instance : instances)
        {
            if (!slots)
                return OVSPATIALQUERY_API_INTERNAL;
            const auto found = slots->find(instance.objectValue);
            if (found == slots->end() || found->second >= shapes.size())
                return OVSPATIALQUERY_API_INTERNAL;
            const OvstageSlotShape& shape = shapes[found->second];
            const double x = instance.rotationXyzw[0], y = instance.rotationXyzw[1], z = instance.rotationXyzw[2],
                         w      = instance.rotationXyzw[3];
            const double length = std::sqrt(x * x + y * y + z * z + w * w);
            if (!(length > 0.0))
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            const double qx = x / length, qy = y / length, qz = z / length, qw = w / length;
            // axes[local][world], as the host's box and capsule sources take them.
            const std::array<std::array<double, 3u>, 3u> axes{{
                {1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qz * qw), 2.0 * (qx * qz - qy * qw)},
                {2.0 * (qx * qy - qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qx * qw)},
                {2.0 * (qx * qz + qy * qw), 2.0 * (qy * qz - qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)},
            }};
            const double                                 scale = instance.uniformScale;
            std::array<double, 16u>                      worldFromLocal{};
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                for (uint32_t local = 0u; local < 3u; ++local)
                    worldFromLocal[axis * 4u + local] = axes[local][axis] * scale;
                worldFromLocal[axis * 4u + 3u] = instance.translation[axis];
            }
            worldFromLocal[15] = 1.0;
            PopulationObject object{};
            object.id                          = objectIdFromValue(instance.objectValue);
            object.geometryReference           = instance.geometry;
            object.participationBits           = instance.participationBits;
            object.uniformScale                = scale;
            ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
            switch (shape.family)
            {
            case ExactGeometryFamily::sphere: {
                SphereGeometrySource source;
                source.center = instance.translation;
                source.radius = shape.parameters[0] * scale;
                status        = createSphereExactGeometry(source, object.exactGeometry);
                break;
            }
            case ExactGeometryFamily::box: {
                BoxGeometrySource source;
                source.center = instance.translation;
                source.axes   = axes;
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                    source.halfExtents[axis] = shape.parameters[axis] * scale;
                status = createBoxExactGeometry(source, object.exactGeometry);
                break;
            }
            case ExactGeometryFamily::capsule: {
                CapsuleGeometrySource source;
                source.radius = shape.parameters[0] * scale;
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    const double offset    = axes[1u][axis] * shape.parameters[1] * scale;
                    source.endpointA[axis] = instance.translation[axis] - offset;
                    source.endpointB[axis] = instance.translation[axis] + offset;
                }
                status = createCapsuleExactGeometry(source, object.exactGeometry);
                break;
            }
            case ExactGeometryFamily::convex: {
                const ConvexConstructionOutcome outcome =
                    createConvexExactGeometry(shape.hull, worldFromLocal, object.exactGeometry);
                status                 = outcome == ConvexConstructionOutcome::success ? OVSPATIALQUERY_API_SUCCESS
                                         : outcome == ConvexConstructionOutcome::outOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                                                                             : OVSPATIALQUERY_API_INVALID_ARGUMENT;
                object.definitionOwner = shape.hull;
                break;
            }
            default:
                status = bindPreparedTriangleGeometry(shape.mesh, worldFromLocal, shape.doubleSided,
                                                      object.triangleGeometry);
                if (status == OVSPATIALQUERY_API_SUCCESS)
                    status = createTriangleMeshExactGeometry(object.triangleGeometry, object.exactGeometry);
                object.geometryKey = instance.geometry.value;
                break;
            }
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            object.bounds = object.exactGeometry->worldBounds();
            objects.push_back(std::move(object));
        }
        std::sort(objects.begin(), objects.end(),
                  [](const PopulationObject& left, const PopulationObject& right)
                  { return left.id.value < right.id.value; });
        outObjects = std::move(objects);
        return OVSPATIALQUERY_API_SUCCESS;
    }
};

// The CUDA resources a population's device publications share: the gather
// kernel, one stream it runs on, and pools of events and scratch, so a
// publication allocates nothing on a busy device (ADR-0021). Kept by every
// capture until its device work has finished.
class OvstageDeviceResources
{
  public:
    std::shared_ptr<OvstageDeviceResources> parent;
    Driver                                  driver;
    DLDevice                                device{kDLCUDA, 0};
    CuContext                               context = nullptr;
    // The device's shared module (acquireCudaContextModule), held by the root;
    // a child reaches it through `parent`.
    std::shared_ptr<CudaContextModule> contextModule;
    CuModule                           module   = nullptr;
    CuFunction                         gather   = nullptr;
    CuStream                           stream   = nullptr;
    CuDevicePtr                        records  = 0u; // OvstageSlotRecord[capacity]
    uint64_t                           capacity = 0u;
    ovspatialquery_instance_handle_t   instance = OVSPATIALQUERY_INVALID_INSTANCE;

    ~OvstageDeviceResources()
    {
        if (context == nullptr)
            return;
        if (parent)
        {
            // Last capture completion makes these buffers idle. Return the whole
            // allocation bucket without CUDA calls or allocation in this destructor;
            // newer gathers may already be waiting on the shared stream.
            if (storage)
            {
                std::lock_guard<std::mutex> lock(parent->mutex);
                storage->next = std::move(parent->idle);
                parent->idle  = std::move(storage);
            }
            return;
        }
        CuContext previous = nullptr;
        (void)driver.contextGetCurrent(&previous);
        (void)driver.contextSetCurrent(context);
        if (stream != nullptr)
            (void)driver.streamSynchronize(stream);
        while (storage)
        {
            for (CuEvent event : storage->events)
                (void)driver.eventDestroy(event);
            for (const auto& allocation : storage->allocations)
                if (allocation.pointer != 0u)
                    (void)driver.memFree(allocation.pointer);
            if (storage->records != 0u)
                (void)driver.memFree(storage->records);
            storage = std::move(idle);
            if (storage)
                idle = std::move(storage->next);
        }
        if (stream != nullptr)
            (void)driver.streamDestroy(stream);
        (void)driver.contextSetCurrent(previous);
        (void)driver.primaryRelease(device.device_id);
    }

    // Takes the device's shared module, finds the kernel and uploads the slot
    // records; NOT_SUPPORTED when the device or the bundled PTX cannot serve it.
    ovspatialquery_api_status_t initialize(ovspatialquery_instance_handle_t owner, DLDevice target,
                                           const std::vector<family::OvstageSlotRecord>& slots,
                                           std::shared_ptr<OvstageDeviceResources>       prior = {})
    {
        instance              = owner;
        device                = target;
        capacity              = slots.size();
        parent                = prior ? (prior->parent ? prior->parent : prior) : nullptr;
        const auto hostStatus = reserveSharedHostLease(instanceState(instance), sizeof(*this) + 128u, hostLease);
        if (hostStatus != OVSPATIALQUERY_API_SUCCESS)
            return hostStatus;
        if (parent)
            context = parent->context;
        if (!driver.open() || (!parent && driver.primaryRetain(&context, device.device_id) != kCuSuccess))
        {
            context = nullptr;
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        if (parent)
        {
            std::lock_guard<std::mutex> lock(parent->mutex);
            storage = std::move(parent->idle);
            if (storage)
                parent->idle = std::move(storage->next);
        }
        if (!storage)
        {
            std::shared_ptr<HostLease> lease;
            const auto charged = reserveSharedHostLease(instanceState(instance), sizeof(Storage) + 128u, lease);
            if (charged != OVSPATIALQUERY_API_SUCCESS)
                return charged;
            storage            = std::make_unique<Storage>();
            storage->hostLease = std::move(lease);
        }
        if (!parent)
        {
            const ovspatialquery_api_status_t loaded = acquireCudaContextModule(device.device_id, contextModule);
            if (loaded != OVSPATIALQUERY_API_SUCCESS)
                return loaded;
            module = contextModule->module;
        }
        CuContext previous = nullptr;
        if (driver.contextGetCurrent(&previous) != kCuSuccess || driver.contextSetCurrent(context) != kCuSuccess)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        CuResult status = kCuSuccess;
        if (parent)
        {
            module = parent->module;
            gather = parent->gather;
            stream = parent->stream;
        }
        if (status == kCuSuccess && !parent)
            status = driver.moduleGetFunction(&gather, module, "ovspatialquery_cuda_gather_ovstage_poses");
        if (status == kCuSuccess && !parent)
            status = driver.streamCreate(&stream, kCuStreamNonBlocking);
        const size_t bytes = slots.size() * sizeof(family::OvstageSlotRecord);
        if (status == kCuSuccess && bytes != 0u && storage->records == 0u)
        {
            const auto charged = reserveInstanceDeviceMemory(instance, device, bytes, storage->recordLease);
            if (charged != OVSPATIALQUERY_API_SUCCESS)
            {
                (void)driver.contextSetCurrent(previous);
                return charged;
            }
            status = driver.memAlloc(&storage->records, bytes);
            if (status != kCuSuccess)
                storage->recordLease.reset();
        }
        records = storage->records;
        if (status == kCuSuccess && bytes != 0u)
            status = driver.copyHtoDAsync(records, slots.data(), bytes, stream);
        if (status == kCuSuccess && !parent)
            status = driver.streamSynchronize(stream);
        (void)driver.contextSetCurrent(previous);
        return status == kCuSuccess       ? OVSPATIALQUERY_API_SUCCESS
               : status == kCuOutOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                          : OVSPATIALQUERY_API_NOT_SUPPORTED;
    }

    CuEvent takeEvent()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!storage->events.empty())
            {
                CuEvent event = storage->events.back();
                storage->events.pop_back();
                return event;
            }
        }
        CuEvent event = nullptr;
        return driver.eventCreate(&event, kCuEventDisableTiming) == kCuSuccess ? event : nullptr;
    }
    void giveEvent(CuEvent event)
    {
        if (event == nullptr)
            return;
        std::lock_guard<std::mutex> lock(mutex);
        storage->events.push_back(event);
    }
    ovspatialquery_api_status_t takeScratch(CuDevicePtr& outBuffer)
    {
        if (parent)
            return parent->takeScratch(outBuffer);
        std::lock_guard<std::mutex> lock(mutex);
        if (!storage->scratch.empty())
        {
            outBuffer = storage->scratch.back();
            storage->scratch.pop_back();
            return OVSPATIALQUERY_API_SUCCESS;
        }
        const size_t          bytes = static_cast<size_t>(std::max<uint64_t>(capacity, 1u) * kScratchBytesPerSlot);
        std::shared_ptr<void> lease;
        const auto            charged = reserveInstanceDeviceMemory(instance, device, bytes, lease);
        if (charged != OVSPATIALQUERY_API_SUCCESS)
            return charged;
        std::shared_ptr<HostLease> allocationHostLease;
        const auto                 hostStatus = reserveSharedHostLease(instanceState(instance),
                                                                       128u + 2u * (sizeof(Storage::Allocation) + sizeof(CuDevicePtr)),
                                                                       allocationHostLease);
        if (hostStatus != OVSPATIALQUERY_API_SUCCESS)
            return hostStatus;
        storage->allocations.push_back({0u, std::move(lease), std::move(allocationHostLease)});
        const auto allocated = driver.memAlloc(&storage->allocations.back().pointer, bytes);
        if (allocated != kCuSuccess)
        {
            storage->allocations.pop_back();
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        outBuffer = storage->allocations.back().pointer;
        return OVSPATIALQUERY_API_SUCCESS;
    }
    void giveScratch(CuDevicePtr buffer)
    {
        if (parent)
        {
            parent->giveScratch(buffer);
            return;
        }
        if (buffer == 0u)
            return;
        std::lock_guard<std::mutex> lock(mutex);
        storage->scratch.push_back(buffer);
    }

  private:
    std::mutex mutex;
    // Recovery keeps the slot capacity. Each allocation keeps its reservation
    // while idle, until the root frees it; reuse needs no second reservation.
    struct Storage
    {
        std::shared_ptr<HostLease> hostLease;
        std::shared_ptr<void>      recordLease;
        CuDevicePtr                records = 0u;
        struct Allocation
        {
            CuDevicePtr                pointer = 0u;
            std::shared_ptr<void>      lease;
            std::shared_ptr<HostLease> hostLease;
        };
        std::vector<Allocation>  allocations;
        std::vector<CuEvent>     events;
        std::vector<CuDevicePtr> scratch;
        std::unique_ptr<Storage> next;
    };
    std::shared_ptr<HostLease> hostLease;
    std::unique_ptr<Storage>   storage;
    std::unique_ptr<Storage>   idle;
};

// A publication owns its report independently of retained device versions. The
// population copies only recovery IDs after the event and keeps them until enqueue.
struct OvstageExclusionReport
{
    std::shared_ptr<OvstageDeviceResources> resources;
    std::shared_ptr<HostLease>              lease;
    std::shared_ptr<CudaPopulationVersion>  version;
    void*                                   storage  = nullptr;
    uint64_t                                capacity = 0;
    uint32_t*                               count    = nullptr;
    uint64_t*                               ids      = nullptr;
    int32_t*                                reasons  = nullptr;
    ~OvstageExclusionReport()
    {
        if (storage)
        {
            CuContext previous = nullptr;
            resources->driver.contextGetCurrent(&previous);
            resources->driver.contextSetCurrent(resources->context);
            resources->driver.memFreeHost(storage);
            resources->driver.contextSetCurrent(previous);
        }
    }
    ovspatialquery_api_status_t initialize()
    {
        CuContext previous = nullptr;
        auto&     d        = resources->driver;
        if (d.contextGetCurrent(&previous) != kCuSuccess || d.contextSetCurrent(resources->context) != kCuSuccess)
            return OVSPATIALQUERY_API_INTERNAL;
        const auto status = d.memHostAlloc(&storage, static_cast<size_t>(8 + capacity * 12), 0u);
        d.contextSetCurrent(previous);
        if (status != kCuSuccess)
            return status == kCuOutOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY : OVSPATIALQUERY_API_NOT_SUPPORTED;
        count   = static_cast<uint32_t*>(storage);
        *count  = 0;
        ids     = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(storage) + 8);
        reasons = reinterpret_cast<int32_t*>(ids + capacity);
        return OVSPATIALQUERY_API_SUCCESS;
    }
};
struct OvstagePendingPublication
{
    ovspatialquery_ordinal_t                ordinal = 0;
    std::shared_ptr<OvstageExclusionReport> report;
    std::vector<uint64_t>                   reshapes;
    std::shared_ptr<HostLease>              lease;
    bool                                    observed = false;
};
struct OvstagePendingRecovery
{
    std::mutex                                                 mutex;
    std::vector<std::shared_ptr<OvstagePendingPublication>>    publications;
    std::vector<uint64_t>                                      prims;
    std::vector<std::pair<uint64_t, ovspatialquery_ordinal_t>> rebuilt;
    std::shared_ptr<HostLease>                                 lease;
    // Caller holds mutex. No host wait: only completion permits reading mapped rows.
    void observe(OvstagePendingPublication& publication)
    {
        if (publication.observed || !publication.report || !publication.report->version)
            return;
        const auto state = publication.report->version->resultState();
        if (state == CudaCompletionState::pending)
            return;
        if (state == CudaCompletionState::complete)
        {
            const auto& report = *publication.report;
            for (uint32_t i = 0; i < *report.count; ++i)
                if (report.reasons[i] == OVSPATIALQUERY_OVSTAGE_EXCLUSION_RESHAPE_PENDING)
                    publication.reshapes.push_back(report.ids[i]);
        }
        publication.observed = true;
        publication.report.reset();
    }
    void fold()
    {
        std::lock_guard<std::mutex> lock(mutex);
        size_t                      consumed = 0;
        for (const auto& publication : publications)
        {
            observe(*publication);
            if (!publication->observed)
                break;
            for (uint64_t id : publication->reshapes)
            {
                const auto built =
                    std::lower_bound(rebuilt.begin(), rebuilt.end(), id,
                                     [](const auto& entry, uint64_t value) { return entry.first < value; });
                if (built != rebuilt.end() && built->first == id && publication->ordinal <= built->second)
                    continue;
                auto at = std::lower_bound(prims.begin(), prims.end(), id);
                if (at == prims.end() || *at != id)
                    prims.insert(at, id);
            }
            ++consumed;
        }
        publications.erase(publications.begin(), publications.begin() + consumed);
    }
};

} // namespace

// The membership the newest admitted device publication published: what the next
// one compares with on the host, the device tables its poses go to, and which
// world-matrix column feeds each slot (ADR-0047). Replaced, never changed, at a
// membership change; `valid` drops when a publication built on it is abandoned,
// so the next one starts from complete rows.
struct OvstageDeviceMembership
{
    std::vector<uint64_t>                     fingerprint;
    std::vector<ovx_token_t>                  definitionAttributes;
    std::vector<uint64_t>                     prototypePaths; // instance proxies' definition prims (ADR-0050)
    std::shared_ptr<const ParticipationTable> participation;
    std::shared_ptr<DeviceBoxPopulation>      boxes;
    std::shared_ptr<OvstageDeviceResources>   resources;
    std::shared_ptr<const OvstageSlotMap>     slotByPath;
    std::vector<uint8_t>                      authored; // per slot: the pose comes from `worldMatrix`
    bool                                      anyAuthored            = false;
    bool                                      droppedPointPrototypes = false;
    std::vector<ExplicitInstanceInput>        instances;
    SourceExclusions                          exclusions;
    std::vector<family::OvstageSlotRecord>    records;
    std::shared_ptr<OvstageObjectPreparer>    preparer;
    std::vector<std::shared_ptr<const void>>  definitions;
    std::shared_ptr<OvstagePendingRecovery>   recovery;
    std::atomic<bool>                         valid{true};
};

namespace
{

// The rows of one placed read group, resolved on the host: which slot each
// logical row feeds and which row of the group's tensor holds its matrix.
struct GatheredGroup
{
    ReadGroup             group{};
    std::vector<uint32_t> pairs; // data row, slot
};

struct PlacedRead
{
    ovstage_read_handle_t      handle = OVSTAGE_INVALID_READ_HANDLE;
    std::vector<GatheredGroup> groups;
};

class OvstageDeviceCapture final : public CudaPopulationCapture
{
  public:
    ~OvstageDeviceCapture() override
    {
        releaseAfterDevice();
        if (resources != nullptr)
        {
            resources->giveScratch(scratch);
            resources->giveEvent(gatherDone);
            resources->giveEvent(available);
        }
    }

    DLDevice device() const override { return execution.device; }
    uint64_t deviceByteCharge() const override { return DeviceBoxPopulation::versionBytes(capacity, true); }
    uint64_t deviceStagingByteCharge() const override { return (2u * capacity + 1u) * sizeof(uint32_t); }
    uint64_t hostStagingByteCharge() const override
    {
        return sizeof(*this) + 4096u + rows * 2u * sizeof(uint32_t) + capacity * 24u;
    }

    ovspatialquery_api_status_t prepareAdmission(const GenerationLease& base) override
    {
        if (!completeSet && (!base.initialized() || base.ordinal() != baseOrdinal))
            return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
        // Complete rows start a new membership; poses build on the newest admitted version of it.
        if (!completeRows)
            deviceBase = base;
        return OVSPATIALQUERY_API_SUCCESS;
    }

    ovspatialquery_api_status_t captureAdmissionInputs() override
    {
        Driver&   driver   = resources->driver;
        CuContext previous = nullptr;
        if (driver.contextGetCurrent(&previous) != kCuSuccess ||
            driver.contextSetCurrent(resources->context) != kCuSuccess)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        ovspatialquery_api_status_t status = launchGather();
        (void)driver.contextSetCurrent(previous);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::array<DeviceColumn, 7> inputs{};
        inputs[0] = {columns.ids, 64u};
        inputs[3] = {columns.translations, 64u};
        inputs[4] = {columns.rotations, 64u};
        inputs[5] = {columns.scales, 64u};
        if (completeRows)
        {
            inputs[1] = {columns.types, 32u};
            inputs[2] = {columns.values, 64u};
            inputs[6] = {columns.bits, 64u};
        }
        const auto launchStatus = membership->boxes->launch(
            targetOrdinal, deviceBase, rows, inputs, !completeRows, !completeRows, hostReadable,
            reinterpret_cast<uintptr_t>(gatherDone), {}, version, true, columns.extents,
            reinterpret_cast<uintptr_t>(report->count), reinterpret_cast<uintptr_t>(report->ids),
            reinterpret_cast<uintptr_t>(report->reasons));
        if (launchStatus == OVSPATIALQUERY_API_SUCCESS)
        {
            report->version       = version;
            pendingReport->report = report;
        }
        return launchStatus;
    }

    void publicationCommitted() noexcept override
    {
        committed = true;
        if (deviceState != nullptr)
            deviceState->commit(membership->boxes);
        if (const std::shared_ptr<DevicePublicationOutcome> outcome = DeviceBoxPopulation::outcome(version))
            outcome->finish(DevicePublicationOutcome::State::published);
    }

    // The runtime calls this when every publication ends; after a commit it only
    // releases the groups.
    void publicationAbandoned() noexcept override
    {

        if (!committed)
            membership->valid.store(false);
        if (const std::shared_ptr<DevicePublicationOutcome> outcome = DeviceBoxPopulation::outcome(version))
            outcome->finish(DevicePublicationOutcome::State::abandoned);
        releaseAfterDevice();
    }

    std::shared_ptr<CudaPopulationVersion> admittedVersion() const override { return version; }

    bool objectCounts(uint64_t& outIndexed, uint64_t& outExcluded) const noexcept override
    {
        const uint64_t count = completed ? *report->count : 0;
        outIndexed           = capacity - count;
        outExcluded          = membership->exclusions.objects.size() + count;
        return true;
    }

    uint64_t exclusionCapacity() const noexcept override { return capacity + membership->exclusions.objects.size(); }
    bool     exclusions(std::vector<uint64_t>& ids, std::vector<int32_t>& reasons) const override
    {
        ids.assign(membership->exclusions.objects.begin(), membership->exclusions.objects.end());
        reasons.assign(membership->exclusions.reasons.begin(), membership->exclusions.reasons.end());
        ids.insert(ids.end(), report->ids, report->ids + *report->count);
        reasons.insert(reasons.end(), report->reasons, report->reasons + *report->count);
        return true;
    }
    void ringWhenReady(const std::shared_ptr<DeviceSignal>& signal) noexcept override
    {

        if (version != nullptr)
            membership->boxes->ringWhenReady(version, signal);
    }

    ovspatialquery_api_status_t execute(ovspatialquery_ordinal_t, const GenerationLease&, const ObjectTable&,
                                        std::shared_ptr<CudaPopulationVersion>& outVersion) override
    {
        const ovspatialquery_api_status_t status = membership->boxes->complete(version);
        if (status == OVSPATIALQUERY_API_NOT_READY)
            return status;
        deviceBase.reset();
        // The version's device work ran after the gather, so the groups are free:
        // a host write of the world matrix waits on them until now.
        releaseAfterDevice();
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            completed = true;
            DeviceBoxPopulation::setOvstageLiveCount(version, capacity - *report->count);
            {
                std::lock_guard<std::mutex> lock(membership->recovery->mutex);
                membership->recovery->observe(*pendingReport);
            }
            outVersion = version;
        }
        return status;
    }

    std::shared_ptr<OvstageExclusionReport>    report;
    std::shared_ptr<OvstagePendingPublication> pendingReport;
    bool                                       completed = false;
    // Admission state.
    ovstage_instance_t*                      stage = nullptr;
    ovspatialquery_dynamic_execution_t       execution{};
    std::shared_ptr<OvstageDeviceMembership> membership;
    std::shared_ptr<OvstageDeviceResources>  resources;
    std::shared_ptr<DevicePreparedState>     deviceState;
    std::vector<PlacedRead>                  reads;
    ovspatialquery_ordinal_t                 baseOrdinal   = 0;
    ovspatialquery_ordinal_t                 targetOrdinal = 0;
    bool                                     completeSet   = true;
    bool                                     completeRows  = true;
    bool                                     hostReadable  = false;
    uint64_t                                 capacity      = 0u;
    uint64_t                                 rows          = 0u;
    // A membership change's binding and bits columns, in slot order.
    std::vector<int32_t>  types;
    std::vector<uint64_t> values;
    std::vector<uint64_t> bits;

  private:
    struct Columns
    {
        CuDevicePtr ids = 0u, translations = 0u, rotations = 0u, scales = 0u, pairs = 0u, types = 0u, values = 0u,
                    bits = 0u, extents = 0u;
    } columns;
    CuDevicePtr                            scratch    = 0u;
    CuEvent                                gatherDone = nullptr;
    CuEvent                                available  = nullptr;
    GenerationLease                        deviceBase;
    std::shared_ptr<CudaPopulationVersion> version;
    std::once_flag                         released;
    bool                                   committed     = false;
    bool                                   gatherStarted = false, gatherRecorded = false;

  public:
    // The event Ovstage records once the read's columns are safe to use.
    uintptr_t availabilityEvent()
    {
        if (available == nullptr)
            available = resources->takeEvent();
        return reinterpret_cast<uintptr_t>(available);
    }

  private:
    ovspatialquery_api_status_t launchGather()
    {
        Driver&    driver        = resources->driver;
        const auto scratchStatus = resources->takeScratch(scratch);
        if (scratchStatus != OVSPATIALQUERY_API_SUCCESS)
            return scratchStatus;
        gatherDone = resources->takeEvent();
        if (scratch == 0u || gatherDone == nullptr || available == nullptr)
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        const uint64_t slots = std::max<uint64_t>(capacity, 1u);
        columns.ids          = scratch;
        columns.translations = columns.ids + slots * 8u;
        columns.rotations    = columns.translations + slots * 24u;
        columns.scales       = columns.rotations + slots * 32u;
        columns.bits         = columns.scales + slots * 8u;
        columns.values       = columns.bits + slots * 16u;
        columns.pairs        = columns.values + slots * 8u;
        columns.types        = columns.pairs + slots * 8u;
        columns.extents      = (columns.types + slots * 4u + 7u) & ~CuDevicePtr(7u);
        CuStream stream      = resources->stream;
        // Ordered behind the read's columns (Ovstage's event, recorded once a group was
        // fetched) and the caller's own event.
        gatherStarted = true;
        if (driver.streamWaitEvent(stream, available, 0u) != kCuSuccess ||
            (execution.cuda_sync.wait_event != 0u &&
             driver.streamWaitEvent(stream, reinterpret_cast<CuEvent>(execution.cuda_sync.wait_event), 0u) !=
                 kCuSuccess))
            return OVSPATIALQUERY_API_INTERNAL;
        if (completeRows && rows != 0u &&
            (driver.copyHtoDAsync(columns.types, types.data(), types.size() * sizeof(int32_t), stream) != kCuSuccess ||
             driver.copyHtoDAsync(columns.values, values.data(), values.size() * sizeof(uint64_t), stream) !=
                 kCuSuccess ||
             driver.copyHtoDAsync(columns.bits, bits.data(), bits.size() * sizeof(uint64_t), stream) != kCuSuccess))
            return OVSPATIALQUERY_API_INTERNAL;
        uint64_t offset = 0u;
        for (const PlacedRead& read : reads)
        {
            for (const GatheredGroup& gathered : read.groups)
            {
                const uint64_t count = gathered.pairs.size() / 2u;
                if (count == 0u)
                    continue;
                CuDevicePtr pairs = columns.pairs + offset * 8u;
                if (driver.copyHtoDAsync(pairs, gathered.pairs.data(), gathered.pairs.size() * sizeof(uint32_t),
                                         stream) != kCuSuccess)
                    return OVSPATIALQUERY_API_INTERNAL;
                const DLTensor& tensor   = gathered.group.data.tensors[0];
                CuDevicePtr     matrices = reinterpret_cast<CuDevicePtr>(tensor.data) + tensor.byte_offset;
                CuDevicePtr     ids = columns.ids + offset * 8u, translations = columns.translations + offset * 24u,
                            rotations = columns.rotations + offset * 32u, scales = columns.scales + offset * 8u;
                CuDevicePtr records      = resources->records;
                uint64_t    work         = count;
                auto        countAddress = reinterpret_cast<CuDevicePtr>(report->count),
                     idsAddress          = reinterpret_cast<CuDevicePtr>(report->ids),
                     reasonsAddress      = reinterpret_cast<CuDevicePtr>(report->reasons);
                CuDevicePtr extents      = columns.extents + offset * 24u;
                void*       args[]       = {&work,         &pairs,        &matrices,       &records,
                                            &ids,          &translations, &rotations,      &scales,
                                            &countAddress, &idsAddress,   &reasonsAddress, &extents};
                if (driver.launchKernel(resources->gather, static_cast<unsigned>((count + 127u) / 128u), 1, 1, 128, 1,
                                        1, 0, stream, args, nullptr) != kCuSuccess)
                    return OVSPATIALQUERY_API_INTERNAL;
                if (takeCudaFailurePoint(kCudaFailureOvstageGatherAfterKernel))
                    return OVSPATIALQUERY_API_INTERNAL;
                offset += count;
            }
        }
        gatherRecorded = driver.eventRecord(gatherDone, stream) == kCuSuccess;
        return gatherRecorded ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INTERNAL;
    }

    void releaseReads() noexcept
    {
        for (PlacedRead& read : reads)
        {
            for (GatheredGroup& gathered : read.groups)
                (void)releaseReadGroup(stage, gathered.group);
            read.groups.clear();
            (void)releaseRead(stage, read.handle);
        }
        reads.clear();
    }

    // Lets Ovstage have its groups back once the gather that reads them has run.
    void releaseAfterDevice() noexcept
    {
        std::call_once(released,
                       [this]()
                       {
                           if (resources != nullptr && resources->context != nullptr && gatherStarted)
                           {
                               CuContext previous = nullptr;
                               (void)resources->driver.contextGetCurrent(&previous);
                               (void)resources->driver.contextSetCurrent(resources->context);
                               if (gatherRecorded)
                                   (void)resources->driver.eventSynchronize(gatherDone);
                               else
                                   // Only submission errors lack an event fencing all accepted work.
                                   (void)resources->driver.streamSynchronize(resources->stream);
                               (void)resources->driver.contextSetCurrent(previous);
                           }
                           releaseReads();
                       });
    }
};

// One placed read of `attribute` over `query`: the snapshot at `target`. RENDERING-domain
// writes such as `apply_usd_time` keep no change history in Ovstage 0.3 and 0.4, so a
// change range would miss them (ADR-0052).
ovspatialquery_api_status_t placedRead(ovstage_instance_t* stage, const ovstage_placement_extension_v1_t& extension,
                                       ovstage_query_handle_t query, ovx_token_t attribute,
                                       ovspatialquery_ordinal_t target, const ovstage_placement_t& placement,
                                       PlacedRead& out)
{
    const ovstage_ordinal_range_t  range{0u, static_cast<ovstage_ordinal_t>(target), false};
    const ovstage_enqueue_result_t enqueued =
        extension.read_attributes_on_device(stageContext(stage), query, &attribute, 1u, range, &placement, &out.handle);
    if (enqueued.status != OVSTAGE_OK)
    {
        out.handle = OVSTAGE_INVALID_READ_HANDLE;
        return mapStageStatus(enqueued.status);
    }
    ovspatialquery_api_status_t status = waitAndRelease(stage, enqueued.op_index);
    ReadCursor                  cursor(stage);
    while (status == OVSPATIALQUERY_API_SUCCESS)
    {
        GatheredGroup              gathered;
        const ovstage_api_status_t fetched = cursor.next(out.handle, OVSTAGE_TIMEOUT_INFINITE, gathered.group);
        if (fetched == OVSTAGE_ERROR_END_OF_ITERATION)
            break;
        if (fetched != OVSTAGE_OK)
        {
            status = mapStageStatus(fetched);
            break;
        }
        out.groups.push_back(std::move(gathered));
    }
    return status;
}

// Resolves every row of a read's groups to the slot its prim feeds. A prim the
// membership does not hold is one the host excluded, because the fingerprint
// already compared which prims have a spatial source; its rows are skipped, as are
// authored rows of a prim the Fabric matrix feeds. A Fabric row of a prim fed by its
// authored matrix, or a tombstone, means the membership changed (`outChanged`).
ovspatialquery_api_status_t resolveRows(path_dictionary_instance_t*    dictionary,
                                        const OvstageDeviceMembership& membership, uint8_t fromAuthored,
                                        PlacedRead& read, std::vector<uint8_t>& covered, bool& outChanged)
{
    std::vector<ovx_primpath_t> paths;
    for (GatheredGroup& gathered : read.groups)
    {
        const ReadGroup& group = gathered.group;
        if (group.is_delete)
        {
            outChanged = true;
            return OVSPATIALQUERY_API_SUCCESS;
        }
        if (group.prims.count == 0u)
            continue;
        if (group.is_array || group.data.tensor_count != 1u)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        const DLTensor& tensor = group.data.tensors[0];
        // One float64 4x4 matrix per row, compact, on the requested device.
        const bool square = tensor.ndim == 3 && tensor.shape[1] == 4 && tensor.shape[2] == 4;
        const bool flat   = tensor.ndim == 1 && tensor.dtype.lanes == 16u;
        if (tensor.device.device_type != kDLCUDA || tensor.dtype.code != kDLFloat || tensor.dtype.bits != 64u ||
            (!square && !flat) || (square && tensor.dtype.lanes != 1u) || tensor.shape[0] < 0)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        ovspatialquery_api_status_t status = resolveGroupPaths(group, dictionary, paths);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        gathered.pairs.reserve(2u * paths.size());
        for (uint32_t row = 0u; row < paths.size(); ++row)
        {
            const auto found = membership.slotByPath->find(handleValue(paths[row]));
            if (found == membership.slotByPath->end())
                continue;
            if (membership.authored[found->second] != fromAuthored)
            {
                // As on the host path, the authored matrix only fills prims without a Fabric one.
                if (fromAuthored != 0u)
                    continue;
                outChanged = true;
                return OVSPATIALQUERY_API_SUCCESS;
            }
            const uint32_t dataRow = group.data.index_map != nullptr ? group.data.index_map[row] : row;
            if (static_cast<int64_t>(dataRow) >= tensor.shape[0] || covered[found->second] != 0u)
                return OVSPATIALQUERY_API_INTERNAL;
            covered[found->second] = 1u;
            gathered.pairs.push_back(dataRow);
            gathered.pairs.push_back(found->second);
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// The new membership a membership-changing publication defines, from the host's
// complete acquisition: one slot per object, its shape scaled by the prim's world
// scales then, and the definition attributes the next publications compare.
struct DefinedMembership
{
    std::vector<ExplicitInstanceInput>     instances;
    std::vector<DeviceShapeDescription>    shapes;
    std::vector<family::OvstageSlotRecord> records;
    std::shared_ptr<OvstageObjectPreparer> preparer;
    std::vector<ovx_token_t>               definitionAttributes;
};

ovspatialquery_api_status_t defineMembership(const OwnedFilter& filter, const CopiedSourceSnapshot& snapshot,
                                             DefinedMembership& out)
{
    out.preparer         = std::make_shared<OvstageObjectPreparer>();
    const size_t n       = snapshot.objects.size();
    bool         spheres = false, cubes = false, capsules = false, meshes = false;
    out.instances.reserve(n);
    out.shapes.reserve(n);
    out.records.reserve(n);
    out.preparer->shapes.reserve(n);
    auto slots = std::make_shared<OvstageSlotMap>();
    slots->reserve(n);
    out.preparer->slots = slots;
    for (size_t index = 0u; index < n; ++index)
    {
        const PopulationObject& object = snapshot.objects[index];
        // Instance proxies of one prototype share one definition (ADR-0050).
        const size_t source = snapshot.definitionSources.empty() ? index : snapshot.definitionSources[index];
        if (source >= snapshot.geometryDefinitions.size())
            return OVSPATIALQUERY_API_INTERNAL;
        const GeometryDefinitionValue& definition = snapshot.geometryDefinitions[source];
        ExactGeometryDescription       description;
        ovspatialquery_api_status_t    status = exportExactGeometryDescription(object.exactGeometry, description);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        family::OvstageSlotRecord record{};
        record.id = object.id.value;
        if (object.ovstageShape)
        {
            std::copy(object.ovstageShape->k.begin(), object.ovstageShape->k.end(), record.k);
            std::copy(object.ovstageShape->inverse.begin(), object.ovstageShape->inverse.end(), record.inverse);
        }
        record.size     = object.ovstageCubeSize;
        cubes           = cubes || record.size > 0;
        record.scale[0] = record.scale[1] = record.scale[2] = 1.0;
        record.rule                                         = family::kOvstageRuleUniform;
        record.frame                                        = family::kOvstageFrameIdentity;
        OvstageSlotShape       shape;
        DeviceShapeDescription device;
        if (const auto* sphere = std::get_if<SphereDefinitionValue>(&definition))
        {
            const auto&  world = std::get<SphereExactGeometryDescription>(description);
            const double scale = world.radius / sphere->radius;
            record.scale[0] = record.scale[1] = record.scale[2] = scale;
            shape.family                                        = ExactGeometryFamily::sphere;
            shape.parameters[0]                                 = sphere->radius;
            record.radius                                       = sphere->radius;
            device.type                                         = OVSPATIALQUERY_GEOMETRY_SPHERE;
            spheres                                             = true;
        }
        else if (const auto* box = std::get_if<BoxDefinitionValue>(&definition))
        {
            const auto& world = std::get<BoxExactGeometryDescription>(description);
            record.rule       = family::kOvstageRuleBox;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                record.scale[axis] = world.halfExtents[axis] / (box->size * 0.5);
            shape.family     = ExactGeometryFamily::box;
            shape.parameters = {box->size * 0.5, box->size * 0.5, box->size * 0.5};
            record.size      = box->size;
            device.type      = OVSPATIALQUERY_GEOMETRY_BOX;
            cubes            = true;
        }
        else if (const auto* capsule = std::get_if<CapsuleDefinitionValue>(&definition))
        {
            const auto& world       = std::get<CapsuleExactGeometryDescription>(description);
            double      halfSegment = 0.0;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                const double half = 0.5 * (world.endpointB[axis] - world.endpointA[axis]);
                halfSegment += half * half;
            }
            halfSegment        = std::sqrt(halfSegment);
            const double scale = world.radius / capsule->radius;
            record.scale[0] = record.scale[1] = record.scale[2] = scale;
            record.frame        = capsule->axis == CapsuleAxis::x   ? family::kOvstageFrameFromX
                                  : capsule->axis == CapsuleAxis::z ? family::kOvstageFrameFromZ
                                                                    : family::kOvstageFrameIdentity;
            shape.family        = ExactGeometryFamily::capsule;
            shape.parameters[0] = capsule->radius;
            shape.parameters[1] = capsule->height * 0.5;
            record.radius       = capsule->radius;
            record.halfHeight   = capsule->height * 0.5;
            device.type         = OVSPATIALQUERY_GEOMETRY_CAPSULE;
            capsules            = true;
        }
        else if (std::holds_alternative<ConvexDefinitionValue>(definition))
        {
            record.rule  = family::kOvstageRuleRigid;
            shape.family = ExactGeometryFamily::convex;
            shape.hull   = std::static_pointer_cast<const PreparedConvexHull>(snapshot.definitionOwners[source]);
            device.type  = OVSPATIALQUERY_GEOMETRY_CONVEX;
            device.hull  = shape.hull;
            meshes       = true;
            if (shape.hull == nullptr || shape.hull->preparedSurface() == nullptr)
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        else if (object.triangleGeometry != nullptr)
        {
            record.rule        = family::kOvstageRuleRigid;
            shape.family       = ExactGeometryFamily::triangleMesh;
            shape.mesh         = object.triangleGeometry->mesh;
            shape.doubleSided  = object.triangleGeometry->doubleSided;
            device.type        = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
            device.localBounds = preparedTriangleMeshLocalBounds(*shape.mesh);
            meshes             = true;
        }
        else
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        if (shape.family == ExactGeometryFamily::sphere || shape.family == ExactGeometryFamily::box ||
            shape.family == ExactGeometryFamily::capsule)
        {
            device.parameters = shape.parameters;
        }
        for (const double scale : record.scale)
        {
            if (!std::isfinite(scale) || !(scale > 0.0))
                return OVSPATIALQUERY_API_INVALID_STATE;
        }
        // The geometry reference and bits are assigned after staging (submitOvstageDevicePublication).
        ExplicitInstanceInput instance;
        instance.objectValue = object.id.value;
        out.instances.push_back(instance);
        out.shapes.push_back(device);
        out.records.push_back(record);
        slots->emplace(object.id.value, static_cast<uint32_t>(index));
        out.preparer->shapes.push_back(std::move(shape));
    }
    // A definition written when a prim is created is in no change range, so the
    // next publications compare these attributes' values (readMembershipFingerprint).
    if (spheres || capsules)
        out.definitionAttributes.push_back(filter.radius);
    if (cubes)
        out.definitionAttributes.push_back(filter.size);
    if (capsules)
    {
        out.definitionAttributes.push_back(filter.height);
        out.definitionAttributes.push_back(filter.axis);
    }
    if (meshes)
    {
        out.definitionAttributes.insert(out.definitionAttributes.end(),
                                        {filter.points, filter.faceVertexCounts, filter.faceVertexIndices,
                                         filter.doubleSided, filter.orientation, filter.holeIndices,
                                         filter.physicsApproximation, filter.hullVertexLimit, filter.usdSchemas});
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace

ovspatialquery_api_status_t submitOvstageDevicePublication(ovspatialquery_instance_handle_t          instanceHandle,
                                                           ovspatialquery_population_handle_t        population,
                                                           const ovspatialquery_ordinal_t*           baseOrdinal,
                                                           ovspatialquery_ordinal_t                  targetOrdinal,
                                                           const ovspatialquery_dynamic_execution_t& execution,
                                                           ovspatialquery_op_id_t&                   outOperation,
                                                           std::string&                              outDiagnostic)
{
    outOperation = OVSPATIALQUERY_INVALID_OP;
    if (execution.device.device_type != kDLCUDA || execution.device.device_id < 0 ||
        execution.cuda_sync.has_stream != 1u || targetOrdinal < 0 || (baseOrdinal != nullptr && *baseOrdinal < 0))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (baseOrdinal != nullptr && targetOrdinal <= *baseOrdinal)
    {
        return OVSPATIALQUERY_API_STALE_ORDINAL;
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (const char* unavailable = cudaBackendUnavailableReason(
            execution.device.device_id,
            instance != nullptr ? instance->cudaModulePins(execution.device.device_id) : nullptr))
    {
        outDiagnostic = unavailable;
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    // Nothing is queued on a stream that is capturing a CUDA graph (ADR-0060).
    if (cudaStreamIsCapturing(execution.device.device_id, execution.cuda_sync.stream))
    {
        outDiagnostic = "cuda_sync.stream is capturing a CUDA graph";
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    PopulationKey               key{};
    PopulationSourceBinding     source;
    ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, key, source);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    if (source.kind != OVSPATIALQUERY_SOURCE_OVSTAGE || source.deviceState == nullptr)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    ovstage_instance_t* stage =
        const_cast<ovstage_instance_t*>(static_cast<const ovstage_instance_t*>(source.instance));
    path_dictionary_instance_t* dictionary =
        const_cast<path_dictionary_instance_t*>(static_cast<const path_dictionary_instance_t*>(source.pathDictionary));
    const std::shared_ptr<OwnedFilter>      filter    = std::static_pointer_cast<OwnedFilter>(source.sourceState);
    const ovstage_placement_extension_v1_t* extension = ovstagePlacementExtension(stage, outDiagnostic);
    if (extension == nullptr)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (instance->handles.hasLiveView(key))
    {
        outDiagnostic = "a population with an ovstage view cannot be prepared on the device, because no view selects "
                        "a device-prepared version";
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    bool                     ready = false, pending = false;
    ovspatialquery_ordinal_t head        = 0;
    uint64_t                 objectCount = 0u;
    uint32_t                 retained    = 0u;
    status = instance->populations.getInfoState(key, ready, head, objectCount, retained, pending);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    std::lock_guard<std::mutex>              lock(filter->deviceMutex);
    std::shared_ptr<OvstageDeviceMembership> current = filter->deviceMembership;
    // A version does not move between the host and the device (ADR-0024 decision 4).
    if (ready && source.deviceState->current() == nullptr)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    // A device version stays on its device: an update names the device its base was prepared on.
    if (baseOrdinal != nullptr && current != nullptr && current->boxes != nullptr &&
        (current->boxes->device().device_type != execution.device.device_type ||
         current->boxes->device().device_id != execution.device.device_id))
    {
        outDiagnostic = "a device-prepared update names another CUDA device than its base was prepared on";
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (baseOrdinal != nullptr && current == nullptr)
    {
        // The first complete set defines the membership an update builds on.
        return pending ? OVSPATIALQUERY_API_BUSY
               : ready ? OVSPATIALQUERY_API_NOT_SUPPORTED
                       : OVSPATIALQUERY_API_NOT_READY;
    }
    std::shared_ptr<const ParticipationTable> participation;
    {
        std::vector<SourceViewRequest>     views;
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        status = instance->handles.beginSourcePublication(key, views, participation);
        (void)instance->handles.finishSourceViewUpdate(views);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
    }
    const ovstage_query_handle_t invalidQuery = OVSTAGE_INVALID_QUERY_HANDLE;
    bool                         rebuild =
        baseOrdinal == nullptr || current == nullptr || !current->valid || current->participation != participation;
    if (current && current->recovery)
        current->recovery->fold();
    std::vector<uint64_t>                 recheckIds;
    std::unordered_map<uint64_t, int32_t> recheckReasons;
    std::string                           dropReport;
    SourceDroppedTriangles                dropCounts;
    if (!rebuild)
        for (size_t i = 0u; i < current->exclusions.objects.size(); ++i)
            if (current->exclusions.reasons[i] != OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE)
            {
                recheckIds.push_back(current->exclusions.objects[i]);
                recheckReasons.emplace(current->exclusions.objects[i], current->exclusions.reasons[i]);
            }
    if (!recheckIds.empty())
    {
        // A filtered-out prim has no slot in the device membership. Recheck just
        // those prims at the target so unchanged bad content does not rebuild all slots.
        std::shared_ptr<OwnedFilter> subset;
        status = copyFilter(stage, ovstage_filter_t{}, dictionary, subset, filter->preparation);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        if (subset->usdPathList != OVX_INVALID_PRIMPATH_LIST)
            dictionary->vtable->release_path_list_reference(dictionary->context, subset->usdPathList);
        std::vector<ovx_primpath_t> recheckIdsPaths;
        recheckIdsPaths.reserve(recheckIds.size());
        for (const uint64_t id : recheckIds)
            recheckIdsPaths.push_back(makeHandle<ovx_primpath_t>(id));
        status = mapDictionaryResult(
            dictionary, dictionary->vtable->create_path_list_from_paths(dictionary->context, recheckIdsPaths.data(),
                                                                        recheckIdsPaths.size(), &subset->usdPathList));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        subset->usdPathCount = recheckIds.size();
        subset->dictionary   = dictionary;
        CopiedSourceSnapshot rechecked;
        rechecked.resolveMaterials = false;
        status = source.acquireSnapshot(source.instance, source.pathDictionary, subset, {}, targetOrdinal, nullptr,
                                        rechecked);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            outDiagnostic = std::move(rechecked.refusal);
            return status;
        }
        rebuild = !rechecked.objects.empty() || rechecked.exclusions.objects.size() != recheckIds.size();
        if (!rebuild)
            for (size_t i = 0u; i < rechecked.exclusions.objects.size(); ++i)
            {
                const auto expected = recheckReasons.find(rechecked.exclusions.objects[i]);
                if (expected == recheckReasons.end() || expected->second != rechecked.exclusions.reasons[i])
                {
                    rebuild = true;
                    break;
                }
            }
    }
    bool                  sparse = false;
    std::vector<uint64_t> recovered;
    if (!rebuild && current->recovery && !current->recovery->prims.empty())
    {
        recovered = current->recovery->prims;
        // The path-list query reads only the prims whose completed rows asked for a reshape.
        std::shared_ptr<OwnedFilter> subset;
        status = copyFilter(stage, ovstage_filter_t{}, dictionary, subset, filter->preparation);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        if (subset->usdPathList != OVX_INVALID_PRIMPATH_LIST)
            dictionary->vtable->release_path_list_reference(dictionary->context, subset->usdPathList);
        std::vector<ovx_primpath_t> recoveredPaths;
        recoveredPaths.reserve(recovered.size());
        for (const uint64_t id : recovered)
            recoveredPaths.push_back(makeHandle<ovx_primpath_t>(id));
        status = mapDictionaryResult(
            dictionary, dictionary->vtable->create_path_list_from_paths(dictionary->context, recoveredPaths.data(),
                                                                        recoveredPaths.size(), &subset->usdPathList));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        subset->usdPathCount = recovered.size();
        subset->dictionary   = dictionary;
        CopiedSourceSnapshot changed;
        DefinedMembership    replacement;
        status = acquireSourceObjects(
            instanceHandle, population, targetOrdinal, changed, participation,
            [&](const CopiedSourceSnapshot& snapshot) { return defineMembership(*filter, snapshot, replacement); },
            &outDiagnostic, subset);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        dropReport = std::move(changed.droppedTriangles);
        dropCounts = std::move(changed.droppedTriangleCounts);
        if (changed.objects.size() != recovered.size())
        {
            // Topology or source availability changed as well; the normal fingerprint rebuild owns that case.
            rebuild = true;
        }
        else
        {
            auto next                    = std::make_shared<OvstageDeviceMembership>();
            next->fingerprint            = current->fingerprint;
            next->definitionAttributes   = current->definitionAttributes;
            next->prototypePaths         = current->prototypePaths;
            next->participation          = current->participation;
            next->slotByPath             = current->slotByPath;
            next->authored               = current->authored;
            next->anyAuthored            = current->anyAuthored;
            next->droppedPointPrototypes = current->droppedPointPrototypes;
            next->exclusions             = current->exclusions;
            next->records                = current->records;
            next->definitions            = current->definitions;
            next->instances              = current->instances;
            next->recovery               = current->recovery;
            next->preparer               = std::make_shared<OvstageObjectPreparer>(*current->preparer);
            std::vector<std::shared_ptr<const void>> owners;
            for (size_t i = 0; i < changed.objects.size(); ++i)
            {
                const auto& object                         = changed.objects[i];
                const auto  slot                           = next->slotByPath->at(object.id.value);
                replacement.instances[i].geometry          = object.geometryReference;
                replacement.instances[i].participationBits = object.participationBits;
                replacement.shapes[i].value                = object.geometryReference.value;
                next->instances[slot]                      = replacement.instances[i];
                next->records[slot]                        = replacement.records[i];
                next->preparer->shapes[slot]               = replacement.preparer->shapes.at(i);
                owners.push_back(object.definitionOwner    ? object.definitionOwner
                                 : object.triangleGeometry ? std::shared_ptr<const void>(object.triangleGeometry->mesh)
                                                           : std::shared_ptr<const void>{});
                next->definitions[slot] = owners.back();
            }
            std::shared_ptr<HostLease> hostLease;
            status = reserveSharedHostLease(instance,
                                            8192u + next->records.size() * (256u + sizeof(family::OvstageSlotRecord)) +
                                                next->exclusions.objects.capacity() * sizeof(uint64_t) +
                                                next->exclusions.reasons.capacity() * sizeof(int32_t),
                                            hostLease);
            std::shared_ptr<void> deviceLease, poolLease;
            if (status == OVSPATIALQUERY_API_SUCCESS)
                status = reserveInstanceDeviceMemory(instanceHandle, execution.device,
                                                     DeviceBoxPopulation::membershipBytes(next->records.size()),
                                                     deviceLease);
            const auto poolBytes = current->boxes->additionalDefinitionBytes(replacement.shapes, true);
            if (status == OVSPATIALQUERY_API_SUCCESS && poolBytes)
                status = reserveInstanceDeviceMemory(instanceHandle, execution.device, poolBytes, poolLease);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            std::shared_ptr<HostLease> stagingLease;
            status = reserveSharedHostLease(
                instance, DeviceBoxPopulation::forkStagingBytes(replacement.instances.size()), stagingLease);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            status = current->boxes->forkOvstage(replacement.instances, replacement.shapes, next->preparer,
                                                 std::move(owners), hostLease, std::move(deviceLease),
                                                 std::move(poolLease), std::move(stagingLease), next->boxes);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            next->resources = std::make_shared<OvstageDeviceResources>();
            status = next->resources->initialize(instanceHandle, execution.device, next->records, current->resources);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            current = next;
            sparse  = true;
        }
    }

    std::shared_ptr<OvstageDeviceMembership> membership = rebuild ? nullptr : current;
    std::shared_ptr<OvstageDeviceCapture>    capture;
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        ovstage_query_handle_t query = invalidQuery;
        std::vector<uint64_t>  fingerprint;
        DefinedMembership      defined;
        if (!rebuild)
        {
            status = readMembershipFingerprint(stage, dictionary, *filter, targetOrdinal, current->definitionAttributes,
                                               current->prototypePaths, fingerprint, query);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            if (fingerprint != current->fingerprint)
            {
                (void)releaseQuery(stage, query);
                rebuild    = true;
                membership = nullptr;
            }
        }
        if (rebuild)
        {
            CopiedSourceSnapshot snapshot;
            status = acquireSourceObjects(
                instanceHandle, population, targetOrdinal, snapshot, participation,
                [&](const CopiedSourceSnapshot& acquired) { return defineMembership(*filter, acquired, defined); },
                &outDiagnostic);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            dropReport = std::move(snapshot.droppedTriangles);
            dropCounts = std::move(snapshot.droppedTriangleCounts);
            // Geometry references and participation bits are assigned only now.
            if (defined.instances.size() != snapshot.objects.size())
                return OVSPATIALQUERY_API_INTERNAL;
            for (size_t index = 0u; index < snapshot.objects.size(); ++index)
            {
                defined.instances[index].geometry          = snapshot.objects[index].geometryReference;
                defined.instances[index].participationBits = snapshot.objects[index].participationBits;
                defined.shapes[index].value                = snapshot.objects[index].geometryReference.value;
            }
            status = readMembershipFingerprint(stage, dictionary, *filter, targetOrdinal, defined.definitionAttributes,
                                               snapshot.prototypePaths, fingerprint, query);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            membership                         = std::make_shared<OvstageDeviceMembership>();
            membership->fingerprint            = std::move(fingerprint);
            membership->definitionAttributes   = defined.definitionAttributes;
            membership->prototypePaths         = snapshot.prototypePaths;
            membership->droppedPointPrototypes = snapshot.droppedPointPrototypes != 0u;
            membership->participation          = participation;
            membership->instances              = defined.instances;
            membership->exclusions             = std::move(snapshot.exclusions);
            membership->records                = defined.records;
            membership->preparer               = defined.preparer;
            for (const auto& object : snapshot.objects)
                membership->definitions.push_back(object.definitionOwner ? object.definitionOwner
                                                  : object.triangleGeometry
                                                      ? std::shared_ptr<const void>(object.triangleGeometry->mesh)
                                                      : std::shared_ptr<const void>{});
            membership->recovery = std::make_shared<OvstagePendingRecovery>();
            status               = reserveSharedHostLease(
                instance,
                4096u + defined.instances.size() *
                            (sizeof(uint64_t) + sizeof(std::pair<uint64_t, ovspatialquery_ordinal_t>)),
                membership->recovery->lease);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            membership->recovery->prims.reserve(defined.instances.size());
            membership->recovery->rebuilt.reserve(defined.instances.size());
            for (const auto& row : defined.instances)
                membership->recovery->rebuilt.emplace_back(row.objectValue, targetOrdinal);
            std::sort(membership->recovery->rebuilt.begin(), membership->recovery->rebuilt.end());
            membership->authored.assign(defined.instances.size(), 0u);
            membership->slotByPath = defined.preparer->slots;
        }
        capture                = std::make_shared<OvstageDeviceCapture>();
        capture->stage         = stage;
        capture->execution     = execution;
        capture->membership    = membership;
        capture->deviceState   = source.deviceState;
        capture->baseOrdinal   = baseOrdinal != nullptr ? *baseOrdinal : 0;
        capture->targetOrdinal = targetOrdinal;
        capture->completeSet   = baseOrdinal == nullptr;
        capture->completeRows  = rebuild || sparse;
        capture->hostReadable  = execution.preparation == OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE;
        capture->capacity      = membership->slotByPath->size();
        if (rebuild)
        {
            capture->resources = std::make_shared<OvstageDeviceResources>();
            status             = capture->resources->initialize(instanceHandle, execution.device, defined.records);
        }
        else
        {
            capture->resources = membership->resources;
        }
        ovstage_placement_t placement OVSTAGE_PLACEMENT_INIT;
        placement.device             = execution.device;
        placement.stream             = execution.cuda_sync.stream;
        placement.availability_event = status == OVSPATIALQUERY_API_SUCCESS ? capture->availabilityEvent() : 0u;
        if (status == OVSPATIALQUERY_API_SUCCESS && placement.availability_event == 0u)
            status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        // A query that also selects excluded prims, authored PointInstancer prototypes, or
        // (for a filter that names its prims) a listed path with no prim, can make Ovstage
        // refuse a device matrix read for the rows it cannot place. Read poses only for slots
        // that this membership actually publishes.
        const bool unpublishedRows =
            !membership->exclusions.objects.empty() || membership->droppedPointPrototypes ||
            (filter->usdPathList != OVX_INVALID_PRIMPATH_LIST && filter->usdPathCount != capture->capacity);
        ovstage_query_handle_t poseQuery = query;
        ovx_primpath_list_t    poseList  = OVX_INVALID_PRIMPATH_LIST;
        if (status == OVSPATIALQUERY_API_SUCCESS && unpublishedRows && capture->capacity != 0u)
        {
            std::vector<ovx_primpath_t> paths;
            paths.reserve(capture->capacity);
            for (const auto& entry : *membership->slotByPath)
                paths.push_back(makeHandle<ovx_primpath_t>(entry.first));
            status = mapDictionaryResult(dictionary, dictionary->vtable->create_path_list_from_paths(
                                                         dictionary->context, paths.data(), paths.size(), &poseList));
            if (status == OVSPATIALQUERY_API_SUCCESS)
                status =
                    mapStageStatus(stageVtable(stage)->query_from_path_list(stageContext(stage), poseList, &poseQuery));
        }
        // Every publication reads the snapshot at the target; an update's prepare kernel
        // finds which slots moved by comparing with its base (ADR-0052).
        std::vector<uint8_t> covered(capture->capacity, 0u);
        bool                 changed = false;
        if (status == OVSPATIALQUERY_API_SUCCESS && capture->capacity != 0u)
        {
            capture->reads.emplace_back();
            status = placedRead(stage, *extension, poseQuery, filter->fabricWorldMatrix, targetOrdinal, placement,
                                capture->reads.back());
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && capture->capacity != 0u)
            status = resolveRows(dictionary, *membership, 0u, capture->reads.back(), covered, changed);
        // Prims without a Fabric world matrix take the authored one, as on the host path.
        const bool needAuthored =
            rebuild ? std::find(covered.begin(), covered.end(), 0u) != covered.end() : membership->anyAuthored;
        if (status == OVSPATIALQUERY_API_SUCCESS && !changed && needAuthored)
        {
            if (rebuild)
            {
                for (size_t slot = 0u; slot < covered.size(); ++slot)
                    membership->authored[slot] = covered[slot] == 0u ? 1u : 0u;
                membership->anyAuthored = true;
            }
            capture->reads.emplace_back();
            status = placedRead(stage, *extension, poseQuery, filter->worldMatrix, targetOrdinal, placement,
                                capture->reads.back());
            if (status == OVSPATIALQUERY_API_SUCCESS)
                status = resolveRows(dictionary, *membership, 1u, capture->reads.back(), covered, changed);
        }
        if (poseQuery != query)
            (void)releaseQuery(stage, poseQuery);
        if (poseList != OVX_INVALID_PRIMPATH_LIST)
            (void)mapDictionaryResult(dictionary,
                                      dictionary->vtable->release_path_list_reference(dictionary->context, poseList));
        (void)releaseQuery(stage, query);
        // A slot no matrix covers lost its pose since the membership was defined.
        if (!rebuild && status == OVSPATIALQUERY_API_SUCCESS &&
            std::find(covered.begin(), covered.end(), 0u) != covered.end())
            changed = true;
        // A prim that appeared, vanished or changed its matrix column: start again from complete rows.
        if (!rebuild && changed)
        {
            capture.reset();
            rebuild    = true;
            membership = nullptr;
            continue;
        }
        // Ovstage 0.3 and 0.4 refuse a placed read of the Fabric matrices while a matched prim has none.
        if (status == OVSPATIALQUERY_API_NOT_SUPPORTED && outDiagnostic.empty())
            outDiagnostic = "the Ovstage runtime refused the device read of the population's world matrices, "
                            "or returned them in a layout the device path cannot read";
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        if (changed)
            return OVSPATIALQUERY_API_INTERNAL;
        if (rebuild && std::find(covered.begin(), covered.end(), 0u) != covered.end())
        {
            outDiagnostic = "a prim of the population has no world matrix the device can read";
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        for (const PlacedRead& read : capture->reads)
            for (const GatheredGroup& group : read.groups)
                capture->rows += group.pairs.size() / 2u;
        if (rebuild)
        {
            // A membership change publishes complete rows into new device tables.
            std::shared_ptr<HostLease> hostLease;
            status = reserveSharedHostLease(instance,
                                            8192u + capture->capacity * (256u + sizeof(family::OvstageSlotRecord)) +
                                                membership->exclusions.objects.capacity() * sizeof(uint64_t) +
                                                membership->exclusions.reasons.capacity() * sizeof(int32_t),
                                            hostLease);
            std::shared_ptr<void> deviceLease, poolLease;
            if (status == OVSPATIALQUERY_API_SUCCESS)
                status =
                    reserveInstanceDeviceMemory(instanceHandle, execution.device,
                                                DeviceBoxPopulation::membershipBytes(capture->capacity), deviceLease);
            const uint64_t poolBytes = DeviceBoxPopulation::definitionBytes(defined.shapes);
            if (status == OVSPATIALQUERY_API_SUCCESS && poolBytes != 0u)
                status = reserveInstanceDeviceMemory(instanceHandle, execution.device, poolBytes, poolLease);
            std::shared_ptr<DeviceBoxPopulation> boxes = std::make_shared<DeviceBoxPopulation>();
            if (status == OVSPATIALQUERY_API_SUCCESS)
                status = boxes->initialize(execution.device, capture->capacity, defined.instances, defined.shapes,
                                           membership->definitions, std::move(hostLease), std::move(deviceLease),
                                           defined.preparer, std::move(poolLease));
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
            membership->boxes     = std::move(boxes);
            membership->resources = capture->resources;
        }
        if (capture->completeRows)
        {
            // Complete rows carry each slot's binding and bits in the gather's row order.
            capture->types.reserve(membership->instances.size());
            capture->values.reserve(membership->instances.size());
            capture->bits.reserve(membership->instances.size() * 2u);
            for (const PlacedRead& read : capture->reads)
                for (const GatheredGroup& group : read.groups)
                    for (size_t pair = 1u; pair < group.pairs.size(); pair += 2u)
                    {
                        const ExplicitInstanceInput& row = membership->instances[group.pairs[pair]];
                        capture->types.push_back(static_cast<int32_t>(row.geometry.family) +
                                                 static_cast<int32_t>(kGeometryTypeFirstBuiltin));
                        capture->values.push_back(row.geometry.value);
                        capture->bits.insert(capture->bits.end(), row.participationBits.words,
                                             row.participationBits.words + 2u);
                    }
        }
        break;
    }
    capture->report            = std::make_shared<OvstageExclusionReport>();
    capture->report->resources = capture->resources;
    capture->report->capacity  = capture->capacity;
    status                     = reserveSharedHostLease(instance, 8u + capture->capacity * 12u, capture->report->lease);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    status = capture->report->initialize();
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    capture->pendingReport          = std::make_shared<OvstagePendingPublication>();
    capture->pendingReport->ordinal = targetOrdinal;
    status = reserveSharedHostLease(instance, sizeof(OvstagePendingPublication) + capture->capacity * sizeof(uint64_t),
                                    capture->pendingReport->lease);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    capture->pendingReport->reshapes.reserve(capture->capacity);
    PreparedPopulationInput input;
    status = baseOrdinal == nullptr
                 ? PreparedPopulationInput::prepareCopiedSet(targetOrdinal, ObjectTable(), input)
                 : PreparedPopulationInput::prepareCopiedCompleteUpdate(*baseOrdinal, targetOrdinal, {}, input);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    std::shared_ptr<OvstageDeviceMembership> admitted = capture->membership;
    input.adoptCudaCapture(capture);
    if (baseOrdinal != nullptr)
    {
        input.chainPendingPublications(); // ADR-0021: the base may be the newest admitted update
    }
    {
        std::lock_guard<std::mutex> recoveryLock(admitted->recovery->mutex);
        admitted->recovery->publications.reserve(admitted->recovery->publications.size() + 1);
    }
    status = submitPopulationInput(instance, key, std::move(input), outOperation, std::move(dropCounts));
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        {
            std::lock_guard<std::mutex> recoveryLock(admitted->recovery->mutex);
            admitted->recovery->publications.push_back(capture->pendingReport);
            if (sparse)
                for (uint64_t id : recovered)
                {
                    auto built =
                        std::lower_bound(admitted->recovery->rebuilt.begin(), admitted->recovery->rebuilt.end(), id,
                                         [](const auto& entry, uint64_t value) { return entry.first < value; });
                    if (built != admitted->recovery->rebuilt.end() && built->first == id)
                        built->second = targetOrdinal;
                }
            if (sparse || rebuild)
                admitted->recovery->prims.clear();
        }
        filter->deviceMembership = std::move(admitted);
        // Names the meshes this call read that dropped triangles without area (ADR-0073).
        outDiagnostic = std::move(dropReport);
    }
    return status;
}

#else

ovspatialquery_api_status_t submitOvstageDevicePublication(ovspatialquery_instance_handle_t,
                                                           ovspatialquery_population_handle_t,
                                                           const ovspatialquery_ordinal_t*, ovspatialquery_ordinal_t,
                                                           const ovspatialquery_dynamic_execution_t&,
                                                           ovspatialquery_op_id_t& outOperation,
                                                           std::string&            outDiagnostic)
{
    outOperation  = OVSPATIALQUERY_INVALID_OP;
    outDiagnostic = kCudaNoBackend;
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
}

#endif

} // namespace detail
} // namespace ovspatialquery
